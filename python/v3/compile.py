"""v3.capture() -- Inductor cpp_wrapper probe.

Two variants:
  - capture(fn, *args, **kwargs):          stock cpp_wrapper (fusion enabled)
  - capture_fallback(fn, *args, **kwargs): cpp_wrapper + every op forced to fallback

Both return a callable produced by torch.compile(backend="inductor",
dynamic=True). The capture call primes the cache by invoking the
compiled fn once with example args so the user's first real call is hot.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Callable

import torch
import torch._inductor.config as inductor_config

from .fallback_hijack import force_all_fallback


_LAST_REPORT: dict | None = None
_LAST_GRAPH_STATS: dict = {}


def _count_aten_nodes(graph: torch.fx.Graph) -> int:
    n = 0
    for node in graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            n += 1
    return n


def _stats_post_grad_pass(graph: torch.fx.Graph) -> None:
    """Called by inductor as post_grad_custom_pre_pass. Runs AFTER AOT
    autograd has produced an aten-canonical graph but BEFORE inductor's
    optimization passes, so node.target == aten OpOverload as expected."""
    _LAST_GRAPH_STATS["fx_nodes"] = _count_aten_nodes(graph)


def capture(fn: Callable, *example_args: Any, **example_kwargs: Any) -> Callable:
    return _capture_common(fn, example_args, example_kwargs, fallback=False)


def capture_fallback(fn: Callable, *example_args: Any, **example_kwargs: Any) -> Callable:
    return _capture_common(fn, example_args, example_kwargs, fallback=True)


def last_capture_report() -> dict | None:
    return dict(_LAST_REPORT) if _LAST_REPORT is not None else None


@contextlib.contextmanager
def _stock_cpp_wrapper_config():
    # Only set cpp_wrapper. Inductor's own get_cpp_wrapper_config()
    # (compile_fx.py) handles cudagraph interaction internally: it
    # preserves the user's triton.cudagraphs setting unless we're in
    # AOTI or graph_partition mode (both incompatible). For JIT +
    # fused Triton kernels, cudagraphs are cudagraph-safe and should
    # stay on if the user enabled them. v3-fallback differs (see
    # fallback_hijack.force_all_fallback) because aten FallbackKernel
    # ops are NOT cudagraph-safe.
    with inductor_config.patch({"cpp_wrapper": True}):
        yield


def _snapshot_pycodecache_module_count() -> int:
    """Return current size of PyCodeCache.modules, or 0 if unavailable."""
    try:
        from torch._inductor.codecache import PyCodeCache
    except ImportError:
        return 0
    return len(getattr(PyCodeCache, "modules", []))


def _resolve_last_inductor_artifact_paths(
    baseline_module_count: int,
) -> tuple[str | None, str | None]:
    """Best-effort: return (so_path, cpp_source_path) for the most recent
    inductor-emitted artifact added to PyCodeCache.modules since the
    baseline snapshot. Returns (None, None) when nothing matches (e.g.
    cache hit -- no new module was loaded).
    """
    import os

    try:
        from torch._inductor.codecache import PyCodeCache
    except ImportError:
        return None, None

    modules = list(getattr(PyCodeCache, "modules", []))
    new_modules = modules[baseline_module_count:]
    if not new_modules:
        return None, None

    # Pick the latest non-None __file__. cpp_wrapper outputs are large
    # Python files containing the embedded C++ source as a raw string.
    cpp_source_path = None
    for mod in reversed(new_modules):
        path = getattr(mod, "__file__", None)
        if path and os.path.exists(path):
            cpp_source_path = path
            break

    if cpp_source_path is None:
        return None, None

    # Matching .so lives next to the .py under the same stem.
    so_path = None
    stem, _ = os.path.splitext(cpp_source_path)
    so_guess = stem + ".so"
    if os.path.exists(so_guess):
        so_path = so_guess
    return so_path, cpp_source_path


def _capture_common(fn, example_args, example_kwargs, fallback: bool):
    global _LAST_REPORT
    _LAST_GRAPH_STATS.clear()
    cm = force_all_fallback() if fallback else _stock_cpp_wrapper_config()

    baseline_module_count = _snapshot_pycodecache_module_count()
    t0 = time.perf_counter()
    with cm, inductor_config.patch({"post_grad_custom_pre_pass": _stats_post_grad_pass}):
        compiled = torch.compile(fn, backend="inductor", dynamic=True)
        # Prime the cache so the user's first call is hot.
        _ = compiled(*example_args, **example_kwargs)
    t1 = time.perf_counter()

    fx_nodes = _LAST_GRAPH_STATS.get("fx_nodes")
    so_path, cpp_source_path = _resolve_last_inductor_artifact_paths(baseline_module_count)
    _LAST_REPORT = {
        "variant": "fallback" if fallback else "stock",
        "capture_seconds": t1 - t0,
        "fx_node_count": fx_nodes,
        # In fallback variant, every aten op becomes a FallbackKernel, so
        # the count equals fx_node_count by construction.
        "fallback_node_count": fx_nodes if fallback else None,
        "so_path": so_path,
        "cpp_source_path": cpp_source_path,
    }
    return compiled
