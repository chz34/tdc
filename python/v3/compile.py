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
    with inductor_config.patch({
        "cpp_wrapper": True,
        "triton.cudagraphs": False,
    }):
        yield


def _capture_common(fn, example_args, example_kwargs, fallback: bool):
    global _LAST_REPORT
    _LAST_GRAPH_STATS.clear()
    cm = force_all_fallback() if fallback else _stock_cpp_wrapper_config()

    t0 = time.perf_counter()
    with cm, inductor_config.patch({"post_grad_custom_pre_pass": _stats_post_grad_pass}):
        compiled = torch.compile(fn, backend="inductor", dynamic=True)
        # Prime the cache so the user's first call is hot.
        _ = compiled(*example_args, **example_kwargs)
    t1 = time.perf_counter()

    fx_nodes = _LAST_GRAPH_STATS.get("fx_nodes")
    _LAST_REPORT = {
        "variant": "fallback" if fallback else "stock",
        "capture_seconds": t1 - t0,
        "fx_node_count": fx_nodes,
        # In fallback variant, every aten op becomes a FallbackKernel, so
        # the count equals fx_node_count by construction.
        "fallback_node_count": fx_nodes if fallback else None,
    }
    return compiled
