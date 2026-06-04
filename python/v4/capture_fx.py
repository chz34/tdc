"""v4 capture_fx -- grab inductor's compiled FX host graph(s).

With inductor's fx_wrapper backend, the host wrapper is built as a
torch.fx.GraphModule (allocs + triton_kernel_wrapper_mutation launches + aten
fallbacks) -- i.e. inductor's fused output expressed as a graph. That gm is only
reachable at the WrapperFxCodegen.compile_graph hook; the upper torch.compile
object does not expose it (see DESIGN.md section 18).

capture_fx hooks compile_graph (via a clean registered subclass) and returns
both the normal torch.compile callable AND the captured gm(s), so the caller can
either run the compiled fn directly or process the fused-op gm by other means
(e.g. feed it to v2).

Design: docs/specs/2026-06-04-v4-fx-capture-design.md.

Works wherever inductor's host graph is FX-convertible: GPU/Triton fused kernels
convert directly; on CPU it works for graphs whose host code has no fused C++
kernel (e.g. all-fallback / pure-extern). A graph with a cpp_fused kernel will
raise inductor's "FX conversion only supports Triton kernels" at prime time --
we let that surface rather than pre-guarding on device.

    r = capture_fx(fn, *example_args)
    out = r.compiled(*example_args)   # run the normal fused result, or
    host_gm = r.gms[0]                # process the host fx graph yourself
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen


# CaptureFxWrapper instances are created fresh per inductor compile (.create()),
# so captured graphs cannot live on the instance -- they go to this context-local
# sink, set up by _capture_context and appended to by compile_graph.
_active_sink: "list | None" = None


class CaptureFxWrapper(WrapperFxCodegen):
    """Clean WrapperFxCodegen subclass that records each compiled gm.

    Only compile_graph is overridden. create() is inherited: it is a classmethod
    using `cls`, so it already instantiates this subclass (and its subgraph
    variant) correctly -- no override needed.
    """

    def compile_graph(self, gm):
        if _active_sink is not None:
            _active_sink.append(gm)
        # Return the normal compiled fn (gm.forward) unchanged, so the upper
        # torch.compile result still runs exactly as it would without capture.
        return super().compile_graph(gm)


@contextlib.contextmanager
def _capture_context(device: str):
    """Install CaptureFxWrapper + fx_wrapper for `device` and yield the gm sink.

    Restores config / registry / sink on exit, even on exception. Swaps
    process-global inductor registries -- not thread-safe (same constraint as
    the v3 fallback backend); fine for single-threaded capture.
    """
    global _active_sink
    import torch._inductor.config as inductor_config
    from torch._inductor.codegen.common import (
        device_codegens,
        init_backend_registration,
    )

    init_backend_registration()
    if device not in device_codegens:
        raise RuntimeError(f"no inductor backend registered for device {device!r}")
    dc = device_codegens[device]
    if dc.fx_wrapper_codegen is None:
        raise RuntimeError(f"device {device!r} has no fx_wrapper_codegen registered")

    saved_wrapper = dc.fx_wrapper_codegen
    saved_sink = _active_sink
    sink: list = []
    dc.fx_wrapper_codegen = CaptureFxWrapper
    _active_sink = sink
    try:
        # size_asserts / alignment_asserts emit RAW string lines
        # (assert_size_stride(...) / "# ... not aligned") that FxConverter
        # cannot consume (it only accepts structured WrapperLines), so an extern
        # kernel like sdpa/conv would abort FX conversion. Disable them so the
        # host graph is FX-convertible. Trade-off: the captured gm and the
        # returned compiled fn run without those runtime size/alignment checks.
        with inductor_config.patch(
            {"fx_wrapper": True, "size_asserts": False, "alignment_asserts": False}
        ):
            yield sink
    finally:
        dc.fx_wrapper_codegen = saved_wrapper
        _active_sink = saved_sink


def _infer_device(example_args: Any) -> str:
    for x in example_args:
        if isinstance(x, torch.Tensor):
            return x.device.type
    return "cpu"


@dataclass
class FxCaptureResult:
    """compiled: the normal torch.compile callable (run it to get fused output).
    gms: captured inductor host FX graphs, in compile_graph call order (forward,
    backward, recompiles, partitions...)."""

    compiled: Callable
    gms: list = field(default_factory=list)


def capture_fx(fn: Callable, *example_args: Any, dynamic: bool = True) -> FxCaptureResult:
    """Compile fn under fx_wrapper and capture the host gm(s).

    Primes the compile once with example_args (that is what triggers
    compile_graph and thus the capture). Returns both the compiled callable and
    the captured graphs; the caller decides what to do with each.
    """
    device = _infer_device(example_args)
    print("[v4.capture_fx] size_asserts & alignment_asserts disabled so the "
          "host graph is FX-convertible; the captured gm and compiled fn run "
          "without those runtime size/alignment checks.")
    with _capture_context(device) as sink:
        compiled = torch.compile(fn, backend="inductor", dynamic=dynamic)
        compiled(*example_args)  # prime -> triggers compile_graph -> capture
        gms = list(sink)
    return FxCaptureResult(compiled=compiled, gms=gms)
