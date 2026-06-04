"""Minimal fx_wrapper probe: capture inductor's compiled FX host graph.

With config.fx_wrapper on, inductor builds its host wrapper as a
torch.fx.GraphModule (allocs + triton_kernel_wrapper_mutation launches +
aten fallbacks). This hooks WrapperFxCodegen.compile_graph to grab that gm
and prints it.

Requires a Triton-capable device (cuda/xpu) -- fx_wrapper cannot convert CPU
C++ kernels ("FX conversion only supports Triton kernels").

Run:  python fx_wrapper_min_probe.py
"""
import os

import torch
import torch._inductor.config as inductor_config
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen
from torch_dispatch_capture.v3.fallback_hijack import (
    NO_FUSION_CONFIG,
    force_all_fallback_lowerings,
)

#DEVICE = os.environ.get("TDC_DEVICE", "cuda")
DEVICE = "cpu"

# Hook the FX wrapper codegen to capture the GraphModule it builds.
_captured = {}
_orig_compile_graph = WrapperFxCodegen.compile_graph


def _hook(self, gm):
    _captured["gm"] = gm
    return _orig_compile_graph(self, gm)


WrapperFxCodegen.compile_graph = _hook


def fn(a, b):
    # mm -> extern aten fallback; the rest fuses into one triton kernel.
    return torch.relu(a @ b + a) * 2.0


def main():
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit("needs a GPU; fx_wrapper is Triton-only (set TDC_DEVICE)")

    a = torch.randn(128, 128, device=DEVICE)
    b = torch.randn(128, 128, device=DEVICE)

    # Decoupled: fallback lowerings + an fx_wrapper config. Crucially keep
    # cpp_wrapper=False -- if it leaks True (as the old bundled
    # force_all_fallback did), FallbackKernel takes the cpp_wrapper
    # runtime-dispatch path and emits a raw Python op call the FX converter
    # cannot consume.
    #
    # size_asserts / alignment_asserts must be OFF: inductor emits those as RAW
    # string lines (assert_size_stride(...) / "# ... not aligned"), and
    # FxConverter only accepts structured WrapperLines -- a raw line aborts FX
    # conversion. Disabling them is what lets an all-fallback graph convert on
    # CPU. Trade-off: the captured gm runs without those runtime checks.
    print("[fx_wrapper_min_probe] size_asserts & alignment_asserts disabled "
          "(raw assert/comment lines are not FX-convertible)")
    with force_all_fallback_lowerings():
        with inductor_config.patch(
            {"fx_wrapper": True, "cpp_wrapper": False,
             "size_asserts": False, "alignment_asserts": False, **NO_FUSION_CONFIG}
        ):
            compiled = torch.compile(fn, backend="inductor", dynamic=True)
            # Prime INSIDE the patch -- torch.compile is lazy, the actual
            # compile (and thus the fx_wrapper capture) happens on first call.
            out = compiled(a, b)
            assert torch.allclose(out, fn(a, b), atol=1e-2), "numeric mismatch"

    gm = _captured.get("gm")
    if gm is None:
        raise SystemExit("no fx_wrapper gm captured (compile may have failed)")

    # Built-in fx debug dumps -- no manual node parsing needed.
    print("=== FX IR (print(gm.graph): node / target / args / kwargs) ===")
    print(gm.graph)  # richest single view; includes triton HOP launch kwargs
    print("\n=== readable module (with shapes/dtypes) ===")
    gm.print_readable()


if __name__ == "__main__":
    main()
