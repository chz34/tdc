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

DEVICE = os.environ.get("TDC_DEVICE", "cuda")

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

    inductor_config.fx_wrapper = True
    a = torch.randn(128, 128, device=DEVICE)
    b = torch.randn(128, 128, device=DEVICE)

    compiled = torch.compile(fn, backend="inductor", dynamic=True)
    out = compiled(a, b)
    assert torch.allclose(out, fn(a, b), atol=1e-2), "numeric mismatch"

    gm = _captured.get("gm")
    if gm is None:
        raise SystemExit("no fx_wrapper gm captured (compile may have failed)")

    print("=== captured fx_wrapper GraphModule ===")
    print(gm.code.strip())
    print("\n=== nodes (op / name / target) ===")
    for n in gm.graph.nodes:
        target = (
            getattr(n.target, "__name__", n.target)
            if n.op == "call_function"
            else n.op
        )
        print(f"  {n.op:14s} {n.name:18s} {target}")


if __name__ == "__main__":
    main()
