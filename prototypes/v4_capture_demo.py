"""Minimal v4 capture_fx demo.

Unlike fx_wrapper_min_probe.py (which manually hooks WrapperFxCodegen.compile_graph
and sets up fx_wrapper / asserts by hand), this just calls the v4 interface:
capture_fx returns both the normal compiled callable AND the captured host FX
graph(s). The caller decides whether to run compiled or process the gm.

GPU/Triton only -- capture_fx raises on CPU (fx_wrapper can't convert C++ kernels).

Run:  python v4_capture_demo.py
"""
import os

import torch
import torch._inductor.config as inductor_config
import torch_dispatch_capture.v4 as tdcv4

from torch_dispatch_capture.v3.fallback_hijack import (
    NO_FUSION_CONFIG,
    force_all_fallback_lowerings,
)

#DEVICE = os.environ.get("TDC_DEVICE", "cuda")
DEVICE = "cpu"


def fn(a, b):
    # mm -> extern aten fallback; the rest fuses into a triton kernel.
    return torch.relu(a @ b + a) * 2.0


def main():
    a = torch.randn(128, 128, device=DEVICE)
    b = torch.randn(128, 128, device=DEVICE)

    with force_all_fallback_lowerings(), inductor_config.patch(
        {"fx_wrapper": True, "cpp_wrapper": False,
            "size_asserts": False, "alignment_asserts": False, **NO_FUSION_CONFIG}
    ):
        result = tdcv4.capture_fx(fn, a, b)  # -> FxCaptureResult(compiled, gms)

        # Choice 1: run the normal fused result.
        out = result.compiled(a, b)
        print("compiled output sum:", float(out.sum()))

    # Choice 2: inspect / process the captured host gm(s) -- built-in fx dumps.
    print(f"\ncaptured {len(result.gms)} host gm(s)")
    for i, gm in enumerate(result.gms):
        print(f"\n===== gm[{i}] FX IR (target / args / kwargs) =====")
        print(gm.graph)
        print(f"===== gm[{i}] readable (with shapes) =====")
        gm.print_readable()


if __name__ == "__main__":
    main()
