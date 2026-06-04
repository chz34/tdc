"""Tests for v4 capture_fx (grab inductor's fx_wrapper host graph).

GPU/Triton only -- fx_wrapper cannot convert CPU C++ kernels, so these skip on
CPU. See docs/specs/2026-06-04-v4-fx-capture-design.md.
"""
import sys
import unittest
from pathlib import Path

import torch

import torch_dispatch_capture.v4 as tdcv4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import DEVICE  # noqa: E402

from torch._higher_order_ops.triton_kernel_wrap import (  # noqa: E402
    triton_kernel_wrapper_mutation,
)


def _fn(a, b):
    # mm -> extern aten fallback; the rest fuses into a triton kernel.
    return torch.relu(a @ b + a) * 2.0


@unittest.skipUnless(DEVICE in ("cuda", "xpu"), "fx_wrapper is Triton-only")
class TestCaptureFx(unittest.TestCase):
    def _inputs(self):
        a = torch.randn(128, 128, device=DEVICE)
        b = torch.randn(128, 128, device=DEVICE)
        return a, b

    def test_result_shape(self):
        a, b = self._inputs()
        r = tdcv4.capture_fx(_fn, a, b)
        self.assertIsInstance(r, tdcv4.FxCaptureResult)
        self.assertTrue(callable(r.compiled))
        self.assertGreater(len(r.gms), 0, "no host gm captured")

    def test_compiled_runs_correctly(self):
        a, b = self._inputs()
        ref = _fn(a, b)
        r = tdcv4.capture_fx(_fn, a, b)
        out = r.compiled(a, b)
        self.assertTrue(torch.allclose(out, ref, atol=1e-2, rtol=1e-2))

    def test_gm_is_graphmodule_with_triton_launch(self):
        a, b = self._inputs()
        r = tdcv4.capture_fx(_fn, a, b)
        gm = r.gms[0]
        self.assertIsInstance(gm, torch.fx.GraphModule)
        # The fused pointwise becomes a triton_kernel_wrapper_mutation HOP node.
        has_triton = any(
            n.op == "call_function" and n.target is triton_kernel_wrapper_mutation
            for n in gm.graph.nodes
        )
        self.assertTrue(has_triton, "expected a triton kernel launch node in the gm")

    def test_restores_registry_and_config_on_exit(self):
        import torch._inductor.config as ic
        from torch._inductor.codegen.common import (
            device_codegens,
            init_backend_registration,
        )

        init_backend_registration()
        before_wrapper = device_codegens[DEVICE].fx_wrapper_codegen
        before_flag = ic.fx_wrapper
        a, b = self._inputs()
        tdcv4.capture_fx(_fn, a, b)
        self.assertIs(device_codegens[DEVICE].fx_wrapper_codegen, before_wrapper)
        self.assertEqual(ic.fx_wrapper, before_flag)


# Note: capture_fx no longer pre-guards on device. On CPU a graph with a
# cpp_fused kernel (like _fn) raises inductor's "FX conversion only supports
# Triton kernels" at prime time; graphs without one (all-fallback / pure-extern)
# convert fine. We don't assert that device-specific behavior here.


if __name__ == "__main__":
    unittest.main()
