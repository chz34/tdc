"""Tests for v3 fallback backend (make_fallback_backend).

See docs/specs/2026-06-03-v3-fallback-backend-design.md.
"""
import sys
import unittest
from pathlib import Path

import torch
from torch._inductor.utils import run_and_get_cpp_code

import torch_dispatch_capture.v3 as tdcv3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import DEVICE  # noqa: E402


def _compile_and_capture(fn, args, mode):
    """Compile fn under the fallback backend and return (output, cpp_code)."""
    torch._dynamo.reset()
    compiled = torch.compile(
        fn, backend=tdcv3.make_fallback_backend(mode), dynamic=True
    )
    return run_and_get_cpp_code(compiled, *args)


def _mm_pointwise(a, b):
    return torch.relu(torch.mm(a, b) * 2.0 + 1.0)


def _pointwise(a, b):
    return torch.relu(a * b + a) * 2.0


def _data_dependent(x):
    # nonzero has a data-dependent output size; must go through fallback dispatch.
    return torch.nonzero(x)


class TestFallbackBackendNumerics(unittest.TestCase):
    """All-fallback cpp_wrapper must stay numerically correct vs eager."""

    def _check(self, fn, args, mode):
        ref = fn(*args)
        out, _ = _compile_and_capture(fn, args, mode)
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-4, rtol=1e-4),
            f"{fn.__name__} mismatch in {mode} mode",
        )

    def test_mm_pointwise_boxed(self):
        a = torch.randn(32, 16, device=DEVICE)
        b = torch.randn(16, 32, device=DEVICE)
        self._check(_mm_pointwise, (a, b), "boxed")

    def test_mm_pointwise_stock(self):
        a = torch.randn(32, 16, device=DEVICE)
        b = torch.randn(16, 32, device=DEVICE)
        self._check(_mm_pointwise, (a, b), "stock")

    def test_pointwise_boxed(self):
        a = torch.randn(64, 64, device=DEVICE)
        b = torch.randn(64, 64, device=DEVICE)
        self._check(_pointwise, (a, b), "boxed")

    def test_data_dependent_boxed(self):
        x = torch.randint(0, 2, (64,), device=DEVICE)
        ref = _data_dependent(x)
        out, _ = _compile_and_capture(_data_dependent, (x,), "boxed")
        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(torch.equal(out, ref))


@unittest.skipUnless(DEVICE == "cpu", "cpp_wrapper codegen markers are CPU-specific")
class TestFallbackBackendCodegen(unittest.TestCase):
    """Inspect the generated host C++ to confirm the intended dispatch shape."""

    def test_boxed_uses_call_dispatcher(self):
        a = torch.randn(32, 16)
        b = torch.randn(16, 32)
        _, code = _compile_and_capture(_mm_pointwise, (a, b), "boxed")
        # Boxed mode forces every fallback op through the by-name dispatcher.
        self.assertIn("aoti_torch_call_dispatcher", code)

    def test_no_fusion_in_either_mode(self):
        # force_all_fallback disables fusion, so no cpp_fused_* kernels should
        # be emitted in either mode -- the wrapper is pure op dispatch.
        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        for mode in ("boxed", "stock"):
            _, code = _compile_and_capture(_pointwise, (a, b), mode)
            self.assertNotIn("cpp_fused_", code, f"fusion leaked in {mode} mode")

    def test_restores_device_codegen_on_exit(self):
        from torch._inductor.codegen.common import (
            device_codegens,
            init_backend_registration,
        )

        init_backend_registration()
        before = device_codegens["cpu"].cpp_wrapper_codegen
        a = torch.randn(8, 8)
        b = torch.randn(8, 8)
        _compile_and_capture(_pointwise, (a, b), "boxed")
        after = device_codegens["cpu"].cpp_wrapper_codegen
        self.assertIs(before, after)


if __name__ == "__main__":
    unittest.main()
