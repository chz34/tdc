"""End-to-end and unit tests for v3 (Inductor cpp_wrapper probe)."""
import sys
import unittest
from pathlib import Path

import torch
import torch._inductor.lowering as _lowering
import torch_dispatch_capture.v3 as tdcv3

# Allow importing test/_device.py when run from repo root or test/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _device import DEVICE  # noqa: E402


class TestForceAllFallback(unittest.TestCase):
    def test_restores_lowerings_on_exit(self):
        before = dict(_lowering.lowerings)
        with tdcv3.force_all_fallback():
            inside = dict(_lowering.lowerings)
        after = dict(_lowering.lowerings)
        self.assertEqual(before.keys(), after.keys())
        # At least one entry must have been rewritten to a fallback handler.
        patched_op_count = sum(
            1
            for k, v in inside.items()
            if isinstance(k, torch._ops.OpOverload)
            and getattr(v, "_is_fallback_handler", False)
        )
        self.assertGreater(patched_op_count, 50)
        # And the original handler must be restored.
        for k, v in before.items():
            self.assertIs(after[k], v)


class TestV3CaptureStock(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_smoke_stock_pointwise(self):
        def fn(x, y):
            return x * 2.0 + y - 1.5

        x = torch.randn(4, 5, device=DEVICE)
        y = torch.randn(4, 5, device=DEVICE)
        captured = tdcv3.capture(fn, x, y)

        ref = x * 2.0 + y - 1.5
        out = captured(x, y)
        self.assertTrue(torch.allclose(out, ref, atol=1e-3, rtol=1e-3))


class TestV3CaptureFallback(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_smoke_fallback_pointwise(self):
        def fn(x, y):
            return x * 2.0 + y - 1.5

        x = torch.randn(4, 5, device=DEVICE)
        y = torch.randn(4, 5, device=DEVICE)
        captured = tdcv3.capture_fallback(fn, x, y)

        ref = x * 2.0 + y - 1.5
        out = captured(x, y)
        # AOT decomp still applies, so the same 1e-3 tolerance v2 uses applies here.
        self.assertTrue(torch.allclose(out, ref, atol=1e-3, rtol=1e-3))
        self.assertEqual(tdcv3.last_capture_report()["variant"], "fallback")


class TestV3CaptureReport(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_fallback_node_count_matches_fx_node_count(self):
        def fn(q, k):
            return torch.matmul(q, k.transpose(-1, -2))

        q = torch.randn(2, 4, 8, device=DEVICE)
        k = torch.randn(2, 4, 8, device=DEVICE)
        tdcv3.capture_fallback(fn, q, k)

        rep = tdcv3.last_capture_report()
        self.assertGreater(rep["fx_node_count"], 0)
        self.assertEqual(rep["fallback_node_count"], rep["fx_node_count"])


if __name__ == "__main__":
    unittest.main()
