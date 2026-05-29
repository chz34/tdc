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

    def test_isolate_fresh_fn_partitions_dynamo_cache(self):
        """Mirrors the v2_benchmark.build_variants() scenario: capture the
        SAME fn under stock then fallback with NO torch._dynamo.reset()
        between them. Without isolate_fresh_fn, Dynamo's per-code-object
        cache_entry_list (extra_state.cpp) returns the first capture's
        artifact for the second, silently collapsing the two variants
        into one. isolate_fresh_fn gives each capture a fresh code
        object so each gets its own ExtraState slot."""
        def fn(x, y):
            return ((x + y) * 0.5 - 1.0).relu()

        x = torch.randn(4, 5, device=DEVICE)
        y = torch.randn(4, 5, device=DEVICE)

        torch._dynamo.reset()
        tdcv3.capture(tdcv3.isolate_fresh_fn(fn), x, y)
        stock = tdcv3.last_capture_report()
        tdcv3.capture_fallback(tdcv3.isolate_fresh_fn(fn), x, y)
        fb = tdcv3.last_capture_report()

        self.assertIsNotNone(stock["cpp_source_path"])
        self.assertIsNotNone(fb["cpp_source_path"])
        self.assertNotEqual(
            stock["cpp_source_path"], fb["cpp_source_path"],
            "stock and fallback returned the same cpp_source_path -- "
            "isolate_fresh_fn did not partition Dynamo's cache",
        )
        self.assertGreater(stock["fused_kernel_count"], 0)
        self.assertEqual(fb["fused_kernel_count"], 0)

    def test_stock_and_fallback_isolate_in_same_process(self):
        """Running stock and fallback back-to-back must produce DIFFERENT
        artifacts (different FxGraphCache hashes). This is what protects
        us against the cache returning stock output to a fallback call
        (or vice versa) when both variants compile the same FX graph.

        Regression guard: if a future edit removes the
        epilogue_fusion / max_fusion_size / freezing flags from
        force_all_fallback's config patch, the cache hashes would
        collapse and one variant would silently return the other's
        artifact. This test fails in that case."""
        def fn(x, y):
            return ((x + y) * 0.5 - 1.0).relu()

        x = torch.randn(4, 5, device=DEVICE)
        y = torch.randn(4, 5, device=DEVICE)

        # Order: stock -> fallback
        torch._dynamo.reset()
        tdcv3.capture(fn, x, y)
        stock_path_1 = tdcv3.last_capture_report()["cpp_source_path"]

        torch._dynamo.reset()
        tdcv3.capture_fallback(fn, x, y)
        fb_path = tdcv3.last_capture_report()["cpp_source_path"]

        # Order: fallback -> stock (repeat to ensure stock path
        # reproduces from cache, not from fallback's cached entry).
        torch._dynamo.reset()
        tdcv3.capture(fn, x, y)
        stock_path_2 = tdcv3.last_capture_report()["cpp_source_path"]

        self.assertNotEqual(
            stock_path_1, fb_path,
            "stock and fallback share the same cpp_source_path -- "
            "FxGraphCache key collision, fallback config flags not "
            "sufficient to distinguish from stock",
        )
        self.assertEqual(
            stock_path_1, stock_path_2,
            "stock capture reproduces a different path across calls -- "
            "indicates non-deterministic cache key or state pollution",
        )

    def test_fused_kernel_count_distinguishes_variants(self):
        """Strongest evidence that fallback really skips inductor fusion:
        inspect the generated cpp_wrapper source and count fused-kernel
        DEFINITIONS. Fallback must have zero; stock must have at least
        one for a workload with obvious fusion opportunities."""
        def fn(x, y):
            return ((x + y) * 0.5 - 1.0).relu()

        x = torch.randn(4, 5, device=DEVICE)
        y = torch.randn(4, 5, device=DEVICE)

        torch._dynamo.reset()
        tdcv3.capture_fallback(fn, x, y)
        fb = tdcv3.last_capture_report()
        self.assertEqual(
            fb["fused_kernel_count"], 0,
            f"fallback variant produced {fb['fused_kernel_count']} fused "
            f"kernel(s); check {fb['cpp_source_path']}"
        )

        torch._dynamo.reset()
        tdcv3.capture(fn, x, y)
        st = tdcv3.last_capture_report()
        self.assertIsNotNone(st["fused_kernel_count"])
        self.assertGreater(
            st["fused_kernel_count"], 0,
            f"stock variant on a fusable workload produced 0 fused kernels; "
            f"check {st['cpp_source_path']}"
        )

    def test_capture_report_paths_exist_on_disk(self):
        import os

        def fn(x):
            return x * 2.0 + 1.0

        x = torch.randn(4, 5, device=DEVICE)
        tdcv3.capture_fallback(fn, x)

        rep = tdcv3.last_capture_report()
        # cpp_source_path is the inductor-emitted Python wrapper file that
        # embeds the cpp source. It must exist on disk when capture succeeded.
        self.assertIsNotNone(rep["cpp_source_path"])
        self.assertTrue(os.path.exists(rep["cpp_source_path"]))


class TestV3DynamicShape(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_capture_once_then_call_with_three_shapes(self):
        def fn(x):
            return (x * 2.0 + 1.0).relu()

        x = torch.randn(4, 5, device=DEVICE)
        captured = tdcv3.capture_fallback(fn, x)

        counters = torch._dynamo.utils.counters
        baseline = counters.get("stats", {}).get("calls_captured", 0)

        for shape in [(4, 5), (8, 5), (12, 5)]:
            xi = torch.randn(*shape, device=DEVICE)
            ref = (xi * 2.0 + 1.0).relu()
            out = captured(xi)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref, atol=1e-3, rtol=1e-3))

        delta = counters.get("stats", {}).get("calls_captured", 0) - baseline
        # Under dynamic=True the trio of shapes should not provoke
        # additional Dynamo captures. Allow 1 to cushion guard quirks
        # but not 3.
        self.assertLessEqual(delta, 1, f"unexpected recompiles: {delta}")


class TestV3Kwargs(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_kwargs_passthrough(self):
        def fn(x, *, scale):
            return x * scale + 1.0

        x = torch.randn(4, 5, device=DEVICE)
        captured = tdcv3.capture_fallback(fn, x, scale=torch.tensor(2.5, device=DEVICE))

        new_x = torch.randn(4, 5, device=DEVICE)
        new_scale = torch.tensor(3.0, device=DEVICE)
        ref = new_x * new_scale + 1.0
        out = captured(new_x, scale=new_scale)
        self.assertTrue(torch.allclose(out, ref, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
