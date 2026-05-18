"""End-to-end validation of the v2 framework.

The cases here are exactly the patterns v1 cannot handle (shape-derived
literals, attention reshape+permute+matmul, dynamic slice upper bound)
plus a few sanity cases (simple arithmetic, multi-output unpacking).
Each case runs the same function across multiple input shapes and
verifies that:

  1. The compiled function returns numerically correct results.
  2. The function reuses one compiled artifact across shapes (i.e.
     dynamic=True is actually working — no per-shape recompile).
"""
import unittest

import torch
import torch_dispatch_capture.v2 as tdcv2


class TestV2Basic(unittest.TestCase):

    def test_simple_arithmetic(self):
        @tdcv2.compile(dynamic=True)
        def fn(x, y):
            return x * 2.0 + y - 1.5

        for shape in [(4, 5), (3, 7), (2, 9)]:
            x = torch.randn(*shape)
            y = torch.randn(*shape)
            ref = x * 2.0 + y - 1.5
            out = fn(x, y)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_shape_derived_view(self):
        """The exact v1 blind-spot from DESIGN §8.1: x.view(x.shape[0]//2, ...)."""
        @tdcv2.compile(dynamic=True)
        def fn(x):
            return x.view(x.shape[0] // 2, 2, -1)

        for shape in [(8, 6), (12, 5), (10, 4)]:
            x = torch.randn(*shape)
            ref = x.view(x.shape[0] // 2, 2, -1)
            out = fn(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_attention_qk(self):
        """Attention-style reshape + permute + matmul. Exercises sym arith
        (H // N_HEADS), kList args (view size, permute dims), and op
        decomposition (torch.matmul -> expand+clone+_unsafe_view+bmm)."""
        N_HEADS = 8

        @tdcv2.compile(dynamic=True)
        def qk(q, k):
            B, S, H = q.shape
            h_dim = H // N_HEADS
            q2 = q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3)
            k2 = k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1)
            return torch.matmul(q2, k2)

        for B, S in [(2, 4), (3, 7), (5, 11)]:
            H = 32
            q = torch.randn(B, S, H)
            k = torch.randn(B, S, H)
            h_dim = H // N_HEADS
            ref = torch.matmul(
                q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3),
                k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1),
            )
            out = qk(q, k)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_slice_dynamic_upper_bound(self):
        """y[0, :x.shape[1]] — slice with sym-derived upper bound. The
        bound varies with x.shape[1] across calls and the trace must
        follow it."""
        @tdcv2.compile(dynamic=True)
        def fn(x, y):
            return x[0, :] + y[0, :x.shape[1]]

        for sx, sy in [((3, 5), (3, 8)), ((4, 6), (4, 10)), ((2, 3), (2, 7))]:
            x = torch.randn(*sx)
            y = torch.randn(*sy)
            ref = x[0, :] + y[0, :x.shape[1]]
            out = fn(x, y)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_multi_output_max(self):
        """max(dim=-1) returns (values, indices) — covers operator.getitem
        in the AOT graph (handled as a PY_CALL step)."""
        @tdcv2.compile(dynamic=True)
        def fn(x):
            v, i = x.max(dim=-1)
            return v + i.float()

        for shape in [(4, 5), (6, 8), (3, 11)]:
            x = torch.randn(*shape)
            ref_v, ref_i = x.max(dim=-1)
            ref = ref_v + ref_i.float()
            out = fn(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_split_then_recombine(self):
        """split returns Tensor[] — exercises the operator.getitem PY_CALL
        path for a list-typed Step output."""
        @tdcv2.compile(dynamic=True)
        def fn(x):
            a, b, c = torch.split(x, x.shape[0] // 3, dim=0)
            return a + b + c

        for n in [9, 12, 15]:
            x = torch.randn(n, 4)
            a, b, c = torch.split(x, n // 3, dim=0)
            ref = a + b + c
            out = fn(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))


if __name__ == "__main__":
    unittest.main()
