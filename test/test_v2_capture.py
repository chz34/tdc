"""End-to-end validation of the v2.capture() direct-replay API.

Mirrors test_v2_basic.py's cases but uses v2.capture(fn, *example_args)
to obtain a callable that skips Dynamo at call time. Each case captures
once, then calls the result against multiple input shapes to confirm
the SymInt extraction recipes carry over correctly.
"""
import unittest

import torch
import torch_dispatch_capture.v2 as tdcv2


class TestV2Capture(unittest.TestCase):

    def setUp(self):
        # Each test captures its own function closure; reset Dynamo's
        # global cache so tests don't share compile artifacts (which
        # leak SymInt symbols across tests and break recipe building).
        torch._dynamo.reset()

    def test_simple_arithmetic(self):
        def fn(x, y):
            return x * 2.0 + y - 1.5
        captured = tdcv2.capture(fn, torch.randn(4, 5), torch.randn(4, 5))
        for shape in [(4, 5), (3, 7), (2, 9)]:
            x = torch.randn(*shape)
            y = torch.randn(*shape)
            ref = x * 2.0 + y - 1.5
            out = captured(x, y)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_shape_derived_view(self):
        def fn(x):
            return x.view(x.shape[0] // 2, 2, -1)
        captured = tdcv2.capture(fn, torch.randn(8, 6))
        for shape in [(8, 6), (12, 5), (10, 4)]:
            x = torch.randn(*shape)
            ref = x.view(x.shape[0] // 2, 2, -1)
            out = captured(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_attention_qk(self):
        N_HEADS = 8
        def qk(q, k):
            B, S, H = q.shape
            h_dim = H // N_HEADS
            q2 = q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3)
            k2 = k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1)
            return torch.matmul(q2, k2)

        captured = tdcv2.capture(qk, torch.randn(2, 4, 32), torch.randn(2, 4, 32))
        for B, S in [(2, 4), (3, 7), (5, 11)]:
            H = 32
            q = torch.randn(B, S, H)
            k = torch.randn(B, S, H)
            h_dim = H // N_HEADS
            ref = torch.matmul(
                q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3),
                k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1),
            )
            out = captured(q, k)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_slice_dynamic_upper_bound(self):
        def fn(x, y):
            return x[0, :] + y[0, :x.shape[1]]
        captured = tdcv2.capture(fn, torch.randn(3, 5), torch.randn(3, 8))
        for sx, sy in [((3, 5), (3, 8)), ((4, 6), (4, 10)), ((2, 3), (2, 7))]:
            x = torch.randn(*sx)
            y = torch.randn(*sy)
            ref = x[0, :] + y[0, :x.shape[1]]
            out = captured(x, y)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_multi_output_max(self):
        def fn(x):
            v, i = x.max(dim=-1)
            return v + i.float()
        captured = tdcv2.capture(fn, torch.randn(4, 5))
        for shape in [(4, 5), (6, 8), (3, 11)]:
            x = torch.randn(*shape)
            ref_v, ref_i = x.max(dim=-1)
            ref = ref_v + ref_i.float()
            out = captured(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_split_then_recombine(self):
        def fn(x):
            a, b, c = torch.split(x, x.shape[0] // 3, dim=0)
            return a + b + c
        captured = tdcv2.capture(fn, torch.randn(9, 4))
        for n in [9, 12, 15]:
            x = torch.randn(n, 4)
            a, b, c = torch.split(x, n // 3, dim=0)
            ref = a + b + c
            out = captured(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))


if __name__ == "__main__":
    unittest.main()
