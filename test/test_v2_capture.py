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

    def test_varied_python_int_arg(self):
        """A plain Python int positional arg must be routed at every
        replay, not frozen at capture. The LLaMA KV-cache pattern
        cache[:, start:start+seqlen] = x relies on this: capture sees
        start_pos=0, but the user calls with start_pos=8, 17, ... and
        v2 has to thread the new value through.

        Pre-fix: _build_recipe_specs put non-shape SymInts into pre_binds
        and the captured value (e.g. 0) was used on every replay --
        silently wrong outputs. Fix routes such SymInts as ("I", arg_idx)
        runtime specs that read args[i] at call time."""
        def fn(x, idx):
            buf = torch.zeros(2, 32)
            buf[:, idx : idx + x.shape[1]] = x
            return buf.clone()

        x = torch.randn(2, 4)
        # wrapper=False: the aot_module path (wrapper=True) has a
        # separate, unfixed problem with non-Tensor scalar args (AOT
        # without Dynamo inlines them as literals and leaves an
        # orphan placeholder); routing fix here covers the
        # Dynamo path only.
        captured = tdcv2.capture(fn, x, 0, wrapper=False)

        for idx in (0, 4, 8, 17, 28):
            ref = fn(x, idx)
            out = captured(x, idx)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref),
                            f"idx={idx} mismatch: max diff "
                            f"{(out-ref).abs().max().item():.3e}")

    def test_varied_int_and_varied_shape_together(self):
        """Combine the two flavours of dynamic-spec routing: a Python
        int arg AND a tensor with varying shape, in the same call.
        Catches any interference between the user_scalar_queue and the
        shape-SymInt routing in _build_recipe_specs."""
        def fn(x, k):
            return x[:k] * 2.0

        x = torch.randn(8, 5)
        captured = tdcv2.capture(fn, x, 3, wrapper=False)

        for k in (1, 2, 3, 5, 7):
            for shape in [(8, 5), (12, 5), (6, 5)]:
                x = torch.randn(*shape)
                if k > x.shape[0]:
                    continue
                ref = x[:k] * 2.0
                out = captured(x, k)
                self.assertEqual(out.shape, ref.shape)
                self.assertTrue(torch.allclose(out, ref),
                                f"k={k}, shape={shape} mismatch")


if __name__ == "__main__":
    unittest.main()
