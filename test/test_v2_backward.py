"""v2.capture(allow_grad=True) — backward replay validation.

The test cases here exercise typical forward-then-backward patterns
that come up in training loops. Each test:
  1. Captures fn once with example_args including requires_grad=True
     tensors. capture(allow_grad=True) materialises both the AOT fw
     and bw graphs, builds two v2 Traces, and wraps them in a
     torch.autograd.Function so loss.backward() drives the bw trace.
  2. Runs the captured callable against fresh inputs (possibly
     different shape from example_args, where the AOT graph is
     shape-polymorphic) and verifies the loss + each requires_grad
     input's grad matches eager.
"""
import unittest

import torch
import torch_dispatch_capture.v2 as tdcv2


class TestV2Backward(unittest.TestCase):

    def setUp(self):
        torch._dynamo.reset()

    def _run_eager_grad(self, fn, *args):
        cloned = []
        for a in args:
            if isinstance(a, torch.Tensor) and a.requires_grad:
                cloned.append(a.detach().clone().requires_grad_(True))
            else:
                cloned.append(a)
        loss = fn(*cloned)
        loss.backward()
        grads = tuple(
            c.grad if isinstance(c, torch.Tensor) and c.requires_grad else None
            for c in cloned
        )
        return loss, grads

    def test_scalar_loss_single_input(self):
        def fn(x):
            return (x * 2.0).sum()

        ex = torch.randn(4, 5, requires_grad=True)
        captured = tdcv2.capture(fn, ex, allow_grad=True)

        for shape in [(4, 5), (6, 3), (8, 8)]:
            x = torch.randn(*shape, requires_grad=True)
            loss = captured(x)
            loss.backward()
            ref_loss, (ref_grad,) = self._run_eager_grad(fn, x)
            self.assertTrue(torch.allclose(loss, ref_loss))
            self.assertTrue(torch.allclose(x.grad, ref_grad))

    def test_attention_qk_loss(self):
        N_HEADS = 8

        def attn_loss(q, k):
            B, S, H = q.shape
            h_dim = H // N_HEADS
            q2 = q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3)
            k2 = k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1)
            return torch.matmul(q2, k2).sum()

        q_ex = torch.randn(2, 4, 32, requires_grad=True)
        k_ex = torch.randn(2, 4, 32, requires_grad=True)
        captured = tdcv2.capture(attn_loss, q_ex, k_ex, allow_grad=True)

        for B, S in [(2, 4), (3, 7), (5, 11)]:
            q = torch.randn(B, S, 32, requires_grad=True)
            k = torch.randn(B, S, 32, requires_grad=True)
            loss = captured(q, k)
            loss.backward()
            ref_loss, (ref_qg, ref_kg) = self._run_eager_grad(attn_loss, q, k)
            self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-5))
            self.assertTrue(torch.allclose(q.grad, ref_qg, atol=1e-5))
            self.assertTrue(torch.allclose(k.grad, ref_kg, atol=1e-5))

    def test_swiglu_loss(self):
        N_HEADS = 8  # unused, just keeps module-level consistency

        def swiglu_loss(x, w_gate, w_up, w_down):
            import torch.nn.functional as F
            gate = F.linear(x, w_gate)
            up = F.linear(x, w_up)
            return F.linear(F.silu(gate) * up, w_down).sum()

        x_ex = torch.randn(2, 4, 16, requires_grad=True)
        w_g = torch.randn(32, 16, requires_grad=True)
        w_u = torch.randn(32, 16, requires_grad=True)
        w_d = torch.randn(16, 32, requires_grad=True)
        captured = tdcv2.capture(swiglu_loss, x_ex, w_g, w_u, w_d, allow_grad=True)

        # Vary only B/S; W shapes stay fixed (H_in/H_out specialised).
        for B, S in [(2, 4), (3, 6), (1, 8)]:
            x = torch.randn(B, S, 16, requires_grad=True)
            wg = w_g.detach().clone().requires_grad_(True)
            wu = w_u.detach().clone().requires_grad_(True)
            wd = w_d.detach().clone().requires_grad_(True)
            loss = captured(x, wg, wu, wd)
            loss.backward()
            ref_loss, (ref_x, ref_wg, ref_wu, ref_wd) = self._run_eager_grad(
                swiglu_loss, x, wg, wu, wd)
            self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-3))
            for got, ref in zip(
                (x.grad, wg.grad, wu.grad, wd.grad),
                (ref_x, ref_wg, ref_wu, ref_wd),
            ):
                self.assertTrue(torch.allclose(got, ref, atol=1e-3))

    def test_requires_grad_required(self):
        """allow_grad=True without any grad input should error early."""
        def fn(x):
            return x.sum()
        x = torch.randn(4, 5)  # NO requires_grad
        with self.assertRaisesRegex(RuntimeError, "requires_grad=True"):
            tdcv2.capture(fn, x, allow_grad=True)


if __name__ == "__main__":
    unittest.main()
