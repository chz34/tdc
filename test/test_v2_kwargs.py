"""v2.capture supports keyword-argument call sites.

torch.compile is transparent to Python's call semantics: kwargs and
keyword-only parameters work just like positional ones. v2.capture
preserves the same contract — at capture time the user can pass
example_args via kwargs (`v2.capture(fn, x=..., y=...)`), and at call
time the returned callable accepts positional, kwarg, or mixed
invocation in the same shape as the original fn.
"""
import unittest

import torch
import torch_dispatch_capture.v2 as tdcv2


class TestV2Kwargs(unittest.TestCase):

    def setUp(self):
        torch._dynamo.reset()

    def test_capture_with_kwargs_call_positionally(self):
        def fn(x, y):
            return x * 2 + y

        captured = tdcv2.capture(fn, x=torch.randn(3, 4), y=torch.randn(3, 4))
        x = torch.randn(5, 6)
        y = torch.randn(5, 6)
        out = captured(x, y)
        self.assertTrue(torch.allclose(out, fn(x, y)))

    def test_capture_with_kwargs_call_with_kwargs(self):
        def fn(x, y):
            return x * 2 + y

        captured = tdcv2.capture(fn, x=torch.randn(3, 4), y=torch.randn(3, 4))
        x = torch.randn(5, 6)
        y = torch.randn(5, 6)
        out = captured(x=x, y=y)
        self.assertTrue(torch.allclose(out, fn(x=x, y=y)))

    def test_capture_with_kwargs_mixed_call(self):
        def fn(x, y, z):
            return x + y + z

        captured = tdcv2.capture(
            fn, torch.randn(3, 4), y=torch.randn(3, 4), z=torch.randn(3, 4))
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        # all positional
        self.assertTrue(torch.allclose(captured(x, y, z), fn(x, y, z)))
        # mixed
        self.assertTrue(torch.allclose(captured(x, y=y, z=z), fn(x, y=y, z=z)))
        # all kwargs
        self.assertTrue(torch.allclose(captured(x=x, y=y, z=z), fn(x=x, y=y, z=z)))

    def test_keyword_only_function(self):
        """A fn declared with `*,` keyword-only params must be captured
        via kwargs; positional capture is a Python-level error from fn
        itself, not from v2."""
        def fn(*, x, y):
            return x * 2 + y

        captured = tdcv2.capture(fn, x=torch.randn(3, 4), y=torch.randn(3, 4))
        x = torch.randn(5, 6)
        y = torch.randn(5, 6)
        out = captured(x=x, y=y)
        self.assertTrue(torch.allclose(out, fn(x=x, y=y)))

    def test_missing_kwarg_at_call_time(self):
        def fn(x, y):
            return x + y

        captured = tdcv2.capture(fn, x=torch.randn(3), y=torch.randn(3))
        # User forgets y → should error with a clear message.
        with self.assertRaises(TypeError):
            captured(x=torch.randn(3))

    def test_allow_grad_with_kwargs(self):
        """Capture-with-backward also works through the kwarg path."""
        def loss_fn(x, scale):
            return (x * scale).sum()

        x_ex = torch.randn(4, 5, requires_grad=True)
        scale_ex = torch.randn(4, 5, requires_grad=True)
        captured = tdcv2.capture(
            loss_fn, x_ex, scale=scale_ex, allow_grad=True)

        x = torch.randn(6, 3, requires_grad=True)
        scale = torch.randn(6, 3, requires_grad=True)
        loss = captured(x, scale=scale)
        loss.backward()

        x_ref = x.detach().clone().requires_grad_(True)
        s_ref = scale.detach().clone().requires_grad_(True)
        ref_loss = loss_fn(x_ref, s_ref)
        ref_loss.backward()

        self.assertTrue(torch.allclose(loss, ref_loss))
        self.assertTrue(torch.allclose(x.grad, x_ref.grad))
        self.assertTrue(torch.allclose(scale.grad, s_ref.grad))


if __name__ == "__main__":
    unittest.main()
