"""v2.capture on nn.Module — exercises parameter placeholders.

Unlike pure-function test cases, real models hold weights as
nn.Parameter on the module. Dynamo lifts those parameters into the
AOT graph as extra placeholders that are NOT part of the user's call
signature. v2.capture has to recognise them (by id() matching against
example_args) and snapshot their observed value as a baked literal in
the flat recipe. Weights stay frozen across replays — re-capture if
they change.

These tests cover:
  - nn.Linear forward (no_grad inference)
  - nn.Linear forward with allow_grad=True — user-input grad path
  - Small 2-layer MLP (multiple param placeholders, ReLU in between)
  - Bias-less Linear (one fewer param) to vary the param count
"""
import unittest

import torch
from torch import nn
import torch_dispatch_capture.v2 as tdcv2


class TestV2NnModule(unittest.TestCase):

    def setUp(self):
        torch._dynamo.reset()

    def test_linear_inference(self):
        linear = nn.Linear(16, 32)
        linear.eval()

        captured = tdcv2.capture(linear, torch.randn(4, 16))

        for B in [4, 7, 12]:
            x = torch.randn(B, 16)
            out = captured(x)
            ref = linear(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_linear_bias_false(self):
        linear = nn.Linear(8, 24, bias=False)
        linear.eval()

        captured = tdcv2.capture(linear, torch.randn(3, 8))

        for B in [3, 5, 9]:
            x = torch.randn(B, 8)
            out = captured(x)
            ref = linear(x)
            self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_mlp_inference(self):
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        model.eval()

        captured = tdcv2.capture(model, torch.randn(4, 16))

        for B in [4, 6, 10]:
            x = torch.randn(B, 16)
            out = captured(x)
            ref = model(x)
            self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_linear_with_backward_user_grad(self):
        """allow_grad=True on a Linear-using fn: the user-input grad
        should be tracked through autograd; parameter grads are NOT
        propagated by the current implementation (snapshot-frozen)."""
        linear = nn.Linear(16, 32)

        def loss_fn(x):
            return linear(x).sum()

        x_ex = torch.randn(4, 16, requires_grad=True)
        captured = tdcv2.capture(loss_fn, x_ex, allow_grad=True)

        x = torch.randn(6, 16, requires_grad=True)
        loss = captured(x)
        loss.backward()

        # Eager reference: clone so the comparison is independent.
        x_ref = x.detach().clone().requires_grad_(True)
        loss_ref = linear(x_ref).sum()
        loss_ref.backward()

        self.assertTrue(torch.allclose(loss, loss_ref, atol=1e-4))
        self.assertTrue(torch.allclose(x.grad, x_ref.grad, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
