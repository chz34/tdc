"""Backward capture/replay tests.

These tests opt into experimental backward support via
``tdc.capture(allow_grad=True)``. Our dispatcher fallback fires at
TESTING_ONLY_GenericMode (priority #3), which is above AutogradFunctionality
(#19), so calling ``.backward()`` inside the capture block records the full
forward + backward op sequence into a single trace.

Two correctness preconditions the user must meet:
  1. ``x.grad`` is pre-allocated before capture, so AccumulateGrad takes
     the in-place ``add_`` branch (a dispatched op we record), not the
     direct C++ assignment branch (which we cannot record).
  2. ``x.grad`` is zeroed before each replay if non-accumulating
     gradient semantics are desired.
"""
import unittest

import torch
import torch_dispatch_capture as tdc


class TestBackward(unittest.TestCase):
    # ------------------------------------------------------------------
    # Sanity
    # ------------------------------------------------------------------

    def test_allow_grad_required(self):
        x = torch.randn(3, requires_grad=True)
        # Default (allow_grad=False) still rejects grad-enabled capture.
        with self.assertRaisesRegex(RuntimeError, "allow_grad=True"):
            with tdc.capture():
                _ = x * x

    def test_allow_grad_lets_capture_proceed(self):
        x = torch.randn(3, requires_grad=True)
        x.grad = torch.zeros_like(x)
        with tdc.capture(allow_grad=True) as trace:
            y = (x * x).sum()
            y.backward()
        # forward + backward both contribute ops
        self.assertGreater(len(trace), 3)

    # ------------------------------------------------------------------
    # Numerical correctness
    # ------------------------------------------------------------------

    def test_basic_backward_replay(self):
        """y = sum(x*x); dy/dx = 2x. Replay should reproduce."""
        torch.manual_seed(0)
        x = torch.randn(3, requires_grad=True)
        x.grad = torch.zeros_like(x)         # pre-alloc for AccumulateGrad

        with tdc.capture(allow_grad=True) as trace:
            y = (x * x).sum()
            y.backward()

        # eager reference
        ref = 2 * x.detach()
        torch.testing.assert_close(x.grad, ref)

        # Multiple replays — note that grads ACCUMULATE because that's the
        # semantics of .backward() in PyTorch. Zero before each replay if
        # you want one-shot semantics.
        x.grad.zero_()
        trace.replay()
        torch.testing.assert_close(x.grad, ref)

        # Change x value, replay, verify the new gradient.
        x.detach().copy_(torch.arange(3, dtype=torch.float32))
        x.grad.zero_()
        trace.replay()
        torch.testing.assert_close(x.grad, 2 * x.detach())

    def test_grad_accumulates_on_repeated_replay(self):
        """Repeated replay without zero_ accumulates — matches eager."""
        x = torch.ones(4, requires_grad=True)
        x.grad = torch.zeros_like(x)

        with tdc.capture(allow_grad=True) as trace:
            (x * 2).sum().backward()

        # After capture: grad should be all 2s.
        torch.testing.assert_close(x.grad, torch.full((4,), 2.0))

        trace.replay()
        torch.testing.assert_close(x.grad, torch.full((4,), 4.0))  # +2

        trace.replay()
        torch.testing.assert_close(x.grad, torch.full((4,), 6.0))  # +2

    def test_linear_backward_replay(self):
        """Single nn.Linear forward + backward, all on the captured trace."""
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4, bias=False).eval()
        for p in model.parameters():
            p.grad = torch.zeros_like(p)        # pre-alloc

        x = torch.randn(2, 4, requires_grad=True)
        x.grad = torch.zeros_like(x)

        with tdc.capture(allow_grad=True) as trace:
            y = model(x)
            y.sum().backward()

        # Capture pass already populated grads. Snapshot them.
        capture_x_grad = x.grad.clone()
        capture_w_grad = model.weight.grad.clone()

        # Zero and replay; result should match capture.
        x.grad.zero_(); model.weight.grad.zero_()
        trace.replay()
        torch.testing.assert_close(x.grad, capture_x_grad)
        torch.testing.assert_close(model.weight.grad, capture_w_grad)

    # ------------------------------------------------------------------
    # Dynamic shape with backward
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dynamic shape, shape-stable backward (element-wise chain)
    # ------------------------------------------------------------------
    #
    # Backward ops that don't reduce dimensions (multiply, add, relu, silu, ...)
    # don't need to bake any shape literals — their gradients are computed
    # purely element-wise. These work cleanly with dynamic shape: capture
    # once, replay across shapes, gradients are correct.
    #
    # To trigger backward without a reduction, we pass an explicit grad_output
    # tensor whose shape tracks the input shape — both are captured tensors,
    # so resizing the input + resizing the grad_output is enough.

    # Note on resizing leaf tensors with requires_grad:
    #   - `x.resize_()`           → raises (requires_grad protected)
    #   - `x.data.resize_()`      → silently resizes a view, NOT x itself
    #   - `x.data = new_tensor`   → ← correct: replaces storage/size on
    #                                 the SAME TensorImpl that the trace
    #                                 holds a strong ref to.
    # `x.grad` doesn't have requires_grad, so `x.grad.resize_as_().zero_()`
    # works fine for it.

    def test_backward_dynamic_shape_elementwise(self):
        """Element-wise forward+backward, dynamic shape via x.data assign."""
        torch.manual_seed(0)
        x = torch.randn(4, requires_grad=True)
        x.grad = torch.zeros_like(x)
        grad_out = torch.ones(4)

        with tdc.capture(allow_grad=True) as trace:
            y = x * x * 2                # y = 2*x*x, dy/dx = 4x
            y.backward(grad_out)         # no .sum() / reductions

        for n in (1, 3, 8, 16, 32):
            x.data = torch.randn(n)      # rebind x's data; same TensorImpl
            grad_out.resize_(n); grad_out.fill_(1.0)
            x.grad.resize_as_(x).zero_()

            trace.replay()
            ref = 4 * x.detach()
            torch.testing.assert_close(
                x.grad, ref, msg=lambda m: f"grad mismatch at n={n}: {m}")

    def test_backward_dynamic_shape_relu_chain(self):
        """ReLU + multiply + add backward across varying shape."""
        torch.manual_seed(0)
        x = torch.randn(5, requires_grad=True)
        x.grad = torch.zeros_like(x)
        grad_out = torch.ones(5)

        with tdc.capture(allow_grad=True) as trace:
            y = torch.relu(x * 3) + x
            y.backward(grad_out)

        for n in (1, 4, 12, 25):
            x.data = torch.randn(n)
            grad_out.resize_(n); grad_out.fill_(1.0)
            x.grad.resize_as_(x).zero_()

            trace.replay()
            x_eager = x.detach().clone().requires_grad_(True)
            x_eager.grad = torch.zeros_like(x_eager)
            (torch.relu(x_eager * 3) + x_eager).backward(torch.ones(n))
            torch.testing.assert_close(
                x.grad, x_eager.grad,
                msg=lambda m: f"grad mismatch at n={n}: {m}")

    # ------------------------------------------------------------------
    # Known limitation: backward through reductions bakes shape literals
    # ------------------------------------------------------------------
    #
    # A `.sum().backward()` path captures `aten::expand(grad_scalar, [shape])`
    # in the trace, where `[shape]` is an IntArrayRef literal baked from the
    # input's capture-time shape. Replay at a different shape uses the stale
    # literal and produces a wrong-shape gradient.
    #
    # This is the same limitation as `as_strided(size=..., stride=...)` in
    # the design doc §8: any backward op whose schema args include shape
    # ints can't track dynamic input shape from a single trace. The user
    # workaround is to recapture per shape, or use SymInt-aware tracing
    # (out of scope for this PoC).

    @unittest.expectedFailure
    def test_backward_dynamic_shape_with_reduction_fails(self):
        """Document the shape-literal limitation: sum().backward() does not
        adapt to new input shapes on replay. Expected failure."""
        torch.manual_seed(0)
        d = 4
        w = torch.randn(d, d, requires_grad=True)
        w.grad = torch.zeros_like(w)
        x = torch.randn(3, d, requires_grad=True)
        x.grad = torch.zeros_like(x)

        with tdc.capture(allow_grad=True) as trace:
            (x @ w).sum().backward()   # sum reduces -> backward expand
                                        # bakes the original [3, d] shape

        # Replay at a different batch -> trace's expand will produce a [3, d]
        # gradient, not [1, d]. add_ into x.grad fails or gives wrong values.
        x.data.resize_(1, d); x.data.normal_()
        x.grad = torch.zeros_like(x)
        w.grad.zero_()
        trace.replay()
        # If we got here without raising, the gradient is silently wrong
        # at the new shape — also a test failure for our purposes.
        ref_x = x.detach().clone().requires_grad_(True)
        ref_w = w.detach().clone().requires_grad_(True)
        (ref_x @ ref_w).sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)


if __name__ == "__main__":
    unittest.main()
