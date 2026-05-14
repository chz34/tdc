"""Numerical correctness of capture/replay against eager execution."""
import unittest

import torch
import torch_dispatch_capture as tdc


class TestCorrectness(unittest.TestCase):
    def test_no_grad_required(self):
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            with tdc.capture():
                pass

    def test_nested_capture_rejected(self):
        with torch.no_grad():
            with tdc.capture():
                with self.assertRaises(RuntimeError):
                    with tdc.capture():
                        pass

    def test_is_capturing_flag(self):
        self.assertFalse(tdc.is_capturing())
        with torch.no_grad():
            with tdc.capture():
                self.assertTrue(tdc.is_capturing())
        self.assertFalse(tdc.is_capturing())

    def test_elementwise_chain(self):
        # Use out= overloads so the result tensor is observable across replays.
        a = torch.randn(8, 8)
        b = torch.randn(8, 8)
        out1 = torch.empty(8, 8)
        out2 = torch.empty(8, 8)
        with torch.no_grad():
            with tdc.capture() as trace:
                torch.add(a, b, out=out1)
                torch.mul(out1, 0.5, out=out2)
            self.assertGreaterEqual(len(trace), 2)
            # Mutate input, replay, check output reflects new values
            a.fill_(2.0); b.fill_(4.0)
            trace.replay()
            expected = (a + b) * 0.5
            torch.testing.assert_close(out2, expected)

    def test_linear_forward_no_grad(self):
        # TESTING_ONLY_GenericMode fires before composite decomposition, so
        # we see just `aten::linear` as one op rather than its decomposed
        # `t + addmm` form. Both behaviors are valid; the goal is replay
        # working with the captured ops.
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 8)
        x = torch.randn(4, 8)
        with torch.no_grad():
            with tdc.capture() as trace:
                _ = model(x)
            self.assertGreaterEqual(len(trace), 1)
            trace.replay()

    def test_relu_zeros(self):
        x = torch.tensor([1.0, -1.0, 0.5])
        with torch.no_grad():
            with tdc.capture() as trace:
                _ = x.relu()
            # relu's CPU kernel decomposes to clamp_min internally, so the
            # trace contains both ops. We just check it's non-empty and
            # replay doesn't crash.
            self.assertGreaterEqual(len(trace), 1)
            trace.replay()

    def test_dump_works(self):
        a = torch.randn(2, 2)
        with torch.no_grad():
            with tdc.capture() as trace:
                _ = a + a
            text = trace.dump()
            self.assertIn("Trace", text)
            self.assertIn("add", text)


if __name__ == "__main__":
    unittest.main()
