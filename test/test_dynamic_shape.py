"""Validate dynamic-shape behavior: same trace, varying input shapes."""
import unittest

import torch
import torch_dispatch_capture as tdc

from _device import DEVICE, print_device_banner


class TestDynamicShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print_device_banner()

    def _capture(self, a, b):
        with torch.no_grad():
            with tdc.capture() as trace:
                _ = a + b
                _ = a * 2.0
            return trace

    def test_inplace_value_mutation(self):
        a = torch.zeros(4, device=DEVICE)
        b = torch.ones(4, device=DEVICE)
        trace = self._capture(a, b)

        a.fill_(10.0)
        trace.replay()
        # We have no return value; correctness is "didn't crash". For full
        # verification of value propagation, we use a tiny custom workload
        # that writes into a known buffer (see test_inplace_out_buffer).

    def test_inplace_out_buffer(self):
        # add(a, b, out=c) writes into c; capturing this and replaying after
        # mutating a/b should update c with new values.
        a = torch.zeros(4, device=DEVICE)
        b = torch.ones(4, device=DEVICE)
        c = torch.empty(4, device=DEVICE)
        with torch.no_grad():
            with tdc.capture() as trace:
                torch.add(a, b, out=c)
            # Initial: 0+1 = 1
            torch.testing.assert_close(c, torch.ones(4, device=DEVICE))

            a.fill_(10.0)
            trace.replay()
            torch.testing.assert_close(c, torch.full((4,), 11.0, device=DEVICE))

            b.fill_(0.5)
            trace.replay()
            torch.testing.assert_close(c, torch.full((4,), 10.5, device=DEVICE))

    def test_resize_same_storage(self):
        # Capture at shape (4,8), replay at different shapes — only inputs
        # need to be resized; `c` (the out= tensor) is auto-resized by the
        # kernel because the trace marks it as a schema-out arg.
        a = torch.zeros(4, 8, device=DEVICE)
        b = torch.ones(4, 8, device=DEVICE)
        c = torch.empty(4, 8, device=DEVICE)
        with torch.no_grad():
            with tdc.capture() as trace:
                torch.add(a, b, out=c)

            # Shrink to (2, 8) — do NOT manually resize c.
            a.resize_(2, 8); a.fill_(3.0)
            b.resize_(2, 8); b.fill_(7.0)
            trace.replay()
            self.assertEqual(tuple(c.shape), (2, 8))
            torch.testing.assert_close(c, torch.full((2, 8), 10.0, device=DEVICE))

            # Grow to (16, 8) — same trace handles it.
            a.resize_(16, 8); a.fill_(1.5)
            b.resize_(16, 8); b.fill_(0.5)
            trace.replay()
            self.assertEqual(tuple(c.shape), (16, 8))
            torch.testing.assert_close(c, torch.full((16, 8), 2.0, device=DEVICE))

    def test_varied_batch(self):
        # Capture a tiny linear at batch=4 and replay at multiple batches.
        #
        # The function is written in fully natural Python style — `out = expr`
        # rebinds a local variable to a freshly-allocated tensor and returns
        # it. This rebinding is NOT visible to the caller, so to make replay
        # results externally observable we add ONE convention inside the
        # capture block: copy the function's return value into a caller-
        # provided buffer. The user's existing function code is untouched.
        torch.manual_seed(0)
        w = torch.randn(8, 8, device=DEVICE)
        b = torch.randn(8, device=DEVICE)

        with torch.no_grad():
            def matmul_plus_b(x, w, b):
                out = torch.matmul(x, w.t())
                out = out + b
                return out

            x = torch.randn(4, 8, device=DEVICE)
            obs = torch.empty(4, 8, device=DEVICE)
            with tdc.capture() as trace:
                result = matmul_plus_b(x, w, b)
                obs.resize_as_(result)   # in-place resize, captured
                obs.copy_(result)        # in-place copy, captured
            ref = (x @ w.t()) + b
            torch.testing.assert_close(obs, ref)

            # Replay across varied batches — resize_as_ + copy_ in the trace
            # make `obs` follow whatever shape `result` takes each replay.
            for batch in (1, 2, 8, 16):
                x.resize_(batch, 8)
                x.normal_()
                trace.replay()
                ref = (x @ w.t()) + b
                torch.testing.assert_close(obs, ref,
                                           msg=f"mismatch at batch={batch}")

            # Same again, mutating x's data via copy_ instead of normal_.
            for batch in (1, 2, 8, 16):
                x_new = torch.randn(batch, 8, device=DEVICE)
                x.resize_(x_new.shape)
                x.copy_(x_new)
                trace.replay()
                ref = (x @ w.t()) + b
                torch.testing.assert_close(obs, ref,
                                           msg=f"mismatch at batch={batch}")


if __name__ == "__main__":
    unittest.main()
