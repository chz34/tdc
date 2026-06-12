"""Unit tests for the CompiledKernelWrapperMutation HOP in isolation.

Uses a synthetic 'compiled kernel' (a plain Python callable that writes in
place) so the HOP is tested without any Inductor integration.
"""
import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

import torch_dispatch_capture.v4.compiled_kernel_hop as hop


def _add_kernel(a, b, out):
    # stand-in for a cpp_fused pybinding: writes its output buffer in place.
    out.copy_(a + b)


class TestCompiledKernelHop(unittest.TestCase):
    def setUp(self):
        hop.compiled_kernel_side_table.reset_table()
        self.idx = hop.compiled_kernel_side_table.add_kernel(_add_kernel)

    def _io(self):
        a = torch.randn(8)
        b = torch.randn(8)
        out = torch.empty(8)
        return a, b, out

    def test_dense_mutates_in_place(self):
        a, b, out = self._io()
        ret = hop.compiled_kernel_wrapper_mutation(self.idx, (2,), (a, b, out))
        self.assertIsNone(ret)
        self.assertTrue(torch.allclose(out, a + b))

    def test_side_table_dedup(self):
        self.assertEqual(
            hop.compiled_kernel_side_table.add_kernel(_add_kernel), self.idx
        )

    def test_functional_clones_and_leaves_input(self):
        a, b, out = self._io()
        out0 = out.clone()
        new_vals = hop.compiled_kernel_wrapper_functional(self.idx, (2,), (a, b, out))
        self.assertIn(2, new_vals)
        self.assertTrue(torch.allclose(new_vals[2], a + b))
        # original out buffer untouched (functional form does not mutate inputs)
        self.assertTrue(torch.equal(out, out0))

    def test_proxy_trace_preserves_hop_node(self):
        a, b, out = self._io()

        def f(a, b, out):
            hop.compiled_kernel_wrapper_mutation(self.idx, (2,), (a, b, out))
            return out

        gm = make_fx(f)(a, b, out)
        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertIn(hop.compiled_kernel_wrapper_mutation, targets)

    def test_functionalized_trace_runs_and_matches(self):
        # make_fx with functionalization must trace through the mutation HOP and
        # the result must be numerically correct.
        a, b, out = self._io()

        def f(a, b, out):
            hop.compiled_kernel_wrapper_mutation(self.idx, (2,), (a, b, out))
            return out

        gm = make_fx(f, tracing_mode="fake", _allow_non_fake_inputs=True)(a, b, out)
        a2, b2, out2 = self._io()
        res = gm(a2, b2, out2)
        self.assertTrue(torch.allclose(res, a2 + b2))


if __name__ == "__main__":
    unittest.main()
