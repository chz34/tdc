"""End-to-end: cpp_fused_* kernels embedded into the fx_wrapper host gm via the
CompiledKernelWrapperMutation HOP, under enable_device_with_fusion on CPU.
"""
import types
import unittest

import torch

import torch_dispatch_capture.v4 as tdcv4
import torch_dispatch_capture.v4.cpp_fusion as cf
from torch_dispatch_capture.v4.compiled_kernel_hop import (
    compiled_kernel_side_table,
    compiled_kernel_wrapper_mutation,
)


def _fn(a, b):
    # a@b is an extern (mm); the rest (add + relu + mul) fuses into one cpp kernel.
    return torch.relu(a @ b + a) * 2.0


class TestCppFusion(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        compiled_kernel_side_table.reset_table()

    def _capture(self, fn, *args):
        captured = {}

        def backend(gm, example_inputs):
            captured["gm"] = gm
            return gm.forward

        with torch.no_grad(), torch._inductor.config.patch(force_disable_caches=True), \
                tdcv4.enable_device_with_fusion("cpu", backend):
            out = torch.compile(fn, backend="inductor", dynamic=False)(*args)
        return out, captured.get("gm")

    def test_host_gm_has_compiled_kernel_hop_and_matches(self):
        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        ref = _fn(a, b)
        out, gm = self._capture(_fn, a, b)

        self.assertTrue(torch.allclose(out, ref, atol=1e-4))
        self.assertIsNotNone(gm, "host gm never reached the backend")
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        # the fused pointwise became a compiled_kernel_wrapper_mutation HOP node
        self.assertIn(compiled_kernel_wrapper_mutation, targets)
        # and the mm stayed an aten extern
        self.assertTrue(
            any("mm" in str(t) for t in targets), f"no extern mm in {targets}"
        )

    def test_side_table_populated(self):
        a = torch.randn(16, 16)
        b = torch.randn(16, 16)
        self._capture(_fn, a, b)
        self.assertGreater(len(compiled_kernel_side_table.id_to_kernel), 0)

    def test_registry_restored_on_exit(self):
        from torch._inductor.codegen.common import (
            device_codegens,
            init_backend_registration,
        )

        init_backend_registration()
        before = device_codegens["cpu"].fx_wrapper_codegen
        a = torch.randn(8, 8)
        self._capture(_fn, a, a)
        self.assertIs(device_codegens["cpu"].fx_wrapper_codegen, before)


class TestBackendRegistry(unittest.TestCase):
    """Adding a compiler backend must not require editing the converter: a newly
    registered backend is selected purely via handles_definition."""

    def test_new_backend_selected_without_converter_changes(self):
        sentinel = object()

        class FakeBackend(cf.CompiledKernelBackend):
            def handles_definition(self, defn_line):
                return getattr(defn_line, "kind", None) == "fake"

            def compile_kernel(self, converter, defn_line):
                return sentinel

            def mutated_arg_indices(self, call_line):
                return (0,)

        fake = FakeBackend()
        cf.register_compiled_kernel_backend(fake)
        try:
            # a line this backend claims -> selected
            self.assertIs(cf._select_backend(types.SimpleNamespace(kind="fake")), fake)
            # a cpp line (gpu=False) -> still the default CppPybindingBackend
            cpp = cf._select_backend(types.SimpleNamespace(gpu=False))
            self.assertIsInstance(cpp, cf.CppPybindingBackend)
            # a triton line (gpu=True, unclaimed) -> None (falls back to super)
            self.assertIsNone(cf._select_backend(types.SimpleNamespace(gpu=True)))
        finally:
            cf._COMPILED_BACKENDS.remove(fake)

    def test_cpp_backend_mutation_from_arg_types(self):
        b = cf.CppPybindingBackend()
        self.assertEqual(
            b.mutated_arg_indices(types.SimpleNamespace(arg_types=["float*"])), (0,)
        )
        self.assertEqual(
            b.mutated_arg_indices(
                types.SimpleNamespace(
                    arg_types=["const float*", "const float*", "float*"]
                )
            ),
            (2,),
        )


if __name__ == "__main__":
    unittest.main()
