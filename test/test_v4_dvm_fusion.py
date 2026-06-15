"""DvmBackend selection / mutation logic. CPU-only: no torch_npu, no NPU needed.

These cover the parts of the dvm path that are device-independent -- which
definition lines the backend claims, that it stays inert for cpp/triton lines,
and how it derives mutated args. The end-to-end embed (definition gate bypass +
HOP node) must be validated on an NPU box with dvm_fxwrapper_runtime_probe.py.
"""
import types
import unittest

import torch_dispatch_capture.v4 as tdcv4
import torch_dispatch_capture.v4.cpp_fusion as cf
from torch_dispatch_capture.v4 import DvmBackend


def _defn(body, **kw):
    return types.SimpleNamespace(kernel_body=body, kernel_name="k", metadata=None, **kw)


class TestDvmBackendSelection(unittest.TestCase):
    def test_claims_native_dvm_kernel_by_name(self):
        # native dvm-codegen mode: body is just the "<name>_build" builder token,
        # the real code is in metadata. Only the dvm_ name prefix identifies it.
        b = DvmBackend()
        line = types.SimpleNamespace(
            kernel_name="dvm_fused_add_mul_relu_0",
            kernel_body="dvm_fused_add_mul_relu_0_build",
            metadata="def dvm_fused_add_mul_relu_0_build(...): ...",
            gpu=True,
        )
        self.assertTrue(b.handles_definition(line))

    def test_claims_fx_fallback_definitions_by_body(self):
        b = DvmBackend()
        bodies = [
            "k = async_compile.mlir('k', '''<src>''', device_str='npu')",
            "k = async_compile.akg_auto_fallback('k', '''<src>''', kernel_meta={})",
            "k = async_compile.import_fx('k', kernel_meta={})",
        ]
        for body in bodies:
            self.assertTrue(b.handles_definition(_defn(body, gpu=True)), body)

    def test_inert_for_triton_and_cpp(self):
        b = DvmBackend()
        triton = _defn("@triton.jit\ndef k(...): ...", gpu=True)
        cpp = _defn("async_compile.cpp_pybinding(['float*'], '''...''')", gpu=False)
        self.assertFalse(b.handles_definition(triton))
        self.assertFalse(b.handles_definition(cpp))

    def test_registry_routes_by_body_not_gpu(self):
        # dvm is auto-registered on import; selection must come from the body,
        # since dvm and triton are both gpu=True.
        mlir = _defn("k = async_compile.mlir('k', '''m''', device_str='npu')", gpu=True)
        self.assertIsInstance(cf._select_backend(mlir), DvmBackend)
        # cpp line still goes to the cpp backend (registered first, gpu=False)
        cpp = _defn("async_compile.cpp_pybinding(['float*'], '''c''')", gpu=False)
        self.assertIsInstance(cf._select_backend(cpp), cf.CppPybindingBackend)
        # a triton line (gpu=True, no dvm api in body) is unclaimed -> super()
        triton = _defn("@triton.jit\ndef k(): ...", gpu=True)
        self.assertIsNone(cf._select_backend(triton))


class TestDvmMutatedArgs(unittest.TestCase):
    def test_uses_arg_types_when_present(self):
        b = DvmBackend()
        line = types.SimpleNamespace(
            kernel_name="k", arg_types=["const float*", "const float*", "float*"]
        )
        self.assertEqual(b.mutated_arg_indices(line), (2,))

    def test_raises_with_guidance_when_arg_types_missing(self):
        b = DvmBackend()
        line = types.SimpleNamespace(kernel_name="k", arg_types=None)
        with self.assertRaises(NotImplementedError) as cm:
            b.mutated_arg_indices(line)
        self.assertIn("arg_types", str(cm.exception))


class TestDvmRegisteredOnce(unittest.TestCase):
    def test_exactly_one_dvm_backend_registered(self):
        n = sum(isinstance(x, DvmBackend) for x in cf._COMPILED_BACKENDS)
        self.assertEqual(n, 1)
        # importing the package must not have perturbed enable_device_with_fusion
        self.assertTrue(hasattr(tdcv4, "enable_device_with_fusion"))


if __name__ == "__main__":
    unittest.main()
