"""torch_dispatch_capture.v4 - capture inductor's compiled FX host graph.

See docs/specs/2026-06-04-v4-fx-capture-design.md for the design.
"""
from .capture_fx import (
    NO_FUSION_CONFIG,
    BackendFxWrapper,
    CaptureFxWrapper,
    FxCaptureResult,
    capture_fx,
    compile_with_gm_backend,
    enable_device_via_fallback,
    force_all_fallback_lowerings,
)
from .compiled_kernel_hop import (
    CompiledKernelSideTable,
    compiled_kernel_side_table,
    compiled_kernel_wrapper_functional,
    compiled_kernel_wrapper_mutation,
    register_kernel_compiler,
)
from .cpp_fusion import (
    CompiledKernelBackend,
    CompiledKernelFxConverter,
    CompiledKernelFxWrapper,
    CppPybindingBackend,
    enable_device_with_fusion,
    register_compiled_kernel_backend,
)
from .dvm_fusion import DvmBackend

__all__ = [
    "capture_fx",
    "compile_with_gm_backend",
    "enable_device_via_fallback",
    "force_all_fallback_lowerings",
    "NO_FUSION_CONFIG",
    "FxCaptureResult",
    "CaptureFxWrapper",
    "BackendFxWrapper",
    "enable_device_with_fusion",
    "register_compiled_kernel_backend",
    "CompiledKernelBackend",
    "CppPybindingBackend",
    "DvmBackend",
    "CompiledKernelFxConverter",
    "CompiledKernelFxWrapper",
    "compiled_kernel_wrapper_mutation",
    "compiled_kernel_wrapper_functional",
    "compiled_kernel_side_table",
    "CompiledKernelSideTable",
    "register_kernel_compiler",
]
