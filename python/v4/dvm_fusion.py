"""DvmBackend: route torch_npu's dvm/mlir fused kernels through the
CompiledKernelWrapperMutation HOP, same as the CPU cpp_pybinding reference.

The dvm backend (TORCHINDUCTOR_NPU_BACKEND=dvm) reuses torch_npu's mlir
compile/codegen path: a fused kernel is defined as
`name = async_compile.mlir('name', '''<mlir>''', device_str='npu')` and is
gpu=True (like Triton). Stock FxConverter rejects it at the definition gate
(_import_kernel: "Unsupported type ... FX conversion only supports Triton
kernels", empirically confirmed on an NPU box). This backend intercepts those
definition lines so the converter never reaches that gate; the kernel is loaded
through PyCodeCache, stored in the global side table, and its call line becomes a
CompiledKernelWrapperMutation HOP node.

Importing this module auto-registers DvmBackend. It is inert unless a kernel
definition body contains async_compile.mlir/akg, so it is harmless on non-NPU
runs. No torch_npu import happens at module load -- the npu stream lookup is
deferred to launch time.

One torch_npu-side prerequisite for mutation handling: dvm's call_kernel must
pass arg_types to generate_kernel_call, marking output buffers as writeable
(non-const) pointers, so KernelCallLine.arg_types is populated. See
mutated_arg_indices below.
"""
from collections.abc import Callable

from torch._inductor.codecache import CodeCacheFuture, LambdaFuture, PyCodeCache
from torch._inductor.codegen.wrapper import PythonWrapperCodegen

from .cpp_fusion import (
    CompiledKernelBackend,
    cpp_mutated_arg_indices,
    register_compiled_kernel_backend,
)

# Compile APIs torch_npu emits for dvm/mlir fused kernels (default + autotune
# fallback + akg). A kernel-definition body containing any of these is ours.
_DVM_COMPILE_APIS = ("async_compile.mlir", "async_compile.akg")


def _current_npu_raw_stream():
    """Current npu raw stream, looked up lazily so tdc imports on CPU-only hosts.
    Returns None if torch_npu is unavailable (caller then launches without a
    stream)."""
    try:
        import torch

        from torch_npu._inductor.utils import get_current_raw_stream

        return get_current_raw_stream(torch.npu.current_device())
    except Exception:
        return None


class _DvmKernelLauncher:
    """Adapts a compiled dvm/mlir kernel to the HOP dense contract: callable on
    the flat positional args, writing its outputs in place.

    The compiled object has been observed in two shapes on the dvm path: a
    launcher exposing .run(*args, stream=...) (the documented mlir/akg
    convention) and a plain function (single-thread / fx-graph fallback). We
    prefer .run with the current npu stream and fall back to calling the object
    directly. The exact stream contract for the plain-callable form must be
    confirmed on an NPU box; this is the single place to adjust if so."""

    def __init__(self, compiled: object, kernel_name: str) -> None:
        self._compiled = compiled
        self._kernel_name = kernel_name

    def __call__(self, *args: object) -> None:
        compiled = self._compiled
        if hasattr(compiled, "run"):
            compiled.run(*args, stream=_current_npu_raw_stream())
        else:
            compiled(*args)


class DvmBackend(CompiledKernelBackend):
    """torch_npu dvm/mlir fused kernels embedded as CompiledKernelWrapperMutation
    HOP nodes."""

    def handles_definition(self, defn_line) -> bool:
        body = getattr(defn_line, "kernel_body", "") or ""
        return any(api in body for api in _DVM_COMPILE_APIS)

    def compile_kernel(self, converter, defn_line) -> Callable:
        code = PythonWrapperCodegen._format_kernel_definition(
            defn_line.kernel_name, defn_line.kernel_body, metadata=defn_line.metadata
        )
        mod = PyCodeCache.load("\n".join([converter.prologue, code]))
        kernel = getattr(mod, defn_line.kernel_name)
        # async_compile.* may hand back a future (NPUTritonFuture /
        # MulitprocessCompileFuture / LambdaFuture); materialize it.
        while isinstance(kernel, (CodeCacheFuture, LambdaFuture)):
            kernel = kernel.result()
        return _DvmKernelLauncher(kernel, defn_line.kernel_name)

    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        # dvm call lines carry no arg_types by default (generate_kernel_call is
        # called positionally), so the meta kernel's output info must reach us as
        # writeable-pointer arg_types. Reuse the cpp rule once they are present.
        arg_types = getattr(call_line, "arg_types", None)
        if not arg_types:
            raise NotImplementedError(
                "DvmBackend needs KernelCallLine.arg_types to know which args are "
                "written. Make torch_npu's NpuMetaKernel.call_kernel pass "
                "arg_types to generate_kernel_call, marking output buffers as "
                "non-const pointers (e.g. 'T*') and inputs as 'const T*'. The "
                "mutated positions come from NpuMetaKernel.mutated_indices."
            )
        return cpp_mutated_arg_indices(arg_types)


register_compiled_kernel_backend(DvmBackend())
