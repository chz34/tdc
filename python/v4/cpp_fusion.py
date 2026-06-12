"""Embed Inductor cpp_fused_* kernels into the fx_wrapper host graph.

Inductor's stock FxConverter only converts Triton kernel-call lines. This module
adds the CPU/compiled path: a non-Triton kernel definition is compiled+cached via
the existing async_compile path (its kernel_body already calls
async_compile.cpp_pybinding, loaded through PyCodeCache), stored in the global
CompiledKernelSideTable, and its call line becomes a CompiledKernelWrapperMutation
HOP node.

enable_device_with_fusion is the fusion-enabled counterpart to v4's
enable_device_via_fallback: same fx_wrapper capture, but fused CPU kernels survive
as HOP nodes instead of being fallback-expanded.
"""
import contextlib
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.codegen.common import FileBackedGraphModule
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.wrapper_fxir import FxConverter, WrapperFxCodegen
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

from .compiled_kernel_hop import (
    compiled_kernel_side_table,
    compiled_kernel_wrapper_mutation,
)

_active_fusion_backend: "Callable | None" = None


def _mutated_arg_indices(arg_types: list) -> tuple[int, ...]:
    """A cpp kernel arg is mutated iff it is a non-const pointer. cpp_argdefs()
    emits writeable buffers (inplace + output) as 'T*' and read-only inputs as
    'const T*'; sizevars have no '*'."""
    return tuple(
        i
        for i, t in enumerate(arg_types)
        if isinstance(t, str) and "*" in t and not t.strip().startswith("const")
    )


class CompiledKernelFxConverter(FxConverter):
    """FxConverter that also handles non-Triton (compiled) kernels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compiled_kernels: dict[str, int] = {}  # kernel_name -> side-table idx

    def _generate_kernel_definition(self, line) -> None:
        if getattr(line, "gpu", True):
            return super()._generate_kernel_definition(line)
        # CPU/compiled kernel: kernel_body is `async_compile.cpp_pybinding(...)`.
        kernel_code = PythonWrapperCodegen._format_kernel_definition(
            line.kernel_name, line.kernel_body, metadata=line.metadata
        )
        mod = PyCodeCache.load("\n".join([self.prologue, kernel_code]))
        kernel = getattr(mod, line.kernel_name)
        if isinstance(kernel, LambdaFuture):
            kernel = kernel.result()
        if isinstance(kernel, CachingAutotuner):
            raise AssertionError("Triton kernel reached the compiled (cpp) path")
        self.compiled_kernels[line.kernel_name] = (
            compiled_kernel_side_table.add_kernel(kernel)
        )

    def _generate_kernel_call(self, line) -> None:
        if line.triton:
            return super()._generate_kernel_call(line)
        idx = self.compiled_kernels[line.kernel_name]
        call_args = tuple(self._lookup_args(line.call_args))
        self.gm.graph.call_function(
            compiled_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": idx,
                "mutated_arg_indices": _mutated_arg_indices(line.arg_types),
                "args": call_args,
            },
        )


class CompiledKernelFxWrapper(WrapperFxCodegen):
    """WrapperFxCodegen using CompiledKernelFxConverter; routes the host gm to an
    optional fusion backend (else runs it via gm.forward)."""

    def _generate(self, is_inference: bool):
        self.run_wrapper_ir_passes(is_inference)
        prologue = "\n".join([self.imports.getvalue(), self.header.getvalue()])
        gm = CompiledKernelFxConverter(
            lines=self.lines,
            prologue=prologue,
            graph_inputs=self.get_fx_graph_inputs(),
            graph_outputs=self.get_graph_outputs(),
            subgms=self.subgms,
            is_subgraph=self.is_subgraph,
        ).generate()
        return FileBackedGraphModule(gm, self.compile_graph(gm)), None

    def compile_graph(self, gm):
        if _active_fusion_backend is None:
            return super().compile_graph(gm)
        example_inputs = [
            n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"
        ]
        return _active_fusion_backend(gm, example_inputs)


@contextlib.contextmanager
def enable_device_with_fusion(
    device: str, gm_backend: "Callable | None" = None
):
    """Bring up inductor on a device type with fx_wrapper + CPU fusion enabled.

    Unlike enable_device_via_fallback (all-fallback, zero fusion), this keeps the
    real CppScheduling, so fused cpp kernels are codegen'd and embedded into the
    host gm as CompiledKernelWrapperMutation HOP nodes. gm_backend is optional;
    when omitted the host gm runs via gm.forward. Restores swapped registries /
    config on exit. Process-global, not thread-safe.
    """
    global _active_fusion_backend
    import torch._inductor.config as inductor_config
    from torch._inductor.codegen.common import (
        custom_backend_passes,
        device_codegens,
        init_backend_registration,
        register_backend_for_device,
    )
    from torch._inductor.codegen.cpp import CppScheduling
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen as PyWrapper

    device = torch.device(device).type

    init_backend_registration()
    had = device in device_codegens
    saved_dc = device_codegens.get(device)
    saved_pass = custom_backend_passes.get(device)
    saved_backend = _active_fusion_backend

    register_backend_for_device(
        device, CppScheduling, PyWrapper, CppWrapperCpu, CompiledKernelFxWrapper
    )
    _active_fusion_backend = gm_backend
    try:
        with inductor_config.patch(
            {"fx_wrapper": True, "size_asserts": False, "alignment_asserts": False}
        ):
            yield
    finally:
        _active_fusion_backend = saved_backend
        if had:
            device_codegens[device] = saved_dc
            custom_backend_passes[device] = saved_pass
        else:
            device_codegens.pop(device, None)
            custom_backend_passes.pop(device, None)
