"""Embed non-Triton compiled fused kernels into the fx_wrapper host graph.

Inductor's stock FxConverter only converts Triton kernel-call lines. This module
adds a backend-agnostic compiled path: a kernel definition handled by a registered
CompiledKernelBackend is compiled to a callable, stored in the global
CompiledKernelSideTable, and its call line becomes a CompiledKernelWrapperMutation
HOP node.

Adding a new compiler backend requires NO change to the converter / wrapper here:
subclass CompiledKernelBackend (three methods -- does it handle this kernel, how to
compile it, which args it mutates) and call register_compiled_kernel_backend(...).
The CPU Inductor cpp_pybinding path is the default backend, registered at import as
the reference implementation.

enable_device_with_fusion is the fusion-enabled counterpart to v4's
enable_device_via_fallback: same fx_wrapper capture, but fused kernels survive as
HOP nodes instead of being fallback-expanded.
"""
import abc
import contextlib
import dataclasses
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


# --- pluggable compiler backends -------------------------------------------
class CompiledKernelBackend(abc.ABC):
    """Teaches the converter how to handle one class of non-Triton fused kernel.

    A backend is selected per kernel-definition line via handles_definition; the
    converter then asks it to compile the kernel to a callable and, at call time,
    which positional args the kernel writes. Nothing else in the converter is
    backend-specific, so a new compiler is added purely by registering one of these.
    """

    @abc.abstractmethod
    def handles_definition(self, defn_line) -> bool:
        """True if this backend owns the given KernelDefinitionLine."""

    @abc.abstractmethod
    def compile_kernel(self, converter: "CompiledKernelFxConverter", defn_line) -> Callable:
        """Compile the kernel to a callable taking the flat positional args and
        writing its outputs in place (the contract the HOP's dense impl calls)."""

    @abc.abstractmethod
    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        """Positions in the call args the kernel writes (for functionalization)."""


_COMPILED_BACKENDS: list[CompiledKernelBackend] = []


def register_compiled_kernel_backend(backend: CompiledKernelBackend) -> None:
    """Register a CompiledKernelBackend. First registered that handles a kernel
    wins, so register more specific backends before more general ones."""
    _COMPILED_BACKENDS.append(backend)


def _select_backend(defn_line) -> "CompiledKernelBackend | None":
    for backend in _COMPILED_BACKENDS:
        if backend.handles_definition(defn_line):
            return backend
    return None


# --- converter / wrapper (backend-agnostic) --------------------------------
class CompiledKernelFxConverter(FxConverter):
    """FxConverter that routes non-Triton kernels through registered backends."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # kernel_name -> (side-table idx, owning backend)
        self.compiled_kernels: dict[str, tuple[int, CompiledKernelBackend]] = {}

    def _generate_kernel_definition(self, line) -> None:
        backend = _select_backend(line)
        if backend is None:
            return super()._generate_kernel_definition(line)  # Triton
        kernel = backend.compile_kernel(self, line)
        idx = compiled_kernel_side_table.add_kernel(kernel)
        self.compiled_kernels[line.kernel_name] = (idx, backend)

    def _generate_kernel_call(self, line) -> None:
        entry = self.compiled_kernels.get(line.kernel_name)
        if entry is None:
            return super()._generate_kernel_call(line)  # Triton
        idx, backend = entry
        self.gm.graph.call_function(
            compiled_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": idx,
                "mutated_arg_indices": tuple(backend.mutated_arg_indices(line)),
                "args": tuple(self._lookup_args(line.call_args)),
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
        for n in gm.graph.nodes:
            if "example_value" not in n.meta and n.meta.get("val") is not None:
                n.meta["example_value"] = n.meta["val"]
        example_inputs = [
            n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"
        ]
        return _active_fusion_backend(gm, example_inputs)


@contextlib.contextmanager
def enable_device_with_fusion(
    device: str, gm_backend: "Callable | None" = None
):
    """Enable fx_wrapper capture on a device while keeping its real fusion.

    Unlike enable_device_via_fallback (all-fallback, zero fusion), this swaps only
    the device's fx_wrapper_codegen for CompiledKernelFxWrapper and leaves its
    scheduling / wrapper / cpp_wrapper untouched, so the device's own fused kernels
    are codegen'd and embedded into the host gm as CompiledKernelWrapperMutation HOP
    nodes. gm_backend is optional; when omitted the host gm runs via gm.forward.
    Restores the swapped registry / config on exit. Process-global, not thread-safe.
    """
    global _active_fusion_backend
    import torch._inductor.config as inductor_config
    from torch._inductor.codegen.common import (
        device_codegens,
        init_backend_registration,
    )

    device = torch.device(device).type

    init_backend_registration()
    if device not in device_codegens:
        raise RuntimeError(f"no inductor backend registered for device {device!r}")
    saved_dc = device_codegens[device]
    saved_backend = _active_fusion_backend

    device_codegens[device] = dataclasses.replace(
        saved_dc, fx_wrapper_codegen=CompiledKernelFxWrapper
    )
    _active_fusion_backend = gm_backend
    try:
        with inductor_config.patch(
            {"fx_wrapper": True, "size_asserts": False, "alignment_asserts": False}
        ):
            yield
    finally:
        _active_fusion_backend = saved_backend
        device_codegens[device] = saved_dc


# --- default backend: Inductor CPU cpp_pybinding ----------------------------
def cpp_mutated_arg_indices(arg_types) -> tuple[int, ...]:
    """A cpp kernel arg is mutated iff it is a non-const pointer. cpp_argdefs()
    emits writeable buffers (inplace + output) as 'T*' and read-only inputs as
    'const T*'; sizevars have no '*'. Reusable by other C-ABI backends."""
    return tuple(
        i
        for i, t in enumerate(arg_types)
        if isinstance(t, str) and "*" in t and not t.strip().startswith("const")
    )


class CppPybindingBackend(CompiledKernelBackend):
    """Inductor CPU cpp_fused_* kernels: kernel_body is async_compile.cpp_pybinding(...),
    loaded+cached through PyCodeCache; a non-const pointer arg is a written buffer."""

    def handles_definition(self, defn_line) -> bool:
        return not getattr(defn_line, "gpu", True)

    def compile_kernel(self, converter, defn_line) -> Callable:
        code = PythonWrapperCodegen._format_kernel_definition(
            defn_line.kernel_name, defn_line.kernel_body, metadata=defn_line.metadata
        )
        mod = PyCodeCache.load("\n".join([converter.prologue, code]))
        kernel = getattr(mod, defn_line.kernel_name)
        if isinstance(kernel, LambdaFuture):
            kernel = kernel.result()
        if isinstance(kernel, CachingAutotuner):
            raise AssertionError("Triton kernel reached the compiled (cpp) backend")
        return kernel

    def mutated_arg_indices(self, call_line) -> tuple[int, ...]:
        return cpp_mutated_arg_indices(call_line.arg_types)


register_compiled_kernel_backend(CppPybindingBackend())
