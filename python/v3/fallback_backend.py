"""v3 fallback backend -- a standalone dynamo backend that runs the
all-fallback cpp_wrapper path through stock inductor.

Goal: every op falls back (no fusion), inductor's cpp_wrapper emits a pure
"allocate + dispatch each op + free" C++ host driver -> fast op dispatch with
no per-op Python overhead.

Design: see docs/specs/2026-06-03-v3-fallback-backend-design.md.

The whole inductor pipeline is reused unchanged; we only
  (1) install a clean CppWrapperCpu subclass as the device's cpp_wrapper
      codegen (so NPU adoption is just register_backend_for_device), and
  (2) set the config bundle (cpp_wrapper on, fusion off) + pick the fallback
      dispatch mechanism (boxed call_dispatcher vs stock c-shim/dispatcher mix).

Usage as a comparison arm:
    compiled = torch.compile(fn, backend=make_fallback_backend("boxed"),
                             dynamic=True)
"""
from __future__ import annotations

import contextlib
import unittest.mock
from typing import Any, Callable, Literal

import torch
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.codegen.wrapper import PythonWrapperCodegen

from .fallback_hijack import force_all_fallback


DispatchMode = Literal["boxed", "stock"]


# Per-device fallback wrapper subclasses.
#
# Each is a thin subclass of the *current upstream* device wrapper, overriding
# only `create` (the base hardcodes `return <Base>()`, so a subclass must
# override it to be instantiated). They add nothing else on purpose: in the
# all-fallback path there are no fused kernels, so the base's device handling
# (CPU: plain calls; GPU: stream / device guard / cuda c-shim naming; the
# Triton-launch machinery stays dormant) is exactly what we need.
#
# Adding a new backend = one subclass + one `_FALLBACK_WRAPPER_BY_DEVICE`
# entry. NPU specifically should register a clean subclass of the upstream
# CppWrapperCpu rather than torch_npu's hand-patched CppWrapperNpu, to avoid
# inheriting its version-drift bugs.


class CppWrapperFallbackCpu(CppWrapperCpu):
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: Any = None,
    ) -> "CppWrapperFallbackCpu":
        return CppWrapperFallbackCpu()


class CppWrapperFallbackGpu(CppWrapperGpu):
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: Any = None,
    ) -> "CppWrapperFallbackGpu":
        return CppWrapperFallbackGpu()


# Backwards-compat alias (original single-class name; CPU is the default).
CppWrapperFallback = CppWrapperFallbackCpu

_FALLBACK_WRAPPER_BY_DEVICE: dict[str, type] = {
    "cpu": CppWrapperFallbackCpu,
    "cuda": CppWrapperFallbackGpu,
    "xpu": CppWrapperFallbackGpu,
}


@contextlib.contextmanager
def _fallback_codegen_context(device: str, mode: str):
    """Scope the config + codegen-class swap for one compile_fx call.

    Everything is restored on exit, even if compilation raises. NOT thread-safe
    (swaps process-global inductor registries) -- same constraint as
    force_all_fallback; fine for single-threaded prototype/benchmark use.
    """
    if mode not in ("boxed", "stock"):
        raise ValueError(f"unknown dispatch mode: {mode!r}")

    from torch._inductor.codegen.common import (
        device_codegens,
        init_backend_registration,
    )

    wrapper_cls = _FALLBACK_WRAPPER_BY_DEVICE.get(device)
    if wrapper_cls is None:
        raise RuntimeError(
            f"no fallback wrapper registered for device {device!r}; "
            f"add an entry to _FALLBACK_WRAPPER_BY_DEVICE "
            f"(have: {sorted(_FALLBACK_WRAPPER_BY_DEVICE)})"
        )

    # Ensure the device's DeviceCodegen entry exists before we patch it.
    init_backend_registration()
    if device not in device_codegens:
        raise RuntimeError(
            f"no inductor backend registered for device {device!r}; "
            "for NPU this is register_backend_for_device('npu', ...)"
        )

    dc = device_codegens[device]
    saved_wrapper = dc.cpp_wrapper_codegen
    dc.cpp_wrapper_codegen = wrapper_cls

    with contextlib.ExitStack() as stack:
        try:
            # cpp_wrapper=True + lowerings all fallback + fusion/cudagraph off.
            stack.enter_context(force_all_fallback())
            if mode == "boxed":
                # Make every aten op look like it has no c-shim, so
                # FallbackKernel.codegen sets use_runtime_dispatch=True and emits
                # the boxed aoti_torch_call_dispatcher path (device-agnostic, no
                # c-shim needed -- ports to NPU as-is). `inductor_fallback_ops`
                # is re-imported inside codegen, so patching the module attr
                # takes effect there.
                import torchgen.aoti.fallback_ops as fallback_ops_mod

                stack.enter_context(
                    unittest.mock.patch.object(
                        fallback_ops_mod, "inductor_fallback_ops", {}
                    )
                )
            yield
        finally:
            dc.cpp_wrapper_codegen = saved_wrapper


def _infer_device(example_inputs: Any) -> str:
    for x in example_inputs:
        if isinstance(x, torch.Tensor):
            return x.device.type
    return "cpu"


def make_fallback_backend(mode: DispatchMode = "boxed") -> Callable:
    """Return a dynamo backend that compiles via stock inductor under the
    all-fallback cpp_wrapper config. `mode` selects the fallback dispatch path:
      - "boxed": every op via aoti_torch_call_dispatcher (portable, no c-shim)
      - "stock": c-shim where available, dispatcher otherwise
    """

    def fallback_inductor_backend(gm: torch.fx.GraphModule, example_inputs: Any):
        from torch._inductor.compile_fx import compile_fx

        device = _infer_device(example_inputs)
        with _fallback_codegen_context(device, mode):
            return compile_fx(gm, example_inputs)

    return fallback_inductor_backend
