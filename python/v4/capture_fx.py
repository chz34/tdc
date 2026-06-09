"""v4 capture_fx -- grab inductor's compiled FX host graph(s).

With inductor's fx_wrapper backend, the host wrapper is built as a
torch.fx.GraphModule (allocs + triton_kernel_wrapper_mutation launches + aten
fallbacks) -- i.e. inductor's fused output expressed as a graph. That gm is only
reachable at the WrapperFxCodegen.compile_graph hook; the upper torch.compile
object does not expose it (see DESIGN.md section 18).

Two entry points:
  - capture_fx(fn, *example_args): returns both the normal torch.compile callable
    AND the captured host gm(s), so the caller can run the compiled fn or process
    the gm by other means (e.g. feed it to v2).
  - compile_with_gm_backend(fn, gm_backend): returns a callable that routes the
    host gm through a user backend on every (re)compile, callable with the
    ORIGINAL fn args (the front-end flattens). Robust to shape/structure changes.

Both hook WrapperFxCodegen.compile_graph via a clean registered subclass.

Design: docs/specs/2026-06-04-v4-fx-capture-design.md.

Works wherever inductor's host graph is FX-convertible: GPU/Triton fused kernels
convert directly; on CPU it works for graphs whose host code has no fused C++
kernel (e.g. all-fallback / pure-extern). A graph with a cpp_fused kernel will
raise inductor's "FX conversion only supports Triton kernels" at prime time --
we let that surface rather than pre-guarding on device.

    r = capture_fx(fn, *example_args)
    out = r.compiled(*example_args)   # run the normal fused result, or
    host_gm = r.gms[0]                # process the host fx graph yourself
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen


# Child of torch's inductor logger: TORCH_LOGS=inductor puts torch._inductor at
# INFO (+inductor -> DEBUG), which this child inherits and propagates to torch's
# log handler. Logging at INFO thus prints iff TORCH_LOGS includes inductor; with
# no TORCH_LOGS the logger sits at WARNING and the line is suppressed for free.
_log = logging.getLogger("torch._inductor.compile_with_gm_backend")


# CaptureFxWrapper instances are created fresh per inductor compile (.create()),
# so captured graphs cannot live on the instance -- they go to this context-local
# sink, set up by _capture_context and appended to by compile_graph.
_active_sink: "list | None" = None

# Backend installed by compile_with_gm_backend (approach B); BackendFxWrapper
# reads it. Context-local, like _active_sink.
_active_gm_backend: "Callable | None" = None

# fx_wrapper on + the asserts that emit raw (non-FX) lines off. Shared by both
# the capture and the backend paths.
_FX_CONFIG = {"fx_wrapper": True, "size_asserts": False, "alignment_asserts": False}


@contextlib.contextmanager
def _install_fx_wrapper(device: str, wrapper_cls: type):
    """Swap `wrapper_cls` in as the device's fx wrapper + enable _FX_CONFIG,
    restoring both on exit. wrapper_cls subclasses the built-in WrapperFxCodegen
    (device-agnostic), so this works for ANY inductor-supporting device --
    including ones that never registered an fx_wrapper_codegen. NOT thread-safe
    (process-global swap)."""
    import torch._inductor.config as inductor_config
    from torch._inductor.codegen.common import (
        device_codegens,
        init_backend_registration,
    )

    init_backend_registration()
    if device not in device_codegens:
        raise RuntimeError(f"no inductor backend registered for device {device!r}")
    dc = device_codegens[device]
    saved_wrapper = dc.fx_wrapper_codegen
    dc.fx_wrapper_codegen = wrapper_cls
    try:
        with inductor_config.patch(_FX_CONFIG):
            yield
    finally:
        dc.fx_wrapper_codegen = saved_wrapper


class CaptureFxWrapper(WrapperFxCodegen):
    """Clean WrapperFxCodegen subclass that records each compiled gm.

    Only compile_graph is overridden. create() is inherited: it is a classmethod
    using `cls`, so it already instantiates this subclass (and its subgraph
    variant) correctly -- no override needed.
    """

    def compile_graph(self, gm):
        if _active_sink is not None:
            _active_sink.append(gm)
        # Return the normal compiled fn (gm.forward) unchanged, so the upper
        # torch.compile result still runs exactly as it would without capture.
        return super().compile_graph(gm)


@contextlib.contextmanager
def _capture_context(device: str):
    """Install CaptureFxWrapper for `device` and yield the gm sink. Restores the
    sink + registry + config on exit (even on exception)."""
    global _active_sink
    saved_sink = _active_sink
    sink: list = []
    _active_sink = sink
    try:
        with _install_fx_wrapper(device, CaptureFxWrapper):
            yield sink
    finally:
        _active_sink = saved_sink


def _infer_device(example_args: Any) -> str:
    for x in example_args:
        if isinstance(x, torch.Tensor):
            return x.device.type
    return "cpu"


@dataclass
class FxCaptureResult:
    """compiled: the normal torch.compile callable (run it to get fused output).
    gms: captured inductor host FX graphs, in compile_graph call order (forward,
    backward, recompiles, partitions...)."""

    compiled: Callable
    gms: list = field(default_factory=list)


def capture_fx(fn: Callable, *example_args: Any, dynamic: bool = True) -> FxCaptureResult:
    """Compile fn under fx_wrapper and capture the host gm(s).

    Primes the compile once with example_args (that is what triggers
    compile_graph and thus the capture). Returns both the compiled callable and
    the captured graphs; the caller decides what to do with each.
    """
    device = _infer_device(example_args)
    print("[v4.capture_fx] size_asserts & alignment_asserts disabled so the "
          "host graph is FX-convertible; the captured gm and compiled fn run "
          "without those runtime size/alignment checks.")
    with _capture_context(device) as sink:
        compiled = torch.compile(fn, backend="inductor", dynamic=dynamic)
        compiled(*example_args)  # prime -> triggers compile_graph -> capture
        gms = list(sink)
    return FxCaptureResult(compiled=compiled, gms=gms)


# ---------------------------------------------------------------------------
# compile_with_gm_backend (approach B): re-route the inductor host gm through a
# user backend on EVERY (re)compile, robust to shape/structure changes.
# ---------------------------------------------------------------------------
class BackendFxWrapper(WrapperFxCodegen):
    """compile_graph applies the active gm_backend to the host gm instead of
    returning gm.forward. example_inputs are taken from the placeholder metas
    (symints + fake tensors) -- the gm's flattened inputs."""

    def compile_graph(self, gm):
        if _active_gm_backend is None:
            return super().compile_graph(gm)
        # The host gm uses the AOT/inductor convention meta["val"]; Dynamo-style
        # backends read meta["example_value"]. Mirror val -> example_value so such
        # a backend finds the fakes it expects. Safe: compile_graph is inductor's
        # last touch of this gm (we only ADD a key, don't change val / graph /
        # gm.code), and we skip val=None nodes (0-return ops) to match Dynamo,
        # which leaves example_value unset on value-less nodes.
        for n in gm.graph.nodes:
            if "example_value" not in n.meta and n.meta.get("val") is not None:
                n.meta["example_value"] = n.meta["val"]
        # Dump the host graph we are about to hand to the user backend; INFO so it
        # prints only when TORCH_LOGS includes inductor (gm.graph __str__ is via %s,
        # deferred -- no cost when suppressed).
        _log.info("compile_with_gm_backend: host gm before gm_backend:\n%s", gm.graph)
        example_inputs = [
            n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"
        ]
        return _active_gm_backend(gm, example_inputs)


@contextlib.contextmanager
def _backend_context(device: str, gm_backend: Callable):
    global _active_gm_backend
    saved = _active_gm_backend
    _active_gm_backend = gm_backend
    try:
        with _install_fx_wrapper(device, BackendFxWrapper):
            yield
    finally:
        _active_gm_backend = saved


def compile_with_gm_backend(
    fn: Callable, gm_backend: Callable, dynamic: bool = True
) -> Callable:
    """Return a callable that runs fn but routes inductor's host FX graph through
    `gm_backend`. Robust (approach B): the capture context is re-entered on EVERY
    call, so any (re)compile -- including for a new shape/structure after the
    first -- still applies gm_backend (no degradation to vanilla inductor). No
    example args needed: it compiles lazily on the first call, inside the context.

    Call the returned callable with the ORIGINAL fn args; the Dynamo/AOT
    front-end flattens them to the gm's inputs, so the caller never handles the
    flattened (symint + tensor) graph inputs.

    gm_backend(gm, example_inputs) -> callable:
      gm            -- inductor host GraphModule (alloc + triton HOP + fallback)
      example_inputs-- the gm's placeholder metas (symints + fake tensors)
      returns       -- a callable taking the flattened inputs (gm.forward's
                       positional convention); must handle host-graph nodes.

    Cost: a per-call inductor_config.patch + fx-wrapper swap (microseconds even
    on cache hits); process-global, not thread-safe.
    """
    compiled = torch.compile(fn, backend="inductor", dynamic=dynamic)

    def runner(*args: Any, **kwargs: Any):
        device = _infer_device(args)
        with _backend_context(device, gm_backend):
            return compiled(*args, **kwargs)

    return runner


# ---------------------------------------------------------------------------
# enable_device_via_fallback: minimal inductor bring-up for a device with NO
# codegen backend -- all-fallback (no fusion) + fx_wrapper to a user backend.
# ---------------------------------------------------------------------------

# Self-contained all-fallback helpers (v4 stays a single-file pure-Python module,
# no cross-version deps). Replacing every OpOverload lowering with its fallback
# handler means no op is fused, so the host graph is all-extern and FX-convertible;
# NO_FUSION_CONFIG turns off the remaining fusion / cudagraph / freezing knobs.
# NOT thread-safe (patches the module-level lowerings dict).
NO_FUSION_CONFIG = {
    "epilogue_fusion": False,
    "max_fusion_size": 1,
    "triton.cudagraphs": False,
    "freezing": False,
}


@contextlib.contextmanager
def force_all_fallback_lowerings():
    """Replace every OpOverload lowering with its fallback_handler (no fusion),
    restoring the lowerings dict on exit. No config changes."""
    from torch._inductor.lowering import fallback_handler, lowerings

    saved = dict(lowerings)
    try:
        for key in list(lowerings.keys()):
            if isinstance(key, torch._ops.OpOverload):
                lowerings[key] = fallback_handler(key, add_to_fallback_set=False)
        yield
    finally:
        lowerings.clear()
        lowerings.update(saved)


class _NoFusionScheduling(CppScheduling):
    """Placeholder device scheduling for fallback-only bring-up. Any attempt to
    codegen a fused (non-extern) kernel hard-errors here, so a fusion that
    slipped past all-fallback fails loudly instead of miscompiling foreign-device
    memory as host pointers. Extern/fallback nodes never reach these methods --
    the scheduler routes them through codegen_extern_call, not the device backend.
    """

    def _reject(self, what: str):
        raise RuntimeError(
            f"enable_device_via_fallback: a {what} reached device codegen -- "
            "all-fallback was not airtight (an op was fused instead of falling "
            "back). Add a fallback for the offending op."
        )

    def codegen_node(self, *args, **kwargs):
        self._reject("fused kernel")

    def codegen_template(self, *args, **kwargs):
        self._reject("template kernel")


def _assert_host_gm_all_extern(gm) -> None:
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_functional,
        triton_kernel_wrapper_mutation,
    )

    fused = {triton_kernel_wrapper_mutation, triton_kernel_wrapper_functional}
    bad = [n.name for n in gm.graph.nodes
           if n.op == "call_function" and n.target in fused]
    if bad:
        raise RuntimeError(
            f"enable_device_via_fallback: host gm is not all-extern; fused "
            f"kernel node(s) present: {bad}"
        )


@contextlib.contextmanager
def enable_device_via_fallback(device: str, gm_backend: "Callable | None" = None):
    """Bring up inductor on a device that has NO codegen backend registered,
    with zero device codegen:
      - register the device with a no-op scheduling (fused kernel -> hard error),
        placeholder python/cpp wrappers, and BackendFxWrapper;
      - force every op to an aten fallback (nothing fuses; compute is plain
        dispatched aten on the device);
      - route the resulting all-extern host gm to gm_backend via fx_wrapper,
        after asserting it really is all-extern.

    gm_backend is optional: when omitted, the host gm runs directly via
    gm.forward (dispatched aten on the device, with inductor's memory planning) --
    i.e. pure enablement, no backend substitution. Pass a gm_backend(gm,
    example_inputs) -> callable only if you want to process/replace the host graph.

    This is an ENABLEMENT path (eager-grade perf, no fusion), not a perf path.
    The device must have aten kernels for every op it runs. Restores all swapped
    registries / lowerings / config on exit. Process-global, not thread-safe.
    """
    global _active_gm_backend
    import torch._inductor.config as inductor_config
    from torch._inductor.codegen.common import (
        custom_backend_passes,
        device_codegens,
        init_backend_registration,
        register_backend_for_device,
    )
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

    def _validating_backend(gm, example_inputs):
        _assert_host_gm_all_extern(gm)
        if gm_backend is None:
            return gm.forward  # default: run the host graph directly
        return gm_backend(gm, example_inputs)

    # Register the standard devices first (idempotent, cached). Without this, a
    # device the stock registration WOULD provide (e.g. cpu) may be absent at
    # enter-time -> had=False -> popped on exit, and since init is @cache it never
    # re-registers, breaking later compiles. After this, had is accurate: True for
    # a stock device (restore it on exit), False for a genuinely new one (pop it).
    init_backend_registration()
    had = device in device_codegens
    saved_dc = device_codegens.get(device)
    saved_pass = custom_backend_passes.get(device)
    saved_backend = _active_gm_backend

    register_backend_for_device(
        device, _NoFusionScheduling, PythonWrapperCodegen, CppWrapperCpu,
        BackendFxWrapper,
    )
    _active_gm_backend = _validating_backend
    try:
        with force_all_fallback_lowerings(), inductor_config.patch(
            {
                "fx_wrapper": True,
                "size_asserts": False,
                "alignment_asserts": False,
                **NO_FUSION_CONFIG,
            }
        ):
            yield
    finally:
        _active_gm_backend = saved_backend
        if had:
            device_codegens[device] = saved_dc
            custom_backend_passes[device] = saved_pass
        else:
            device_codegens.pop(device, None)
            custom_backend_passes.pop(device, None)
