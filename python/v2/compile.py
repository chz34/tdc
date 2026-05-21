"""v2.capture() — direct-replay entry that bypasses Dynamo at call time.

The historical @tdcv2.compile decorator (a thin wrapper around
torch.compile + fw_compiler) was removed: it was strictly slower than
the dynamo eager backend and provided no value v2.capture doesn't
already cover. capture() runs torch.compile internally exactly once to
materialise the trace, then returns a callable whose per-call overhead
is just IValue conversion + C++ replay.

Trade-off: the user must guarantee future calls match example_args in
arg structure (rank, dtype, device). Sym dimensions may vary; concrete
ints that Dynamo specialised on are baked as constants.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd

from .translator import translate_graph


# A recipe spec is a tagged tuple describing how to fetch one placeholder
# slot at replay time. _compile_flat_recipe collapses a list of specs into
# a single source-generated function so that materialising all N
# placeholders costs one Python call instead of N.
#
#   ("T", user_arg_idx)            : Tensor placeholder
#   ("S", user_arg_idx, dim)       : SymInt from a Tensor's shape
#   ("L", value)                   : literal (closure constant or concrete int)
RecipeSpec = Union[
    Tuple[str, int],         # ("T", i)
    Tuple[str, int, int],    # ("S", i, d)
    Tuple[str, Any],         # ("L", value)
]


def capture(
    fn,
    *example_args,
    allow_grad: bool = False,
    wrapper: bool = True,
    **example_kwargs,
):
    """Run `fn(*example_args, **example_kwargs)` once through torch.compile
    to extract a Trace, then return a callable that replays the trace
    on fresh args.

    `wrapper` (default True) controls how the replay callable handles
    the AOT runtime layer:

      - `wrapper=True`  (default): wrap the trace with
        `torch._functorch.aot_autograd.aot_function`. The returned
        callable goes through PyTorch's native RuntimeWrapper on every
        call — input mutations get written back, output aliases get
        rebuilt, the user-visible pytree is restored automatically.
        Adds ~5-15us per call vs the bare path but matches eager
        semantics exactly. Required for training (optimizer.step,
        any in-place parameter / KV-cache mutation).

      - `wrapper=False`: return our direct_replay callable
        (flat_recipe + param pre-bind + id()-based output_shaper). No
        AOT RuntimeWrapper on the call path: fastest possible per-call
        overhead (~5-10us total) but the caller is responsible for
        anything mutation-related. Suitable for pure inference with
        no in-place side effects.

    Caveats for `wrapper=True`:
      - Functions that reference nn.Modules through a non-Module
        container (list, dict, dataclass, plain object attr-chain)
        aren't detected by our closure-scan shim. The error message
        will tell you to use wrapper=False or restructure. Modules
        held directly in closure cells, globals, or as an outer
        Module's attribute *are* handled.
      - Models with tied weights (e.g. BERT's shared attention) hit
        aot_module's `_reparametrize_module` tied-key error; use
        wrapper=False for those.
      - Combined with allow_grad=True we silently downgrade to
        wrapper=False (aot_function carries its own autograd path).

    Either way, Dynamo guards / cache / recompile are NOT on the call
    path — the caller must guarantee future args match example_args
    in rank/dtype/device.

    `allow_grad=True`: also capture the backward graph. Requires at
    least one example_arg with requires_grad=True. The returned
    callable is wrapped in torch.autograd.Function. Only compatible
    with `wrapper=False` for now (the wrapped path uses aot_function
    which already wires up backward via its own autograd.Function).

    Kwargs: example_kwargs are flattened into the positional arg list
    in declared order so the trace's recipes can address them by
    index. The returned callable accepts the same kwarg names as fn
    (mixing positional and kwarg-style invocation is allowed).
    """
    # allow_grad uses its own torch.autograd.Function-based dual-graph
    # path; the aot_function wrapper has its own incompatible mechanism
    # for backward (it returns a callable that already triggers bw via
    # autograd.Function on `.backward()`). Silently fall back to the
    # direct path so the user doesn't have to thread two flags.
    if allow_grad and wrapper:
        wrapper = False

    # If the user only passed positional args AND fn isn't keyword-only,
    # the existing positional-only path is enough.
    if not example_kwargs:
        return _capture_positional(fn, example_args, allow_grad, wrapper)

    # Mixed/kwarg path: use inspect.signature to canonicalise parameter
    # ordering so both the capture call and every later call site can
    # use any mix of positional / keyword and we'll route via param name.
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError) as e:
        raise RuntimeError(
            f"v2.capture: can't introspect signature of {fn!r} but "
            f"example_kwargs were passed; pass example_args positionally "
            f"or provide a fn with a Python-introspectable signature.") from e

    bound = sig.bind(*example_args, **example_kwargs)
    bound.apply_defaults()
    param_names = tuple(bound.arguments.keys())
    ordered_values = tuple(bound.arguments[n] for n in param_names)

    def _wrapped(*flat):
        return fn(**dict(zip(param_names, flat)))

    _wrapped.__name__ = getattr(fn, "__name__", "fn") + "__kwflat"
    inner = _capture_positional(_wrapped, ordered_values, allow_grad, wrapper)

    def call(*args, **kwargs):
        bound_call = sig.bind(*args, **kwargs)
        bound_call.apply_defaults()
        flat = [bound_call.arguments[n] for n in param_names]
        return inner(*flat)

    # Forward debug attrs from inner so introspection still works.
    for attr in ("trace", "recipe_specs", "flat_recipe",
                 "fw_trace", "bw_trace", "fw_recipe_specs"):
        if hasattr(inner, attr):
            setattr(call, attr, getattr(inner, attr))
    return call


def _capture_positional(fn, example_args, allow_grad: bool, wrapper: bool = True):
    """Internal: capture path with strictly-positional example_args."""
    if allow_grad:
        # allow_grad uses its own autograd.Function path and is
        # incompatible with the aot_function wrapper. Callers will
        # have hit the NotImplementedError in `capture` already.
        return _capture_with_backward(fn, example_args)

    if wrapper:
        return _capture_via_aot_wrapper(fn, example_args)

    # ---- inference-only path (no backward graph) ----
    captured: list = []

    def grab_compiler(gm, sample_inputs):
        trace = translate_graph(gm)
        state = {
            "trace": trace,
            "gm": gm,
            "observed_args": None,
            "last_trace_out": None,
        }
        captured.append(state)
        def wrapping_cb(*args):
            if state["observed_args"] is None:
                state["observed_args"] = list(args)
            result = trace.v2_replay(list(args))
            # Snapshot every wrapping_cb invocation; the final one
            # corresponds to the result AOT will reshape into the
            # user-visible structure (see below).
            state["last_trace_out"] = list(result)
            return result[0] if len(result) == 1 else tuple(result)
        return wrapping_cb

    with torch.no_grad():
        compiled_fn = torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=grab_compiler),
            dynamic=True,
        )
        # AOT's runtime layer turns wrapping_cb's flat tuple into the
        # final user-visible structure (drops saved-for-backward
        # intermediates, restores the original Python container).
        # Capture that structure so direct_replay can reproduce it.
        user_visible_out = compiled_fn(*example_args)
    if not captured:
        raise RuntimeError(
            "v2.capture: fw_compiler was never called; torch.compile may have "
            "graph-broken on the example function. Inspect with "
            "TORCH_LOGS='graph_breaks' python your_script.py")
    state = captured[0]
    trace = state["trace"]
    if state["observed_args"] is None:
        raise RuntimeError(
            "v2.capture: example call did not exercise the trace; "
            "torch.compile may have specialized to a different cache entry.")
    output_shaper = _build_output_shaper(
        user_visible_out, state["last_trace_out"])
    runtime_specs, pre_binds, _ = _build_recipe_specs(
        state["gm"], example_args, state["observed_args"])
    # Push frozen values (params / Dynamo-specialised constants) into
    # the trace once; subsequent replays skip those slots entirely.
    for arg_idx, value in pre_binds:
        trace.v2_pre_bind(arg_idx, value)
    flat_recipe = _compile_flat_recipe(runtime_specs)

    def direct_replay(*user_args):
        flat = flat_recipe(user_args)
        result = trace.v2_replay(flat)
        return output_shaper(result)

    # Expose internals for introspection / debug.
    direct_replay.trace = trace                  # type: ignore[attr-defined]
    direct_replay.recipe_specs = runtime_specs   # type: ignore[attr-defined]
    direct_replay.pre_binds = pre_binds          # type: ignore[attr-defined]
    direct_replay.flat_recipe = flat_recipe      # type: ignore[attr-defined]
    direct_replay.output_shaper = output_shaper  # type: ignore[attr-defined]
    return direct_replay


class _ClosureModuleShim(torch.nn.Module):
    """Thin nn.Module that registers fn's closure-captured nn.Modules
    as submodules so aot_module can find their Parameters/Buffers.

    Why this works: aot_module's _reparametrize_module mutates module
    attrs via setattr; since both `self._closure_mod_N` and the cell
    captured by fn refer to the *same* nn.Module object, the in-place
    parameter swap is visible through both. Calling self(*args) hits
    self.forward which delegates to fn — fn picks up the swapped
    FakeTensor params through its closure exactly as if they were its
    own attrs.
    """
    def __init__(self, fn, modules):
        super().__init__()
        # add_module dedupes-by-name only; do NOT use the same name
        # twice. Iteration order matches the closure scan.
        for i, m in enumerate(modules):
            self.add_module(f"_closure_mod_{i}", m)
        # Store fn as a plain attr (not a submodule) so __setattr__'s
        # nn.Module / Tensor specialisation doesn't try to register it.
        object.__setattr__(self, "_fn", fn)

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _scan_closure_modules(fn):
    """Return the list of unique nn.Module objects fn references via
    its closure cells OR globals.

    `inspect.getclosurevars` walks fn's bytecode and resolves every
    LOAD_DEREF / LOAD_GLOBAL name to its current value, so we catch
    both `def fn(x): return linear(x)` at module scope (linear is in
    .globals) and the same idiom inside another function (linear is
    in .nonlocals).

    Best-effort: nn.Modules reached only through an intermediate
    object (e.g. a list of models, or `cfg.model`) aren't found by
    name lookup; those still hit aot_function's FakeTensor assert
    and should fall back to wrapper=False."""
    try:
        cv = inspect.getclosurevars(fn)
    except TypeError:
        # Built-ins, C-level callables, etc.
        return []
    seen: set = set()
    modules: list = []
    for source in (cv.nonlocals, cv.globals):
        for val in source.values():
            if isinstance(val, torch.nn.Module) and id(val) not in seen:
                modules.append(val)
                seen.add(id(val))
    return modules


def _capture_via_aot_wrapper(fn, example_args):
    """wrapper=True path. Hand fn to torch's aot_function / aot_module
    so the call site goes through PyTorch's native RuntimeWrapper —
    input mutation writeback, output alias regen, pytree unflatten
    all come for free.

    fw_compiler swaps gm.forward for trace.v2_replay so we still get
    the C++ replay speed-up at the innermost layer; the per-call cost
    is bare RuntimeWrapper, not Dynamo guards.
    """
    from torch._functorch.aot_autograd import aot_function, aot_module

    captured: list = []

    def grab_compiler(gm, example_inputs):
        trace = translate_graph(gm)
        captured.append({"trace": trace, "gm": gm})
        def run_via_trace(*flat_args):
            return trace.v2_replay(list(flat_args))
        return run_via_trace

    # nn.Module parameters live in attributes aot_function can't see
    # (FakeTensor sees the real Parameter and bails). aot_module is the
    # dedicated entry that lifts module params/buffers as explicit
    # graph inputs.
    #
    # The same problem hits plain functions that close over an nn.Module
    # (e.g. `def fn(x): return linear(x)` where `linear` is a Linear in
    # the surrounding scope). aot_function can't see those Parameters
    # any more than it could when they live in `fn`'s attrs. Detect
    # closure-captured Modules and wrap them into a shim nn.Module so
    # aot_module's parameter-lifting handles them. The shim adds each
    # Module via add_module, so aot_module's _reparametrize_module
    # walks them; since the closure cell and the shim attr point to
    # the same object, mutating one is observed by the other.
    if isinstance(fn, torch.nn.Module):
        target = fn
    else:
        closure_mods = _scan_closure_modules(fn)
        target = _ClosureModuleShim(fn, closure_mods) if closure_mods else None

    if target is not None:
        aot_compiled = aot_module(
            target, fw_compiler=grab_compiler, dynamic=True)
    else:
        aot_compiled = aot_function(
            fn, fw_compiler=grab_compiler, dynamic=True)

    # Trigger AOT trace + RuntimeWrapper wiring. The first call also
    # forces fw_compiler to fire, populating `captured`.
    try:
        with torch.no_grad():
            aot_compiled(*example_args)
    except AssertionError as e:
        # FakeTensorMode rejecting a real Parameter is the signature of
        # "fn references an nn.Module through a path our closure scan
        # missed" — e.g. self.cfg.model where cfg isn't an nn.Module,
        # or models stored in a list/dict. Re-raise with the actual
        # remediation rather than the cryptic FakeTensor assert.
        if "convert all Tensors to FakeTensors" in str(e):
            raise RuntimeError(
                "v2.capture(wrapper=True): aot_function/aot_module's "
                "FakeTensorMode hit a real Parameter that our closure-"
                "scan shim didn't lift. fn likely references an "
                "nn.Module through a non-Module container (list, dict, "
                "dataclass, plain object attr-chain). Either:\n"
                "  - move the Module into a closure cell / global / "
                "self.X attr so it's name-resolvable, or\n"
                "  - pass the outer Module directly to v2.capture, or\n"
                "  - use wrapper=False which goes through torch.compile "
                "(Dynamo can resolve any attribute chain).\n"
                f"original error: {e}"
            ) from e
        raise

    if not captured:
        raise RuntimeError(
            "v2.capture(wrapper=True): fw_compiler was never called; "
            "aot_function/aot_module may have graph-broken on the "
            "example function.")

    state = captured[0]
    # Expose internals so callers can still introspect the trace even
    # though the call path now goes through aot_function.
    aot_compiled.trace = state["trace"]    # type: ignore[attr-defined]
    aot_compiled.gm = state["gm"]          # type: ignore[attr-defined]
    return aot_compiled


def _build_output_shaper(user_visible_out, trace_out):
    """Given the structure returned by AOT's runtime layer and the
    list of raw trace outputs that fed into it, produce a function
    `shaper(result) -> structure` that reshapes a fresh trace.v2_replay
    result the same way on every call.

    Why: AOT's FW graph often emits more outputs than the user sees
    (intermediates saved for the backward, or simply outputs the AOT
    runtime drops/reorders). trace_v2 always returns ALL graph outputs,
    so direct_replay must drop / reorder / repack them to match what
    the user got out of compiled_fn(*example_args).

    Matching is by tensor `id()` — AOT's runtime layer aliases (does
    not clone) when packaging outputs, so a Tensor in user_visible_out
    is the *same* Python object as the corresponding entry in
    trace_out. We record (kind, index) tuples that direct_replay
    consumes against a fresh trace_out list."""
    id_to_idx = {
        id(v): i for i, v in enumerate(trace_out) if isinstance(v, torch.Tensor)
    }

    def _plan(v):
        if isinstance(v, torch.Tensor):
            idx = id_to_idx.get(id(v))
            if idx is None:
                # AOT cloned/derived a tensor we never returned to it
                # — fall back to keeping all trace outputs as a flat
                # tuple. Loses the user-visible structure but keeps
                # correctness; warn so this is visible.
                raise _OutputShaperBail(
                    f"AOT returned a Tensor (shape={tuple(v.shape)}) that "
                    "doesn't match any trace output by id(); AOT may have "
                    "cloned/aliased an output, which v2.capture's id()-based "
                    "matching doesn't handle.")
            return ("T", idx)
        if isinstance(v, tuple):
            return ("tuple", [_plan(e) for e in v])
        if isinstance(v, list):
            return ("list", [_plan(e) for e in v])
        if v is None:
            return ("none",)
        # Scalars, etc. — best to pass-through as a literal-by-value
        # snapshot. Rare in AOT outputs.
        return ("literal", v)

    try:
        plan = _plan(user_visible_out)
    except _OutputShaperBail as e:
        print(f"# v2.capture: output_shaper fallback ({e})")
        n = len(trace_out)
        def fallback(result):
            return result[0] if len(result) == 1 else tuple(result[:n])
        return fallback

    def apply(plan, result):
        kind = plan[0]
        if kind == "T":
            return result[plan[1]]
        if kind == "tuple":
            return tuple(apply(p, result) for p in plan[1])
        if kind == "list":
            return [apply(p, result) for p in plan[1]]
        if kind == "none":
            return None
        if kind == "literal":
            return plan[1]
        raise AssertionError(f"unhandled plan kind: {kind}")

    def shaper(result):
        return apply(plan, result)
    return shaper


class _OutputShaperBail(Exception):
    """Sentinel signalling that the id()-based output mapping couldn't
    cover the user-visible structure. Caught by _build_output_shaper
    which then returns a flat-passthrough fallback shaper."""
    pass


def _capture_with_backward(fn, example_args):
    """allow_grad=True path. Capture fw + bw graphs in one example call
    and return an autograd.Function-wrapped callable so the captured
    pair runs as a normal differentiable PyTorch op."""
    if not any(isinstance(a, torch.Tensor) and a.requires_grad for a in example_args):
        raise RuntimeError(
            "v2.capture(allow_grad=True): at least one example arg must "
            "have requires_grad=True so AOTAutograd produces a backward "
            "graph during the example call.")

    captured: list = []

    def grab_compiler(gm, sample_inputs):
        trace = translate_graph(gm)
        state = {"trace": trace, "gm": gm, "observed_args": None}
        captured.append(state)
        def wrapping_cb(*args):
            if state["observed_args"] is None:
                state["observed_args"] = list(args)
            result = trace.v2_replay(list(args))
            return result[0] if len(result) == 1 else tuple(result)
        return wrapping_cb

    compiled_fn = torch.compile(
        fn,
        backend=aot_autograd(
            fw_compiler=grab_compiler,
            bw_compiler=grab_compiler,
        ),
        dynamic=True,
    )
    out = compiled_fn(*example_args)
    # Force a backward so AOTAutograd materialises the bw graph and our
    # grab_compiler gets invoked a second time.
    if isinstance(out, torch.Tensor):
        out.sum().backward()
    elif isinstance(out, (tuple, list)):
        torch.stack([
            o.flatten().sum() for o in out if isinstance(o, torch.Tensor)
        ]).sum().backward()
    else:
        raise RuntimeError(
            f"v2.capture(allow_grad=True): unsupported output type {type(out)}")

    if len(captured) < 2:
        raise RuntimeError(
            f"v2.capture(allow_grad=True): expected fw + bw compile (2 entries), "
            f"got {len(captured)}. backward() may not have triggered a "
            f"separate bw compile in this AOTAutograd configuration.")

    fw_state = captured[0]
    bw_state = captured[1]

    fw_specs, fw_pre_binds, fw_ph_to_user_input = _build_recipe_specs(
        fw_state["gm"], example_args, fw_state["observed_args"])
    fw_trace_obj = fw_state["trace"]
    for arg_idx, value in fw_pre_binds:
        fw_trace_obj.v2_pre_bind(arg_idx, value)
    fw_flat = _compile_flat_recipe(fw_specs)

    n_user_inputs = len(example_args)

    # Count tangent placeholders in bw_gm to learn how many of fw's
    # outputs are user-visible (the rest are saved-for-backward).
    n_tangents = sum(
        1 for n in bw_state["gm"].graph.nodes
        if n.op == "placeholder" and n.name.startswith("tangents_")
    )
    if n_tangents == 0:
        raise RuntimeError(
            "v2.capture(allow_grad=True): bw graph has no 'tangents_*' "
            "placeholders; cannot tell user outputs from saved tensors.")

    # AOTAutograd internally re-orders fw outputs before feeding them
    # to bw — bw's placeholder order does not match fw's output order.
    # Build per-bw-placeholder routing by matching FX node names:
    # bw placeholder 'primals_1' references fw output node 'primals_1';
    # bw placeholder 'tangents_N' is the (N-1)-th element of grad_outputs.
    fw_out_node = [n for n in fw_state["gm"].graph.nodes if n.op == "output"][0]
    fw_output_args = fw_out_node.args[0]  # tuple of fx.Node
    fw_output_names = [n.name for n in fw_output_args]
    bw_placeholders = [n for n in bw_state["gm"].graph.nodes if n.op == "placeholder"]
    bw_arg_sources: list = []   # (kind, idx) per bw placeholder
    for n in bw_placeholders:
        if n.name.startswith("tangents_"):
            t_idx = int(n.name.split("_")[1]) - 1   # tangents_1 -> 0
            bw_arg_sources.append(("tangent", t_idx))
        elif n.name in fw_output_names:
            bw_arg_sources.append(("fw_out", fw_output_names.index(n.name)))
        else:
            raise RuntimeError(
                f"v2.capture(allow_grad=True): bw placeholder '{n.name}' "
                f"is neither a tangent nor a known fw output. "
                f"Known fw outputs: {fw_output_names}")

    fw_trace = fw_state["trace"]
    bw_trace = bw_state["trace"]

    class _CapturedFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *user_args):
            fw_outputs = fw_trace.v2_replay(fw_flat(user_args))
            # Save the full fw_outputs list — split into tensors (via
            # save_for_backward, autograd requirement) + non-tensors
            # (stashed as ctx attributes), with positions preserved so
            # we can reconstruct the full list in backward.
            tensors = []
            tensor_positions = []
            other_values = []
            other_positions = []
            for i, v in enumerate(fw_outputs):
                if isinstance(v, torch.Tensor):
                    tensors.append(v)
                    tensor_positions.append(i)
                else:
                    other_values.append(v)
                    other_positions.append(i)
            ctx.save_for_backward(*tensors)
            ctx.tensor_positions = tensor_positions
            ctx.other_values = other_values
            ctx.other_positions = other_positions
            ctx.fw_out_len = len(fw_outputs)
            user_outputs = fw_outputs[:n_tangents]
            if len(user_outputs) == 1:
                return user_outputs[0]
            return tuple(user_outputs)

        @staticmethod
        def backward(ctx, *grad_outputs):
            # Reconstruct fw_outputs list from ctx's split storage.
            fw_outputs_full: list = [None] * ctx.fw_out_len
            for idx, t in zip(ctx.tensor_positions, ctx.saved_tensors):
                fw_outputs_full[idx] = t
            for idx, v in zip(ctx.other_positions, ctx.other_values):
                fw_outputs_full[idx] = v
            # Assemble bw inputs in bw's placeholder order using the
            # name-based routing built at capture time.
            bw_args = []
            for kind, idx in bw_arg_sources:
                if kind == "fw_out":
                    bw_args.append(fw_outputs_full[idx])
                else:  # tangent
                    bw_args.append(grad_outputs[idx])
            bw_outputs = bw_trace.v2_replay(bw_args)
            # bw outputs are aligned with fw INPUT placeholders.
            # Map back to user_input grads via fw recipes.
            input_grads = [None] * n_user_inputs
            for fw_ph_idx, user_idx in fw_ph_to_user_input.items():
                if fw_ph_idx < len(bw_outputs):
                    input_grads[user_idx] = bw_outputs[fw_ph_idx]
            return tuple(input_grads)

    def call(*user_args):
        return _CapturedFn.apply(*user_args)

    call.fw_trace = fw_trace                  # type: ignore[attr-defined]
    call.bw_trace = bw_trace                  # type: ignore[attr-defined]
    call.fw_recipe_specs = fw_specs           # type: ignore[attr-defined]
    return call


def _build_recipe_specs(
    gm: fx.GraphModule, example_args, observed_args
):
    """For each gm placeholder, decide whether it varies with the user
    call (runtime spec) or is constant across calls (pre-bind).

    Returns (runtime_specs, pre_binds) where:
      - runtime_specs: list of ("T", i) / ("S", i, dim) tuples. Each
        becomes an expression in the generated flat_recipe (one entry
        in v2_replay's args list per call).
      - pre_binds: list of (arg_idx, value) tuples. Applied via
        trace.v2_pre_bind() once at capture; replays skip these slots
        entirely (no pybind round-trip, no captured_tensors_ overwrite).

    Tensor placeholders:
      id() match with example_args  -> ("T", user_idx) runtime spec
      otherwise (module param/buffer)-> (arg_idx, observed_tensor) pre-bind

    SymInt placeholders:
      symbol appears in a USER-INPUT Tensor's FakeTensor.shape
        -> ("S", user_idx, dim) runtime spec
      otherwise (Dynamo-specialised closure const, etc.)
        -> (arg_idx, observed_int) pre-bind
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(observed_args):
        raise RuntimeError(
            f"v2.capture: gm has {len(placeholders)} placeholders but "
            f"observed_args has {len(observed_args)}.")

    # id()-based map of user-input Tensors -> position in example_args.
    user_tensor_ids: dict = {
        id(a): i for i, a in enumerate(example_args)
        if isinstance(a, torch.Tensor)
    }

    # Pre-pass: classify each placeholder as user-input vs param/other.
    is_user_input: List[bool] = []
    for ph_idx, n in enumerate(placeholders):
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor):
            obs = observed_args[ph_idx]
            is_user_input.append(
                isinstance(obs, torch.Tensor) and id(obs) in user_tensor_ids
            )
        else:
            is_user_input.append(False)

    # SymInt symbol -> (user_arg_idx, dim), built from USER-INPUT shapes
    # only — module params have static shapes so their dims aren't sym.
    symbol_source: dict[str, Tuple[int, int]] = {}
    for ph_idx, n in enumerate(placeholders):
        if not is_user_input[ph_idx]:
            continue
        fake_t = n.meta.get("val")
        user_idx = user_tensor_ids[id(observed_args[ph_idx])]
        for dim, size in enumerate(fake_t.shape):
            if isinstance(size, torch.SymInt):
                key = str(size.node.expr)
                symbol_source.setdefault(key, (user_idx, dim))

    # Sanity: every example_args Tensor must be matched to a placeholder.
    matched = {
        user_tensor_ids[id(observed_args[ph_idx])]
        for ph_idx, b in enumerate(is_user_input) if b
    }
    expected = set(user_tensor_ids.values())
    if matched != expected:
        raise RuntimeError(
            f"v2.capture: example_args[{sorted(expected - matched)}] did "
            f"not appear as Tensor placeholders. The call site must pass "
            f"the same tensor objects through (matched by id()).")

    runtime_specs: List[RecipeSpec] = []
    pre_binds: List[Tuple[int, Any]] = []
    # ph_idx (the original gm placeholder index) -> user_arg_idx, for
    # the subset of placeholders that ARE user-input Tensors. Needed by
    # the backward-replay path which maps bw outputs back to user-input
    # grads — bw outputs are aligned with the FW graph's placeholder
    # positions, NOT with the (smaller) runtime_specs.
    ph_to_user_input: Dict[int, int] = {}
    for ph_idx, n in enumerate(placeholders):
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor):
            if is_user_input[ph_idx]:
                user_idx = user_tensor_ids[id(observed_args[ph_idx])]
                runtime_specs.append(("T", user_idx))
                ph_to_user_input[ph_idx] = user_idx
            else:
                # Module parameter / buffer: pre-bind once.
                pre_binds.append((ph_idx, observed_args[ph_idx]))
        elif isinstance(val, torch.SymInt):
            key = str(val.node.expr)
            if key in symbol_source:
                ua_idx, dim = symbol_source[key]
                runtime_specs.append(("S", ua_idx, dim))
            else:
                pre_binds.append((ph_idx, observed_args[ph_idx]))
        elif isinstance(val, int):
            pre_binds.append((ph_idx, val))
        else:
            raise RuntimeError(
                f"v2.capture: unsupported placeholder val type "
                f"{type(val).__name__} for node {n.name}")
    return runtime_specs, pre_binds, ph_to_user_input


def _compile_flat_recipe(specs: List[RecipeSpec]) -> Callable[[Tuple[Any, ...]], list]:
    """Generate and exec() a single function that materialises all
    NON-pre-bound placeholder values in one Python call.

    Output shape: `def _flat(args): return [<expr_0>, <expr_1>, ...]`
    where each expression is either `args[i]` (T spec) or
    `args[i].size(d)` (S spec). Pre-bound placeholders (param tensors,
    Dynamo-specialised constants) are NOT in the list — they live in
    the trace's persistent buffers, set once via v2_pre_bind.

    Why exec/eval: this collapses N lambda calls (one per placeholder)
    into a single function invocation. On accelerators where kernel
    time is amortised across calls and host-side Python overhead is
    the bottleneck, this can save 1-3us per replay (≈ N * 300-500ns).
    """
    expr_parts: list[str] = []
    for spec in specs:
        if spec[0] == "T":
            expr_parts.append(f"args[{spec[1]}]")
        elif spec[0] == "S":
            expr_parts.append(f"args[{spec[1]}].size({spec[2]})")
        else:
            raise AssertionError(
                f"flat_recipe only handles T/S specs after pre-bind extraction; "
                f"got {spec!r}")

    src = (
        "def _flat_recipe(args):\n"
        f"    return [{', '.join(expr_parts)}]\n"
    )
    ns: dict = {}
    exec(src, ns)
    fn = ns["_flat_recipe"]
    fn._source = src                   # type: ignore[attr-defined]
    return fn
