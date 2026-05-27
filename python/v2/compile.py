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

from .fx_passes import (
    eliminate_dead_clones,
    rewrite_prims_in_gm,
    rewrite_slice_scatter_to_inplace,
)
from .translator import translate_graph


# Note: we deliberately do NOT pass core_aten_decompositions() to
# aot_function / aot_autograd. That table also decomposes high-level
# ops like aten.linear -> mm + add, which reorders fp32 FMA and
# changes numerical results by ~1e-2 absolute on near-zero outputs of
# 2048-dim matmul (fully within fp32 machine precision but visible at
# tight tolerances). The only prim we actually need to eliminate is
# prims.convert_element_type, and rewrite_prims_in_gm handles it
# directly without touching high-level aten ops. See DESIGN.md §17.6.9.


# A recipe spec is a tagged tuple describing how to fetch one placeholder
# slot at replay time. _compile_flat_recipe collapses a list of specs into
# a single source-generated function so that materialising all N
# placeholders costs one Python call instead of N.
#
#   ("T", user_arg_idx)            : Tensor placeholder
#   ("S", user_arg_idx, dim)       : SymInt from a Tensor's shape
#   ("I", user_arg_idx)            : Python scalar (int / bool) the user
#                                    passed as a positional arg. Dynamo
#                                    lifts it as a SymInt placeholder
#                                    that flows into shape arith, slice
#                                    bounds, etc. -- so a fresh call
#                                    value must be plumbed through, not
#                                    frozen at capture time.
#   ("L", value)                   : literal (closure constant or concrete int)
RecipeSpec = Union[
    Tuple[str, int],         # ("T", i) / ("I", i)
    Tuple[str, int, int],    # ("S", i, d)
    Tuple[str, Any],         # ("L", value)
]


def _check_no_python_scalar_args_for_wrapper(example_args, example_kwargs):
    """Raise if any example arg is a Python scalar (int / bool / float).

    Under wrapper=True we hand `fn` to aot_module/aot_function directly,
    bypassing Dynamo. AOT bakes Python scalars into the FX graph as
    literals at trace time; the corresponding placeholder is left with
    num_users=0 and meta['val']=None, so any value the user passes at
    replay is silently discarded. We refuse rather than capture into a
    silently-wrong trace.

    Tensor inputs (including 0-d / scalar tensors) are always fine.
    """
    def _bad_arg_iter():
        for i, a in enumerate(example_args):
            if not isinstance(a, torch.Tensor) and isinstance(a, (int, float)):
                # bool is a subclass of int; covered by the above.
                yield (f"positional arg #{i}", a)
        for k, v in example_kwargs.items():
            if not isinstance(v, torch.Tensor) and isinstance(v, (int, float)):
                yield (f"keyword arg {k!r}", v)

    bad = list(_bad_arg_iter())
    if not bad:
        return
    details = "; ".join(f"{loc}={v!r}" for loc, v in bad)
    raise RuntimeError(
        "v2.capture(wrapper=True): refusing to capture because the "
        "example args include Python scalar(s) that aot_module would "
        "bake as literals -- subsequent calls with different values "
        "would silently return stale results.\n"
        f"  Offending args: {details}\n"
        "  Fix: pass wrapper=False (Dynamo path) which routes scalar "
        "args through a runtime spec, so they can vary at every replay. "
        "If you genuinely want the scalar frozen at capture, wrap it as "
        "a closure constant (move it out of the forward signature)."
    )


def capture(
    fn,
    *example_args,
    allow_grad: bool = False,
    wrapper: bool = False,
    **example_kwargs,
):
    """Run `fn(*example_args, **example_kwargs)` once through torch.compile
    to extract a Trace, then return a callable that replays the trace
    on fresh args.

    `wrapper` (default False) controls how the replay callable handles
    the AOT runtime layer:

      - `wrapper=False` (default, recommended): return our direct_replay
        callable (flat_recipe + param pre-bind + pytree-based
        output_shaper). The captured trace goes through Dynamo + AOT,
        so Python scalar args get sym-ified and route through
        ("I", arg_idx) recipe specs (varying start_pos and similar
        works correctly). No AOT RuntimeWrapper on the call path:
        smallest possible per-call overhead.

      - `wrapper=True`: wrap the trace with
        `torch._functorch.aot_autograd.aot_function`. The returned
        callable goes through PyTorch's native RuntimeWrapper on every
        call. Useful for narrow scenarios where direct_replay can't
        cover the user's semantics:
          * Tensor subclasses (DTensor, custom __torch_dispatch__)
            that need AOT's runtime layer for metadata propagation.
        Adds ~3-100us per call (more for small workloads) vs the
        bare path. Refuses to capture if the example args include
        Python scalars (aot_module without Dynamo bakes them as
        graph literals and would silently return stale results).

    Caveats for `wrapper=True`:
      - Refused if `example_args` / `example_kwargs` contain Python
        scalars (int, bool, float): aot_module bakes them at trace
        time. Use wrapper=False (the default) for KV-cache and
        similar patterns with int args.
      - Functions that reference nn.Modules through a non-Module
        container (list, dict, dataclass, plain object attr-chain)
        aren't detected by our closure-scan shim.
      - Models with tied weights (e.g. BERT's shared attention) hit
        aot_module's `_reparametrize_module` tied-key error.
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

    # wrapper=True traces through aot_module/aot_function WITHOUT
    # Dynamo. Without Dynamo's int-sym-ification, every Python scalar
    # arg (start_pos, mask flags, ...) gets baked into the graph as a
    # literal at capture time -- and the corresponding placeholder
    # becomes an orphan (num_users=0, meta['val'] is None). Calling
    # with a different scalar value silently uses the captured one.
    # Refuse the capture loudly so users can't be bitten by this.
    # The Dynamo-driven path (wrapper=False) handles scalar args
    # correctly via ("I", arg_idx) runtime specs in
    # _build_recipe_specs.
    if wrapper:
        _check_no_python_scalar_args_for_wrapper(example_args, example_kwargs)

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
        # disable_functionalization=True (passed to aot_autograd below)
        # already keeps in-place ops as-is, so slice_scatter never
        # appears in the graph -- no need for the rewrite here.
        gm = rewrite_prims_in_gm(gm)
        # Drop dead clones (nn.Dropout in eval, etc.) before translation.
        # These are real per-call memcpys on accelerator devices; without
        # this pass timm ViT replay is dominated by ~37 spurious 19MB
        # tensor copies (DESIGN: docstring of eliminate_dead_clones).
        gm = eliminate_dead_clones(gm)
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
            backend=aot_autograd(
                fw_compiler=grab_compiler,
                # Inference path: skip AOT functionalize entirely. AOT
                # still decomposes Dynamo's high-level Python ops
                # (__setitem__, etc.) to core aten and adds sym_size
                # for dynamic shape, but leaves user-written in-place
                # mutations (aten.copy_, aten.add_, aten.index_put_,
                # aten.scatter_, ...) as in-place ops -- avoiding the
                # entire class of "functional copy + slice_scatter +
                # writeback" patterns that would force ~MB-scale
                # tensor reallocation per replay. Inductor handles
                # the same problem via auto_functionalized_v2 pattern
                # matching; we sidestep it by not creating the patterns
                # in the first place. See DESIGN.md §17.6.9.
                disable_functionalization=True,
            ),
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
    pre_binds = _promote_scalar_pre_binds_to_device(pre_binds, example_args)
    for arg_idx, value in pre_binds:
        trace.v2_pre_bind(arg_idx, value)
    flat_recipe = _compile_flat_recipe(runtime_specs)
    # Persistent buffer reused across every replay -- zero per-call
    # list allocation. flat_recipe writes into it in-place; we hand
    # it to v2_replay which only reads. The slots get overwritten on
    # every call, so the previous call's tensor references release
    # exactly when the new call starts (matches eager lifetime).
    flat_buf: list = [None] * flat_recipe._buf_len

    def direct_replay(*user_args):
        flat_recipe(user_args, flat_buf)
        result = trace.v2_replay(flat_buf)
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
        # disable_functionalization=True (passed to aot_function /
        # aot_module below) keeps in-place ops as-is, so the
        # slice_scatter rewrite isn't needed here.
        gm = rewrite_prims_in_gm(gm)
        gm = eliminate_dead_clones(gm)
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

    # Same disable_functionalization rationale as _capture_positional
    # (see DESIGN.md §17.6.9). Inference-only; the backward path
    # (_capture_with_backward) keeps functionalize on because
    # autograd's partition_fn requires a pure-functional graph.
    if target is not None:
        aot_compiled = aot_module(
            target, fw_compiler=grab_compiler, dynamic=True,
            disable_functionalization=True,
        )
    else:
        aot_compiled = aot_function(
            fn, fw_compiler=grab_compiler, dynamic=True,
            disable_functionalization=True,
        )

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

    Pytree-based: we hand user_visible_out to torch.utils._pytree to
    flatten into leaves + a treespec. This handles all built-in
    containers (tuple, list, dict, OrderedDict), namedtuples,
    dataclasses (when registered), and any custom type the user
    registered via pytree.register_pytree_node -- without us writing
    per-container code. The treespec is reapplied at every call to
    rebuild the same nested structure.

    Tensor leaves match against trace_out by `id()` — AOT's runtime
    layer aliases (does not clone) when packaging outputs, so a
    Tensor in user_visible_out is the *same* Python object as the
    corresponding entry in trace_out. Non-Tensor leaves (None,
    scalars, strings) are snapshotted by value into the plan."""
    import torch.utils._pytree as pytree

    id_to_idx = {
        id(v): i for i, v in enumerate(trace_out) if isinstance(v, torch.Tensor)
    }

    def _fallback_shaper(reason: str):
        # Lossy fallback: keep all trace outputs as a flat tuple. Loses
        # the user-visible structure but keeps correctness; surface the
        # reason so the user can decide whether to wrap in wrapper=True
        # or register their custom type with pytree.
        print(f"# v2.capture: output_shaper fallback ({reason})")
        n = len(trace_out)
        def fallback(result):
            return result[0] if len(result) == 1 else tuple(result[:n])
        return fallback

    try:
        leaves, treespec = pytree.tree_flatten(user_visible_out)
    except Exception as e:
        return _fallback_shaper(f"pytree.tree_flatten failed: {e}")

    # Per-leaf plan: ("T", idx) for tensors mapped to a trace_out slot,
    # ("L", value) for snapshotted literals (None, scalars, strings, ...).
    leaf_plans: list = []
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            idx = id_to_idx.get(id(leaf))
            if idx is None:
                return _fallback_shaper(
                    f"unmatched Tensor leaf (shape={tuple(leaf.shape)}); "
                    "AOT may have cloned/aliased an output, which "
                    "v2.capture's id()-based matching doesn't handle")
            leaf_plans.append(("T", idx))
        else:
            leaf_plans.append(("L", leaf))

    def shaper(result):
        materialized = [
            result[p[1]] if p[0] == "T" else p[1]
            for p in leaf_plans
        ]
        return pytree.tree_unflatten(materialized, treespec)
    return shaper


def _capture_with_backward(fn, example_args):
    """allow_grad=True path. Capture fw + bw graphs in one example call
    and return an autograd.Function-wrapped callable so the captured
    pair runs as a normal differentiable PyTorch op.

    Supports two ways of supplying parameters for the gradient graph:
      1. As positional `example_args` with requires_grad=True (e.g.
         passing model.parameters() positionally and using
         torch.func.functional_call inside `fn`).
      2. As Module attributes referenced via fn's closure (the natural
         nn.Module form, no functional_call boilerplate required).
         Dynamo lifts these Parameters as graph inputs; we detect them
         after tracing (isinstance check on observed_args) and surface
         them as positional leaf args of the wrapped autograd.Function
         so backward grads route to `param.grad` through the standard
         autograd accumulator path. From the user's perspective, the
         returned callable still takes only the original user_args:
         `captured(*user_args)`. Params are stitched in internally.

    Either way, opt.step()'s in-place updates to the Parameters are
    automatically reflected at replay time -- the trace stores the
    Parameter tensor object itself (same TensorImpl), not a snapshot.
    Replacing a Parameter (model.fc = new_layer) is NOT reflected; treat
    the trace as bound to the Parameter objects present at capture time.
    """
    captured: list = []

    def grab_compiler(gm, sample_inputs):
        # backward path: autograd's partition_fn needs a pure-
        # functional graph to split fw/bw, so we CAN'T pass
        # disable_functionalization here. slice_scatter shows up
        # in this graph -- rewrite_slice_scatter_to_inplace
        # de-functionalises it back to in-place form before we
        # translate.
        gm = rewrite_prims_in_gm(gm)
        gm = rewrite_slice_scatter_to_inplace(gm)
        gm = eliminate_dead_clones(gm)
        trace = translate_graph(gm)
        # Snapshot AOTAutograd's ViewAndMutationMeta off the active
        # TracingContext. AOT stashes fw_metadata on the context just
        # before calling our compiler (see torch._functorch._aot_autograd
        # .graph_compile.py around the `compiled_fw_func = compiler(...)`
        # call) -- this is the same metadata RuntimeWrapper consumes to
        # do mutation write-back and output-aliasing fixup. Capturing
        # it here lets v2 reproduce that bookkeeping at replay time
        # WITHOUT heuristics. Only the FW compile pass populates
        # mutated_inp_runtime_indices, so the bw call's snapshot is
        # discarded.
        from torch._guards import TracingContext
        tc = TracingContext.try_get()
        aot_fw_metadata = getattr(tc, "fw_metadata", None) if tc else None
        state = {
            "trace": trace,
            "gm": gm,
            "observed_args": None,
            "aot_fw_metadata": aot_fw_metadata,
        }
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

    # Post-compile requires_grad check: covers both user-input-grad and
    # closure-captured Parameter cases. The previous pre-check (requiring
    # an example_arg with requires_grad=True) was too strict for the
    # common nn.Module training pattern where x/y carry no grad and only
    # the closure-lifted Parameters do.
    def _has_grad_path(x):
        if isinstance(x, torch.Tensor):
            return x.requires_grad
        if isinstance(x, (tuple, list)):
            return any(_has_grad_path(o) for o in x)
        return False
    if not _has_grad_path(out):
        raise RuntimeError(
            "v2.capture(allow_grad=True): example call's output has "
            "requires_grad=False; AOTAutograd will not produce a "
            "backward graph. Either pass an example_arg with "
            "requires_grad=True, or ensure fn references an nn.Module "
            "with at least one Parameter (requires_grad=True).")

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
    fw_pre_binds = _promote_scalar_pre_binds_to_device(fw_pre_binds, example_args)
    for arg_idx, value in fw_pre_binds:
        fw_trace_obj.v2_pre_bind(arg_idx, value)

    # Extract nn.Parameter pre-binds for autograd-leaf routing. They
    # REMAIN pre-bound in the trace (v2_pre_bind above) so the trace's
    # captured_tensors_ slot points to the actual Parameter object --
    # opt.step's in-place mutation propagates through TensorImpl identity
    # without any per-call work. Separately, we surface the same params
    # as positional leaf args of _CapturedFn.apply so autograd records
    # them as inputs to the Function; backward grads then route to
    # `param.grad` through PyTorch's standard accumulator path. This is
    # the v2 analogue of aot_eager's mechanism: AOT lifts the params,
    # the wrapping autograd.Function exposes them as leaves, autograd
    # routes grads back.
    fw_param_specs: List[Tuple[int, torch.nn.Parameter]] = [
        (ph_idx, value) for ph_idx, value in fw_pre_binds
        if isinstance(value, torch.nn.Parameter)
    ]
    captured_params: List[torch.nn.Parameter] = [p for _, p in fw_param_specs]
    # ph_idx -> slot in captured_params (== position in trailing apply args).
    ph_to_param_slot: Dict[int, int] = {
        ph_idx: slot for slot, (ph_idx, _) in enumerate(fw_param_specs)
    }

    fw_flat = _compile_flat_recipe(fw_specs)
    # See _capture_positional for the rationale; persistent buffer
    # reused across every forward replay invocation under autograd.
    fw_flat_buf: list = [None] * fw_flat._buf_len

    n_user_inputs = len(example_args)
    n_params = len(captured_params)

    # Count tangent placeholders in bw_gm to learn how many of fw's
    # outputs are user-visible (the rest are mutated-input copies or
    # saved-for-backward intermediates).
    n_tangents = sum(
        1 for n in bw_state["gm"].graph.nodes
        if n.op == "placeholder" and n.name.startswith("tangents_")
    )
    if n_tangents == 0:
        raise RuntimeError(
            "v2.capture(allow_grad=True): bw graph has no 'tangents_*' "
            "placeholders; cannot tell user outputs from saved tensors.")

    # AOT's fw output ordering is documented as
    #     [mutated_input_copies, user_outputs, saved_for_backward]
    # where the mutated_input_copies block has length
    # `num_mutated_inp_runtime_indices` and the i-th entry of that block
    # is the new value of the input at position
    # `mutated_inp_runtime_indices[i]`. This metadata is the same one
    # RuntimeWrapper consumes for its epilogue write-back (see
    # torch._functorch._aot_autograd.runtime_wrappers
    # ._apply_input_mutations). v2 bypasses RuntimeWrapper, so we must
    # reproduce the write-back manually -- otherwise BN's running_mean
    # / running_var / num_batches_tracked stay at their initial values
    # across replays (visible after bn.eval() or model checkpointing).
    #
    # We snapshotted ViewAndMutationMeta off TracingContext during the
    # FW compile callback (grab_compiler). The bw compile snapshot is
    # discarded because backward doesn't carry input-mutation info.
    aot_fw_meta = fw_state.get("aot_fw_metadata")
    if aot_fw_meta is None:
        # Defensive fallback: previously this path tried heuristics
        # over the FX graph (getitem-arg-N + signature matching) which
        # mis-resolves on deep BN stacks. With TracingContext-based
        # metadata available in the supported PyTorch versions, this
        # branch should be unreachable; warn loudly so a regression
        # surfaces as a slow/silent buffer drift rather than getting
        # missed.
        print("# v2.capture: WARNING -- no AOT fw_metadata on "
              "TracingContext; mutated-buffer write-back is disabled "
              "for this trace. BN-style training will fail to update "
              "running stats across replays.")
        num_mutated_inputs = 0
        mutated_inp_indices: List[int] = []
    else:
        num_mutated_inputs = aot_fw_meta.num_mutated_inp_runtime_indices
        mutated_inp_indices = list(aot_fw_meta.mutated_inp_runtime_indices)

    # User outputs live immediately after the mutated_input_copies in
    # fw_outputs.
    user_out_start = num_mutated_inputs

    # Note: aot_fw_meta.num_outputs counts ALL user-output positions
    # including aliases, but only the differentiable ones get tangents
    # in bw_gm. We deliberately don't assert num_outputs == n_tangents
    # (resnet etc. produce alias outputs that count in num_outputs but
    # carry no tangent). v2's user-visible output slice still uses
    # n_tangents below, matching the previous behavior.

    # Build the write-back map: fw_output[i] for i in
    # [0..num_mutated_inputs) -> placeholder ph_idx (which is the same
    # index in the fw input list).
    mutated_output_to_ph_idx: Dict[int, int] = {
        i: mutated_inp_indices[i] for i in range(num_mutated_inputs)
    }

    # Build ph_idx -> actual buffer Tensor map. pre_binds stores the
    # real input tensors keyed by ph_idx; for mutated_input positions
    # (typically buffers like running_mean), this gives us the source-
    # of-truth tensor to copy_() into.
    ph_to_writeback_target: Dict[int, torch.Tensor] = {}
    pre_bind_by_phidx = {ph_idx: v for ph_idx, v in fw_pre_binds}
    for out_idx, ph_idx in mutated_output_to_ph_idx.items():
        v = pre_bind_by_phidx.get(ph_idx)
        if isinstance(v, torch.Tensor):
            ph_to_writeback_target[ph_idx] = v

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
        def forward(ctx, *all_args):
            # all_args layout: (*user_args, *params)
            #   - user_args feed into fw_flat -> fw_flat_buf -> v2_replay
            #     (params are NOT routed through fw_flat -- they're already
            #     pre-bound to the trace's captured_tensors_ slots).
            #   - params are still passed here so autograd records them
            #     as leaf inputs to this Function; backward returns grads
            #     for them, autograd routes to param.grad accumulators.
            user_args = all_args[:n_user_inputs]
            fw_flat(user_args, fw_flat_buf)
            fw_outputs = fw_trace.v2_replay(fw_flat_buf)
            # AOT-RuntimeWrapper replacement: write mutated-input
            # outputs back to their source buffers. Functionalize
            # turned `buffer.copy_(new)` into `new = ...; output new`.
            # The C++ trace's captured_tensors_ slot still points to
            # the original buffer (via pre_bind), and the new value is
            # at fw_outputs[i] for each mutated-input slot. .copy_()
            # is in-place on the underlying Storage, so subsequent
            # replays see the updated buffer value through the same
            # TensorImpl identity.
            for out_idx, ph_idx in mutated_output_to_ph_idx.items():
                target = ph_to_writeback_target.get(ph_idx)
                new_val = fw_outputs[out_idx]
                if target is not None and isinstance(new_val, torch.Tensor):
                    target.detach().copy_(new_val.detach())
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
            user_outputs = fw_outputs[user_out_start:user_out_start + n_tangents]
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
            # bw outputs are aligned with fw INPUT placeholders. Route to
            # the corresponding apply position so autograd can hand grads
            # to the right leaf (user_input grads -> user .grad,
            # param grads -> param.grad).
            user_grads = [None] * n_user_inputs
            for fw_ph_idx, user_idx in fw_ph_to_user_input.items():
                if fw_ph_idx < len(bw_outputs):
                    user_grads[user_idx] = bw_outputs[fw_ph_idx]
            param_grads = [None] * n_params
            for ph_idx, slot in ph_to_param_slot.items():
                if ph_idx < len(bw_outputs):
                    param_grads[slot] = bw_outputs[ph_idx]
            return (*user_grads, *param_grads)

    def call(*user_args):
        return _CapturedFn.apply(*user_args, *captured_params)

    call.fw_trace = fw_trace                  # type: ignore[attr-defined]
    call.bw_trace = bw_trace                  # type: ignore[attr-defined]
    call.fw_recipe_specs = fw_specs           # type: ignore[attr-defined]
    call.captured_params = captured_params    # type: ignore[attr-defined]
    return call


def _infer_target_device(example_args) -> torch.device | None:
    """Pick the device captured tensors should live on at replay time.
    First Tensor in example_args wins. Returns None if the user passed
    no Tensors (degenerate trace -- nothing to promote anyway)."""
    for a in example_args:
        if isinstance(a, torch.Tensor):
            return a.device
    return None


def _promote_scalar_pre_binds_to_device(
    pre_binds: List[Tuple[int, Any]],
    example_args,
) -> List[Tuple[int, Any]]:
    """Move 0-d CPU pre-bind tensors to the user input's device.

    Why: AOT functionalization lifts Python scalars (e.g. RMSNorm's
    `self.eps = 1e-6`, attention's `math.sqrt(head_dim)`) into 0-d
    Tensors as graph placeholders. `torch.tensor(1e-6)` defaults to
    float64 / CPU. Without promotion, every replay sees an
    `aten::add.Tensor(npu_t, cpu_scalar_t, 1)` and the NPU backend
    forces a synchronous H2D copy of the scalar before running the
    add -- ~500us per occurrence. LLaMA-class models hit this 30+
    times per replay (DESIGN.md §17.6.9 quantifies it as ~15ms
    device-side overhead vs eager).

    Promote at capture time so the captured tensor in
    `Trace::captured_tensors_` is already on the target device.
    Replay then sees `aten::add.Tensor(npu_t, npu_scalar_t, 1)` --
    pure on-device computation, no H2D sync.

    Only 0-d tensors are promoted intentionally:
      - They represent config constants (eps, scale, temperature, ...)
        that are conceptually device-agnostic.
      - Higher-rank CPU tensors might be the user's intentional CPU
        data being passed in deliberately; we don't second-guess.

    Caveat: after promotion, the captured 0-d tensor is a NEW
    TensorImpl independent from any source the user might still hold.
    If the user later does `module.layer.eps_buf.fill_(new)` on a
    registered CPU Buffer, that mutation won't propagate to our
    promoted NPU copy (DESIGN.md §17.6.9 走法 A). This is rare in
    practice -- the common pattern is `self.eps = 1e-6` (Python float
    which can't be mutated in-place anyway) or `register_buffer(...,
    torch.tensor(eps, device=target_device))` (already on device, no
    promotion needed). The mutation-reflection path C from §17.6.9
    is deferred until a real use case asks for it.
    """
    target = _infer_target_device(example_args)
    if target is None or target.type == "cpu":
        return pre_binds
    result: List[Tuple[int, Any]] = []
    for arg_idx, value in pre_binds:
        if (
            isinstance(value, torch.Tensor)
            and value.dim() == 0
            and value.device.type == "cpu"
        ):
            value = value.to(device=target)
        result.append((arg_idx, value))
    return result


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

    # Positional indices of Python scalar args (int / bool) the user
    # passed. Under Dynamo+dynamic=True these get lifted as SymInt
    # placeholders that participate in shape arith / slice bounds; we
    # need to route a fresh value at every replay, not freeze the
    # capture-time value. Consumed positionally (Dynamo preserves
    # user-arg relative order in the placeholder list) and matched via
    # value as a sanity check. Note: bool subclasses int in Python, so
    # this covers both. Floats are not lifted by Dynamo into the
    # captured graph in any way that varies the kernel call, so we
    # don't include them.
    user_scalar_queue: List[int] = [
        i for i, a in enumerate(example_args)
        if isinstance(a, int) and not isinstance(a, torch.Tensor)
    ]

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
                # SymInt derived from a user-input Tensor's shape.
                ua_idx, dim = symbol_source[key]
                runtime_specs.append(("S", ua_idx, dim))
            elif (user_scalar_queue
                  and observed_args[ph_idx] == example_args[user_scalar_queue[0]]):
                # SymInt that maps to the next user-passed Python int.
                # Value-equality with the queue head + positional order
                # gives a stable mapping under Dynamo's arg-lifting:
                # shape SymInts are lifted before scalar arg SymInts,
                # and within each group the original positional order
                # is preserved. The == check guards against the
                # (theoretical) reorder.
                user_idx = user_scalar_queue.pop(0)
                runtime_specs.append(("I", user_idx))
                # Intentionally NOT added to ph_to_user_input: that map
                # routes bw outputs back to user-input GRADs, and ints
                # don't carry grads.
            else:
                # No matching user scalar -- it's a Dynamo-specialised
                # closure constant or similar. Pre-bind it as before.
                pre_binds.append((ph_idx, observed_args[ph_idx]))
        elif isinstance(val, int):
            # Concrete int placeholder (non-dynamic compile, or AOT
            # specialised the int). Same routing as the SymInt branch:
            # map to a user scalar if value matches, else pre-bind.
            if (user_scalar_queue
                    and val == example_args[user_scalar_queue[0]]):
                user_idx = user_scalar_queue.pop(0)
                runtime_specs.append(("I", user_idx))
            else:
                pre_binds.append((ph_idx, val))
        else:
            raise RuntimeError(
                f"v2.capture: unsupported placeholder val type "
                f"{type(val).__name__} for node {n.name}")
    return runtime_specs, pre_binds, ph_to_user_input


def _compile_flat_recipe(specs: List[RecipeSpec]) -> Callable[..., list]:
    """Generate and exec() a single function that materialises all
    NON-pre-bound placeholder values into a caller-provided buffer.

    Output signature: `def _flat(args, buf): buf[0] = ...; ... ; return buf`
    where each line is either `args[i]` (T spec) or `args[i].size(d)`
    (S spec). Pre-bound placeholders (param tensors, Dynamo-specialised
    constants) are NOT touched — they live in the trace's persistent
    buffers, set once via v2_pre_bind.

    Why exec/eval: collapses N attribute / index ops into a single
    function invocation. On accelerators where kernel time is
    amortised across calls and host-side Python overhead is the
    bottleneck, this can save 1-3us per replay (~N * 300-500ns).

    Why writes-into-buf instead of returning a fresh list: in the hot
    replay path we want zero list allocations per call. The caller
    holds the buf across replays, hands the same object in each time;
    we just overwrite the slots. Tensor slots churn through the user
    Tensor objects (one ref each), int slots get the result of
    .size() (cached for small ints by CPython). No new list, no new
    tuple — only the .size() ints when sizes don't fit the small-int
    cache (<= 256).
    """
    expr_parts: list[str] = []
    for spec in specs:
        if spec[0] == "T":
            expr_parts.append(f"args[{spec[1]}]")
        elif spec[0] == "S":
            expr_parts.append(f"args[{spec[1]}].size({spec[2]})")
        elif spec[0] == "I":
            # Same expression as T but the underlying slot in the trace
            # is an int slot, not a tensor slot -- routing is set up
            # in translator._translate_placeholder based on the FX
            # node's meta['val'] type. flat_recipe just writes the
            # value; the trace knows where it goes.
            expr_parts.append(f"args[{spec[1]}]")
        else:
            raise AssertionError(
                f"flat_recipe only handles T/S/I specs after pre-bind extraction; "
                f"got {spec!r}")

    if not expr_parts:
        # Degenerate case: nothing to materialise (no user-input slots).
        # Still return the buffer for caller uniformity.
        src = (
            "def _flat_recipe(args, buf):\n"
            "    return buf\n"
        )
    else:
        body_lines = [
            f"    buf[{i}] = {expr}" for i, expr in enumerate(expr_parts)
        ]
        src = (
            "def _flat_recipe(args, buf):\n"
            + "\n".join(body_lines)
            + "\n    return buf\n"
        )
    ns: dict = {}
    exec(src, ns)
    fn = ns["_flat_recipe"]
    fn._source = src                   # type: ignore[attr-defined]
    fn._buf_len = len(expr_parts)      # type: ignore[attr-defined]
    return fn
