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

from typing import Any, Callable, List, Tuple, Union

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


def capture(fn, *example_args, allow_grad: bool = False):
    """Run `fn(*example_args)` once through torch.compile to extract a
    Trace, then return a callable that replays the trace directly with
    fresh args — no Dynamo/AOT machinery on the call path.

    Caller is responsible for: future args matching example_args in
    rank/dtype/device. Sym dimensions may vary; concrete int dimensions
    that Dynamo guarded on must stay the same.

    allow_grad=True: also capture the backward graph. Requires at least
    one example_arg to have requires_grad=True. The returned callable
    is wrapped in torch.autograd.Function so callers can do the usual
    `loss = captured(*args); loss.backward()` pattern.
    """
    if allow_grad:
        return _capture_with_backward(fn, example_args)

    # ---- inference-only path (no backward graph) ----
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

    with torch.no_grad():
        compiled_fn = torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=grab_compiler),
            dynamic=True,
        )
        compiled_fn(*example_args)
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
    recipe_specs = _build_recipe_specs(state["gm"], example_args, state["observed_args"])
    flat_recipe = _compile_flat_recipe(recipe_specs)

    def direct_replay(*user_args):
        flat = flat_recipe(user_args)
        result = trace.v2_replay(flat)
        return result[0] if len(result) == 1 else tuple(result)

    # Expose internals for introspection / debug.
    direct_replay.trace = trace                # type: ignore[attr-defined]
    direct_replay.recipe_specs = recipe_specs  # type: ignore[attr-defined]
    direct_replay.flat_recipe = flat_recipe    # type: ignore[attr-defined]
    return direct_replay


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

    fw_specs = _build_recipe_specs(fw_state["gm"], example_args, fw_state["observed_args"])
    fw_flat = _compile_flat_recipe(fw_specs)

    # bw outputs at fw-placeholder positions are grads for fw inputs.
    # Only tensor placeholders carry user-visible grads.
    fw_ph_to_user_input: dict = {
        i: spec[1] for i, spec in enumerate(fw_specs) if spec[0] == "T"
    }
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
) -> List[RecipeSpec]:
    """For each gm placeholder, return a tagged spec describing how to
    fetch its runtime value.

    Per-placeholder resolution:

    Tensor placeholders:
      (a) id() matches one of example_args -> user input, recipe ("T", i).
      (b) Otherwise = module parameter / buffer lifted into the graph
          by Dynamo. Snapshot the observed Tensor as a literal — recipe
          ("L", tensor). Weights stay frozen across replays; re-capture
          after a parameter update. (Sufficient for inference; training
          would need to expose param grads explicitly.)

    SymInt placeholders:
      (a) Symbol appears in a USER-INPUT Tensor's FakeTensor.shape:
          recipe ("S", user_idx, dim). Reads from the live caller arg.
      (b) Otherwise: fall back to the observed value as a literal
          (closure constants, Dynamo-specialised ints).
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

    specs: List[RecipeSpec] = []
    for ph_idx, n in enumerate(placeholders):
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor):
            if is_user_input[ph_idx]:
                specs.append(("T", user_tensor_ids[id(observed_args[ph_idx])]))
            else:
                # Module parameter / buffer snapshot.
                specs.append(("L", observed_args[ph_idx]))
        elif isinstance(val, torch.SymInt):
            key = str(val.node.expr)
            if key in symbol_source:
                ua_idx, dim = symbol_source[key]
                specs.append(("S", ua_idx, dim))
            else:
                specs.append(("L", observed_args[ph_idx]))
        elif isinstance(val, int):
            specs.append(("L", val))
        else:
            raise RuntimeError(
                f"v2.capture: unsupported placeholder val type "
                f"{type(val).__name__} for node {n.name}")
    return specs


def _compile_flat_recipe(specs: List[RecipeSpec]) -> Callable[[Tuple[Any, ...]], list]:
    """Generate and exec() a single function that materialises all
    placeholder values in one Python call.

    Output shape: `def _flat(args): return [<expr_0>, <expr_1>, ...]`
    where each expression is either `args[i]`, `args[i].size(d)`, or a
    literal value (baked as a constant reference into a closure cell).

    Why exec/eval: this collapses N lambda calls (one per placeholder)
    into a single function invocation. On accelerators where kernel
    time is amortised across calls and host-side Python overhead is
    the bottleneck, this can save 1-3us per replay (≈ N * 300-500ns).
    """
    # Build the expressions. Literals can't always be inlined safely
    # (an int 8 is fine via repr(); arbitrary objects may not have a
    # parseable repr). Stash them in a local list referenced by index.
    literal_table: list = []
    expr_parts: list[str] = []
    for spec in specs:
        if spec[0] == "T":
            expr_parts.append(f"args[{spec[1]}]")
        elif spec[0] == "S":
            expr_parts.append(f"args[{spec[1]}].size({spec[2]})")
        elif spec[0] == "L":
            idx = len(literal_table)
            literal_table.append(spec[1])
            expr_parts.append(f"_L[{idx}]")
        else:
            raise AssertionError(f"unknown recipe spec tag: {spec[0]!r}")

    src = (
        "def _flat_recipe(args):\n"
        f"    return [{', '.join(expr_parts)}]\n"
    )
    ns: dict = {"_L": literal_table}
    exec(src, ns)
    fn = ns["_flat_recipe"]
    # Keep the generated source available for debugging.
    fn._source = src                   # type: ignore[attr-defined]
    fn._literal_table = literal_table  # type: ignore[attr-defined]
    return fn
