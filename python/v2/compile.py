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


def capture(fn, *example_args):
    """Run `fn(*example_args)` once through torch.compile to extract a
    Trace, then return a callable that replays the trace directly with
    fresh args — no Dynamo/AOT machinery on the call path.

    Caller is responsible for: future args matching example_args in
    rank/dtype/device. Sym dimensions may vary; concrete int dimensions
    that Dynamo guarded on must stay the same.
    """
    captured: list = []

    def grab_compiler(gm, sample_inputs):
        trace = translate_graph(gm)
        state = {"trace": trace, "gm": gm, "observed_args": None}
        captured.append(state)
        def wrapping_cb(*args):
            # Record what AOTAutograd-prelude resolves each placeholder
            # to on the example_args call. Used as the literal-fallback
            # value for SymInts that don't appear in any input Tensor's
            # shape (e.g., Dynamo-specialized closure-captured ints like
            # a module's hidden-dim constant).
            if state["observed_args"] is None:
                state["observed_args"] = list(args)
            result = trace.v2_replay(list(args))
            return result[0] if len(result) == 1 else tuple(result)
        return wrapping_cb

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


def _build_recipe_specs(
    gm: fx.GraphModule, example_args, observed_args
) -> List[RecipeSpec]:
    """For each gm placeholder, return a tagged spec describing how to
    fetch its runtime value. Specs feed _compile_flat_recipe, which
    generates a single Python function that materialises all values in
    one call (eliminates per-recipe lambda overhead).

    Resolution priority:
      (1) SymInts that appear in some input Tensor's FakeTensor.shape
          -> ("S", user_arg_idx, dim)
      (2) SymInts we cannot derive from any shape (e.g. Dynamo-
          specialised closure constants) -> ("L", observed_value)
      (3) Tensors -> ("T", user_arg_idx)
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    tensor_placeholders = [
        (n, n.meta.get("val")) for n in placeholders
        if isinstance(n.meta.get("val"), torch.Tensor)
    ]

    if len(tensor_placeholders) != len(example_args):
        raise RuntimeError(
            f"v2.capture: example_args has {len(example_args)} Tensors but "
            f"the AOT graph has {len(tensor_placeholders)} Tensor placeholders. "
            f"Mixed-arg-type signatures (Tensor + Python scalar) are not "
            f"supported yet."
        )

    # Symbol -> (user_arg_idx, dim) from inspecting Tensor placeholder shapes.
    symbol_source: dict[str, Tuple[int, int]] = {}
    for user_idx, (_, fake_t) in enumerate(tensor_placeholders):
        for dim, size in enumerate(fake_t.shape):
            if isinstance(size, torch.SymInt):
                key = str(size.node.expr)
                symbol_source.setdefault(key, (user_idx, dim))

    specs: List[RecipeSpec] = []
    tensor_user_iter = 0
    for ph_idx, n in enumerate(placeholders):
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor):
            specs.append(("T", tensor_user_iter))
            tensor_user_iter += 1
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
