"""Public v2 API — @tdcv2.compile() decorator and v2.capture() direct entry.

Two paths are provided:

  @tdcv2.compile(dynamic=True)
    The transparent path. Each call goes through torch.compile's full
    runtime: Dynamo guard check, prelude bytecode, AOTAutograd runtime
    wrapper, then our replay. Overhead is dominated by the torch.compile
    pipeline (~7-10us/call on small workloads) regardless of how the
    trace itself runs.

  v2.capture(fn, *example_args)
    The direct-replay path. Capture the trace once via torch.compile,
    then return a callable that bypasses Dynamo at call time. Per-call
    overhead is just IValue conversion + C++ replay. Trade-off: the
    user must guarantee future calls have the same arg structure as
    example_args (same rank, same dtype, ...).
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd

from .translator import translate_graph


# ---------------------------------------------------------------------------
# Path 1: torch.compile-integrated entry
# ---------------------------------------------------------------------------
def fw_compiler(gm, _sample_inputs):
    """AOTAutograd fw_compiler: AOT GraphModule -> C++ Trace -> callable."""
    trace = translate_graph(gm)

    def replay_callable(*args):
        result = trace.v2_replay(list(args))
        return result[0] if len(result) == 1 else tuple(result)

    return replay_callable


def compile(fn=None, *, dynamic: bool = True):
    """Decorator. Equivalent to:

        torch.compile(fn,
                      backend=aot_autograd(fw_compiler=tdcv2.fw_compiler),
                      dynamic=dynamic)
    """
    def wrap(f):
        return torch.compile(
            f,
            backend=aot_autograd(fw_compiler=fw_compiler),
            dynamic=dynamic,
        )

    return wrap if fn is None else wrap(fn)


# ---------------------------------------------------------------------------
# Path 2: v2.capture() direct-replay entry
# ---------------------------------------------------------------------------
Recipe = Callable[[Tuple[Any, ...]], Any]


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
    recipes = _build_recipes(state["gm"], example_args, state["observed_args"])

    def direct_replay(*user_args):
        flat = [recipe(user_args) for recipe in recipes]
        result = trace.v2_replay(flat)
        return result[0] if len(result) == 1 else tuple(result)

    # Expose internals for introspection / debug.
    direct_replay.trace = trace            # type: ignore[attr-defined]
    direct_replay.recipes = recipes        # type: ignore[attr-defined]
    return direct_replay


def _build_recipes(gm: fx.GraphModule, example_args, observed_args) -> List[Recipe]:
    """For each gm placeholder, return a recipe(user_args) -> value.

    Resolution priority:
      (1) For SymInts that appear in some input Tensor's FakeTensor.shape,
          use a shape-extraction recipe (e.g., args[i].size(d)). These
          change with user input.
      (2) For SymInt placeholders that come immediately before a Tensor
          placeholder (positional grouping), assume they describe that
          Tensor's leading dims. Recipe: args[k].size(d).
      (3) For SymInt placeholders we still can't derive (e.g., Dynamo-
          specialized closure constants like a module's N_HEADS=8), fall
          back to a literal recipe with the value observed during the
          example call. The caller has implicitly agreed by passing
          example_args that this constant won't vary across replays.
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

    # Symbol -> (user_arg_idx, dim) from inspecting Tensor placeholder
    # shapes. SymInts not present in any Tensor shape (e.g., closure-
    # captured constants that Dynamo specialized to SymInt) fall back
    # to observed-args below.
    symbol_source: dict[str, Tuple[int, int]] = {}
    for user_idx, (_, fake_t) in enumerate(tensor_placeholders):
        for dim, size in enumerate(fake_t.shape):
            if isinstance(size, torch.SymInt):
                key = str(size.node.expr)
                symbol_source.setdefault(key, (user_idx, dim))

    # Emit recipes in placeholder order.
    recipes: List[Recipe] = []
    tensor_user_iter = 0
    for ph_idx, n in enumerate(placeholders):
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor):
            i = tensor_user_iter
            recipes.append(lambda args, i=i: args[i])
            tensor_user_iter += 1
        elif isinstance(val, torch.SymInt):
            key = str(val.node.expr)
            if key in symbol_source:
                ua_idx, dim = symbol_source[key]
                recipes.append(lambda args, i=ua_idx, d=dim: args[i].size(d))
            else:
                # Fallback: use the value Dynamo's prelude resolved for
                # this placeholder during the example call. Treated as
                # a specialized literal.
                v = observed_args[ph_idx]
                recipes.append(lambda args, v=v: v)
        elif isinstance(val, int):
            v = val
            recipes.append(lambda args, v=v: v)
        else:
            raise RuntimeError(
                f"v2.capture: unsupported placeholder val type "
                f"{type(val).__name__} for node {n.name}")
    return recipes
