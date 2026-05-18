"""Translate an AOT FX GraphModule into a v2 Trace (DESIGN.md §17.6.3).

The translator processes each FX node by `node.op`:

  placeholder            -> append to captured_tensors / captured_ints
  call_function          -> emit one Step (TENSOR_OP or PY_CALL by target type)
  output                 -> set trace.outputs
  call_method / call_module / get_attr / HOP -> fail-fast (DESIGN §17.6.8)

list/tuple structures in node.args become kList refs; literals become
kLiteral; Node references become kPrevStepOutput (one-level slot=0
because pytree flattens nested outputs at every AOT boundary).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import fx

from .trace import RefKind, Step, StepInputRef, StepKind, Trace


def translate_graph(gm: fx.GraphModule) -> Trace:
    trace = Trace()
    node_to_ref: Dict[fx.Node, StepInputRef] = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            _translate_placeholder(node, trace, node_to_ref)
        elif node.op == "call_function":
            _translate_call_function(node, trace, node_to_ref)
        elif node.op == "output":
            _translate_output(node, trace, node_to_ref)
        elif node.op in ("call_method", "call_module", "get_attr"):
            raise NotImplementedError(
                f"v2 does not support FX node.op='{node.op}' "
                f"(node={node.name}, target={node.target}). "
                f"AOT graphs rarely emit these; if you hit this, file a bug "
                f"or fall back to torch.compile(backend='inductor')."
            )
        else:
            raise AssertionError(f"unknown FX node.op: {node.op}")

    return trace


def _translate_placeholder(node, trace: Trace, node_to_ref):
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        idx = trace.n_captured_tensors
        trace.n_captured_tensors += 1
        ref = StepInputRef(kind=RefKind.CAPTURED_TENSOR, idx=idx)
        trace.placeholder_routing.append((RefKind.CAPTURED_TENSOR, idx))
    elif isinstance(val, (torch.SymInt, int)):
        idx = trace.n_captured_ints
        trace.n_captured_ints += 1
        ref = StepInputRef(kind=RefKind.CAPTURED_INT, idx=idx)
        trace.placeholder_routing.append((RefKind.CAPTURED_INT, idx))
    else:
        raise NotImplementedError(
            f"v2 placeholder val type not supported yet: {type(val).__name__} "
            f"(node {node.name}). Handles Tensor + SymInt only; "
            f"SymFloat / SymBool are listed in DESIGN §17.6.8 defensive checks."
        )
    node_to_ref[node] = ref


def _translate_call_function(node, trace: Trace, node_to_ref):
    target = node.target

    if isinstance(target, torch._ops.HigherOrderOperator):
        raise NotImplementedError(
            f"v2 does not support HigherOrderOperator {target} (e.g. cond / "
            f"while_loop / scan). Use torch.compile(backend='inductor') for "
            f"control-flow workloads. See DESIGN §17.6.8."
        )

    inputs = [_node_arg_to_ref(a, node_to_ref) for a in node.args]
    kwargs = {k: _node_arg_to_ref(v, node_to_ref) for k, v in node.kwargs.items()}
    step_idx = len(trace.steps)

    if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        trace.steps.append(Step(
            kind=StepKind.TENSOR_OP,
            inputs=inputs,
            kwargs=kwargs,
            op=target,
            name=str(target),
        ))
    elif callable(target):
        # operator.* / torch.sym_* / whitelisted torch APIs — all the same path
        trace.steps.append(Step(
            kind=StepKind.PY_CALL,
            inputs=inputs,
            kwargs=kwargs,
            fn=target,
            name=f"{getattr(target, '__module__', '?')}."
                 f"{getattr(target, '__name__', repr(target))}",
        ))
    else:
        raise NotImplementedError(
            f"v2 cannot translate call_function target of type {type(target)}: "
            f"{target!r}"
        )

    node_to_ref[node] = StepInputRef(
        kind=RefKind.PREV_STEP_OUTPUT, step=step_idx, slot=0)


def _translate_output(node, trace: Trace, node_to_ref):
    # node.args is conventionally a 1-element tuple containing the
    # return tuple/list, e.g. ((view,),) or ((add, idx),).
    assert len(node.args) == 1, f"unexpected output arity: {node.args}"
    output_value = node.args[0]
    if isinstance(output_value, (tuple, list)):
        trace.outputs = [_node_arg_to_ref(v, node_to_ref) for v in output_value]
    else:
        trace.outputs = [_node_arg_to_ref(output_value, node_to_ref)]


def _node_arg_to_ref(value: Any, node_to_ref) -> StepInputRef:
    """Convert a node.args element into a StepInputRef.

    Cases:
      - fx.Node      -> look up the ref recorded when that node was visited
      - list / tuple -> kList wrapping recursive sub-refs
      - anything else (int, float, bool, None, str, slice, dtype, ...) -> kLiteral
    """
    if isinstance(value, fx.Node):
        return node_to_ref[value]
    if isinstance(value, (list, tuple)):
        return StepInputRef(
            kind=RefKind.LIST,
            list_elements=[_node_arg_to_ref(v, node_to_ref) for v in value],
        )
    return StepInputRef(kind=RefKind.LITERAL, literal=value)
