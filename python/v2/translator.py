"""Translate an AOT FX GraphModule into a C++ Trace (DESIGN.md §17.6.9).

The translator walks each FX node in `gm.graph.nodes` and emits a Step
into a C++ Trace via the v2_* methods exposed in csrc/bindings.cpp.
After the walk, Trace.v2_replay(args) runs the unified C++ replay engine
(csrc/trace_v2.cpp) — no Python loop over steps at run time.
"""
from __future__ import annotations

import operator
from typing import Any, Dict

import torch
from torch import fx

from torch_dispatch_capture import _C  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Python target -> BuiltinKind for kPyCall steps
# ---------------------------------------------------------------------------
_OP_TO_BUILTIN = {
    operator.floordiv: _C.BuiltinKind.FLOORDIV,
    operator.truediv:  _C.BuiltinKind.TRUEDIV,
    operator.add:      _C.BuiltinKind.ADD,
    operator.sub:      _C.BuiltinKind.SUB,
    operator.mul:      _C.BuiltinKind.MUL,
    operator.mod:      _C.BuiltinKind.MOD,
    operator.neg:      _C.BuiltinKind.NEG,
    operator.getitem:  _C.BuiltinKind.GETITEM,
    operator.eq:       _C.BuiltinKind.EQ,
    operator.lt:       _C.BuiltinKind.LT,
    operator.le:       _C.BuiltinKind.LE,
    operator.gt:       _C.BuiltinKind.GT,
    operator.ge:       _C.BuiltinKind.GE,
    operator.ne:       _C.BuiltinKind.NE,
    torch.sym_max:     _C.BuiltinKind.SYM_MAX,
    torch.sym_min:     _C.BuiltinKind.SYM_MIN,
    torch.sym_int:     _C.BuiltinKind.SYM_INT,
    torch.sym_float:   _C.BuiltinKind.SYM_FLOAT,
}


def translate_graph(gm: fx.GraphModule) -> _C.Trace:
    trace = _C.Trace()
    node_to_ref: Dict[fx.Node, Any] = {}    # fx.Node -> _C.StepInputRef

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


def _translate_placeholder(node, trace, node_to_ref):
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        idx = trace.v2_add_placeholder_tensor()
        node_to_ref[node] = _C.v2_ref_captured_tensor(idx)
    elif isinstance(val, (torch.SymInt, int)):
        idx = trace.v2_add_placeholder_int()
        node_to_ref[node] = _C.v2_ref_captured_int(idx)
    else:
        raise NotImplementedError(
            f"v2 placeholder val type not supported yet: {type(val).__name__} "
            f"(node {node.name}). Handles Tensor + SymInt only."
        )


def _translate_call_function(node, trace, node_to_ref):
    target = node.target

    if isinstance(target, torch._ops.HigherOrderOperator):
        raise NotImplementedError(
            f"v2 does not support HigherOrderOperator {target}. Use "
            f"torch.compile(backend='inductor') for control-flow workloads."
        )

    inputs = [_node_arg_to_ref(a, node_to_ref) for a in node.args]

    if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        # Merge kwargs into positional via schema (DESIGN §17.6.9). C++
        # replay uses positional-only callBoxed, so we expand kwargs here.
        positional_inputs = _merge_kwargs_into_positional(target, node.args, node.kwargs, node_to_ref)
        op_name = _qualified_op_name(target)
        n_out = len(target._schema.returns)
        step_idx = trace.v2_add_tensor_op_step(op_name, positional_inputs, n_out)
    elif callable(target):
        builtin = _OP_TO_BUILTIN.get(target)
        kwargs_refs = [_node_arg_to_ref(v, node_to_ref) for v in node.kwargs.values()]
        if builtin is not None:
            # Only positional inputs make sense for C++ builtin dispatch;
            # the small set of builtins we map (operator.* / torch.sym_*)
            # don't take kwargs in practice.
            if kwargs_refs:
                raise NotImplementedError(
                    f"v2 builtin {target} unexpectedly has kwargs: {node.kwargs}"
                )
            step_idx = trace.v2_add_pycall_step(
                kind=builtin,
                inputs=inputs,
                name=str(target),
            )
        else:
            # Fallback: opaque py::object call. kwargs aren't piped through
            # for now (rare in AOT graphs).
            if kwargs_refs:
                raise NotImplementedError(
                    f"v2 pyfallback for {target} with kwargs not yet supported"
                )
            step_idx = trace.v2_add_pycall_step(
                kind=_C.BuiltinKind.PY_FALLBACK,
                inputs=inputs,
                py_fn=target,
                name=str(target),
            )
    else:
        raise NotImplementedError(
            f"v2 cannot translate call_function target of type {type(target)}: "
            f"{target!r}"
        )

    node_to_ref[node] = _C.v2_ref_prev_step(step_idx, 0)


def _translate_output(node, trace, node_to_ref):
    assert len(node.args) == 1, f"unexpected output arity: {node.args}"
    output_value = node.args[0]
    if isinstance(output_value, (tuple, list)):
        out_refs = [_node_arg_to_ref(v, node_to_ref) for v in output_value]
    else:
        out_refs = [_node_arg_to_ref(output_value, node_to_ref)]
    trace.v2_set_outputs(out_refs)


def _node_arg_to_ref(value, node_to_ref):
    if isinstance(value, fx.Node):
        return node_to_ref[value]
    if isinstance(value, (list, tuple)):
        return _C.v2_ref_list([_node_arg_to_ref(v, node_to_ref) for v in value])
    return _C.v2_ref_literal(value)


def _qualified_op_name(op) -> str:
    """Build 'aten::view.<overload>' style name expected by
    v2_add_tensor_op_step. Schema's overload_name may be empty string —
    that's the default overload and C++ findOp wants the empty string."""
    schema = op._schema
    return f"{schema.name}.{schema.overload_name}"


def _merge_kwargs_into_positional(op, args, kwargs, node_to_ref):
    """For a kTensorOp step, lift kwargs into the positional slot order
    dictated by the op's schema (so C++ replay can use callBoxed with a
    flat positional stack). Missing args get filled with their schema
    default value as a literal ref."""
    schema_args = op._schema.arguments
    n_positional = len(args)
    refs = [_node_arg_to_ref(a, node_to_ref) for a in args]

    # Walk the remaining schema args in declaration order and pull from
    # node.kwargs by name; for absent ones, append the default value as a
    # literal. Stop once we've covered all kwargs.
    remaining_kwargs = dict(kwargs)
    for i in range(n_positional, len(schema_args)):
        sa = schema_args[i]
        name = sa.name
        if name in remaining_kwargs:
            refs.append(_node_arg_to_ref(remaining_kwargs.pop(name), node_to_ref))
        else:
            if not sa.has_default_value():
                break
            refs.append(_C.v2_ref_literal(sa.default_value))
    if remaining_kwargs:
        raise NotImplementedError(
            f"v2 schema kwarg merge: unconsumed kwargs {list(remaining_kwargs)} "
            f"for op {_qualified_op_name(op)}; schema args: "
            f"{[a.name for a in schema_args]}"
        )
    return refs
