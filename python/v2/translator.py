"""Translate an AOT FX GraphModule into a C++ Trace (DESIGN.md §17.6.9).

The translator walks each FX node in `gm.graph.nodes` and emits a Step
into a C++ Trace via the v2_* methods exposed in csrc/bindings.cpp.
After the walk, Trace.v2_replay(args) runs the unified C++ replay engine
(csrc/trace_v2.cpp) — no Python loop over steps at run time.

For kTensorOp steps we also precompute an ArgCoercion tag per input
slot (whether to leave the IValue alone, wrap Scalar->0-d Tensor,
convert GenericList -> IntList, or convert GenericList -> TensorList).
This lets C++ replay skip schema introspection entirely on the hot
path; see csrc/trace_v2.cpp::apply_coercion.
"""
from __future__ import annotations

import operator
from typing import Any, Dict, List, Tuple

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

# Map operator/torch.sym targets to the IValue *kind* their output will
# have at replay time. Used by coercion prediction for downstream ops.
_BUILTIN_OUTPUT_KIND = {
    operator.floordiv: "int",
    operator.mod:      "int",
    operator.add:      "int",   # ints in graph; floats would have own overload
    operator.sub:      "int",
    operator.mul:      "int",
    operator.neg:      "int",
    operator.truediv:  "float",
    operator.eq:       "bool",
    operator.lt:       "bool",
    operator.le:       "bool",
    operator.gt:       "bool",
    operator.ge:       "bool",
    operator.ne:       "bool",
    operator.getitem:  "tensor",    # multi-output op tuple unpack — defaults to Tensor
    torch.sym_max:     "int",
    torch.sym_min:     "int",
    torch.sym_int:     "int",
    torch.sym_float:   "float",
}


def translate_graph(gm: fx.GraphModule) -> _C.Trace:
    trace = _C.Trace()
    node_to_ref: Dict[fx.Node, Any] = {}        # fx.Node -> _C.StepInputRef
    node_to_kind: Dict[fx.Node, str] = {}       # fx.Node -> predicted IValue kind

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            _translate_placeholder(node, trace, node_to_ref, node_to_kind)
        elif node.op == "call_function":
            _translate_call_function(node, trace, node_to_ref, node_to_kind)
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


def _translate_placeholder(node, trace, node_to_ref, node_to_kind):
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        idx = trace.v2_add_placeholder_tensor()
        node_to_ref[node] = _C.v2_ref_captured_tensor(idx)
        node_to_kind[node] = "tensor"
    elif isinstance(val, (torch.SymInt, int)):
        idx = trace.v2_add_placeholder_int()
        node_to_ref[node] = _C.v2_ref_captured_int(idx)
        node_to_kind[node] = "int"
    else:
        raise NotImplementedError(
            f"v2 placeholder val type not supported yet: {type(val).__name__} "
            f"(node {node.name}). Handles Tensor + SymInt only."
        )


def _translate_call_function(node, trace, node_to_ref, node_to_kind):
    target = node.target

    if isinstance(target, torch._ops.HigherOrderOperator):
        raise NotImplementedError(
            f"v2 does not support HigherOrderOperator {target}. Use "
            f"torch.compile(backend='inductor') for control-flow workloads."
        )

    if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        # Merge args+kwargs into the schema's positional slot order, then
        # build (refs, predicted_kinds) in parallel so we can compute
        # coercions slot-by-slot.
        positional_refs, positional_kinds = _merge_args_kwargs_via_schema(
            target, node.args, node.kwargs, node_to_ref, node_to_kind)
        coercions = _compute_coercions(target, positional_kinds)
        op_name = _qualified_op_name(target)
        n_out = len(target._schema.returns)
        step_idx = trace.v2_add_tensor_op_step(
            op_name, positional_refs, n_out, coercions=coercions)
        node_to_kind[node] = "tuple" if n_out > 1 else "tensor"
    elif callable(target):
        inputs = [_node_arg_to_ref(a, node_to_ref) for a in node.args]
        builtin = _OP_TO_BUILTIN.get(target)
        kwargs_refs = [_node_arg_to_ref(v, node_to_ref) for v in node.kwargs.values()]
        if builtin is not None:
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
        node_to_kind[node] = _BUILTIN_OUTPUT_KIND.get(target, "other")
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


# ---------------------------------------------------------------------------
# Ref + kind helpers
# ---------------------------------------------------------------------------
def _node_arg_to_ref(value, node_to_ref):
    if isinstance(value, fx.Node):
        return node_to_ref[value]
    if isinstance(value, (list, tuple)):
        return _C.v2_ref_list([_node_arg_to_ref(v, node_to_ref) for v in value])
    return _C.v2_ref_literal(value)


def _predict_value_kind(value, node_to_kind) -> str:
    """Predicted runtime IValue kind for a node-arg value.

    Returns one of: tensor, int, float, bool, list, tuple, other.
    """
    if isinstance(value, fx.Node):
        return node_to_kind.get(value, "other")
    # bool must come before int (bool is int subclass)
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, (list, tuple)):
        return "list"
    return "other"


def _qualified_op_name(op) -> str:
    """Build 'aten::view.<overload>' style name. Schema's overload_name
    may be empty string — that's the default overload and C++ findOp
    wants the empty string."""
    schema = op._schema
    return f"{schema.name}.{schema.overload_name}"


def _merge_args_kwargs_via_schema(op, args, kwargs, node_to_ref, node_to_kind):
    """For a kTensorOp step, lift kwargs into the positional slot order
    dictated by the op's schema. Return (refs, kinds) parallel lists in
    schema-positional order. Missing args get filled with their schema
    default value as a literal ref / kind."""
    schema_args = op._schema.arguments
    n_positional = len(args)

    refs: List[Any] = [_node_arg_to_ref(a, node_to_ref) for a in args]
    kinds: List[str] = [_predict_value_kind(a, node_to_kind) for a in args]

    remaining_kwargs = dict(kwargs)
    for i in range(n_positional, len(schema_args)):
        sa = schema_args[i]
        name = sa.name
        if name in remaining_kwargs:
            v = remaining_kwargs.pop(name)
            refs.append(_node_arg_to_ref(v, node_to_ref))
            kinds.append(_predict_value_kind(v, node_to_kind))
        elif sa.has_default_value():
            v = sa.default_value
            refs.append(_C.v2_ref_literal(v))
            kinds.append(_predict_value_kind(v, node_to_kind))
        else:
            break
    if remaining_kwargs:
        raise NotImplementedError(
            f"v2 schema kwarg merge: unconsumed kwargs {list(remaining_kwargs)} "
            f"for op {_qualified_op_name(op)}; schema args: "
            f"{[a.name for a in schema_args]}"
        )
    return refs, kinds


def _compute_coercions(op, positional_kinds) -> List[Any]:
    """For a kTensorOp step's merged positional refs, return one
    ArgCoercion tag per slot based on (schema arg type, predicted kind
    of the IValue we'll resolve to).

    The result is frozen at translation time so C++ replay just switches
    on the tag rather than running schema().arguments()->kind() on each
    call (DESIGN §17.6.9 opt #3)."""
    schema_args = op._schema.arguments
    NONE = _C.ArgCoercion.NONE
    SCALAR_T = _C.ArgCoercion.SCALAR_TO_TENSOR
    LIST_I = _C.ArgCoercion.LIST_TO_INT_LIST
    LIST_T = _C.ArgCoercion.LIST_TO_TENSOR_LIST

    out: List[Any] = []
    for k, kind in enumerate(positional_kinds):
        if k >= len(schema_args):
            out.append(NONE)
            continue
        sa_type = schema_args[k].type
        # Unwrap Optional[T] -- coercion follows the inner type.
        if sa_type.kind() == "OptionalType":
            sa_type = sa_type.getElementType()
        kind_str = sa_type.kind()
        if kind_str == "TensorType":
            if kind == "tensor":
                out.append(NONE)
            elif kind in ("int", "float", "bool", "other"):
                out.append(SCALAR_T)
            else:
                # Lists / tuples reaching a Tensor slot are unusual;
                # leave alone and let callBoxed surface a clear error.
                out.append(NONE)
        elif kind_str == "ListType":
            if kind == "list":
                elem_kind = sa_type.getElementType().kind()
                # SymInt[] / int[] both surface as IntType at this layer.
                if elem_kind in ("IntType", "SymIntType"):
                    out.append(LIST_I)
                elif elem_kind == "TensorType":
                    out.append(LIST_T)
                else:
                    out.append(NONE)
            else:
                out.append(NONE)
        else:
            out.append(NONE)
    return out
