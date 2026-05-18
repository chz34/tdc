"""Trace data structures for v2 (DESIGN.md §17.6.2).

A Trace is a flat sequence of Steps that — when given concrete
positional inputs in graph-placeholder order — reproduces the user
function. Each Step is either a TENSOR_OP (an aten / prims / custom
OpOverload dispatched through the normal callable interface) or a
PY_CALL (any Python callable: operator.* / torch.sym_* / getitem).

Pure-Python implementation; no C++ extension. The cost is one Python
function call per step at replay time, which is acceptable for the
framework PoC and adequate for non-hot-loop workloads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class RefKind(Enum):
    CAPTURED_TENSOR = "captured_tensor"     # captured_tensors[idx]
    CAPTURED_INT = "captured_int"           # captured_ints[idx]  (Dynamo prelude)
    PREV_STEP_OUTPUT = "prev_step_output"   # outputs[step][slot]
    LITERAL = "literal"                     # constant baked into trace
    LIST = "list"                           # Python list of sub-refs


@dataclass
class StepInputRef:
    kind: RefKind
    idx: int = 0
    step: int = 0
    slot: int = 0
    literal: Any = None
    list_elements: List["StepInputRef"] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.kind is RefKind.CAPTURED_TENSOR:
            return f"T[{self.idx}]"
        if self.kind is RefKind.CAPTURED_INT:
            return f"I[{self.idx}]"
        if self.kind is RefKind.PREV_STEP_OUTPUT:
            return f"Step[{self.step},{self.slot}]"
        if self.kind is RefKind.LITERAL:
            return f"Lit({self.literal!r})"
        if self.kind is RefKind.LIST:
            return f"List({self.list_elements!r})"
        return "?"


class StepKind(Enum):
    TENSOR_OP = "tensor_op"     # op(*args, **kwargs)  where op is OpOverload
    PY_CALL = "py_call"         # fn(*args, **kwargs)  where fn is any Python callable


@dataclass
class Step:
    kind: StepKind
    inputs: List[StepInputRef]
    kwargs: Dict[str, StepInputRef] = field(default_factory=dict)
    op: Optional[Any] = None            # TENSOR_OP: torch._ops.OpOverload
    fn: Optional[Callable] = None       # PY_CALL: Python callable
    name: str = ""

    def __repr__(self) -> str:
        target = self.op or self.fn
        return f"{self.kind.value}({self.name or target}, inputs={self.inputs}, kwargs={self.kwargs})"


@dataclass
class Trace:
    """A captured trace plus the metadata needed to route positional inputs.

    `placeholder_routing[i]` tells us where the i-th positional arg goes
    at replay time: into captured_tensors (if `Tensor` placeholder) or
    captured_ints (if `SymInt` placeholder). The order matches FX
    graph's placeholder order which equals AOTAutograd's call signature.
    """
    steps: List[Step] = field(default_factory=list)
    outputs: List[StepInputRef] = field(default_factory=list)

    placeholder_routing: List[Tuple[RefKind, int]] = field(default_factory=list)
    n_captured_tensors: int = 0
    n_captured_ints: int = 0

    def dump(self) -> str:
        lines = [
            f"Trace: {len(self.steps)} steps, "
            f"{self.n_captured_tensors} captured tensors, "
            f"{self.n_captured_ints} captured ints"
        ]
        for i, s in enumerate(self.steps):
            lines.append(f"  [{i}] {s}")
        lines.append(f"  outputs = {self.outputs}")
        return "\n".join(lines)


def _resolve(
    ref: StepInputRef,
    outputs: List[List[Any]],
    captured_tensors: List[Any],
    captured_ints: List[int],
) -> Any:
    if ref.kind is RefKind.CAPTURED_TENSOR:
        return captured_tensors[ref.idx]
    if ref.kind is RefKind.CAPTURED_INT:
        return captured_ints[ref.idx]
    if ref.kind is RefKind.PREV_STEP_OUTPUT:
        return outputs[ref.step][ref.slot]
    if ref.kind is RefKind.LITERAL:
        return ref.literal
    if ref.kind is RefKind.LIST:
        return [_resolve(e, outputs, captured_tensors, captured_ints)
                for e in ref.list_elements]
    raise AssertionError(f"unknown RefKind: {ref.kind}")


def replay(trace: Trace, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Run `trace` against fresh positional args (graph placeholder order).

    Each Step always produces exactly one slot in `outputs[i]`:
    multi-output aten ops (max.dim etc.) are stored as a Python tuple
    in slot 0, and downstream getitem PY_CALL steps extract elements.
    This keeps the slot dimension flat (always 0) — see DESIGN §17.6.3.
    """
    n_tensors = trace.n_captured_tensors
    n_ints = trace.n_captured_ints
    captured_tensors: List[Any] = [None] * n_tensors
    captured_ints: List[int] = [0] * n_ints

    if len(args) != len(trace.placeholder_routing):
        raise ValueError(
            f"trace expects {len(trace.placeholder_routing)} positional args, "
            f"got {len(args)}"
        )
    for arg, (kind, idx) in zip(args, trace.placeholder_routing):
        if kind is RefKind.CAPTURED_TENSOR:
            captured_tensors[idx] = arg
        else:
            captured_ints[idx] = int(arg)

    outputs: List[List[Any]] = [[] for _ in range(len(trace.steps))]

    def resolve(r: StepInputRef) -> Any:
        return _resolve(r, outputs, captured_tensors, captured_ints)

    for i, step in enumerate(trace.steps):
        positional = [resolve(r) for r in step.inputs]
        kwargs = {k: resolve(v) for k, v in step.kwargs.items()}
        if step.kind is StepKind.PY_CALL:
            assert step.fn is not None
            result = step.fn(*positional, **kwargs)
        else:  # TENSOR_OP
            assert step.op is not None
            result = step.op(*positional, **kwargs)
        outputs[i] = [result]   # always exactly one slot — see docstring

    return tuple(resolve(r) for r in trace.outputs)
