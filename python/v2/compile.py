"""Public v2 API — @tdc.v2.compile() decorator and the fw_compiler hook.

Wires the AOT graph translator into AOTAutograd via the standard
fw_compiler interface, so users see the familiar `torch.compile`
decorator shape (DESIGN.md §17.6.5).
"""
from __future__ import annotations

import torch
from torch._dynamo.backends.common import aot_autograd

from .trace import replay
from .translator import translate_graph


def fw_compiler(gm, _sample_inputs):
    """AOTAutograd fw_compiler: AOT FX GraphModule -> v2 Trace -> callable.

    The returned callable is what AOTAutograd substitutes for the user's
    forward function. It accepts positional args matching the graph's
    placeholder order (SymInts already resolved to Python ints by the
    Dynamo prelude) and returns a tuple of Tensors.

    `_sample_inputs` is part of AOTAutograd's interface but unused here —
    every value we need is on the graph itself (placeholder metadata).
    """
    trace = translate_graph(gm)

    def replay_callable(*args):
        result = replay(trace, args)
        return result[0] if len(result) == 1 else result

    return replay_callable


def compile(fn=None, *, dynamic: bool = True):
    """Decorator. Equivalent to:

        torch.compile(fn,
                      backend=aot_autograd(fw_compiler=tdc.v2.fw_compiler),
                      dynamic=dynamic)

    Usage:

        @tdc.v2.compile(dynamic=True)
        def fn(x):
            return x.view(x.shape[0] // 2, 2, -1)
    """
    def wrap(f):
        return torch.compile(
            f,
            backend=aot_autograd(fw_compiler=fw_compiler),
            dynamic=dynamic,
        )

    return wrap if fn is None else wrap(fn)
