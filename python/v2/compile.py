"""Public v2 API — @tdcv2.compile() decorator and the fw_compiler hook.

The fw_compiler walks the AOT FX graph (DESIGN.md §17.6.3) and emits a
C++ Trace; the returned callable invokes Trace.v2_replay(args) which
runs the unified C++ replay engine (csrc/trace_v2.cpp). No per-step
Python overhead at run time.
"""
from __future__ import annotations

import torch
from torch._dynamo.backends.common import aot_autograd

from .translator import translate_graph


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
