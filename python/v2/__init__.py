"""torch_dispatch_capture.v2 — AOTAutograd-backed capture/replay.

Status: framework PoC. Pure-Python translator + replayer over the AOT
forward graph delivered by `aot_autograd(fw_compiler=...)`. See
DESIGN.md §17.6 for the design and `prototypes/` for the empirical
findings this framework is built on.

Quick start:

    import torch
    import torch_dispatch_capture.v2 as tdcv2

    @tdcv2.compile(dynamic=True)
    def fn(x):
        # exact case that v1 cannot handle:
        return x.view(x.shape[0] // 2, 2, -1)

    # Reused across shapes — no recompile, no shape baking:
    fn(torch.randn(8, 6))
    fn(torch.randn(12, 5))
"""
from .compile import compile, fw_compiler
from .trace import (
    RefKind, Step, StepInputRef, StepKind, Trace, replay,
)
from .translator import translate_graph

__all__ = [
    "compile",
    "fw_compiler",
    "translate_graph",
    "replay",
    "Trace",
    "Step",
    "StepInputRef",
    "StepKind",
    "RefKind",
]
