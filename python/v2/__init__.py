"""torch_dispatch_capture.v2 — AOTAutograd-backed capture/replay.

Status: C++ replay engine, Python translator (DESIGN.md §17.6.9).

Quick start:

    import torch
    import torch_dispatch_capture.v2 as tdcv2

    @tdcv2.compile(dynamic=True)
    def fn(x):
        return x.view(x.shape[0] // 2, 2, -1)   # v1 cannot handle this

    fn(torch.randn(8, 6))      # first call: torch.compile machinery
    fn(torch.randn(12, 5))     # subsequent: one C++ trace, replayed
"""
from .compile import compile, fw_compiler
from .translator import translate_graph

__all__ = [
    "compile",
    "fw_compiler",
    "translate_graph",
]
