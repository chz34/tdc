"""torch_dispatch_capture.v2 — AOTAutograd-backed capture/replay.

Status: C++ replay engine, Python translator (DESIGN.md §17.6.9).

Quick start:

    import torch
    import torch_dispatch_capture.v2 as tdcv2

    captured = tdcv2.capture(fn, *example_args)   # one-time
    out = captured(*real_args)                    # direct replay, no torch.compile per call

A torch.compile-style decorator was historically also exposed but turned
out to be strictly slower than 'dynamo eager backend', so it was removed
in favour of v2.capture's direct-replay path.
"""
from .compile import capture
from .translator import translate_graph

__all__ = [
    "capture",
    "translate_graph",
]
