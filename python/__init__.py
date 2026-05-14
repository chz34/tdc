"""torch_dispatch_capture — C++ dispatcher-level capture/replay.

Usage:
    import torch
    import torch_dispatch_capture as tdc

    with torch.no_grad(), tdc.capture() as trace:
        out = my_function(a, b)
    trace.replay()                      # replays the captured op sequence
    a.fill_(3.0); trace.replay()        # auto-reflects in-place mutation
    a.resize_(16, 16); b.resize_(16, 16)
    trace.replay()                      # dynamic shape: no recapture

See agent_space/cpp_dispatch_capture_design.md for the design.
"""
from __future__ import annotations

import contextlib

import torch

from . import _C as _ext

Trace = _ext.Trace
is_capturing = _ext.is_capturing


@contextlib.contextmanager
def capture():
    """Capture all aten ops in scope into a replayable Trace.

    Must be called inside ``torch.no_grad()`` — the v1 implementation skips
    autograd at replay, so any autograd graph built during capture would be
    bound to capture-time tensors and produce wrong gradients on replay.
    Backward support is tracked as a future extension (see design doc §17.1).

    Raises:
        RuntimeError: if called with grad enabled.
        RuntimeError: if another capture is already active on this thread.
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            "tdc.capture() must be inside torch.no_grad(); v1 does not "
            "support autograd. See design doc §7 / §17.1."
        )
    trace = _ext.begin_capture()
    try:
        yield trace
    finally:
        _ext.end_capture()


__all__ = ["Trace", "capture", "is_capturing"]
