"""torch_dispatch_capture — C++ dispatcher-level capture/replay.

Capture every aten op a block of code dispatches, then replay them later
without paying the Python interpreter cost between ops. The captured
trace re-reads each input Tensor's metadata on every replay, so an input
that has been mutated in-place or resized between replays is reflected
automatically — there is no per-shape recapture.


Quick start (functions with side effects):
==========================================

    import torch
    import torch_dispatch_capture as tdc

    a = torch.zeros(4); b = torch.ones(4); c = torch.empty(4)
    with torch.no_grad(), tdc.capture() as trace:
        torch.add(a, b, out=c)        # writes into c

    trace.replay()                    # c is now [1, 1, 1, 1]
    a.fill_(10.0); trace.replay()     # c is now [11, 11, 11, 11]
    a.resize_(8); b.resize_(8); a.fill_(2); b.fill_(3)
    trace.replay()                    # dynamic shape; c becomes [5]*8


Quick start (natural Python style):
===================================

`replay()` returns nothing on purpose — a trace records *side effects*,
and silently returning "the last step's output" would be misleading
whenever the captured function writes to more than one tensor (e.g., KV
cache + Q buffer + attention output in a transformer block).

When your function is written in the natural Python style:

    def my_fn(x, w, b):
        out = torch.matmul(x, w.t())   # local rebind, NOT visible to caller
        out = out + b                   # another local rebind
        return out                      # caller discards return if not assigned

the rebindings inside `my_fn` are invisible to the caller. To make
replay results externally observable WITHOUT modifying `my_fn`, allocate
a buffer outside and copy the return value into it inside the capture
block:

    obs = torch.empty(...)             # observation buffer; outlives trace
    with torch.no_grad(), tdc.capture() as trace:
        result = my_fn(x, w, b)        # natural code, no `out=` parameter
        obs.resize_as_(result)         # capture-time convention 1
        obs.copy_(result)              # capture-time convention 2

    trace.replay()
    print(obs)                         # reflects the new replay output

    # Dynamic shape — same trace, no recapture:
    x.resize_(new_shape); x.copy_(new_data)
    trace.replay()
    print(obs)                         # auto-resized and refilled

The two extra in-place calls (`resize_as_` + `copy_`) are recorded as
trace steps; on every replay they update `obs` to mirror whatever the
function returned at that replay. The model / function itself stays
unchanged.

You can do this for as many output tensors as you care about:

    out_a = torch.empty(...); out_b = torch.empty(...)
    with tdc.capture() as trace:
        result_a, result_b = my_fn(x)
        out_a.resize_as_(result_a); out_a.copy_(result_a)
        out_b.resize_as_(result_b); out_b.copy_(result_b)

Caveat: `copy_` itself costs memory bandwidth on every replay. For tiny
tensors this is free; for very large outputs (e.g., LLM prefill at long
seqlen) the copy can dominate any dispatcher savings. Pick observation
points consciously.


Limitations (v1):
=================

- Must be inside `torch.no_grad()`. Autograd support is future work
  (see design doc §17.1).
- The captured Tensor *objects* (Python identity) must be the same
  ones used at replay. Their metadata (sizes/strides/data_ptr) can
  change freely.
- Changing dtype / device / layout between capture and replay is not
  supported (changes the dispatch keyset).
- Captured `out=` arguments are auto-resized to zero on replay so the
  kernel's `resize_output()` can re-allocate to the current shape.

See cpp_dispatch_capture_design.md (in the design discussion repo) for
the full design rationale.
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
    Backward support is tracked as a future extension.

    Observing replay results
    ------------------------
    ``replay()`` is intentionally void — a trace records side effects,
    not a return value. There are two patterns:

    1. **Explicit `out=` buffer** (PyTorch native style):

       .. code-block:: python

           with tdc.capture() as trace:
               torch.add(a, b, out=c)
           # `c` is auto-resized and overwritten on every replay.

    2. **Observation buffer convention** for code that returns new
       tensors via natural ``out = expr`` rebinding:

       .. code-block:: python

           obs = torch.empty(...)
           with tdc.capture() as trace:
               result = my_function(x)
               obs.resize_as_(result); obs.copy_(result)
           # `obs` reflects the latest replay; model code stays natural.

    Both patterns support dynamic shape — the trace re-reads each
    input's metadata on every replay.

    Raises:
        RuntimeError: if called with grad enabled.
        RuntimeError: if another capture is already active on this thread.
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            "tdc.capture() must be inside torch.no_grad(); v1 does not "
            "support autograd."
        )
    trace = _ext.begin_capture()
    try:
        yield trace
    finally:
        _ext.end_capture()


__all__ = ["Trace", "capture", "is_capturing"]
