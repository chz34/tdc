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


Quick start (forward + backward, experimental):
===============================================

The dispatcher fallback sits above AutogradFunctionality in priority,
so backward aten ops dispatched by the autograd engine are also
visible. Opt in with `allow_grad=True`. Use the warmup-then-capture
idiom (same shape as CUDA Graph):

    x = torch.randn(8, requires_grad=True)
    grad_out = torch.ones(8)

    # 1. Warmup — runs the workload eagerly once so .grad is allocated
    #    and AccumulateGrad will take the dispatched in-place add_ path
    #    for all subsequent backwards.
    (x * x * 2).backward(grad_out)
    x.grad.zero_()

    # 2. Capture — both forward and backward aten ops are recorded.
    with tdc.capture(allow_grad=True) as trace:
        y = x * x * 2
        y.backward(grad_out)

    # 3. Replay — zero grads if non-accumulating semantics is desired.
    x.grad.zero_(); trace.replay()

Replay internally pushes `at::AutoDispatchBelowAutograd` so autograd
wrappers are skipped — replay is pure aten-op execution, no second
backward graph is built.

Caveats:
  - The warmup pass is essential for AccumulateGrad: its first-time
    branch is a direct C++ assignment (NOT dispatched), so without
    warmup the first replay's gradient write would be missed. After
    one eager backward, all subsequent .backward() calls go through
    the dispatched add_ accumulate path that we record.
  - `.grad` accumulates across replays. Zero it manually for one-shot
    semantics.
  - For leaf tensors with `requires_grad=True`, `resize_()` is
    forbidden. Use `x.data = new_tensor` to swap storage while
    keeping the same TensorImpl identity that the trace captured.
  - Backward through reductions (`sum`, `mean`, `norm`, ...) bakes
    shape literals in trace and does NOT support dynamic shape on
    replay. Element-wise / matmul backward chains are fine.


Limitations (v1):
=================

- Default `allow_grad=False` requires `torch.no_grad()`. Use
  `allow_grad=True` for the experimental backward support above.
- The captured Tensor *objects* (Python identity) must be the same
  ones used at replay. Their metadata (sizes/strides/data_ptr) can
  change freely.
- Changing dtype / device / layout between capture and replay is not
  supported (changes the dispatch keyset).
- Captured `out=` arguments are auto-resized to zero on replay so the
  kernel's `resize_output()` can re-allocate to the current shape.

See DESIGN.md (next to this file in the repo root) for the full
design rationale.
"""
from __future__ import annotations

import contextlib

import torch

from . import _C as _ext

Trace = _ext.Trace
is_capturing = _ext.is_capturing


@contextlib.contextmanager
def capture(allow_grad: bool = False):
    """Capture all aten ops in scope into a replayable Trace.

    By default this must be called inside ``torch.no_grad()`` — capture +
    replay is most predictable for inference workloads. Set
    ``allow_grad=True`` to enable experimental forward+backward capture:
    the dispatcher fallback fires before AutogradFunctionality (priority
    #3 vs #19), so calling ``loss.backward()`` inside the capture block
    records all backward aten ops too. Replay re-runs the full op
    sequence, updating ``.grad`` on the captured leaf tensors.

    Recommended usage — warmup, capture, replay (CUDA-Graph idiom):

    .. code-block:: python

        # warmup
        loss = compute_loss(model, x)
        loss.backward()
        optimizer.zero_grad()

        # capture
        with tdc.capture(allow_grad=True) as trace:
            loss = compute_loss(model, x)
            loss.backward()

        # replay per training step
        for step in steps:
            optimizer.zero_grad()
            trace.replay()
            optimizer.step()

    Caveats with ``allow_grad=True``:
      - The warmup pass is the canonical way to ensure ``.grad`` is
        allocated for every parameter / leaf input. AccumulateGrad's
        first-time branch is a direct C++ assignment NOT visible to
        the dispatcher; after one eager backward, AccumulateGrad takes
        the in-place ``add_`` accumulate path which IS dispatched and
        recorded. (You can also manually pre-allocate via
        ``x.grad = torch.zeros_like(x)`` if you have a reason to.)
      - ``.grad`` accumulates on every replay (just like calling
        ``.backward()`` repeatedly in eager). Zero it manually between
        replays if you want non-accumulating semantics.
      - ``requires_grad`` of captured tensors must not change between
        capture and replay (it influences which dispatch keys are in
        the keyset, hence which kernels are selected).
      - For dynamic-shape leaf inputs, prefer ``x.data = new_tensor``
        over ``x.resize_(...)`` (leaf tensors with ``requires_grad``
        refuse in-place resize).
      - Backward ops whose schemas include shape literals — most
        notably ``sum``/``mean``/``norm`` backward (the ``expand``
        with saved shape) — bake those literals at capture and will
        not adapt to new shapes on replay. Element-wise and matmul
        backward chains are dynamic-shape safe.
      - Replay internally pushes ``at::AutoDispatchBelowAutograd``,
        so autograd wrappers are skipped on replay (no second
        backward graph gets built and discarded).

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
        RuntimeError: if called with grad enabled and ``allow_grad=False``.
        RuntimeError: if another capture is already active on this thread.
    """
    if not allow_grad and torch.is_grad_enabled():
        raise RuntimeError(
            "tdc.capture() requires torch.no_grad() by default; pass "
            "allow_grad=True for experimental forward+backward capture."
        )
    trace = _ext.begin_capture()
    try:
        yield trace
    finally:
        _ext.end_capture()


__all__ = ["Trace", "capture", "is_capturing"]
