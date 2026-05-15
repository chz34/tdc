# torch_dispatch_capture

C++ dispatcher-level capture/replay for PyTorch — PoC of the design
discussed in `cpp_dispatch_capture_design.md`.

## What it does

Capture every aten op a block of code dispatches, then replay them later
without paying the Python interpreter / framework cost between ops. The
captured trace re-reads each input Tensor's metadata on every replay,
so an input that has been mutated in-place or resized between replays
is reflected automatically — there is no per-shape recapture, unlike
cudagraph.

## Build

```bash
cd agent_space/torch_dispatch_capture
MAX_JOBS=4 pip install -e . --no-build-isolation -v
```

Incremental rebuild only:

```bash
MAX_JOBS=4 python setup.py build_ext --inplace
```

The build uses `torch.utils.cpp_extension.CppExtension`, so it picks up
the PyTorch installation in the active venv. `MAX_JOBS=4` is clamped by
`setup.py` to avoid blowing up parallel compile load.

## Usage patterns

### Pattern A: PyTorch native `out=` style

The simplest pattern — write the op using its `out=` overload so the
result lands in a buffer you already hold:

```python
import torch
import torch_dispatch_capture as tdc

a = torch.zeros(4); b = torch.ones(4); c = torch.empty(4)
with torch.no_grad(), tdc.capture() as trace:
    torch.add(a, b, out=c)

trace.replay()                 # c is now [1, 1, 1, 1]
a.fill_(10.0); trace.replay()  # c is now [11, 11, 11, 11]
a.resize_(8); b.resize_(8); a.fill_(2); b.fill_(3)
trace.replay()                 # dynamic shape: c becomes [5]*8 — same trace
```

Captured `out=` args are auto-resized to zero on replay so the kernel
re-allocates to whatever shape the current inputs require. Users don't
need to manually `c.resize_(...)`.

This pattern is what KV-cache writes / output buffer reuse in LLM
serving look like.

### Pattern B: Natural Python style + observation buffer

When your function is written in the natural Python style with `out =
expr` rebinding (the way most existing models are written), the local
rebinding is invisible to the caller — Python semantics, not a flaw of
this library:

```python
def my_fn(x, w, b):
    out = torch.matmul(x, w.t())   # local rebind; outer `out` unaffected
    out = out + b                   # another local rebind
    return out                      # caller must capture the return
```

You don't need to modify `my_fn`. Inside the capture block, allocate an
observation buffer and copy the function's return value into it:

```python
obs = torch.empty(...)             # observation buffer; outlives trace

with torch.no_grad(), tdc.capture() as trace:
    result = my_fn(x, w, b)        # natural code, untouched
    obs.resize_as_(result)         # capture-time convention 1
    obs.copy_(result)              # capture-time convention 2

trace.replay()
print(obs)                         # the latest replay's output

# Same trace works for new shapes too:
x.resize_(new_shape); x.copy_(new_data)
trace.replay()
print(obs)                         # auto-resized and refilled
```

The two extra in-place calls (`resize_as_` + `copy_`) are recorded as
trace steps; on every replay they update `obs` to mirror whatever the
function returned at that replay. The user's function/model code itself
stays unchanged.

Observe as many outputs as you want:

```python
out_a = torch.empty(...); out_b = torch.empty(...)
with tdc.capture() as trace:
    result_a, result_b = my_fn(x)
    out_a.resize_as_(result_a); out_a.copy_(result_a)
    out_b.resize_as_(result_b); out_b.copy_(result_b)
```

**Cost note**: `copy_` on every replay costs memory bandwidth. For tiny
tensors this is negligible; for very large outputs (e.g., LLM prefill
returning a `[B, S=2048, L=256]` tensor = several MB) the copy itself
can dominate the dispatcher savings. Pick observation points
consciously — only copy what you actually need to inspect.

### Pattern C: Forward + backward capture (experimental, `allow_grad=True`)

The dispatcher fallback fires at `TESTING_ONLY_GenericMode` (priority
#3), which is **above** `AutogradFunctionality` (#19). When grad is
enabled inside the capture block, the autograd engine dispatches
backward aten ops through the dispatcher, and our fallback records
them just like forward ops. A single trace can therefore contain the
entire forward + backward of one training-style step.

**Recommended pattern — warmup, then capture, then replay** (same idiom
as CUDA Graph):

```python
x = torch.randn(8, requires_grad=True)
grad_out = torch.ones(8)

# 1. Warmup: run the forward + backward once eagerly. This brings
#    everything into steady state — most importantly, it allocates
#    `x.grad`. Subsequent .backward() calls go through the in-place
#    AccumulateGrad.add_ path (a dispatched op we can record).
(x * x * 2).backward(grad_out)
x.grad.zero_()                       # zero before capture (optional)

# 2. Capture
with tdc.capture(allow_grad=True) as trace:
    y = x * x * 2
    y.backward(grad_out)

# 3. Replay — reproduces the gradient on demand.
x.grad.zero_(); trace.replay()
# x.grad now contains 4*x (dy/dx for y = 2x^2 with grad seed 1)
```

Replay internally pushes `at::AutoDispatchBelowAutograd` so the
dispatcher skips VariableType wrappers — autograd does NOT re-run at
replay time. This avoids:
  - building a fresh backward graph that nobody traverses (wasted work)
  - attaching `grad_fn` to `.grad` (which would prevent subsequent
    `resize_` between replays)

**Caveats**:

1. **Warmup is the canonical way to set up `.grad`**. AccumulateGrad's
   first-time path is a direct C++ assignment (`x.grad = grad_var`)
   that is NOT dispatched and cannot be recorded; replay would silently
   fail to update `.grad`. Any of these makes the warmup happen:

   ```python
   (x * x).sum().backward()              # easiest: just run it once
   # — or —
   x.grad = torch.zeros_like(x)          # manual pre-alloc also works
   # — or, for a model —
   compute_loss(model, x).backward()     # warmup forward+backward step
   optimizer.zero_grad()
   ```

   Once `.grad` exists as a real tensor, every subsequent backward
   uses the dispatched `add_` accumulate path that we record.

2. **`.grad` accumulates across replays** (same semantics as calling
   `.backward()` repeatedly in eager). You have two choices for
   controlling this, both fine — pick whichever fits your training
   loop:

   **(a) zero outside the trace**, the standard PyTorch idiom. The
   user's existing training loop calls `optimizer.zero_grad()` (or a
   manual `.grad.zero_()`) between replays exactly as it would
   between eager `.backward()` calls:

   ```python
   for batch in dataloader:
       x.data = batch
       optimizer.zero_grad(set_to_none=False)   # MUST be False
       trace.replay()
       optimizer.step()
   ```

   ⚠ The default `set_to_none=True` rebinds `.grad = None`, which
   destroys the Tensor reference the trace is holding. Always pass
   `set_to_none=False` (or use a manual loop `for p in params:
   p.grad.zero_()`).

   **(b) zero inside the trace.** Put `optimizer.zero_grad(set_to_none=False)`
   (or any in-place zero) inside the capture block; those zero ops
   are recorded as steps and replay will perform them automatically
   on every call. The training loop then has nothing to do between
   replays:

   ```python
   with tdc.capture(allow_grad=True) as trace:
       optimizer.zero_grad(set_to_none=False)   # also captured
       loss = compute_loss(model, x)
       loss.backward()

   for batch in dataloader:
       x.data = batch
       trace.replay()              # zero + forward + backward, atomically
       optimizer.step()
   ```

   This costs N extra `zero_` ops per replay (one per parameter),
   each one a cheap memset, but the training-loop code is shorter
   and there's no way to forget the `set_to_none=False` flag. For
   gradient-accumulation training (where you intentionally do NOT
   zero between mini-batches), use form (a) and skip the
   `zero_grad` call as usual.

3. **`requires_grad` must not change** between capture and replay
   (it affects which dispatch keys are in the keyset).

4. **Dynamic-shape leaves**: a leaf tensor with `requires_grad=True`
   refuses `resize_()`. Use `x.data = new_tensor` instead — this
   keeps the same `TensorImpl` (so the trace's captured ref stays
   valid) and replaces storage/sizes:

   ```python
   x.data = torch.randn(new_n)        # ✓ correct
   x.grad.resize_as_(x).zero_()       # x.grad doesn't need this trick
   ```

5. **Backward through reductions bakes shape literals**: ops like
   `sum().backward()` capture the input's shape as an
   `IntArrayRef` literal inside the backward's `expand` op. Replay
   at a different shape uses the stale literal and produces wrong
   gradients. **Element-wise** backward chains (`mul`, `add`,
   `relu`, `silu`, plain `matmul`, ...) work cleanly with dynamic
   shape; backward through `sum`/`mean`/`norm` does not, and you
   need to re-capture per shape.

   See `test/test_backward.py::test_backward_dynamic_shape_with_reduction_fails`
   for the documented expected-failure example.

## Why `replay()` returns nothing

A trace is a recording of **side effects**, not a pure function. A
typical captured block in production writes to multiple tensors (KV
cache slices, attention output buffer, sometimes statistics counters);
returning "the last step's output" would silently hide all the other
writes and mislead callers. Patterns A/B/C above are explicit about
which tensors are observation points.

## Tests

```bash
python -m unittest discover test -v        # all tests
python -m unittest test.test_correctness   # 7 correctness tests
python -m unittest test.test_dynamic_shape # 4 dynamic-shape tests
python -m unittest test.test_backward      # 8 backward tests
python test/test_benchmark.py              # 5 benchmarks with numbers
```

Benchmarks accept `TDC_DEVICE` for cross-device runs:

```bash
TDC_DEVICE=cuda  python -m unittest test.test_benchmark
TDC_DEVICE=npu   python -m unittest test.test_benchmark
TDC_DEVICE=mps   python -m unittest test.test_benchmark
```

## Layout

```
csrc/
  capture_context.{h,cpp}   data structures + TLS, Trace::dump
  capture_fallback.cpp      boxed fallback (TESTING_ONLY_GenericMode)
  trace.cpp                 Trace::replay (the hot path)
  bindings.cpp              pybind11 module
python/__init__.py          capture() context manager + usage docs
setup.py                    CppExtension config (MAX_JOBS<=4)
test/
  test_correctness.py       7 tests, no_grad guard, nested rejection
  test_dynamic_shape.py     4 tests, varied batch / resize / mutation
  test_backward.py          8 tests, allow_grad capture + replay
  test_benchmark.py         5 benchmarks, eager vs replay timings;
                             reads TDC_DEVICE for cross-device runs
```

## Known v1 limitations (intentional)

- **Forward + autograd default**: `capture()` defaults to
  `allow_grad=False` and requires `torch.no_grad()`. Backward support
  is opt-in via `allow_grad=True` (see Pattern C above) and has its
  own caveats listed there.
- The captured Tensor **objects** (Python identity) must be the same
  ones used at replay. Their metadata (sizes/strides/data_ptr) can
  change freely. Use Pattern B to track output-side identity for code
  that returns new tensors.
- Changing dtype / device / layout between capture and replay is not
  supported (would change the dispatch keyset; cached lookups would be
  wrong). Shape changes are fine.
- `as_strided` and similar ops whose `size`/`stride` args are baked as
  Python ints are frozen at capture-time values. Use shape-derived
  parameters (e.g., `view(-1, 8)`) if you need shape-following views.
  The same limitation hits backward ops whose schemas include shape
  literals — most notably reductions' backward (`sum_backward` calls
  `expand(saved_shape)`). Pattern C documents this.
- TensorList args (e.g., `aten::cat([t1, t2, ...])`) are recorded as
  literal IValues; in-place mutation of list elements may not
  propagate. Common patterns work fine; corner-case workloads might.

## How it works (3-line summary)

1. Boxed fallback registered on `TESTING_ONLY_GenericMode` fires on
   every aten op while capture is active; it classifies inputs
   (captured tensor / prior-step output / literal IValue), records the
   step, and runs the op normally. Because GenericMode sits above
   AutogradFunctionality in priority, autograd's backward dispatches
   are also visible when `allow_grad=True`.
2. `Trace::replay()` re-pushes inputs onto a stack (reading current
   metadata from the captured Tensor objects, so mutations propagate)
   and invokes each step's op via `op.callBoxed(stack)`. Replay also
   pushes `at::AutoDispatchBelowAutograd` so autograd wrappers do not
   re-execute — every replay is pure aten-op execution.
3. Dynamic shape is automatic because each push reads the Tensor's
   current `sizes/strides/data_ptr`, and the kernel adapts. The few
   ops that bake shape into literals (`as_strided`,
   `sum_backward`'s `expand`, ...) are the documented exceptions.
