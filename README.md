# torch_dispatch_capture

C++ dispatcher-level capture/replay for PyTorch — PoC of the design
described in [`DESIGN.md`](DESIGN.md).

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

All test files (not just the benchmark) honor the `TDC_DEVICE` env
var. Default is `cpu`; set to any of `cuda` / `xpu` / `mps` / `npu` /
`privateuseone` to run the whole suite on an accelerator:

```bash
TDC_DEVICE=cuda  python -m unittest discover test
TDC_DEVICE=npu   python -m unittest discover test
TDC_DEVICE=mps   python -m unittest discover test
```

Each test class prints a one-line `>>> TDC_DEVICE = ...` banner at
setUp so it's clear what device the run is on. The device-resolution
and synchronize helpers live in `test/_device.py`.

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
  _device.py                shared DEVICE / SYNC helper (reads TDC_DEVICE)
  test_correctness.py       15 tests, no_grad guard, view family
  test_dynamic_shape.py     4 tests, varied batch / resize / mutation
  test_backward.py          8 tests, allow_grad capture + replay
  test_benchmark.py         5 benchmarks, eager vs replay timings
```

## What this PoC is **for**

| Scenario | v1 verdict |
|---|---|
| Inference with host-overhead-dominated small ops | ✓ sweet spot |
| LLM decode (batch=1, seq=1) repeated calls | ✓ |
| KV cache writes / in-place mutation patterns | ✓ |
| `view(-1, N)` / `transpose` / `permute` / `squeeze` (dim-index ops) | ✓ |
| Fixed-shape training loop with warmup-then-capture | ✓ |
| Cross-shape replay where shape changes are along dim-index ops | ✓ |

## What this PoC is **not** for

| Scenario | v1 verdict | Use this instead |
|---|---|---|
| `x.view(x.shape[0]//2, 2, -1)` (shape-derived literal) | ✗ | `torch.compile` (see v2 below) |
| Heavy reductions like `.sum().backward()` with dynamic shape | ✗ | `torch.compile` |
| `as_strided(size, stride)` with shape-following stride | ✗ | rewrite or `torch.compile` |
| Models whose forward has data-dependent control flow | ✗ | `torch.compile(dynamic=True)` |

**Why these limits exist**: our fallback intercepts at the dispatcher,
which sees args **after Python has already evaluated them**. When user
code writes `x.view(x.shape[0]//2, ...)`, the `//2` is a Python int
operation that runs before the `view` call, so the dispatcher gets a
literal int — we have no way to recover the lineage back to `x.shape`.
Solving this requires SymInt-level symbolic tracing, which lives at
the Python bytecode layer (Dynamo / FX), not at the dispatcher.

See `DESIGN.md` §8.1–8.3 for the full discussion.

## Known constraints (within the supported scope)

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
  wrong). Shape changes are fine **when they fall under the supported
  patterns above**.
- TensorList args (e.g., `aten::cat([t1, t2, ...])`) are recorded as
  literal IValues; in-place mutation of list elements may not
  propagate. Common patterns work fine; corner-case workloads might.

## v2 direction (if needed): compose with `torch.compile`, don't compete

Full dynamic-shape support (handling shape-derived literals like
`x.view(x.shape[0]//2, ...)`) is **out of scope for v1 by design**. The
right place to add it is not to evolve v1 with self-written FX graph
parsing — that would replicate a large part of Dynamo and Inductor.
Instead, the proposed v2 is a thin **custom backend for `torch.compile`**:

```python
@torch.compile(backend=tdc_backend, dynamic=True)
def fn(x):
    return x.view(x.shape[0] // 2, 2, -1)
```

In this form:
- `torch.compile` (Dynamo + AOTAutograd) does all the symbolic tracing,
  functionalization, and decomposition for us — handing our backend a
  clean SymInt-bearing functional aten FX graph.
- Our backend translates the graph into an extended trace where size
  computations are explicit Steps (rather than baked literals), then
  uses the existing dispatcher-fallback capture for the Tensor-op
  steps.
- Estimated work: ~1000 LoC of additional graph-translation code,
  vs ~3000+ if we tried to build our own FX parser.

This isn't currently planned — v1 is shipped as a focused PoC for the
host-overhead-bound workloads above. v2 is documented as the natural
next step if production needs full dynamic shape with our dispatcher
acceleration. See design doc §17.6 for the detailed v2 layout.

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
