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

## Why `replay()` returns nothing

A trace is a recording of **side effects**, not a pure function. A
typical captured block in production writes to multiple tensors (KV
cache slices, attention output buffer, sometimes statistics counters);
returning "the last step's output" would silently hide all the other
writes and mislead callers. Patterns A and B above are explicit about
which tensors are observation points.

## Tests

```bash
python -m unittest discover test -v        # all tests
python -m unittest test.test_correctness   # 7 correctness tests
python -m unittest test.test_dynamic_shape # 4 dynamic-shape tests
python test/test_benchmark.py              # 5 benchmarks with numbers
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
  test_benchmark.py         5 benchmarks, eager vs replay timings
```

## Known v1 limitations (intentional)

- Must capture inside `torch.no_grad()`. Autograd support is future
  work — see design doc §17.1 (route B1: cache autograd wrapper as
  kernel; route B2: capture forward + backward together AOT-style).
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
- TensorList args (e.g., `aten::cat([t1, t2, ...])`) are recorded as
  literal IValues; in-place mutation of list elements may not
  propagate. Common patterns work fine; corner-case workloads might.

## How it works (3-line summary)

1. Boxed fallback registered on `TESTING_ONLY_GenericMode` fires on
   every aten op while capture is active; it classifies inputs
   (captured tensor / prior-step output / literal IValue), records the
   step, and runs the op normally.
2. `Trace::replay()` re-pushes inputs onto a stack (reading current
   metadata from the captured Tensor objects, so mutations propagate)
   and invokes each step's op via `op.callBoxed(stack)`.
3. Dynamic shape is automatic because each push reads the Tensor's
   current `sizes/strides/data_ptr`, and the kernel adapts.
