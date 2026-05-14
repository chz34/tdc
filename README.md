# torch_dispatch_capture

C++ dispatcher-level capture/replay for PyTorch — PoC of the design in
`agent_space/cpp_dispatch_capture_design.md`.

## Build

```bash
cd agent_space/torch_dispatch_capture
MAX_JOBS=4 pip install -e . --no-build-isolation -v
```

`MAX_JOBS=4` is enforced by `setup.py` (clamped down if higher in env).
The build uses `torch.utils.cpp_extension.CppExtension`, so it picks up
the PyTorch installation in the active venv.

If the build fails on missing torch headers, make sure your venv has a
matching PyTorch installed: `python -c "import torch; print(torch.__path__)"`.

## Quick check

```python
import torch
import torch_dispatch_capture as tdc

a = torch.zeros(4); b = torch.ones(4); out = torch.empty(4)
with torch.no_grad(), tdc.capture() as trace:
    torch.add(a, b, out=out)

print(trace)
trace.replay()       # out is now [1,1,1,1]
a.fill_(10.0)
trace.replay()       # out is now [11,11,11,11]
a.resize_(8); b.resize_(8); out.resize_(8)
a.fill_(2.0); b.fill_(3.0)
trace.replay()       # dynamic shape: out is [5]*8, same trace
```

## Tests

```bash
cd agent_space/torch_dispatch_capture
python -m unittest discover test -v
python test/test_benchmark.py           # for benchmark numbers
```

## Layout

```
csrc/
  capture_context.{h,cpp}   data structures + TLS, Trace::dump
  capture_fallback.cpp      boxed fallback registered on PrivateUse2
  trace.cpp                 Trace::replay (the hot path)
  bindings.cpp              pybind11 module
python/__init__.py          capture() context manager
setup.py                    CppExtension config (MAX_JOBS<=4)
test/
  test_correctness.py       numerical correctness vs eager
  test_dynamic_shape.py     resize / varied batch / in-place mutation
  test_benchmark.py         eager vs replay timing
```

## Known v1 limitations (intentional)

- Must capture inside `torch.no_grad()` — see design doc §7.
- Captured `OperatorHandle` / `KernelFunction` are valid as long as no
  op gets unregistered between capture and replay (true in normal use).
- TensorList args (e.g. `aten::cat`) are recorded as literal IValues:
  in-place mutation of list elements may not propagate across replays.
  Add a list-classify path if needed (see `classify_input`).
- `as_strided` and other ops whose `size`/`stride` args are baked as
  Python ints will be frozen at capture-time values. Document for users.

## How it works (3-line summary)

1. Boxed fallback on `PrivateUse2` fires on every aten op while capture
   is active; it does `OperatorEntry::lookup` once per op, stashes the
   `KernelFunction`, classifies inputs (external/prev-step/literal),
   then runs the op normally.
2. `Trace::replay()` re-pushes inputs onto the stack (reading current
   metadata from external Tensor objects) and invokes each cached
   `SafeKernelFunction` directly — no key extraction, no alias
   resolution, no profiler hooks, no reentrancy bookkeeping.
3. Dynamic shape is automatic because the kernel reads sizes/strides/
   data_ptr from the current Tensor on each call.
