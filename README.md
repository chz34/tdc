# torch_dispatch_capture

C++ dispatcher-level capture/replay for PyTorch — PoC of the design
described in [`DESIGN.md`](DESIGN.md). Replay a captured aten-op
trace without paying Dynamo / AOTAutograd / Python-interpreter cost
per call.

## Two paths

The project ships two capture mechanisms that share the same C++
`Trace` / `Step` data structures and replay engine:

| | v2 (recommended) | v1 |
|---|---|---|
| Entry | `tdcv2.capture(fn, *example_args, allow_grad=...)` | `with tdc.capture():` context manager |
| Capture mechanism | `torch.compile(backend=aot_autograd(fw_compiler=...))` runs once; AOT FX graph is translated to a C++ `Trace` | Boxed dispatcher fallback records every aten op as it dispatches |
| Dynamic shape | Full SymInt support (shape-derived literals like `x.view(x.shape[0]//2, ...)` work) | Only shape changes that propagate through dim-index ops (`view(-1, ...)`, `permute`, `transpose`, ...) |
| Data-dependent shape (`masked_select` / `nonzero` / `unique`, where output size depends on input *values*) | Works, but Dynamo refuses by default — set `torch._dynamo.config.capture_dynamic_output_shape_ops = True` before capture (see `test/test_v2_capture.py::TestV2DataDependentShape`) | Works out of the box with no flags — the dispatcher records the op as-is and replay reads the kernel-computed output size each call |
| Backward | `allow_grad=True` — fw + bw double-graph capture, wrapped in `torch.autograd.Function`. nn.Module Parameters auto-routed to autograd; BN running stats / other mutated buffers written back via AOT `ViewAndMutationMeta` | `allow_grad=True` — autograd's backward ops captured as they dispatch through the fallback. Warmup required to allocate `.grad` first. |
| Cost per call | C++ replay only; no Dynamo / AOT runtime overhead | C++ replay only; no Python overhead between ops |
| Coupling with PyTorch internals | Medium — depends on Dynamo/AOT interfaces incl. `TracingContext.fw_metadata` | Low — only uses dispatcher |
| Best for | Training (incl. BN), models with complex shape arithmetic, anything torch.compile can already trace | Tiny / KV-cache-style decode loops, fixed-shape patterns where Dynamo+AOT overhead at capture is too much |

**Start with v2.** Drop to v1 only if you have a workload where the
`torch.compile` one-time capture cost is unacceptable, or you need the
absolute floor of PyTorch-internals coupling.

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

## v2: the primary path

### Quick start — inference

```python
import torch
import torch_dispatch_capture.v2 as tdcv2

def fn(x):
    # Shape-derived literals work — Dynamo's SymInt tracing handles them.
    return x.view(x.shape[0] // 2, 2, -1)

x_example = torch.randn(8, 16)
captured = tdcv2.capture(fn, x_example)

# Subsequent calls bypass Dynamo / AOT runtime entirely:
out = captured(torch.randn(8, 16))
out = captured(torch.randn(12, 16))    # different shape, same trace
```

`captured` is a plain callable. Internally, `tdcv2.capture` runs
`torch.compile(backend=aot_autograd(...))` once, intercepts the
forward-graph compile callback, translates the FX graph into a C++
trace, and returns a wrapper that calls `Trace::replay_v2()`. After
the one-time capture, there is no Dynamo, no AOT runtime wrapper, no
Python guard tower on the call path — just IValue conversion and C++
replay.

### Training — nn.Module, backward, optimizer step

```python
import torch.nn as nn
import torch.nn.functional as F
import torch_dispatch_capture.v2 as tdcv2

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.BatchNorm1d(16),       # BN running stats are written back across replays
    nn.ReLU(),
    nn.Linear(16, 4),
)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

def train_step(x, y):
    return F.cross_entropy(model(x), y)    # natural closure form

x_ex = torch.randn(32, 8)
y_ex = torch.randint(0, 4, (32,))

# allow_grad=True captures both forward and backward graphs and wraps
# them in a torch.autograd.Function. nn.Module parameters lifted by
# Dynamo are surfaced to autograd as leaf inputs, so `param.grad` is
# populated on `loss.backward()` exactly as in eager.
captured = tdcv2.capture(train_step, x_ex, y_ex, allow_grad=True)

for x, y in batches:
    opt.zero_grad()
    captured(x, y).backward()
    opt.step()        # in-place .data update is visible to the next replay
```

Two things work that look like they shouldn't:

- **`model.parameters()` get gradients** even though they're not in the
  positional args of `captured(x, y)`. v2 detects Parameters lifted by
  Dynamo, pre-binds them into the trace's `captured_tensors_` slot AND
  surfaces them as `_CapturedFn.apply` leaf args. Backward returns one
  grad per leaf; autograd routes them into `param.grad` the same way
  `aot_eager` does.
- **`opt.step()` is observed by the next replay**. SGD's
  `param.data.add_(grad, alpha=-lr)` is an in-place mutation on the
  same TensorImpl the trace holds, so it lands in `captured_tensors_`
  automatically. Same mechanism handles BN's `running_mean` /
  `running_var` updates: v2 reads AOT's `ViewAndMutationMeta` off
  `TracingContext.fw_metadata` and `.copy_()`s the new values back
  into the buffer tensors at the end of each forward replay — the
  same epilogue RuntimeWrapper does, just done manually because v2
  bypasses the wrapper for speed.

No `torch.func.functional_call` boilerplate. No manual parameter
threading. Switch `bn.eval()` after training and the BN running stats
are correctly accumulated.

### What v2 does NOT cover

- **Module replacement**: `model.fc = nn.Linear(...)` after capture
  doesn't take effect; the trace still references the old Parameter
  objects. In-place updates (`opt.step()`, `param.data.copy_(...)`)
  are fine.
- **Capture-time mode is frozen**: if you capture with
  `model.train()`, switching to `model.eval()` afterwards doesn't
  change the recorded BN code path (training-mode BN uses batch
  stats; eval-mode uses running stats; these are different ops
  inside the trace). Re-capture if you need to switch modes.
- **Torchbench HuggingFace `outputs.loss`-style returns**: the bench's
  `forward()` is supported (no positional Tensors needed — closure
  lifts everything), but if your benchmark wrapper returns
  `BaseModelOutputWith*` dataclasses and you're consuming `.loss`
  through that wrapper, you may need a small adapter.

## v3: Inductor `cpp_wrapper` probe (forward-only)

`tdcv3` is a thin adapter around `torch.compile(backend="inductor",
dynamic=True)` with `inductor_config.cpp_wrapper=True`. It exists for
benchmarking, not to replace v2 — its purpose is to put the
cpp_wrapper-emitted C++ host driver next to v2's boxed `callBoxed`
trace on the same workload and measure the difference.

Two variants:

```python
import torch_dispatch_capture.v3 as tdcv3

# Stock: cpp_wrapper on, Inductor still fuses & emits Triton/C++ kernels.
captured = tdcv3.capture(fn, *example_args)

# Fallback: cpp_wrapper on AND every op forced through aten via FallbackKernel.
# This is the shape that mirrors v2 most directly (no fusion, aten-only).
captured = tdcv3.capture_fallback(fn, *example_args)

out = captured(*new_args)
report = tdcv3.last_capture_report()
# {'variant', 'capture_seconds', 'fx_node_count', 'fallback_node_count',
#  'so_path', 'cpp_source_path'}
```

Use v3 when you want to quantify *what does the cpp_wrapper host-side
path actually buy us, and how much of inductor's win comes from fusion
vs. wrapper shape?* Use v2 for everything else — v3 inherits Dynamo's
recompile semantics, has no zero-recompile guarantee, and is
forward-only in v0.1.

See `docs/specs/2026-05-28-v3-design.md` for the design and
`prototypes/v3_benchmark.py` for the canonical comparison.

## v4: capture Inductor's `fx_wrapper` host graph (pure Python)

`tdcv4` is a pure-Python module (no C++ extension) that grabs the
`torch.fx.GraphModule` Inductor builds when `config.fx_wrapper=True` —
the *host* graph of allocations + Triton-kernel-launch HOPs + aten
extern calls, captured at Inductor's only stable hook,
`WrapperFxCodegen.compile_graph(gm)`. Unlike v1/v2/v3 it does not
translate to a C++ `Trace`; it hands you the host gm itself. Requires
PyTorch >= 2.9 (when `config.fx_wrapper` and per-device
`fx_wrapper_codegen` landed).

Three entry points:

```python
import torch_dispatch_capture.v4 as tdcv4

# 1. Capture the host gm(s) for inspection.
res = tdcv4.capture_fx(fn, *example_args)        # -> FxCaptureResult
res.compiled(*args)         # the normal fused callable
res.gms                     # captured host GraphModules, in compile order

# 2. Route the host gm through your own backend on EVERY (re)compile.
runner = tdcv4.compile_with_gm_backend(fn, gm_backend=my_backend)
# my_backend(gm, example_inputs) -> callable; robust to recompiles.

# 3. Bring up Inductor on a device with NO codegen backend.
with tdcv4.enable_device_via_fallback("cpu", my_backend, decompose=False):
    torch.compile(model, backend="inductor")(x)
```

`enable_device_via_fallback` forces every op to an aten extern (no
fusion -> all-extern host graph -> FX-convertible) and routes the
result through `fx_wrapper` to your backend, after asserting the graph
really is all-extern. `gm_backend=None` just runs the host gm via
`gm.forward` (pure enablement, no substitution). `decompose=False`
empties Inductor's decomposition table so big aten ops
(native_layer_norm, gelu, ...) survive as a single fallback dispatch;
those de-decomposed ops are registered as explicit fallbacks
(`force_all_fallback_lowerings(extra_ops=...)`) so they do not reach
Inductor's implicit-fallback path, whose debug log eagerly stringifies
the deeply nested IR and effectively hangs.

### Keep fused kernels (CompiledKernel HOP)

`enable_device_with_fusion` is the fusion-*preserving* counterpart: it
keeps the device's real scheduling, so fused kernels survive in the host
gm as `compiled_kernel_wrapper_mutation` HOP nodes (a non-Triton analog
of Inductor's `triton_kernel_wrapper_mutation`) instead of being
fallback-expanded.

```python
import torch_dispatch_capture.v4 as tdcv4

# CPU C++ fused kernels -> HOP nodes (zero PyTorch changes):
with tdcv4.enable_device_with_fusion("cpu", my_backend):
    out = torch.compile(fn, backend="inductor")(a, b)
# host gm now has compiled_kernel_wrapper_mutation node(s); the cpp
# kernel is compiled via PyCodeCache and stored in a global side table.
```

Adding another compiler is "subclass + register", no edit to the
converter/wrapper:

```python
from torch_dispatch_capture.v4 import (
    CompiledKernelBackend, register_compiled_kernel_backend,
)

class MyBackend(CompiledKernelBackend):
    def handles_definition(self, defn_line): ...   # claim a kernel def
    def compile_kernel(self, converter, defn_line): ...  # -> callable
    def mutated_arg_indices(self, call_line): ...  # which args it writes

register_compiled_kernel_backend(MyBackend())
```

`CppPybindingBackend` (CPU cpp) and `DvmBackend` (torch_npu
`TORCHINDUCTOR_NPU_BACKEND=dvm`) ship registered. The dvm path is
validated end to end on NPU (`relu(a@b+a)*2` -> 1 HOP node, numerics
match eager) and needs one small torch_npu change so its call line
carries `arg_types` (the written-arg set); see the probes
`prototypes/dvm_fxwrapper_{runtime,static}_probe.py` and design section
10. Mutation indices otherwise come from the existing `arg_types`
(writeable buffers are non-const pointers).

Pure-Python install (no C++ build) for v4 only:

```bash
TDC_PURE_PYTHON=1 pip install -e .
```

See `docs/specs/2026-06-04-v4-fx-capture-design.md` and DESIGN.md
section 18 for the fx_wrapper interaction points, and
`docs/specs/2026-06-11-compiled-kernel-hop-design.md` for the
CompiledKernel HOP and its cpp/dvm backends.

## v1: dispatcher-level capture

Use v1 when the one-time `torch.compile` cost of v2 is unacceptable —
typically very small ops, KV-cache decode loops, or anywhere you need
the lowest possible coupling with PyTorch internals.

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

### v1 — when NOT to use it

| Scenario | v1 verdict | Use v2 instead |
|---|---|---|
| `x.view(x.shape[0]//2, 2, -1)` (shape-derived literal) | ✗ | ✓ |
| Heavy reductions like `.sum().backward()` with dynamic shape | ✗ | ✓ |
| `as_strided(size, stride)` with shape-following stride | ✗ | ✓ |
| Models whose forward has data-dependent control flow | ✗ | ✓ (via Dynamo's compile-time specialization) |
| nn.Module training with BN running stats | works but caveat-heavy | ✓ recommended |

**Why these limits exist for v1**: our fallback intercepts at the
dispatcher, which sees args **after Python has already evaluated them**.
When user code writes `x.view(x.shape[0]//2, ...)`, the `//2` is a
Python int operation that runs before the `view` call, so the
dispatcher gets a literal int — we have no way to recover the lineage
back to `x.shape`. Solving this requires SymInt-level symbolic tracing,
which lives at the Python bytecode layer (Dynamo / FX) — exactly what
v2 plugs into. See `DESIGN.md` §8.1–8.3 for the full discussion.

## Why `replay()` returns nothing

A trace is a recording of **side effects**, not a pure function. A
typical captured block in production writes to multiple tensors (KV
cache slices, attention output buffer, sometimes statistics counters);
returning "the last step's output" would silently hide all the other
writes and mislead callers. Patterns A/B/C above are explicit about
which tensors are observation points.

v2 follows the same model internally (the trace records side effects
on `captured_tensors_` slots), but its user-facing API wraps the trace
in a callable that returns the user's expected outputs — the
side-effects-only design is hidden behind `_CapturedFn.forward` /
`_CapturedFn.backward`.

## Tests

```bash
# all tests
python -m unittest discover test -v

# single test file -- test/ has no __init__.py, so the
# `python -m unittest test.test_v2_capture` form errors;
# use one of these instead:
( cd test && python -m unittest test_v2_capture -v )
python -m unittest discover -s test -p test_v2_capture.py -v

# single test case
( cd test && python -m unittest \
  test_v2_backward.TestV2Backward.test_batchnorm_training_buffer_mutation -v )
```

All test files (not just the benchmarks) honor the `TDC_DEVICE` env
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
csrc/                       C++ source (built as one _C.so)
  capture_context.{h,cpp}   data structures + TLS capture context + dump
  capture_fallback.cpp      v1 boxed fallback (TESTING_ONLY_GenericMode)
  trace.cpp                 v1 Trace::replay hot path
  trace_v2.cpp              v2 Trace::replay_v2 engine + builtin dispatch
  bindings.cpp              pybind11 module (both v1 and v2)
python/
  __init__.py               v1 capture() context manager + usage docs
  v2/__init__.py            v2 entry (re-exports capture, translate_graph)
  v2/compile.py             v2 capture() + recipe building
  v2/translator.py          FX graph -> C++ Trace translator
  v2/fx_passes.py           FX rewrites used before translation
  v4/__init__.py            v4 entry (re-exports capture_fx etc.)
  v4/capture_fx.py          capture_fx / compile_with_gm_backend / enable_device_via_fallback
setup.py                    CppExtension config (MAX_JOBS<=4; TDC_PURE_PYTHON=1 for v4-only)
test/
  _device.py                shared DEVICE / SYNC helper (reads TDC_DEVICE)
  test_correctness.py       v1 correctness
  test_dynamic_shape.py     v1 dynamic shape
  test_backward.py          v1 allow_grad capture + replay
  test_benchmark.py         v1 timing benchmarks
  test_v2_capture.py        v2 end-to-end
  test_v2_kwargs.py         v2 kwargs handling
  test_v2_nn_module.py      v2 nn.Module wrapper=True path
  test_v2_inplace_mutation.py  v2 in-place op cross-replay behaviour
  test_v2_backward.py       v2 backward + parameter routing + BN buffer writeback
  test_v2_triton.py         v2 + torch.library.triton_op custom ops
  test_v4_capture_fx.py     v4 fx_wrapper capture + enable_device_via_fallback
prototypes/                 exploratory / benchmark scripts (gitignored runs)
  v2_benchmark.py           v1 / v2 / inductor speed comparison
  training_benchmark.py     fw+bw+SGD across all variants (in-house + torchbench)
  run.py                    torchbench-style capture validation harness
```

## How it works (in 3 lines each)

**v2:** `tdcv2.capture` runs `torch.compile(backend=aot_autograd(fw_compiler=...))`
once. The fw_compiler callback receives the FX graph and AOT's
`ViewAndMutationMeta` (via `TracingContext.fw_metadata`); both are
translated into a C++ `Trace` that knows where each call-time arg
plugs in, where mutated buffers need to be written back at the end of
forward, and which input slots are autograd leaves for the backward
graph. After capture, `captured(*args)` calls `Trace::replay_v2()`
directly — no Dynamo, no AOT runtime wrapper.

**v1:** Boxed fallback registered on `TESTING_ONLY_GenericMode` fires
on every aten op while capture is active; it classifies inputs
(captured tensor / prior-step output / literal IValue), records the
step, and runs the op normally. Because GenericMode sits above
AutogradFunctionality in priority, autograd's backward dispatches are
also visible when `allow_grad=True`. `Trace::replay()` re-pushes
inputs onto a stack (reading current metadata from the captured Tensor
objects, so mutations propagate) and invokes each step's op via
`op.callBoxed(stack)`.

Dynamic shape is automatic in v1 for ops that read shape from input
TensorImpls; v2 handles the full SymInt case because the FX graph
preserves shape-derived literals as explicit `sym_*` Steps.

**v4:** Registers a `WrapperFxCodegen` subclass for the target device
and flips `config.fx_wrapper=True`, so Inductor emits its host wrapper
as a `torch.fx.GraphModule` instead of Python/C++ text. The subclass
overrides `compile_graph(gm)` — Inductor's last and only touch of that
gm — to capture it or route it to a user backend. No C++ `Trace`,
no replay engine; the captured object is a real FX GraphModule whose
nodes are `empty_strided` allocs, `triton_kernel_wrapper_mutation`
HOP launches, and aten extern calls.
