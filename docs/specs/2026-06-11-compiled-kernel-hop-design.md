# CompiledKernelWrapperMutation HOP — embedding non-Triton fused kernels into the fx_wrapper graph

Status: implemented and tested in `torch_dispatch_capture.v4` (zero PyTorch
changes -- see section 9).
Date: 2026-06-11.
Venue: prototype lives in `torch_dispatch_capture.v4`. The PyTorch-side seams
described in section 5 turned out to be *unnecessary* for the prototype; they
remain as optional upstreaming work if this becomes a first-class Inductor
feature.

## 1. Motivation

Inductor's `fx_wrapper` backend (`WrapperFxCodegen`) turns the host wrapper into a
`torch.fx.GraphModule`. Its `FxConverter` can only convert **Triton** kernel-call
lines: `FxConverter._generate_kernel_call` raises
`"FX conversion only supports Triton kernels"` whenever `not line.triton`. A fused
kernel launch becomes a `triton_kernel_wrapper_mutation` HOP node, but there is no
analogous HOP for a CPU C++ fused kernel (`cpp_fused_*`), so the host graph cannot
represent it.

v4's `enable_device_via_fallback` worked around this by forcing every op to an aten
extern (all-fallback, zero fusion) so the host graph is all-extern and
FX-convertible. That trades away all fusion. This design lifts the limitation:
a new HOP lets a *compiled* (non-Triton) fused kernel be embedded into the host FX
graph, so CPU C++ fusion can flow through `fx_wrapper`.

## 2. Goal / non-goals

Goal:
- A new HOP `CompiledKernelWrapperMutation` (plus functional sibling
  `CompiledKernelWrapperFunctional`) modeled exactly on Triton's
  `triton_kernel_wrapper_mutation` / `_functional`, with full dispatch parity
  (dense / FakeTensorMode / Meta / ProxyTorchDispatchMode / functionalize).
- `FxConverter` converts a non-Triton kernel-call line into that HOP.
- Kernel source generation reuses the normal scheduler flow (no change to how the
  C++ kernel src is produced). Compilation + caching reuse Inductor's existing
  `async_compile` + code cache, exactly as the Triton path does.
- A global side table (key -> compiled callable), mirroring Triton's
  `kernel_side_table`, that the wrapper populates at kernel-definition time and the
  HOP resolves at call time via the key.
- Experimental target: CPU `cpp_fused_*` kernels. Extensible to other compilers.

Non-goals:
- Autotuning of compiled kernels (Triton's compile-time autotune path is not
  mirrored; compiled kernels have a single launch form).
- Backward / training through the HOP (forward inference first; the functionalize
  impl is provided for FX re-processing, not for autograd codegen).

## 3. Background (grounded in torch 2.12 source)

### 3.1 The Triton HOP we mirror (`torch/_higher_order_ops/triton_kernel_wrap.py`)

- `KernelSideTable` (line 152): global, lock-guarded; `add_kernel(k)->idx`,
  `get_kernel(idx)`, `add_constant_args`, `reset_table`. Module singleton
  `kernel_side_table`.
- `TritonKernelWrapperMutation(HigherOrderOperator)` (1250), name
  `"triton_kernel_wrapper_mutation"`, `cacheable=True`. It is a **mutation** HOP:
  returns `None`; effects are in-place writes into the tensors in `kwargs`. The
  kernel is referenced by `kernel_idx` (callables are not graphable).
- Five dispatch impls:
  - `dense` (`CompositeExplicitAutograd`, 1307): resolve kernel + constants, build
    grid_fn, handle TMA, reorder kwargs into positional args by `kernel.arg_names`,
    launch `kernel[grid_fn](*args, ...)`.
  - `FakeTensorMode` (1395): `with mode: return None`.
  - `Meta` (1409): `return None`.
  - `ProxyTorchDispatchMode` (1445): re-emit the HOP as a `call_function` proxy node
    via `trace_triton_kernel_wrapper` (so re-tracing preserves the node).
  - `py_functionalize_impl` (1490): unwrap tensors, compute the written-tensor set
    via `get_mutated_tensors` (which parses the Triton TTIR), call the functional
    sibling with `tensors_to_clone`, then `ctx.replace` / `commit_update` / `sync`
    each mutated input.
- `TritonKernelWrapperFunctional` (1276): functional sibling with its own
  dense/fake/proxy/functionalize impls; returns a dict of cloned outputs.

### 3.2 The fx_wrapper conversion seam (`torch/_inductor/codegen/wrapper_fxir.py`)

- `FxConverter._generate_kernel_definition` (1178): `_import_kernel(code, name)` ->
  `PyCodeCache.load(prologue + code)`, `getattr(mod, name)`, resolve `LambdaFuture`,
  assert result is a `CachingAutotuner`, store `TritonKernel(tuner, wrap_triton(...))`
  in `self.kernels`.
- `FxConverter._generate_kernel_call` (1171): if `not line.triton` ->
  `raise NotImplementedError("FX conversion only supports Triton kernels.")`; else
  `_generate_triton_call` builds the `triton_kernel_wrapper_mutation` node.
- `WrapperFxCodegen._generate` (57): hardcodes `FxConverter(...)`.

### 3.3 The CPU C++ kernel define/compile contract (`codegen/cpp.py`, `async_compile.py`)

- `CppScheduling.define_kernel` (cpp.py:5533): `_, _, arg_types = args.cpp_argdefs()`;
  emits `kernel_body = "async_compile.cpp_pybinding({arg_types!r}, r'''<src>''')"`;
  `wrapper.define_kernel(name, kernel_body, gpu=False, cpp_definition=<C decl>)`.
- `AsyncCompile.cpp_pybinding(argtypes, src)` (async_compile.py:530): single-thread
  -> `CppPythonBindingsCodeCache.load_pybinding(...)`; multi-thread -> `LambdaFuture`.
  `CppPythonBindingsCodeCache` is the C++ analog of `PyCodeCache` (in-mem + on-disk).
  Returns a callable taking the buffers as flat positional tensors; it writes
  outputs in place.
- `KernelArgs` (common.py:1571): `output_buffers`, `inplace_buffers`, `input_buffers`.
  The output + inplace buffers are exactly the mutated args; `cpp_argdefs()` /
  `python_argdefs()` order them, so the mutated positions among `call_args` are known
  at kernel-call codegen time.
- `KernelCallLine` (wrapper.py:688) already carries `triton: bool`, `kernel_name`,
  `call_args`, `arg_types`. `KernelDefinitionLine` (724) carries `kernel_body`,
  `gpu`, `cpp_definition`.

Key consequence: because `kernel_body` already *is* an `async_compile.cpp_pybinding(...)`
call, the existing `_import_kernel` (PyCodeCache.load) mechanism compiles and caches
a C++ kernel with no new machinery -- the only change is to relax its
"must be a CachingAutotuner" assertion. The compiler used is whatever
`async_compile.<method>` the scheduling backend writes into `kernel_body`; that is the
natural extension point for other compilers.

## 4. Design

### 4.1 New HOPs (`python/v4/compiled_kernel_hop.py`)

```python
class CompiledKernelWrapperMutation(HigherOrderOperator):
    # name "compiled_kernel_wrapper_mutation", cacheable=True
    def __call__(self, kernel_idx: int, mutated_arg_indices: tuple[int, ...],
                 args: tuple) -> None: ...

class CompiledKernelWrapperFunctional(HigherOrderOperator):
    # name "compiled_kernel_wrapper_functional"
    def __call__(self, kernel_idx: int, mutated_arg_indices: tuple[int, ...],
                 args: tuple) -> dict[int, Tensor]: ...
```

Signature differences from Triton (all simplifications): no `grid`, no
`tma_descriptor_metadata`, no `constant_args_idx`, no per-name `kwargs`. A compiled
fused kernel takes a flat positional tensor list and writes outputs in place.
`mutated_arg_indices` are the positions in `args` the kernel writes.

### 4.2 Dispatch impls (full Triton parity)

| impl | mutation | functional |
|---|---|---|
| dense (`CompositeExplicitAutograd`) | `side_table.get_kernel(idx)(*resolve(args))` -> None | clone `args[i]` for `i in mutated_arg_indices`; run kernel with clones substituted; return `{i: clone}` |
| FakeTensorMode | `with mode: return None` | return cloned fakes for mutated indices |
| Meta | `return None` | cloned metas |
| ProxyTorchDispatchMode | re-emit self as `call_function` proxy (helper mirrors `trace_triton_kernel_wrapper`) | same |
| py_functionalize_impl | `ctx.unwrap_tensors`; written set = `mutated_arg_indices` (no source parsing); call functional sibling; `ctx.replace`/`commit_update`/`sync` each mutated input | functional form's own functionalize |

The only mechanism that differs from Triton: the written-tensor set comes from
`mutated_arg_indices` (provided by Inductor) instead of `get_mutated_tensors` parsing
the kernel source. Capability is equivalent; implementation is shorter.

### 4.3 Global side table + compiler extensibility

```python
class CompiledKernelSideTable:           # mirrors KernelSideTable
    id_to_kernel: dict[int, Callable]; kernel_to_id; lock
    def add_kernel(self, k) -> int: ...
    def get_kernel(self, idx) -> Callable: ...
    def reset_table(self) -> None: ...    # tests only

compiled_kernel_side_table = CompiledKernelSideTable()
```

The wrapper populates the side table at kernel-definition time (with the compiled
callable) and the HOP carries the integer idx; the dense impl resolves
idx -> callable at run time. This is what "the wrapper maintains its own key->kernel
list" means.

Compiler extensibility is provided by a small `CompiledKernelBackend` registry (added
after the second backend, see section 10). A backend answers three questions about a
kernel-definition line -- does it own this kernel (`handles_definition`), how to
compile it to a callable (`compile_kernel`), and which call args it writes
(`mutated_arg_indices`). `register_compiled_kernel_backend(...)` appends it;
`_select_backend(line)` returns the first match (else `None` -> the Triton path).
Adding a compiler is then "subclass + register" with no edit to the converter or
wrapper. The CPU `CppPybindingBackend` is registered at import as the reference
backend; `DvmBackend` (section 10) is the second. The original
`register_kernel_compiler(kind, fn)` hook remains for compilers that produce their
callable outside `async_compile`.

### 4.4 FxConverter integration

`FxConverter._generate_kernel_definition`: relax `_import_kernel` so a non-Triton
result (a plain callable / pybinding, possibly a `LambdaFuture`) is accepted; store
it via `compiled_kernel_side_table.add_kernel(...)` and record
`kernel_name -> kernel_idx`.

`FxConverter._generate_kernel_call`: replace the `not line.triton` raise with a
`_generate_compiled_kernel_call` that emits a `CompiledKernelWrapperMutation` node
with `kernel_idx`, the resolved arg FX nodes, and `mutated_arg_indices`.

### 4.5 mutated_arg_indices source

No new `KernelCallLine` field is needed: `cpp_argdefs()` already encodes write-ness
in the arg types it emits. Writeable buffers (inplace + output) are emitted as a
non-const pointer `T*`; read-only inputs as `const T*`; sizevars have no `*`. So
the converter derives the mutated set directly from the existing
`KernelCallLine.arg_types`:

```python
mutated = tuple(i for i, t in enumerate(arg_types)
                if "*" in t and not t.strip().startswith("const"))
```

This keeps the feature fully out-of-tree **for the CPU cpp backend** (the
`KernelCallLine`-field plumbing in the original plan is unnecessary). (Verified: a
fused in-place kernel -> `['float*']` -> `(0,)`; a two-input one-output kernel ->
`['const float*', 'const float*', 'float*']` -> `(2,)`.)

Caveat (generalized after dvm, section 10): `arg_types` is only "free" when the
device scheduling already emits it. cpp does (via `cpp_argdefs()`); dvm did not, and
making it emit `arg_types` is the one small torch_npu change the dvm backend needs.
The HOP-side rule above is unchanged -- only the *source* of `arg_types` differs.

### 4.6 Entry point and positioning

New context manager `enable_device_with_fusion(device, gm_backend=None)`:
- registers the *real* `CppScheduling` (fusion enabled), `CompiledKernelFxWrapper`,
  python/cpp placeholder wrappers;
- flips `config.fx_wrapper=True` (+ disables size/alignment asserts, as v4 does);
- does NOT force all-fallback and does NOT assert all-extern.

`CompiledKernelFxWrapper(WrapperFxCodegen)` overrides `_generate` to instantiate
`CompiledKernelFxConverter` instead of `FxConverter`, and keeps v4's `compile_graph`
routing (run `gm.forward`, or hand the gm to `gm_backend`).

This is the fusion-enabled counterpart to `enable_device_via_fallback`: same
fx_wrapper capture, but fused CPU kernels are preserved as HOP nodes instead of being
fallback-expanded.

## 5. PyTorch changes (NOT required; optional upstreaming)

The prototype needs **zero** PyTorch changes (section 9). The seams below are only
relevant if this becomes a first-class Inductor feature rather than an out-of-tree
package, in which case they form a small additive PR:

1. `torch/_inductor/codegen/wrapper_fxir.py`: relax `_import_kernel` to accept
   non-`CachingAutotuner` callables; replace the `not line.triton` raise in
   `_generate_kernel_call` with dispatch to the compiled-kernel path (instead of
   subclassing `FxConverter`).
2. The two HOPs + side table: upstream candidate location
   `torch/_higher_order_ops/compiled_kernel_wrap.py`.

The original plan also proposed adding a `KernelCallLine.mutated_arg_indices` field
(+ cpp computing it from `KernelArgs`); this is unnecessary because the mutated set
is recoverable from the existing `arg_types` (section 4.5).

## 6. Data flow

Compile time: scheduler fuses CPU pointwise/reduction nodes -> `CppScheduling`
emits `KernelDefinitionLine` (body = `async_compile.cpp_pybinding(...)`) +
`KernelCallLine(triton=False)` -> `CompiledKernelFxConverter` loads the kernel via
PyCodeCache (compiles + caches the C++), stores the callable in the side table, and
emits a `CompiledKernelWrapperMutation` node with `mutated_arg_indices` derived from
`arg_types`.

Run time: `gm.forward` reaches the HOP node -> `__call__` dispatches to dense
(`CompositeExplicitAutograd`) -> side table -> `kernel(*args)` writes outputs in
place -> `None`. No fake/proxy mode is active. Handing the gm to another FX
processor for re-tracing triggers the proxy/fake/functionalize impls so the HOP is
preserved / functionalized correctly.

## 7. Testing

HOP unit tests (`compiled_kernel_hop`):
- dense: a trivial compiled C++ kernel writes the expected output buffer in place;
  numerics correct.
- functionalize: after functionalization the mutation is expressed via `ctx.replace`
  (input replaced by functional output), no input aliasing surprises.
- proxy: `make_fx` over a call retains a `compiled_kernel_wrapper_mutation` node.
- fake/meta: returns `None`, no spurious tensors.

End to end (`enable_device_with_fusion`):
- LN+GELU (or `relu(a@b + a) * 2`) compiled on CPU; the captured host gm contains a
  `compiled_kernel_wrapper_mutation` node (contrast: the all-fallback path is fully
  extern); output matches eager within tolerance.
- registry/config restored on context exit; `compiled_kernel_side_table.reset_table()`
  between tests.

## 8. Open questions / future

- Multi-output / inplace-aliasing edge cases in `mutated_arg_indices` (a buffer that is
  both read and written) -- validate the index set matches the kernel's true writes.
- Whether to also mirror Triton's constant-args side table; not needed for cpp_fused
  (no non-graphable constants) but may be for other compilers.
- Autotuning and backward are explicitly deferred.

## 9. Implementation outcome

Implemented entirely in `torch_dispatch_capture.v4`. The CPU cpp backend needs no
PyTorch changes; the dvm backend needs one small torch_npu change (section 10.4).

- `python/v4/compiled_kernel_hop.py`: `CompiledKernelSideTable` +
  `CompiledKernelWrapperMutation` / `CompiledKernelWrapperFunctional` HOPs with the
  full Triton dispatch set (dense / FakeTensorMode / Meta / ProxyTorchDispatchMode /
  functionalize) and autograd-key fallthroughs.
- `python/v4/cpp_fusion.py`: the `CompiledKernelBackend` ABC + registry
  (`register_compiled_kernel_backend`, `_select_backend`), the backend-agnostic
  `CompiledKernelFxConverter` (routes a definition to the first claiming backend, else
  the Triton path), `CompiledKernelFxWrapper` (subclasses `WrapperFxCodegen._generate`
  to use the converter), `enable_device_with_fusion`, and the reference
  `CppPybindingBackend` (`gpu=False` kernels).
- `python/v4/dvm_fusion.py`: `DvmBackend` (name-prefix `dvm_` selection,
  builder-function compile, `_DvmKernelLauncher`), registered at import (section 10).
- Tests: `test/test_v4_compiled_kernel_hop.py` (5), `test/test_v4_cpp_fusion.py`
  (registry + cpp e2e), `test/test_v4_dvm_fusion.py` (dvm selection/mutation, CPU-only);
  demos `prototypes/v4_cpp_fusion_demo.py`, `prototypes/dvm_fxwrapper_*_probe.py`.

Two findings that shaped the final code:

1. `arg_types` is a sufficient mutation source (section 4.5), removing the need for
   any `KernelCallLine` change.
2. For `relu(a@b + a) * 2`, the `+ a` is absorbed into an `addmm` extern, so only
   `relu(...) * 2` fuses -- as an in-place cpp kernel on the addmm output buffer
   (one `float*` in/out arg). The HOP correctly carries `mutated_arg_indices=(0,)`.
   A kernel with distinct inputs and output (`relu(a) + sigmoid(b)`) yields
   `['const float*', 'const float*', 'float*']` -> `mutated=(2,)`.

Subclassing `FxConverter` + `WrapperFxCodegen._generate` (rather than monkeypatching
torch internals) keeps the feature self-contained and copy-migratable, consistent
with v4's design philosophy.

## 10. Second backend: torch_npu dvm/mlir (validated on NPU, 2026-06-15)

The CPU cpp backend proved the HOP path; the dvm/mlir Inductor backend
(`TORCHINDUCTOR_NPU_BACKEND=dvm`, an out-of-tree torch_npu graph-fusion compiler)
proved its **extensibility to a non-Triton, non-cpp compiler on a real accelerator**.
This section records what was found, the resulting refactor, and the one torch_npu
change required.

### 10.1 Result

`relu(a @ b + a) * 2` compiled on NPU under `enable_device_with_fusion("npu", ...)`:
the dvm-fused pointwise kernel (`dvm_fused_add_mul_relu_0`, the `a@b` stays an extern)
is embedded as **one `compiled_kernel_wrapper_mutation` HOP node**, the side table
holds one kernel, and `out` matches eager (`numerics match eager: True`). The dvm
launch ran at run time through the HOP's dense impl, so the launcher contract and the
mutation set are both correct end to end.

End-to-end validation lives out of tree in
`prototypes/dvm_fxwrapper_runtime_probe.py` (Phase 1 = stock WrapperFxCodegen blocks
at the definition gate; Phase 2 = DvmBackend succeeds) and a static
source-analysis companion `prototypes/dvm_fxwrapper_static_probe.py`.

### 10.2 How dvm differs from cpp (and why each assumption broke)

1. **`gpu=True` like Triton.** dvm fused kernels are not `gpu=False`, so the cpp
   discriminator (`not defn_line.gpu`) cannot select them. DvmBackend keys on the
   kernel *name* prefix `dvm_` (both dvm define modes emit `"dvm_" +
   get_fused_kernel_name(...)`), with an `async_compile.import_fx/mlir/akg` body check
   as fallback.
2. **The define body is a builder-name token, not an `async_compile.*` call.** The
   native dvm-codegen mode (`_define_dvm_kernel`) puts the actual builder function in
   the *metadata* arg and sets `kernel_body = "<name>_build"`. `_format_kernel_definition`
   runs the metadata as code and binds the kernel name to the builder function, so
   `getattr(mod, name)` is a plain `<class 'function'>` -- which is exactly the type the
   stock Triton gate (`_import_kernel`) rejects. DvmBackend's `compile_kernel` reuses
   the cpp recipe (`PyCodeCache.load(prologue + _format_kernel_definition(...))` +
   `getattr` + resolve any `CodeCacheFuture`) and wraps the result in
   `_DvmKernelLauncher`.
3. **No `arg_types` on the call line.** dvm's `generate_kernel_call(name, call_args)`
   passed no arg types, so the section-4.5 mutation recovery had nothing to read. This
   is the one gap that required a torch_npu change (10.4).

### 10.3 Launcher contract (confirmed on NPU)

`_DvmKernelLauncher` prefers `compiled.run(*args, stream=...)` and falls back to a
plain `compiled(*args)`; the npu stream is fetched lazily (`get_current_raw_stream`)
so the module still imports on CPU-only hosts. For the validated kernel the compiled
object is a plain builder function, so the plain-call path runs and writes outputs in
place with no explicit stream -- numerics confirm it. The `.run`+stream and import_fx
fallback shapes are coded but not yet exercised (10.5).

### 10.4 The one required torch_npu change

dvm must emit `arg_types` so the mutation set is recoverable. `call_args` is
`inputs ++ outputs` (`find_common_positions`), so the trailing `num_outputs` entries
are exactly the written buffers -- the direct analog of cpp's writeable pointers.

Pitfall that cost two iterations: the call is **not** emitted by the mlir meta kernel.
`NpuMetaScheduling.codegen_node_schedule` ends with
`final_kernel.call_kernel(call_args, name)`, where `final_kernel` is the
`NpuTritonKernel` from `create_kernel_choices`. `num_outputs` lives on the meta kernel
(`compile_kwargs`), so it is stashed onto that kernel
(`kernel.num_outputs = compile_kwargs.get("num_outputs", 0)`), and
`NpuTritonKernel.call_kernel` builds
`arg_types = ["const void*"] * num_inputs + ["void*"] * num_outputs` and passes it.
`NpuMlirWrapperCodeGen.generate_kernel_call` already accepts and ignores `arg_types`
in its non-triton branch, so the normal dvm run path is unaffected; only fx_wrapper
consumers read it. (torch_npu branch `migrate-dvm-to-master`.)

So the honest extensibility cost of a new compiler backend is: **tdc** = one file +
one `register_compiled_kernel_backend(...)`; **the compiler** = make its scheduling
emit `arg_types` (or otherwise surface its written-arg set) if it does not already.

### 10.5 Open edges for dvm (not yet covered)

- In-place mutated buffers: dvm duplicates such a buffer into both the input and the
  output section of `call_args`; arg_types marks the output occurrence written and the
  input occurrence const. The validated kernel is pure pointwise (no in-place), so the
  duplicate-arg case vs HOP functionalization is unverified. Overlaps with the
  section-8 multi-output/aliasing edge.
- Dynamic shape / `_uwu_` symbolic-arg lines (R1): dvm's `call_kernel` writes those as
  bare-string host lines, which FxConverter cannot convert. Not reached by the static
  example; a dynamic-shape subgraph would surface it (see the static probe's R1
  finding).
- Larger / real models (T5-scale): multi-kernel, reductions, extern interleaving --
  same validation we did for cpp.
- The `.run`+stream launcher form and the `import_fx` fallback kernel.
