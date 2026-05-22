"""Dynamic-input-shape benchmark for v2.capture and peer backends.

Companion to v2_benchmark.py:
  - v2_benchmark.py covers fixed-shape workloads (timing reproducibility,
    profiler traces).
  - This file covers varying-shape workloads to validate the runtime
    behaviour of each path when calls come with different shapes
    (dynamo/aot/inductor under `dynamic=True`; v2's `("S", user_idx, dim)`
    runtime spec reading `.size(dim)` at replay time).

Imports the workload primitives, variant builder, correctness helpers,
and device infra from v2_benchmark so we don't duplicate them. Run
this file directly to execute just the dynamic suite, or have
v2_benchmark's `__main__` call the public functions exposed here.

v1 is intentionally excluded from the dynamic suite: its replay()
returns the capture-time output tensor and ignores user args, so it
can't express shape variation.
"""
import io
import sys
import time
import traceback

import torch

# Reuse the static benchmark's primitives. v2_benchmark builds the
# static WORKLOADS dict at import time (allocates a few module
# instances on DEVICE); that's the price of sharing the workload
# code and is amortised across both files when both run in one
# process.
from v2_benchmark import (
    SYNC,
    _build_llama_attention,
    _build_transformer_block,
    _compare_outputs,
    _flatten_output,
    _rand,
    build_variants,
    fmt_us,
    print_device_banner,
    workload_attention,
    workload_pointwise,
    workload_swiglu_ffn,
)


# ---------------------------------------------------------------------------
# Dynamic-shape workloads.
#
# Each entry provides:
#   fn:             function or nn.Module under test.
#   capture_inputs: args fed to build_variants() at construction time;
#                   establishes which dims become SymInt placeholders.
#   input_pool:     list of arg tuples with varied shapes along the
#                   sym dims. The timing loop cycles through these so
#                   every iteration sees a (possibly) different shape.
# ---------------------------------------------------------------------------
def _build_dynamic_workloads():
    workloads = {}

    # Pointwise with both dims varying. Reveals symbolic dim propagation
    # overhead per call (multiple .size() reads, multiple sym arith
    # placeholders feeding into the underlying aten kernels).
    workloads["pointwise dyn-(B,H)"] = dict(
        fn=workload_pointwise,
        capture_inputs=(_rand(32, 64), _rand(32, 64)),
        input_pool=[
            (_rand(b, h), _rand(b, h))
            for b, h in [(16, 32), (32, 64), (48, 96),
                         (64, 64), (96, 128), (128, 64)]
        ],
    )

    # Attention QK with seqlen varying. matmul on the sym dim exercises
    # the more interesting case: kernel-launch-bound (small S) vs
    # arith-bound (larger S) on the same compiled graph.
    workloads["attention QK dyn-S (B=4,H=32)"] = dict(
        fn=workload_attention,
        capture_inputs=(_rand(4, 64, 32), _rand(4, 64, 32)),
        input_pool=[
            (_rand(4, s, 32), _rand(4, s, 32))
            for s in (32, 48, 64, 96, 128)
        ],
    )

    # SwiGLU FFN with varied seqlen. Weights are SAME tensor objects
    # across the pool (id() stable) so v2.capture treats them as
    # constant user-input tensors; only x.shape[1] varies.
    _swi_wg = _rand(512, 128)
    _swi_wu = _rand(512, 128)
    _swi_wd = _rand(128, 512)
    workloads["SwiGLU FFN dyn-S (B=2,H=128,Hi=512)"] = dict(
        fn=workload_swiglu_ffn,
        capture_inputs=(_rand(2, 64, 128), _swi_wg, _swi_wu, _swi_wd),
        input_pool=[
            (_rand(2, s, 128), _swi_wg, _swi_wu, _swi_wd)
            for s in (32, 64, 128, 192, 256)
        ],
    )

    # Transformer block with varied seqlen — the canonical LLM serving
    # variation. Module nn.Parameter/Buffer attrs are stable; only the
    # x tensor's seqlen dim varies. Stresses the same sym dim through
    # every layer: 2 LayerNorms, 4+3 nn.Linear, 2 matmul, softmax.
    workloads["Transformer block dyn-S (B=2,H=256,Hi=1024)"] = dict(
        fn=_build_transformer_block(hidden=256, n_heads=8, ffn_inner=1024),
        capture_inputs=(_rand(2, 128, 256),),
        input_pool=[
            (_rand(2, s, 256),) for s in (32, 64, 128, 192, 256)
        ],
    )

    # LLaMA attention with KV cache, varied seqlen AND start_pos.
    # Both dimensions of variation matter for real KV-cache decode:
    #   - prefill: one big seqlen at start_pos=0
    #   - decode:  seqlen=1 (or a chunk) at incrementing start_pos
    # Dynamo lifts start_pos as a SymInt placeholder (it participates
    # in the cache slice bounds: cache_k[:bsz, start_pos:start_pos+seqlen]),
    # and compile.py:_build_recipe_specs maps it to the user-arg
    # position via ("I", arg_idx) so each replay reads the fresh
    # value. Sequence below exercises a realistic prefill+decode pattern.
    workloads["LLaMA attn dyn-(S,startpos) (B=2,dim=256)"] = dict(
        fn=_build_llama_attention(dim=256, n_heads=8, max_batch=4, max_seq=128),
        capture_inputs=(_rand(2, 16, 256), 0),
        input_pool=[
            (_rand(2, 16, 256), 0),    # prefill 16 tokens
            (_rand(2,  1, 256), 16),   # decode token 17
            (_rand(2,  1, 256), 17),
            (_rand(2,  4, 256), 18),   # chunk 4 tokens
            (_rand(2,  1, 256), 22),
            (_rand(2,  8, 256), 23),   # chunk 8 tokens
        ],
    )

    return workloads


DYNAMIC_WORKLOADS = _build_dynamic_workloads()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_iters_dynamic(callable_, input_pool, n_warmup=12, n_iters=120):
    """Like time_iters in v2_benchmark, but cycles through input_pool
    to exercise varied shapes between calls.

    Warmup covers the full pool at least twice so any per-shape compile
    (dynamo retrace at a new sym dim hint, inductor codegen for a
    previously-unseen guard) is amortised before the timed window
    starts. Each timed iteration picks pool[i % pool_len]; median is
    reported in microseconds.

    On accelerator devices, SYNC() brackets every call so wall-clock
    reflects kernel completion rather than dispatch enqueueing."""
    pool_len = len(input_pool)
    warmup_count = max(n_warmup, pool_len * 2)
    try:
        for i in range(warmup_count):
            callable_(*input_pool[i % pool_len])
            SYNC()
    except Exception:
        traceback.print_exc()
        return None
    samples = []
    for i in range(n_iters):
        inputs = input_pool[i % pool_len]
        t0 = time.perf_counter()
        try:
            callable_(*inputs)
        except Exception:
            traceback.print_exc()
            return None
        SYNC()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def run_dynamic_correctness_check():
    """Per-shape correctness check for the dynamic-shape suite.

    For each workload:
      1. build_variants() once using capture_inputs (each compile-flavour
         backend retraces lazily; v1 / v2 do their single capture here).
      2. For every shape in input_pool, call eager(*shape) to get a
         reference, then every other variant with the same shape and
         compare. Dynamo / aot / inductor will recompile lazily under
         dynamic=True; v2 reuses its single trace by reading .size(dim)
         at call time. v1 is excluded (shape-baked at capture).

    The check verifies *correctness under shape variation*, not
    timing. Skip-on-failure semantics mirror v2_benchmark's static
    run_correctness_check().
    """
    error_buf = io.StringIO()
    orig_stderr = sys.stderr
    results: list[tuple] = []
    all_ok = True

    for label, spec in DYNAMIC_WORKLOADS.items():
        fn = spec["fn"]
        capture_inputs = spec["capture_inputs"]
        input_pool = spec["input_pool"]
        torch._dynamo.reset()
        sys.stderr = error_buf
        try:
            variants = build_variants(fn, capture_inputs)
        finally:
            sys.stderr = orig_stderr
        if "eager" not in variants:
            results.append((label, "", "", "no eager variant, skipping", True))
            continue
        for shape_idx, inputs in enumerate(input_pool):
            # Tag with the first Tensor arg's shape — for all workloads
            # in DYNAMIC_WORKLOADS the varying tensor is the first arg.
            first = inputs[0]
            shape_tag = (
                str(tuple(first.shape))
                if isinstance(first, torch.Tensor) else str(first)
            )
            with torch.no_grad():
                ref = _flatten_output(variants["eager"](*inputs))
            ref = [t.detach().clone() for t in ref]
            for name, callable_ in variants.items():
                if name == "eager":
                    continue
                if name == "v1":
                    # v1 bakes shape; output tensor is sized at capture.
                    if shape_idx == 0:
                        results.append((label, name, "",
                                        "skipped (shape-baked)", True))
                    continue
                if callable_ is None:
                    if shape_idx == 0:
                        results.append((label, name, "",
                                        "skipped (variant not available)", True))
                    continue
                # Do NOT reset Dynamo between shapes -- the whole point
                # of the dynamic test is the SAME compiled callable
                # handles every shape.
                with torch.no_grad():
                    sys.stderr = error_buf
                    try:
                        got = _flatten_output(callable_(*inputs))
                    except Exception:
                        results.append((label, name, shape_tag,
                                        "N/A (exception during call)", False))
                        all_ok = False
                        continue
                    finally:
                        sys.stderr = orig_stderr
                ok, msg = _compare_outputs(ref, got)
                status = "ok" if ok else f"MISMATCH ({msg})"
                results.append((label, name, shape_tag, status, ok))
                if not ok:
                    all_ok = False

    print("\n# dynamic-shape correctness check vs eager")
    print("-" * 110)
    for label, name, shape_tag, status, _ok in results:
        if name:
            print(f"  {label:<42} {name:<13} {shape_tag:<22} {status}")
        else:
            print(f"  {label:<42} {status}")

    errors = error_buf.getvalue()
    if errors:
        print(f"\n{'='*110}")
        print("Errors during dynamic correctness check:")
        print(f"{'='*110}")
        print(errors)

    if not all_ok:
        print("\n*** Dynamic correctness check had failures — see lines above ***")


# ---------------------------------------------------------------------------
# Speed table
# ---------------------------------------------------------------------------
def run_dynamic_speed_table():
    """Per-workload x per-mode timing table for the dynamic-shape suite.

    Differences from v2_benchmark.run_speed_table:
      - Each timed call cycles through input_pool (via
        time_iters_dynamic), so the median reflects steady-state cost
        when shape varies between calls.
      - v1 is excluded from the column set — its replay path can't
        express varied shapes.
      - capture_inputs (not the pool) is used to build_variants(), so
        v2's recipe is set up against a single representative shape
        and then reused across the pool.
    """
    sample_spec = next(iter(DYNAMIC_WORKLOADS.values()))
    error_buf = io.StringIO()
    orig_stderr = sys.stderr
    sys.stderr = error_buf
    try:
        sample_variants = build_variants(
            sample_spec["fn"], sample_spec["capture_inputs"]
        )
    finally:
        sys.stderr = orig_stderr
    variant_names = [n for n in sample_variants.keys() if n != "v1"]
    has_eager = "eager" in variant_names
    ratio_names = [n for n in variant_names if n != "eager"] if has_eager else []

    col_w = 9
    time_header = " ".join(f"{n:>{col_w}}" for n in variant_names)
    ratio_header = " ".join(f"{(n + '/eager'):>{col_w + 1}}" for n in ratio_names)
    if ratio_header:
        header = f"{'workload':<42} {time_header} | {ratio_header}"
    else:
        header = f"{'workload':<42} {time_header}"

    rows: list[tuple[str, dict]] = []
    for label, spec in DYNAMIC_WORKLOADS.items():
        torch._dynamo.reset()
        orig_stderr = sys.stderr
        sys.stderr = error_buf
        try:
            variants = build_variants(spec["fn"], spec["capture_inputs"])
        finally:
            sys.stderr = orig_stderr
        times: dict = {}
        for name in variant_names:
            callable_ = variants.get(name)
            if callable_ is None:
                times[name] = None
                continue
            # No Dynamo reset between shapes within one workload --
            # we want the same compiled artefact to handle every shape.
            sys.stderr = error_buf
            try:
                times[name] = time_iters_dynamic(callable_, spec["input_pool"])
            finally:
                sys.stderr = orig_stderr
        rows.append((label, times))

    print(f"\n# dynamic-shape speed table")
    print(f"\n{header}")
    sub = "(times in us; pool cycled per iter"
    if ratio_header:
        sub += "; ratios relative to eager"
    sub += "; v1 omitted -- shape-baked at capture)"
    print(sub)
    print("-" * len(header))

    for label, times in rows:
        def cell(t):
            return "     N/A" if t is None else fmt_us(t)
        time_strs = " ".join(f"{cell(times[n]):>{col_w}}" for n in variant_names)
        if has_eager:
            eg = times["eager"]
            def ratio(n):
                t = times[n]
                if t is None or eg is None or eg <= 0:
                    return "    N/A "
                return f"{(t/eg):>{col_w}.2f}x"
            ratio_strs = " ".join(f"{ratio(n):>{col_w}}" for n in ratio_names)
            print(f"{label:<42} {time_strs} | {ratio_strs}")
        else:
            print(f"{label:<42} {time_strs}")

    errors = error_buf.getvalue()
    if errors:
        print(f"\n{'='*110}")
        print("Errors during dynamic benchmark:")
        print(f"{'='*110}")
        print(errors)


if __name__ == "__main__":
    print("# v2 dynamic-shape benchmark")
    print_device_banner()
    sample_spec = next(iter(DYNAMIC_WORKLOADS.values()))
    _names = list(build_variants(sample_spec["fn"], sample_spec["capture_inputs"]).keys())
    print(f"# modes: {' / '.join(_names)}")
    run_dynamic_correctness_check()
    run_dynamic_speed_table()
