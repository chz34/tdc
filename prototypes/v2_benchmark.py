"""v2 replay performance benchmark.

Compares wall-clock per call across six modes on the same function:

    eager        : plain function call, no compile pipeline
    dynamo       : torch.compile(backend="eager")
                   -- Dynamo prelude (call_size etc.) + raw gm
    aot_eager    : torch.compile(backend="aot_eager")
                   -- Dynamo prelude + AOTAutograd graph + boxed_nop runner
    inductor     : torch.compile(backend="inductor")
                   -- Full Dynamo + AOT + Inductor codegen (fused kernels)
    v1           : with tdc.capture(): ...; trace.replay()
                   -- Dispatcher-level capture, C++ callBoxed in a loop
    v2_cap       : v2.capture(fn, *example_args) direct-replay
                   -- AOT graph translated to C++ trace, replay only

Device is controlled by TDC_DEVICE (default cpu). When running on an
accelerator, tensors are allocated on DEVICE and each timed call is
wrapped in SYNC() so wall-clock measurements reflect kernel completion,
not just dispatch enqueueing.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch_dispatch_capture as tdc           # v1
import torch_dispatch_capture.v2 as tdcv2      # v2
from torch.profiler import profile, ProfilerActivity

# Share the test suite's device helper. test/ is a sibling of prototypes/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))
from _device import DEVICE, SYNC, print_device_banner  # noqa: E402


# ---------------------------------------------------------------------------
# Workloads — three sizes, intentionally CPU-only so timing is reproducible.
# ---------------------------------------------------------------------------
def workload_tiny(x):
    """Smallest case: 1 sym arith + 1 aten op."""
    return x.view(x.shape[0] // 2, 2, -1)


def workload_pointwise(x, y):
    """Several pointwise ops on small tensors; Python overhead dominant."""
    return x * 2.0 + y - 1.5


N_HEADS = 8

def workload_attention(q, k):
    """Attention QK projection — many aten ops + matmul decomposition."""
    B, S, H = q.shape
    h_dim = H // N_HEADS
    q2 = q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3)
    k2 = k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1)
    return torch.matmul(q2, k2)


def workload_swiglu_ffn(x, w_gate, w_up, w_down):
    """SwiGLU feed-forward block, as used in LLaMA / Mistral / Qwen:

        SwiGLU(x) = (silu(x W_gate^T) * x W_up^T) W_down^T

    Exercises 3 matmuls + 1 silu + 1 elementwise mul. Decomposes in AOT
    to expand/clone/_unsafe_view/bmm sequences for each linear, plus
    sigmoid+mul for silu. Representative of a non-trivial LLM block."""
    gate = F.linear(x, w_gate)
    up = F.linear(x, w_up)
    return F.linear(F.silu(gate) * up, w_down)


def _rand(*shape):
    return torch.randn(*shape, device=DEVICE)


def _swiglu_inputs(B, S, H, H_inner):
    """Allocate the (x, w_gate, w_up, w_down) tuple for a SwiGLU FFN.
    Weight shapes mirror torch.nn.Linear's convention: [out, in]."""
    return (
        _rand(B, S, H),
        _rand(H_inner, H),
        _rand(H_inner, H),
        _rand(H, H_inner),
    )


WORKLOADS = {
    "tiny (view+floordiv)":            (workload_tiny,      (_rand(8, 6),)),
    "pointwise (small 64x64)":         (workload_pointwise, (_rand(64, 64), _rand(64, 64))),
    "pointwise (medium 512x512)":      (workload_pointwise, (_rand(512, 512), _rand(512, 512))),
    "attention QK (B=4,S=64,H=32)":    (workload_attention, (_rand(4, 64, 32), _rand(4, 64, 32))),
    "attention QK (B=8,S=512,H=128)":  (workload_attention, (_rand(8, 512, 128), _rand(8, 512, 128))),
    "SwiGLU FFN (B=2,S=64,H=128,Hi=512)":   (workload_swiglu_ffn, _swiglu_inputs(2, 64, 128, 512)),
    "SwiGLU FFN (B=4,S=256,H=512,Hi=2048)": (workload_swiglu_ffn, _swiglu_inputs(4, 256, 512, 2048)),
}


# ---------------------------------------------------------------------------
# Compile each workload under each mode
# ---------------------------------------------------------------------------
def _v1_capture(fn, example_inputs):
    """Capture fn under v1's dispatcher-fallback path.

    Returns a callable that on each invocation replays the captured trace,
    ignoring the passed-in args (v1 replays against the tensors stashed
    at capture time, not fresh ones). The benchmark loop will still call
    it with `*inputs` to keep parity with the other modes.

    Workloads here are fixed-shape so v1's shape-baking is not a concern.
    """
    with torch.no_grad():
        with tdc.capture() as trace:
            fn(*example_inputs)

    def cb(*_unused):
        trace.replay()

    return cb


def build_variants(fn, example_inputs):
    """Return dict of {label: callable} for fn under each mode."""
    return {
        "eager":     fn,
        "dynamo":    torch.compile(fn, backend="eager", dynamic=True),
        "aot_eager": torch.compile(fn, backend="aot_eager", dynamic=True),
        "inductor":  torch.compile(fn, backend="inductor", dynamic=True),
        "v1":        _v1_capture(fn, example_inputs),
        # Direct-replay path: capture once with example_inputs, then call
        # the resulting callable without going through Dynamo/AOT again.
        "v2_cap":    tdcv2.capture(fn, *example_inputs),
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_iters(callable_, inputs, n_warmup=50, n_iters=500):
    """Pre-generated inputs reused across all iterations so the timing
    excludes randn() cost. Median of n_iters samples in microseconds.

    On accelerator devices, SYNC() bracketing each call ensures we
    measure kernel completion rather than just dispatch enqueueing."""
    for _ in range(n_warmup):
        callable_(*inputs)
    SYNC()
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        callable_(*inputs)
        SYNC()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def fmt_us(v):
    return f"{v:8.2f}"


def run_speed_table():
    header = (f"{'workload':<33} {'eager':>10} {'dynamo':>10} {'aot_eager':>10} "
              f"{'inductor':>10} {'v1':>10} {'v2_cap':>10} {'cap/eager':>10}")
    print(f"\n{header}")
    print(f"{'(all numbers in us)':<33}")
    print("-" * len(header))
    for label, (fn, inputs) in WORKLOADS.items():
        torch._dynamo.reset()
        variants = build_variants(fn, inputs)
        times = {}
        for name, callable_ in variants.items():
            if name not in ("v1", "v2_cap"):  # both already captured above
                torch._dynamo.reset()
            times[name] = time_iters(callable_, inputs)
        cap_over_eager = times["v2_cap"] / times["eager"] if times["eager"] > 0 else float("nan")
        print(f"{label:<33} {fmt_us(times['eager']):>10} {fmt_us(times['dynamo']):>10} "
              f"{fmt_us(times['aot_eager']):>10} {fmt_us(times['inductor']):>10} "
              f"{fmt_us(times['v1']):>10} {fmt_us(times['v2_cap']):>10} "
              f"{cap_over_eager:>9.2f}x")


# ---------------------------------------------------------------------------
# Profile: open the v2 mode under torch.profiler and dump a chrome trace
# ---------------------------------------------------------------------------
def _profile_activities():
    """Profile activities matching the current device. CPU is always
    included; accelerator activity is added when one is detected."""
    acts = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda" and hasattr(ProfilerActivity, "CUDA"):
        acts.append(ProfilerActivity.CUDA)
    elif DEVICE.type == "xpu" and hasattr(ProfilerActivity, "XPU"):
        acts.append(ProfilerActivity.XPU)
    elif DEVICE.type == "mps" and hasattr(ProfilerActivity, "MPS"):
        acts.append(ProfilerActivity.MPS)
    elif DEVICE.type == "privateuseone" and hasattr(ProfilerActivity, "PrivateUse1"):
        acts.append(ProfilerActivity.PrivateUse1)
    return acts


def _safe_filename(label):
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def _profile_one(label, callable_, inputs, out_path, n_warmup=50, n_iters=100):
    """Warmup, then profile n_iters calls; export Chrome-trace and print
    the top events. Returns the prof object for caller introspection."""
    for _ in range(n_warmup):
        callable_(*inputs)
    SYNC()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with profile(
        activities=_profile_activities(),
        record_shapes=False,
        with_stack=True,
    ) as prof:
        for _ in range(n_iters):
            callable_(*inputs)
        SYNC()
    prof.export_chrome_trace(out_path)

    sort_key = "self_cuda_time_total" if DEVICE.type == "cuda" else "self_cpu_time_total"
    print(f"\n{'='*78}")
    print(f"  {label} profile — device: {DEVICE}    trace: {out_path}")
    print(f"{'='*78}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    return prof


def run_profile(workload_label="attention QK (B=8,S=512,H=128)",
                out_dir="prototypes/traces"):
    """Profile the same workload under four modes and export four
    Chrome-traces so the timelines can be loaded side-by-side in
    perfetto/chrome://tracing for direct visual comparison.

      <workload>__eager.json     — baseline; no compile pipeline
      <workload>__inductor.json  — torch.compile inductor (fused kernels)
      <workload>__v1.json        — v1 capture/replay (dispatcher level)
      <workload>__v2_cap.json    — v2.capture direct-replay
    """
    fn, inputs = WORKLOADS[workload_label]
    variants = build_variants(fn, inputs)
    stem = _safe_filename(workload_label)

    _profile_one("eager",      variants["eager"],     inputs, f"{out_dir}/{stem}__eager.json")
    _profile_one("inductor",   variants["inductor"],  inputs, f"{out_dir}/{stem}__inductor.json")
    _profile_one("v1",         variants["v1"],        inputs, f"{out_dir}/{stem}__v1.json")
    _profile_one("v2 capture", variants["v2_cap"],    inputs, f"{out_dir}/{stem}__v2_cap.json")


if __name__ == "__main__":
    print("# v2 framework benchmark")
    print_device_banner()
    print("# eager / dynamo / aot_eager / inductor / v1 / v2_cap")
    run_speed_table()
    run_profile()
