"""v2 replay performance benchmark.

Compares wall-clock per call across five modes on the same function:

    eager        : plain function call, no compile pipeline
    dynamo       : torch.compile(backend="eager")
                   -- Dynamo prelude (call_size etc.) + raw gm
    aot_eager    : torch.compile(backend="aot_eager")
                   -- Dynamo prelude + AOTAutograd graph + boxed_nop runner
    v2           : torch.compile(backend=aot_autograd(fw_compiler=v2.fw_compiler))
                   -- Dynamo prelude + AOTAutograd graph + C++ trace replay
    v2_cap       : v2.capture(fn, *example_args) direct-replay
                   -- Trace replay only, bypasses Dynamo at call time

Device is controlled by TDC_DEVICE (default cpu). When running on an
accelerator, tensors are allocated on DEVICE and each timed call is
wrapped in SYNC() so wall-clock measurements reflect kernel completion,
not just dispatch enqueueing.
"""
import os
import sys
import time

import torch
import torch_dispatch_capture.v2 as tdcv2
from torch._dynamo.backends.common import aot_autograd
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


def _rand(*shape):
    return torch.randn(*shape, device=DEVICE)


WORKLOADS = {
    "tiny (view+floordiv)":            (workload_tiny,      (_rand(8, 6),)),
    "pointwise (small 64x64)":         (workload_pointwise, (_rand(64, 64), _rand(64, 64))),
    "pointwise (medium 512x512)":      (workload_pointwise, (_rand(512, 512), _rand(512, 512))),
    "attention QK (B=4,S=64,H=32)":    (workload_attention, (_rand(4, 64, 32), _rand(4, 64, 32))),
    "attention QK (B=8,S=512,H=128)":  (workload_attention, (_rand(8, 512, 128), _rand(8, 512, 128))),
}


# ---------------------------------------------------------------------------
# Compile each workload under each mode
# ---------------------------------------------------------------------------
def build_variants(fn, example_inputs):
    """Return dict of {label: callable} for fn under each mode."""
    return {
        "eager":     fn,
        "dynamo":    torch.compile(fn, backend="eager", dynamic=True),
        "aot_eager": torch.compile(fn, backend="aot_eager", dynamic=True),
        "v2":        torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=tdcv2.fw_compiler),
            dynamic=True,
        ),
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
              f"{'v2':>10} {'v2_cap':>10} {'cap/eager':>10}")
    print(f"\n{header}")
    print(f"{'(all numbers in us)':<33}")
    print("-" * len(header))
    for label, (fn, inputs) in WORKLOADS.items():
        torch._dynamo.reset()
        variants = build_variants(fn, inputs)
        times = {}
        for name, callable_ in variants.items():
            if name != "v2_cap":   # v2_cap already captured by build_variants
                torch._dynamo.reset()
            times[name] = time_iters(callable_, inputs)
        cap_over_eager = times["v2_cap"] / times["eager"] if times["eager"] > 0 else float("nan")
        print(f"{label:<33} {fmt_us(times['eager']):>10} {fmt_us(times['dynamo']):>10} "
              f"{fmt_us(times['aot_eager']):>10} {fmt_us(times['v2']):>10} "
              f"{fmt_us(times['v2_cap']):>10} {cap_over_eager:>9.2f}x")


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


def run_profile(workload_label="attention QK (B=8,S=512,H=128)",
                out="prototypes/traces/v2_replay.json"):
    fn, inputs = WORKLOADS[workload_label]
    variants = build_variants(fn, inputs)
    cfn_v2 = variants["v2"]
    cfn_cap = variants["v2_cap"]

    # warmup both to amortize compile
    for _ in range(50):
        cfn_v2(*inputs)
        cfn_cap(*inputs)
    SYNC()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with profile(activities=_profile_activities(), record_shapes=False) as prof:
        for _ in range(100):
            cfn_cap(*inputs)
        SYNC()
    prof.export_chrome_trace(out)

    print(f"\n{'='*78}")
    print(f"  v2.capture direct-replay profile — {workload_label}, 100 iterations")
    print(f"  device: {DEVICE}    trace: {out}")
    print(f"{'='*78}")
    sort_key = "self_cuda_time_total" if DEVICE.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    print("# v2 framework benchmark")
    print_device_banner()
    print("# eager / dynamo / aot_eager / v2 (compile) / v2_cap (direct-replay)")
    run_speed_table()
    run_profile()
