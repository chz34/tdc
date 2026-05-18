"""v2 replay performance benchmark.

Honest test of the question: does v2's pure-Python replay actually
go faster than alternatives? Or does it have the same overhead pattern
as v1's earlier __torch_dispatch__ pure-Python replay experiment, where
"capture once, replay in Python" gained nothing over plain eager?

Four modes are compared on the same function with the same inputs:

    eager        : plain function call, no compile pipeline
    dynamo       : torch.compile(backend="eager")
                   -- Dynamo prelude (call_size etc.) + raw gm
    aot_eager    : torch.compile(backend="aot_eager")
                   -- Dynamo prelude + AOTAutograd graph + boxed_nop runner
    v2           : torch.compile(backend=aot_autograd(fw_compiler=v2.fw_compiler))
                   -- Dynamo prelude + AOTAutograd graph + our pure-Python replay

Expected outcome (we want to verify, not preassume):

  1. v2 is NOT faster than eager. The Python overhead of replaying each
     Step is comparable to Python calling the op itself. The expectation
     matches the user's prior __torch_dispatch__ experiment.
  2. v2 is comparable to aot_eager since both go through the same
     compile path; the only difference is how the post-compile gm is run.
  3. The interesting metric is not raw speed but "how much overhead
     does the v2 translator + replayer add over aot_eager?" — that's
     the cost of using v2 over just letting aot_eager run the gm.
"""
import os
import time

import torch
import torch_dispatch_capture.v2 as tdcv2
from torch._dynamo.backends.common import aot_autograd
from torch.profiler import profile, ProfilerActivity


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


WORKLOADS = {
    "tiny (view+floordiv)":            (workload_tiny,      (torch.randn(8, 6),)),
    "pointwise (small 64x64)":         (workload_pointwise, (torch.randn(64, 64), torch.randn(64, 64))),
    "pointwise (medium 512x512)":      (workload_pointwise, (torch.randn(512, 512), torch.randn(512, 512))),
    "attention QK (B=4,S=64,H=32)":    (workload_attention, (torch.randn(4, 64, 32), torch.randn(4, 64, 32))),
    "attention QK (B=8,S=512,H=128)":  (workload_attention, (torch.randn(8, 512, 128), torch.randn(8, 512, 128))),
}


# ---------------------------------------------------------------------------
# Compile each workload under each mode
# ---------------------------------------------------------------------------
def build_variants(fn):
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
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_iters(callable_, inputs, n_warmup=50, n_iters=500):
    """Pre-generated inputs reused across all iterations so the timing
    excludes randn() cost (which was ~27% of profile noise). Median of
    n_iters samples in microseconds."""
    for _ in range(n_warmup):
        callable_(*inputs)
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        callable_(*inputs)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def fmt_us(v):
    return f"{v:8.2f}"


def run_speed_table():
    header = (f"{'workload':<33} {'eager':>10} {'dynamo':>10} {'aot_eager':>10} "
              f"{'v2':>10} {'v2/eager':>10} {'v2-aot':>10}")
    print(f"\n{header}")
    print(f"{'(all numbers in us)':<33}")
    print("-" * len(header))
    for label, (fn, inputs) in WORKLOADS.items():
        variants = build_variants(fn)
        times = {}
        for name, callable_ in variants.items():
            torch._dynamo.reset()
            times[name] = time_iters(callable_, inputs)
        v2_over_eager = times["v2"] / times["eager"] if times["eager"] > 0 else float("nan")
        v2_minus_aot = times["v2"] - times["aot_eager"]
        print(f"{label:<33} {fmt_us(times['eager']):>10} {fmt_us(times['dynamo']):>10} "
              f"{fmt_us(times['aot_eager']):>10} {fmt_us(times['v2']):>10} "
              f"{v2_over_eager:>9.1f}x {v2_minus_aot:>9.2f}")


# ---------------------------------------------------------------------------
# Profile: open the v2 mode under torch.profiler and dump a chrome trace
# ---------------------------------------------------------------------------
def run_profile(workload_label="attention QK (B=8,S=512,H=128)",
                out="prototypes/traces/v2_replay.json"):
    fn, inputs = WORKLOADS[workload_label]
    variants = build_variants(fn)
    cfn_v2 = variants["v2"]
    cfn_aot = variants["aot_eager"]

    # warmup both to amortize compile
    for _ in range(50):
        cfn_v2(*inputs)
        cfn_aot(*inputs)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for _ in range(100):
            cfn_v2(*inputs)
    prof.export_chrome_trace(out)

    print(f"\n{'='*78}")
    print(f"  v2 replay profile — {workload_label}, 100 iterations")
    print(f"  trace: {out}")
    print(f"{'='*78}")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


if __name__ == "__main__":
    print("# v2 framework benchmark")
    print("# Pure-Python replay against eager / Dynamo-only / aot_eager baselines")
    run_speed_table()
    run_profile()
