"""v3 benchmark -- focused comparison for the cpp_wrapper probe.

Variants: eager, inductor, v3-stock, v3-fallback, v2.
Scenarios (each gated on whether the variant is buildable on the device):
  A -- fixed-shape micro-op hot loop
  B -- dynamic seq_len
  C -- Transformer block / Llama attention

Run with:
    TDC_DEVICE=cpu python prototypes/v3_benchmark.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import median

import torch

# Make `_device` importable when invoked as a script from any cwd.
_PROTO = Path(__file__).resolve().parent
_REPO = _PROTO.parent
sys.path.insert(0, str(_REPO / "test"))
from _device import DEVICE, SYNC  # noqa: E402

import torch_dispatch_capture.v2 as tdcv2
import torch_dispatch_capture.v3 as tdcv3


def _time_call(callable_, args, n_warmup=100, n_iters=1000):
    for _ in range(n_warmup):
        callable_(*args)
    SYNC()
    samples_us = []
    for _ in range(n_iters):
        SYNC()
        t0 = time.perf_counter_ns()
        callable_(*args)
        SYNC()
        samples_us.append((time.perf_counter_ns() - t0) / 1000.0)
    return median(samples_us)


def _capture_variant(name, fn, example_inputs):
    """Returns (callable_or_None, capture_seconds_or_None, error_message_or_None)."""
    try:
        torch._dynamo.reset()
        t0 = time.perf_counter()
        if name == "eager":
            return fn, 0.0, None
        if name == "inductor":
            c = torch.compile(fn, backend="inductor", dynamic=True)
            _ = c(*example_inputs)
            return c, time.perf_counter() - t0, None
        if name == "v3-stock":
            c = tdcv3.capture(fn, *example_inputs)
            return c, time.perf_counter() - t0, None
        if name == "v3-fallback":
            c = tdcv3.capture_fallback(fn, *example_inputs)
            return c, time.perf_counter() - t0, None
        if name == "v2":
            c = tdcv2.capture(fn, *example_inputs)
            return c, time.perf_counter() - t0, None
        raise ValueError(name)
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


VARIANTS = ["eager", "inductor", "v3-stock", "v3-fallback", "v2"]


def workload_pointwise(x, y):
    return ((x + y) * 0.5 - 1.0).relu()


def run_scenario_a():
    print(f"### Scenario A -- workload_pointwise (B=4, dim=256) on {DEVICE}")
    x = torch.randn(4, 256, device=DEVICE)
    y = torch.randn(4, 256, device=DEVICE)
    eager_us = None
    rows = []
    for name in VARIANTS:
        c, cap_s, err = _capture_variant(name, workload_pointwise, (x, y))
        if c is None:
            rows.append((name, None, None, err))
            continue
        per_call_us = _time_call(c, (x, y))
        if name == "eager":
            eager_us = per_call_us
        rows.append((name, per_call_us, cap_s, None))

    print()
    print("| variant       | per_call_us | speedup_vs_eager | capture_s |")
    print("| ------------- | ----------- | ---------------- | --------- |")
    for name, per_us, cap_s, err in rows:
        if err is not None:
            print(f"| {name:<13} | -           | -                | skipped: {err}")
            continue
        speedup = f"{eager_us / per_us:.2f}x" if eager_us else "-"
        cap_str = f"{cap_s:.2f}" if cap_s is not None else "-"
        print(f"| {name:<13} | {per_us:>10.2f}  | {speedup:<16} | {cap_str:>9} |")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="A", help="comma-separated subset of {A,B,C}")
    args = parser.parse_args()
    scenarios = {s.strip().upper() for s in args.scenarios.split(",")}
    if "A" in scenarios:
        run_scenario_a()


if __name__ == "__main__":
    main()
