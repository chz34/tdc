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
import os
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

# Reuse model builders + torchbench loader from v2_benchmark to avoid duplication.
sys.path.insert(0, str(_PROTO))
from v2_benchmark import _build_transformer_block, _load_torchbench  # noqa: E402


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
    print("| variant       | per_call_us | time_vs_eager | capture_s |")
    print("| ------------- | ----------- | ------------- | --------- |")
    for name, per_us, cap_s, err in rows:
        if err is not None:
            print(f"| {name:<13} | -           | -             | skipped: {err}")
            continue
        ratio = f"{per_us / eager_us:.2f}x" if eager_us else "-"
        cap_str = f"{cap_s:.2f}" if cap_s is not None else "-"
        print(f"| {name:<13} | {per_us:>10.2f}  | {ratio:<13} | {cap_str:>9} |")
    print()


def workload_gemm_chain(x, w):
    return torch.matmul(x, w).relu() + 1.0


def run_scenario_b():
    print(f"### Scenario B -- gemm+pointwise, varying seq_len on {DEVICE}")
    hidden = 64
    seqs = [128, 256, 512, 1024]
    x_example = torch.randn(seqs[0], hidden, device=DEVICE)
    w = torch.randn(hidden, hidden, device=DEVICE)

    captured: dict[str, object] = {}
    cap_seconds: dict[str, float] = {}
    capture_errors: dict[str, str] = {}
    recompile_baseline: dict[str, int] = {}
    counters = torch._dynamo.utils.counters

    for name in VARIANTS:
        torch._dynamo.reset()
        recompile_baseline[name] = counters.get("stats", {}).get("calls_captured", 0)
        c, cap_s, err = _capture_variant(name, workload_gemm_chain, (x_example, w))
        if c is None:
            capture_errors[name] = err
            continue
        captured[name] = c
        cap_seconds[name] = cap_s

    per_shape_us: dict[str, dict[int, float]] = {name: {} for name in VARIANTS}
    for seq in seqs:
        x_seq = torch.randn(seq, hidden, device=DEVICE)
        for name, c in captured.items():
            per_shape_us[name][seq] = _time_call(c, (x_seq, w), n_warmup=5, n_iters=50)

    recompiles_total: dict[str, int] = {}
    for name in captured:
        delta = counters.get("stats", {}).get("calls_captured", 0) - recompile_baseline[name]
        recompiles_total[name] = delta

    print()
    header = "| variant       | " + " | ".join(f"seq={s:>4} us" for s in seqs) + " | recompiles |"
    sep = "| ------------- | " + " | ".join(["-----------"] * len(seqs)) + " | ---------- |"
    print(header)
    print(sep)
    for name in VARIANTS:
        if name in capture_errors:
            cells = " | ".join(["-          "] * len(seqs))
            print(f"| {name:<13} | {cells} | -          |")
            continue
        cells = " | ".join(f"{per_shape_us[name][s]:>10.2f} " for s in seqs)
        print(f"| {name:<13} | {cells} | {recompiles_total[name]:>10} |")
    print()


def run_scenario_c():
    print(f"### Scenario C -- Transformer block on {DEVICE}")
    block = _build_transformer_block(hidden=512, n_heads=8, ffn_inner=2048).to(DEVICE).eval()
    x = torch.randn(2, 16, 512, device=DEVICE)
    with torch.no_grad():
        ref = block(x).detach()

    def fn(inp):
        with torch.no_grad():
            return block(inp)

    eager_us = None
    rows = []
    for name in VARIANTS:
        c, cap_s, err = _capture_variant(name, fn, (x,))
        if c is None:
            rows.append((name, None, None, None, err))
            continue
        try:
            with torch.no_grad():
                out = c(x)
            max_abs = (out - ref).abs().max().item()
        except Exception as e:  # noqa: BLE001
            rows.append((name, None, cap_s, None, f"call failed: {type(e).__name__}: {e}"))
            continue
        if not torch.allclose(out, ref, atol=1e-3, rtol=1e-3):
            rows.append((name, None, cap_s, max_abs, "numerics drift atol/rtol 1e-3"))
            continue
        per_call_us = _time_call(c, (x,))
        if name == "eager":
            eager_us = per_call_us
        rows.append((name, per_call_us, cap_s, max_abs, None))

    print()
    print("| variant       | per_call_us | time_vs_eager | capture_s | max_abs_diff |")
    print("| ------------- | ----------- | ------------- | --------- | ------------ |")
    for name, per_us, cap_s, max_abs, err in rows:
        if err is not None:
            print(f"| {name:<13} | -           | -             | -         | {err}")
            continue
        ratio = f"{per_us / eager_us:.2f}x" if eager_us else "-"
        cap_str = f"{cap_s:.2f}" if cap_s is not None else "-"
        max_abs_str = f"{max_abs:.2e}" if max_abs is not None else "-"
        print(f"| {name:<13} | {per_us:>10.2f}  | {ratio:<13} | {cap_str:>9} | {max_abs_str:>12} |")
    print()


def _run_torchbench_one(model_name: str, batch_size: int):
    """Run all 5 variants on a torchbench model. Returns (eager_us_or_None, rows).
    Rows shape mirrors run_scenario_c: (name, per_us, cap_s, max_abs, err)."""
    loaded = _load_torchbench(model_name, batch_size=batch_size)
    if loaded is None:
        return None, None     # message already printed by _load_torchbench
    fn, example_inputs = loaded

    with torch.no_grad():
        try:
            ref = fn(*example_inputs)
        except Exception as e:  # noqa: BLE001
            print(f"# torchbench:{model_name} eager call failed: "
                  f"{type(e).__name__}: {e}")
            return None, None

    eager_us = None
    rows = []
    for name in VARIANTS:
        c, cap_s, err = _capture_variant(name, fn, example_inputs)
        if c is None:
            rows.append((name, None, None, None, err))
            continue
        try:
            with torch.no_grad():
                out = c(*example_inputs)
        except Exception as e:  # noqa: BLE001
            rows.append((name, None, cap_s, None,
                         f"call failed: {type(e).__name__}: {e}"))
            continue

        # Output may be a tensor, tuple, or HF-style dataclass. Best-effort
        # numerical diff: flatten ref/out via pytree, ignore if structure differs
        # so we don't bail entire timing on output shape quirks.
        max_abs = None
        try:
            from torch.utils._pytree import tree_flatten
            ref_leaves = [t for t in tree_flatten(ref)[0]
                          if isinstance(t, torch.Tensor)]
            out_leaves = [t for t in tree_flatten(out)[0]
                          if isinstance(t, torch.Tensor)]
            if len(ref_leaves) == len(out_leaves) and ref_leaves:
                max_abs = max(
                    (rl - ol).abs().max().item()
                    for rl, ol in zip(ref_leaves, out_leaves)
                    if rl.shape == ol.shape
                )
        except Exception:  # noqa: BLE001
            pass

        per_call_us = _time_call(c, example_inputs, n_warmup=5, n_iters=20)
        if name == "eager":
            eager_us = per_call_us
        rows.append((name, per_call_us, cap_s, max_abs, None))

    return eager_us, rows


def run_scenario_d():
    """Scenario D -- torchbench models. Gated on TDC_TORCHBENCH=1 to keep
    casual sweeps fast (these models are 1M..116M params)."""
    if os.environ.get("TDC_TORCHBENCH", "0") != "1":
        print("### Scenario D -- skipped (set TDC_TORCHBENCH=1 to enable)")
        print()
        return

    # Light-to-heavy ordering so an early failure on the heaviest model
    # doesn't lose the data we already gathered.
    models = [
        ("squeezenet1_1", 8),
        ("BERT_pytorch",  4),
        ("hf_GPT2",       4),
    ]

    for model_name, batch_size in models:
        print(f"### Scenario D -- torchbench:{model_name} (B={batch_size}) on {DEVICE}")
        eager_us, rows = _run_torchbench_one(model_name, batch_size)
        if rows is None:
            print()
            continue

        print()
        print("| variant       | per_call_us | time_vs_eager | capture_s | max_abs_diff |")
        print("| ------------- | ----------- | ------------- | --------- | ------------ |")
        for name, per_us, cap_s, max_abs, err in rows:
            if err is not None:
                print(f"| {name:<13} | -           | -             | -         | {err}")
                continue
            ratio = f"{per_us / eager_us:.2f}x" if eager_us else "-"
            cap_str = f"{cap_s:.2f}" if cap_s is not None else "-"
            max_abs_str = f"{max_abs:.2e}" if max_abs is not None else "-"
            print(f"| {name:<13} | {per_us:>10.2f}  | {ratio:<13} | {cap_str:>9} | {max_abs_str:>12} |")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="A,B,C,D",
                        help="comma-separated subset of {A,B,C,D}; D = torchbench, "
                             "requires TDC_TORCHBENCH=1")
    args = parser.parse_args()
    scenarios = {s.strip().upper() for s in args.scenarios.split(",")}
    if "A" in scenarios:
        run_scenario_a()
    if "B" in scenarios:
        run_scenario_b()
    if "C" in scenarios:
        run_scenario_c()
    if "D" in scenarios:
        run_scenario_d()


if __name__ == "__main__":
    main()
