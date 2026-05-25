"""timm_vision_transformer (torchbench) compatibility probe for v1 / v2.

Loads the timm_vision_transformer model from torchbench (~22M params,
vit_base-style, eval mode, batch=4, 224x224 input -> 1000-class logits)
and exercises each capture path:

  - eager           : reference baseline
  - v1              : tdc.capture context manager + trace.replay()
  - v2 (direct)     : tdcv2.capture(..., wrapper=False)
  - v2 (wrapper)    : tdcv2.capture(..., wrapper=True)

Reports per-path:
  - Whether capture succeeded (and if not, why)
  - Whether replay matches eager output (allclose)
  - Capture wall-time (including any first-call compile/trace cost)
  - Median replay time

Run:
    python prototypes/timm_vit_smoketest.py [BATCH]

The default batch size is 4 to keep CPU runs tractable; pass any int
as the first arg to override.
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch_dispatch_capture as tdc       # noqa: E402  # v1
import torch_dispatch_capture.v2 as tdcv2  # noqa: E402  # v2


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_vit(batch_size: int):
    """Load timm_vision_transformer in eval mode on CPU. torchbench's
    Model wrapper does the timm.create_model dance, weight init, and
    builds example_inputs. We just unwrap to (model, example_inputs)
    -- the same convention the existing v2_benchmark workload table
    uses."""
    mod = importlib.import_module("torchbenchmark.models.timm_vision_transformer")
    bench = mod.Model(test="eval", device="cpu", batch_size=batch_size)
    model, example_inputs = bench.get_module()
    model.eval()
    return model, tuple(example_inputs)


# ---------------------------------------------------------------------------
# Per-path probe
# ---------------------------------------------------------------------------
def probe_eager(fn, inputs, *, n_warmup: int = 1, n_iters: int = 5):
    """Compute reference output + median wall time. Single source of
    truth for downstream correctness checks."""
    with torch.no_grad():
        for _ in range(n_warmup):
            fn(*inputs)
        samples = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            ref = fn(*inputs)
            samples.append(time.perf_counter() - t0)
    samples.sort()
    return ref.detach().clone(), samples[len(samples) // 2] * 1000.0


def probe_v1(fn, inputs):
    """v1 capture: tdc.capture() context manager; trace.replay() re-runs
    each Step via op.callBoxed. v1 captures the same nn.Module instance
    that holds the eager weights, so the comparison is direct.

    Caveats v1 has on this kind of workload:
      - Sym shapes are baked at capture (timm ViT has fixed shapes
        through the network once batch is fixed, so this doesn't bite).
      - Mutation of captured TensorImpl identities at replay time
        means we read the captured output buffer, not a fresh one.
    """
    print("\n# v1 probe")
    try:
        t_cap = time.perf_counter()
        with torch.no_grad():
            with tdc.capture() as trace:
                captured_out = fn(*inputs)
        cap_ms = (time.perf_counter() - t_cap) * 1000.0
        print(f"  capture: ok ({cap_ms:.1f} ms)")
        print(f"  trace:   {trace}")
    except Exception:
        print(f"  capture: FAILED")
        traceback.print_exc()
        return None, None, None

    # Time replay.
    try:
        for _ in range(1):  # warmup
            trace.replay()
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            trace.replay()
            samples.append(time.perf_counter() - t0)
        samples.sort()
        replay_ms = samples[len(samples) // 2] * 1000.0
        print(f"  replay:  ok (median {replay_ms:.1f} ms)")
    except Exception:
        print(f"  replay:  FAILED")
        traceback.print_exc()
        return cap_ms, None, captured_out

    return cap_ms, replay_ms, captured_out


def probe_v2(fn, inputs, *, wrapper: bool):
    label = "v2 (wrapper)" if wrapper else "v2 (direct)"
    print(f"\n# {label} probe")
    try:
        t_cap = time.perf_counter()
        torch._dynamo.reset()
        captured = tdcv2.capture(fn, *inputs, wrapper=wrapper)
        cap_ms = (time.perf_counter() - t_cap) * 1000.0
        print(f"  capture: ok ({cap_ms:.1f} ms)")
    except Exception:
        print(f"  capture: FAILED")
        traceback.print_exc()
        return None, None, None

    try:
        with torch.no_grad():
            for _ in range(1):  # warmup
                captured(*inputs)
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                out = captured(*inputs)
                samples.append(time.perf_counter() - t0)
        samples.sort()
        replay_ms = samples[len(samples) // 2] * 1000.0
        print(f"  replay:  ok (median {replay_ms:.1f} ms)")
        return cap_ms, replay_ms, out
    except Exception:
        print(f"  replay:  FAILED")
        traceback.print_exc()
        return cap_ms, None, None


def compare(name: str, ref, got, atol: float = 1e-3, rtol: float = 1e-3):
    if got is None:
        return
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    if ok:
        print(f"  {name}: matches eager (allclose atol={atol}, rtol={rtol})")
    else:
        max_diff = (ref - got).abs().max().item()
        ref_max = ref.abs().max().item()
        print(f"  {name}: MISMATCH (max abs diff {max_diff:.3e}, "
              f"ref max {ref_max:.3e}, rel {max_diff/max(ref_max, 1e-12):.2e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(batch_size: int):
    print(f"# timm_vision_transformer probe (batch={batch_size}, device=cpu)")
    print(f"# torch {torch.__version__}")

    t0 = time.perf_counter()
    model, inputs = load_vit(batch_size)
    print(f"# loaded model in {(time.perf_counter()-t0)*1000:.0f} ms "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"# input: {[tuple(a.shape) for a in inputs]}")

    # Eager reference. Use separate model clones for each capture path so
    # that v1's capture-time mutation of captured TensorImpls doesn't leak
    # into v2's capture or vice versa.
    m_eager = copy.deepcopy(model)
    ref, eager_ms = probe_eager(m_eager, inputs)
    print(f"\n# eager: median {eager_ms:.1f} ms, output {tuple(ref.shape)}")

    m_v1 = copy.deepcopy(model)
    v1_cap, v1_replay, v1_out = probe_v1(m_v1, inputs)
    compare("v1 replay vs eager", ref, v1_out)

    m_v2d = copy.deepcopy(model)
    v2d_cap, v2d_replay, v2d_out = probe_v2(m_v2d, inputs, wrapper=False)
    compare("v2 (direct) replay vs eager", ref, v2d_out)

    m_v2w = copy.deepcopy(model)
    v2w_cap, v2w_replay, v2w_out = probe_v2(m_v2w, inputs, wrapper=True)
    compare("v2 (wrapper) replay vs eager", ref, v2w_out)

    print("\n# summary (ms)")
    print(f"  {'path':<16} {'capture':>10} {'replay':>10}  {'replay/eager':>14}")
    rows = [
        ("eager",         None,     eager_ms),
        ("v1",            v1_cap,   v1_replay),
        ("v2 (direct)",   v2d_cap,  v2d_replay),
        ("v2 (wrapper)",  v2w_cap,  v2w_replay),
    ]
    for name, cap, rep in rows:
        cap_s = "    -    " if cap is None else f"{cap:9.1f}"
        rep_s = "    -    " if rep is None else f"{rep:9.1f}"
        ratio = "    -    "
        if rep is not None and eager_ms is not None and eager_ms > 0:
            ratio = f"{rep/eager_ms:>12.2f}x"
        print(f"  {name:<16} {cap_s:>10} {rep_s:>10}  {ratio:>14}")


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(batch)
