"""Training-loop benchmark for v1 / v2 capture+replay.

Companion to v2_benchmark.py (inference, static shapes) and
dynamic_benchmark.py (inference, varied shapes). This file exercises
the training loop: each iteration runs forward + backward (and
optionally an optimizer step) on a small training-capable model,
then compares the final weight state across modes to verify
correctness.

Design choices:

1. **Functional workloads**. Both v1 and v2 produce gradients only
   for tensors that are EXPLICIT user-input args to the captured
   function. nn.Module parameters that AOT lifts as graph inputs
   internally get gradients in the bw graph, but v2's
   _CapturedFn.backward only forwards user-input grads back to
   autograd, so we'd lose parameter grads if we passed an nn.Module.
   Instead we define `fn(x, y, *weights)` with weights as positional
   args so v2's allow_grad path produces a .grad for each.

2. **Optimizer step is opt-in** (--include-optimizer). The default
   captures fw+bw only and applies weight updates in Python with
   torch.no_grad() outside the trace. With --include-optimizer the
   update happens inside the captured trace via in-place arithmetic
   (limited to SGD-style: w -= lr * w.grad).

3. **Correctness via final weight comparison**. Each variant starts
   from an identical clone of the initial weights, runs N training
   iterations on the same fixed batch, and the final weights are
   compared against eager's. Iteration count is small (~5) to keep
   wall time reasonable while still exercising grad accumulation.

4. **Per-step timing**. After correctness check, the speed table
   times one training iteration (fw+bw[+update]) across all modes,
   reporting median over a warmup+iters window.

Known limitation: v1's allow_grad path currently fails on multi-Linear
training (chained F.linear with autograd backward) -- the dispatcher
fallback misidentifies one of the saved-for-backward tensor
identities, causing a transpose+matmul in the captured bw to consume
the wrong operand at replay. The error surfaces as
"mat1 and mat2 shapes cannot be multiplied (B x C and H x B)". v1's
single-Linear backward (test_backward.py) still works; the
chained-Linear case is a known follow-up.

Run:
    python prototypes/training_benchmark.py
    python prototypes/training_benchmark.py --include-optimizer
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))
from _device import DEVICE, SYNC, print_device_banner  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch_dispatch_capture as tdc            # noqa: E402  # v1
import torch_dispatch_capture.v2 as tdcv2       # noqa: E402  # v2


# ---------------------------------------------------------------------------
# Functional training workloads
# ---------------------------------------------------------------------------
def mlp_loss_fn(x, y, w1, b1, w2, b2):
    """2-layer MLP + cross-entropy loss, fully functional."""
    h = F.relu(F.linear(x, w1, b1))
    logits = F.linear(h, w2, b2)
    return F.cross_entropy(logits, y)


def attn_loss_fn(x, y, w_q, w_k, w_v, w_o, w_proj, b_proj):
    """Single attention head + mean-pool + classifier head."""
    H = x.shape[-1]
    q = F.linear(x, w_q)
    k = F.linear(x, w_k)
    v = F.linear(x, w_v)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (H ** 0.5)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    out = F.linear(out, w_o)
    pooled = out.mean(dim=1)
    logits = F.linear(pooled, w_proj, b_proj)
    return F.cross_entropy(logits, y)


# ---------------------------------------------------------------------------
# Weight + input initialisation (reproducible)
# ---------------------------------------------------------------------------
def make_mlp_weights(in_dim=64, hidden=128, out_dim=10, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [
        (torch.randn(hidden, in_dim, generator=g) * 0.1).to(DEVICE).requires_grad_(True),
        torch.zeros(hidden, device=DEVICE, requires_grad=True),
        (torch.randn(out_dim, hidden, generator=g) * 0.1).to(DEVICE).requires_grad_(True),
        torch.zeros(out_dim, device=DEVICE, requires_grad=True),
    ]


def make_mlp_data(B=8, in_dim=64, num_classes=10, seed=1000):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, in_dim, generator=g).to(DEVICE)
    y = torch.randint(0, num_classes, (B,), generator=g).to(DEVICE)
    return x, y


def make_attn_weights(H=64, num_classes=10, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    weights = []
    for _ in range(4):  # w_q, w_k, w_v, w_o
        weights.append(
            (torch.randn(H, H, generator=g) * 0.1).to(DEVICE).requires_grad_(True)
        )
    weights.append(
        (torch.randn(num_classes, H, generator=g) * 0.1).to(DEVICE).requires_grad_(True)
    )
    weights.append(torch.zeros(num_classes, device=DEVICE, requires_grad=True))
    return weights


def make_attn_data(B=4, S=16, H=64, num_classes=10, seed=1000):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, S, H, generator=g).to(DEVICE)
    y = torch.randint(0, num_classes, (B,), generator=g).to(DEVICE)
    return x, y


# ---------------------------------------------------------------------------
# Common helpers shared across variants
# ---------------------------------------------------------------------------
def _zero_grads(weights):
    for w in weights:
        if w.grad is not None:
            w.grad.zero_()


def _apply_sgd(weights, lr):
    """One SGD step with no momentum, outside the autograd graph."""
    with torch.no_grad():
        for w in weights:
            if w.grad is not None:
                w.add_(w.grad, alpha=-lr)


# ---------------------------------------------------------------------------
# Variant builders -- each returns (step_fn, weights_ref)
#
# step_fn(x, y) -> loss_scalar:
#     - zeros .grad on every captured weight
#     - runs one fw+bw iteration on the variant's path
#     - leaves .grad populated on weights for the caller to consume
#
# The optimizer step (if any) is the caller's responsibility -- the
# benchmark drives `for _ in range(N): loss = step(x, y); if include_opt:
# _apply_sgd(weights, lr)` so each variant sees the same iteration
# structure.
# ---------------------------------------------------------------------------
def build_eager(workload_fn, weights):
    def step(x, y):
        _zero_grads(weights)
        loss = workload_fn(x, y, *weights)
        loss.backward()
        return loss.detach()
    return step


def build_compiled(workload_fn, weights, backend, dynamic=True):
    """Generic torch.compile-based variant (dynamo / aot_eager / inductor)."""
    compiled = torch.compile(workload_fn, backend=backend, dynamic=dynamic)
    def step(x, y):
        _zero_grads(weights)
        loss = compiled(x, y, *weights)
        loss.backward()
        return loss.detach()
    return step


def build_v1(workload_fn, weights, x_ex, y_ex):
    """v1 capture+replay. Needs an eager warmup so AccumulateGrad's
    first-call shortcut allocates .grad on every weight; subsequent
    backward passes go through dispatched aten::add_ which our
    fallback CAN record."""
    # Warmup: eager fw+bw, allocates .grad for x_ex (if it had
    # requires_grad) and every weight.
    loss = workload_fn(x_ex, y_ex, *weights)
    loss.backward()
    _zero_grads(weights)
    # Capture
    with tdc.capture(allow_grad=True) as trace:
        captured_loss = workload_fn(x_ex, y_ex, *weights)
        captured_loss.backward()
    # The capture-time backward also incremented .grad; clear so the
    # first timed iteration starts from a clean state matching eager.
    _zero_grads(weights)

    def step(x, y):
        _zero_grads(weights)
        # v1 replays against the captured input tensors; mutate in place
        # so each replay sees the caller-supplied batch.
        x_ex.detach().copy_(x)
        y_ex.copy_(y)
        trace.replay()
        # captured_loss is the same TensorImpl across replays; its
        # value reflects the most recent replay's loss.
        return captured_loss.detach().clone()
    return step


def build_v2(workload_fn, weights, x_ex, y_ex):
    """v2 capture+replay via tdcv2.capture(..., allow_grad=True)."""
    captured = tdcv2.capture(
        workload_fn, x_ex, y_ex, *weights, allow_grad=True
    )

    def step(x, y):
        _zero_grads(weights)
        loss = captured(x, y, *weights)
        loss.backward()
        return loss.detach()
    return step


# ---------------------------------------------------------------------------
# Workload registry
#
# Each entry encodes "how to build the data + weights + step variants
# for one model". Keeping the spec in a dict so the correctness and
# timing loops are uniform.
# ---------------------------------------------------------------------------
def _build_mlp_spec():
    return dict(
        name="MLP (B=8, in=64, hidden=128, out=10)",
        loss_fn=mlp_loss_fn,
        make_weights=make_mlp_weights,
        make_data=make_mlp_data,
    )


def _build_attn_spec():
    return dict(
        name="Attention block (B=4, S=16, H=64)",
        loss_fn=attn_loss_fn,
        make_weights=make_attn_weights,
        make_data=make_attn_data,
    )


WORKLOADS = [_build_mlp_spec(), _build_attn_spec()]


# Variant ordering -- also the column order of the speed table.
# v2 (wrapper) is omitted: capture() silently downgrades allow_grad=True
# to wrapper=False because aot_function carries its own backward path.
VARIANT_NAMES = [
    "eager",
    "dynamo",
    "aot_eager",
    "inductor",
    "v1",
    "v2",
]


def build_variants(spec, *, x_ex, y_ex):
    """Build {name: (step_fn, weights)} for one workload. The variant
    builders need x_ex / y_ex for capture-time examples (v1, v2)."""
    fn = spec["loss_fn"]
    make_weights = spec["make_weights"]

    # Each variant gets an independent clone of the initial weights so
    # post-iteration weight comparisons are well-defined.
    variants = {}
    for name in VARIANT_NAMES:
        weights = make_weights(seed=0)
        try:
            if name == "eager":
                step = build_eager(fn, weights)
            elif name == "dynamo":
                step = build_compiled(fn, weights, "eager")
            elif name == "aot_eager":
                step = build_compiled(fn, weights, "aot_eager")
            elif name == "inductor":
                step = build_compiled(fn, weights, "inductor")
            elif name == "v1":
                step = build_v1(fn, weights, x_ex.clone(), y_ex.clone())
            elif name == "v2":
                step = build_v2(fn, weights, x_ex.clone(), y_ex.clone())
            else:
                raise ValueError(name)
            variants[name] = (step, weights)
        except Exception:
            traceback.print_exc()
            variants[name] = (None, weights)
    return variants


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def _weights_close(a_list, b_list, atol=1e-4, rtol=1e-4):
    if len(a_list) != len(b_list):
        return False, f"weight count mismatch: {len(a_list)} vs {len(b_list)}"
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if a.shape != b.shape:
            return False, f"weight[{i}] shape {tuple(b.shape)} vs {tuple(a.shape)}"
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a - b).abs().max().item()
            return False, (
                f"weight[{i}] max abs diff {diff:.3e} "
                f"(shape={tuple(a.shape)}, atol={atol})"
            )
    return True, ""


def run_correctness(*, n_iters=5, lr=1e-2, include_optimizer=False):
    """For each workload, run n_iters of fw+bw[+SGD] on every variant
    using the same fixed batch, then compare final weights vs eager."""
    print("\n# training correctness check")
    print("-" * 78)
    print(f"# n_iters={n_iters}, lr={lr}, include_optimizer={include_optimizer}")

    for spec in WORKLOADS:
        print(f"\n## workload: {spec['name']}")
        x, y = spec["make_data"](seed=42)
        variants = build_variants(spec, x_ex=x, y_ex=y)

        # Drive eager first to set the reference.
        ref_step, ref_weights = variants["eager"]
        if ref_step is None:
            print("  eager build failed -- skipping workload")
            continue
        ref_losses = []
        for it in range(n_iters):
            loss = ref_step(x, y)
            ref_losses.append(loss.item())
            if include_optimizer:
                _apply_sgd(ref_weights, lr)
        print(f"  eager final loss: {ref_losses[-1]:.4f}  (path: "
              + ' -> '.join(f"{l:.3f}" for l in ref_losses) + ")")

        # Compare every other variant to eager's final weights.
        for name in VARIANT_NAMES:
            if name == "eager":
                continue
            step, weights = variants[name]
            if step is None:
                print(f"  {name:<12} BUILD FAILED")
                continue
            try:
                losses = []
                for it in range(n_iters):
                    loss = step(x, y)
                    losses.append(loss.item())
                    if include_optimizer:
                        _apply_sgd(weights, lr)
            except Exception as e:
                print(f"  {name:<12} STEP FAILED ({type(e).__name__}: "
                      f"{str(e)[:120]})")
                continue
            ok, msg = _weights_close(ref_weights, weights)
            loss_match = abs(losses[-1] - ref_losses[-1]) < max(
                1e-4, abs(ref_losses[-1]) * 1e-3
            )
            status = "ok" if (ok and loss_match) else "MISMATCH"
            detail = msg if not ok else (
                f"(loss diff |{losses[-1]:.4f} - {ref_losses[-1]:.4f}|)"
                if not loss_match else ""
            )
            print(f"  {name:<12} {status:<10} final_loss={losses[-1]:.4f}  {detail}")


# ---------------------------------------------------------------------------
# Speed table
# ---------------------------------------------------------------------------
def time_step(step_fn, x, y, *, n_warmup=5, n_iters=50,
              include_optimizer=False, weights=None, lr=1e-2):
    """Median wall-time of one step in microseconds. The optimizer step
    is included if `include_optimizer` is True, so the timing reflects
    the actual per-iteration cost the user would see."""
    try:
        for _ in range(n_warmup):
            step_fn(x, y)
            if include_optimizer and weights is not None:
                _apply_sgd(weights, lr)
            SYNC()
    except Exception:
        traceback.print_exc()
        return None
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        try:
            step_fn(x, y)
            if include_optimizer and weights is not None:
                _apply_sgd(weights, lr)
        except Exception:
            traceback.print_exc()
            return None
        SYNC()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def run_speed_table(*, include_optimizer=False):
    print("\n# per-iteration training speed (median, us)")
    col_w = 9
    header_cols = " ".join(f"{n:>{col_w}}" for n in VARIANT_NAMES)
    ratio_cols = " ".join(
        f"{(n + '/eager'):>{col_w + 1}}" for n in VARIANT_NAMES if n != "eager"
    )
    header = f"{'workload':<38} {header_cols} | {ratio_cols}"
    print(header)
    sub = "(times in us; ratios relative to eager"
    sub += "; includes optimizer step)" if include_optimizer else ")"
    print(sub)
    print("-" * len(header))

    for spec in WORKLOADS:
        x, y = spec["make_data"](seed=42)
        variants = build_variants(spec, x_ex=x, y_ex=y)
        times = {}
        for name in VARIANT_NAMES:
            step, weights = variants[name]
            if step is None:
                times[name] = None
                continue
            times[name] = time_step(
                step, x, y,
                include_optimizer=include_optimizer,
                weights=weights,
            )
        def cell(t):
            return "     N/A" if t is None else f"{t:8.2f}"
        time_strs = " ".join(f"{cell(times[n]):>{col_w}}" for n in VARIANT_NAMES)
        eg = times["eager"]
        def ratio(n):
            t = times[n]
            if t is None or eg is None or eg <= 0:
                return "    N/A "
            return f"{(t/eg):>{col_w}.2f}x"
        ratio_strs = " ".join(
            f"{ratio(n):>{col_w}}" for n in VARIANT_NAMES if n != "eager"
        )
        print(f"{spec['name']:<38} {time_strs} | {ratio_strs}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-optimizer", action="store_true",
        help="Apply SGD weight updates inside the timed step. "
             "Default off -- captures fw+bw only."
    )
    parser.add_argument(
        "--n-iters", type=int, default=5,
        help="Training iterations for the correctness check."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2,
        help="SGD learning rate (only used with --include-optimizer)."
    )
    args = parser.parse_args()

    print("# training benchmark")
    print_device_banner()
    print(f"# variants: {' / '.join(VARIANT_NAMES)}")

    run_correctness(
        n_iters=args.n_iters,
        lr=args.lr,
        include_optimizer=args.include_optimizer,
    )
    run_speed_table(include_optimizer=args.include_optimizer)
