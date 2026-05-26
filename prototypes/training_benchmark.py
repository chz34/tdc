"""Training-loop benchmark for v1 / v2 capture+replay.

Companion to v2_benchmark.py (inference, static shapes) and
dynamic_benchmark.py (inference, varied shapes). This file runs a
full training loop -- forward, backward, optional optimizer step --
on a small nn.Module workload across all modes, and compares the
final weight state for correctness.

The CLI mirrors torchbench's run.py:

    python training_benchmark.py <model> [-d cpu|cuda|npu]
                                         [--include-optimizer]
                                         [--n-iters N] [--lr LR]

`<model>` selects from a registry of small training-capable workloads
(MLPClassifier, TransformerClassifier; the latter wraps v2_benchmark's
TransformerBlock with a pooled classifier head). User code follows
idiomatic PyTorch -- nn.Linear / nn.LayerNorm modules with weights
stored as parameters, F.cross_entropy as the loss, torch.optim.SGD
as the optimizer.

The default mode (-t train without --include-optimizer) captures
forward + backward only and skips optimizer.step(). Add
--include-optimizer to also fold the SGD update into the timed
iteration (the update itself stays in Python, outside the captured
trace).

How each mode handles training:

  - eager / dynamo / aot_eager / inductor: drop the model into
    torch.compile (or use eager) and call as `loss = model(x); ...;
    loss.backward(); optimizer.step()`. The optimizer sees
    model.parameters() and updates them in place.

  - v1: capture the entire fw+bw inside the `tdc.capture(allow_grad=
    True)` context manager. trace.replay() re-runs every aten op
    including the AccumulateGrad add_ into .grad. (Known limitation:
    multi-Linear chain backward currently misidentifies one of the
    saved-for-backward tensor identities; see the printed STEP
    FAILED diagnostics.)

  - v2 (direct): use torch.func.functional_call to thread the
    module's parameters through as positional user-input args of a
    wrapper fn. This is needed because v2's _CapturedFn.backward
    only routes gradients for tensors that appear in user-input
    positions; module parameters lifted internally by AOT don't get
    their grads forwarded back to autograd. The user still defines
    the model normally -- the param-positional adaptation happens
    inside the variant builder.

  - v2 (wrapper) is omitted: capture() silently downgrades
    allow_grad=True to wrapper=False per compile.py's existing logic.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))
from _device import DEVICE, SYNC, print_device_banner  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch_dispatch_capture as tdc            # noqa: E402  # v1
import torch_dispatch_capture.v2 as tdcv2       # noqa: E402  # v2

# Reuse the TransformerBlock from v2_benchmark.py so workload code
# stays a single source of truth.
sys.path.insert(0, os.path.dirname(__file__))
from v2_benchmark import TransformerBlock       # noqa: E402


# ---------------------------------------------------------------------------
# Trainable nn.Module workloads
# ---------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    """Small 2-layer MLP classifier. Idiomatic style: weights are
    stored as nn.Linear parameters, optimizer iterates over
    model.parameters()."""

    def __init__(self, in_dim=64, hidden=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerClassifier(nn.Module):
    """v2_benchmark.TransformerBlock + mean-pool + classifier head.

    Uses the same TransformerBlock the inference benchmark uses, so
    the workload code is a single source of truth. We add a small
    classifier head on top for cross-entropy training."""

    def __init__(self, hidden=128, n_heads=4, ffn_inner=256, num_classes=10):
        super().__init__()
        self.block = TransformerBlock(hidden, n_heads, ffn_inner)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.block(x)
        return self.head(h.mean(dim=1))


# ---------------------------------------------------------------------------
# Workload registry
#
# Each entry knows how to build the model + a (x, y) batch. Looked
# up by name via the CLI.
# ---------------------------------------------------------------------------
def _make_mlp():
    model = MLPClassifier(in_dim=64, hidden=128, num_classes=10).to(DEVICE)
    g = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(8, 64, generator=g).to(DEVICE)
    y = torch.randint(0, 10, (8,), generator=g).to(DEVICE)
    return model, x, y


def _make_transformer():
    model = TransformerClassifier(
        hidden=128, n_heads=4, ffn_inner=256, num_classes=10,
    ).to(DEVICE)
    g = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(4, 16, 128, generator=g).to(DEVICE)
    y = torch.randint(0, 10, (4,), generator=g).to(DEVICE)
    return model, x, y


WORKLOADS = {
    "mlp": ("MLP (B=8, in=64, hidden=128, out=10)", _make_mlp),
    "transformer": (
        "Transformer (B=4, S=16, H=128, ffn=256)", _make_transformer,
    ),
}


# ---------------------------------------------------------------------------
# Variant builders
#
# Each builder takes a freshly-built model + initial batch and returns
# a step(x, y) closure that runs one training iteration on that
# variant's path. The optimizer (if any) lives in the closure.
#
# Convention: step(x, y) zeroes .grad, runs fw+bw (and optimizer step
# if include_optimizer was set at build time), and returns the loss
# value as a detached scalar so the benchmark can record convergence.
# ---------------------------------------------------------------------------
def _build_optimizer(params, lr, include_optimizer):
    if not include_optimizer:
        return None
    return torch.optim.SGD(params, lr=lr)


def build_eager(model, x_ex, y_ex, *, lr, include_optimizer):
    opt = _build_optimizer(model.parameters(), lr, include_optimizer)

    def step(x, y):
        if opt is not None:
            opt.zero_grad(set_to_none=False)
        else:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        if opt is not None:
            opt.step()
        return loss.detach()
    return step


def build_compiled(model, x_ex, y_ex, *, backend, lr, include_optimizer):
    """torch.compile-based variant (dynamo / aot_eager / inductor).

    For the training case we compile just the forward; backward is
    autograd-driven and runs on the same compiled artifact when
    backends support it (inductor / aot_eager)."""
    compiled_forward = torch.compile(model, backend=backend, dynamic=True)
    opt = _build_optimizer(model.parameters(), lr, include_optimizer)

    def step(x, y):
        if opt is not None:
            opt.zero_grad(set_to_none=False)
        else:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        loss = F.cross_entropy(compiled_forward(x), y)
        loss.backward()
        if opt is not None:
            opt.step()
        return loss.detach()
    return step


def build_v1(model, x_ex, y_ex, *, lr, include_optimizer):
    """v1 captures the entire fw+bw via the dispatcher fallback.

    Pattern from test_backward.py: warmup with one eager fw+bw to
    allocate .grad on every parameter (AccumulateGrad's first call
    bypasses dispatch); then capture; then replay re-runs everything
    including the captured grad accumulations.

    Optimizer step is applied OUTSIDE the captured trace -- v1's
    capture only covers aten ops that go through the dispatcher,
    and torch.optim's parameter update path is in Python."""
    # Warmup: eager fw+bw to allocate .grad
    loss = F.cross_entropy(model(x_ex), y_ex)
    loss.backward()
    for p in model.parameters():
        p.grad.zero_()

    # Capture
    with tdc.capture(allow_grad=True) as trace:
        captured_loss = F.cross_entropy(model(x_ex), y_ex)
        captured_loss.backward()
    for p in model.parameters():
        p.grad.zero_()

    opt = _build_optimizer(model.parameters(), lr, include_optimizer)

    def step(x, y):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        # v1 replays against the same captured input tensors; copy
        # the caller-supplied batch into them in place.
        x_ex.detach().copy_(x)
        y_ex.copy_(y)
        trace.replay()
        if opt is not None:
            opt.step()
        return captured_loss.detach().clone()
    return step


def build_v2(model, x_ex, y_ex, *, lr, include_optimizer):
    """v2 + idiomatic nn.Module training.

    v2's _CapturedFn.backward only routes gradients for tensors that
    appeared as USER-INPUT positional args at capture time -- nn.Module
    parameters that AOT lifts internally are absorbed into pre_binds
    and their bw outputs are discarded. To bridge that without
    contorting the model code, we wrap the captured fn in
    torch.func.functional_call: parameters are extracted from the
    Module and threaded through positionally, with the model code
    itself unchanged.

    .grad still lands on the same Tensor objects model.parameters()
    references, so torch.optim works against them transparently.
    """
    from torch.func import functional_call

    # Snapshot the model's parameter names + tensors. Order matters
    # because we use positional args.
    param_names = list(dict(model.named_parameters()).keys())
    params = list(model.parameters())

    def train_step(x, y, *param_list):
        p_dict = dict(zip(param_names, param_list))
        out = functional_call(model, p_dict, x)
        return F.cross_entropy(out, y)

    captured = tdcv2.capture(
        train_step, x_ex, y_ex, *params, allow_grad=True,
    )

    opt = _build_optimizer(params, lr, include_optimizer)

    def step(x, y):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss = captured(x, y, *params)
        loss.backward()
        if opt is not None:
            opt.step()
        return loss.detach()
    return step


# Variant ordering -- also the column order in the speed table.
VARIANT_NAMES = ["eager", "dynamo", "aot_eager", "inductor", "v1", "v2"]


def build_variant(name, model, x_ex, y_ex, *, lr, include_optimizer):
    """Dispatch to the right builder for one variant. Returns the
    step closure, or None on build failure (variant is skipped)."""
    try:
        if name == "eager":
            return build_eager(
                model, x_ex, y_ex, lr=lr, include_optimizer=include_optimizer)
        if name == "dynamo":
            return build_compiled(
                model, x_ex, y_ex, backend="eager",
                lr=lr, include_optimizer=include_optimizer)
        if name == "aot_eager":
            return build_compiled(
                model, x_ex, y_ex, backend="aot_eager",
                lr=lr, include_optimizer=include_optimizer)
        if name == "inductor":
            return build_compiled(
                model, x_ex, y_ex, backend="inductor",
                lr=lr, include_optimizer=include_optimizer)
        if name == "v1":
            return build_v1(
                model, x_ex, y_ex, lr=lr, include_optimizer=include_optimizer)
        if name == "v2":
            return build_v2(
                model, x_ex, y_ex, lr=lr, include_optimizer=include_optimizer)
    except Exception:
        traceback.print_exc()
        return None
    return None


def _fresh_model_and_batch(make_model_fn, seed=42):
    """Build a fresh (model, x, y) triple for one variant. Seeds the
    global torch RNG so each variant's model gets identical initial
    weights -- without this the random init for nn.Linear differs
    between variants and the correctness check sees init-noise, not
    backward-correctness. Also resets Dynamo's cache so previous
    compilations from other variants don't leak in."""
    torch._dynamo.reset()
    torch.manual_seed(seed)
    return make_model_fn()


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def _params_close(model_ref, model_var, atol=1e-4, rtol=1e-4):
    ref_params = dict(model_ref.named_parameters())
    var_params = dict(model_var.named_parameters())
    if set(ref_params) != set(var_params):
        return False, (
            f"param-name set differs: ref - var = "
            f"{set(ref_params) - set(var_params)}, "
            f"var - ref = {set(var_params) - set(ref_params)}"
        )
    for name in ref_params:
        a = ref_params[name].detach()
        b = var_params[name].detach()
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a - b).abs().max().item()
            return False, (
                f"{name} max abs diff {diff:.3e} "
                f"(shape={tuple(a.shape)}, atol={atol})"
            )
    return True, ""


def run_correctness(model_key, *, n_iters, lr, include_optimizer):
    label, make_model = WORKLOADS[model_key]
    print(f"\n## correctness: {label}")
    print(f"#  n_iters={n_iters}, lr={lr}, include_optimizer={include_optimizer}")

    # Eager reference
    model_ref, x, y = _fresh_model_and_batch(make_model)
    ref_step = build_variant(
        "eager", model_ref, x.clone(), y.clone(),
        lr=lr, include_optimizer=include_optimizer)
    ref_losses = []
    for _ in range(n_iters):
        loss = ref_step(x, y)
        SYNC()
        ref_losses.append(loss.item())
    print("   eager final loss: {:.4f}  (path: {})".format(
        ref_losses[-1], " -> ".join(f"{l:.3f}" for l in ref_losses)))

    # Each variant gets an independent model with the same init seed.
    for name in VARIANT_NAMES:
        if name == "eager":
            continue
        model_var, x_v, y_v = _fresh_model_and_batch(make_model)
        step = build_variant(
            name, model_var, x_v.clone(), y_v.clone(),
            lr=lr, include_optimizer=include_optimizer)
        if step is None:
            print(f"   {name:<11} BUILD FAILED")
            continue
        try:
            losses = []
            for _ in range(n_iters):
                loss = step(x, y)
                SYNC()
                losses.append(loss.item())
        except Exception as e:
            print(f"   {name:<11} STEP FAILED  ({type(e).__name__}: "
                  f"{str(e)[:120]})")
            continue
        ok, msg = _params_close(model_ref, model_var)
        loss_match = abs(losses[-1] - ref_losses[-1]) < max(
            1e-4, abs(ref_losses[-1]) * 1e-3
        )
        status = "ok" if (ok and loss_match) else "MISMATCH"
        detail = msg if not ok else (
            f"(loss diff |{losses[-1]:.4f} - {ref_losses[-1]:.4f}|)"
            if not loss_match else ""
        )
        print(f"   {name:<11} {status:<10} final_loss={losses[-1]:.4f}  {detail}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _time_one(step_fn, x, y, *, n_warmup=5, n_iters=50):
    try:
        for _ in range(n_warmup):
            step_fn(x, y)
            SYNC()
    except Exception:
        traceback.print_exc()
        return None
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        try:
            step_fn(x, y)
        except Exception:
            traceback.print_exc()
            return None
        SYNC()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def run_speed(model_key, *, lr, include_optimizer):
    label, make_model = WORKLOADS[model_key]

    times = {}
    for name in VARIANT_NAMES:
        model, x, y = _fresh_model_and_batch(make_model)
        step = build_variant(
            name, model, x.clone(), y.clone(),
            lr=lr, include_optimizer=include_optimizer)
        times[name] = None if step is None else _time_one(step, x, y)

    col_w = 9
    print(f"\n## per-iteration training speed: {label} (median, us)")
    sub = "(includes optimizer step)" if include_optimizer else "(fw+bw only)"
    print(f"#  {sub}")
    header_cols = " ".join(f"{n:>{col_w}}" for n in VARIANT_NAMES)
    ratio_cols = " ".join(
        f"{(n + '/eager'):>{col_w + 1}}"
        for n in VARIANT_NAMES if n != "eager"
    )
    print(f"   {header_cols} | {ratio_cols}")

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
    print(f"   {time_strs} | {ratio_strs}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "model", nargs="?", default=None,
        choices=list(WORKLOADS.keys()),
        help=(
            "Workload name. If omitted, run all registered workloads in turn. "
            "Available: " + ", ".join(WORKLOADS.keys()) + "."
        ),
    )
    parser.add_argument(
        "-d", "--device", default=None,
        help="Device override (cpu/cuda/npu/...). Default reads "
             "TDC_DEVICE env var (currently: " + str(DEVICE) + ").",
    )
    parser.add_argument(
        "--include-optimizer", action="store_true",
        help="Run torch.optim.SGD.step() inside the timed iteration. "
             "Default off -- the benchmark covers fw+bw only.",
    )
    parser.add_argument(
        "--n-iters", type=int, default=5,
        help="Training iterations for the correctness check.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2,
        help="SGD learning rate (only used with --include-optimizer).",
    )
    args = parser.parse_args()

    if args.device:
        # Respect CLI override by overwriting the env var the _device
        # helper read. This module's DEVICE / SYNC were resolved at
        # import time, so this only affects newly-spawned subprocesses
        # / future imports; print a note and require env-based setup.
        print(f"# Note: -d {args.device} sets TDC_DEVICE for child "
              f"processes only. Re-run with TDC_DEVICE={args.device} "
              "to change the runtime device.")

    print("# training benchmark")
    print_device_banner()
    print(f"# variants: {' / '.join(VARIANT_NAMES)}")

    models_to_run = [args.model] if args.model else list(WORKLOADS.keys())
    for key in models_to_run:
        run_correctness(
            key, n_iters=args.n_iters, lr=args.lr,
            include_optimizer=args.include_optimizer,
        )
        run_speed(
            key, lr=args.lr,
            include_optimizer=args.include_optimizer,
        )


if __name__ == "__main__":
    main()
