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

Every mode runs `optimizer.zero_grad() / forward / loss.backward() /
optimizer.step()` -- a real training step where weights actually
move and loss drops between iterations.

`--include-optimizer` controls whether the optimizer.step() is
folded into the captured/compiled artifact (where supported), or
runs in normal eager Python outside the trace:

  - default (no --include-optimizer): optimizer.step() runs in
    eager Python after each replay. v1's trace covers fw+bw;
    v2's captured fn covers fw+bw; the weight update is a
    separate Python-side call that mutates parameters via
    aten::add_ as usual.

  - --include-optimizer: where possible, fold optimizer.step()'s
    aten ops into the capture so replay does the whole step in
    one dispatch sequence.
        v1: opt.step() is called INSIDE the
            `with tdc.capture(allow_grad=True)` block, so the SGD
            add_ ops are recorded as Steps; replay applies them.
        v2: not currently supported -- v2's autograd.Function
            wrapper only owns fw + bw graphs; folding the
            optimizer would need a different capture API. v2
            falls back to eager-step behaviour and prints a notice.
        eager / dynamo / aot_eager / inductor: no-op (no "capture"
            concept beyond what each compile backend already does).

How each mode handles training:

  - eager / dynamo / aot_eager / inductor: idiomatic
    `optimizer.step()` after backward. dynamo / aot_eager /
    inductor compile the forward (and backward via autograd), but
    optimizer always runs in Python.

  - v1: capture fw+bw (or fw+bw+step with --include-optimizer)
    inside the `tdc.capture(allow_grad=True)` context manager.
    (Known limitation: multi-Linear chain backward currently
    misidentifies one of the saved-for-backward tensor identities;
    the printed STEP FAILED diagnostics make this visible.)

  - v2 (direct): use torch.func.functional_call to thread the
    module's parameters through as positional user-input args of a
    wrapper fn. v2's _CapturedFn.backward only routes gradients
    for tensors that appear in user-input positions; functional_call
    bridges that gap without forcing the model code to be functional.

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


# ---------------------------------------------------------------------------
# torchbench integration (opt-in via TDC_TORCHBENCH=1)
#
# Mirrors v2_benchmark._load_torchbench's pattern: import the model's
# torchbenchmark.models.<name> module, construct its Model wrapper with
# test='train', and surface the bench as the workload's "model". The
# variant builders detect spec.is_torchbench and dispatch to bench's
# own forward()/backward()/optimizer (bench.forward() returns the loss
# tensor directly).
# ---------------------------------------------------------------------------
def _load_torchbench_train(name: str, batch_size: int):
    """Load a torchbench model in train mode. Returns the bench object
    on success or None on any failure (missing dep, OOM on cpu, etc.)
    so the rest of the suite isn't blocked."""
    try:
        import importlib
        mod = importlib.import_module(f"torchbenchmark.models.{name}")
        bench = mod.Model(
            test="train", device=DEVICE.type, batch_size=batch_size,
        )
        bench.model.train()
        return bench
    except Exception as e:
        print(f"# torchbench: skipping {name!r} train "
              f"({type(e).__name__}: {str(e)[:160]})")
        return None


class WorkloadSpec:
    """Captures everything a variant builder needs for one workload.

    Two flavours, kept in one struct so the iterator in run_correctness
    / run_speed stays uniform:

      - is_torchbench=False: make_model() returns (nn.Module, x, y)
        where the variant builder runs `F.cross_entropy(model(x), y)`
        as the loss.
      - is_torchbench=True: make_model() returns (bench, None, None)
        where the variant builder runs `bench.forward()` for the
        loss. x, y are sentinel None values; we never pass them to
        anything.
    """
    def __init__(self, name, label, make_model, *, is_torchbench=False):
        self.name = name
        self.label = label
        self.make_model = make_model
        self.is_torchbench = is_torchbench


WORKLOADS = {
    "mlp": WorkloadSpec(
        name="mlp",
        label="MLP (B=8, in=64, hidden=128, out=10)",
        make_model=_make_mlp,
    ),
    "transformer": WorkloadSpec(
        name="transformer",
        label="Transformer (B=4, S=16, H=128, ffn=256)",
        make_model=_make_transformer,
    ),
}


def _bench_lazy_loader(name, bs):
    """Returns a factory that loads the bench on every call. Each
    variant gets its own bench instance so optimizer/state don't
    cross-contaminate."""
    def factory():
        bench = _load_torchbench_train(name, bs)
        return bench, None, None
    return factory


if os.environ.get("TDC_TORCHBENCH", "0") == "1":
    # Same model list as v2_benchmark.py's torchbench rotation; small
    # batch sizes because training is slower than inference and our
    # n_iters loop sees each batch many times.
    for _tb_name, _bs in [
        ("squeezenet1_1", 8),
        ("alexnet",       8),
        ("BERT_pytorch",  2),
        ("hf_GPT2",       2),
        ("timm_vision_transformer", 4),
    ]:
        # Probe-load once at import time so unavailable models don't
        # populate the registry. The variant builders re-load to get
        # an independent bench per variant.
        _probe = _load_torchbench_train(_tb_name, _bs)
        if _probe is None:
            continue
        del _probe   # discard; per-variant builders will load their own
        _key = f"torchbench:{_tb_name}"
        WORKLOADS[_key] = WorkloadSpec(
            name=_key,
            label=f"torchbench:{_tb_name} (B={_bs}, train)",
            make_model=_bench_lazy_loader(_tb_name, _bs),
            is_torchbench=True,
        )


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
def _real_model(spec, model_or_bench):
    """Resolve to the actual nn.Module. For in-house workloads
    spec.is_torchbench=False and we get the model directly; for
    torchbench we unwrap bench.model."""
    return model_or_bench.model if spec.is_torchbench else model_or_bench


def _compute_loss(spec, model_or_bench, x, y):
    """Run forward+loss under the workload's convention. For
    in-house: F.cross_entropy(model(x), y). For torchbench:
    bench.forward() (which uses bench.example_inputs internally and
    returns the loss tensor directly)."""
    if spec.is_torchbench:
        return model_or_bench.forward()
    return F.cross_entropy(model_or_bench(x), y)


def _snapshot_params(spec, model_or_bench):
    """Return a dict of cloned parameter + buffer tensors -- used to
    reset every variant back to the same initial state after any
    capture-time warmup or example-call mutations.

    Buffers must be snapshotted alongside params because some variants
    (notably v2, which runs an example forward inside `capture()`)
    advance stateful buffers like BatchNorm's running_mean/running_var
    before the timed loop starts. Without resetting buffers the variant
    enters its first step "1 forward ahead" of the eager reference and
    the loss comparison sees BN-state drift rather than backward
    correctness.
    """
    m = _real_model(spec, model_or_bench)
    snap = {f"P/{n}": p.detach().clone() for n, p in m.named_parameters()}
    for n, b in m.named_buffers():
        snap[f"B/{n}"] = b.detach().clone()
    return snap


def _restore_params(spec, model_or_bench, snapshot):
    """In-place copy snapshot values back into the model's parameters
    AND buffers, then clear .grad. Identity is preserved so any captured
    trace that references model state by TensorImpl identity keeps
    working against the reset values.

    Strips a leading `_orig_mod.` from the param/buffer name if the
    direct lookup misses: build_compiled monkey-patches `bench.model =
    torch.compile(real, ...)` for torchbench, which wraps state under
    that prefix in named_parameters() / named_buffers(). The canonical
    snapshot was taken from the pre-wrap eager model so its keys don't
    carry the prefix.
    """
    PREFIX = "_orig_mod."

    def _lookup(kind, name):
        key = f"{kind}/{name}"
        if key in snapshot:
            return snapshot[key]
        if name.startswith(PREFIX):
            stripped = f"{kind}/{name[len(PREFIX):]}"
            if stripped in snapshot:
                return snapshot[stripped]
        raise KeyError(
            f"_restore_params: no snapshot entry for {kind}/{name!r}. "
            f"snapshot keys (first 5): {list(snapshot.keys())[:5]}")

    m = _real_model(spec, model_or_bench)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(_lookup("P", n))
        for n, b in m.named_buffers():
            b.copy_(_lookup("B", n))
        for p in m.parameters():
            p.grad = None


def build_eager(spec, model, x_ex, y_ex, *, lr, include_optimizer):
    # Eager has no "capture", so --include-optimizer is a no-op here:
    # the optimizer always runs in Python.
    opt = torch.optim.SGD(_real_model(spec, model).parameters(), lr=lr)

    def step(x, y):
        opt.zero_grad()
        loss = _compute_loss(spec, model, x, y)
        loss.backward()
        opt.step()
        return loss.detach()
    return step


def build_compiled(spec, model, x_ex, y_ex, *, backend, lr, include_optimizer):
    """torch.compile-based variant (dynamo / aot_eager / inductor).

    Compile the forward; backward is autograd-driven, optimizer
    runs in eager Python. --include-optimizer has no effect on
    these modes (would require torch.compile(opt.step) which is a
    separate gesture).

    For torchbench: monkey-patch bench.model with a compiled wrapper
    so bench.forward()'s internal `self.model(**self.example_inputs)`
    call hits the compiled path. This is the standard pattern
    torchbench itself uses for its --torchdynamo runs."""
    real = _real_model(spec, model)
    if spec.is_torchbench:
        model.model = torch.compile(real, backend=backend, dynamic=True)
        callable_for_loss = model  # bench, whose .forward() runs compiled
    else:
        callable_for_loss = torch.compile(model, backend=backend, dynamic=True)
    opt = torch.optim.SGD(real.parameters(), lr=lr)

    def step(x, y):
        opt.zero_grad()
        loss = _compute_loss(spec, callable_for_loss, x, y)
        loss.backward()
        opt.step()
        return loss.detach()
    return step


def build_v1(spec, model, x_ex, y_ex, *, lr, include_optimizer):
    """v1 captures fw+bw (and optimizer.step() under
    --include-optimizer) via the dispatcher fallback.

    For torchbench workloads we capture `bench.forward() +
    bench_loss.backward()` inside the context manager. bench.forward()
    internally calls bench.model(**bench.example_inputs), so the
    dispatcher fallback sees the full forward + backward op sequence."""
    real = _real_model(spec, model)
    opt = torch.optim.SGD(real.parameters(), lr=lr)

    # Warmup: eager fw+bw+step. Needed so AccumulateGrad's special-
    # first-call path doesn't appear inside the capture.
    opt.zero_grad()
    loss = _compute_loss(spec, model, x_ex, y_ex)
    loss.backward()
    opt.step()
    opt.zero_grad()

    if include_optimizer:
        # Capture fw + bw + optimizer.step() -- the SGD update's
        # aten::add_ on each parameter is dispatched and recorded.
        with tdc.capture(allow_grad=True) as trace:
            captured_loss = _compute_loss(spec, model, x_ex, y_ex)
            captured_loss.backward()
            opt.step()
    else:
        with tdc.capture(allow_grad=True) as trace:
            captured_loss = _compute_loss(spec, model, x_ex, y_ex)
            captured_loss.backward()
    opt.zero_grad()

    def step(x, y):
        opt.zero_grad()
        if not spec.is_torchbench:
            # v1 replays against the captured input tensors; copy
            # the caller-supplied batch into them in place. For
            # torchbench, bench.example_inputs lives inside the
            # bench object -- the user batch isn't separately fed.
            x_ex.detach().copy_(x)
            y_ex.copy_(y)
        trace.replay()
        if not include_optimizer:
            opt.step()
        return captured_loss.detach().clone()
    return step


def build_v2(spec, model, x_ex, y_ex, *, lr, include_optimizer):
    """v2 + idiomatic nn.Module training.

    Uses the natural closure form: v2.capture detects Parameters lifted
    by AOT and routes their backward grads to .grad through autograd's
    accumulator path (aot_eager-style). No functional_call needed.

    --include-optimizer is not folded into the captured trace -- the
    autograd.Function wrapper only owns fw + bw graphs, optimizer.step
    runs in eager. opt.step's in-place .data mutation IS visible to the
    next captured replay (v2 holds the Parameter object in
    captured_tensors_, not a snapshot), so multi-iteration training
    converges identically to eager.

    For torchbench: capture `bench.forward` directly. It's a zero-arg
    method that closure-references bench.model + bench.example_inputs.
    Dynamo lifts both as graph inputs; parameters route through the
    autograd-leaf path, example_inputs Tensors stay pre-bound (same
    batch per replay, which matches torchbench's benchmark loop
    semantics anyway). Capture is wrapped in try/except so individual
    bench wrappers that graph-break (HF amp_context, exotic output
    dataclasses, etc.) fail loud but don't crash the benchmark.
    """
    if include_optimizer:
        print("   v2          NOTICE     --include-optimizer not folded "
              "into the captured trace; running optimizer in eager "
              "(in-place updates ARE reflected in the next replay)")

    real = _real_model(spec, model)

    if spec.is_torchbench:
        # bench.forward() is zero-arg; bench.example_inputs and
        # bench.model are both closure-captured. Capture with no
        # positional args -- v2 will lift everything via Dynamo +
        # the new param-leaf routing.
        try:
            captured = tdcv2.capture(model.forward, allow_grad=True)
        except Exception as e:
            print(f"   v2          CAPTURE FAILED  "
                  f"({type(e).__name__}: {str(e).splitlines()[0][:140]})")
            return None
        for p in real.parameters():
            p.grad = None
        opt = torch.optim.SGD(real.parameters(), lr=lr)

        def step(x, y):
            # bench owns its own inputs; ignore the caller-supplied
            # x, y (mirrors build_v1's torchbench branch).
            opt.zero_grad()
            loss = captured()
            loss.backward()
            opt.step()
            return loss.detach()
        return step

    # In-house path: closure form with x, y as positional args.
    def train_step(x, y):
        return F.cross_entropy(model(x), y)
    try:
        captured = tdcv2.capture(train_step, x_ex, y_ex, allow_grad=True)
    except Exception as e:
        print(f"   v2          CAPTURE FAILED  "
              f"({type(e).__name__}: {str(e).splitlines()[0][:140]})")
        return None
    # capture-time example call already ran a .backward(); clear
    # the accumulated grads so the first timed step starts clean.
    for p in real.parameters():
        p.grad = None

    opt = torch.optim.SGD(real.parameters(), lr=lr)

    def step(x, y):
        opt.zero_grad()
        loss = captured(x, y)
        loss.backward()
        opt.step()
        return loss.detach()
    return step


# Variant registry. Each entry is (name, builder) where builder has
# the signature (model, x_ex, y_ex, *, lr, include_optimizer) -> step.
# Order doubles as the column order in the speed table. The three
# compile-backend variants share build_compiled via functools.partial
# so adding a new backend (or a new variant entirely) is a single
# line here -- no separate dispatch function to update.
from functools import partial

VARIANTS = [
    ("eager",     build_eager),
    ("dynamo",    partial(build_compiled, backend="eager")),
    ("aot_eager", partial(build_compiled, backend="aot_eager")),
    ("inductor",  partial(build_compiled, backend="inductor")),
    ("v1",        build_v1),
    ("v2",        build_v2),
]


def _safe_build(builder, spec, model, x_ex, y_ex, *, lr, include_optimizer):
    """Invoke a variant builder, printing the traceback and returning
    None on failure so the rest of the run can continue."""
    try:
        return builder(
            spec, model, x_ex, y_ex,
            lr=lr, include_optimizer=include_optimizer)
    except Exception:
        traceback.print_exc()
        return None


def _fresh_model_and_batch(spec, seed=42):
    """Build a fresh (model_or_bench, x, y) triple for one variant.
    Seeds the global torch RNG so each variant's model gets identical
    initial weights -- without this the random init for nn.Linear
    differs between variants and the correctness check sees
    init-noise, not backward-correctness. Also resets Dynamo's cache
    so previous compilations from other variants don't leak in.

    For torchbench specs, make_model() loads a fresh bench instance
    and returns (bench, None, None). The bench's own init pulls
    weights from torchbench's stored checkpoints (not from the
    torch.manual_seed), so two variants of the same torchbench
    workload start identical by construction."""
    torch._dynamo.reset()
    torch.manual_seed(seed)
    return spec.make_model()


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def _params_close(model_ref, model_var, atol=1e-4, rtol=1e-4):
    # Strip a leading "_orig_mod." from var keys: build_compiled
    # monkey-patches bench.model with torch.compile, which wraps params
    # under that prefix in named_parameters(). The pre-wrap eager model
    # has no such prefix, so a direct set comparison would always fail
    # for the dynamo/aot_eager/inductor variants on torchbench.
    def _strip(n):
        prefix = "_orig_mod."
        return n[len(prefix):] if n.startswith(prefix) else n

    ref_params = {n: p for n, p in model_ref.named_parameters()}
    var_params = {_strip(n): p for n, p in model_var.named_parameters()}
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
    spec = WORKLOADS[model_key]
    print(f"\n## correctness: {spec.label}")
    print(f"#  n_iters={n_iters}, lr={lr}, include_optimizer={include_optimizer}")

    # One canonical initial snapshot drives every variant -- this way
    # variants whose build dirties weights (v1's warmup + capture-time
    # example call, v2's example call's .backward) get reset to the
    # same initial state before the timed loop starts.
    ref_obj, _, _ = _fresh_model_and_batch(spec)
    init_snapshot = _snapshot_params(spec, ref_obj)

    eager_builder = dict(VARIANTS)["eager"]
    ref_obj, x, y = _fresh_model_and_batch(spec)
    _restore_params(spec, ref_obj, init_snapshot)
    ref_step = _safe_build(
        eager_builder, spec, ref_obj,
        x.clone() if x is not None else None,
        y.clone() if y is not None else None,
        lr=lr, include_optimizer=include_optimizer)
    _restore_params(spec, ref_obj, init_snapshot)
    ref_losses = []
    for _ in range(n_iters):
        loss = ref_step(x, y)
        SYNC()
        ref_losses.append(loss.item())
    print("   eager final loss: {:.4f}  (path: {})".format(
        ref_losses[-1], " -> ".join(f"{l:.3f}" for l in ref_losses)))

    for name, builder in VARIANTS:
        if name == "eager":
            continue
        var_obj, x_v, y_v = _fresh_model_and_batch(spec)
        step = _safe_build(
            builder, spec, var_obj,
            x_v.clone() if x_v is not None else None,
            y_v.clone() if y_v is not None else None,
            lr=lr, include_optimizer=include_optimizer)
        if step is None:
            print(f"   {name:<11} BUILD FAILED")
            continue
        # Reset weights AFTER build (v1's warmup / v2's example call
        # both mutate the model). Captured traces keep their TensorImpl
        # references; in-place .copy_ preserves identity.
        _restore_params(spec, var_obj, init_snapshot)
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
        ok, msg = _params_close(
            _real_model(spec, ref_obj), _real_model(spec, var_obj),
        )
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
    spec = WORKLOADS[model_key]

    # Use the same canonical snapshot every variant resets to so
    # timing isn't biased by post-build dirty weight state.
    ref_obj, _, _ = _fresh_model_and_batch(spec)
    init_snapshot = _snapshot_params(spec, ref_obj)

    times = {}
    for name, builder in VARIANTS:
        obj, x, y = _fresh_model_and_batch(spec)
        step = _safe_build(
            builder, spec, obj,
            x.clone() if x is not None else None,
            y.clone() if y is not None else None,
            lr=lr, include_optimizer=include_optimizer)
        if step is not None:
            _restore_params(spec, obj, init_snapshot)
        times[name] = None if step is None else _time_one(step, x, y)

    col_w = 9
    names = [n for n, _ in VARIANTS]
    print(f"\n## per-iteration training speed: {spec.label} (median, us)")
    sub = (
        "(optimizer.step() in the captured trace; v1 only)"
        if include_optimizer
        else "(optimizer.step() in eager after captured fw+bw replay)"
    )
    print(f"#  {sub}")
    header_cols = " ".join(f"{n:>{col_w}}" for n in names)
    ratio_cols = " ".join(
        f"{(n + '/eager'):>{col_w + 1}}" for n in names if n != "eager"
    )
    print(f"   {header_cols} | {ratio_cols}")

    def cell(t):
        return "     N/A" if t is None else f"{t:8.2f}"
    time_strs = " ".join(f"{cell(times[n]):>{col_w}}" for n in names)
    eg = times["eager"]
    def ratio(n):
        t = times[n]
        if t is None or eg is None or eg <= 0:
            return "    N/A "
        return f"{(t/eg):>{col_w}.2f}x"
    ratio_strs = " ".join(
        f"{ratio(n):>{col_w}}" for n in names if n != "eager"
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
        help=(
            "Fold optimizer.step() into the captured trace (v1 only). "
            "Default off: optimizer.step() runs in normal eager Python "
            "after the captured fw+bw replay -- weights still update "
            "and loss drops between iterations. Setting this flag puts "
            "the SGD add_ ops INSIDE the capture window so replay does "
            "the entire training step in one dispatch sequence. v2's "
            "current autograd.Function path cannot fold the optimizer "
            "and will print a notice + run optimizer in eager."
        ),
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
    print(f"# variants: {' / '.join(n for n, _ in VARIANTS)}")

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
