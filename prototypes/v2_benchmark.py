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
    v2           : v2.capture(fn, *example_args)
                   -- AOT graph translated to C++ trace, direct replay
                      (bypasses Dynamo at call time)

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
from torch import nn
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


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block — one of the most common building
    blocks in modern LLMs:

        h = x + Attn(LayerNorm(x))
        y = h + FFN(LayerNorm(h))

    Attn is multi-head self-attention with linear Q/K/V/O projections;
    FFN is SwiGLU. Weights are held as nn.Linear / nn.LayerNorm
    parameters (the realistic shape). torch.compile and v2.capture both
    lift these parameters as graph placeholders behind the scenes;
    v2.capture's param-snapshot path freezes the weight values at
    capture time, which is the inference contract here.

    Exercises in one block:
      - 2 LayerNorms (decompose to several aten ops each)
      - 4 + 3 = 7 nn.Linear (Q/K/V/O + FFN gate/up/down)
      - 2 matmul + softmax for attention
      - silu + element-wise mul (SwiGLU)
      - 2 residual additions
    """

    def __init__(self, hidden: int, n_heads: int, ffn_inner: int):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError("hidden must be divisible by n_heads")
        self.hidden = hidden
        self.n_heads = n_heads
        self.h_dim = hidden // n_heads
        self.ln1 = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn_gate = nn.Linear(hidden, ffn_inner, bias=False)
        self.ffn_up = nn.Linear(hidden, ffn_inner, bias=False)
        self.ffn_down = nn.Linear(ffn_inner, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.h_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, self.hidden)
        x = x + self.o_proj(out)

        h2 = self.ln2(x)
        h2 = self.ffn_down(F.silu(self.ffn_gate(h2)) * self.ffn_up(h2))
        return x + h2


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


def _build_transformer_block(hidden, n_heads, ffn_inner):
    """Instantiate a TransformerBlock on DEVICE in eval mode."""
    return TransformerBlock(hidden, n_heads, ffn_inner).to(DEVICE).eval()


WORKLOADS = {
    "tiny (view+floordiv)":            (workload_tiny,      (_rand(8, 6),)),
    "pointwise (small 64x64)":         (workload_pointwise, (_rand(64, 64), _rand(64, 64))),
    "pointwise (medium 512x512)":      (workload_pointwise, (_rand(512, 512), _rand(512, 512))),
    "attention QK (B=4,S=64,H=32)":    (workload_attention, (_rand(4, 64, 32), _rand(4, 64, 32))),
    "attention QK (B=8,S=512,H=128)":  (workload_attention, (_rand(8, 512, 128), _rand(8, 512, 128))),
    "SwiGLU FFN (B=2,S=64,H=128,Hi=512)":   (workload_swiglu_ffn, _swiglu_inputs(2, 64, 128, 512)),
    "SwiGLU FFN (B=4,S=256,H=512,Hi=2048)": (workload_swiglu_ffn, _swiglu_inputs(4, 256, 512, 2048)),
    # Transformer block uses an nn.Module instance as `fn`. Weights live
    # on the module's nn.Parameter slots; Dynamo / v1 / v2 each lift
    # them into the graph in their own way (v2 snapshots them via the
    # id()-based param classification — see compile.py:_build_recipe_specs).
    "Transformer block (B=2,S=128,H=256,Hi=1024)": (
        _build_transformer_block(hidden=256, n_heads=8, ffn_inner=1024),
        (_rand(2, 128, 256),),
    ),
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

    The captured output tensor(s) are held in the closure so the timing
    loop / correctness check can read v1's result post-replay — v1's
    replay rewrites these tensors in place.
    """
    with torch.no_grad():
        with tdc.capture() as trace:
            captured_out = fn(*example_inputs)

    def cb(*_unused):
        trace.replay()
        return captured_out

    return cb


def build_variants(fn, example_inputs):
    """Return dict of {label: callable} for fn under each mode.

    Comment out a line to skip that mode entirely — the speed table and
    profile section read the dict's keys at runtime, so partial coverage
    (e.g., NPU where inductor is not available) just works without
    touching either of those functions.

    Key order here is the column order in the printed table.
    """
    return {
        "eager":     fn,
        "dynamo":    torch.compile(fn, backend="eager", dynamic=True),
        "aot_eager": torch.compile(fn, backend="aot_eager", dynamic=True),
        "inductor":  torch.compile(fn, backend="inductor", dynamic=True),
        "v1":        _v1_capture(fn, example_inputs),
        # v2 direct-replay path: capture once with example_inputs, then
        # call the resulting callable without going through Dynamo/AOT.
        "v2":        tdcv2.capture(fn, *example_inputs),
    }


# Modes that don't reuse Dynamo's compile cache between workloads — we
# reset before timing each. v1/v2 capture once and stash their own state,
# so resetting Dynamo's cache afterwards would be a no-op for them.
_CAPTURE_MODES = {"v1", "v2"}


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


# ---------------------------------------------------------------------------
# Correctness check against eager
# ---------------------------------------------------------------------------
def _flatten_output(out):
    """Flatten the return value of a workload to a list of Tensors so we
    can compare element-wise. Non-Tensor leaves are dropped — workloads
    here only return Tensor / tuple-of-Tensor / module-of-Tensor."""
    if out is None:
        return []
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        flat = []
        for o in out:
            flat.extend(_flatten_output(o))
        return flat
    return []


def _compare_outputs(ref, got, atol=5e-4, rtol=5e-4):
    """Return (ok, message). Tolerances are loose enough that inductor's
    fused/reordered kernels don't trigger false negatives on fp32 ops."""
    if len(ref) != len(got):
        return False, f"flat output count {len(got)} vs eager {len(ref)}"
    for i, (a, b) in enumerate(zip(ref, got)):
        if a.shape != b.shape:
            return False, f"out[{i}] shape {tuple(b.shape)} vs eager {tuple(a.shape)}"
        if a.dtype != b.dtype:
            return False, f"out[{i}] dtype {b.dtype} vs eager {a.dtype}"
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a.float() - b.float()).abs().max().item()
            return False, f"out[{i}] max abs diff {diff:.3e} (atol={atol})"
    return True, ""


def run_correctness_check():
    """Run each variant once and compare its output against eager.

    Aborts the benchmark on any mismatch so a timing table for broken
    captures is never printed. v1 returns the in-place-updated capture
    tensors via the closure in _v1_capture; the other variants return
    fresh tensors from the call itself."""
    print("\n# correctness check vs eager")
    print("-" * 78)
    all_ok = True
    for label, (fn, inputs) in WORKLOADS.items():
        torch._dynamo.reset()
        variants = build_variants(fn, inputs)
        if "eager" not in variants:
            print(f"  {label}: no eager variant, skipping")
            continue
        with torch.no_grad():
            ref = _flatten_output(variants["eager"](*inputs))
        # Clone so a later in-place variant (v1) can't accidentally alias
        # the reference tensors and mask a real discrepancy.
        ref = [t.detach().clone() for t in ref]

        for name, callable_ in variants.items():
            if name == "eager":
                continue
            if name not in _CAPTURE_MODES:
                torch._dynamo.reset()
            with torch.no_grad():
                got = _flatten_output(callable_(*inputs))
            ok, msg = _compare_outputs(ref, got)
            status = "ok" if ok else f"MISMATCH ({msg})"
            print(f"  {label:<44} {name:<10} {status}")
            if not ok:
                all_ok = False
    if not all_ok:
        raise RuntimeError(
            "v2_benchmark correctness check failed — see lines above")


def run_speed_table():
    """Print a per-workload x per-mode timing table.

    Column order is the insertion order of build_variants(); modes can
    be added / commented out there without touching this function.
    Ratio columns relative to 'eager' are appended for every non-eager
    mode that's present. If 'eager' is not in the variant set, the
    ratio block is omitted entirely.
    """
    # Discover the column structure once from a representative workload.
    sample_fn, sample_inputs = next(iter(WORKLOADS.values()))
    sample_variants = build_variants(sample_fn, sample_inputs)
    variant_names = list(sample_variants.keys())
    has_eager = "eager" in variant_names
    ratio_names = [n for n in variant_names if n != "eager"] if has_eager else []

    col_w = 9
    time_header = " ".join(f"{n:>{col_w}}" for n in variant_names)
    ratio_header = " ".join(f"{(n + '/eager'):>{col_w + 1}}" for n in ratio_names)
    if ratio_header:
        header = f"{'workload':<33} {time_header} | {ratio_header}"
    else:
        header = f"{'workload':<33} {time_header}"

    print(f"\n{header}")
    sub = "(times in us"
    if ratio_header:
        sub += "; ratios relative to eager)"
    else:
        sub += ")"
    print(sub)
    print("-" * len(header))

    for label, (fn, inputs) in WORKLOADS.items():
        torch._dynamo.reset()
        variants = build_variants(fn, inputs)
        # Sanity: schema is expected to be stable across workloads.
        if list(variants.keys()) != variant_names:
            raise RuntimeError(
                f"build_variants returned different keys for workload "
                f"'{label}': {list(variants.keys())} vs {variant_names}")
        times = {}
        for name, callable_ in variants.items():
            if name not in _CAPTURE_MODES:
                torch._dynamo.reset()
            times[name] = time_iters(callable_, inputs)

        time_strs = " ".join(f"{fmt_us(times[n]):>{col_w}}" for n in variant_names)
        if has_eager:
            eg = times["eager"]
            ratio_strs = " ".join(
                f"{(times[n]/eg if eg > 0 else float('nan')):>{col_w}.2f}x"
                for n in ratio_names
            )
            print(f"{label:<33} {time_strs} | {ratio_strs}")
        else:
            print(f"{label:<33} {time_strs}")


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
    """Profile the chosen workload under every mode that build_variants
    enables, exporting one Chrome-trace per mode for side-by-side
    timeline inspection in perfetto / chrome://tracing.

    File naming: <workload>__<mode>.json, e.g. `..__inductor.json`,
    `..__v1.json`. If a mode is commented out in build_variants, no
    trace is produced for it.
    """
    fn, inputs = WORKLOADS[workload_label]
    variants = build_variants(fn, inputs)
    stem = _safe_filename(workload_label)

    for name, callable_ in variants.items():
        _profile_one(name, callable_, inputs, f"{out_dir}/{stem}__{name}.json")


if __name__ == "__main__":
    print("# v2 framework benchmark")
    print_device_banner()
    # Reflect the live build_variants() output so commenting out a
    # backend (e.g. inductor on NPU) is reflected in the banner too.
    sample_fn, sample_inputs = next(iter(WORKLOADS.values()))
    _names = list(build_variants(sample_fn, sample_inputs).keys())
    print(f"# modes: {' / '.join(_names)}")
    run_correctness_check()
    run_speed_table()
    run_profile()
