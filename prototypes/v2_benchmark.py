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
    v2 (direct)  : v2.capture(fn, *example_args, wrapper=False)
                   -- AOT graph translated to C++ trace, bare direct_replay
                      (fastest, no AOT RuntimeWrapper overhead)
    v2 (wrapper) : v2.capture(fn, *example_args, wrapper=True)
                   -- direct_replay wrapped in aot_function for full eager
                      semantics (input mutations, output alias rebuild)

Device is controlled by TDC_DEVICE (default cpu). When running on an
accelerator, tensors are allocated on DEVICE and each timed call is
wrapped in SYNC() so wall-clock measurements reflect kernel completion,
not just dispatch enqueueing.
"""
import argparse
import io
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tv_models
import torch_dispatch_capture as tdc           # v1
import torch_dispatch_capture.v2 as tdcv2      # v2
import torch_dispatch_capture.v3 as tdcv3      # v3
from torch.profiler import profile, ProfilerActivity

# Share the test suite's device helper. test/ is a sibling of prototypes/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))
from _device import DEVICE, SYNC, print_device_banner  # noqa: E402


# ---------------------------------------------------------------------------
# Triton softmax kernel + torch.library.triton_op registration.
# Mirrors test/test_v2_triton.py: defines a row-wise softmax kernel and
# registers it as a dispatcher-visible op so v1/v2 can capture/replay it
# like any other OpOverload. The op appears as a single call_function
# node in the FX graph (no decomposition), so the translator emits one
# kTensorOp Step per kernel call -- dispatch overhead is the same whether
# the kernel is aten or triton.
#
# Sizes are chosen so one block covers the whole row (n_cols <= 1024),
# avoiding multi-block reductions. 512 -> BLOCK_SIZE=512, 1024 ->
# BLOCK_SIZE=1024.
# ---------------------------------------------------------------------------
def _detect_triton_available():
    try:
        import triton
    except ImportError as e:
        return False, f"triton not installed ({e})"
    try:
        target = triton.runtime.driver.active.get_current_target()
        return True, f"target={target}"
    except Exception as e:
        return False, (
            f"triton has no active driver for {DEVICE.type} "
            f"({type(e).__name__}: {str(e)[:80]})"
        )


TRITON_AVAILABLE, TRITON_REASON = _detect_triton_available()

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _softmax_kernel(
        x_ptr, output_ptr, n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = row * n_cols + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
        x_max = tl.max(x, axis=0)
        x = x - x_max
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(output_ptr + offsets, num / den, mask=mask)

    @torch.library.triton_op("tdc_bench::softmax_triton", mutates_args={})
    def softmax_triton(x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        torch.library.wrap_triton(_softmax_kernel)[grid](
            x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

    @softmax_triton.register_fake
    def _(x):
        return torch.empty_like(x)

    def workload_triton_softmax(x):
        """Triton softmax -> scale + bias (attention post-processing)."""
        return softmax_triton(x) * 1.5 + 0.1

else:
    softmax_triton = None           # type: ignore[assignment]
    workload_triton_softmax = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Workloads -- three sizes, intentionally CPU-only so timing is reproducible.
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


class LlamaAttentionWithKVCache(nn.Module):
    """Self-attention block lifted from torchbench's LLaMA reference
    (torchbenchmark/models/llama/model.py:Attention), reduced to the
    KV-cache-relevant essentials so the workload exercises the exact
    pattern v2's slice_scatter rewrite targets.

    Each forward:
      1. Q/K/V projection
      2. ★ KV cache write:
            self.cache_k[:bsz, start_pos:start_pos+seqlen] = xk
            self.cache_v[:bsz, start_pos:start_pos+seqlen] = xv
         AOT functionalize rewrites this as
            slice + copy + slice_scatter + (copy_ writeback)
         which without our rewrite would alloc + memcpy a fresh
         [max_batch, max_seq, n_heads, head_dim] tensor every step.
      3. ★ KV cache read for the current context:
            keys   = self.cache_k[:bsz, : start_pos+seqlen]
            values = self.cache_v[:bsz, : start_pos+seqlen]
         Dynamic-shape slice that grows with start_pos.
      4. Scaled dot-product attention (no mask; the upstream model
         also disables mask for shape compatibility), then output
         projection.

    Differences from the upstream Attention (intentional, to keep
    the workload focused on cache-pattern perf):
      - No rotary embedding (apply_rotary_emb pulls in complex tensor
        ops that aren't on the hot path we're measuring).
      - No type_as round-trip in softmax (this is an fp32 path here).
      - cache_k/cache_v as register_buffer instead of plain attrs
        so dynamo/aot lift them via the buffer-input path.

    What the workload validates:
      - Correctness of the slice_scatter -> slice + copy_ rewrite.
      - End-to-end speedup on a workload dominated by the rewritten
        pattern (the per-step cost ratio of cache update vs matmul
        is fairly close, so the rewrite should be visible).
    """

    def __init__(self, dim: int, n_heads: int, max_batch: int, max_seq: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.register_buffer(
            "cache_k", torch.zeros(max_batch, max_seq, n_heads, self.head_dim)
        )
        self.register_buffer(
            "cache_v", torch.zeros(max_batch, max_seq, n_heads, self.head_dim)
        )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        # KV cache update.
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # KV cache read.
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # Scaled dot-product attention.
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) * self.scale
        scores = torch.softmax(scores, dim=-1)
        out = torch.matmul(scores, values)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(out)


def _rand(*shape):
    return torch.randn(*shape, device=DEVICE)


# ---------------------------------------------------------------------------
# torchbench models
# ---------------------------------------------------------------------------
def _build_alexnet():
    """AlexNet (torchvision) with ImageNet weights, eval mode."""
    return tv_models.alexnet(
        weights=tv_models.AlexNet_Weights.IMAGENET1K_V1
    ).to(DEVICE).eval()


def _load_torchbench(name: str, batch_size=None, test: str = "eval"):
    """Load a torchbench model via its standardised Model API. Returns
    (fn, example_inputs) on success or None if anything goes wrong
    (torchbench not installed, weight-download failure, model init
    error, incompatibility with the local torch version, ...). Failures
    are logged but never raise — the workload is just skipped.

    Uses `bench.eval` as the captured callable -- the same zero-arg
    entry point torchbench's own benchmarks use. All three framework
    bases (HuggingFaceModel / TimmModel / TorchVisionModel) expose
    eval() with the same signature: `eval(self) -> Tuple[Tensor]`
    (vision returns the raw Tensor for some models). The method
    handles the kwarg expansion (model(**self.example_inputs)) and
    output post-processing (HF strips .logits) internally, so we
    don't need to thread the dict/tuple/kwarg distinction through
    our own pipeline. Returning `(bench.eval, ())` keeps the rest
    of the benchmark loop's `fn(*inputs)` convention intact.

    Trade-off: inputs are baked into `bench.example_inputs` at
    construction; we can't replace them per call (which the dynamic-
    shape benchmark doesn't need anyway).

    Expects the `torchbenchmark` package on sys.path (e.g. installed
    from the pytorch/benchmark repo via `pip install .`)."""
    try:
        import importlib
        mod = importlib.import_module(f"torchbenchmark.models.{name}")
        kwargs: dict = {"test": test, "device": DEVICE.type}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        bench = mod.Model(**kwargs)
        # bench.eval is a bound method; capture/replay paths only call
        # it as fn(); v2's closure-scan / Dynamo tracing follows
        # __self__ into bench's .model attribute to lift parameters.
        return bench.eval, ()
    except Exception as e:
        print(f"# torchbench: skipping {name!r} ({type(e).__name__}: {str(e)[:160]})")
        return None


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


def _build_llama_attention(dim, n_heads, max_batch, max_seq):
    """Instantiate a LlamaAttentionWithKVCache on DEVICE in eval mode."""
    return LlamaAttentionWithKVCache(
        dim, n_heads, max_batch, max_seq
    ).to(DEVICE).eval()


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
    # LLaMA-style attention with KV cache write+read. Specifically
    # crafted to exercise the slice_scatter rewrite path (see
    # compile.py:_rewrite_slice_scatter_to_inplace and DESIGN.md
    # §17.6.9). The cache is sized [max_batch=4, max_seq=128, ...]
    # = ~1MB so the rewrite's payoff is measurable but the workload
    # stays quick. start_pos=4 leaves room for the read slice
    # cache_k[:, :start_pos+seqlen] to grow with each retrace.
    "LLaMA attn KV (B=2,S=32,dim=256,maxS=128)": (
        _build_llama_attention(dim=256, n_heads=8, max_batch=4, max_seq=128),
        (_rand(2, 32, 256), 4),
    ),
    # torchvision-direct: weights frozen at capture via v2.capture's
    # param-snapshot path.
    "alexnet (B=64)": (
        _build_alexnet(),
        (_rand(64, 3, 224, 224),),
    ),
}
# Triton softmax workloads: exercised only when TRITON_AVAILABLE is True
# (GPU required). Each entry runs a triton softmax kernel followed by a
# small aten post-processing chain, comparing eager vs dynamo/aot_eager/
# inductor/v1/v2. The triton_op appears as a single opaque OpOverload in
# the graph, so the dispatch-to-compute ratio is high -- v2's C++ replay
# should show the largest relative speedup here.
if TRITON_AVAILABLE:
    WORKLOADS.update({
        "triton softmax (512x512)": (
            workload_triton_softmax,
            (_rand(512, 512),),
        ),
        "triton softmax (1024x1024)": (
            workload_triton_softmax,
            (_rand(1024, 1024),),
        ),
    })


# ---------------------------------------------------------------------------
# Append torchbench-sourced workloads, best-effort.
#
# Off by default — these are full models (e.g. BERT_pytorch is ~116M
# params) and running them through the eager / dynamo / inductor /
# v1 / v2 matrix with the full warmup+iter budget makes a CPU run take
# many minutes. Set TDC_TORCHBENCH=1 to enable; on accelerators they're
# usually fast enough to keep on by default but we still gate on the
# env var to keep CI predictable.
#
# Each entry calls _load_torchbench(); if the model can't load (e.g.
# weight-download hash failure, missing torchbench checkout, hitting a
# v2 unsupported FX op), we skip it instead of failing the whole
# benchmark. Models listed in _TB_SKIP_CORRECTNESS still participate
# in timing but are excluded from the eager-vs-everything diff. The
# set is empty by default — v2.capture now reshapes its raw trace
# outputs back to the user-visible structure (compile.py's
# _build_output_shaper), so AOT's extra saved-for-backward outputs
# no longer leak through and BERT_pytorch passes correctness.
# ---------------------------------------------------------------------------
_TB_SKIP_CORRECTNESS: set[str] = set()

if os.environ.get("TDC_TORCHBENCH", "0") == "1":
    # When TDC_TORCHBENCH=1, the benchmark switches to torchbench-only
    # mode: built-in workloads are cleared and we run only the
    # torchbench models below. This keeps the printed table focused on
    # real-model results when that's what the user asked for.
    #
    # Label is auto-derived as f"torchbench:{name} (B={bs})". llama is
    # torchbench's reference Llama decoder; llava is the HF multimodal
    # model — metadata.yaml marks CPU unsupported (OOM on CI), so on
    # cpu it skips via _load_torchbench's broad except.
    WORKLOADS.clear()
    for _name, _bs in [
        ("squeezenet1_1", 64),
        ("BERT_pytorch",  64),
        ("llama",         64),
        # ("llava",         64), // entry does not match, aot will remove all model
        ("dlrm",         128),
        ("dcgan",         32),
        ("stable_diffusion_unet",         None),
        ("stable_diffusion_text_encoder",         None),
        ("timm_vision_transformer",         64),
        ("hf_GPT2",         8),
        ("hf_Whisper",         1024),

    ]:
        _label = f"torchbench:{_name} (B={_bs})"
        _loaded = _load_torchbench(_name, batch_size=_bs)
        if _loaded is not None:
            WORKLOADS[_label] = _loaded
    if not WORKLOADS:
        raise SystemExit(
            "TDC_TORCHBENCH=1 set but no torchbench models loaded -- "
            "is the torchbenchmark package importable, and were the "
            "model weights downloaded? See _load_torchbench()'s "
            "per-model skip messages above for details."
        )


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
    try:
        with torch.no_grad():
            with tdc.capture() as trace:
                captured_out = fn(*example_inputs)
    except Exception:
        traceback.print_exc()
        return None

    def cb(*_unused):
        trace.replay()
        return captured_out

    return cb


def _export_capture(fn, example_inputs):
    """Export fn via torch.export and return ep.module() as the callable.

    torch.export captures a self-contained fx.GraphModule (Dynamo
    bytecode trace, all params lifted to graph inputs) plus an
    input/output spec. ep.module() returns an InterpreterModule that
    walks the graph eagerly on each call — no Dynamo guards, no AOT
    functionalization wrapper, but also no C++ replay. Useful to
    isolate "graph snapshot + Python interpretation" overhead vs the
    eager-everywhere or fully-compiled extremes.

    torch.export only accepts nn.Module; wrap plain functions in a
    thin Module. Returns None on failure (some workloads have control
    flow torch.export refuses to spec)."""
    try:
        if isinstance(fn, torch.nn.Module):
            mod = fn
        else:
            # torch.export needs named forward parameters to match
            # dynamic_shapes against, so synthesize an arity-correct
            # forward with positional names a0, a1, ... Closures over
            # `fn` keep dispatching to the original.
            n = len(example_inputs)
            arg_names = [f"a{i}" for i in range(n)]
            src = (
                "def forward(self, " + ", ".join(arg_names) + "):\n"
                "    return _fn(" + ", ".join(arg_names) + ")\n"
            )
            ns = {"_fn": fn}
            exec(src, ns)
            cls = type("_FnAsModule", (torch.nn.Module,), {"forward": ns["forward"]})
            mod = cls()
            forward_arg_names = arg_names
        # Declare every Tensor input's dim 0 as dynamic so the export
        # tolerates the same kind of batch variation the other backends
        # (dynamic=True) tolerate.
        from torch.export import Dim
        if isinstance(fn, torch.nn.Module):
            # Use the module's own forward signature.
            import inspect
            sig = inspect.signature(fn.forward)
            forward_arg_names = [
                n for n in sig.parameters if n != "self"
            ][:len(example_inputs)]
        dynamic_shapes = {}
        for name, a in zip(forward_arg_names, example_inputs):
            if isinstance(a, torch.Tensor) and a.dim() > 0:
                # Declare every dim as Dim.AUTO so export matches the
                # dynamic=True behaviour the other backends use. AUTO
                # specialises on constant dims and leaves varied dims
                # symbolic, so this is strictly more permissive than
                # the {0: Dim.AUTO} variant we used to set -- needed
                # for dynamic-S workloads where dim 1 (seqlen) varies.
                dynamic_shapes[name] = {i: Dim.AUTO for i in range(a.dim())}
            else:
                dynamic_shapes[name] = None
        ep = torch.export.export(
            mod, tuple(example_inputs),
            dynamic_shapes=dynamic_shapes,
        )
        return ep.module()
    except Exception as e:
        print(f"# export: skipping ({type(e).__name__}: {str(e)[:160]})")
        return None


def build_variants(fn, example_inputs, only=None):
    """Return dict of {label: callable} for fn under each mode.

    Comment out a line to skip that mode entirely — the speed table and
    profile section read the dict's keys at runtime, so partial coverage
    (e.g., NPU where inductor is not available) just works without
    touching either of those functions.

    Key order here is the column order in the printed table.

    ``only`` is an optional iterable of variant names to keep -- builders
    not in ``only`` are skipped entirely (no compile cost paid). When
    ``only`` is None all variants are built.
    """
    # torch.export: Dynamo bytecode trace once into a self-contained
    # gm + spec; ep.module() interprets it on each call. May fail per
    # workload (control flow torch.export refuses) — _export_capture
    # then returns None, and the speed table skips that cell while
    # keeping the column.
    only_set = set(only) if only is not None else None
    variants = {}
    for name, builder in [
        ("eager",     lambda: fn),
        ("dynamo",    lambda: torch.compile(fn, backend="eager", dynamic=True)),
        ("aot_eager", lambda: torch.compile(fn, backend="aot_eager", dynamic=True)),
        ("inductor",  lambda: torch.compile(fn, backend="inductor", dynamic=True)),
        # Same backend as 'inductor', but mode="reduce-overhead" sets
        # triton.cudagraphs=True. On CUDA/XPU it wraps the compiled graph
        # in a cudagraph; on NPU it lands on torch_npu's monkey-patched
        # cudagraphify -> npugraphify (aclgraph). On CPU the flag has no
        # effect, so this column degrades to plain inductor.
        ("reduce-overhead", lambda: torch.compile(
            fn, backend="inductor", mode="reduce-overhead", dynamic=True)),
        # v3 variants both compile the SAME fn in the same process; without
        # isolate_fresh_fn each torch.compile would share fn.__code__'s
        # Dynamo cache_entry_list, and the second capture would silently
        # reuse the first's compiled artifact (see python/v3/compile.py's
        # isolate_fresh_fn docstring).
        ("v3-stock",    lambda: tdcv3.capture(tdcv3.isolate_fresh_fn(fn), *example_inputs)),
        ("v3-fallback", lambda: tdcv3.capture_fallback(tdcv3.isolate_fresh_fn(fn), *example_inputs)),
        # ("v1",             lambda: _v1_capture(fn, example_inputs)),
        ("v2",    lambda: tdcv2.capture(fn, *example_inputs, wrapper=False)),
        # ("v2 (wrapper)",   lambda: tdcv2.capture(fn, *example_inputs, wrapper=True)),
        ("export",         lambda: _export_capture(fn, example_inputs)),
    ]:
        if only_set is not None and name not in only_set:
            continue
        try:
            variants[name] = builder()
        except Exception:
            traceback.print_exc()
            variants[name] = None
    return variants


# Authoritative ordered list of variant names. Mirrors the build_variants()
# builder order so CLI listing and resolution stays in lock-step with what
# the benchmark actually knows how to build.
_ALL_VARIANT_NAMES = [
    "eager", "dynamo", "aot_eager", "inductor", "reduce-overhead",
    "v3-stock", "v3-fallback",
    "v2", "export",
]


def _resolve_variants(specs):
    """Parse one-or-more --variants arguments into a deduplicated ordered
    list of variant names. Supports exact match and case-insensitive
    substring match ('v3' picks both v3-stock and v3-fallback). Errors
    out on unrecognized patterns to fail loudly rather than silently run
    fewer variants than the user asked for. Returns None when no spec
    was given, meaning 'all variants'."""
    if not specs:
        return None
    requested: list[str] = []
    for spec in specs:
        requested.extend(s.strip() for s in spec.split(",") if s.strip())
    keep: list[str] = []
    not_found: list[str] = []
    for r in requested:
        if r in _ALL_VARIANT_NAMES:
            if r not in keep:
                keep.append(r)
            continue
        r_lower = r.lower()
        matches = [n for n in _ALL_VARIANT_NAMES if r_lower in n.lower()]
        if not matches:
            not_found.append(r)
            continue
        for m in matches:
            if m not in keep:
                keep.append(m)
    if not_found:
        raise SystemExit(
            f"Unrecognized --variants entries: {not_found}\n"
            f"Available: {_ALL_VARIANT_NAMES}"
        )
    return keep


def _filter_workloads(specs):
    """Parse --workloads arguments into a filtered WORKLOADS dict. Uses
    case-insensitive substring match against the workload label so e.g.
    '--workloads hf_GPT2' matches 'torchbench:hf_GPT2 (B=8)'. Errors out
    on unrecognized patterns. Returns None when no spec was given."""
    if not specs:
        return None
    requested: list[str] = []
    for spec in specs:
        requested.extend(s.strip() for s in spec.split(",") if s.strip())
    keep: dict = {}
    not_found: list[str] = []
    for r in requested:
        r_lower = r.lower()
        matches = [label for label in WORKLOADS if r_lower in label.lower()]
        if not matches:
            not_found.append(r)
            continue
        for label in matches:
            keep[label] = WORKLOADS[label]
    if not_found:
        raise SystemExit(
            f"Unrecognized --workloads entries: {not_found}\n"
            f"Available:\n  " + "\n  ".join(WORKLOADS)
        )
    return keep


# Variants for which we MUST NOT torch._dynamo.reset() between calls.
# Two reasons end up in the same set:
#   - v1 / v2: capture into their own C++ Trace; Dynamo's cache is
#     irrelevant after capture, but resetting it is also harmless.
#   - v3-stock / v3-fallback: the captured object is an OptimizedModule.
#     dynamo.reset() invalidates its compile entry, the next call
#     re-traces under whatever inductor_config is current at that moment
#     (cpp_wrapper=False by default), and the variant silently degrades
#     to a vanilla python-wrapper inductor compile. Verified empirically
#     -- without this the v3 columns are indistinguishable from
#     'inductor' because they ARE the same compile.
_CAPTURE_MODES = {"v1", "v2 (direct)", "v2 (wrapper)", "v3-stock", "v3-fallback"}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_iters(callable_, inputs, n_warmup=10, n_iters=100):
    """Pre-generated inputs reused across all iterations so the timing
    excludes randn() cost. Median of n_iters samples in microseconds.

    On accelerator devices, SYNC() bracketing each call ensures we
    measure kernel completion rather than just dispatch enqueueing."""
    try:
        for _ in range(n_warmup):
            callable_(*inputs)
            SYNC()
    except Exception:
        traceback.print_exc()
        return None
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        try:
            callable_(*inputs)
        except Exception:
            traceback.print_exc()
            return None
        SYNC()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def fmt_us(v):
    if v is None:
        return "     N/A"
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


def _compare_outputs(ref, got, atol=1e-3, rtol=1e-3):
    """Return (ok, message). Tolerances allow for fp32 FMA reordering
    introduced by inductor fusion or by v2's core_aten_decompositions
    rewrites (e.g. aten.linear -> mm + add, aten.silu -> mul +
    sigmoid). Both 1e-3 keeps the check strict enough to catch real
    algorithmic bugs (wrong shape/dtype, NaN, order-1 errors) while
    tolerating FMA reordering on typical fp32 nets."""
    if len(ref) != len(got):
        return False, f"flat output count {len(got)} vs eager {len(ref)}"
    for i, (a, b) in enumerate(zip(ref, got)):
        if a.shape != b.shape:
            return False, f"out[{i}] shape {tuple(b.shape)} vs eager {tuple(a.shape)}"
        if a.dtype != b.dtype:
            return False, f"out[{i}] dtype {b.dtype} vs eager {a.dtype}"
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a.float() - b.float()).abs().max().item()
            ref_max = a.float().abs().max().item()
            return False, (
                f"out[{i}] max abs diff {diff:.3e} (atol={atol}, "
                f"ref max abs {ref_max:.3e}, "
                f"rel {diff/max(ref_max, 1e-12):.2e})"
            )
    return True, ""


def run_correctness_check(only=None):
    """Run each variant once and compare its output against eager.

    Collects all results first, then prints them at once. Failures are
    reported but do not abort the run — the caller decides whether to
    stop or continue with the speed table.

    ``only``, if set, restricts which variants are built. Eager is
    always force-included as the comparison reference."""
    error_buf = io.StringIO()
    orig_stderr = sys.stderr
    results: list[tuple] = []
    all_ok = True
    only_eff = (
        list(dict.fromkeys(["eager", *only])) if only is not None else None
    )

    for label, (fn, inputs) in WORKLOADS.items():
        if label in _TB_SKIP_CORRECTNESS:
            results.append((label, "", "skipped — v2 / AOT output-arity mismatch", True))
            continue
        torch._dynamo.reset()
        sys.stderr = error_buf
        try:
            variants = build_variants(fn, inputs, only=only_eff)
        finally:
            sys.stderr = orig_stderr
        if "eager" not in variants:
            results.append((label, "", "no eager variant, skipping", True))
            continue
        with torch.no_grad():
            ref = _flatten_output(variants["eager"](*inputs))
        ref = [t.detach().clone() for t in ref]

        for name, callable_ in variants.items():
            if name == "eager":
                continue
            if callable_ is None:
                results.append((label, name, "skipped (variant not available)", True))
                continue
            if name not in _CAPTURE_MODES:
                torch._dynamo.reset()
            with torch.no_grad():
                sys.stderr = error_buf
                try:
                    got = _flatten_output(callable_(*inputs))
                except Exception:
                    results.append((label, name, "N/A (exception during call)", False))
                    all_ok = False
                    continue
                finally:
                    sys.stderr = orig_stderr
            ok, msg = _compare_outputs(ref, got)
            status = "ok" if ok else f"MISMATCH ({msg})"
            results.append((label, name, status, ok))
            if not ok:
                all_ok = False

    # Print all results at once.
    print("\n# correctness check vs eager")
    print("-" * 78)
    for label, name, status, _ok in results:
        if name:
            print(f"  {label:<44} {name:<10} {status}")
        else:
            print(f"  {label:<44} {status}")

    # Dump any errors captured during variant construction or invocation.
    errors = error_buf.getvalue()
    if errors:
        print(f"\n{'='*78}")
        print("Errors during correctness check:")
        print(f"{'='*78}")
        print(errors)

    if not all_ok:
        print("\n*** Correctness check had failures — see lines above ***")


def run_speed_table(only=None):
    """Print a per-workload x per-mode timing table.

    Column order is the insertion order of build_variants(); modes can
    be added / commented out there without touching this function.
    Ratio columns relative to 'eager' are appended for every non-eager
    mode that's present. If 'eager' is not in the variant set, the
    ratio block is omitted entirely.

    Errors during variant construction or timing are captured silently
    and printed after the complete table so tracebacks never interrupt
    the table layout.

    ``only`` is the user-selected variant filter (or None for all).
    """
    # Discover the column structure once from a representative workload.
    sample_fn, sample_inputs = next(iter(WORKLOADS.values()))
    error_buf = io.StringIO()
    orig_stderr = sys.stderr
    sys.stderr = error_buf
    try:
        sample_variants = build_variants(sample_fn, sample_inputs, only=only)
    finally:
        sys.stderr = orig_stderr
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

    # Collect all rows and errors first, then print everything together.
    rows: list[tuple[str, dict]] = []  # (label, times)

    for label, (fn, inputs) in WORKLOADS.items():
        torch._dynamo.reset()
        orig_stderr = sys.stderr
        sys.stderr = error_buf
        try:
            variants = build_variants(fn, inputs, only=only)
        finally:
            sys.stderr = orig_stderr
        if list(variants.keys()) != variant_names:
            raise RuntimeError(
                f"build_variants returned different keys for workload "
                f"'{label}': {list(variants.keys())} vs {variant_names}")
        times: dict = {}
        for name, callable_ in variants.items():
            if callable_ is None:
                times[name] = None
                continue
            if name not in _CAPTURE_MODES:
                torch._dynamo.reset()
            sys.stderr = error_buf
            try:
                times[name] = time_iters(callable_, inputs)
            finally:
                sys.stderr = orig_stderr
        rows.append((label, times))

    # Print the full table.
    print(f"\n{header}")
    sub = "(times in us"
    if ratio_header:
        sub += "; ratios relative to eager)"
    else:
        sub += ")"
    print(sub)
    print("-" * len(header))

    for label, times in rows:
        def cell(t):
            return "     N/A" if t is None else fmt_us(t)

        time_strs = " ".join(f"{cell(times[n]):>{col_w}}" for n in variant_names)
        if has_eager:
            eg = times["eager"]
            def ratio(n):
                t = times[n]
                if t is None or eg is None or eg <= 0:
                    return "    N/A "
                return f"{(t/eg):>{col_w}.2f}x"
            ratio_strs = " ".join(f"{ratio(n):>{col_w}}" for n in ratio_names)
            print(f"{label:<33} {time_strs} | {ratio_strs}")
        else:
            print(f"{label:<33} {time_strs}")

    # Dump any errors captured during variant construction or timing.
    errors = error_buf.getvalue()
    if errors:
        print(f"\n{'='*78}")
        print("Errors during benchmark:")
        print(f"{'='*78}")
        print(errors)


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
    elif DEVICE.type == "npu" and hasattr(ProfilerActivity, "NPU"):
        acts.append(ProfilerActivity.NPU)
    return acts


def _safe_filename(label):
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def _profile_one(label, callable_, inputs, out_path, n_warmup=5, n_iters=30):
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
                out_dir="prototypes/traces",
                only=None):
    """Profile the chosen workload under every mode that build_variants
    enables, exporting one Chrome-trace per mode for side-by-side
    timeline inspection in perfetto / chrome://tracing.

    File naming: <workload>__<mode>.json, e.g. `..__inductor.json`,
    `..__v1.json`. If a mode is commented out in build_variants, no
    trace is produced for it.
    """
    fn, inputs = WORKLOADS[workload_label]
    variants = build_variants(fn, inputs, only=only)
    stem = _safe_filename(workload_label)

    for name, callable_ in variants.items():
        if callable_ is None:
            continue
        try:
            _profile_one(name, callable_, inputs, f"{out_dir}/{stem}__{name}.json")
        except Exception:
            traceback.print_exc()
            print(f"# profile: {name} skipped due to exception (see traceback above)")


def _parse_cli_args(argv=None):
    """Parse command-line arguments for the benchmark driver.

    Designed so a bash launcher can shard work over many processes
    (one variant x one workload per process) for full crash isolation:

        for v in v3-stock v3-fallback v2 inductor; do
            for w in hf_GPT2 BERT_pytorch llama; do
                python prototypes/v2_benchmark.py \\
                    --variants "$v" --workloads "$w" \\
                    --skip-profile > out/${v}_${w}.log 2>&1
            done
        done

    Each (variant, workload) pair runs in its own Python process; a
    crash in v3-fallback's compile (e.g. the upstream proxy executor
    codegen bug we saw on CUDA/NPU for hf_GPT2) only loses its own log,
    not the whole sweep.
    """
    parser = argparse.ArgumentParser(
        description="v2/v3 capture-vs-compile benchmark driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variants", action="append", default=[],
        help="Variant name(s) to include. May be repeated or comma-separated. "
             "Case-insensitive substring match: 'v3' picks v3-stock + v3-fallback. "
             "Default: all variants.",
    )
    parser.add_argument(
        "--workloads", action="append", default=[],
        help="Workload label substring(s). May be repeated or comma-separated. "
             "e.g. '--workloads hf_GPT2' matches 'torchbench:hf_GPT2 (B=8)'. "
             "Default: all workloads (which is the TDC_TORCHBENCH-gated set).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List variant names and workload labels, then exit.",
    )
    parser.add_argument(
        "--skip-correctness", action="store_true",
        help="Skip the correctness-check vs eager pass.",
    )
    parser.add_argument(
        "--skip-speed", action="store_true",
        help="Skip the per-workload x per-variant timing table.",
    )
    parser.add_argument(
        "--skip-profile", action="store_true",
        help="Skip the profile / chrome-trace export step (default is to "
             "profile the last workload).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli_args()

    if args.list:
        print("Variants:")
        for n in _ALL_VARIANT_NAMES:
            print(f"  {n}")
        print("\nWorkloads:")
        for label in WORKLOADS:
            print(f"  {label}")
        raise SystemExit(0)

    only = _resolve_variants(args.variants)
    filtered = _filter_workloads(args.workloads)
    if filtered is not None:
        # Replace WORKLOADS in place so the existing run_* code that
        # iterates WORKLOADS directly sees the filtered subset.
        WORKLOADS.clear()
        WORKLOADS.update(filtered)
    if not WORKLOADS:
        raise SystemExit("No workloads selected -- nothing to run.")

    print("# v2 framework benchmark")
    print_device_banner()
    if only is not None:
        print(f"# variant filter: {only}")
    if filtered is not None:
        print(f"# workload filter: {list(WORKLOADS)}")
    # Reflect the live build_variants() output so commenting out a
    # backend (e.g. inductor on NPU) is reflected in the banner too.
    sample_fn, sample_inputs = next(iter(WORKLOADS.values()))
    _names = list(build_variants(sample_fn, sample_inputs, only=only).keys())
    print(f"# modes: {' / '.join(_names)}")
    if not args.skip_correctness:
        run_correctness_check(only=only)
    if not args.skip_speed:
        run_speed_table(only=only)
    if not args.skip_profile:
        run_profile(list(WORKLOADS)[-1], only=only)
