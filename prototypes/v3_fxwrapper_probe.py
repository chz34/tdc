"""Puncture-probe the inductor FX wrapper backend (WrapperFxCodegen).

Run on a GPU (CUDA) box -- inductor emits Triton kernels there, which is the
ONLY kernel form the FX wrapper can convert (CPU C++ kernels raise
"FX conversion only supports Triton kernels"; see wrapper_fxir.py:_import_kernel).

What it does:
  1. Turns on `torch._inductor.config.fx_wrapper`.
  2. Hooks `WrapperFxCodegen.compile_graph(gm)` to grab the FX GraphModule
     inductor builds as its *host wrapper* (allocations + kernel launches +
     fallback aten ops), for a few representative workloads.
  3. Dumps each graph node (op / target / args) and prints a category
     breakdown that answers the design question for an fx_wrapper-based v3:

        - aten_fallback        -> v2 translator already replays via callBoxed
        - triton_kernel_launch -> needs a NEW C++ Trace step (launch cubin)
        - alloc / view / misc  -> infra a v2-style replay must also model

Usage:
    python v3_fxwrapper_probe.py                # all workloads
    python v3_fxwrapper_probe.py pointwise mm   # subset
    TDC_DEVICE=cuda python v3_fxwrapper_probe.py # explicit device (default cuda)

No torch_dispatch_capture imports -- depends only on torch, so it can be
copied to any GPU machine standalone.
"""
from __future__ import annotations

import operator
import os
import sys
from collections import Counter

import torch
import torch._inductor.config as inductor_config
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen


DEVICE = os.environ.get("TDC_DEVICE", "cuda")


# --------------------------------------------------------------------------
# Workloads: each mixes fused pointwise (-> Triton kernels) with ops that
# inductor lowers to extern/fallback kernels (mm, conv, sdpa) so the captured
# graph contains BOTH triton_kernel_launch and aten_fallback nodes.
# --------------------------------------------------------------------------
def _mlp(x, w1, b1, w2):
    # addmm (extern) + fused gelu/relu pointwise (triton) + mm (extern)
    h = torch.relu(torch.addmm(b1, x, w1))
    return torch.nn.functional.gelu(h @ w2)


def _attention(q, k, v):
    # sdpa often lowers to a fallback/extern kernel; surrounding scale fuses
    return torch.nn.functional.scaled_dot_product_attention(q, k, v) * 0.5


def _pointwise(a, b):
    # pure elementwise -> should collapse to a single fused Triton kernel
    return torch.relu(a * b + a) * 2.0 - b.sigmoid()


def _layernorm(x, w, b):
    # reduction (mean/var) fuses; may emit reduction Triton kernel(s)
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b)


def _make_inputs(name):
    d = DEVICE
    if name == "mlp":
        return _mlp, (torch.randn(64, 128, device=d), torch.randn(128, 256, device=d),
                      torch.randn(256, device=d), torch.randn(256, 64, device=d))
    if name == "attention":
        return _attention, (torch.randn(2, 4, 64, 32, device=d),
                            torch.randn(2, 4, 64, 32, device=d),
                            torch.randn(2, 4, 64, 32, device=d))
    if name == "pointwise":
        return _pointwise, (torch.randn(512, 512, device=d), torch.randn(512, 512, device=d))
    if name == "layernorm":
        return _layernorm, (torch.randn(64, 256, device=d),
                           torch.randn(256, device=d), torch.randn(256, device=d))
    raise SystemExit(f"unknown workload: {name}")


WORKLOADS = ["pointwise", "mlp", "layernorm", "attention"]


# --------------------------------------------------------------------------
# Node categorization
# --------------------------------------------------------------------------
def _categorize(node):
    if node.op == "placeholder":
        return "input"
    if node.op == "output":
        return "output"
    if node.op == "get_attr":
        return "get_attr"
    if node.op != "call_function":
        return f"other({node.op})"

    t = node.target
    if t is triton_kernel_wrapper_mutation:
        return "triton_kernel_launch"
    if isinstance(t, torch._ops.OpOverload):
        name = str(t)
        if name in ("aten.empty_strided.default",):
            return "alloc"
        return "aten_fallback"
    if t is torch.empty_strided:
        return "alloc"
    if t is operator.getitem:
        return "getitem"
    tmod = getattr(t, "__module__", "") or ""
    tname = getattr(t, "__name__", str(t))
    if "reinterpret" in tname or "as_strided" in tname:
        return "view/reinterpret"
    # SymInt size/grid arithmetic: operator.*/math.*/sympy/torch.sym_*.
    # In the host wrapper graph these compute buffer sizes and launch grids
    # (e.g. ceildiv(numel, BLOCK) = -(-numel // BLOCK)); they are infra, not
    # compute, and a v2-style replay already models SymInt-derived literals.
    if tmod in ("operator", "_operator", "math") or "sympy" in tmod \
            or "sym" in tname.lower():
        return "sym/size"
    return f"misc({tname})"


def _fmt_arg(a):
    if isinstance(a, torch.fx.Node):
        return f"%{a.name}"
    if isinstance(a, (list, tuple)):
        return "[" + ", ".join(_fmt_arg(x) for x in a) + "]"
    s = repr(a)
    return s if len(s) <= 48 else s[:45] + "..."


def _describe_triton(node):
    """Resolve a triton_kernel_wrapper_mutation node into the data a v2-style
    Trace step would have to capture: which compiled kernel, the launch grid,
    and the tensor-arg -> buffer routing. The kernel itself is not inline -- it
    lives in kernel_side_table indexed by kernel_idx."""
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    kw = node.kwargs
    idx = kw.get("kernel_idx")
    name = None
    try:
        kernel = kernel_side_table.get_kernel(idx)
        # JITFunction: .fn.__name__; Autotuner: .fn.fn.__name__
        for chain in (("fn", "__name__"), ("__name__",), ("fn", "fn", "__name__")):
            obj = kernel
            for a in chain:
                obj = getattr(obj, a, None)
            if isinstance(obj, str):
                name = obj
                break
        name = name or type(kernel).__name__
    except Exception as e:  # GPU-only table; never let probing crash the dump
        name = f"<unresolved: {type(e).__name__}>"

    tkw = kw.get("kwargs", {})
    args = {k: _fmt_arg(v) for k, v in tkw.items()} if isinstance(tkw, dict) else tkw
    return {"kernel_idx": idx, "kernel": name, "grid": _fmt_arg(kw.get("grid")),
            "args": args}


def _dump_graph(gm, tag):
    print(f"\n{'=' * 72}\n  fx_wrapper GraphModule -- {tag}\n{'=' * 72}")
    cats = Counter()
    for node in gm.graph.nodes:
        cat = _categorize(node)
        cats[cat] += 1
        tgt = getattr(node.target, "__name__", str(node.target)) \
            if node.op == "call_function" else node.op
        args = ", ".join(_fmt_arg(a) for a in node.args)
        kw = ""
        if node.op == "call_function" and node.kwargs:
            # for the triton HOP, kwargs carry kernel/grid -- show keys
            kw = "  kwargs={" + ", ".join(node.kwargs.keys()) + "}"
        print(f"  [{cat:20s}] {node.name:18s} = {tgt}({args}){kw}")
        if cat == "triton_kernel_launch":
            d = _describe_triton(node)
            print(f"       -> kernel_idx={d['kernel_idx']} kernel={d['kernel']} "
                  f"grid={d['grid']}")
            print(f"       -> arg routing: {d['args']}")

    print(f"\n  -- category counts ({tag}) --")
    for c, n in cats.most_common():
        print(f"     {n:3d}  {c}")

    # The design-relevant ratio for an fx_wrapper-based v3:
    aten = cats["aten_fallback"]
    triton = cats["triton_kernel_launch"]
    infra = cats["alloc"] + cats["view/reinterpret"] + cats["getitem"] + cats["sym/size"]
    print(f"\n  -- v2-translator readiness ({tag}) --")
    print(f"     aten_fallback (callBoxed-ready) : {aten}")
    print(f"     triton_kernel_launch (NEW step) : {triton}")
    print(f"     infra alloc/view/getitem/sym    : {infra}")
    return cats


def main():
    selected = [a for a in sys.argv[1:] if not a.startswith("-")] or WORKLOADS

    print(f"torch={torch.__version__}  device={DEVICE}")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. Run on a GPU box, or set TDC_DEVICE to a Triton-"
            "capable device. The FX wrapper cannot convert CPU C++ kernels.")

    captured: list[tuple[str, torch.fx.GraphModule]] = []
    orig = WrapperFxCodegen.compile_graph

    current = {"tag": "?"}

    def hook(self, gm):
        captured.append((current["tag"], gm))
        return orig(self, gm)

    WrapperFxCodegen.compile_graph = hook
    inductor_config.fx_wrapper = True

    total = Counter()
    failures: list[tuple[str, str]] = []
    try:
        for name in selected:
            fn, inputs = _make_inputs(name)
            ref = fn(*inputs)
            torch._dynamo.reset()
            current["tag"] = name
            before = len(captured)
            # Isolate each workload: an fx_wrapper conversion gap in one
            # (e.g. sdpa's assert_size_stride raw line) must not abort the run
            # or drop the aggregate over the workloads that did convert.
            try:
                compiled = torch.compile(fn, backend="inductor", dynamic=True)
                with torch.no_grad():
                    out = compiled(*inputs)
                ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
                n_graphs = len(captured) - before
                print(f"\n### workload={name}: numeric={'MATCH' if ok else 'MISMATCH'}"
                      f", fx graphs captured={n_graphs}")
                for i, (tag, gm) in enumerate(captured[before:]):
                    total += _dump_graph(gm, f"{tag}#{i}")
            except Exception as e:
                last = str(e).strip().splitlines()[-1][:120]
                failures.append((name, last))
                print(f"\n### workload={name}: FAILED to convert -> {last}")
    finally:
        WrapperFxCodegen.compile_graph = orig

    print(f"\n{'#' * 72}\n  AGGREGATE over {len(captured)} captured graph(s)\n{'#' * 72}")
    for c, n in total.most_common():
        print(f"  {n:4d}  {c}")
    aten, triton = total["aten_fallback"], total["triton_kernel_launch"]
    denom = aten + triton or 1
    print(f"\n  aten_fallback / (aten_fallback + triton_launch) = "
          f"{aten}/{denom} = {aten / denom:.0%}")
    print("  -> share of kernel-bearing nodes v2 translator already handles "
          "vs needs a new Triton-launch step.")
    if failures:
        print(f"\n  -- fx_wrapper conversion gaps ({len(failures)}) --")
        for name, msg in failures:
            print(f"     {name}: {msg}")


if __name__ == "__main__":
    main()
