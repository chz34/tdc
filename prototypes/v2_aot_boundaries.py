"""
v2 AOT-graph boundary probe.

For each boundary case we believed v2 design might be incomplete on, this
script:
  - Compiles the case via torch.compile + aot_autograd backend.
  - Captures the AOT forward graph.
  - Prints the graph (via gm.print_readable()).
  - Categorizes every node by op + target and reports which categories
    appeared, so we can confirm or invalidate the design's coverage.

The cases follow the analysis in the chat session. We deliberately include
patterns the current DESIGN.md §17.6 either doesn't mention or marks as
out-of-scope:

  (A) Multi-output ops + operator.getitem unpacking
        max(dim=...), split, topk, var_mean, sort
  (B) torch.sym_* helpers (not just operator.*)
        sym_max, sym_min
  (C) Higher-order operators (HOPs)
        torch.cond -> torch.ops.higher_order.cond
  (D) Functionalized in-place / aliasing
        x.add_, view + add_
  (E) Composite torch APIs that may stay un-decomposed
        einsum, layer_norm, dropout
  (F) Tensor indexing / slicing
        x[0, :], y[0, :x.shape[1]] — Python __getitem__ with dynamic bound

Each section ends with `summarize()` printing exactly what was observed,
so the report doubles as a regression check: if a later PyTorch version
changes the AOT lowering, the diff jumps out immediately.

Optional profiling: set TDC_PROFILE=1 to wrap each case's *post-compile*
invocation in torch.profiler and export a Chrome-trace timeline per case
into prototypes/traces/. Open the JSON at https://ui.perfetto.dev or
chrome://tracing to see the actual call chain: Dynamo prelude (call_size
extracting shapes) -> compiled graph entry -> aten op sequence -> kernel
launches. This confirms the wrapper-then-graph execution model rather
than just inferring it from source.
"""

import os
import pathlib
import torch
from collections import defaultdict
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch.profiler import profile, ProfilerActivity, record_function


PROFILE_ENABLED = os.environ.get("TDC_PROFILE", "") == "1"
TRACE_DIR = pathlib.Path(__file__).parent / "traces"


# ---------------------------------------------------------------------------
# Node target classification
# ---------------------------------------------------------------------------
def classify_target(t):
    """Return (kind, repr_string) describing a call_function/call_method target.

    The category names match the five buckets v2 design must consider:
       OpOverload, HOP, builtin (_operator), python_callable, other.
    """
    if isinstance(t, torch._ops.OpOverload):
        return "OpOverload", str(t)
    if isinstance(t, torch._ops.OpOverloadPacket):
        return "OpOverloadPacket", str(t)
    if isinstance(t, torch._ops.HigherOrderOperator):
        return "HOP", str(t)
    tname = type(t).__name__
    if tname == "builtin_function_or_method":
        mod = getattr(t, "__module__", "")
        return f"builtin ({mod})", f"{mod}.{t.__name__}"
    if callable(t):
        mod = getattr(t, "__module__", "")
        name = getattr(t, "__name__", repr(t))
        return "python_callable", f"{mod}.{name}"
    if isinstance(t, str):
        return "method-name (str)", t
    return "other", repr(t)


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------
def _safe_filename(label):
    """Sanitize a case label for use as a filename."""
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def run_case(label, fn, *inputs):
    """Compile `fn` with aot_autograd recording backend, exercise it with
    the given inputs, print the AOT forward graph + a node-category
    breakdown. If TDC_PROFILE=1, also profile a steady-state invocation
    and export a Chrome-trace timeline. Returns the category dict for
    cross-case aggregation."""
    rec = AotEagerAndRecordGraphs()
    cfn = torch.compile(fn, backend=rec, dynamic=True)
    print(f"\n{'=' * 78}")
    print(f"  CASE: {label}")
    print(f"{'=' * 78}")
    try:
        # First call: triggers compile. Second call: steady state (the one
        # we want to profile because it exercises only the dispatch path,
        # not the compiler).
        cfn(*inputs)
        if PROFILE_ENABLED:
            TRACE_DIR.mkdir(exist_ok=True)
            trace_path = TRACE_DIR / f"{_safe_filename(label)}.json"
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True,
            ) as prof:
                with record_function(f"case::{label}"):
                    cfn(*inputs)
            prof.export_chrome_trace(str(trace_path))
            print(f"  [trace] {trace_path}")
    except Exception as e:
        print(f"  [compilation/run failed] {type(e).__name__}: {e}")
        return None
    if not rec.fw_graphs:
        print("  [no forward graph captured]")
        return None

    gm = rec.fw_graphs[0]
    gm.print_readable()

    # Per-case category summary.
    cats = defaultdict(set)
    for n in gm.graph.nodes:
        if n.op in ("call_function", "call_method"):
            kind, repr_str = classify_target(n.target)
            cats[f"{n.op} / {kind}"].add(repr_str)
        else:
            cats[n.op].add("(structural)")

    print("\n  Node categories observed:")
    for cat, targets in sorted(cats.items()):
        print(f"    {cat}  ({len(targets)})")
        for t in sorted(targets):
            print(f"        {t}")
    return cats


# ---------------------------------------------------------------------------
# Cases — each tagged with the design-implication question it answers.
# ---------------------------------------------------------------------------

# (A) Multi-output ops -- expect operator.getitem in graph
def case_max_dim(x):
    vals, idx = x.max(dim=-1)
    return vals + idx.float()

def case_split(x):
    a, b, c = torch.split(x, x.shape[0] // 3, dim=0)
    return a + b + c

def case_topk(x):
    vals, idx = torch.topk(x, 3, dim=-1)
    return vals + idx.float()

def case_var_mean(x):
    var, mean = torch.var_mean(x, dim=-1)
    return var + mean

def case_sort(x):
    vals, idx = torch.sort(x, dim=-1)
    return vals + idx.float()


# (B) torch.sym_* helpers — non-operator-module sym arith
def case_sym_max(x):
    n = torch.sym_max(x.shape[0], 4)
    return x[:n]

def case_sym_min(x):
    n = torch.sym_min(x.shape[0], 8)
    return x[:n]


# (C) HOP — torch.cond
def case_cond(x):
    return torch.cond(x.sum() > 0,
                      lambda x: x.sin(),
                      lambda x: x.cos(),
                      (x,))


# (D) Functionalized in-place / aliasing
def case_inplace_add(x, y):
    y.add_(x)
    return y * 2

def case_view_then_inplace(x):
    v = x.view(-1)
    v.add_(1)
    return x


# (F) Tensor indexing / slicing — primary use case from chat
def case_slice_with_dynamic_bound(x, y):
    # x[0, :]   uses int + full-slice → select then slice (or just select)
    # y[0, :N]  uses int + bounded-slice with sym-derived upper → slice with sym arg
    return x[0, :] + y[0, :x.shape[1]]

def case_slice_basic(x):
    # No sym in slice: x[0, :] alone — should be pure select / slice with literals
    return x[0, :]

def case_slice_stride(x):
    # Strided slice: x[::2, 1::3] — exercises start/stop/step all three
    return x[::2, 1::3]


# (E) Composite APIs — may stay un-decomposed
def case_einsum(x):
    return torch.einsum("ij,jk->ik", x, x.t())

def case_layer_norm(x):
    return torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[-1],))

def case_dropout_train(x):
    return torch.nn.functional.dropout(x, p=0.5, training=True)


# ---------------------------------------------------------------------------
# Driver — categories pulled together at the end
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    all_results = {}

    # (A) Multi-output ops
    all_results["max(dim=-1)"]   = run_case("max(dim=-1)", case_max_dim,   torch.randn(4, 5))
    all_results["split"]         = run_case("split into 3", case_split,    torch.randn(9, 4))
    all_results["topk"]          = run_case("topk(k=3, dim=-1)", case_topk, torch.randn(4, 6))
    all_results["var_mean"]      = run_case("var_mean(dim=-1)", case_var_mean, torch.randn(4, 5))
    all_results["sort"]          = run_case("sort(dim=-1)", case_sort, torch.randn(4, 5))

    # (B) torch.sym_*
    all_results["sym_max"]       = run_case("torch.sym_max", case_sym_max, torch.randn(3, 5))
    all_results["sym_min"]       = run_case("torch.sym_min", case_sym_min, torch.randn(12, 5))

    # (C) HOP
    all_results["torch.cond"]    = run_case("torch.cond (HOP)", case_cond, torch.randn(8))

    # (D) Functional in-place
    all_results["in-place add_"] = run_case("y.add_(x)", case_inplace_add,
                                            torch.randn(4, 4), torch.randn(4, 4).clone())
    all_results["view + add_"]   = run_case("view().add_(1)", case_view_then_inplace,
                                            torch.randn(2, 3))

    # (E) Composite APIs
    all_results["einsum"]        = run_case("einsum 'ij,jk->ik'", case_einsum, torch.randn(3, 4))
    all_results["layer_norm"]    = run_case("layer_norm last dim", case_layer_norm, torch.randn(2, 5))
    all_results["dropout train"] = run_case("dropout p=0.5 train", case_dropout_train,
                                            torch.randn(4, 4))

    # (F) Tensor slicing
    all_results["slice basic"]        = run_case("x[0, :]", case_slice_basic,
                                                 torch.randn(3, 5))
    all_results["slice dyn bound"]    = run_case("x[0, :] + y[0, :x.shape[1]]",
                                                 case_slice_with_dynamic_bound,
                                                 torch.randn(3, 5), torch.randn(3, 8))
    all_results["slice stride"]       = run_case("x[::2, 1::3]", case_slice_stride,
                                                 torch.randn(5, 7))

    # ------------------- final cross-case summary -------------------
    print("\n\n" + "#" * 78)
    print("#  Cross-case summary: which target categories appeared, where")
    print("#" * 78)
    # category -> list of case labels that exhibited it
    cat_to_cases = defaultdict(list)
    for label, cats in all_results.items():
        if cats is None:
            continue
        for cat in cats:
            cat_to_cases[cat].append(label)

    for cat in sorted(cat_to_cases):
        cases = cat_to_cases[cat]
        print(f"\n  {cat}")
        for c in cases:
            print(f"      - {c}")

    # ---------------- regression assertions ----------------
    # Each assertion codifies a design-relevant empirical fact. If PyTorch
    # changes the lowering for one of these cases, the failure tells us our
    # v2 plan needs to be revisited for that category.

    def cat_present(case_label, cat_substr):
        cats = all_results.get(case_label)
        return cats is not None and any(cat_substr in c for c in cats)

    # (A) Multi-output ops always introduce operator.getitem to unpack.
    for case in ("max(dim=-1)", "split", "topk", "var_mean", "sort"):
        assert cat_present(case, "builtin (_operator)"), (
            f"Expected operator.getitem in '{case}' — multi-output unpacking lost?"
        )

    # (B) torch.sym_max / sym_min surface as python_callable, NOT operator.
    for case in ("sym_max", "sym_min"):
        assert cat_present(case, "python_callable"), (
            f"Expected torch.sym_* python_callable in '{case}'"
        )

    # (C) torch.cond produces a HOP target AND get_attr nodes for sub-graphs.
    assert cat_present("torch.cond", "HOP"), "torch.cond should produce a HOP node"
    assert cat_present("torch.cond", "get_attr"), (
        "torch.cond should produce get_attr for true/false sub-graphs"
    )

    # (D) Functionalized in-place leaves aten.copy_.default as a fence at
    #     graph tail. We can't see the exact target name from cat keys
    #     alone, but the in-place case must contain at least 2 distinct
    #     OpOverloads (the functional op + the copy_ epilogue).
    inplace_targets = set()
    for cats_label in ("in-place add_", "view + add_"):
        if all_results.get(cats_label) is None:
            continue
        for cat, ts in all_results[cats_label].items():
            if "OpOverload" in cat:
                inplace_targets.update(ts)
    assert any("copy_" in t for t in inplace_targets), (
        "Expected aten.copy_.default fence in functionalized in-place graphs"
    )

    # (E) Composite torch APIs (einsum, layer_norm, dropout) get decomposed
    #     to aten — none of them should leave a python_callable behind.
    for case in ("einsum", "layer_norm", "dropout train"):
        assert not cat_present(case, "python_callable"), (
            f"Did NOT expect python_callable in '{case}' — composite stayed un-decomposed"
        )

    print("\n\nAll regression assertions passed.")


if __name__ == "__main__":
    main()
