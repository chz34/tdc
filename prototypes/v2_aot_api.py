"""
v2 user-API verification: torch.compile + aot_autograd backend hook.

Standalone prototype — not dependent on the v1 C++ extension. Verifies the
proposed v2 interface (DESIGN.md §17.6) against real torch.compile +
AOTAutograd pipelines, and dumps both pipeline stages side-by-side:

  - Stage 1: Dynamo graph (pre-AOT) — what `backend=fn` would see.
  - Stage 2: AOT graph (post-AOTAutograd) — what v2 will translate to traces.

Two cases run through both stages:
  (1) `view(x.shape[0] // 2, 2, -1)`  — minimal shape-derived literal.
  (2) attention QK projection           — multi-stage shape derivation.
"""

import torch
from torch._dynamo.testing import EagerAndRecordGraphs, AotEagerAndRecordGraphs


# ----- case 1: minimal shape-derived literal --------------------------------
def shape_derived_view(x):
    # The exact v1 dispatcher-PoC blind spot from §8.1.
    return x.view(x.shape[0] // 2, 2, -1)


# ----- case 2: attention QK projection --------------------------------------
N_HEADS = 8


def attention_qk(q, k):
    # q, k : [B, S, H], H must be divisible by N_HEADS.
    # Reshape to [B, S, n_h, h_dim], permute to attention layout,
    # then batched matmul Q @ K^T -> [B, n_h, S, S].
    B, S, H = q.shape
    h_dim = H // N_HEADS
    q2 = q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3)    # [B, n_h, S, h_dim]
    k2 = k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1)    # [B, n_h, h_dim, S]
    return torch.matmul(q2, k2)


# ----- comparison harness ---------------------------------------------------
def run_comparison(label, fn, sample_inputs_list):
    """Compile `fn` once with the Dynamo-only recording backend and once
    with the AOT recording backend, run on each input set, then dump both
    captured graphs side-by-side. Returns the two recorders."""
    dynamo_recorder = EagerAndRecordGraphs()
    aot_recorder = AotEagerAndRecordGraphs()

    fn_dynamo = torch.compile(fn, backend=dynamo_recorder, dynamic=True)
    fn_aot = torch.compile(fn, backend=aot_recorder, dynamic=True)

    print(f"\n{'#' * 70}")
    print(f"# Case: {label}")
    print(f"{'#' * 70}")

    for inputs in sample_inputs_list:
        ref = fn(*inputs)
        y_dynamo = fn_dynamo(*inputs)
        y_aot = fn_aot(*inputs)
        torch.testing.assert_close(y_dynamo, ref)
        torch.testing.assert_close(y_aot, ref)
        in_shapes = [tuple(t.shape) for t in inputs]
        print(f"  inputs {in_shapes} -> output {tuple(y_aot.shape)}  OK")

    print(f"\nDynamo graph captures : {len(dynamo_recorder.graphs)}")
    print(f"AOT fw graph captures : {len(aot_recorder.fw_graphs)}")

    print(f"\n--- {label}: Dynamo graph (pre-AOT) ---")
    dynamo_recorder.graphs[0].print_readable()
    print(f"\n--- {label}: AOT graph (post-AOTAutograd) ---")
    aot_recorder.fw_graphs[0].print_readable()
    return dynamo_recorder, aot_recorder


def main():
    torch.manual_seed(0)

    # ----- case 1 -----
    _, aot_case1 = run_comparison(
        "view(x.shape[0] // 2, 2, -1)",
        shape_derived_view,
        [(torch.randn(8, 6),), (torch.randn(12, 5),), (torch.randn(10, 4),)],
    )

    # Structural assertions on the simple case — these are the invariants
    # v2's translation depends on:
    #   - dynamic dims are graph-input SymInts (Dynamo lifts when dynamic=True,
    #     AOT inherits the signature). sym_size lives in the wrapper, not the
    #     graph body — v2 needs kCapturedSymInt input ref kind.
    #   - sym arithmetic stays as call_function (operator.floordiv etc.) —
    #     v2 needs the SymExpr Step kind.
    #   - aten.view's size arg is a Python list mixing sym refs and literals —
    #     v2 needs the kIntList input ref kind.
    gm = aot_case1.fw_graphs[0]
    symint_placeholders = [
        n for n in gm.graph.nodes
        if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.SymInt)
    ]
    arith_nodes = [
        n for n in gm.graph.nodes
        if n.op == "call_function"
        and any(kw in str(n.target) for kw in ("floordiv", "truediv", "mul", "add", "sub"))
    ]
    aten_nodes = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and "aten." in str(n.target)
    ]
    assert symint_placeholders, "AOT graph: missing SymInt placeholders"
    assert arith_nodes, "AOT graph: missing sym arithmetic nodes"
    assert aten_nodes, "AOT graph: missing aten nodes"

    # ----- case 2 -----
    # B, S vary; H is held constant (=32) across inputs because changing
    # the divisor's quotient (H // N_HEADS) tends to fail an int-equality
    # guard and force a recompile. Keeping H fixed lets us see "one graph
    # reused across (B, S) shapes" — the core dynamic-shape claim.
    run_comparison(
        f"attention QK [B,S,H] -> Q@K^T [B,{N_HEADS},S,S]",
        attention_qk,
        [
            (torch.randn(2, 4, 32), torch.randn(2, 4, 32)),
            (torch.randn(3, 7, 32), torch.randn(3, 7, 32)),
            (torch.randn(5, 11, 32), torch.randn(5, 11, 32)),
        ],
    )

    # Differences observed in the two stages across both cases:
    #
    #   Aspect         | Dynamo stage              | AOT stage
    #   ---------------+---------------------------+----------------------
    #   placeholders   | source-derived (s77, L_q_)| normalized (arg0_1...)
    #   op invocation  | method (q.view, q.permute)| qualified call_function
    #                  | + operator.matmul         |   (aten.view, aten.permute
    #                  |                           |    aten.expand, aten.bmm)
    #   size args      | unpacked positionals      | IntList literal
    #   decomposition  | high-level matmul stays   | matmul -> expand + bmm
    #                  |                           |   (broadcast lowering)
    #   sym arithmetic | call_function (operator.*)| same
    #
    # Notes for v2 translation against the AOT stage:
    #   - matmul broadcasts to bmm via aten.expand. Both the expand size and
    #     bmm shapes use sym-derived IntLists; kIntList must support nested
    #     SymExpr edges.
    #   - permute's dims arg is a literal IntList (always ints) — no
    #     SymInt-mixing needed there.
    #   - reshape/view's size arg is sym-typed and needs kIntList with
    #     sym refs.


if __name__ == "__main__":
    main()
