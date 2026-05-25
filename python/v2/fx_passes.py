"""Pre-translate FX graph passes for v2.capture.

These passes run after AOT hands us a GraphModule and before our
translator turns it into a C++ Trace. They sit at a well-defined
boundary: input is an fx.GraphModule, output is an equivalent
fx.GraphModule with the target patterns rewritten. They know nothing
about pre_binds, output_shapers, captured tensor slots, or any other
v2.capture internal -- which is what lets us test and evolve them
independently of the capture pipeline.

The passes are exposed as module-level functions:

  rewrite_prims_in_gm(gm) -> gm
      Replace prims.convert_element_type with aten._to_copy.
      Eliminates a Python decomposition jump on every replay.

  rewrite_slice_scatter_to_inplace(gm) -> gm
      De-functionalize slice_scatter + copy_(input, ss_result) back
      to slice + copy_(slice_view, val). Used as a fallback on the
      backward path; inference paths skip functionalize at the AOT
      call so slice_scatter never appears there.

Both passes are idempotent: running them on a graph that has no
matches is a no-op. They mutate the gm in-place via fx.Graph editing
and call recompile() when something changed.

See DESIGN.md §17.6.9 for the perf analysis that motivated each.
"""
from __future__ import annotations

import torch
from torch import fx


__all__ = [
    "rewrite_prims_in_gm",
    "rewrite_slice_scatter_to_inplace",
    "eliminate_dead_clones",
]


# ---------------------------------------------------------------------------
# prims.convert_element_type -> aten._to_copy
# ---------------------------------------------------------------------------
def rewrite_prims_in_gm(gm: fx.GraphModule) -> fx.GraphModule:
    """Replace prims.convert_element_type nodes with aten._to_copy.

    AOT functionalization inserts prims.convert_element_type when it
    lifts Python scalars (e.g. RMSNorm's `self.eps = 1e-6`) into 0-d
    Tensors. Without inductor's codegen lowering, this prim falls
    back to a Python decomposition path at replay time -- much
    slower than the equivalent aten._to_copy which has direct C++
    backend kernels.

    Passing decompositions={prims.convert_element_type: ...} to
    aot_function doesn't help because functionalization-inserted ops
    skip the decomposition table. We rewrite post-AOT, pre-translate.

    Same-dtype rewrites collapse to a no-op (we drop the node and
    rewire users to the input). Different-dtype becomes aten._to_copy.
    See DESIGN.md §17.6.9.
    """
    convert_op = torch.ops.prims.convert_element_type.default
    to_copy_op = torch.ops.aten._to_copy.default
    changed = False
    for node in list(gm.graph.nodes):
        if not (node.op == "call_function" and node.target is convert_op):
            continue
        src, target_dtype = node.args
        src_dtype = (
            src.meta["val"].dtype
            if "val" in src.meta and hasattr(src.meta["val"], "dtype")
            else None
        )
        if src_dtype == target_dtype:
            node.replace_all_uses_with(src)
            gm.graph.erase_node(node)
            changed = True
            continue
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                to_copy_op, args=(src,), kwargs={"dtype": target_dtype}
            )
        # Propagate meta so downstream consumers (including subsequent
        # rewrites) see the right dtype.
        new_node.meta = dict(node.meta)
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm


# ---------------------------------------------------------------------------
# slice_scatter + copy_ -> slice + copy_(slice_view, val)
# ---------------------------------------------------------------------------
def rewrite_slice_scatter_to_inplace(gm: fx.GraphModule) -> fx.GraphModule:
    """De-functionalize KV-cache style writes:
        slice_scatter(t, val, dim, start, end, step)
        copy_(t, slice_scatter_result)
    ->
        slice_view = slice(t, dim, start, end, step)
        copy_(slice_view, val)

    The primary mitigation for this problem now lives in the AOT
    call: inference paths pass `disable_functionalization=True` so
    in-place mutations stay as in-place ops in the captured graph
    and slice_scatter never appears (DESIGN.md §17.6.9). This pass
    is the *fallback* for the backward path (allow_grad=True), where
    autograd's partition_fn requires a pure-functional graph and
    we can't disable functionalize. It's also harmless on inference
    graphs that have no slice_scatter -- it just walks the graph
    and finds nothing.

    Why: LLaMA / GPT-style decoders write into a KV cache via
    `cache_k[:, start:end] = xk`. With functionalize on, AOT
    transforms this into `slice + slice_scatter`, which **allocates
    a full-size new cache_k and memcpy's the unchanged portion** --
    on a [64, 1024, 8, 64] cache (128 MB), that's 128 MB of pointless
    copy per layer per replay. The rewrite restores eager's in-place
    behavior. Other consumers of the slice_scatter result are
    rewired to read from `base` directly (post-mutation value).
    """
    slice_op = torch.ops.aten.slice.Tensor
    slice_scatter_op = torch.ops.aten.slice_scatter.default
    copy_inplace_op = torch.ops.aten.copy_.default

    # AOT often emits *nested* slice_scatter for multi-dim assignments
    # like `cache_k[:bsz, start:end] = val`:
    #     outer_ss = slice_scatter(arg, inner_ss, outer_dim_args)
    #     inner_ss = slice_scatter(slice(arg, outer), copy, inner_dim_args)
    #     copy_(arg, outer_ss)
    # One pass eliminates only the outer scatter. After that, the
    # remaining copy_ has form (inner_slice_view, inner_ss) which
    # matches pattern 2; the next iteration eliminates the inner
    # scatter. Loop until no more rewrites fire.
    any_changed = False
    while True:
        changed = _rewrite_slice_scatter_pass(
            gm, slice_op, slice_scatter_op, copy_inplace_op,
        )
        any_changed = any_changed or changed
        if not changed:
            break
    if any_changed:
        gm.graph.lint()
        gm.recompile()
    return gm


def _rewrite_slice_scatter_pass(
    gm: fx.GraphModule,
    slice_op,
    slice_scatter_op,
    copy_inplace_op,
) -> bool:
    """One pass of the slice_scatter rewrite. Returns True if any
    rewrite was applied. Caller wraps with a fixed-point loop."""

    def _ss_inner_args(src):
        a = list(src.args) + [None] * (6 - len(src.args))
        kw = dict(src.kwargs)
        return (
            a[1],
            a[2] if a[2] is not None else kw.get("dim", 0),
            a[3] if a[3] is not None else kw.get("start", None),
            a[4] if a[4] is not None else kw.get("end", None),
            a[5] if a[5] is not None else kw.get("step", 1),
        )

    def _is_slice_node(n):
        return (
            isinstance(n, fx.Node)
            and n.op == "call_function"
            and n.target is slice_op
        )

    def _slice_normalized(n):
        """Return (base, dim, start, end, step) with schema defaults
        applied so two semantically equivalent slice nodes match even
        if one omits trailing args."""
        a = list(n.args)
        kw = dict(n.kwargs)
        base = a[0]
        tail = a[1:] + [None] * (4 - len(a[1:]))
        return (
            base,
            tail[0] if tail[0] is not None else kw.get("dim", 0),
            tail[1] if tail[1] is not None else kw.get("start", None),
            tail[2] if tail[2] is not None else kw.get("end", None),
            tail[3] if tail[3] is not None else kw.get("step", 1),
        )

    def _equivalent_slice(a, b):
        if not (_is_slice_node(a) and _is_slice_node(b)):
            return False
        na = _slice_normalized(a)
        nb = _slice_normalized(b)
        # Compare base by identity, the rest by equality (literals or
        # identical FX-node refs).
        return na[0] is nb[0] and na[1:] == nb[1:]

    changed = False
    for node in list(gm.graph.nodes):
        if not (node.op == "call_function" and node.target is copy_inplace_op):
            continue
        if len(node.args) < 2:
            continue
        copy_dst = node.args[0]
        src = node.args[1]
        if not (isinstance(src, fx.Node)
                and src.op == "call_function"
                and src.target is slice_scatter_op):
            continue
        ss_base = src.args[0] if src.args else None
        ss_val, ss_dim, ss_start, ss_end, ss_step = _ss_inner_args(src)

        # Two patterns AOT emits for `cache[outer][inner] = val` (or
        # the 1-level variant `cache[inner] = val`):
        #
        #   1) copy_(base, slice_scatter(base, val, inner))
        #      Single-dim in-place write to base.
        #
        #   2) copy_(slice(base, outer), slice_scatter(slice(base, outer), val, inner))
        #      Two-dim in-place write: outer slice then inner slice.
        #      The two slice nodes are semantically the same view (same
        #      base, same outer params) but are distinct FX nodes.
        #
        # Rewrite goal in both cases: in-place write `val` into the
        # appropriate slice view of the original base tensor.
        if copy_dst is ss_base:
            # Pattern 1: 1-level.
            outer_slice = copy_dst
        elif _equivalent_slice(copy_dst, ss_base):
            # Pattern 2: 2-level. Use the existing outer-slice node
            # already in copy_'s args (copy_dst) -- it's an outer-slice
            # view of the original base.
            outer_slice = copy_dst
        else:
            continue

        # Inner slice into the outer view at the slice_scatter inner
        # dim args. For pattern 1 this becomes the only slice; for
        # pattern 2 it's the inner of a 2-level view (cache[:bsz][...]).
        with gm.graph.inserting_before(node):
            inner_slice = gm.graph.call_function(
                slice_op,
                args=(outer_slice, ss_dim, ss_start, ss_end, ss_step),
            )
        # Rewire other consumers of slice_scatter result to read
        # through outer_slice. After the in-place copy_ runs, the
        # outer slice view sees the mutated values. FX node ordering
        # preserves the dependency since copy_ stays at its original
        # graph position.
        src.replace_all_uses_with(outer_slice)
        # Rewrite copy_ to be the in-place write through inner_slice.
        new_args = (inner_slice, ss_val)
        if len(node.args) > 2:
            new_args = new_args + tuple(node.args[2:])
        node.args = new_args
        # slice_scatter is now unused -> remove. The old outer-slice
        # node feeding slice_scatter (ss_base under pattern 2) may
        # still have other users; only erase if dead.
        gm.graph.erase_node(src)
        if (isinstance(ss_base, fx.Node)
                and ss_base is not copy_dst
                and not ss_base.users):
            gm.graph.erase_node(ss_base)
        changed = True
    return changed


# ---------------------------------------------------------------------------
# aten::clone(x, memory_format=None) elimination
# ---------------------------------------------------------------------------
def eliminate_dead_clones(gm: fx.GraphModule) -> fx.GraphModule:
    """Remove `aten::clone(x, memory_format=None)` nodes that AOT
    decomposes nn.Dropout (and similar identity-in-inference modules)
    into, when no downstream user mutates the clone's result.

    Why these clones exist: PyTorch's autograd contract requires
    dropout/dropoutXd/drop_path to return a NEW tensor with its own
    grad_fn even when train=False / p=0 (so user code that does
    `y = dropout(x); y.add_(1)` doesn't silently mutate x). The
    decomposition therefore emits `aten::clone(x, None)` in inference
    mode -- a full-tensor memcpy with no semantic effect. timm ViT
    eval triggers ~3 of these per transformer block (proj_drop,
    act_drop, mlp_drop), each ~19MB on a (B=64, S=197, dim=384) fp32
    tensor. On accelerators where dispatch isn't the bottleneck (NPU,
    big-batch GPU) this becomes a 700MB+ extra HBM traffic per
    forward and makes replay slower than eager (eager calls the
    dropout C++ op directly, which fuses the clone with the autograd
    no-op into one cheap kernel; we expand it into a free-standing
    clone Step).

    Safety: only eliminate when ALL of
      - memory_format kwarg is None / absent (don't drop
        format-conversion clones used to materialize a contiguous
        copy before _unsafe_view -- those are real work eager would
        also do via reshape)
      - every user is a non-mutating OpOverload (no schema arg with
        alias_info.is_write referencing the clone). copy_, add_,
        mul_, etc. with the clone result as the mutated slot keep
        the clone.

    Idempotent.
    """
    clone_op = torch.ops.aten.clone.default
    changed = False
    for node in list(gm.graph.nodes):
        if not (node.op == "call_function" and node.target is clone_op):
            continue
        # Resolve memory_format from positional/kwarg.
        mem_fmt = None
        if len(node.args) >= 2:
            mem_fmt = node.args[1]
        elif "memory_format" in node.kwargs:
            mem_fmt = node.kwargs["memory_format"]
        if mem_fmt is not None:
            continue
        if not node.args:
            continue
        src = node.args[0]
        # Audit every user: refuse to eliminate if any consumer takes
        # the clone result into a mutated slot. Be defensive about
        # non-OpOverload targets (operator.getitem etc.) -- those
        # don't mutate but we still want a schema to inspect, so we
        # accept them only when their behaviour is read-only by
        # construction.
        if not _all_users_treat_as_read_only(node):
            continue
        node.replace_all_uses_with(src)
        gm.graph.erase_node(node)
        changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm


def _all_users_treat_as_read_only(node: fx.Node) -> bool:
    import operator
    for user in node.users:
        # The graph's `output` node is just "yield this value to the
        # caller" -- read-only by construction. Recognise it before
        # the schema dispatch (its .target is the string "output", not
        # an OpOverload).
        if user.op == "output":
            continue
        tgt = user.target
        if tgt is operator.getitem:
            continue
        if not isinstance(tgt, torch._ops.OpOverload):
            # Unknown callable -- be safe and keep the clone.
            return False
        schema_args = tgt._schema.arguments
        # Find every positional slot in `user.args` that points at
        # `node`. If the corresponding schema arg has alias_info.is_write
        # set, this user mutates the clone's result.
        for slot, arg in enumerate(user.args):
            if arg is not node:
                continue
            if slot >= len(schema_args):
                # Variadic / out-of-schema -- be safe.
                return False
            sa = schema_args[slot]
            if sa.alias_info is not None and sa.alias_info.is_write:
                return False
        # kwargs path: check kwargs that reference the clone.
        for name, arg in user.kwargs.items():
            if arg is not node:
                continue
            # Find schema arg by name.
            matched = next((sa for sa in schema_args if sa.name == name), None)
            if matched is None:
                return False
            if matched.alias_info is not None and matched.alias_info.is_write:
                return False
    return True
