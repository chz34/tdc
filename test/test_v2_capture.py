"""End-to-end validation of the v2.capture() direct-replay API.

Mirrors test_v2_basic.py's cases but uses v2.capture(fn, *example_args)
to obtain a callable that skips Dynamo at call time. Each case captures
once, then calls the result against multiple input shapes to confirm
the SymInt extraction recipes carry over correctly.
"""
import unittest

import torch
import torch_dispatch_capture.v2 as tdcv2


class TestV2Capture(unittest.TestCase):

    def setUp(self):
        # Each test captures its own function closure; reset Dynamo's
        # global cache so tests don't share compile artifacts (which
        # leak SymInt symbols across tests and break recipe building).
        torch._dynamo.reset()

    def test_simple_arithmetic(self):
        def fn(x, y):
            return x * 2.0 + y - 1.5
        captured = tdcv2.capture(fn, torch.randn(4, 5), torch.randn(4, 5))
        for shape in [(4, 5), (3, 7), (2, 9)]:
            x = torch.randn(*shape)
            y = torch.randn(*shape)
            ref = x * 2.0 + y - 1.5
            out = captured(x, y)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_shape_derived_view(self):
        def fn(x):
            return x.view(x.shape[0] // 2, 2, -1)
        captured = tdcv2.capture(fn, torch.randn(8, 6))
        for shape in [(8, 6), (12, 5), (10, 4)]:
            x = torch.randn(*shape)
            ref = x.view(x.shape[0] // 2, 2, -1)
            out = captured(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_attention_qk(self):
        N_HEADS = 8
        def qk(q, k):
            B, S, H = q.shape
            h_dim = H // N_HEADS
            q2 = q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3)
            k2 = k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1)
            return torch.matmul(q2, k2)

        captured = tdcv2.capture(qk, torch.randn(2, 4, 32), torch.randn(2, 4, 32))
        for B, S in [(2, 4), (3, 7), (5, 11)]:
            H = 32
            q = torch.randn(B, S, H)
            k = torch.randn(B, S, H)
            h_dim = H // N_HEADS
            ref = torch.matmul(
                q.view(B, S, N_HEADS, h_dim).permute(0, 2, 1, 3),
                k.view(B, S, N_HEADS, h_dim).permute(0, 2, 3, 1),
            )
            out = captured(q, k)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_slice_dynamic_upper_bound(self):
        def fn(x, y):
            return x[0, :] + y[0, :x.shape[1]]
        captured = tdcv2.capture(fn, torch.randn(3, 5), torch.randn(3, 8))
        for sx, sy in [((3, 5), (3, 8)), ((4, 6), (4, 10)), ((2, 3), (2, 7))]:
            x = torch.randn(*sx)
            y = torch.randn(*sy)
            ref = x[0, :] + y[0, :x.shape[1]]
            out = captured(x, y)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_multi_output_max(self):
        def fn(x):
            v, i = x.max(dim=-1)
            return v + i.float()
        captured = tdcv2.capture(fn, torch.randn(4, 5))
        for shape in [(4, 5), (6, 8), (3, 11)]:
            x = torch.randn(*shape)
            ref_v, ref_i = x.max(dim=-1)
            ref = ref_v + ref_i.float()
            out = captured(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_split_then_recombine(self):
        def fn(x):
            a, b, c = torch.split(x, x.shape[0] // 3, dim=0)
            return a + b + c
        captured = tdcv2.capture(fn, torch.randn(9, 4))
        for n in [9, 12, 15]:
            x = torch.randn(n, 4)
            a, b, c = torch.split(x, n // 3, dim=0)
            ref = a + b + c
            out = captured(x)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref))

    def test_varied_python_int_arg(self):
        """A plain Python int positional arg must be routed at every
        replay, not frozen at capture. The LLaMA KV-cache pattern
        cache[:, start:start+seqlen] = x relies on this: capture sees
        start_pos=0, but the user calls with start_pos=8, 17, ... and
        v2 has to thread the new value through.

        Pre-fix: _build_recipe_specs put non-shape SymInts into pre_binds
        and the captured value (e.g. 0) was used on every replay --
        silently wrong outputs. Fix routes such SymInts as ("I", arg_idx)
        runtime specs that read args[i] at call time."""
        def fn(x, idx):
            buf = torch.zeros(2, 32)
            buf[:, idx : idx + x.shape[1]] = x
            return buf.clone()

        x = torch.randn(2, 4)
        # wrapper=False: the aot_module path (wrapper=True) has a
        # separate, unfixed problem with non-Tensor scalar args (AOT
        # without Dynamo inlines them as literals and leaves an
        # orphan placeholder); routing fix here covers the
        # Dynamo path only.
        captured = tdcv2.capture(fn, x, 0, wrapper=False)

        for idx in (0, 4, 8, 17, 28):
            ref = fn(x, idx)
            out = captured(x, idx)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.allclose(out, ref),
                            f"idx={idx} mismatch: max diff "
                            f"{(out-ref).abs().max().item():.3e}")

    def test_varied_int_and_varied_shape_together(self):
        """Combine the two flavours of dynamic-spec routing: a Python
        int arg AND a tensor with varying shape, in the same call.
        Catches any interference between the user_scalar_queue and the
        shape-SymInt routing in _build_recipe_specs."""
        def fn(x, k):
            return x[:k] * 2.0

        x = torch.randn(8, 5)
        captured = tdcv2.capture(fn, x, 3, wrapper=False)

        for k in (1, 2, 3, 5, 7):
            for shape in [(8, 5), (12, 5), (6, 5)]:
                x = torch.randn(*shape)
                if k > x.shape[0]:
                    continue
                ref = x[:k] * 2.0
                out = captured(x, k)
                self.assertEqual(out.shape, ref.shape)
                self.assertTrue(torch.allclose(out, ref),
                                f"k={k}, shape={shape} mismatch")

    # ---- pytree output structure preservation ------------------------
    # _build_output_shaper uses torch.utils._pytree to flatten the
    # user's return value into leaves + treespec and rebuild it on
    # every replay. These tests lock in the supported shapes; adding
    # a custom container is a matter of pytree.register_pytree_node
    # at the call site, no v2 code change.

    def test_output_dict(self):
        def fn(x):
            return {"loss": x.sum(), "scaled": x * 2.0}
        captured = tdcv2.capture(fn, torch.randn(3, 4))
        x = torch.randn(3, 4)
        got = captured(x)
        ref = fn(x)
        self.assertIsInstance(got, dict)
        self.assertEqual(set(got), {"loss", "scaled"})
        self.assertTrue(torch.allclose(got["loss"], ref["loss"]))
        self.assertTrue(torch.allclose(got["scaled"], ref["scaled"]))

    def test_output_nested_dict_list_tuple(self):
        def fn(x, y):
            return {"a": [x.sin(), x.cos()], "b": (y.tan(),)}
        captured = tdcv2.capture(fn, torch.randn(3), torch.randn(3))
        x = torch.randn(3); y = torch.randn(3)
        got = captured(x, y)
        ref = fn(x, y)
        self.assertIsInstance(got, dict)
        self.assertIsInstance(got["a"], list)
        self.assertIsInstance(got["b"], tuple)
        self.assertTrue(torch.allclose(got["a"][0], ref["a"][0]))
        self.assertTrue(torch.allclose(got["a"][1], ref["a"][1]))
        self.assertTrue(torch.allclose(got["b"][0], ref["b"][0]))

    def test_output_namedtuple(self):
        from collections import namedtuple
        Result = namedtuple("Result", ["mean", "std"])

        def fn(x):
            return Result(mean=x.mean(), std=x.std())
        captured = tdcv2.capture(fn, torch.randn(8))
        x = torch.randn(8)
        got = captured(x)
        ref = fn(x)
        # pytree preserves the namedtuple subclass exactly.
        self.assertIsInstance(got, Result)
        self.assertTrue(torch.allclose(got.mean, ref.mean))
        self.assertTrue(torch.allclose(got.std, ref.std))

    def test_output_dataclass_after_pytree_register(self):
        """dataclass requires explicit registration with pytree (PyTorch
        doesn't auto-register user dataclasses). Once registered, v2's
        output_shaper threads it through transparently."""
        from dataclasses import dataclass
        import torch.utils._pytree as pytree

        @dataclass
        class Outputs:
            loss: torch.Tensor
            logits: torch.Tensor

        if not hasattr(pytree, "register_dataclass"):
            self.skipTest("pytree.register_dataclass not in this PyTorch")
        pytree.register_dataclass(Outputs)

        def fn(x):
            return Outputs(loss=x.sum(), logits=x * 2.0)
        captured = tdcv2.capture(fn, torch.randn(3, 4))
        x = torch.randn(3, 4)
        got = captured(x)
        ref = fn(x)
        self.assertIsInstance(got, Outputs)
        self.assertTrue(torch.allclose(got.loss, ref.loss))
        self.assertTrue(torch.allclose(got.logits, ref.logits))

    # ---- scaled_dot_product_attention (Optional[Tensor] arg routing)  ----
    # SDPA is the canonical case where Tensor? args reach the C++ replay
    # as None: attn_mask is Optional[Tensor]; on CPU SDPA lowers to
    # aten::_scaled_dot_product_flash_attention_for_cpu whose schema has
    # Tensor? attn_mask and float? scale. _compute_coercions used to
    # mark None at a Tensor? slot as SCALAR_TO_TENSOR, then apply_coercion
    # called iv.toScalar() on a None IValue and raised "IValue is not a
    # Scalar". The fix in _predict_value_kind + _compute_coercions
    # surfaces None as a distinct "none" kind and skips coercion when
    # the schema is Optional[T] and value is None.

    def test_sdpa_without_mask(self):
        """F.scaled_dot_product_attention with no attn_mask -- the
        attn_mask=None Optional[Tensor] arg is what used to break."""
        import torch.nn.functional as F

        def fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

        q = torch.randn(2, 4, 16, 8)
        k = torch.randn(2, 4, 16, 8)
        v = torch.randn(2, 4, 16, 8)
        ref = fn(q, k, v)
        captured = tdcv2.capture(fn, q, k, v)
        with torch.no_grad():
            got = captured(q, k, v)
        self.assertTrue(torch.allclose(got, ref, atol=1e-4))

    def test_sdpa_with_explicit_none(self):
        """Mirror timm's call site: attn_mask=None passed explicitly,
        dropout_p=0.0, is_causal=False. AOT lowers this to a graph
        where the SDPA op's Optional[Tensor] slot is fed a None IValue
        from a literal, exercising the None+OptionalType coercion path."""
        import torch.nn.functional as F

        def fn(q, k, v):
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

        q = torch.randn(2, 4, 32, 8)
        k = torch.randn(2, 4, 32, 8)
        v = torch.randn(2, 4, 32, 8)
        ref = fn(q, k, v)
        captured = tdcv2.capture(fn, q, k, v)
        with torch.no_grad():
            got = captured(q, k, v)
        self.assertTrue(torch.allclose(got, ref, atol=1e-4))

    def test_sdpa_inside_attention_module(self):
        """Reproduce timm.models.vision_transformer.Attention's shape,
        which exercises SDPA + reshape + linear sequence end-to-end.
        Catches any future regression where wrapping SDPA inside an
        nn.Module + param lift changes how AOT emits the Optional[T]
        literals."""
        import torch.nn as nn
        import torch.nn.functional as F

        class Attention(nn.Module):
            def __init__(self, dim, n_heads):
                super().__init__()
                self.n_heads = n_heads
                self.head_dim = dim // n_heads
                self.qkv = nn.Linear(dim, dim * 3, bias=False)
                self.proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(
                    B, N, 3, self.n_heads, self.head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                # The SDPA call with Optional[Tensor] attn_mask=None.
                out = F.scaled_dot_product_attention(q, k, v)
                out = out.transpose(1, 2).reshape(B, N, C)
                return self.proj(out)

        m = Attention(dim=64, n_heads=4).eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            ref = m(x)
        captured = tdcv2.capture(m, x)
        with torch.no_grad():
            got = captured(x)
        self.assertTrue(torch.allclose(got, ref, atol=1e-4),
                        f"max diff {(got-ref).abs().max().item():.3e}")

    # ---- dead-clone elimination (eliminate_dead_clones FX pass) ------
    # nn.Dropout in eval mode (and other identity-in-inference modules)
    # decomposes to aten::clone(x, None) under AOT. The clone is a real
    # per-replay memcpy with no semantic effect. eliminate_dead_clones
    # removes these but must KEEP aten::clone(x, contiguous_format)
    # used to materialize a non-contiguous view (e.g. after transpose
    # before _unsafe_view).

    def test_dead_clone_from_dropout_in_eval_is_removed(self):
        """nn.Dropout in eval mode should not contribute any clone Step
        to the captured trace. Without the pass the dropout decomp
        leaves an aten::clone(x, None) in the graph."""
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16, bias=False)
                self.drop = nn.Dropout(p=0.1)

            def forward(self, x):
                return self.drop(self.linear(x))

        m = M().eval()
        x = torch.randn(4, 16)
        captured = tdcv2.capture(m, x)
        # Trace exposes its repr; assert no aten::clone step survived.
        s = repr(captured.trace)
        self.assertNotIn(
            "aten::clone", s,
            f"unexpected clone in trace (dropout decomp not eliminated):\n{s}"
        )
        with torch.no_grad():
            self.assertTrue(torch.allclose(captured(x), m(x), atol=1e-5))

    def test_clone_with_contiguous_format_is_preserved(self):
        """A clone(memory_format=torch.contiguous_format) is real work
        the eager path also does (it's what `.contiguous()` lowers to).
        The pass must keep it; removing it would break downstream
        _unsafe_view operations that require contiguous storage."""
        def fn(x):
            # transpose is a view, contiguous() forces a copy, reshape
            # then does the layout the kernel needs. AOT typically
            # lowers this to view + clone(contiguous) + _unsafe_view.
            return x.transpose(0, 1).contiguous().reshape(-1)

        x = torch.randn(4, 5)
        captured = tdcv2.capture(fn, x)
        with torch.no_grad():
            ref = fn(x)
            got = captured(x)
        self.assertTrue(torch.allclose(got, ref))

    def test_clone_whose_result_is_mutated_in_place_is_preserved(self):
        """A clone whose result is the destination of an in-place op
        cannot be removed -- removing it would forward the mutation
        onto the source tensor. The pass's per-user schema check
        (alias_info.is_write) is what catches this."""
        def fn(x):
            y = x.clone()
            y.add_(1.0)
            return y

        x = torch.randn(8)
        x_before = x.clone()
        captured = tdcv2.capture(fn, x)
        with torch.no_grad():
            y = captured(x)
        # x must NOT have been mutated -- if the clone were elided,
        # the in-place add_ would have flowed onto x.
        self.assertTrue(torch.allclose(x, x_before),
                        "source x was mutated, indicating the clone was wrongly elided")
        self.assertTrue(torch.allclose(y, x_before + 1.0))

    # ---- AOT get_attr nodes (FX-baked Tensor constants) -------------
    # AOT sometimes emits FX get_attr nodes for tensor literals that
    # survive FakeTensorMode -- e.g. HuggingFace GPT2's KV-cache concat
    # path uses torch.empty(0) as initial cache when past_kv is None,
    # which AOT freezes as `_tensor_constant<N>` get_attr nodes.
    # _translate_get_attr routes these through v2_add_constant_tensor.

    def test_get_attr_tensor_constant(self):
        """An FX get_attr node referring to a Tensor attribute must
        flow through the trace's captured_tensors_ as a constant slot.
        Repros GPT2's `cat([torch.empty(0), x], dim=...)` pattern: a
        zero-element tensor is inlined as a graph constant; the cat
        with x then returns x unchanged."""
        const = torch.empty(0, dtype=torch.float32)

        def fn(x):
            # AOT typically routes a tensor literal seen by Dynamo
            # through a get_attr on the gm. Force the same path by
            # closing over `const`.
            return torch.cat([const, x], dim=0)

        x = torch.randn(8, 4)
        ref = fn(x)
        captured = tdcv2.capture(fn, x)
        with torch.no_grad():
            got = captured(x)
        self.assertTrue(torch.allclose(got, ref))

    # ---- SCALAR_TO_TENSOR coercion preserves scalar dtype ------------
    # at::scalar_tensor(scalar) without an explicit dtype defaults to
    # fp32. That breaks dtype propagation: `arange(seq).long() + 0`
    # where 0 is an Int literal would otherwise widen Long arange to
    # Float, silently flipping a later embedding lookup's indices
    # dtype. GPT2's positional embedding hit this; the kernel rejected
    # Float indices with "argument #1 'indices' ... got FloatTensor".

    def test_scalar_to_tensor_preserves_long_dtype(self):
        """`Long arange + Int 0` must stay Long. Without the dtype-
        preserving fix in apply_coercion's kScalarToTensor case the
        result is Float (default dtype) and downstream Long-only ops
        like embedding fail."""
        def fn(seq):
            r = torch.arange(seq, dtype=torch.long)
            return r + 0       # the 0 is an Int Python literal

        seq = 8
        ref = fn(seq)
        captured = tdcv2.capture(fn, seq)
        with torch.no_grad():
            got = captured(seq)
        self.assertEqual(got.dtype, torch.long,
                         f"expected Long, got {got.dtype}")
        self.assertTrue(torch.equal(got, ref))

    # ---- ops with bool[] schema (LayerNorm backward, etc.) -----------
    # `aten::native_layer_norm_backward`'s `output_mask: bool[3]` is the
    # canonical surface; any training workload with LayerNorm hits it
    # the moment .backward() runs. Without LIST_TO_BOOL_LIST the boxed
    # dispatcher's iv.toBoolList() raises an INTERNAL ASSERT
    # "Expected BoolList but got GenericList".

    def test_native_layer_norm_backward(self):
        """LayerNorm + .backward() exercises native_layer_norm_backward's
        `output_mask: bool[3]` schema. The captured backward graph
        emits the bool[True, True, True] literal that needs
        LIST_TO_BOOL_LIST coercion; without it the boxed dispatcher's
        iv.toBoolList() raises an INTERNAL ASSERT
        ("Expected BoolList but got GenericList")."""
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.func import functional_call

        torch.manual_seed(0)
        ln = nn.LayerNorm(8)
        params = list(ln.parameters())
        names = list(dict(ln.named_parameters()).keys())

        def fn(x, *p):
            # .sum() picks one path through the bw graph that
            # exercises native_layer_norm_backward.
            return functional_call(ln, dict(zip(names, p)), x).sum()

        x = torch.randn(4, 8, requires_grad=True)
        captured = tdcv2.capture(fn, x, *params, allow_grad=True)
        # capture-time example call already did a .backward(); clear
        # grads so v2_grads reflects exactly one replay's accumulation.
        for p in params:
            p.grad = None
        loss = captured(x, *params)
        loss.backward()
        v2_grads = [p.grad.clone() for p in params]
        # Eager reference: clear grads, re-run, compare.
        for p in params:
            p.grad = None
        ref_loss = fn(x, *params)
        ref_loss.backward()
        for p, v2_grad in zip(params, v2_grads):
            self.assertTrue(
                p.grad is not None
                and torch.allclose(p.grad, v2_grad, atol=1e-4),
                f"grad mismatch on shape {tuple(p.shape)}: max diff "
                f"{(p.grad - v2_grad).abs().max().item():.3e}"
            )
        self.assertTrue(torch.allclose(ref_loss, loss, atol=1e-4))

    # ---- aten::index.Tensor with Tensor?[] indices (Optional list) ---
    # The schema is `aten::index.Tensor(Tensor self, Tensor?[] indices)`.
    # `Tensor?[]` is `List<Optional<Tensor>>`; without
    # LIST_TO_OPTIONAL_TENSOR_LIST coercion the boxed dispatcher
    # raises "Tried to cast a List<Any> to a List<Tensor?>".

    def test_index_with_tensor_indices(self):
        """Fancy indexing with a tensor: x[idx]. AOT lowers this to
        aten::index.Tensor whose `indices` arg expects
        List<Optional<Tensor>>. The translator must emit
        LIST_TO_OPTIONAL_TENSOR_LIST so apply_coercion builds the
        strongly-typed list at replay."""
        def fn(x, idx):
            return x[idx]

        x = torch.randn(10, 4)
        idx = torch.tensor([0, 2, 5])
        ref = fn(x, idx)
        captured = tdcv2.capture(fn, x, idx)
        with torch.no_grad():
            got = captured(x, idx)
        self.assertTrue(torch.allclose(got, ref))

    def test_index_with_none_in_indices_list(self):
        """x[:, idx] form passes a None alongside a Tensor in the
        indices list. The Optional<Tensor> list must accept both,
        with None resolving to std::nullopt in C++ apply_coercion."""
        def fn(x, idx):
            return x[:, idx]

        x = torch.randn(6, 8)
        idx = torch.tensor([1, 3, 7])
        ref = fn(x, idx)
        captured = tdcv2.capture(fn, x, idx)
        with torch.no_grad():
            got = captured(x, idx)
        self.assertTrue(torch.allclose(got, ref))

    def test_output_none_mixed_with_tensor(self):
        """None leaves intermixed with Tensors are preserved by the
        shaper without consuming a trace_out slot."""
        def fn(x):
            return (x.sum(), None, x.mean())
        captured = tdcv2.capture(fn, torch.randn(4))
        x = torch.randn(4)
        got = captured(x)
        ref = fn(x)
        self.assertIsInstance(got, tuple)
        self.assertEqual(len(got), 3)
        self.assertIsNone(got[1])
        self.assertTrue(torch.allclose(got[0], ref[0]))
        self.assertTrue(torch.allclose(got[2], ref[2]))


class TestV2DataDependentShape(unittest.TestCase):
    """v2 + ops whose OUTPUT SHAPE depends on input DATA values, not
    just input shapes (masked_select, nonzero, unique, etc.).

    Background: Dynamo refuses to trace these ops by default --
    they're called "dynamic output shape ops" and they graph-break
    with a hint to set `capture_dynamic_output_shape_ops = True`.
    Once that's enabled Dynamo emits an "unbacked SymInt" for the
    output size and the AOT FX graph carries it through. v2's trace
    + C++ replay then handles the runtime size automatically because
    each step reads its inputs' current metadata from the live
    Tensor objects (the same mechanism that gives v1 dynamic-shape
    coverage for dim-index ops).

    These tests validate that:
      1. With the flag on, capture succeeds (no graph break).
      2. Replays with masks/inputs that select DIFFERENT numbers of
         elements all produce outputs of the correct runtime shape.
      3. Values match eager exactly.
    """

    def setUp(self):
        torch._dynamo.reset()

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_masked_select_varies_output_size(self):
        def fn(x, mask):
            return torch.masked_select(x, mask)

        x_ex = torch.arange(12, dtype=torch.float32)
        mask_ex = x_ex > 5                  # 6 True
        captured = tdcv2.capture(fn, x_ex, mask_ex)

        # Same args -> 6 outputs
        out = captured(x_ex, mask_ex)
        self.assertEqual(tuple(out.shape), (6,))
        self.assertTrue(torch.equal(out, fn(x_ex, mask_ex)))

        # Different masks at same input shape: output size flexes with
        # the number of True values.
        for cutoff, expected_count in [(3, 8), (-1, 12), (100, 0)]:
            mask = x_ex > cutoff
            out = captured(x_ex, mask)
            self.assertEqual(
                tuple(out.shape), (expected_count,),
                f"cutoff={cutoff}: got shape {tuple(out.shape)}, expected ({expected_count},)",
            )
            self.assertTrue(torch.equal(out, fn(x_ex, mask)))

        # Different INPUT shape AND different True count.
        x2 = torch.arange(20, dtype=torch.float32)
        out = captured(x2, x2 > 10)         # 9 True
        self.assertEqual(tuple(out.shape), (9,))
        self.assertTrue(torch.equal(out, fn(x2, x2 > 10)))

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_masked_select_downstream_consumer(self):
        """The data-dependent SymInt produced by masked_select must
        flow correctly into a downstream op that consumes the runtime
        size (here: .sum() over the variably-sized output)."""
        def fn(x, mask):
            return torch.masked_select(x, mask).sum()

        x = torch.arange(12, dtype=torch.float32)
        captured = tdcv2.capture(fn, x, x > 5)

        for cutoff in [5, 3, -1, 100]:        # 6, 8, 12, 0 True
            mask = x > cutoff
            got = captured(x, mask)
            ref = fn(x, mask)
            self.assertTrue(
                torch.allclose(got, ref),
                f"cutoff={cutoff}: v2={got.item()} eager={ref.item()}",
            )

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_indices(self):
        """torch.nonzero is the canonical data-dependent op: output
        is (count_of_true, ndim)."""
        def fn(mask):
            return torch.nonzero(mask)

        mask_ex = torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool)
        captured = tdcv2.capture(fn, mask_ex)

        for mask, expected_count in [
            (torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool), 3),
            (torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool), 5),
            (torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool), 0),
        ]:
            out = captured(mask)
            self.assertEqual(out.shape, (expected_count, 1))
            self.assertTrue(torch.equal(out, fn(mask)))

    def test_masked_select_without_flag_errors_clearly(self):
        """Without capture_dynamic_output_shape_ops=True the capture
        should fail predictably -- this is the "diagnostic guard"
        test so a future PyTorch version that flips the default
        doesn't quietly break this expectation."""
        # Default config: Dynamo graph-breaks on masked_select.
        def fn(x, mask):
            return torch.masked_select(x, mask)
        x = torch.arange(8, dtype=torch.float32)
        with self.assertRaises(RuntimeError) as cm:
            tdcv2.capture(fn, x, x > 3)
        # We don't pin the exact message (Dynamo phrasing evolves);
        # it should be the "fw_compiler was never called" form since
        # Dynamo bailed before getting to AOT.
        msg = str(cm.exception)
        self.assertTrue(
            "fw_compiler was never called" in msg
            or "graph" in msg.lower(),
            f"unexpected error: {msg}",
        )


if __name__ == "__main__":
    unittest.main()
