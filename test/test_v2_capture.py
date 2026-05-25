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


if __name__ == "__main__":
    unittest.main()
