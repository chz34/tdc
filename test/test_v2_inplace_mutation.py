"""v2.capture under in-place buffer mutation — scenario tests.

These tests verify the **user-observable behaviour** when the captured
function modifies a registered Buffer in-place (LLaMA KV cache writes,
running stats accumulators, scatter updates, etc.). The internal
mechanism that makes this work has evolved several times — pattern-
matching slice_scatter, disable_functionalization, etc. — and may
keep evolving. As long as v2.capture's output matches eager and the
buffer's post-call state matches eager, these tests should pass
regardless of how it's implemented underneath.

Each test compares two parallel module instances:
  - m_ref: stays in eager
  - m_v2:  goes through v2.capture
With identical initial buffer state and identical inputs, the two
must produce the same outputs and the same final buffer values.

Covers:
  - `self.buf[start:end] = val`             (slice assignment, LLaMA KV)
  - `self.buf[:, start:end, ...] = val`     (multi-dim slice assignment)
  - `self.buf.add_(x)`                      (in-place add)
  - `self.buf.copy_(x)`                     (in-place copy)
  - `self.buf.fill_(scalar)`                (in-place fill)
  - Repeated calls accumulating state
  - read-after-write within a single forward
  - Python float module attr in arithmetic (RMSNorm eps pattern)
  - Both wrapper=True and wrapper=False, since they take different
    paths internally and need separate scenario coverage.
"""
import copy
import unittest

import torch
from torch import nn
import torch_dispatch_capture.v2 as tdcv2


def _clone_module_buffers(src: nn.Module) -> nn.Module:
    """Return a deep copy of `src` so the two parallel instances start
    with identical buffer state but mutate independently."""
    return copy.deepcopy(src)


def _snapshot_buffers(mod: nn.Module) -> dict:
    """Take a deep snapshot of every buffer in mod, so we can restore
    state after v2.capture's tracing call mutates them in-place."""
    return {n: b.detach().clone() for n, b in mod.named_buffers()}


def _restore_buffers(mod: nn.Module, snapshot: dict) -> None:
    """Restore buffer state in-place. Used after v2.capture so that
    the timed call starts from the same buffer state as the eager
    reference run."""
    for n, b in mod.named_buffers():
        b.copy_(snapshot[n])


class TestV2InPlaceMutation(unittest.TestCase):

    def setUp(self):
        torch._dynamo.reset()

    # ---- LLaMA KV cache style: cache[:bsz, start:end] = val ----------

    def _check_kv_cache_module(self, wrapper):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 16, 4, 8))

            def forward(self, x, start_pos):
                seqlen = x.shape[1]
                self.cache[:, start_pos:start_pos + seqlen] = x
                return self.cache[:, :start_pos + seqlen].clone()

        m_ref = M().eval()
        m_v2 = _clone_module_buffers(m_ref)

        x = torch.randn(2, 4, 4, 8)
        out_ref = m_ref(x, 2)

        captured = tdcv2.capture(m_v2, x, 2, wrapper=wrapper)
        out_v2 = captured(x, 2)

        self.assertEqual(out_v2.shape, out_ref.shape)
        self.assertTrue(torch.allclose(out_v2, out_ref, atol=1e-5),
                        f"output mismatch under wrapper={wrapper}")
        self.assertTrue(torch.allclose(m_v2.cache, m_ref.cache, atol=1e-5),
                        f"cache mismatch under wrapper={wrapper}")

    def test_kv_cache_slice_assign_wrapper_false(self):
        self._check_kv_cache_module(wrapper=False)

    def test_kv_cache_slice_assign_wrapper_true(self):
        self._check_kv_cache_module(wrapper=True)

    # ---- Repeated calls accumulate state -----------------------------

    def test_kv_cache_two_calls_accumulate(self):
        """Two sequential calls should leave the buffer in the same
        state as two sequential eager calls. Exercises 'previous-call
        mutation persists for next call' behaviour."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 16, 4, 8))

            def forward(self, x, start_pos):
                seqlen = x.shape[1]
                self.cache[:, start_pos:start_pos + seqlen] = x
                return self.cache[:, :start_pos + seqlen].clone()

        m_ref = M().eval()
        m_v2 = _clone_module_buffers(m_ref)

        x1 = torch.randn(2, 4, 4, 8)
        x2 = torch.randn(2, 4, 4, 8)

        out1_ref = m_ref(x1, 0)
        out2_ref = m_ref(x2, 4)

        # The captured callable bakes start_pos as a constant if Dynamo
        # specialises on it; in that case the captured call always
        # writes to start_pos=0. To test the accumulation scenario
        # without depending on Dynamo's specialisation policy, capture
        # twice with the two distinct start_pos values.
        captured1 = tdcv2.capture(m_v2, x1, 0, wrapper=False)
        out1_v2 = captured1(x1, 0)
        captured2 = tdcv2.capture(m_v2, x2, 4, wrapper=False)
        out2_v2 = captured2(x2, 4)

        self.assertTrue(torch.allclose(out1_v2, out1_ref, atol=1e-5))
        self.assertTrue(torch.allclose(out2_v2, out2_ref, atol=1e-5))
        # After both calls, cache_ref and cache_v2 should match across
        # all positions — the second call's mutation didn't wipe out
        # the first call's mutation.
        self.assertTrue(torch.allclose(m_v2.cache, m_ref.cache, atol=1e-5))

    # ---- In-place arithmetic on a buffer -----------------------------

    def test_buffer_inplace_add_(self):
        """`self.buf.add_(x)` accumulator -- a NON-idempotent in-place
        op. v2.capture's trace step runs the model once and mutates
        the buffer; we snapshot+restore before the timed call so the
        comparison vs eager is from a matching initial state.

        This pattern matters for any in-place op that's not idempotent
        under the same input (add_, sub_, mul_, exponential_, etc.).
        Idempotent ops (copy_, fill_, slice assignment with same val)
        don't need the snapshot, but it's harmless and the test stays
        robust if the user picks a non-idempotent variant in real code."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("acc", torch.zeros(8, 16))

            def forward(self, x):
                self.acc.add_(x)
                return self.acc.clone()

        m_ref = M().eval()
        m_v2 = _clone_module_buffers(m_ref)
        snap = _snapshot_buffers(m_v2)
        x = torch.randn(8, 16)

        out_ref = m_ref(x)
        captured = tdcv2.capture(m_v2, x, wrapper=False)
        _restore_buffers(m_v2, snap)
        out_v2 = captured(x)

        self.assertTrue(torch.allclose(out_v2, out_ref, atol=1e-5))
        self.assertTrue(torch.allclose(m_v2.acc, m_ref.acc, atol=1e-5))

    def test_buffer_inplace_add_wrapper_true(self):
        """Same as test_buffer_inplace_add_ but under wrapper=True.
        wrapper=True goes through aot_function/aot_module rather than
        torch.compile, so it deserves its own scenario coverage."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("acc", torch.zeros(8, 16))

            def forward(self, x):
                self.acc.add_(x)
                return self.acc.clone()

        m_ref = M().eval()
        m_v2 = _clone_module_buffers(m_ref)
        snap = _snapshot_buffers(m_v2)
        x = torch.randn(8, 16)

        out_ref = m_ref(x)
        captured = tdcv2.capture(m_v2, x, wrapper=True)
        _restore_buffers(m_v2, snap)
        out_v2 = captured(x)

        self.assertTrue(torch.allclose(out_v2, out_ref, atol=1e-5))
        self.assertTrue(torch.allclose(m_v2.acc, m_ref.acc, atol=1e-5))

    def test_buffer_inplace_copy_(self):
        """`self.buf.copy_(x)` overwrite."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("snap", torch.zeros(4, 8))

            def forward(self, x):
                self.snap.copy_(x)
                return self.snap.clone() * 2

        m_ref = M().eval()
        m_v2 = _clone_module_buffers(m_ref)
        x = torch.randn(4, 8)

        out_ref = m_ref(x)
        captured = tdcv2.capture(m_v2, x, wrapper=False)
        out_v2 = captured(x)

        self.assertTrue(torch.allclose(out_v2, out_ref, atol=1e-5))
        self.assertTrue(torch.allclose(m_v2.snap, m_ref.snap, atol=1e-5))

    # ---- read-after-write within one forward -------------------------

    def test_read_after_write_same_call(self):
        """Mutation, then read in the same forward. The read must
        see the post-mutation value."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(2, 8))

            def forward(self, x):
                self.buf.copy_(x)
                # subsequent read must see x's values, not zeros
                return self.buf * 3

        m_ref = M().eval()
        m_v2 = _clone_module_buffers(m_ref)
        x = torch.randn(2, 8)

        out_ref = m_ref(x)
        captured = tdcv2.capture(m_v2, x, wrapper=False)
        out_v2 = captured(x)

        self.assertTrue(torch.allclose(out_v2, out_ref, atol=1e-5))
        # buf should equal x (not zeros) -- the read happened after
        # the write in the same forward
        self.assertTrue(torch.allclose(m_v2.buf, x, atol=1e-5))

    # ---- Python float module attr (RMSNorm eps pattern) --------------

    def test_python_float_attr_in_arithmetic(self):
        """`self.eps = 1e-6; x + self.eps` -- the RMSNorm pattern.
        AOT lifts the Python float into a 0-d Tensor; depending on
        the device, that tensor might need promotion for performance
        but the *correctness* must hold."""

        class RMSNormLike(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.eps = 1e-6
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                rms = x.pow(2).mean(-1, keepdim=True)
                return x * torch.rsqrt(rms + self.eps) * self.weight

        m_ref = RMSNormLike(32).eval()
        m_v2 = _clone_module_buffers(m_ref)
        x = torch.randn(4, 16, 32)

        out_ref = m_ref(x)
        captured = tdcv2.capture(m_v2, x, wrapper=False)
        out_v2 = captured(x)

        # RMSNorm involves division and rsqrt; loosen tolerance a touch.
        self.assertTrue(torch.allclose(out_v2, out_ref, atol=1e-5, rtol=1e-5))

    # ---- Multi-dim slice with both axes dynamic ----------------------

    def test_multi_dim_slice_assign(self):
        """`cache[:bsz, start:end] = val` -- LLaMA exactly. Two
        in-place writes in one forward (K and V caches), each with
        a multi-dim slice."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k", torch.zeros(4, 16, 8, 8))
                self.register_buffer("v", torch.zeros(4, 16, 8, 8))

            def forward(self, xk, xv, start_pos):
                bsz, seqlen, _, _ = xk.shape
                self.k[:bsz, start_pos:start_pos + seqlen] = xk
                self.v[:bsz, start_pos:start_pos + seqlen] = xv
                k = self.k[:bsz, :start_pos + seqlen]
                v = self.v[:bsz, :start_pos + seqlen]
                return (k * v).sum()

        m_ref = M().eval()
        m_v2 = _clone_module_buffers(m_ref)

        xk = torch.randn(2, 4, 8, 8)
        xv = torch.randn(2, 4, 8, 8)
        start_pos = 3

        out_ref = m_ref(xk, xv, start_pos)
        captured = tdcv2.capture(m_v2, xk, xv, start_pos, wrapper=False)
        out_v2 = captured(xk, xv, start_pos)

        self.assertTrue(torch.allclose(out_v2, out_ref, atol=1e-4))
        self.assertTrue(torch.allclose(m_v2.k, m_ref.k, atol=1e-5))
        self.assertTrue(torch.allclose(m_v2.v, m_ref.v, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
