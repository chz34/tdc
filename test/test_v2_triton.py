"""v1 + v2 capture/replay over a Triton custom op.

Triton kernels register with the dispatcher via torch.library.triton_op
and appear as ordinary OpOverload nodes in the FX graph (and as ordinary
Step entries in v1's trace). These tests lock in that integration so
future translator / fallback changes don't quietly break opaque-op
handling.

The whole module skips when Triton has no usable driver for the current
TDC_DEVICE -- typically a CPU-only torch build. On CUDA / NPU / XPU
the tests run end-to-end (real kernel launch in both capture and
replay) and verify the result matches eager.

Both `@triton.jit` and `@torch.library.triton_op` are evaluated at
module load: the former requires the kernel be defined in a real
Python file, and the latter performs a global op registration. The
registration ns/name (`tdc_triton_test::add_triton`) is unique to
this file so it can't collide with another test file's triton_op.
"""
import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))
from _device import DEVICE, SYNC


# ---------------------------------------------------------------------------
# Triton availability detection
# ---------------------------------------------------------------------------
def _detect_triton_available() -> tuple[bool, str]:
    """Return (ok, reason). Detects whether Triton can actually launch
    a kernel on DEVICE. Two failure modes we care about:

      1. triton import fails (package not installed).
      2. triton imports but `driver.active` raises (no usable backend
         for this device -- the common case on CPU-only torch builds).

    We do NOT attempt a real launch here (the @triton.jit decorator
    requires module-level definition which we can't do conditionally
    without restructuring imports). Instead we probe the driver state,
    which mirrors what generate_ttir() does internally."""
    try:
        import triton  # noqa: F401
    except ImportError as e:
        return False, f"triton not installed ({e})"
    try:
        target = triton.runtime.driver.active.get_current_target()
        return True, f"target={target}"
    except Exception as e:  # noqa: BLE001
        return False, (f"triton has no active driver for {DEVICE.type} "
                       f"({type(e).__name__}: {str(e)[:80]})")


TRITON_AVAILABLE, TRITON_REASON = _detect_triton_available()


# ---------------------------------------------------------------------------
# Kernel + triton_op registration
#
# Always evaluate this at module load (the @triton.jit decorator needs
# `inspect.getsourcelines(fn)` to succeed, which requires a real file).
# Even when TRITON_AVAILABLE is False the import side-effects are
# harmless -- registration completes but no test exercises the op.
# ---------------------------------------------------------------------------
if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(
        x_ptr, y_ptr, output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)

    @torch.library.triton_op("tdc_triton_test::add_triton", mutates_args={})
    def add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
        torch.library.wrap_triton(add_kernel)[grid](
            x, y, output, n_elements, BLOCK_SIZE=128
        )
        return output

    @add_triton.register_fake
    def _(x, y):
        return torch.empty_like(x)
else:
    # Sentinel so tests that reference add_triton at class-body time
    # still parse. Skip decorator on the class prevents these from
    # actually running.
    add_triton = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _rand(*shape):
    return torch.randn(*shape, device=DEVICE)


# ---------------------------------------------------------------------------
# v2 tests
# ---------------------------------------------------------------------------
@unittest.skipUnless(TRITON_AVAILABLE, f"triton/driver unavailable: {TRITON_REASON}")
class TestV2Triton(unittest.TestCase):
    """Validate that v2.capture treats torch.library.triton_op the same
    as any other dispatcher-registered op: the FX graph carries a single
    OpOverload node, translator emits a single kTensorOp Step, and
    replay re-dispatches to the triton wrapper which launches the
    underlying kernel."""

    def setUp(self):
        torch._dynamo.reset()

    def test_eager_baseline(self):
        """Sanity: the triton_op runs in eager and matches CPU/aten
        semantics. If this fails the rest of the suite can't be
        trusted."""
        x = _rand(256)
        y = _rand(256)
        got = add_triton(x, y)
        SYNC()
        self.assertTrue(torch.allclose(got, x + y, atol=1e-5))

    def test_v2_direct(self):
        """wrapper=False (default) path. The captured trace records a
        kTensorOp Step for `tdc_triton_test::add_triton` and replays
        it through op.callBoxed -> dispatcher -> triton wrapper."""
        import torch_dispatch_capture.v2 as tdcv2

        def fn(x, y):
            return add_triton(x, y) * 2.0

        x = _rand(64, 64)
        y = _rand(64, 64)
        ref = fn(x, y)
        SYNC()

        captured = tdcv2.capture(fn, x, y, wrapper=False)
        got = captured(x, y)
        SYNC()
        self.assertTrue(torch.allclose(got, ref, atol=1e-5),
                        f"max diff {(got - ref).abs().max().item():.3e}")

    def test_v2_wrapper(self):
        """wrapper=True path goes through aot_module. The user fn here
        has only Tensor args (no Python scalars), so the loud-fail
        check in capture() lets it through and we exercise the RuntimeWrapper
        call path over the same triton-emitting graph."""
        import torch_dispatch_capture.v2 as tdcv2

        def fn(x, y):
            return add_triton(x, y) * 2.0

        x = _rand(64, 64)
        y = _rand(64, 64)
        ref = fn(x, y)
        SYNC()

        captured = tdcv2.capture(fn, x, y, wrapper=True)
        got = captured(x, y)
        SYNC()
        self.assertTrue(torch.allclose(got, ref, atol=1e-5))

    def test_v2_triton_with_varied_int(self):
        """triton_op composes with the ("I", arg_idx) Python-scalar
        routing introduced for KV-cache-style patterns. Capture at
        scale=2, replay across {1, 2, 5} and verify each."""
        import torch_dispatch_capture.v2 as tdcv2

        def fn(x, y, scale):
            return add_triton(x, y) * scale

        x = _rand(32, 32)
        y = _rand(32, 32)
        captured = tdcv2.capture(fn, x, y, 2, wrapper=False)
        for scale in (1, 2, 5):
            ref = fn(x, y, scale)
            SYNC()
            got = captured(x, y, scale)
            SYNC()
            self.assertTrue(torch.allclose(got, ref, atol=1e-5),
                            f"scale={scale} max diff "
                            f"{(got - ref).abs().max().item():.3e}")

    def test_v2_triton_opaque_in_fx_graph(self):
        """Structural assertion: the AOT FX graph must carry exactly one
        call_function node whose target is our triton_op (NOT one
        decomposed into wrap_triton + empty_like + ...). If this fails,
        AOT or torch.library.triton_op changed how triton ops are
        represented and the translator may need to adapt."""
        from torch._dynamo.backends.common import aot_autograd

        captured_gms = []
        def grab(gm, sample_inputs):
            captured_gms.append(gm)
            def noop(*a):
                return [torch.zeros(o.shape, device=DEVICE)
                        if hasattr(o, "shape") else o
                        for o in gm.graph.output_node().args[0]]
            return noop

        def fn(x, y):
            return add_triton(x, y)

        compiled = torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=grab,
                                 disable_functionalization=True),
            dynamic=True,
        )
        with torch.no_grad():
            try:
                compiled(_rand(32), _rand(32))
            except Exception:
                # The noop callable may have shape mismatches; ignore.
                pass

        self.assertTrue(captured_gms, "AOT did not invoke grab_compiler")
        gm = captured_gms[0]
        triton_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            #and str(n.target).startswith("tdc_triton_test::")
            and ("tdc_triton_test::" in str(n.target)
                 or "tdc_triton_test." in str(n.target))
        ]
        self.assertEqual(
            len(triton_nodes), 1,
            f"expected exactly one tdc_triton_test op node, got "
            f"{[n.target for n in triton_nodes]}"
        )


# ---------------------------------------------------------------------------
# v1 tests
# ---------------------------------------------------------------------------
@unittest.skipUnless(TRITON_AVAILABLE, f"triton/driver unavailable: {TRITON_REASON}")
class TestV1Triton(unittest.TestCase):
    """v1's GenericMode boxed fallback should intercept `tdc_triton_test::
    add_triton` at the dispatcher boundary and record it as a single
    Step. The fallback's redispatch then runs the actual implementation
    (Python wrapper -> wrap_triton -> triton launch) to produce the
    captured output tensor."""

    def test_v1_capture_then_replay(self):
        """End-to-end v1 path."""
        import torch_dispatch_capture as tdc

        def fn(x, y, buf):
            buf.copy_(add_triton(x, y) * 2.0)

        x = _rand(64, 64)
        y = _rand(64, 64)
        buf = torch.empty_like(x)

        # Capture writes into buf; replay overwrites buf in-place
        # (v1's design: captured_out is the same TensorImpl across
        # replays for in-place / out= patterns).
        with torch.no_grad(), tdc.capture() as trace:
            fn(x, y, buf)
        SYNC()

        expected_buf = (x + y) * 2.0
        self.assertTrue(torch.allclose(buf, expected_buf, atol=1e-5))

        # Mutate inputs, then replay -- buf should pick up the new sum.
        x2 = _rand(64, 64)
        x.copy_(x2)
        trace.replay()
        SYNC()
        new_expected = (x2 + y) * 2.0
        self.assertTrue(torch.allclose(buf, new_expected, atol=1e-5),
                        f"max diff {(buf - new_expected).abs().max().item():.3e}")


if __name__ == "__main__":
    unittest.main()
