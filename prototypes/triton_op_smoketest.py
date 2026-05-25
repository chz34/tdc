"""Triton custom op + v2.capture interaction probe.

Goal: register a Triton kernel via torch.library.triton_op, then
observe how v2.capture's pipeline treats it -- specifically:

  1. Does the triton_op appear as a single OpOverload node in
     the AOT FX graph (vs being lowered / decomposed) ?
  2. Does python/v2/translator.translate_graph accept it without
     raising (it treats every OpOverload as a kTensorOp Step) ?
  3. Does v2.capture (wrapper=False) produce a callable whose
     replay output matches eager?
  4. Does start_pos-style varied scalar args still work when
     mixed with triton_op call sites?

This script is environment-aware:
  - Needs Triton installed (just `import triton` at module load).
  - The .jit kernel attempts a real launch. On CPU-only torch
    builds Triton has no usable backend and will raise at launch
    time; we then short-circuit to a graph-only inspection so the
    pipeline analysis can still run.

Run:
    python prototypes/triton_op_smoketest.py
"""
from __future__ import annotations

import sys
import os

import torch
import triton
import triton.language as tl

# Make our project importable from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch_dispatch_capture as tdc       # noqa: E402  # v1
import torch_dispatch_capture.v2 as tdcv2  # noqa: E402


# ---------------------------------------------------------------------------
# Triton kernel + custom op registration
# ---------------------------------------------------------------------------
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@torch.library.triton_op("mylib::add_triton", mutates_args={})
def add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add two tensors via a Triton kernel. The triton_op decorator
    wires this into PyTorch's dispatcher so torch.compile / AOT see
    the call as a single OpOverload node, and the fake-tensor /
    functionalize / autograd plumbing works automatically.

    The wrap_triton call below is what tells torch.compile this is
    a triton kernel (vs a regular custom Python op): the compiler
    can then either lower it through Inductor or pass it through
    unchanged."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    torch.library.wrap_triton(add_kernel)[grid](
        x, y, output, n_elements, BLOCK_SIZE=128
    )
    return output


# fake/meta function so AOT trace + FakeTensorMode know the output
# shape/dtype without actually launching the kernel.
@add_triton.register_fake
def _(x, y):
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def probe_eager_launch() -> bool:
    """Returns True if Triton can actually launch on the local device.
    On a CPU-only torch build Triton has no usable backend and raises
    at the first launch -- we then degrade to graph-only inspection."""
    try:
        x = torch.randn(1024)
        y = torch.randn(1024)
        out = add_triton(x, y)
        ok = torch.allclose(out, x + y, atol=1e-5)
        print(f"# probe_eager_launch: launched on {x.device}, allclose={ok}")
        return ok
    except Exception as e:
        print(f"# probe_eager_launch: launch failed ({type(e).__name__}: "
              f"{str(e)[:160]}...)")
        return False


def probe_fx_graph_shape() -> None:
    """Capture the FX graph via aot_autograd and dump every node so we
    can see how triton_op shows up. We use a minimal grab_compiler that
    only inspects the graph -- never runs it -- so this works even
    when Triton can't launch."""
    from torch._dynamo.backends.common import aot_autograd

    captured = []
    def grab(gm, sample_inputs):
        captured.append(gm)
        # Return a callable that errors if invoked. We only need the
        # graph; we don't want to trigger a launch.
        def must_not_run(*a, **kw):
            raise RuntimeError("graph-only probe: don't run me")
        return must_not_run

    def fn(x, y):
        return add_triton(x, y) * 2.0

    compiled = torch.compile(
        fn,
        backend=aot_autograd(
            fw_compiler=grab,
            # Match v2.capture(wrapper=False)'s settings so what we
            # see here is what v2 sees.
            disable_functionalization=True,
        ),
        dynamic=True,
    )
    try:
        with torch.no_grad():
            compiled(torch.randn(32, 32), torch.randn(32, 32))
    except RuntimeError as e:
        if "graph-only probe" not in str(e):
            raise

    if not captured:
        print("# probe_fx_graph_shape: no graph was captured (graph break?)")
        return

    gm = captured[0]
    print("\n# FX graph (under aot_autograd + disable_functionalization=True):")
    print(gm.graph)
    print("\n# Per-node breakdown:")
    for n in gm.graph.nodes:
        if n.op == "call_function":
            t = n.target
            kind = (type(t).__name__
                    if not callable(t) else
                    getattr(t, "__qualname__", repr(t)))
            print(f"  {n.name:<16}  op={n.op:<14}  target={kind}")
        else:
            print(f"  {n.name:<16}  op={n.op:<14}")


def probe_translator_acceptance() -> None:
    """Run python/v2/translator.translate_graph on the captured graph,
    confirming it accepts the triton_op node (treats it as a kTensorOp
    Step under the OpOverload generic path)."""
    from torch._dynamo.backends.common import aot_autograd
    from torch_dispatch_capture.v2 import translator as t_mod

    captured = []
    def grab(gm, sample_inputs):
        captured.append(gm)
        def must_not_run(*a, **kw):
            raise RuntimeError("graph-only probe: don't run me")
        return must_not_run

    def fn(x, y):
        return add_triton(x, y) * 2.0

    compiled = torch.compile(
        fn,
        backend=aot_autograd(fw_compiler=grab, disable_functionalization=True),
        dynamic=True,
    )
    try:
        with torch.no_grad():
            compiled(torch.randn(32, 32), torch.randn(32, 32))
    except RuntimeError as e:
        if "graph-only probe" not in str(e):
            raise

    gm = captured[0]
    try:
        trace = t_mod.translate_graph(gm)
        print(f"\n# probe_translator_acceptance: translate_graph succeeded.")
        print(f"  trace object: {trace!r}")
    except Exception as e:
        print(f"\n# probe_translator_acceptance: translate_graph FAILED.")
        print(f"  {type(e).__name__}: {str(e)[:200]}")


def probe_end_to_end_capture(can_launch: bool) -> None:
    """The full v2.capture(wrapper=False) -> replay pipeline. Only
    meaningful if Triton can actually launch on the local device."""
    if not can_launch:
        print("\n# probe_end_to_end_capture: SKIPPED (Triton can't launch on "
              "this build; capture would succeed but replay would error at "
              "kernel launch time -- same failure as eager).")
        return

    def fn(x, y):
        return add_triton(x, y) * 2.0

    x = torch.randn(64, 64)
    y = torch.randn(64, 64)

    ref = fn(x, y)
    captured = tdcv2.capture(fn, x, y)
    got = captured(x, y)

    ok = torch.allclose(got, ref, atol=1e-5)
    print(f"\n# probe_end_to_end_capture: replay vs eager allclose={ok}")
    if not ok:
        print(f"  max abs diff: {(got - ref).abs().max().item():.3e}")


def probe_v1_with_meta_tensors() -> None:
    """Structural-only v1 probe that doesn't need a working Triton
    backend. Use meta tensors as inputs; triton_op's register_fake
    impl runs instead of launching the kernel. The boxed fallback
    is supposed to fire BEFORE backend dispatch (GenericMode priority
    3, before Meta), so we should still see the op recorded as one
    Step. Useful to answer the "does v1 see triton_op as opaque?"
    question without GPU."""
    x = torch.empty(64, 64, device="meta")
    y = torch.empty(64, 64, device="meta")

    def fn(x, y):
        # No trailing multiply -- we want to see exactly how many
        # Steps the triton_op call decomposes into.
        return add_triton(x, y)

    try:
        with torch.no_grad(), tdc.capture() as trace:
            captured_out = fn(x, y)
    except Exception as e:
        print(f"\n# probe_v1_with_meta_tensors: capture raised "
              f"({type(e).__name__}: {str(e)[:160]})")
        return

    print(f"\n# probe_v1_with_meta_tensors (meta tensors, no real launch):")
    print(f"  user-visible output: shape={tuple(captured_out.shape)} "
          f"device={captured_out.device}")
    print(f"  trace summary: {trace}")


def probe_v1_capture(can_launch: bool) -> None:
    """v1 capture works at the dispatcher boxed-fallback level. The
    interesting questions are:

      1. Does the GenericMode fallback intercept mylib::add_triton.default
         as a single Step (treating it opaquely) -- or does it recurse
         into the wrapper's body (empty_like + the HOP launch)?
      2. If it does intercept, does replay correctly re-launch the
         kernel via op.callBoxed -> dispatcher -> triton_op impl?

    We inspect the captured trace's Step list to answer #1; the actual
    replay (#2) needs Triton to be launchable on the local device."""
    if not can_launch:
        # The capture step itself needs to call the kernel (the boxed
        # fallback re-dispatches after recording) so a launch failure
        # blocks capture too. Skip but explain.
        print("\n# probe_v1_capture: SKIPPED (Triton can't launch on this "
              "build; v1's boxed fallback needs a working call to capture "
              "the Step's output identity).")
        return

    def fn(x, y):
        return add_triton(x, y) * 2.0

    x = torch.randn(64, 64)
    y = torch.randn(64, 64)

    with tdc.capture() as trace:
        captured_out = fn(x, y)

    print(f"\n# probe_v1_capture: trace = {trace!r}")
    print(f"  user-visible output shape: {tuple(captured_out.shape)}")
    print(f"  Steps recorded by v1 fallback:")
    # Dump each step's op name. v1's Trace exposes step inspection
    # through __repr__; for a structured view we'd need bindings.
    # The op count alone tells us whether triton_op was captured as
    # ONE step (opaque) or expanded (we'd see empty_like + the HOP).
    print(f"    {trace}")

    # Eager re-run for reference. Note this MUTATES the same captured
    # output buffer because v1's replay writes into it in-place.
    ref = fn(x, y)
    trace.replay()
    ok = torch.allclose(captured_out, ref, atol=1e-5)
    print(f"  replay vs eager allclose: {ok}")
    if not ok:
        print(f"  max abs diff: {(captured_out - ref).abs().max().item():.3e}")


def probe_combined_int_arg(can_launch: bool) -> None:
    """Combine triton_op with a varied int arg, to make sure the two
    machineries (("I", arg_idx) routing + OpOverload step) compose."""
    if not can_launch:
        print("\n# probe_combined_int_arg: SKIPPED (no Triton launch)")
        return

    def fn(x, y, scale):
        return add_triton(x, y) * scale

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    captured = tdcv2.capture(fn, x, y, 2)
    for scale in (1, 2, 3, 5):
        ref = fn(x, y, scale)
        got = captured(x, y, scale)
        print(f"  scale={scale}: allclose={torch.allclose(got, ref, atol=1e-5)}")


if __name__ == "__main__":
    print("=" * 70)
    print("Triton custom op + v2.capture interaction probe")
    print("=" * 70)
    print(f"# torch {torch.__version__}")
    print(f"# triton {triton.__version__}")
    print(f"# cuda available: {torch.cuda.is_available()}")

    can_launch = probe_eager_launch()
    probe_fx_graph_shape()
    probe_translator_acceptance()
    probe_end_to_end_capture(can_launch)
    probe_combined_int_arg(can_launch)
    probe_v1_with_meta_tensors()
    probe_v1_capture(can_launch)
