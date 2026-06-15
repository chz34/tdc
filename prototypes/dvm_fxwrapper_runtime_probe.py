"""Runtime go/no-go probe for the dvm backend + fx_wrapper path. RUN ON AN NPU BOX.

Companion to dvm_fxwrapper_static_probe.py. The static probe predicts that the
dvm scheduling emits bare-string lines into the host wrapper (the '_uwu_'
symbolic-arg lines), which FxConverter cannot convert. This script proves it
empirically, WITHOUT writing any DvmBackend yet:

  Phase 1 -- swap only fx_wrapper_codegen for the npu device to the STOCK
  WrapperFxCodegen, turn config.fx_wrapper on, compile a tiny mlir-fused
  subgraph, and capture the exact exception FxConverter raises (and on which
  line/kernel). That is the blocker we must fix on the dvm side.

It imports nothing torch_npu-specific at module top level; if torch_npu / dvm
are unavailable it prints SKIP and exits 0, so it is safe to drop into CI.

Usage (on NPU):
  TORCHINDUCTOR_NPU_BACKEND=dvm python agent_space/dvm_fxwrapper_runtime_probe.py
"""
import dataclasses
import os
import sys
import traceback


def _skip(reason: str):
    print(f"[SKIP] {reason}")
    sys.exit(0)


def main():
    os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "dvm")
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    requested_backend = os.environ["TORCHINDUCTOR_NPU_BACKEND"]

    try:
        import torch
    except Exception as e:  # pragma: no cover
        _skip(f"torch import failed: {e}")

    try:
        import torch_npu  # noqa: F401
        # NOTE: this import overwrites os.environ["TORCHINDUCTOR_NPU_BACKEND"] to
        # 'mlir' as a side effect (mlir_compiler.py top-level), because dvm reuses
        # the mlir compile/codegen path. The user's real choice survives only on
        # _InductorNpuRegistry._loaded_backend, so check that, not the env var.
        import torch_npu._inductor  # registers the npu backend
    except Exception as e:
        _skip(f"torch_npu/_inductor unavailable: {e}")

    if not torch.npu.is_available():
        _skip("no NPU device available")

    try:
        from torch_npu.utils._dynamo import _InductorNpuRegistry

        loaded_backend = _InductorNpuRegistry._loaded_backend
    except Exception:
        loaded_backend = requested_backend
    if loaded_backend != "dvm":
        _skip(f"active npu backend is {loaded_backend!r}, not dvm")

    from torch._inductor import config as inductor_config
    from torch._inductor.codegen.common import (
        device_codegens,
        init_backend_registration,
    )
    from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen

    init_backend_registration()
    if "npu" not in device_codegens:
        _skip("no inductor backend registered for 'npu' (dvm not active?)")

    dc = device_codegens["npu"]
    print(f"[info] npu scheduling = {getattr(dc, 'scheduling', None)}")
    print(f"[info] npu wrapper    = {getattr(dc, 'wrapper_codegen', None)}")
    print(f"[info] npu fx_wrapper = {getattr(dc, 'fx_wrapper_codegen', None)}")

    # A tiny pointwise chain that dvm fuses into a single mlir kernel; a@b stays
    # an extern so we also exercise the extern path under the fx wrapper.
    def fn(a, b):
        return torch.relu(a @ b + a) * 2.0

    a = torch.randn(64, 64, device="npu")
    b = torch.randn(64, 64, device="npu")
    ref = fn(a, b)

    captured = {}

    def capture_backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    saved = device_codegens["npu"]
    device_codegens["npu"] = dataclasses.replace(
        saved, fx_wrapper_codegen=WrapperFxCodegen
    )
    err = None
    try:
        torch._dynamo.reset()
        with torch.no_grad(), inductor_config.patch(
            {
                "fx_wrapper": True,
                "size_asserts": False,
                "alignment_asserts": False,
                "force_disable_caches": True,
                # avoid the run_intermediate_hooks host-write path (R1 #3)
                "generate_intermediate_hooks": False,
            }
        ):
            out = torch.compile(fn, backend="inductor", dynamic=False)(a, b)
            torch.npu.synchronize()
    except Exception as e:  # the predicted FxConverter failure
        err = e
    finally:
        device_codegens["npu"] = saved

    print("\n" + "=" * 70)
    print("PHASE 1 RESULT (stock WrapperFxCodegen on dvm, no DvmBackend yet)")
    print("=" * 70)
    if err is not None:
        msg = str(err)
        print(f"[blocked] {type(err).__name__}: {msg.splitlines()[0] if msg else ''}")
        lowered = msg.lower()
        if "unsupported type for kernel" in lowered:
            print(
                "  => DEFINITION GATE (_import_kernel): the mlir kernel def loaded"
                "\n     fine but is not a CachingAutotuner, so stock FxConverter"
                "\n     rejects it. This is the FIRST blocker and is exactly what"
                "\n     DvmBackend solves: CompiledKernelFxConverter checks"
                "\n     _select_backend BEFORE super()._generate_kernel_definition,"
                "\n     so _import_kernel is never reached. NOTE: R1 (bare-string"
                "\n     host lines) is downstream and was NOT reached here."
            )
        elif "wrapper ir lines" in lowered:
            print(
                "  => CONFIRMS R1: a bare-string host line reached FxConverter."
                "\n     Most likely the '_uwu_' symbolic-arg writeline in"
                " call_kernel."
            )
        elif "only supports triton kernels" in lowered:
            print(
                "  => CALL GATE: the mlir KernelCallLine hit the triton-only gate;"
                "\n     a DvmBackend (handles_definition on async_compile.mlir)"
                " would intercept it."
            )
        else:
            print("  => unexpected failure; full traceback below.")
        print("-" * 70)
        traceback.print_exception(type(err), err, err.__traceback__)
    else:
        gm = captured.get("gm")
        print("[ok] fx_wrapper host gm produced without error.")
        if gm is not None:
            targets = [
                n.target for n in gm.graph.nodes if n.op == "call_function"
            ]
            print(f"  host gm call_function targets: {len(targets)}")
            print(f"  numerics match eager: {torch.allclose(out, ref, atol=1e-3)}")
        print(
            "  => R1 not triggered for THIS subgraph (no '_uwu_' args emitted)."
            "\n     Try a dynamic-shape / indexed subgraph to force symbolic args."
        )

    # ---- Phase 2: DvmBackend via enable_device_with_fusion -----------------
    print("\n" + "=" * 70)
    print("PHASE 2 RESULT (DvmBackend + CompiledKernelFxWrapper)")
    print("=" * 70)
    try:
        import torch_dispatch_capture.v4 as tdcv4  # auto-registers DvmBackend
        import torch_dispatch_capture.v4.cpp_fusion as cf
        from torch_dispatch_capture.v4.compiled_kernel_hop import (
            compiled_kernel_side_table,
            compiled_kernel_wrapper_mutation,
        )
    except Exception as e:
        print(f"[skip] torch_dispatch_capture not importable: {e}")
        return

    # ---- diagnostics: which backends are registered, and how each kernel
    # definition line is classified (so a "None" selection is explainable). ----
    registered = [type(b).__name__ for b in cf._COMPILED_BACKENDS]
    print(f"[diag] registered backends: {registered}")
    has_dvm = any(type(b).__name__ == "DvmBackend" for b in cf._COMPILED_BACKENDS)
    print(f"[diag] DvmBackend present : {has_dvm}")
    if not has_dvm:
        print(
            "  => DvmBackend is NOT registered in this tdc checkout. Pull the"
            "\n     commit that adds python/v4/dvm_fusion.py (and its import in"
            "\n     __init__.py), then re-run."
        )

    sel_log = []
    _orig_select = cf._select_backend

    def _logging_select(line):
        result = _orig_select(line)
        body = (getattr(line, "kernel_body", "") or "").replace("\n", " ")
        sel_log.append(
            (
                getattr(line, "kernel_name", None),
                getattr(line, "gpu", None),
                body[:140],
                type(result).__name__ if result is not None else None,
            )
        )
        return result

    captured2 = {}

    def capture_backend2(gm, example_inputs):
        captured2["gm"] = gm
        return gm.forward

    compiled_kernel_side_table.reset_table()
    cf._select_backend = _logging_select
    err2 = None
    try:
        torch._dynamo.reset()
        with torch.no_grad(), inductor_config.patch(
            {"generate_intermediate_hooks": False}
        ), tdcv4.enable_device_with_fusion("npu", capture_backend2):
            out2 = torch.compile(fn, backend="inductor", dynamic=False)(a, b)
            torch.npu.synchronize()
    except Exception as e:
        err2 = e
    finally:
        cf._select_backend = _orig_select

    if sel_log:
        print("[diag] kernel-definition lines seen by _select_backend:")
        for name, gpu, body, sel in sel_log:
            print(f"    - {name} (gpu={gpu}) -> {sel}")
            print(f"        body[:140]: {body}")

    if err2 is not None:
        msg = str(err2)
        lowered = msg.lower()
        print(f"[blocked] {type(err2).__name__}: {msg.splitlines()[0] if msg else ''}")
        if "wrapper ir lines" in lowered:
            print(
                "  => DEFINITION GATE BYPASSED; now hitting R1 (a bare-string host"
                "\n     line, the '_uwu_' symbolic-arg writeline). Route those lines"
                "\n     through WrapperLine IR on the dvm side."
            )
        elif "arg_types" in lowered:
            print(
                "  => DvmBackend reached mutated_arg_indices but the call line has no"
                "\n     arg_types. Apply the torch_npu call_kernel change (pass"
                "\n     arg_types marking output buffers as non-const pointers)."
            )
        else:
            print("  => next blocker; traceback below.")
        print("-" * 70)
        traceback.print_exception(type(err2), err2, err2.__traceback__)
    else:
        gm = captured2.get("gm")
        print("[ok] DvmBackend produced an fx_wrapper host gm (definition gate bypassed).")
        if gm is not None:
            hop = [
                n
                for n in gm.graph.nodes
                if n.op == "call_function"
                and n.target is compiled_kernel_wrapper_mutation
            ]
            print(f"  CompiledKernelWrapperMutation HOP nodes: {len(hop)}")
            print(f"  side table size: {len(compiled_kernel_side_table.id_to_kernel)}")
            print(f"  numerics match eager: {torch.allclose(out2, ref, atol=1e-3)}")
            if not hop:
                print(
                    "  => no HOP nodes: dvm produced no fused mlir kernel for this"
                    " subgraph (all extern?)."
                )


if __name__ == "__main__":
    main()
