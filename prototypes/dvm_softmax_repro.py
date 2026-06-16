"""Minimal reproducer for the dvm softmax compile-time coredump.

FINDING (2026-06-16): this is a torch_npu DVM build bug, NOT a tdc/fx_wrapper
issue. MODE=native (plain dvm torch.compile, no tdc) segfaults on the same kernel
with an identical fused subgraph, so the crash is in the dvm C build (kobj.setup)
of the bool/select/-inf 'vector' kernel. Use MODE=native as a pure upstream repro.

Distilled from t5_cpu.py's T5Attention at the decode step (seq_len=1): the fused
kernel that segfaults is
  dvm_fused__softmax_div_eq_masked_fill_matmul_ones_tril_*
i.e. matmul/scale -> tril(ones)==0 -> masked_fill(-inf) -> softmax -> matmul.

The causal mask is built INSIDE forward (torch.tril(torch.ones(...))) so tril +
ones fuse into the kernel (hence the `_ones_tril` suffix), and the mask compare
+ masked_fill(-inf) introduce the bool / select / -inf ops that distinguish this
kernel (ktype='vector') from the layernorm kernels (ktype='spec') that compile
fine. Shapes default to the exact crashing decode shapes; override via env.

Pure dvm repro on NPU (no tdc; segfaults in the dvm build):
  TORCHINDUCTOR_NPU_BACKEND=dvm TDC_DEVICE=npu MODE=native python dvm_softmax_repro.py

Via the tdc fx_wrapper path on NPU (same crash):
  TORCHINDUCTOR_NPU_BACKEND=dvm TDC_DEVICE=npu python dvm_softmax_repro.py

Control on CPU (cpp backend; compiles + runs, NO crash):
  TDC_DEVICE=cpu python dvm_softmax_repro.py

Env knobs: HEADS (8), HEAD_DIM (128), SEQ (1), KV (1), DYNAMIC (0).
"""
import math
import os
import sys

import torch
from torch import nn


class AttnSoftmaxRepro(nn.Module):
    """The score -> causal-mask -> softmax -> value core of T5Attention. q/k/v are
    already projected + head-split, matching T5Attention after q_proj/k_proj/v_proj
    and the view/transpose to [batch, heads, seq, head_dim]."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.scale = math.sqrt(head_dim)

    def forward(self, query, key, value):
        seq = query.shape[2]
        kv = key.shape[2]
        # built in-graph -> tril + ones fuse into the kernel (the `_ones_tril`)
        mask = torch.tril(torch.ones(seq, kv, device=query.device))
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)


def main():
    device = os.environ.get("TDC_DEVICE", "npu")
    os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "dvm")
    heads = int(os.environ.get("HEADS", "8"))
    head_dim = int(os.environ.get("HEAD_DIM", "128"))
    seq = int(os.environ.get("SEQ", "1"))
    kv = int(os.environ.get("KV", "1"))
    dynamic = os.environ.get("DYNAMIC", "0") == "1"

    if device == "npu":
        try:
            import torch_npu  # noqa: F401
            import torch_npu._inductor  # noqa: F401
        except Exception as e:
            print(f"[skip] torch_npu unavailable: {e}")
            return
        if not torch.npu.is_available():
            print("[skip] no NPU device")
            return

    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, head_dim, device=device)
    k = torch.randn(1, heads, kv, head_dim, device=device)
    v = torch.randn(1, heads, kv, head_dim, device=device)
    model = AttnSoftmaxRepro(head_dim).to(device)

    ref = model(q, k, v)

    captured = {}

    def capture_backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    # MODE=native  -> plain dvm compile (no fx_wrapper, no DvmBackend): the
    #                 baseline that "works". If THIS segfaults too, the dvm build
    #                 of the kernel is the bug, independent of our integration.
    # FINDING (2026-06-16): MODE=native ALSO segfaults on this kernel, with an
    # identical fused subgraph -- so this is a torch_npu DVM build bug (kobj.setup
    # for the bool/select/-inf 'vector' kernel), NOT a tdc/fx_wrapper issue. Use
    # this script (MODE=native, no tdc needed) as the upstream repro. The full T5
    # native run avoids the crash only because its richer fused subgraph trips the
    # is_node_dvm_supported fallback to import_fx; this minimal subgraph is all-
    # supported, so dvm actually builds it and crashes.
    # MODE=fusion  -> our enable_device_with_fusion path (default; same crash).
    mode = os.environ.get("MODE", "fusion")

    print(
        f"[info] device={device} mode={mode} heads={heads} head_dim={head_dim} "
        f"seq={seq} kv={kv} dynamic={dynamic}"
    )
    print("[info] compiling -- if it segfaults here, it died in a dvm kernel build")
    sys.stdout.flush()

    import torch._inductor.config as inductor_config

    if mode == "native":
        with torch.no_grad(), inductor_config.patch({"force_disable_caches": True}):
            out = torch.compile(model, backend="inductor", dynamic=dynamic)(q, k, v)
            if device == "npu":
                torch.npu.synchronize()
        # Note: for the default softmax kernel this line is NOT reached -- the dvm
        # build segfaults above (a pure dvm bug; no tdc / fx_wrapper involved). It
        # prints only for kernels dvm can build (or with shapes that fall back).
        print(
            f"[ok] native dvm compiled + ran, no crash. "
            f"numerics match eager: {torch.allclose(out, ref, atol=1e-3)}"
        )
        return

    # fusion mode only: import tdc lazily so MODE=native is a pure dvm repro.
    import torch_dispatch_capture.v4 as tdcv4

    with torch.no_grad(), inductor_config.patch(
        {"force_disable_caches": True, "generate_intermediate_hooks": False}
    ), tdcv4.enable_device_with_fusion(device, capture_backend):
        out = torch.compile(model, backend="inductor", dynamic=dynamic)(q, k, v)
        if device == "npu":
            torch.npu.synchronize()

    gm = captured.get("gm")
    from torch_dispatch_capture.v4.compiled_kernel_hop import (
        compiled_kernel_side_table,
        compiled_kernel_wrapper_mutation,
    )

    hop = (
        [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is compiled_kernel_wrapper_mutation
        ]
        if gm is not None
        else []
    )
    print(f"[ok] compiled. CompiledKernelWrapperMutation HOP nodes: {len(hop)}")
    print(f"     side table size: {len(compiled_kernel_side_table.id_to_kernel)}")
    print(f"     numerics match eager: {torch.allclose(out, ref, atol=1e-3)}")


if __name__ == "__main__":
    main()
