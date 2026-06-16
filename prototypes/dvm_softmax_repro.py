"""Minimal reproducer for the dvm-fxwrapper softmax compile-time coredump.

Distilled from t5_cpu.py's T5Attention at the decode step (seq_len=1): the fused
kernel that segfaults during DvmBackend.compile_kernel is
  dvm_fused__softmax_div_eq_masked_fill_matmul_ones_tril_*
i.e. matmul/scale -> tril(ones)==0 -> masked_fill(-inf) -> softmax -> matmul.

The causal mask is built INSIDE forward (torch.tril(torch.ones(...))) so tril +
ones fuse into the kernel (hence the `_ones_tril` suffix), and the mask compare
+ masked_fill(-inf) introduce the bool / select / -inf ops that distinguish this
kernel (ktype='vector') from the layernorm kernels (ktype='spec') that compile
fine. Shapes default to the exact crashing decode shapes; override via env.

Run on NPU (reproduces the coredump in the dvm build of the softmax kernel):
  TORCHINDUCTOR_NPU_BACKEND=dvm TDC_DEVICE=npu python dvm_softmax_repro.py

Control on CPU (cpp backend; should compile + run, NO crash):
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

    import torch_dispatch_capture.v4 as tdcv4

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

    print(
        f"[info] device={device} heads={heads} head_dim={head_dim} "
        f"seq={seq} kv={kv} dynamic={dynamic}"
    )
    print("[info] compiling -- if it segfaults here, it died in a dvm kernel build")
    sys.stdout.flush()

    import torch._inductor.config as inductor_config

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
