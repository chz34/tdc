"""Reproduce the native-vs-fx_wrapper divergence on the T5 decode softmax kernel.

The earlier dvm_softmax_repro.py fed the causal mask in as a bool tensor, so its
fused kernel had only bool/float inputs -- dvm built it in BOTH native and
fx_wrapper, and both crashed. That hid the real divergence.

The actual T5 kernel computes the causal mask IN-GRAPH from int64 position
indices: pos_diff = q_pos - k_pos; keep = pos_diff <= 0 (the `sub` + `le.Scalar`
seen in the native import_fx graph). Because int64 is NOT in DVM_SUPPORT_TYPE
([bf16, f16, f32, int32, bool]), is_node_dvm_supported returns False for the
int64 placeholders, so native sets dvm_codegen=None and falls back to
async_compile.import_fx -- it never builds the dvm kernel, so it does not crash.

Under fx_wrapper the scheduler splits the int64 mask math out of the softmax
kernel; the remaining kernel has only bool/float inputs (all DVM-supported), so
dvm actually builds it and hits the DVM build/exec bug -> coredump.

So this repro keeps the int64 position arithmetic in the model to reproduce the
divergence:
  MODE=native  -> kernel keeps int64 -> import_fx fallback -> runs, no crash
  MODE=fusion  -> int64 split out -> bool/float kernel built -> coredump

Run on NPU:
  TORCHINDUCTOR_NPU_BACKEND=dvm TDC_DEVICE=npu MODE=native python dvm_softmax_causal_repro.py
  TORCHINDUCTOR_NPU_BACKEND=dvm TDC_DEVICE=npu              python dvm_softmax_causal_repro.py

CPU control (cpp backend; compiles + runs, no crash):
  TDC_DEVICE=cpu python dvm_softmax_causal_repro.py

Env knobs: HEADS (8), HEAD_DIM (128), SEQ (1), KV (1), DYNAMIC (0).
"""
import math
import os
import sys

import torch
from torch import nn


class CausalAttnRepro(nn.Module):
    """Attention with a causal mask derived from int64 position indices, mirroring
    the native T5 decode kernel (sub -> le.Scalar -> where -> eq -> masked_fill)."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.scale = math.sqrt(head_dim)

    def forward(self, query, key, value, q_pos, k_pos):
        # int64 position arithmetic -> bool causal mask (the int64 "poison" that
        # makes native fall back to import_fx; keep it in-graph).
        pos_diff = q_pos.reshape(-1, 1) - k_pos.reshape(1, -1)  # i64[seq, kv]
        keep = pos_diff <= 0  # bool[seq, kv]  (le.Scalar)
        mask = torch.where(keep, 1.0, 0.0)  # f32  (where)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(mask == 0, float("-inf"))  # eq + masked_fill
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
    mode = os.environ.get("MODE", "fusion")

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
    q_pos = torch.arange(seq, dtype=torch.int64, device=device)
    k_pos = torch.arange(kv, dtype=torch.int64, device=device)
    model = CausalAttnRepro(head_dim).to(device)

    ref = model(q, k, v, q_pos, k_pos)

    captured = {}

    def capture_backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    print(
        f"[info] device={device} mode={mode} heads={heads} head_dim={head_dim} "
        f"seq={seq} kv={kv} dynamic={dynamic}"
    )
    print("[info] compiling/running -- a segfault means the dvm kernel was built")
    sys.stdout.flush()

    import torch._inductor.config as inductor_config

    if mode == "native":
        with torch.no_grad(), inductor_config.patch({"force_disable_caches": True}):
            out = torch.compile(model, backend="inductor", dynamic=dynamic)(
                q, k, v, q_pos, k_pos
            )
            if device == "npu":
                torch.npu.synchronize()
        print(
            f"[ok] native dvm ran, no crash (int64 kept -> import_fx fallback). "
            f"numerics match eager: {torch.allclose(out, ref, atol=1e-3)}"
        )
        return

    import torch_dispatch_capture.v4 as tdcv4

    with torch.no_grad(), inductor_config.patch(
        {"force_disable_caches": True, "generate_intermediate_hooks": False}
    ), tdcv4.enable_device_with_fusion(device, capture_backend):
        out = torch.compile(model, backend="inductor", dynamic=dynamic)(
            q, k, v, q_pos, k_pos
        )
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
