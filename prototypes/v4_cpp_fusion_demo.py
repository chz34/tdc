"""Demo: CPU cpp-fused kernels embedded into the fx_wrapper host graph via the
CompiledKernelWrapperMutation HOP.

Contrast with v4_capture_demo.py / enable_device_via_fallback, which force every
op to an aten extern (no fusion) so the host graph is all-extern. Here we keep
the real CppScheduling: Inductor fuses CPU pointwise ops into cpp_fused_* kernels,
and enable_device_with_fusion turns each fused kernel into a
`compiled_kernel_wrapper_mutation` HOP node in the host gm -- which then runs.

Run:  python v4_cpp_fusion_demo.py
"""
import torch
import torch._inductor.config as inductor_config

import torch_dispatch_capture.v4 as tdcv4
from torch_dispatch_capture.v4.compiled_kernel_hop import (
    compiled_kernel_side_table,
    compiled_kernel_wrapper_mutation,
)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)

    def forward(self, x):
        # fc1/fc2 -> addmm externs; the gelu + residual add fuse into cpp kernels.
        h = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(h) + x


def main():
    torch.manual_seed(0)
    model = MLP().eval()
    x = torch.randn(8, 64)
    ref = model(x)

    compiled_kernel_side_table.reset_table()
    torch._dynamo.reset()

    # The host gm is exposed by passing a gm_backend; it receives the converted
    # FX GraphModule. Returning gm.forward just runs it (pure enablement).
    captured = {}

    def gm_backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    # ---- the actual usage: one context manager + ordinary torch.compile -------
    with torch.no_grad(), inductor_config.patch(force_disable_caches=True), \
            tdcv4.enable_device_with_fusion("cpu", gm_backend):
        compiled = torch.compile(model, backend="inductor", dynamic=False)
        out = compiled(x)
    # --------------------------------------------------------------------------

    gm = captured["gm"]
    print("=== host FX graph (fx_wrapper output) ===")
    gm.print_readable()
    print("-----------------------------------------")
    print(gm.graph)
    print("=========================================")


    hop_nodes = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and n.target is compiled_kernel_wrapper_mutation
    ]
    extern_nodes = [
        n for n in gm.graph.nodes
        if n.op == "call_function" and "aten" in str(n.target)
    ]
    print(f"\ncompiled cpp-fused HOP nodes : {len(hop_nodes)}")
    for n in hop_nodes:
        mai = n.kwargs["mutated_arg_indices"]
        print(f"  kernel_idx={n.kwargs['kernel_idx']} "
              f"n_args={len(n.kwargs['args'])} mutated={mai}")
    print(f"aten extern nodes            : {len(extern_nodes)}  "
          f"{[str(n.target).split('.')[-2] for n in extern_nodes]}")
    print(f"side table size              : {len(compiled_kernel_side_table.id_to_kernel)}")
    print(f"\nnumerics match eager         : {torch.allclose(out, ref, atol=1e-4)}")


if __name__ == "__main__":
    main()
