"""CPU cpp_wrapper dispatch demo.

Builds one function that forces inductor's cpp_wrapper to emit all three
operator-dispatch mechanisms, then dumps the generated C++ host code:

  1. fused kernel      -> pointwise ops compiled into a `cpp_fused_*` C++
                          function, called directly.
  2. c-shim aten op    -> mm lowers to an extern kernel invoked via the
                          unboxed C ABI shim `aoti_torch_cpu_mm_out(...)`.
  3. proxy executor    -> a custom op inductor cannot lower and has no shim
                          for; dispatched via the boxed
                          `aoti_torch_proxy_executor_call_function(...)`.

Run:  python cpp_wrapper_dispatch_demo.py
Writes the full generated C++ to cpp_wrapper_dump.py next to this file.
"""
import os

import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_cpp_code


# (3) custom op with no inductor lowering and no c-shim -> proxy executor
@torch.library.custom_op("tdcdemo::scaled_add", mutates_args=())
def scaled_add(x: torch.Tensor, y: torch.Tensor, s: float) -> torch.Tensor:
    return x + y * s


@scaled_add.register_fake
def _(x, y, s):
    return torch.empty_like(x)


def fn(a, b, x, y):
    m = torch.mm(a, b)                 # (2) extern aten -> c-shim
    m = torch.permute(m, (1, 0))
    p = torch.relu(m * 2.0 + 1.0)      # (1) pointwise -> fused cpp kernel
    c = torch.ops.tdcdemo.scaled_add(p, x, 0.5)   # (3) custom -> proxy executor
    return c + y                       # (1) more pointwise


def main():
    inductor_config.cpp_wrapper = True

    a = torch.randn(64, 128)
    b = torch.randn(128, 64)
    x = torch.randn(64, 64)
    y = torch.randn(64, 64)
    ref = fn(a, b, x, y)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor", dynamic=True)
    result, code = run_and_get_cpp_code(compiled, a, b, x, y)
    print("numeric:", "MATCH" if torch.allclose(result, ref, atol=1e-4) else "MISMATCH")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp_wrapper_dump.py")
    with open(out, "w") as f:
        f.write(code)
    print(f"dumped {len(code.splitlines())} lines of generated C++ to {out}")

    # Highlight the three dispatch mechanisms in the generated code.
    def grep(label, needles):
        hits = [ln.strip() for ln in code.splitlines()
                if any(n in ln for n in needles)]
        print(f"\n=== {label} ({len(hits)} line(s)) ===")
        for h in hits[:12]:
            print("  ", h)

    grep("(1) fused C++ kernel def + call", ["cpp_fused_", "kernels.cpp_fused"])
    grep("(2) c-shim aten op (unboxed C ABI)", ["aoti_torch_cpu_", "aoti_torch_cpu_mm"])
    # JIT cpp_wrapper uses the live dispatcher for boxed fallbacks
    # (aoti_torch_call_dispatcher). The proxy executor
    # (aoti_torch_proxy_executor_call_function) is the AOTI-mode form of the
    # same boxed fallback, used when there is no Python runtime.
    grep("(3) boxed fallback (JIT=call_dispatcher / AOTI=proxy_executor)",
         ["aoti_torch_call_dispatcher", "aoti_torch_proxy_executor_call_function"])


if __name__ == "__main__":
    main()
