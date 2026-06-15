"""Static go/no-go probe for routing the dvm (mlir/akg) inductor backend through
the CompiledKernelWrapperMutation HOP / fx_wrapper path.

Runs anywhere (no NPU, no torch_npu import needed) -- it only parses the
ascend_npu_ir codegen source with `ast`. It answers the three questions that
decide whether the dvm backend can join the HOP path, and where it would break:

  R1. Does the dvm scheduling write *bare strings* into the HOST wrapper?
      Under config.fx_wrapper=True the host wrapper is replaced by a
      WrapperFxCodegen; FxConverter only understands WrapperLine IR objects, so
      any `V.graph.wrapper_code.writeline("...")` (or `wrapper.writeline(...)`
      where `wrapper = V.graph.wrapper_code`) becomes an un-convertible line and
      raises "FX conversion only supports Wrapper IR lines". Writes into a LOCAL
      IndentedBuffer (e.g. `compile_wrapper`) are fine -- they become kernel_body.

  R2. What does NpuMlirWrapperCodeGen specialize that gets lost when the wrapper
      is swapped for the fx wrapper?

  R3. Is a mutated-arg source available on the meta kernel (needed for the HOP's
      mutated_arg_indices, since dvm call lines carry no arg_types)?

Usage:
  python agent_space/dvm_fxwrapper_static_probe.py [DVM_CODEGEN_DIR]
"""
import ast
import os
import sys

DEFAULT_DIR = (
    "/home/chz34/src/ai-framework/pytorch_npu/torch_npu/_inductor/"
    "ascend_npu_ir/ascend_npu_ir/npu/codegen"
)

HOST_WRAPPER_EXPR = "V.graph.wrapper_code"
# Methods whose body, under fx_wrapper, runs against the host wrapper instance.
HOST_WRITE_METHODS = {"call_kernel"}
SAFE_LOCAL_RECEIVERS = {"compile_wrapper", "code", "self.body", "buf", "wrapper_body"}


def _expr_src(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<?>"


class HostWriteVisitor(ast.NodeVisitor):
    """Find writeline-style calls whose receiver resolves to the host wrapper."""

    def __init__(self, filename: str):
        self.filename = filename
        self.findings: list[tuple[int, str, str, str]] = []  # line, method, kind, src
        self._method_stack: list[str] = []
        # local var name -> True if bound to V.graph.wrapper_code in this function
        self._host_aliases: list[set[str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._method_stack.append(node.name)
        self._host_aliases.append(set())
        # pre-scan assignments: `wrapper = V.graph.wrapper_code`
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign) and _expr_src(stmt.value) == HOST_WRAPPER_EXPR:
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        self._host_aliases[-1].add(tgt.id)
        self.generic_visit(node)
        self._host_aliases.pop()
        self._method_stack.pop()

    def _receiver_is_host(self, recv: ast.AST) -> bool:
        src = _expr_src(recv)
        if src == HOST_WRAPPER_EXPR:
            return True
        if isinstance(recv, ast.Name) and self._host_aliases:
            return recv.id in self._host_aliases[-1]
        return False

    def _receiver_is_safe_local(self, recv: ast.AST) -> bool:
        return _expr_src(recv) in SAFE_LOCAL_RECEIVERS

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("writeline", "writelines"):
            recv = func.value
            method = self._method_stack[-1] if self._method_stack else "<module>"
            src = _expr_src(node)
            if self._receiver_is_host(recv):
                # a raw-string arg (not a WrapperLine object) is the blocker
                is_str = node.args and isinstance(
                    node.args[0], (ast.Constant, ast.JoinedStr, ast.BinOp)
                )
                kind = "HOST-RAW-STRING" if is_str else "HOST-OBJECT"
                self.findings.append((node.lineno, method, kind, src))
            elif not self._receiver_is_safe_local(recv) and method in HOST_WRITE_METHODS:
                self.findings.append(
                    (node.lineno, method, "HOST?(unresolved receiver)", src)
                )
        self.generic_visit(node)


def scan_file(path: str):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    v = HostWriteVisitor(os.path.basename(path))
    v.visit(tree)
    return v.findings


def list_overrides(path: str, class_name: str):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            bases = [_expr_src(b) for b in node.bases]
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            return bases, methods
    return None, None


def class_has_attrs(path: str, class_name: str, attrs):
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in attrs:
                    if _expr_src(sub.value) == "self":
                        found[sub.attr] = True
    return {a: found.get(a, False) for a in attrs}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    if not os.path.isdir(root):
        print(f"[ERROR] dvm codegen dir not found: {root}")
        print("pass the path as argv[1]")
        sys.exit(2)

    scan_files = ["meta_kernel.py", "mlir.py", "akg.py", "wrapper.py"]

    print("=" * 78)
    print("R1. RAW WRITES INTO THE HOST WRAPPER (fx_wrapper blockers)")
    print("=" * 78)
    total_blockers = 0
    for fn in scan_files:
        path = os.path.join(root, fn)
        if not os.path.exists(path):
            continue
        findings = scan_file(path)
        host = [f for f in findings if f[2].startswith("HOST")]
        if not host:
            continue
        for lineno, method, kind, src in host:
            blocker = kind == "HOST-RAW-STRING"
            total_blockers += blocker
            flag = "  <== BLOCKER" if blocker else ""
            print(f"  {fn}:{lineno}  [{method}]  {kind}{flag}")
            print(f"      {src}")
    print(f"\n  raw-string host writes (hard blockers): {total_blockers}")
    print("  NOTE: writes into local 'compile_wrapper' buffers are NOT listed")
    print("        (they become kernel_body and are safe).")

    print("\n" + "=" * 78)
    print("R2. NpuMlirWrapperCodeGen SPECIALIZATION LOST UNDER fx_wrapper")
    print("=" * 78)
    wpath = os.path.join(root, "wrapper.py")
    if os.path.exists(wpath):
        bases, methods = list_overrides(wpath, "NpuMlirWrapperCodeGen")
        print(f"  class NpuMlirWrapperCodeGen({', '.join(bases or [])})")
        for m in methods or []:
            note = ""
            if m in ("write_get_raw_stream", "write_header", "write_triton_header_once"):
                note = "   (npu stream/device setup -- verify fx wrapper provides it)"
            elif m == "generate_kernel_call":
                note = "   (moot for fused kernels: they become HOP nodes)"
            elif m == "_generate_extern_kernel_alloc_helper":
                note = "   (externs -- must still work under FxConverter)"
            print(f"    - {m}{note}")

    print("\n" + "=" * 78)
    print("R3. MUTATED-ARG SOURCE ON THE META KERNEL (for HOP mutated_arg_indices)")
    print("=" * 78)
    mpath = os.path.join(root, "meta_kernel.py")
    if os.path.exists(mpath):
        attrs = class_has_attrs(
            mpath, "NpuMetaKernel", {"mutated_indices", "num_outputs", "_call_args"}
        )
        for a, present in attrs.items():
            print(f"    NpuMetaKernel.{a:<16} : {'PRESENT' if present else 'MISSING'}")
        if attrs.get("mutated_indices"):
            print(
                "\n  => mutated_arg_indices can be sourced from NpuMetaKernel.mutated_indices;"
                "\n     plumb it into the side table at define time (no arg_types needed)."
            )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if total_blockers:
        print(
            f"  NO-GO as-is: {total_blockers} raw-string host write(s) will make"
            "\n  FxConverter raise 'FX conversion only supports Wrapper IR lines'."
            "\n  Required dvm-side change: route the '_uwu_' symbolic-arg lines"
            "\n  through WrapperLine IR (e.g. SymbolicCallArg) instead of"
            "\n  wrapper.writeline(f'{arg} = {expr}'), OR ensure no '_uwu_' args are"
            "\n  emitted for the subgraphs you intend to capture."
        )
    else:
        print("  No raw-string host writes found -- fx_wrapper path is clear on R1.")
    print(
        "  Confirm R1 empirically on an NPU box with"
        " dvm_fxwrapper_runtime_probe.py."
    )


if __name__ == "__main__":
    main()
