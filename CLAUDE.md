# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

C++ dispatcher 级别的 PyTorch 算子捕获/重放（capture/replay）PoC。记录一个代码块中所有 aten op 的 dispatcher 调用，后续无需 Python 解释器开销即可重放。重放时从捕获的 Tensor 对象实时读取 metadata（sizes/strides/data_ptr），因此 in-place mutation、resize 等动态 shape 变化自动反映——无需像 cudagraph 那样按 shape 重新捕获。

## 构建与测试

```bash
# 完整构建（在包含 PyTorch 的 venv 中）
MAX_JOBS=4 pip install -e . --no-build-isolation -v

# 仅增量重编译 C++ 部分
MAX_JOBS=4 python setup.py build_ext --inplace

# 运行所有测试
python -m unittest discover test -v

# 运行单个测试文件
python -m unittest test.test_correctness
python -m unittest test.test_v2_capture

# 在 GPU 上运行测试
TDC_DEVICE=cuda python -m unittest discover test
TDC_DEVICE=npu python -m unittest discover test
```

`MAX_JOBS` 被 `setup.py` 自动 clamp 到 <=4。所有测试文件通过 `test/_device.py` 读取 `TDC_DEVICE` 环境变量来切换设备。

## 架构

### v1：直接 dispatcher 级捕获

入口：`tdc.capture()` context manager → 返回 `Trace` 对象，调用 `trace.replay()` 重放。

核心流程：
1. **`csrc/capture_fallback.cpp`** — boxed fallback 注册在 `TESTING_ONLY_GenericMode`（优先级 #3，高于 AutogradFunctionality 的 #19）。捕获期间每个 aten op 经过 dispatcher 时触发，分类输入（captured tensor / prev-step output / literal IValue），执行 op，记录 output TensorImpl 身份供后续步骤引用。
2. **`csrc/trace.cpp`** — `Trace::replay()` 热路径：遍历所有 Step，从 `StepInputRef` 重建 stack（实时读取 Tensor 当前 metadata），通过 `op.callBoxed()` 直调 kernel。跳过 autograd（push `AutoDispatchBelowAutograd`）。
3. **`csrc/capture_context.h`** — 核心数据结构：`StepInputRef`（五类 ref kind：kCapturedTensor / kPrevStepOutput / kLiteral / kCapturedInt / kList），`Step`（两种 step kind：kTensorOp / kPyCall），`Trace`（持有 steps + captured tensors + 各种 bookkeeping）。

关键约束：out-of-tree C++ 扩展，不修改 PyTorch 核心。使用 `torch.utils.cpp_extension.CppExtension` 构建。不对 shape 做符号化跟踪——所有 Python 层计算出的 size 参数在 dispatcher 看到时已是具体 int literal。

### v2：torch.compile + AOTAutograd 后端

入口：`tdcv2.capture(fn, *example_args)` → 内部运行一次 `torch.compile(backend=aot_autograd(fw_compiler=...))`，将 AOT FX graph 翻译为 C++ Trace，返回可直接调用的 callable。后续调用绕过 Dynamo/AOTAutograd，直接走 C++ replay。

核心文件：
- **`python/v2/compile.py`** — `capture()` 入口，处理 positional/kwargs 参数规范化，支持 `allow_grad=True`（fwd+bwd 双图捕获，封装为 `torch.autograd.Function`）。核心：`_build_recipe_specs()` 区分 runtime spec（按 id() 匹配用户输入 Tensor，按 FakeTensor shape 匹配 SymInt）和 pre-bind（module 参数、Dynamo 特化的常量）。
- **`python/v2/translator.py`** — FX graph → C++ Trace 翻译器：遍历 FX node，emit kTensorOp step（OpOverload）或 kPyCall step（operator.* / torch.sym_* builtin）。预计算 `ArgCoercion` 标签（Scalar→0-d Tensor, GenericList→IntList/TensorList），避免 replay 时 schema 自省。
- **`csrc/trace_v2.cpp`** — `Trace::replay_v2()` 统一重放引擎：根据 `placeholder_routing_` 将 args 路由到 captured_tensors_/captured_ints_，遍历 steps，支持 kTensorOp（带 coercion）和 kPyCall（builtin switch 或 py::object fallback）。

### 两套版本的关系

| | v1 | v2 |
|---|---|---|
| 捕获方式 | dispatcher fallback 直接记录 | torch.compile → AOT graph → 翻译为 Trace |
| dynamic shape | 仅 shape 变化沿 dim-index op 传播的可用 | 完整 SymInt 支持（含 shape-derived literal） |
| backward | 实验性 `allow_grad=True`，warmup 必须 | `allow_grad=True`，自动双图捕获 + autograd.Function |
| 用例 | 小 op、decoder、KV cache | 训练 loop、复杂 shape 推导 |

两者共享同一套 C++ Trace/Step/StepInputRef 数据结构，v1 只用前三种 ref kind，v2 额外使用 kCapturedInt 和 kList。

### 目录结构

```
csrc/                          C++ 源码（全部编译为一个 _C.so）
  capture_context.{h,cpp}      数据结构 + TLS 捕获上下文 + dump
  capture_fallback.cpp         v1 boxed fallback 注册
  trace.cpp                    v1 Trace::replay() 热路径
  trace_v2.cpp                 v2 replay 引擎 + builtin dispatch
  bindings.cpp                 pybind11 模块（v1+v2 所有 Python 接口）
python/
  __init__.py                  v1 capture() context manager
  v2/__init__.py               v2 入口（暴露 capture, translate_graph）
  v2/compile.py                v2.capture() + recipe building
  v2/translator.py             FX graph → C++ Trace 翻译器
test/
  _device.py                   共享设备/同步工具（读 TDC_DEVICE env var）
  test_correctness.py          v1 正确性测试
  test_dynamic_shape.py        v1 动态 shape 测试
  test_backward.py             v1 backward 测试
  test_benchmark.py            v1 性能基准
  test_v2_capture.py           v2 端到端测试
  test_v2_kwargs.py            v2 kwargs 调用测试
  test_v2_nn_module.py         v2 nn.Module 参数占位符测试
  test_v2_backward.py          v2 backward 测试
prototypes/                    探索性脚本与 trace 样本
  v2_benchmark.py              v1/v2/inductor 多模式性能对比
  v2_aot_api.py                AOTAutograd 边界实验
```

### 关键设计决策

- **`replay()` 不返回值**：trace 记录的是副作用（写到 `out=` buffer、in-place mutation），不是纯函数。返回"最后一个 step 的输出"会静默忽略其他 buffer 写入。
- **v1 `allow_grad=True` 需要 warmup**：AccumulateGrad 首次调用走非 dispatch 的 C++ 直接赋值路径，必须先用 eager backward 跑一次让 `.grad` 分配好，后续 backward 才走 dispatched `add_` accumulate 路径被记录。
- **v2 pre-bind vs runtime spec**：nn.Module 参数/Dynamo 特化常量通过 `id()` 匹配识别后 pre-bind 到 trace（一次设定，跨 replay 不变）；用户输入 Tensor 和 SymInt 维度作为 runtime spec（每次 replay 从 args 提取）。
- **full-literal list 冻结**：translator 遇到所有元素均为 Python literal 的 list arg（如 `permute([0,2,1,3])`），在翻译时直接构建 `c10::List<int64_t>` 存为 kLiteral，replay 跳过 list 重建+coercion。
- **多输出 op + getitem 折叠**：translator 对已知多输出的 OpOverload 节点记录 `multi_output_step` map，后续 `operator.getitem(node, k)` 折叠为 `PrevStepOutput(step_idx, k)` 引用，不 emit 额外 PyCall step。