# v3 Fallback Backend 设计文档

日期:2026-06-03
状态:已评审,待实现

## 1. 背景与动机

现有 v3-fallback(`python/v3/`)通过 `force_all_fallback()`(monkeypatch `lowerings` dict)
+ stock 上游 `CppWrapperCpu`,在 CPU 上验证了"全量 fallback + cpp_wrapper"能去掉 per-op
Python 开销。但要把这条路用到 NPU 上时,torch_npu 现有的 `CppWrapperNpu` 是一份**基于旧
版本拷贝改写的补丁实现**,并为规避 NPU 缺陷做了大量改动(`super_write_header_rewrite`、
注释掉 `c_shim_npu`、版本漂移导致 `add_device_include` 泄漏不存在的 `cpp_wrapper/npu.h`
等)。

本设计构建一个**干净的、直接继承当前上游 `CppWrapperCpu` 的 wrapper 子类**,配一个
**独立的 compile backend**,目标:

- 全量 fallback + cpp_wrapper,生成纯"分配 + 逐 op 下发 + 释放"的 C++ host 驱动,
  实现快速算子下发、消除 host/Python 开销;
- 全程复用 inductor 的 device 适配流程,**NPU 移植只需 `register_backend_for_device`**;
- 作为一个**新的对照 arm**,与现有 v3-fallback / v2 / inductor / eager 同台对比,
  不替换现有 v3-fallback。

非目标(YAGNI):AOTI 模式(本设计只面向 JIT);NPU 的实际移植(本设计只到 CPU 原型
+ 留好挂点);补 c-shim(正交工作,见 `2026-05-29-npu-cpp-wrapper-adaptation.md`)。

## 2. 总体架构

```
torch.compile(fn, backend=make_fallback_backend(mode))
        |
        v
fallback_inductor_backend(gm, example_inputs)        # 唯一新外层
        |  with _fallback_codegen_context(device, mode):
        |    - inductor_config.cpp_wrapper = True
        |    - force_all_fallback()  (复用现有,lowerings 全 fallback + 关融合)
        |    - swap device_codegens[device].cpp_wrapper_codegen = CppWrapperFallback
        |    - mode=="boxed": patch FallbackKernel.codegen -> use_runtime_dispatch=True
        v
torch._inductor.compile_fx(gm, example_inputs)       # 完全是 stock inductor 流程
        |   GraphLowering -> init_wrapper_code
        |     -> get_wrapper_codegen_for_device(device, cpp_wrapper=True)
        |     -> 返回 CppWrapperFallback(已注册)
        v
CppWrapperFallback(CppWrapperCpu)                     # 干净子类,CPU 上近乎 identity
        v
编译好的 callable(host 驱动 = C++,无 per-op Python)
```

设备适配归属:inductor 内部已按 device 适配(分配 / stream / guard / fallback 下发)。
- CPU 原型:backend 在编译作用域内临时 swap `device_codegens['cpu'].cpp_wrapper_codegen`,
  退出恢复(不永久劫持 cpu)。
- NPU 移植:torch_npu 侧 `register_backend_for_device('npu', NPUScheduling, NPUWrapper,
  CppWrapperFallback)` 永久注册即可,backend 一行不改。

## 3. 组件

放在 `python/v3/fallback_backend.py`,三个小单元:

### 3.1 `CppWrapperFallback(CppWrapperCpu)`
- 职责:干净继承当前上游 `CppWrapperCpu`;CPU 上近乎 identity。唯一必须的覆写是
  `create` classmethod(返回自身类型,inductor 子类约定)。为 NPU 移植预留覆写挂点
  (设备 include / `kernel_driver` / stream),CPU 阶段留空不实现。
- 接口:标准 wrapper codegen 接口(全继承)。
- 依赖:`CppWrapperCpu`。

### 3.2 `fallback_inductor_backend(gm, example_inputs)` / `make_fallback_backend(mode)`
- 职责:dynamo backend。进入配置作用域 -> 调 `torch._inductor.compile_fx(gm,
  example_inputs)` -> 返回编译 callable。
- 接口:标准 backend 签名 `(GraphModule, list) -> Callable`。下发模式通过工厂
  `make_fallback_backend(mode="boxed"|"stock") -> backend` 绑定。
- 依赖:`compile_fx`、单元 3.3。

### 3.3 `_fallback_codegen_context(device, mode)`
- 职责(上下文管理器,`try/finally` 全恢复):
  - `inductor_config.patch({"cpp_wrapper": True})`;
  - 复用现有 `force_all_fallback()`(lowerings 全 fallback + 关融合 + `triton.cudagraphs=False`);
  - snapshot 并 swap `device_codegens[device].cpp_wrapper_codegen = CppWrapperFallback`;
  - mode=="boxed":作用域内 patch `FallbackKernel.codegen` 强制 `use_runtime_dispatch=True`
    (全部走 `call_dispatcher`);mode=="stock":不动(有 c-shim 走 c-shim,否则 dispatcher)。
- 接口:`with _fallback_codegen_context("cpu", "boxed"): ...`。
- 依赖:`force_all_fallback`、`CppWrapperFallback`、inductor 的 `device_codegens` /
  `FallbackKernel`。

对外用法(新对照 arm):
```python
compiled = torch.compile(fn, backend=make_fallback_backend(mode="boxed"), dynamic=True)
```

## 4. 数据流与两种下发模式

端到端(mode="boxed"):dynamo trace -> backend 进配置作用域 -> `compile_fx`(stock):
AOT -> aten 图 -> lowering 每个 op 成 FallbackKernel(无融合)-> `CppWrapperFallback`
逐节点 codegen -> 编译 .so;退出作用域全恢复;调用时跑 C++ host 驱动(分配 + 逐 op
下发 + 释放,无 per-op Python)。

同一个 `relu(mm(a,b))` 的产物差异:

| | boxed 模式 | stock 模式 |
|---|---|---|
| mm | `aoti_torch_call_dispatcher("aten::mm", ...)` | `aoti_torch_cpu_mm_out(...)`(有 shim 走 unboxed) |
| relu | `aoti_torch_call_dispatcher("aten::relu", ...)` | 有 shim 走 shim,否则 dispatcher |
| 可移植性 | 设备无关,NPU 无需 c-shim 即可用 | NPU 无 c-shim 时退化 |
| host 开销 | boxed 派发(约等于 v2 callBoxed) | 有 shim 的更快 |

两种模式都无融合核(force_all_fallback 关掉)。

## 5. 错误处理与边界

1. 作用域恢复:`try/finally` 确保 `compile_fx` 抛异常时也恢复 `device_codegens`、
   `lowerings`、config patch、`FallbackKernel.codegen` patch。
2. 进程全局 / 非线程安全:swap `device_codegens` 和 patch `FallbackKernel.codegen`
   是进程级,与现有 `force_all_fallback` 同等约束;原型/benchmark 单线程编译可接受。
3. Dynamo cache 污染:同一 fn 多 arm 同进程跑共享 per-code-object cache,复用现有
   `isolate_fresh_fn` 隔离。
4. 数据依赖 / list 返回算子:boxed 模式 `call_dispatcher` 是通用 boxed 路径,
   `nonzero`/`split`/`_foreach_*` 都能跑;stock 模式无 c-shim 的也回落 dispatcher。
   无正确性缺口。
5. cudagraph:`force_all_fallback` 已设 `triton.cudagraphs=False`,沿用。

## 6. 测试

用项目的 `TestCase` + `run_tests`,新增 `test/test_v3_fallback_backend.py`:

1. 正确性:代表性负载(pointwise、mm+pointwise、`nonzero` 数据依赖、一个无 c-shim
   算子)× `@parametrize` 两种 mode,`assertEqual` 对齐 eager。
2. 诊断断言:`run_and_get_cpp_code` 抓产物 —— boxed 模式断言含
   `aoti_torch_call_dispatcher`;两种模式都断言无 `cpp_fused_`(确认融合关闭)。
3. benchmark arm:`prototypes/v2_benchmark.py` 加 `v3-fallback-boxed` /
   `v3-fallback-stock` 两个 variant,与 eager / inductor / v2 / 现有 v3-fallback
   同台对比 host 开销。

## 7. NPU 移植改动面(验证"加个 register 就行")

- 单元 3.2 / 3.3 一行不改;
- torch_npu 侧 `register_backend_for_device('npu', ..., CppWrapperFallback)`;
- 在 `CppWrapperFallback` 补 NPU 的覆写挂点(设备 include / `kernel_driver` / stream);
- boxed 模式天然可用(无需 c-shim);stock 模式待 `c_shim_npu` 补齐后才有意义。
