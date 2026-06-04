# v4 FX Capture 设计文档

日期:2026-06-04
状态:已评审,直接实现

## 1. 背景与动机

inductor 的 fx_wrapper(`config.fx_wrapper=True`)把宿主 wrapper 编译成一张
`torch.fx.GraphModule`(分配 + `triton_kernel_wrapper_mutation` HOP 启动 + aten
fallback)。这张 gm **同时包含了 inductor 的融合产物**(融合后的 Triton 核以 HOP
节点出现)和宿主编排,是把"inductor 编译结果"作为可程序化处理的图二次消费的入口。

但从 `torch.compile` 返回的上层对象**拿不到这个 gm 对象**——它埋在
`FileBackedGraphModule` 里、被 Dynamo/AOT 缓存包住。gm 对象唯一的进出口是
`WrapperFxCodegen.compile_graph(gm)`(详见 DESIGN.md 的 fx_wrapper 交互点一节)。

v4 通过 hook `compile_graph`,在编译时抓住 gm,让用户**同时拿到**:
- 正常的 `torch.compile` compiled callable(可直接运行融合产物);
- 抓到的 gm 列表(可自行处理含融合算子的宿主图,例如送到 v2 做 C++ Trace replay)。

用户据此自行判断:直接跑 compiled,还是用其它方式处理 gm。

设备:不预先按设备拦截。GPU/Triton 融合核可直接转 FX;CPU 上当宿主图无 cpp 融合核
(全 fallback / 纯 extern)时也能转,有 cpp 融合核时由 inductor 在 prime 时抛
"FX conversion only supports Triton kernels",我们让它自然冒出而不提前 guard。

非目标(YAGNI):gm 的具体下游
处理(v2 集成是后续工作,v4 只负责"拿到 gm");gm 与 compiled_fn 的精确一一映射
元数据(只按顺序平铺)。

## 2. 架构(方案 A:子类 + 作用域注册 swap + 共享 sink)

复用 v3-fb 的注册-swap 套路。每次 inductor 编译会 `.create()` 出新的 wrapper
实例,所以 gm 不能存实例上,必须存到 context 管理的**共享 sink**。

```
capture_fx(fn, *example_args)
  └ with _capture_context(device):
        inductor_config.fx_wrapper = True
        swap device_codegens[device].fx_wrapper_codegen = CaptureFxWrapper
        装好 context-local 共享 sink (_active_sink)
        compiled = torch.compile(fn, backend="inductor", dynamic=True)
        compiled(*example_args)            # prime -> 触发编译
          └ inductor 每个图: CaptureFxWrapper.compile_graph(gm)
                _active_sink.append(gm)            # 抓 gm
                return super().compile_graph(gm)   # = gm.forward,正常上交
    退出 context: 恢复 fx_wrapper / device_codegens / sink
  return FxCaptureResult(compiled=compiled, gms=list(sink))
```

## 3. 组件

放在 `python/v4/capture_fx.py`。

### 3.1 `CaptureFxWrapper(WrapperFxCodegen)`
- 职责:干净子类,只覆写两个方法:
  - `compile_graph(gm)`:把 gm 追加到当前活跃 sink,然后 `return
    super().compile_graph(gm)`(= `gm.forward`,正常 compiled fn 不变)。
  - `create(...)`:返回 `CaptureFxWrapper()`(inductor 子类约定)。
- 依赖:`WrapperFxCodegen`;模块级的 `_active_sink`(context-local 列表)。

### 3.2 `_capture_context(device)`(上下文管理器)
- 职责(`try/finally` 全恢复):patch `inductor_config.fx_wrapper=True`;snapshot
  并 swap `device_codegens[device].fx_wrapper_codegen = CaptureFxWrapper`;把
  `_active_sink` 指向一个新列表,退出时还原为先前值(支持嵌套/复用安全)。
- 接口:`with _capture_context("cuda") as sink: ...`,sink 是收集 gm 的列表。
- 依赖:inductor 的 `device_codegens` / `init_backend_registration`、`CaptureFxWrapper`。

### 3.3 `capture_fx(fn, *example_args, dynamic=True) -> FxCaptureResult`
- 职责:推断 device(不预先 guard,见 §1);`with _capture_context(device)`
  内 `torch.compile(fn, backend="inductor", dynamic=dynamic)` 并 prime 一次;返回
  `FxCaptureResult(compiled, gms=list(sink))`。
- 接口:`capture_fx(fn, *example_args) -> FxCaptureResult`。

### 3.4 `FxCaptureResult`(dataclass)
```python
@dataclass
class FxCaptureResult:
    compiled: Callable             # 正常 torch.compile callable
    gms: list[torch.fx.GraphModule]  # 按 compile_graph 顺序平铺的宿主 FX 图
```

对外用法:
```python
r = v4.capture_fx(fn, *args)
out = r.compiled(*args)    # 选择1:直接跑正常融合产物
host_gm = r.gms[0]         # 选择2:拿融合产物宿主图自行处理(如喂 v2)
```

## 4. 约束与错误处理

1. **不预先按设备 guard**:GPU/Triton 融合核可直接转;CPU 上宿主图无 cpp 融合核时
   也能转(全 fallback / 纯 extern),有 cpp 融合核时 inductor 在 prime 时抛
   "FX conversion only supports Triton kernels" —— 让它自然冒出,不提前拦截。
   (早期版本硬卡 cuda/xpu,实验证明 CPU 在 size_asserts/alignment_asserts 关闭后
   也能转换无融合核的图,故移除该 guard。)
2. **作用域恢复**:`try/finally` 确保 prime 抛异常时也恢复 fx_wrapper 配置、
   device_codegens、sink。进程全局 swap,非线程安全(同 v3-fb 约束)。
2b. **关闭 size_asserts / alignment_asserts**:这两类断言由 inductor 以**裸字符串行**
   发出(`assert_size_stride(...)` / `# ... not aligned`),而 FxConverter 只接受结构化
   WrapperLine,裸行会中止 FX 转换(extern 算子如 sdpa/conv 会踩到)。capture_fx 在
   作用域内关掉它们,使宿主图可转 FX。代价:抓到的 gm 和返回的 compiled fn 运行时**不带**
   这些 size/alignment 检查。capture_fx 会打印一行提示说明此事。
3. **prime 失败如实抛出**:某些 fallback 触发 fx_wrapper 覆盖盲区(如 sdpa 的
   assert_size_stride 裸字符串)会编译失败,不静默吞掉。
4. **空 gms**:若 prime 后 sink 为空(理论上不应发生),返回空列表,由用户判断。

## 5. 测试(`test/test_v4_capture_fx.py`,GPU 跑;CPU skip)

1. `capture_fx` 返回 `FxCaptureResult`;`r.compiled(*args)` 数值对齐 eager。
2. `r.gms` 非空,且至少一张图含 `triton_kernel_wrapper_mutation` HOP 节点(确认抓到
   融合产物宿主图)。
3. `r.gms[0]` 是 `torch.fx.GraphModule`,节点可遍历(为喂 v2 铺路)。
4. context 退出后 `device_codegens[device].fx_wrapper_codegen` 与
   `config.fx_wrapper` 恢复原值。

## 6. 与 v2 的衔接(后续,非本设计范围)

`r.gms[0]` 是宿主 FX 图,节点为 `empty_strided` 分配、`triton_kernel_wrapper_mutation`
HOP(融合核启动)、aten `OpOverload` fallback。把它喂给 v2 需要 v2 translator 新增
"Triton-kernel-launch step"(从 C++ 启动 cubin),详见
`2026-05-28-v3-design.md` / fx_wrapper 版 v3 的讨论。v4 只负责把 gm 交到用户手里。
