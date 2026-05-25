# C++ Dispatcher-Level Capture/Replay — Design

## 1. 目的

把 `agent_space/dispatch_capture_replay.py` 的 Python 层 PoC 下沉到 C++,
真正消除 dispatcher 在 replay 期的开销,且**原生支持动态 shape**——同一份
trace 可在不同 input shape 下反复 replay,无须重 capture。

## 2. 核心观察

`Dispatcher::call` 每次 op 调用都重做下列工作:
1. 从 input tensors 的 `key_set()` ∪ TLS include 中提取 dispatch keyset
2. `OperatorEntry::lookup(keyset)` 在表里查 `KernelFunction*`
3. alias key 展开(`Autograd` → `AutogradCPU`,`CompositeImplicitAutograd`
   → 具体后端)
4. 沿 dispatch 链 redispatch:AutogradCPU → ADInplaceOrView → CPU/CUDA...

关键事实:**第 1–3 步的输出只依赖 `(device, dtype, layout, requires_grad)`,
不依赖 shape / strides / data_ptr**。也就是说,只要这四个属性不变,resolved
`KernelFunction*` 就是不变的。

shape 不参与 dispatch 决策,只在 kernel **内部** 被读取(算 output sizes、
broadcast、分配 workspace 等)。kernel 是 boxed 接口,每次调用时从 stack
上的 `Tensor` 实时读 `sizes()`/`strides()`/`data_ptr()`。

**结论**:capture 一次拿到 `KernelFunction*` 缓存住,replay 时绕过
`Dispatcher::call` 直调 `KernelFunction::callBoxed`,每次都把 input tensor
现读现 box。shape / data 动态变化自动反映到 kernel,无任何额外机制。

## 3. 非目标

- **不**消除 backend kernel 本身的耗时(`at::native::add` 还得跑一次)。
- **不**合并 launch 或做 graph 级优化(那是 inductor/cudagraph 的事)。
- **不**支持 input tensor 对象被换掉。trace 持有 capture 时输入 tensor 的
  强引用;replay 用的是同一个 Python `Tensor` 对象——其 metadata 可以变,
  对象身份不能变。详见 §5 关于 placeholder 机制的讨论。
- **不**支持 `(device, dtype, layout, requires_grad)` 在 capture 与 replay
  之间发生变化。这些变化改 dispatch keyset,cached `KernelFunction*` 不再
  正确。replay 入口 validate,不匹配 raise(或 fallback 到 eager-recapture,
  视未来扩展决定)。
- **v1 不**做 autograd capture——capture 区域必须 `torch.no_grad()`,否则
  raise(详见 §7;不是物理限制,是速度/复杂度取舍,backward 支持作为 v2
  扩展,见 §17.1,触发条件:v1 收益验证达预期)。
- **v1 不**支持 "shape-derived literal"(任何由 Python 整数运算从 `x.shape[i]`
  推算出的 size 参数,如 `x.view(x.shape[0]//2, 2, -1)` 中的第一维)。Python
  在调用 view 之前已经把 `x.shape[0]//2` 求值成具体 int,dispatcher 看到的就
  是字面值,我们也只能按字面值录入 trace。详见 §8。要解决这个需要 SymInt 级
  别的符号化跟踪——但那条路最合理的落地形态不是自研 FX graph 解析,而是把
  PoC 包装成 `torch.compile` 的 backend(见 §17.6)。
- **不**改 PyTorch 核心,作为 out-of-tree C++ 扩展实现(`torch.utils.cpp_extension`
  风格)。

## 4. 与已有机制对比

| 方案 | dispatcher 开销 | Python 入口 | 真正 kernel | dynamic shape | 多次 replay |
|---|---|---|---|---|---|
| eager | 全付 | 全付 | 全付 | ✓ | n/a |
| Python `__torch_dispatch__` PoC | 全付 | 全付 + loop 开销 | 全付 | ✓ | ✓ |
| **本方案** | **几乎 0** | **0** (C++ 内闭环) | 全付 | **✓ in-place + shape 自由** | ✓ |
| cudagraph / aclgraph | 几乎 0 | 0 | 合并打包 | ✗ (按 shape 分桶重建) | ✓ |
| AOTInductor | 几乎 0 | 0 | 部分融合 | ✓ (SymInt) | ✓ |

定位:**与 cudagraph 类似的"replay 期 host 几乎 0 开销"档,但保留逐 op
launch、对 dynamic shape 零成本**。适合 cudagraph 不能用的 data-dependent
控制流 / 动态 shape 场景。

## 5. 输入 metadata 的动态读取

trace 内每个 step 用一个 `StepInputRef` 描述 input 来源:

```cpp
struct StepInputRef {
    enum class Kind { kCapturedTensor, kPrevStepOutput, kLiteral };
    Kind kind;
    union {
        size_t captured_tensor_idx;   // index into Trace::captured_tensors_
        struct { size_t step; size_t output_slot; } prev;
        c10::IValue literal;          // int/float/Scalar/bool/...
    };
};
```

- `kCapturedTensor`:外部输入(用户传入 / 模型参数)。**trace 持有这个
  `Tensor` 的强引用**;每次 replay 时从这个 `Tensor` 当场读 sizes/strides/
  data_ptr。如果用户 `a.fill_(...)` 或 `a.resize_(...)` 改了它,replay 看到
  新值。
- `kPrevStepOutput`:前面某个 step 在本次 replay 产生的输出。维护一个
  `outputs[step]` 表,每个 step 把 boxed kernel 的输出存进去,后续 step
  按 `(step, output_slot)` 索引取出。
- `kLiteral`:常量参数(reduction 的 `dim`、`alpha` Scalar 等)。capture 时
  从 stack 拷一份 IValue 即可。

replay 一次的执行循环:

```cpp
void Trace::replay() {
    validate_capture_keys_still_valid();   // §10 风险 1/2
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKeySet(kCaptureKey));

    std::vector<std::vector<c10::IValue>> outputs(steps_.size());
    torch::jit::Stack stack;
    stack.reserve(MAX_ARITY);

    for (size_t i = 0; i < steps_.size(); ++i) {
        auto& step = steps_[i];
        stack.clear();
        // 重建 stack:tensor 输入按引用实时读 metadata
        for (auto& ref : step.inputs) {
            switch (ref.kind) {
                case Kind::kCapturedTensor:
                    stack.push_back(captured_tensors_[ref.captured_tensor_idx]);
                    break;
                case Kind::kPrevStepOutput:
                    stack.push_back(outputs[ref.prev.step][ref.prev.output_slot]);
                    break;
                case Kind::kLiteral:
                    stack.push_back(ref.literal);
                    break;
            }
        }
        // 直调 cached KernelFunction,完全跳过 Dispatcher::call
        step.kernel->callBoxed(
            c10::OperatorHandle(step.op_def),
            step.effective_ks,
            &stack);
        // 收输出
        outputs[i].assign(stack.begin(), stack.end());
    }
}
```

**动态 shape 全部由 `stack.push_back(tensor)` 这一步处理**——`Tensor` 是
`TensorImpl*` 智能指针,push 时只增加引用计数,kernel 在 stack 上读到的是
当下最新的 metadata。**不需要任何额外机制。**

### 5.1 支持的动态变化

| 变化类型 | 是否支持 | 备注 |
|---|---|---|
| `a.fill_(x)` / in-place value mutation | ✓ | 自动 |
| `a.copy_(b)` | ✓ | 自动 |
| `a.resize_(new_shape)` (storage cap 内) | ✓ | kernel 读 new sizes |
| `a.resize_(new_shape)` (超过 storage cap) | ✓ | PyTorch 自动 reallocate,新 data_ptr,但同一个 Tensor 对象——replay 自动用新指针 |
| 完全换对象 `a = torch.empty(...)` | ✗ | trace 持有的是旧对象;需用 §5.2 的 placeholder |
| 改 dtype/device/layout | ✗ | 改 dispatch key,cached KernelFunction* 失效;raise |
| 改 requires_grad | ✗ | 同上(影响 Autograd key) |

### 5.2 Placeholder(可选扩展)

如果业务场景是"每次循环重新分配 `out = torch.empty(new_shape)`",需要在
capture 入口显式声明 placeholder:

```python
with tdc.capture() as trace:
    placeholder_a = tdc.placeholder("a")
    placeholder_b = tdc.placeholder("b")
    out = my_function(placeholder_a, placeholder_b)

# replay 时绑定真实 tensor
trace.bind(a=real_a_v1, b=real_b_v1); trace.replay()
trace.bind(a=real_a_v2, b=real_b_v2); trace.replay()   # 完全不同的对象
```

placeholder 是一个特殊 Tensor 子类(空 storage,Python 端用 `_make_wrapper_subclass`
做),在 capture fallback 里被识别并存为 `StepInputRef::kPlaceholder("a")`;
`trace.bind()` 时把名字映射到真实 tensor,replay 用映射后的对象 push 入 stack。

placeholder 机制属于 v2,初版不包含。初版的约束就是"原对象不变,metadata
可变"。

## 6. 架构

### 6.1 Capture 点的选择

权衡过的几个 hook 位置:

| 位置 | 优点 | 缺点 | 选 |
|---|---|---|---|
| `c10::Dispatcher::call` 内 patch | 抓得最全 | 改核心,版本绑定 | ✗ |
| 自定义 DispatchKey + fallback | 标准外部扩展 | 仍要被 dispatcher 调一次(capture 期,正是我们要的) | **✓** |
| Python key 的 `__torch_dispatch__` hook | Python 实现简单 | 已被 PoC 证伪——位置在 dispatcher 外,省不到 dispatcher | ✗ |

**选**:用 `TESTING_ONLY_GenericMode` 作为 capture key,配合 TLS
`IncludeDispatchKeyGuard` 启用。capture 期 fallback 每个 op 都进一次,
之后 replay 期把它从 TLS 里排除,直调 cached kernel。

> ⚠ 历史:最初设计选用 `PrivateUse2`(backend bit),但有三个本质问题
> 让它不可行,见 §6.1.1 — 现在和未来都不应使用 PU2 作为 capture key。

### 6.1.1 为什么不选 PrivateUse2(及 PrivateUse 系列 backend bit)

PrivateUse1/2/3 是 backend bit,位于 dispatcher 链路最底端(优先级低于
所有 functionality keys,包括 AutogradFunctionality #333、ADInplaceOrView
#304、AutocastCPU #349、Tracer 等)。最初设计倾向用它,实施时发现三重问题:

**问题 1:backend bit 与 Dense functionality 共享 bit 位**

DispatchKeySet 内部把 backend bit 和 Dense functionality 用同一组 bit 位
表示。`keyset - DispatchKeySet(PrivateUse2)` 这种 subtraction 会同时**清掉
Dense 功能位**,导致 redispatch 时 effective_ks 退化为空,composite kernel
内部的 redispatch 报"no kernel for Undefined"——之前实施时踩过的真实坑。

要解决得自己实现"只清 backend bit、保留 Dense"的工具(`remove_backend()`
是可用 API),但每个用 keyset 计算的地方都得记得用对——本质上是把
backend bit 当 functionality key 用,违反设计意图。

**问题 2:位置低于 autograd → 看不到 forward 高层 op,被动接收 autograd
内部行为**

PU2 fallback 在 dispatcher 链中**先经过 AutogradCPU wrapper、ADInplaceOrView、
Autocast 等所有 functionality keys 之后**才触发。这意味着:

- forward 时,我们看到的是 autograd wrapper 已经做完 `collect_next_edges`、
  `SavedVariable(self)` 之后的 decomposed leaf op
- autograd 内部把 forward 输入 shape 通过 `SavedVariable` **作为 IntArrayRef
  保存到 backward node**,这事**在 dispatcher 之外发生**——我们无论用哪个
  key 都看不到。但 PU2 因为更晚被触发,**多了一层"我们和 autograd
  内部行为已经无法解耦"的耦合**

**问题 3:autograd wrapper 的 reduced_ks 不保证保留 PU2 backend bit**

autograd codegen 出来的 wrapper 内部用 `at::redispatch::xxx(reduced_ks, ...)`
继续往下走。reduced_ks 的构造逻辑(`ks & after_ADInplaceOrView_keyset` 等)
**不保证保留 PU2 backend bit**——取决于具体实现细节和 PyTorch 版本。这会导致
某些 autograd 内部 dispatch 调用绕过我们的 fallback,trace 不完整。

GenericMode 在 #411(仅次于 PythonDispatcher / PreDispatch),**位于所有
wrapper 之上**,几乎所有 redispatch chain 都会先经过它,trace 完整性更可靠。

**结论**:PrivateUse2 看似是"独立 backend"的自然选择,但实际上是三重坑的
组合,且没有任何场景下它能做 GenericMode 做不到的事。**当前实现不用,未来
方向也不用**(包括 v2 走 aot_autograd backend 路线时,fallback 仍应选
GenericMode)。

如果要彻底脱离 `TESTING_ONLY_*` 这种测试预留 key(详见 §13 风险 10),正确
做法是**向 PyTorch 上游提案新加一个 functionality key**(如 `OpDispatchCapture`),
不是回退到 PU2。

### 6.2 capture fallback 内部逻辑

```cpp
void capture_fallback(const c10::OperatorHandle& op,
                      c10::DispatchKeySet ks,
                      torch::jit::Stack* stack) {
    auto& ctx = CaptureContext::current();

    // 算"如果 capture key 不在 TLS 里,这次 op 实际会走的 keyset"
    auto effective_ks = ks - DispatchKeySet(kCaptureKey);

    // 关键:从 OperatorEntry 拿到该 keyset 对应的 KernelFunction
    // 这一步就是 Dispatcher::call 每次都做的 routing 工作 —— capture
    // 只做一次,replay 复用
    const auto& kernel = op.operatorDef().op.lookup(effective_ks);

    // 录每个 input 的来源:captured / prev_step / literal
    auto input_refs = classify_inputs(*stack, ctx);

    // 在 capture 期 eager 跑一次,产生真实输出供后续 step 链接
    op.callBoxed(stack);
    auto output_ivalues = collect_outputs(*stack, ctx);

    ctx.record({
        .kernel        = &kernel,         // 已 resolved
        .effective_ks  = effective_ks,
        .op_def        = &op.operatorDef(),
        .inputs        = std::move(input_refs),
        .n_outputs     = output_ivalues.size(),
    });
    // op.callBoxed 已经把输出 push 回 stack,fallback 自然返回
}
```

`OperatorEntry::lookup(keyset)` 这一行是整个设计的核心——dispatcher 每次都
做、replay 时跳过、capture 一次就够。

## 7. 与 autograd 的关系

capture key (`TESTING_ONLY_GenericMode`, #411) 位于 dispatcher 链最顶端
附近,**高于** `AutogradFunctionality`(#333)和 `ADInplaceOrView`(#304)。
capture 时 dispatcher 链如下:

```
CaptureFallback (GenericMode)   ← 我们在这里:第一个被触发,record step
  ↓ exclude(GenericMode) + redispatch(effective_ks)
AutogradCPU wrapper             ← 跑 collect_next_edges / save_for_backward /
                                  构造 backward node;挂到 capture-time 的 output
  ↓ AutoDispatchBelowADInplaceOrView, redispatch
ADInplaceOrView                 ← 跑 view 元信息 / in-place version bump
  ↓ redispatch
CPU kernel                       ← 真实计算
```

即:**capture 期间 autograd 是真的跑了的**,我们的 fallback 在 autograd
之**前**触发,先 record,然后 redispatch 让 autograd 正常处理。replay 时
为了避免重新跑一遍 autograd wrapper、防止 `.grad` 被加上新 grad_fn 而
不能 resize_,我们用 `at::AutoDispatchBelowAutograd` 跳过 autograd
wrapper,直接走到 backend kernel。

这就是 autograd 不被支持的真正原因:**capture-time 建好的 backward graph
绑定到了 capture-time 的 input/output 对象,replay 产生的新 output 没有
grad_fn,且用户即便持有 capture-time output 调 backward,`save_for_backward`
存的也是 capture-time 的 input snapshot,与 replay 期变化过的 input 无关 ——
任何方向都得到错误梯度或抛错**。

**结论 (v1)**:capture 必须 `torch.no_grad()`。Python API 强制:

```python
with tdc.capture() as trace:
    if torch.is_grad_enabled():
        raise RuntimeError("capture() must be inside torch.no_grad()")
    ...
```

这是**速度 vs 复杂度**的取舍,不是物理限制——支持 autograd 的方案(让
replay 跑 autograd wrapper 或把 fw+bw 一起 capture)记入 §17 后续扩展,
等 v1 收益验证后再决定是否做。

## 8. View ops 与动态 shape 的相互作用

view ops(`t`, `view`, `reshape`, `as_strided`, ...)的 boxed kernel 不分配
新 storage,返回共享 storage 的新 `TensorImpl`。**每次 replay 都会产生新
的 TensorImpl**,因为它是基于当前 input 的 metadata 现算的。

我们的 `outputs[step][slot]` 表正好 handle:后续 step 通过 `kPrevStepOutput`
索引拿到的是**这次 replay 的 new view**,不是 capture 时的 old view。

动态 shape 下的特殊情形:

- `view(...)` 的目标 shape 在 capture 时是常量(literal IValue 存进 args),
  replay 时这个常量不变。如果原 tensor 在两次 replay 间从 `[4,8]` 变
  `[8,8]`,而 view 的目标是 `[-1, 8]`,kernel 会算出新的实际 shape——OK。
- 但如果 view 目标是 `[2,4,4]` 硬编码,replay 时 input 变 `[8,8]`(32
  elements 不等于 32... 这里假设一致),view 仍按 `[2,4,4]` reshape——OK,
  数值是用户责任。
- `reshape` 在某些情况下会触发 copy,某些情况下纯 view。kernel 内部依旧
  依赖当时 metadata 决定;cached `KernelFunction*` 还是同一个(reshape 的
  CompositeImplicitAutograd kernel),没问题。

特殊情形——`as_strided` 的 size/stride 参数:这两个参数是 args 里的
literal `IntArrayRef`,**在 capture 时被冻结**。如果用户期望 stride 跟着
input shape 变,他必须在每次 replay 之间手动调整 input,然后 trace 内
`as_strided` 还是用 capture 时的 stride 常量。这是真正的限制,**文档列入
**:不要在 capture 区域出现 size/stride 依赖动态 shape 的 view ops。如果
非要支持,需要在 Python 侧把 size/stride 也做成 placeholder。

### 8.1 shape-derived literal:dispatcher capture 的根本盲点

`as_strided(size, stride)` 是这类问题的一个特例。更普遍的形式是 **size 参数
由 Python 端从 `tensor.shape[i]` 派生出来**:

```python
y = x.view(x.shape[0] // 2, 2, -1)    # 第一维由 x.shape[0]//2 推出
z = w.reshape(batch * seq_len, hidden)  # 由两个维度乘积推出
```

Python 在调用 `view` / `reshape` 之前,**已经把 `x.shape[0]//2` 求值成具体 int**:

```
x.shape          → tensor.sizes() 直读 TensorImpl 字段,不经 dispatcher
x.shape[0]       → Python int,不经 dispatcher
... // 2         → Python int 运算,不经 dispatcher
x.view(<int>, 2, -1)  → 经 dispatcher,但 size 参数已是字面值
```

我们的 fallback 在 dispatcher 层接到的 size 参数是 `IntArrayRef([4, 2, -1])`,
**完全没有"从 x.shape 来的"这条 lineage 信息**。trace 把它存为 `kLiteral`
冻结到 capture-time shape。

这是 v1 的**根本盲点**,不是一个能用更聪明的 capture 逻辑修复的 bug —— Python
端的求值发生在我们能介入的最低点之前。

### 8.2 为什么 torch.compile 没这个问题

`torch.compile` 通过 **Dynamo 在 Python 字节码层接管执行**,实现"延迟整数求
值":

1. `x.shape[0]` 在 Dynamo trace 时返回 `SymInt(s0)`,不是 `int(4)`
2. `s0 // 2` 不计算,返回新的 `SymInt(s0 // 2)`
3. 生成的 FX graph 节点里 `view` 的 size 参数**是 SymInt 表达式**,不是字面值
4. 运行时按 input 当下的 size 实例化 SymInt,再喂给 aten 内核

这条机制无法在 dispatcher 层重建——dispatcher **看不到** Python 字节码,看不到
`//`,看不到 `x.shape` 的属性读取。要拿到 lineage,**必须从 FX graph 读**,而 FX
graph 只能由 Dynamo 这样的字节码级 trace 框架产生。

### 8.3 v1 PoC 的有意识取舍

下面这些场景在 v1 是**明确不支持**的(归入 §3 非目标):

| 用户写法 | v1 行为 |
|---|---|
| `x.view(M, N)` 中 `M`、`N` 都是 Python 字面常量 | ✓ shape 不变就 OK,变就要重 capture |
| `x.view(-1, N)` | ✓ kernel 内 `-1` 自动算,动态 OK |
| `x.transpose(0, 1)` / `permute([1,0,2])` / `squeeze()` / `unsqueeze(d)` | ✓ dim 索引与 shape 大小无关,完全动态 |
| `x.view(x.shape[0]//2, 2, -1)` | ✗ 第一维烘焙在 trace,shape 变即错 |
| `x.reshape(b*s, h)` 其中 b/s 由 shape 派生 | ✗ 同上 |
| `as_strided(size=..., stride=...)` | ✗ 全部烘焙 |
| `.sum().backward()` 中 backward 的 `expand` | ✗ 同样烘焙了 forward 的 shape |

**用户应对**:
1. 优先用 `-1` marker(`view(-1, N)`、`reshape(-1, ...)`)避开 shape-derived literal
2. 用 dim-index 类操作(transpose/permute/squeeze)替代 size-list 类操作
3. 必须用 shape-derived literal 的代码,**按 shape 桶分别 capture**(类似 cudagraph)
4. 真要完整 dynamic shape,**用 `torch.compile`,不要用 v1 PoC**

v1 PoC 的价值定位:**"dispatcher overhead 是瓶颈,且 shape pattern 简单"** 的场景。
重叠完全 dynamic shape 不是 v1 的目标。

## 9. Python API

```python
import torch_dispatch_capture as tdc

# 唯一形态:context manager,默认 dynamic 语义
with tdc.capture() as trace:
    out = my_function(a, b)

trace.replay()                                # 用 a, b 的当前 metadata
a.fill_(3.0); trace.replay()                  # 自动反映新 value
a.resize_(16, 16); b.resize_(16, 16)          # 自动反映新 shape
trace.replay()                                # 不用重 capture

# 显式 begin/end
handle = tdc.begin()
my_function(a, b)
trace = tdc.end(handle)

# 内省
trace.size()                # int,op 数量
trace.entries               # list[dict] of (op_name, dispatch_keys, input_kinds)
trace.discard()             # 主动释放
print(trace)                # 打印 op 序列
```

## 10. C++ API

```cpp
namespace torch_dispatch_capture {

class Trace {
public:
    void replay();
    size_t size() const;
    std::string dump() const;
    ~Trace();      // 释放所有 captured Tensor 强引用
private:
    struct Step {
        const c10::KernelFunction* kernel;
        c10::DispatchKeySet effective_ks;
        c10::impl::OperatorEntry* op_def;
        std::vector<StepInputRef> inputs;
        size_t n_outputs;
    };
    std::vector<Step> steps_;
    std::vector<at::Tensor> captured_tensors_;
};

class CaptureContext {
public:
    static CaptureContext& current();
    static void begin();
    static std::unique_ptr<Trace> end();
    static bool is_active();
    void record(Step&&);
};

}
```

## 11. 实现文件清单

```
torch_dispatch_capture/
├── csrc/
│   ├── capture_context.h         # CaptureContext + Trace 声明
│   ├── capture_context.cpp       # TLS context 实现
│   ├── capture_fallback.cpp      # boxed fallback,注册到 TESTING_ONLY_GenericMode
│   ├── trace.cpp                 # Trace::replay 实现
│   └── bindings.cpp              # pybind11 → Python
├── python/
│   └── __init__.py               # capture() context manager
├── test/
│   ├── test_correctness.py       # 与 eager 数值对齐
│   ├── test_dynamic_shape.py     # 不同 shape 反复 replay
│   ├── test_mutation.py          # in-place 传播
│   ├── test_view_ops.py          # view 在动态 shape 下正确
│   └── test_benchmark.py         # eager vs replay
├── setup.py                      # torch.utils.cpp_extension
└── README.md
```

构建用 `torch.utils.cpp_extension.CppExtension`,不需要改 PyTorch 源码。

## 12. 验证用例

### 12.1 正确性

- `test_arithmetic` — 元素级 op 链 replay 与 eager 数值完全一致。
- `test_view_ops_static_shape` — `t/view/reshape/transpose` chain,replay
  期间 input shape 不变,数值正确。
- `test_inplace_propagation` — `a.fill_(10); trace.replay()` 用新 a 算
  (对齐 Python PoC EXP 4)。
- `test_no_grad_required` — `grad_enabled=True` 下 capture 抛 RuntimeError。
- `test_dtype_change_rejected` — capture 时 float32 → replay 时 a 变 float64,
  raise(因为 dispatch key 变了)。

### 12.2 动态 shape 专项

- `test_resize_same_storage` — capture `(4,8) → (8,8)` add。replay 时
  `a.resize_(2, 8); b.resize_(2, 8)`,replay 应得 `(2,8)` 输出,数值与
  在新 shape 下 eager 跑的一致。
- `test_resize_realloc` — `a.resize_(1024, 1024)` 超过原 storage,PyTorch
  重新 allocate,data_ptr 变。replay 自动拿新 ptr,数值正确。
- `test_varied_batch` — capture 时 batch=4,replay 时 batch ∈ {1,2,4,8,16}
  各跑一遍,与对应 eager 一致。模拟 KV cache 滑动 / variable seq len。
- `test_view_with_shape_dep` — 如果 trace 包含 `tensor.view(-1, 8)` 这种
  shape 自适应 op,replay 在不同 input shape 下正确。
- `test_view_with_shape_literal` — 如果 trace 包含 `view(2,4,4)` 硬编码,
  replay 时 input shape 变得不兼容,raise(kernel 自然 raise)。文档明确。
- `test_as_strided_frozen` — `as_strided(size=[4,4], stride=[8,1])` 这种
  stride 硬编码,文档明确不支持动态 shape;测试断言行为符合"shape 变了
  就出错"的预期。

### 12.3 Benchmark

工作负载:
- 8×8 elementwise add × 64(纯 dispatch overhead)
- DeepSequential(32 × `Linear(8,8)`)
- variable seq len:capture 一次,replay 在 batch ∈ {1,4,16,64} 各 100 次

```
[cpp_dispatch_capture benchmark]
n_ops=128 workload=elementwise
  eager_med    = X.X µs/call    (Y.Y ns/op)
  replay_med   = A.A µs/call    (B.B ns/op)
  speedup      = N.NN×
  per_op_save  = (Y.Y - B.B) ns

[dynamic shape]
n_ops=128 workload=elementwise(batch=B)
  B=1:   eager=X1 us  replay=A1 us  speedup=N1×
  B=4:   ...
  B=16:  ...
  B=64:  ...
  (一次 capture,N 次 replay,无重 capture)
```

预期(基于 Python PoC 推算):
- elementwise per-op 节省 0.9–1.2 µs:基线 1.8 µs/op → 0.6–0.9 µs/op (2–3×)
- DeepSequential per-op 节省 1.2–1.5 µs

硬性 assert:**任何 batch size 下,replay 不能慢于 eager**。**replay 在不
同 batch 间共用同一份 trace,不允许触发重 capture**。

## 13. 风险与已知限制

1. **`OperatorEntry*` / `KernelFunction*` 的生命周期**。op (re-)registration
   会更新 `OperatorEntry::dispatchTable_` 里的 slot。存裸指针不安全:存
   **(OperatorEntry*, dispatch_key, version)**;replay 入口 `version ==
   entry.version()` 检查,不匹配则 fallback 到一次 lookup 并刷新缓存(O(1),
   不影响热路径)。

2. **alias key 展开**。`OperatorEntry::lookup` 已 handle,我们拿到的
   `KernelFunction&` 是最终具体 kernel 的引用,**不**是 alias key 的 stub。

3. **fallthrough kernel**。`BackendSelect` 等 key 的 kernel 是 fallthrough,
   `lookup` 自动跳过,返回下一层。OK。

4. **跨 stream 一致性**。CPU 无问题;CUDA / XPU 需要 capture 时 stream 与
   replay 时 stream 关系明确(可用当前 `getCurrentStream`)。与 cudagraph
   同样限制,文档说明。

5. **Custom autograd Function / hook**。capture 区域 `no_grad`,
   `torch.autograd.Function.forward` 内部的 dispatcher 调用会被 capture,
   backward 不会(因为根本没建)。预期行为。

6. **TorchDispatchMode 与 capture 共存**。如用户在 capture 时压了一个
   `TorchDispatchMode`(`Python` key,enum 229),由于 `TESTING_ONLY_GenericMode`
   (#411) **高于** `Python` key,我们的 capture fallback **先 fire**,
   redispatch 后才轮到用户 mode。即:我们看到的是 mode 处理**之前**的原始
   op。要让 mode 先跑,需要用户先 enter mode 再 enter capture——但即便如此,
   mode 的 redispatch 还会再回到我们(因为 TLS 里我们的 key 一直在),
   可能产生不预期的双重拦截。**建议不与 TorchDispatchMode 嵌套使用**。

7. **`as_strided` 类硬编码 size/stride 的 view ops**。capture 时 size/stride
   作为 literal IValue 冻结,replay 时不跟随 input shape 变化。文档明确
   不在动态 shape 支持范围。如有必要可加 placeholder 扩展(§5.2)。

8. **shape 不兼容时 kernel 抛错**。如果 capture 时 `a:[4,8] @ b:[8,4]`,
   replay 时改成 `a:[4,7]`,`addmm` kernel 自己会 raise。我们不预先校验
   shape——这是 kernel 的责任。**好处**:对合法变化零开销;**代价**:错误
   信息来自 kernel,不是我们。文档建议用户对 trace 的 shape 兼容性自己负责。

9. **多次 replay 间的中间 tensor 内存**。trace 在 capture 期会产生中间
   output;只要后续 step 引用这些 output(`kPrevStepOutput`),它们就会被
   引用计数维持。但**每次 replay** 时这些中间 output 是新分配的;PyTorch
   caching allocator 会复用。整体行为与 eager 一致,不"钉死"内存。

10. **不与 `torch.compile` 嵌套**。`torch.compile` 内 FunctionalTensor /
    ProxyTensor mode 与我们的 capture key 共存关系复杂。detect 到
    `torch._dynamo.is_compiling()` 时拒绝 capture,raise。

## 14. 与 Mode / Subclass 的优先级关系

`torch/csrc/utils/python_arg_parser.cpp:588` 的
"Note [__torch_dispatch__ dispatching order]":
**user mode → user subclass → infra mode → infra subclass**。`Python` key
(enum 229)的处理位于 dispatcher 中游,我们的 `TESTING_ONLY_GenericMode`
(#411) 位置更高(仅次于 PythonDispatcher / PreDispatch)。

实际触发顺序:
```
CaptureFallback (TESTING_ONLY_GenericMode)  ← 我们最先触发 (#411)
  ↓ exclude(GenericMode) + redispatch(effective_ks)
TorchDispatchMode (if any)                   ← Python key (#229)
  ↓ redispatch
AutogradCPU / ADInplaceOrView / ...          ← #333 / #304
  ↓ redispatch
真实 backend kernel (CPU/CUDA/...)            ← Dense + backend bit
```

意义:capture 看到的是用户写的**原始** op(autograd / mode 都还没处理),
record 之后我们 redispatch,让 dispatcher chain 继续正常处理。这样 trace 里
存的是用户层面的 op 序列,而**不是** mode / autograd 分解后的 leaf。replay
时仍然走完整 chain(autograd 重新处理一遍)——除非用户启用 v1 的
`AutoDispatchBelowAutograd` 路径(`allow_grad=True` 时 replay 端默认开启)。

历史说明:早期设计中考虑过用 `PrivateUse2`(backend bit,优先级低于
所有 wrapper),那条路有三重问题(详见 §6.1.1),所以选了 GenericMode 这条
"在所有 wrapper 之上"的路径。
autograd、view-meta 都处理过了。这是想要的——replay 不需要再过这些层。

## 15. 实施分阶段

| 阶段 | 内容 | 风险 | 输出 |
|---|---|---|---|
| 1 | C++ 扩展骨架 + capture fallback + 仅录 op 名 + Python `capture()` 上下文 | 低 | smoke test |
| 2 | replay 实现:cached KernelFunction 直调,outputs rewire | 中 (view 处理) | 正确性测试全过 |
| 3 | 动态 shape 测试 (resize / varied batch) | 低 | dynamic 测试全过 |
| 4 | benchmark + 报告 | 低 | 数值表,确认 ≥ 2× per-op |
| 5 | 边角:reset hook、no_grad 强制、dtype change rejection | 低 | 测试套件 GA |
| 6 | 文档 + setup.py packaging | 低 | pip install 可用 |

## 16. 验证通过标准

- [ ] `pip install -e torch_dispatch_capture` 在 PyTorch 2.x 上构建通过。
- [ ] `import torch_dispatch_capture` 不报错,`capture()` 可用。
- [ ] 所有 `test/test_*.py` 通过,**尤其 `test_dynamic_shape.py` 全套**。
- [ ] elementwise benchmark:**replay per-op < 0.5 × eager per-op**。
- [ ] DeepSequential benchmark:**replay per-op < 0.6 × eager per-op**。
- [ ] dynamic shape:**capture 1 次,4 个不同 batch 各 100 次 replay,
      累计耗时 < 同 workload eager 累计耗时 × 0.6**。
- [ ] 内存:`trace = None; gc.collect()` 后,capture 时分配的 storage
      被全部释放。

## 17. 后续扩展

按预期收益与改动量排序,**触发条件:v1 验证 elementwise / DeepSequential
benchmark 至少 ≥ 1.5× per-op 节省**。如果 v1 收益不达预期,以下都不做。

### 17.1 Backward capture(优先级最高,前提条件:v1 收益足够)

支持 `requires_grad=True` 下 capture。两条可选路线,各有取舍:

**路线 B1:replay 期重跑 autograd wrapper**

- capture 时把 `lookup` 用的 keyset **保留** Autograd 系列,cached 的
  `KernelFunction*` 是 `AutogradCPU wrapper`,不是底层 CPU kernel。
- replay 时不 exclude Autograd,直调 wrapper —— wrapper 内部 `save_for_backward`
  / `collect_next_edges` / 构造新 backward node 都正常发生,**snapshot 是
  replay 期当下的 input 值**,grad_fn 挂在 replay-time 的 new output 上。
- 节省幅度:只剩 `OperatorEntry::lookup` 那一层(≈ 15% per-op,基于 Python
  PoC `_op_dk` 实测)。autograd wrapper 内部仍要 redispatch 一次,那次走
  完整 dispatcher。
- 实现改动:capture_fallback 把 `effective_ks` 从"减掉 Autograd 后"改成
  "完整 ks"。其余几乎不变。

**路线 B2:capture forward + backward(AOT 式)**

- capture API 扩展:
  ```python
  with tdc.capture() as trace:
      out = fn(a, b)
      trace.mark_backward(out, grad_outputs=[grad_out])
  ```
- capture 期 `mark_backward` 触发 `torch.autograd.grad(out, [a, b], grad_out)`,
  让 autograd 真的跑一遍 backward —— 所有 backward kernel 也通过 dispatcher,
  也被我们的 capture_fallback 录进 trace。
- replay 一次 = forward 算子序列 + backward 算子序列连跑,**完全不需要
  autograd**。等价于 AOTAutograd 的轻量版。
- 节省幅度:同 v1(几乎全部 dispatcher),且覆盖 backward。
- 实现改动:中等。需要在 trace 内部区分 fw step 与 bw step,replay 时如果
  用户只要 forward 就停在 fw 终点。需要把 grad input/output 也作为
  placeholder 处理(参考 §5.2)。
- 限制:capture 时 loss 形状要与 replay 时一致(实际上由 §3 的
  `(device, dtype, layout)` 一致性约束自然保证)。

**v2 倾向选 B1 先做**,因为改动最小且支持原生 PyTorch autograd 语义;B2
做为更激进的优化路线,等 B1 也有数据后再考虑。

### 17.2 Placeholder + bind(§5.2)

支持 capture 后绑定不同 tensor 对象。等价于一个简化版 LazyTensor。优先
级中等——大部分场景靠 in-place mutation 已够,只在"每次循环重新分配
output buffer"这种特定 pattern 下需要。

### 17.3 跨 trace 复用 KernelFunction cache

全局 `(op_name, effective_ks) → KernelFunction*` cache,后续 capture 跳过
lookup,capture 期也加速。降低 capture cost,与 replay 性能无关。低优先级。

### 17.4 与 NPU OpDispatchCapture 串联

本设计抓 PyTorch dispatcher 层,NPU 的 `OpDispatchCapture`
(`pytorch_npu/docs/dynamic_capture_demo_design.md`)抓 `EXEC_NPU_CMD` 层。
两者正交,可以同时启用让 dispatcher overhead + ACL host prep 一起省;需
要在 NPU OpCommand hook 里识别我们的 capture key 并跳过自己的 capture
路径。NPU 团队完成 v1 后再对接。

### 17.5 序列化 trace

把 trace 落盘(op 名 + dispatch keyset + literal args),下次 load 时按 op
名重新 `findOp + lookup`,等价于一种轻量 AOT 格式。低优先级,与 AOTI 重叠。

### 17.6 完整动态 shape:作为 `torch.compile` 的 backend(v2 推荐方向)

要解决 §8.1 描述的 shape-derived literal 问题,**没有可行的 dispatcher-only
路径**。但有一个 PyTorch 早已铺好的工业路径:把 PoC 包装成 `torch.compile`
的自定义 backend,**借力 Dynamo + AOTAutograd 已经做完的符号化工作**。

#### 17.6.1 接入点选择

`torch.compile` 的 backend 钩子有三档可选:

| 接入点 | graph 形态 | 我们需要处理的复杂度 |
|---|---|---|
| 直接接 `backend=fn`(Dynamo 原始 graph) | call_method / call_function / operator.* / aten 混杂 | **高**(节点类型十几种,Python op 语义全集) |
| **接 `aot_autograd(fw_compiler=...)`** | functionalized + decomposed 的 core aten graph,SymInt 表达式显式 | **中**(基本只剩 `call_function` + 大约 150 个 core aten op + 一打 `operator.*`) |
| 接 inductor 之后 | 已 codegen,失去 graph 结构 | **不可能** |

**v2 选第二档**:`torch._dynamo.backends.common.aot_autograd` 已经把 graph
处理到"几乎全是 functional aten + 少量 Python operator"的程度。我们写一个
shim 把这个 graph 翻译成扩展版的 trace。

#### 17.6.2 扩展后的 trace 结构

```cpp
struct Step {
    enum Kind {
        kTensorOp,    // op.callBoxed(stack) — 走 dispatcher (aten / prims / 自定义 op)
        kPyCall,      // step.fn(*resolved_args) — Python 解释器内直接调用
    };
    Kind kind;

    // kTensorOp 字段
    c10::OperatorHandle op;
    c10::DispatchKey target_dk;

    // kPyCall 字段:覆盖 _operator.*(含 getitem)、torch.sym_*、白名单 torch API
    py::object fn;

    std::vector<StepInputRef> inputs;
    size_t n_outputs;        // 几乎所有 step 是 1;
                             // 静态多元组返回的 op (max.dim 等) 在 schema 上即 >1
};

struct StepInputRef {
    enum Kind {
        kCapturedTensor,     // Dynamo prelude 传入的 Tensor
        kCapturedInt,        // Dynamo prelude 从 .shape/.size 提出来的具体 int
        kPrevStepOutput,     // (step_idx, slot) — 引用 prev step 的输出
        kLiteral,            // 字面量 IValue (int / float / bool / None / str)
        kList,               // 嵌套列表,元素递归是 StepInputRef
    };
    size_t idx;                                 // kCapturedTensor / kCapturedInt
    size_t step;                                // kPrevStepOutput
    size_t slot;                                // kPrevStepOutput;多数 step slot=0
    c10::IValue literal;                        // kLiteral
    std::vector<StepInputRef> list_elements;    // kList
};
```

设计原则说明:

**(1) Step 只有 2 种 kind**。早期草案在 Step 上单独区分 `SymExpr` 与可能的 `GetItem`,
但实测表明 `operator.getitem` 与 `operator.floordiv` 在 Python 层是完全同型的对象
(`type(...).__name__ == "builtin_function_or_method"`),AOTAutograd 也把它们归入
同一桶 (`autograd_cache.py:212` 的 `builtin_function_or_method` 分支)。统一到
`kPyCall` 后,`fn` 字段承载所有 Python callable,翻译器无需特殊路径区分。
原 `SymExpr` 这个名字也是错的——`operator.getitem` 不是符号运算,是数据访问;
`kPyCall` 更准确表达"在 Python 层调用,不进 dispatcher"。

**(2) StepInputRef 有 5 种 kind**:
- `kCapturedTensor` / `kCapturedInt`:graph 入参,运行时从 Dynamo prelude 拿到具体值
- `kPrevStepOutput(step, slot)`:flat 寻址,不嵌套
- `kLiteral` / `kList`:graph 中的 Python 数据

**(3) `kList` 不限定元素类型**(原 `kIntList` 改名)。下列三类 args 底层数据结构相同:
- `view`/`expand` 的 size:list of int/SymInt
- `cat`/`stack` 的 tensors:list of Tensor
- `permute` 的 dims:list of pure int literal

trace 层不做静态类型区分,dispatcher 在 push IValue 进栈时按 op schema 自然校验。

#### 17.6.3 翻译规则(从 aot graph FX 节点到 trace step)

##### 17.6.3.1 节点类型完备度的源码依据

v2 翻译器需要处理的 FX 节点是一个**封闭有限集**,这件事由 PyTorch 内部的两层
枚举保证,翻译器只需覆盖这些枚举即可声称完备。

**第一层:`node.op` 的 6 种取值**

来源:`torch/fx/interpreter.py:294` 的 `Interpreter.run_node`,通过
`getattr(self, n.op)` 分派,接口固定为以下 6 个方法:

```python
def run_node(self, n: Node) -> Any:
    args, kwargs = self.fetch_args_kwargs_from_env(n)
    return getattr(self, n.op)(n.target, args, kwargs)
```

`n.op` 的合法取值定义在 `torch/fx/graph.py` 的 `Node` 类,共 6 个:
`placeholder` / `get_attr` / `call_function` / `call_method` / `call_module` / `output`。
新增节点类型需要修改 FX 核心,极其罕见。

**第二层:`call_function` 的 target 五大类**

绝大多数计算节点是 `call_function`。它的 `target` 可以是任意 Python callable,
看似无穷,但 AOTAutograd 用 `autograd_cache.py:202` 的 `is_cacheable_function`
做了**显式白名单分类**,任何不在白名单里的 target 会抛 `BypassAOTAutogradCache`
(同文件第 240 行),也就是说**graph 里不会出现白名单之外的 target**:

```python
def is_cacheable_function(target):
    if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):  # 1
        return True
    if is_public_torch_api(target):                                               # 2
        return True
    if isinstance(target, torch._ops.HigherOrderOperator):                        # 3
        return target.cacheable()
    if type(target).__name__ == "builtin_function_or_method":                     # 4
        return True
    if is_safe_torch_function(target):                                            # 5
        return True
    if function_name in SAFE_NON_TORCH_FUNCTIONS:                                 # 5b
        return True
    return False  # ← BypassAOTAutogradCache raised
```

逐类拆解:

| # | 类别 | 实例 | 在 graph 里出现的方式 |
|---|---|---|---|
| 1 | `OpOverload` / `OpOverloadPacket` | `aten.view.default`, `aten.bmm.default`, `prims.convert_element_type.default`, 用户 `torch.library` 自定义 op | dispatcher 注册过的真实算子 |
| 2 | public torch API (限于 `torch.functional` / `torch.nn.functional` 两个模块,`SAFE_TORCH_MODULES`) | `torch.nn.functional.relu` 偶尔(通常已分解) | 极少;一般已被 dynamo 进一步 trace 成 OpOverload |
| 3 | `HigherOrderOperator` | `torch.ops.higher_order.cond` / `while_loop` / `scan` / `invoke_subgraph` / `auto_functionalized_v2` | 控制流 / 子图;伴随 `get_attr` 节点引用子 GraphModule |
| 4 | `builtin_function_or_method` | `_operator.floordiv` / `add` / `mul` / `sub` / `mod` / `getitem` / `eq` / `lt` / `not_` 等 | sym 算术 / 多输出解包 / 比较 |
| 5 | `SAFE_TORCH_FUNCTIONS` 白名单 (`autograd_cache.py:161`) | `torch.sym_max` / `sym_min` / `sym_int` / `sym_float` / `sym_sum` / `_sym_sqrt` / `Size` / `Tensor` / `autograd.grad` | sym helper 函数;前 5 个在 graph 里常见,后 3 个罕见 |
| 5b | `SAFE_NON_TORCH_FUNCTIONS` 白名单 | `einops.rearrange` / `einops.repeat` | 极少;通常已被进一步 trace 成 aten 序列 |

`torch_non_c_binding_in_graph_functions` 也是白名单的一部分(同 `is_safe_torch_function`
分支),涵盖一些非 C 绑定的 torch 内部函数,实测罕见出现。

**完备度结论**

| `node.op` | 在 v2 里如何处理 | 探针覆盖 |
|---|---|---|
| `placeholder` | 翻译为 `kCapturedTensor` / `kCapturedInt`;`SymFloat`/`SymBool` 不支持 | `v2_aot_api.py` 所有用例 |
| `call_function` (类 1) | 翻译为 `kTensorOp` | 所有用例 |
| `call_function` (类 4 sym 部分) | 翻译为 `kPyCall` | sym arith、attention QK 用例 |
| `call_function` (类 4 `getitem`) | 翻译为 `kPyCall(operator.getitem)` | max/split/topk/var_mean/sort 等 |
| `call_function` (类 5 `torch.sym_*`) | 翻译为 `kPyCall` | sym_max/sym_min 用例 |
| `call_function` (类 2 nn.functional 残留) | 翻译为 `kPyCall`(同 fn 通用路径) | layer_norm/dropout(实测已分解) |
| `call_function` (类 3 HOP) | **不支持**,fail-fast | torch.cond 用例 |
| `call_function` (类 5b einops) | 通用 `kPyCall` 路径或不支持 | 未实测(实测易分解) |
| `call_method` | **不支持**,fail-fast | 实测未出现 |
| `call_module` | **不支持**,fail-fast | 实测未出现 |
| `get_attr` (模型参数) | 加入 `captured_tensors_` | (v2 训练支持时启用) |
| `get_attr` (HOP 子图) | 仅伴随 HOP 出现,HOP 已 fail | torch.cond 用例 |
| `output` | 把 args 里 Node 转为 `kPrevStepOutput`,挂到 trace.outputs | 所有用例 |

**总计 `node.op` 6 种,每种都有明确处理(支持或显式 fail)**。`call_function`
里 5 大类只支持 1 + 4 + 5 这三类(及 2 类的少数残留),其余通过显式断言拒绝,
与 §17.6.8 的防御性断言列表完全对应。这构成 v2 翻译器的**封闭完备性证明**。

##### 17.6.3.2 完整翻译规则表

| `node.op` | target 类型 | 翻译为 |
|---|---|---|
| `placeholder` | `val` is `Tensor` | 加入 `captured_tensors_`, ref = `kCapturedTensor(idx)` |
| `placeholder` | `val` is `SymInt` | 加入 `captured_ints_`, ref = `kCapturedInt(idx)` |
| `placeholder` | `val` is `SymFloat` / `SymBool` | **不支持**(v2 范围内未覆盖),翻译时 fail |
| `call_function` | `OpOverload` (aten.* / prims.* / 自定义) | `kTensorOp` Step, inputs 按 args 递归翻译 |
| `call_function` | `operator.*` (含 `floordiv`/`add`/`mul`/`sub`/`mod`/`getitem`/`eq`/`lt` 等) | `kPyCall` Step, `fn` = 该 operator |
| `call_function` | `torch.sym_max` / `sym_min` / `sym_int` / `sym_float` / `sym_ite` / `sym_not` / `sym_sum` 等 | `kPyCall` Step, `fn` = 该 torch.sym 函数 |
| `call_function` | `HigherOrderOperator` (`cond` / `while_loop` / `scan` / `invoke_subgraph` ...) | **不支持**,fail-fast 报错引导用 `backend="inductor"` |
| `call_method` | 任意 | **不支持**(AOT graph 中极罕见),fail-fast |
| `call_module` | 任意 | **不支持**(仅 built-in nn module,实际几乎不出现),fail-fast |
| `get_attr` | HOP 子图引用 | 不会单独到达(HOP 整体已 fail) |
| `get_attr` | 模型参数 | 加入 `captured_tensors_`, ref = `kCapturedTensor(idx)` |
| `output` | — | 把 args 里每个 Node 转成 `kPrevStepOutput`,挂到 trace.outputs |
| 任何 args 中出现 Python `list` / `tuple` | — | 整体翻译成 `kList`,元素递归构造 sub-ref |

补充说明:

**关于 list/tuple 构造**:FX graph 里**没有**对应 `getitem` 的"make_tuple/make_list"
节点。集合构造在 `node.args` 上以 Python `immutable_list` / `tuple` 形态原地存在,
不进 graph 本身。翻译时直接读取 `args` 上的容器结构,构造 `kList` ref。

**关于 `operator.getitem`**:与早期草案不同,getitem **不被翻译器折叠**,而是和其他
`operator.*` 一样翻译为 `kPyCall` Step。理由:

- 多输出 op (如 `max.dim`、`var_mean`) 的输出在 IValue 层是 1 个 tuple,
  `operator.getitem(prev, i)` 是取元素的正常步骤,无需特殊路径。
- `Tensor[]` 返回 op (如 `split`) 输出 1 个 `List<Tensor>` IValue,同样靠
  `getitem` 取元素,**capture_fallback 不需要按 schema 拆 List**。
- AOT graph 经 pytree 摊平后,**永远只产生 1-level getitem**(实测见 §17.6.8),
  所以 `kPrevStepOutput.slot` 始终是 schema 上静态多元组的 slot,**不会嵌套**。

**关于 `kPrevStepOutput.slot`**:仅当 `kTensorOp` 的 schema 静态声明多元组返回时
slot > 0(只有少数 op:`max.dim`、`min.dim`、`sort`、`topk`、`var_mean` 等)。
其余情况一律 slot=0。**绝不应出现 `slot` 指向"上一 step 的内部嵌套元素"** ——
那类访问只能通过显式 `getitem` step 完成。

#### 17.6.4 Replay 算法(扩展自 v1)

```python
def replay(trace, captured_tensors, captured_ints):
    """captured_tensors / captured_ints 来自 Dynamo prelude:
    它在 graph 入口之前已经从用户 Tensor 上提取了 .size() 等具体 int。
    runtime 不再有 SymInt,全是 concrete int + 真实 Tensor。"""
    outputs = [None] * len(trace.steps)

    def resolve(r):
        if r.kind == kCapturedTensor:   return captured_tensors[r.idx]
        if r.kind == kCapturedInt:      return captured_ints[r.idx]
        if r.kind == kPrevStepOutput:   return outputs[r.step][r.slot]
        if r.kind == kLiteral:          return r.literal
        if r.kind == kList:             return [resolve(s) for s in r.list_elements]

    for i, step in enumerate(trace.steps):
        if step.kind == kPyCall:
            # Python callable 直接调用,fn 已经覆盖了 operator.* / torch.sym_* / getitem 等
            args = [resolve(r) for r in step.inputs]
            outputs[i] = [step.fn(*args)]                  # 默认单输出
        else:  # kTensorOp
            stack = [IValue(resolve(r)) for r in step.inputs]
            step.op.callBoxed(&stack)
            # 多元组返回 (max.dim / var_mean / sort 等) 时 schema 决定 slot 数
            outputs[i] = [stack.pop() for _ in range(step.n_outputs)]

    return [resolve(r) for r in trace.outputs]
```

每次 replay,SymInt 算术 step (kPyCall, fn=operator.floordiv 等) 重跑一次得到新的
具体 int;IntList 等容器在 `resolve(kList)` 时现场构造。**SymInt 计算依赖关系通过
step 之间的 `kPrevStepOutput` 边显式保留在 trace 里**,而不是被字面化。

注:从 dispatcher 视角看,`SymInt[]` 形参收到普通 `IntList` IValue 时通过隐式转换
包成"常量 SymInt"(`SymInt(c)` 退化态),没有 runtime 开销。所以 replay 不需要
特意构造 SymInt — 用普通 Python int 喂进栈即可,dispatcher 自然识别。

#### 17.6.5 用户 API

```python
import torch_dispatch_capture as tdc
from torch._dynamo.backends.common import aot_autograd

@torch.compile(backend=aot_autograd(fw_compiler=tdc.tdc_compiler), dynamic=True)
def fn(x):
    return x.view(x.shape[0] // 2, 2, -1)    # ← v1 不支持,v2 支持
```

或者更简洁的封装:

```python
@tdc.compile(dynamic=True)
def fn(x): ...
```

内部就是 `torch.compile + aot_autograd + tdc_compiler` 的组合。

#### 17.6.6 工程量估计

| 子任务 | LoC | 风险 |
|---|---|---|
| `tdc_compiler(gm, sample_inputs)` 主框架 | ~150 行 Python | 低(已有 dynamo 套路) |
| `Step::kPyCall` + `kList` 的 C++ 类型扩展 | ~200 行 C++ | 低 |
| Replay loop 处理 kPyCall / kList | ~80 行 C++ | 低 |
| Aten op 节点翻译 + 输入分类 | ~300 行 Python | 中(core aten 大约 150 个,但绝大多数 schema 很规整) |
| operator.* / torch.sym_* 节点翻译(统一为 kPyCall) | ~50 行 Python | 低 |
| 输出处理(`output` 节点 + 多元组 schema 的 slot) | ~30 行 Python | 低(无需 getitem 折叠) |
| 防御性断言 + HOP/call_method 等不支持节点的错误信息 | ~50 行 Python | 低 |
| 测试 + LLM-style workload 验证 | ~300 行 | 中 |
| **合计** | **~1160 行** | 中 |

相比"自研 dynamo + FX graph 解析"的~3000+ 行 + 几乎所有 Python 语义,v2 走
aot_autograd 接入点是**显著更小的工程**,且与 inductor 共享上游处理逻辑,
PyTorch 升级时的维护成本也低得多。

#### 17.6.7 v1 与 v2 的定位

| 维度 | v1(当前 PoC) | v2(借 torch.compile) |
|---|---|---|
| 触发方式 | `with tdc.capture():` 显式 | `@torch.compile(backend=tdc)` |
| 动态 shape 支持范围 | dim-index 类 ops + `view(-1, ...)` | **完整动态 shape**(含 shape-derived literal) |
| 反向支持 | 实验性 `allow_grad=True` | AOTAutograd 已自带,fw/bw graph 都接到 |
| 工程量 | 已完成,~600 LoC | ~1100 LoC 增量 |
| 性能边界 | LLM decode / 小算子 / 固定 shape 训练 | 同 v1 + 通用动态 shape 推理 |
| 与 PyTorch 的耦合 | 低(只用 dispatcher) | 中(依赖 Dynamo/AOT 接口稳定性) |

**v2 不是 v1 的替代,而是叠加**:
- 简单 / 固定 shape / 极致小 host overhead → 用 v1(没有 compile 开销)
- 复杂 / 真动态 shape / 训练 → 用 v2(走 compile pipeline)

实施触发条件:v1 在产品场景跑通后,如果 LLM 训练或动态 shape 推理有明确需求,
再启动 v2。否则 v1 就够了。

#### 17.6.8 经实测验证的 AOT graph 不变量

`prototypes/v2_aot_api.py` 与 `prototypes/v2_aot_boundaries.py` 通过 16 个边界 case
(多输出 op / sym_* helper / HOP / 函数化 in-place / 复合 API / 切片) 实测得出的
不变量,本节作为 §17.6.2~§17.6.4 设计假设的实证依据。

| 不变量 | 探针用例 | v2 设计依赖于此 |
|---|---|---|
| `dynamic=True` 下 SymInt 被 AOTAutograd 提到 placeholder | view(x.shape[0]//2)、attention QK、切片用例 | trace 入口签名 = (Int×N, Tensor×M);`sym_size` 不进 graph 内部 |
| `operator.*` 与 `torch.sym_*` 在 Python 层同构,且 AOTAutograd 同桶分类 | sym_max/sym_min vs floordiv/mul/etc. | 统一 `kPyCall` step,`fn` 字段承载;不需要按子类型分派 |
| AOT graph 永远只产生 1-level `getitem`(pytree 在所有边界摊平) | max/split/topk/var_mean/sort、cond 多元组分支、嵌套用户代码 | `kPrevStepOutput.slot` 不嵌套;无需链式 ref kind |
| `Tensor[]` 返回 op (split 等) 后续每个元素**独立** getitem 节点 | split 后接 3 个 getitem | capture 端无需"按 schema destructure List<Tensor>";由 kPyCall(getitem) 自然解 |
| Python `__getitem__` (tensor 切片) 完全分解为 `aten.select.int` + `aten.slice.Tensor` | `x[0, :] + y[0, :x.shape[1]]`、`x[::2, 1::3]` | 不需要为 indexing 单独建模;`:` 出现为 `INT64_MAX` 字面量;动态上界自然变 `kCapturedInt` |
| 函数化 in-place 在 graph 末尾留 `aten.copy_.default` fence | y.add_(x)、view().add_(1) | trace 必须保留尾部 copy_ 步骤,否则 caller 看不到 mutation |
| 复合 API (einsum / layer_norm / dropout / `matmul`) 几乎全被分解为 core aten | 6 个独立 case | 不需要为 `torch.nn.functional` / `torch.functional` 单独路径,落到 `kTensorOp` 即可 |
| HOP 与 `get_attr` 共存,且 HOP 把 GraphModule 子图作为 arg | torch.cond | v2 显式拒绝 HOP;不引入子图递归翻译 |
| 没有 `make_tuple` / `make_list` 节点,集合是 `node.args` 上的 Python 容器 | 显式 list 构造 / unbind+stack / 元组返回 | trace 直接把 args 上的 list/tuple 翻译为 `kList`,无对偶"构造 step" |

**v2 翻译器必须实施的防御性断言**(无任一条满足都应 fail-fast,报错指引用户改用
`backend="inductor"` 或回退到 v1):

1. `placeholder` 的 `val` 不是 `Tensor` 也不是 `SymInt` (例如 `SymFloat`/`SymBool`)。
2. `call_function` 的 target 是 `HigherOrderOperator`。
3. `call_method` / `call_module` 节点出现。
4. `operator.getitem(prev, i)` 的 `prev` 又是一个 `operator.getitem` 节点 (链式访问)。
5. `output` 的 args 是嵌套 tuple/list (违反 pytree 摊平假设)。

这五条断言一旦在真实 graph 上触发,说明 PyTorch 上游对 AOT graph 形态做了破坏性
变更,需要回看 §17.6.3 的翻译表是否需要扩展。**只要这些假设成立,v2 翻译器就只
处理 `placeholder / call_function(OpOverload | builtin | python_callable) / output`
这 4 类 FX 节点**,设计闭合。

#### 17.6.9 已知性能缺陷:AOT 对 Python scalar 的 functionalize 处理

在 torchbench llama 等含 RMSNorm / attention-scale 的模型上,实测 v2 trace
里出现明显异常的 op 序列(`TDC_TRACE_DEBUG=1` 抓到的真实 dump,设备 NPU):

```
[v2][N]   aten::mean.dim          → Tensor(float,[64,32,1],npu:0)
[v2][N+1] prims::convert_element_type. stack=[Tensor(double,[],cpu), 7]
[v2][N+2] prims::convert_element_type. stack=[Tensor(double,[],cpu), 6]
[v2][N+3] aten::add.Tensor        stack=[Tensor(float,[64,32,1],npu:0),
                                         Tensor(float,[],cpu),         ← !!
                                         1]
```

每个 RMSNorm + attention scaling 都会重复这一段。

**根因**

源代码层是 `x + self.eps` (其中 `self.eps = 1e-6` 是 Python float),或者
`scores / math.sqrt(head_dim)` (math.sqrt 返回 Python float)。

Dynamo / AOT 处理这条路径分三步:

1. Dynamo bytecode trace 把 `self.eps` 读为 `ConstantVariable(1e-6)`,作为
   constant 注入 graph
2. AOT functionalize 阶段要求 graph 每条边都是 Tensor (subclass tracing
   的固有约束),所以 Python scalar 被 **lift 成 0-d Tensor 作为 graph
   input** (placeholder)
3. lift 出来的 Tensor 用 `torch.tensor(1e-6)` 的默认行为构造,**dtype=float64,
   device=cpu**——Python float 的默认值

v2 在 `_build_recipe_specs` 里把这个 placeholder 走 pre-bind 路径:
`is_user_input` 检查匹配不上,落到 "module parameter / buffer" 分支
`pre_binds.append(...)`。pre-bind 的 value 就是 capture 时的 cpu 0-d
double tensor,被永久存到 `captured_tensors_` 里,**device 在 capture 时
钉死成 CPU**。

紧跟的两个 `prims::convert_element_type` 是 AOT functionalize 出来的:
- 第 1 个 (`double → 7=float64`) 实际是 no-op,AOT 出于保守加的占位
  cast
- 第 2 个 (`double → 6=float32`) 把 dtype 对齐到 compute dtype

`prims::*` 不带后端 C++ kernel,replay 时走 `CompositeImplicitAutograd`
Python decomposition (回调进 Python,转发 `aten._to_copy`,再 dispatch
一遍才到 backend kernel)。每次 ~30-100 μs Python 跳板开销。

最终的 `aten::add.Tensor(npu_tensor, cpu_scalar_tensor, 1)` 是 **跨 device
混合操作**:NPU backend kernel 检测到 RHS 是 CPU 标量,必须先 H2D 同步,
才能在设备上做加法。

**costs 累加**

LLaMA decoder block 含 2~3 个 RMSNorm + 1 个 attention-scale,每个都重复
这一套。8~12 个 transformer block 下来,**单次 replay 累积 30+ 次** "CPU
标量 + NPU tensor" 混合操作。

| 单次开销 | 来源 |
|---|---|
| ~30-100 μs | `prims::convert_element_type` (double→double, no-op) Python decomposition |
| ~30-100 μs | `prims::convert_element_type` (double→float) 同上 |
| ~50-200 μs (取决于 backend) | `aten::add.Tensor` 跨 device 同步 |

整体 latency 可能在毫秒级,跟"绕过 Dynamo 节省的 host overhead"在同一个数量级
甚至更高,完全可能让 v2 比 eager / aot_eager 还慢。

**为什么 v1 没这个问题**

v1 是 dispatcher fallback 捕获,看到的是用户 Python 代码实际 dispatch 出去
的 op。`x + 1e-6` 在 Python 层走 overload resolution,RHS 是 Python number
⇒ 直接 dispatch 到 `aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1)`。
scalar 是 `c10::IValue(Scalar)`,**纯 host-side 数字,不是 Tensor**:

```
[v1][N] aten::add.Scalar  stack=[Tensor(float,[64,32,1],npu:0), 1e-06, 1]
                                                                ^^^^^^
                                                  c10::Scalar (kLiteral)
```

无 prim、无 device、无 H2D。整套问题是 AOT functionalize 强制 Tensor-only
IR 引入的。

**为什么 inductor 没这个问题**

inductor 在 lower 阶段有完整的 decomposition + pattern rewrite,会把
`aten.add.Tensor(t, cpu_scalar_t)` 识别为 "标量+张量" 模式并 codegen 成
scalar-on-host kernel,prim 也被它的 decomposition 表展平到 aten 后再 fuse
进生成 kernel。aot_eager 和我们 v2 都不带这一层 rewrite,保留了 functionalize
后的"丑"形态。

**修复方案**

落地中的关键认知:**这些"丑形态"在源头上都是 AOT functionalize 这一步引入
的**。Python scalar 被 lift 成 Tensor、in-place 被改写为 slice_scatter +
copy_ 都是 functionalize 为了让 graph 变成纯函数才做的事。Dynamo 的原始 FX
graph 里没有这些问题(实测:Dynamo backend 直接看 gm,`x[i:j] = y` 仍是单个
`<built-in setitem>` 节点;经 AOT functionalize 后才裂成 slice + slice_scatter
+ copy_)。

按收益从大到小:

| 级别 | 改动 | 解决的问题 |
|---|---|---|
| ★ 已落地 | 推理路径(`_capture_positional` / `_capture_via_aot_wrapper`)给 AOT 传 **`disable_functionalization=True`** | **通用解** —— 所有 in-place op(`__setitem__` / `copy_` / `add_` / `index_put_` / `scatter_` / `fill_` ...) 都保持原貌,不再被 functionalize 翻成 `<functional op> + copy_(input, result)` 形态。LLaMA KV cache 的 slice_scatter 直接消失,trace op 数显著下降。 |
| ★ 已落地 | `_capture_positional` / `_capture_with_backward` 在 `trace.v2_pre_bind()` 前扫 `pre_binds`,把 device 是 cpu / dim==0 的 Tensor `.to(target_device)` (`_promote_scalar_pre_binds_to_device`) | 消除 RMSNorm / attention scale 的 CPU 标量在每次 replay 触发的 H2D 同步。 |
| ★ 已落地 | translator pre-translate FX pass(`_rewrite_prims_in_gm`)将 `prims.convert_element_type` 重写为 `aten._to_copy` | 消除 prim 的 Python decomposition 跳板;同 dtype 的 no-op 转换在 capture 时直接被折叠掉。 |
| ◯ 兜底 | `_rewrite_slice_scatter_to_inplace` FX pass | 仅在 backward 路径(`allow_grad=True`)生效 —— autograd 的 partition_fn 要求纯函数图,不能 disable_functionalization,所以那条路仍然会出现 slice_scatter。 |
| 中 | translator 检测 "Tensor + (kLiteral/kCapturedTensor 是 0-d scalar tensor)" 模式,rewrite 成 `aten.add.Scalar` + Scalar literal;同理 div / mul / sub / pow | 完全消除 0-d tensor placeholder,达到 v1 的 op 形态。device-promote 之后这个的边际收益变小,优先级降。 |
| 长期 | 在 v2 翻译器中维护"Python scalar literal 回退表":Dynamo 标记 `kCapturedConst` 的 placeholder 不走 pre-bind,直接 inline 成 Scalar IValue | 等价于把 v1 的 overload 选择能力反过来推到 v2。 |

**为什么 disable_functionalization 是真正的通用解,而 inductor 选了打补丁**

Inductor 的 `auto_functionalized_v2` 走的是"全 functionalize → 再 pattern
match 还原"的路径。原因是 inductor 后端 codegen 对纯函数 IR 有强依赖
(CSE、buffer 复用、kernel fusion 都基于"边是 SSA value"的假设)。所以
inductor 不能 disable functionalize,只能事后挨个 pattern 还原。

**v2.capture 的目标不是 codegen,只是 replay**。我们要的就是"用户写了什么
op 我就跑什么 op",in-place 保持 in-place,view 保持 view 关系。这种场景下
disable functionalize 不仅可行,而且**消除了一整类需要后续打补丁的根源问题**。

**禁用 functionalize 的边界**

| 场景 | functionalize | 原因 |
|---|---|---|
| 推理(`allow_grad=False`,wrapper True/False) | 关 | 没有反向传播需求,纯函数 IR 不是必要 |
| 训练(`allow_grad=True`) | 开 | AOTAutograd 的 partition_fn 要求纯函数图来切分 fw/bw |

backward 路径仍然保留 `_rewrite_slice_scatter_to_inplace` 作为兜底,等到训练
场景的 KV cache 真的成为热点再说(目前训练通常 batch 大、host overhead 占比
小,优先级低)。

**该缺陷的诊断方法**

设 `TDC_TRACE_DEBUG=1` 跑 v2:

| 看到 | 含义 |
|---|---|
| `slice_scatter` 出现在 trace 里(且 wrapper=False / wrapper=True 推理路径) | 不应该发生 —— disable_functionalization 没生效。检查 AOT 版本对 kwarg 的支持。 |
| `Tensor(...,[],cpu)` 出现在 `add.Tensor` / `div.Tensor` 等 op 的 stack 里 | device-promote 没覆盖到 —— 可能是 0-d 之外的 cpu tensor,或者 target device 推断失败。 |
| `prims::*` 出现在 trace 里 | `_rewrite_prims_in_gm` 没覆盖到此 prim,加规则。 |
| `aten::copy_` 紧邻 `slice` 出现 | 这是**正确的 in-place 写**,跟 eager 一致,**不要去除**。 |

可以作为接入新模型时的标准 sanity check。

#### 17.6.10 已知性能缺陷:nn.Dropout(eval) 被 AOT 展开为 aten::clone

在 timm_vision_transformer 等含大量 `nn.Dropout` 的模型上(eval 模式),实测
v2 trace 出现每个 transformer block 重复约 3 次的 `aten::clone` 序列。
`TDC_TRACE_DEBUG=1` 抓到的真实 dump(NPU,ViT-B,B=64,S=197,dim=384):

```
[v2][24] aten::addmm.            → Tensor(float,[12608, 384],npu:0)   # WO 投影
[v2][25] aten::view.             → Tensor(float,[64, 197, 384],npu:0)
[v2][26] aten::clone.    stack=[Tensor(float,[64, 197, 384],npu:0), None]   ← !!
[v2][27] aten::add.Tensor                                                # residual

[v2][33] aten::gelu.             → Tensor(float,[64, 197, 1536],npu:0)  # MLP 激活
[v2][34] aten::clone.    stack=[Tensor(float,[64, 197, 1536],npu:0), None]  ← !!
[v2][35] aten::view.             → ...                                  # FC2 输入

[v2][38] aten::view.             → Tensor(float,[64, 197, 384],npu:0)  # FC2 输出
[v2][39] aten::clone.    stack=[Tensor(float,[64, 197, 384],npu:0), None]  ← !!
[v2][40] aten::add.Tensor                                                # residual
```

ViT-B 12 个 block × 3 个 clone + 入口 pos_embed dropout = **~38 个多余 clone
step**,每个 `64*197*384*4 ≈ 19 MB`,**单次 forward 累计 ~720 MB 多余 HBM
流量**。在 NPU 这种 dispatch 不是瓶颈的设备上,完全可以让 replay 比 eager 慢
(eager 走 dropout C++ fast-path,带宽开销是另一个数量级)。

**根因**

PyTorch autograd 契约要求 `aten::dropout(input, p, train)` 在 `train=False`
/ `p=0` 时**也要返回一个新的 Tensor**(有独立 grad_fn),否则用户做
`y = dropout(x); y.add_(1)` 会悄悄改到 `x`。eager 路径下这个契约由 C++
dropout 实现内部 fast-path 兑现(一次 cheap kernel 调用,clone 与 autograd
no-op 融合);AOT trace 阶段则把"clone(input)"暴露成一个独立的 FX 节点,
我们 v2 翻译时变成独立 Step,带宽 + dispatch 都成本翻倍。

`drop_path` / `nn.Identity` / `F.dropout(... p=0.0)` 等同款问题。

**修复方案**

落地一个 FX pass `eliminate_dead_clones`(`python/v2/fx_passes.py`),在
translate_graph 之前清理掉**`memory_format=None` 且所有用户都不写入它**的
`aten::clone` 节点。判别"用户不写入"靠 op schema 的 `alias_info.is_write`
位:

```python
clone(x, memory_format=None) → x       # 当 user.schema_arg.alias_info.is_write 全为 False
clone(x, memory_format=contiguous) → keep    # 真要 materialize,eager 也会做
clone(x, memory_format=None) → keep    # 当任一 user 写它(copy_ / add_ / ...)
```

FX `output` 节点视为安全 read-only(它就是把值传给调用方,没有 mutation 语义)。
`operator.getitem` 同理。

**实测收益**(timm_vit B=4 CPU,见 prototypes/timm_vit_smoketest.py)

| 指标 | 改前 | 改后 |
|---|---|---|
| trace step 数 | 410 | 372(-38) |
| `aten::clone` step 数 | 38 | 0 |
| v2 (direct) replay vs eager | 1.13x | **1.01x** |

NPU 上 720MB 带宽减少,预期收益放大若干倍,可让 v2 (direct) 反超 eager。

**该缺陷的诊断方法**

设 `TDC_TRACE_DEBUG=1`,在 trace 中出现以下模式:

| 看到 | 含义 |
|---|---|
| `aten::clone. stack=[..., None]` 紧跟 addmm/view/gelu,后接 add/view/native_layer_norm | **dead clone** —— pass 没生效或没覆盖到此 op |
| `aten::clone. stack=[..., 0]`(或非 None 的 memory_format) | **必要的 materialize** —— eager 也做,不要删 |
| `aten::clone. stack=[..., None]` 紧跟 `aten::copy_` / `aten::add_` 之类 in-place op | **alias 写入目标** —— pass 已通过 alias_info.is_write 保留,正确 |

#### 17.6.11 已知正确性缺陷:Optional[Tensor]=None 触发 IValue::toScalar

在 timm_vision_transformer 这类使用 `F.scaled_dot_product_attention` 的模型
上,实测 v2 在 capture 阶段(AOT 的第一次 example call)直接 raise:

```
RuntimeError: IValue is not a Scalar
  ... in trace.v2_replay(...)
  ... in wrapping_cb
  ... in aot_compiled(...)
```

trace 翻译本身**全部成功**(整个 ViT 的 447 个 call_function 节点都翻译到了
trace step);失败在 C++ replay 阶段的 IValue 类型校验。

**根因**

CPU 上 `F.scaled_dot_product_attention` 被 lower 为
`aten::_scaled_dot_product_flash_attention_for_cpu`,schema:

```
_scaled_dot_product_flash_attention_for_cpu(
    Tensor query, Tensor key, Tensor value,
    float dropout_p=0.0, bool is_causal=False,
    *, Tensor? attn_mask=None, float? scale=None
) -> (Tensor output, Tensor logsumexp)
```

`attn_mask` 是 `Optional[Tensor]`,默认 None。timm 的 Attention 不传 mask,
所以 graph 里这个 slot 拿到 `lit(None)`。translator 旧版 `_compute_coercions`
的处理(简化):

```python
sa_type = schema_args[k].type
if sa_type.kind() == "OptionalType":
    sa_type = sa_type.getElementType()    # 解包成 TensorType
if sa_type.kind() == "TensorType":
    if kind == "tensor":
        out.append(NONE)
    elif kind in ("int", "float", "bool", "other"):    # ← 把 None 当成 "other"
        out.append(SCALAR_TO_TENSOR)
    ...
```

`_predict_value_kind(None)` 旧版落到 `return "other"` 兜底分支。**"other"
被错误地归入 SCALAR_TO_TENSOR**:replay 时 `apply_coercion` 的 `kScalarToTensor`
分支调 `iv.toScalar()`,对 None IValue 触发 "IValue is not a Scalar"。

任何带 `Tensor?` 参数且实际传 None 的 op 都触发,SDPA 只是首先暴露的典型。

**修复方案**

translator 两处改动:

1. `_predict_value_kind` 增加单独的 `"none"` 分类,与 "other" 区分:
   ```python
   if value is None:
       return "none"
   ```
2. `_compute_coercions` 在 schema 是 `OptionalType` 且 value kind 是 `"none"`
   时,**直接 NONE coercion 短路掉**,不再解包内层 type:
   ```python
   is_optional = sa_type.kind() == "OptionalType"
   if is_optional and kind == "none":
       out.append(NONE)
       continue
   if is_optional:
       sa_type = sa_type.getElementType()
   ...
   ```

`float?` / `int?` 之类的 Optional[Scalar] 传 None 不受影响 —— 原代码在解包后
不会进 TensorType 分支,所以是 NONE coercion;改动后行为一致。

**该缺陷的诊断方法**

| 看到 | 含义 |
|---|---|
| `RuntimeError: IValue is not a Scalar` 出现在 wrapping_cb 调用栈 | **可能是 Optional[Tensor] + None 的本 bug**;先查 trace 中是否有 `Tensor?` 参数收到 lit(None) |
| 同样错误但 trace 里没有 None 参数 | 别的 SCALAR_TO_TENSOR 误标 —— 检查 `_compute_coercions` 是否覆盖此 op 的所有 arg 类型 |

#### 17.6.12 支持矩阵:torch.library.triton_op 自定义算子

v2 与 v1 都**原生支持** `@torch.library.triton_op` 注册的 Triton kernel,
零额外适配代码。这一节记录"为什么支持是免费的"以及环境约束。

**机制**

`@torch.library.triton_op("namespace::name", mutates_args={...})` 把 Triton
kernel 包装成一个 dispatcher-可见的 `torch._ops.OpOverload`,kwarg 之外的
arity 就是用户函数的签名(`(x, y, ...) -> Tensor`)。

- **v2 路径**:AOT trace 把 triton_op call 当成单个 OpOverload 节点保留
  (不会展开成 wrap_triton + empty_like 等内部细节)。translator 走通用
  `_translate_call_function` → `aten/<custom-ns>::name` step,**与普通 aten op
  同一条路径**。
- **v1 路径**:`TESTING_ONLY_GenericMode` boxed fallback 在 op 进入实现之前
  拦截(优先级 #411,高于所有 backend key)。redispatch 后内部的
  `wrap_triton` / `empty_like` 因为 `ExcludeDispatchKeyGuard` 不再被二次记录,
  整个 triton_op 作为单个 Step 落到 trace,**完全 opaque**。

**replay 调用链(两条路径相同)**

```
trace step → op.callBoxed(stack) → dispatcher
                                      ↓
                       triton_op 实现(Python wrapper)
                                      ↓
                                 wrap_triton(grid)
                                      ↓
                                  triton kernel launch
```

triton 的 JIT 缓存 / autotune / grid lambda 都在 wrapper 内,与 v2 解耦。
`BLOCK_SIZE: tl.constexpr` 这类只在 grid 求值时用的常量不会进入 trace 的
参数列表。

**v2 与 v1 的细微差异**

| 方面 | v1 | v2 (wrapper=False) |
|---|---|---|
| capture 是否需要 GPU | **需要**(fallback redispatch 必须真启动 kernel 拿输出 TensorImpl id) | 不需要(FakeTensorMode trace,kernel 不真启动) |
| replay 是否需要 GPU | 需要 | 需要 |
| 是否需要 `@register_fake` | 不需要(走真 backend) | 需要(AOT 走 FakeTensorMode 推导输出形状/dtype) |
| 动态 shape | wrapper 内 `output.numel()` 每次 replay 重算,grid 跟随 | SymInt 经 `("S", arg_idx, dim)` 路由,grid 同样跟随 |
| BLOCK_SIZE / constexpr | 不进 trace | 不进 trace |

**测试覆盖**

`test/test_v2_triton.py` 通过 module-load 时探测 `triton.runtime.driver.active`
来 device-conditional skip:CPU-only torch build 自动 skip 全部 6 个用例,
CUDA / NPU / XPU 上启用 end-to-end real launch。覆盖:

- 单 OpOverload 节点结构断言(`test_v2_triton_opaque_in_fx_graph`)
- v2 (direct) + v2 (wrapper) end-to-end correctness
- v1 capture + replay,验证输入修改后输出跟随
- triton_op + 变更 int 参数(组合 `("I", arg_idx)` 标量路由 + opaque op step)

**已知限制**

- 必须用 `@torch.library.triton_op` 注册而**不是**直接调用 `@triton.jit` kernel。
  裸 `add_kernel[grid](...)` 不是 dispatcher 可见的 op,会在 Dynamo 启动点
  graph-break。
- 必须配合 `@triton_op.register_fake` 提供输出形状推导(v2 走 FakeTensorMode
  必需;v1 可省略)。
- v2 (wrapper=True) 在 Python 标量入参的 case 下会被入口 loud-fail 拦截
  (§17.6.11 的 sibling 问题:aot_module 无 Dynamo,scalar 会被烤死)。
  triton kernel 自身的 constexpr 不算 —— 它们在 wrapper 内不进入用户 fn
  的 example_args。

#### 17.6.13 已知翻译缺陷:AOT 嵌 Tensor 常量为 get_attr 节点

在 HuggingFace GPT2(以及多数有 KV-cache 概念的 transformer 模型)上,实测 v2
translate_graph 在第一次 example call 之前就 raise:

```
NotImplementedError: v2 does not support FX node.op='get_attr'
  (node=_tensor_constant0, target=_tensor_constant0)
```

GPT2-base 的 FX graph 里有 **24 个 `_tensor_constant<N>` get_attr 节点**
(12 层 × 2 KV cache),每个被 `aten::lift_fresh_copy` 消费后参与 `aten::cat`:

```
[189] op=get_attr        target=_tensor_constant0
[190] aten::lift_fresh_copy(_tensor_constant0)
[191] op=get_attr        target=_tensor_constant1
[192] aten::lift_fresh_copy(_tensor_constant1)
[193] aten::cat([lift_fresh_copy, transpose], -2)
```

**根因**

HuggingFace GPT2 attention 在 `past_key_values is None` 时用 `torch.empty(0)`
作为初始 KV cache,与新的 k/v 沿 seq dim 拼接。AOT trace 看到这两个 `torch.empty(0)`
是字面量 tensor —— FakeTensorMode 能正常推导出 (0,) 形状 + dtype —— 但**值是
trace 时就确定的**,不依赖 graph 输入。AOT 把它们当作 **graph 常量** 处理,
不放进 placeholder,而是把真实 tensor 挂到 `gm._tensor_constant<N>` 属性上,在
graph 节点用 `op="get_attr"` 引用。

translator 旧版 `translate_graph` 主循环对 `node.op == "get_attr"` 直接
NotImplementedError:

```python
elif node.op in ("call_method", "call_module", "get_attr"):
    raise NotImplementedError(...)
```

注释还乐观地说 "AOT graphs rarely emit these" —— **HF GPT2 是反例**,任何用到
"trace 时确定值的小 tensor 常量"的代码都会触发(空 cache 初始化、padding mask、
预计算的 sinusoidal pos_embed 等)。

**修复方案**

`_translate_get_attr` 函数 + `v2_add_constant_tensor` C++ binding:

```python
def _translate_get_attr(node, gm, trace, node_to_ref, node_to_kind):
    target_value = getattr(gm, node.target, None)
    if isinstance(target_value, torch.Tensor):
        idx = trace.v2_add_constant_tensor(target_value)
        node_to_ref[node] = _C.v2_ref_captured_tensor(idx)
        node_to_kind[node] = "tensor"
        return
    raise NotImplementedError(...)   # 非 Tensor 属性(目前不支持)
```

`v2_add_constant_tensor` = 直接 `captured_tensors_.emplace_back(t)` 并返回新 slot
索引,**不在 `placeholder_routing_` 添加条目** —— 这个 slot 是冻在常量值上的,
不会被 v2_replay 用 args 覆盖。引用类型仍是 `kCapturedTensor`,replay 路径无任何
变化。

**安全性**

- 翻译顺序:FX 节点遍历是按 placeholder → 中间 call_function / get_attr 混排 →
  output 这样进行的。`_translate_get_attr` 在第一个 get_attr 出现时(GPT2 是 idx
  189,所有 152 个 placeholder 已经处理完)首次调用,新常量插入到 slot 150 之后,
  与 placeholder slot 完全不冲突。
- 替换路径:目前只支持 `target_value` 是 Tensor 的情形;遇到 Module 片段 /
  ScriptModule fragment / Python literal 等其他类型 raise 明确错误,留给未来扩展。

**该缺陷的诊断方法**

| 看到 | 含义 |
|---|---|
| `NotImplementedError: v2 does not support FX node.op='get_attr'` | 命中本节;升级到含 `_translate_get_attr` 的 translator 即可 |
| `get_attr` 数量等于 layer × 2 (k+v cache) 或 layer × 3 (q+k+v 都缓存) | 是 KV-cache 初始化模式;典型 HF transformer |
| `get_attr` 数量 ≈ layer 数 | 是 per-layer attention mask / sinusoidal pos_embed 类常量 |
| `get_attr` 数量为奇数或杂数 | 可能是用户代码里的 `torch.tensor([...])` 字面量;无害 |

#### 17.6.14 已知正确性缺陷:SCALAR_TO_TENSOR coercion 丢 dtype

在 HF GPT2 上,即便修复了 §17.6.13 的 get_attr,v2 capture 仍然在第一次
example call 时 raise:

```
RuntimeError: Expected tensor for argument #1 'indices' to have one of
the following scalar types: Long, Int; but got torch.FloatTensor instead
(while checking arguments for embedding)
```

错误位点(embedding 的 indices 收到 Float)**与污染源相距数步**。`TDC_TRACE_DEBUG=1`
抓到的关键片段:

```
[2] aten::arange.       stack=[64, None, None, cpu, False]                  → Long[64]
[3] aten::add.Tensor    stack=[Tensor(long,[64]), Tensor(float,[])<S>T, 1]  ← !!
[4] aten::unsqueeze.    stack=[Tensor(float,[64]), 0]                        ← 已变 Float
[5] aten::embedding.    stack=[Tensor(float,[1024,768]), Tensor(float,[1,64]), ...]
                                                       ^^^^^^^^^^^^^^^^^^^
                                                       拒收非 Long indices
```

step 3 的第二个入参标了 `<S>T` 后缀(SCALAR_TO_TENSOR coercion),但 stack 显示
dtype 是 `float` —— **literal `0` 被烤成了 fp32 0-d tensor**。Long + Float 经
PyTorch 类型推导广播成 Float,后续 unsqueeze / embedding 看到的就是 Float
indices。

**根因**

源代码层是 GPT2 的位置编码路径:

```python
position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
# 等价于
position_ids = (torch.arange(seq_len) + 0).long()   # AOT 简化路径
```

AOT 看到 `arange + 0` 中的 `0` 是 Python int 字面量,但 `aten::add.Tensor` schema
的 `other` 参数是 `Tensor`(不是 `Scalar`),所以 translator 标
`SCALAR_TO_TENSOR` coercion。C++ 端 `apply_coercion`:

```cpp
case ArgCoercion::kScalarToTensor:
    if (iv.isTensor()) return iv;
    return c10::IValue(at::scalar_tensor(iv.toScalar()));   // ← 无 dtype hint
```

`at::scalar_tensor(scalar)` 无 dtype 参数时,**落到全局默认 dtype = fp32**,
不管 Scalar 自己的类型(kLong vs kDouble vs kBool ...)。`Int(0)` 因此被烤成
`Tensor(float, [], cpu)`,Long arange + Float scalar 经 PyTorch 标准类型推导广
播到 Float,后续整条链路 dtype 错误传播。

**为什么 eager 没事**

eager 走 dispatcher overload resolution,`tensor + 0` 在 Python 层会 dispatch
到 `aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1)` 而不是
`add.Tensor` —— scalar 直接以 `c10::Scalar` 留在 IValue 里,kernel 内部按
self 的 dtype 处理,不会触发任何类型转换。AOT 的 functionalize 强制 IR
是"全 Tensor 边",才把 `0` 升级为 Tensor,触发了 coercion 路径。

**修复方案**

`apply_coercion` 在 `kScalarToTensor` case 中传 Scalar 自己的 dtype 给
`scalar_tensor`:

```cpp
case ArgCoercion::kScalarToTensor: {
    if (iv.isTensor()) return iv;
    const auto s = iv.toScalar();
    return c10::IValue(
        at::scalar_tensor(s, at::TensorOptions().dtype(s.type())));
}
```

`Scalar.type()` 是 IValue 内置追踪的类型(kInt / kLong / kDouble / kBool),
能准确反映"用户原本是 Python int 还是 float"。修复后:

- `Int(0)` → `Tensor(long, [], cpu)`(Scalar.type() = kLong,Python int 在 C++
  Scalar 中默认是 kLong)
- `Float(0.5)` → `Tensor(double, [], cpu)`(后续若需要再走类型推导降到 fp32)

PyTorch 自己的类型推导规则现在能正确工作:
- `Long + Long(0d) → Long` ✓ (arange + 0 保持 Long)
- `Float + Long(0d) → Float` ✓ (PyTorch 让 Float 赢类型推导)

**该缺陷的诊断方法**

| 看到 | 含义 |
|---|---|
| `TDC_TRACE_DEBUG=1` dump 中 `<S>T` 后跟 `Tensor(float,[]...)`,但源代码是 int 字面量 | 命中本节 |
| embedding / scatter / index 等 Long-only op 在 v2 replay 报 "got FloatTensor" 错,但 trace 起始几步看着正常 | 大概率 dtype 污染从某个 SCALAR_TO_TENSOR 开始;翻几步 stack 找 `<S>T` 标记 |
| 错误信息说的"参数 N"不是真正的污染点 | dtype 错误的传播链可能有 5~10 步距离,从 op 出错位置反向找 |

**与 §17.6.9 的关系**

§17.6.9 的"Python scalar 被 lift 成 0-d Tensor"问题是同一个根源 ——
functionalize 让 scalar 进入 IR。那一节的修复是 device-promote(把 cpu 0-d
tensor 在 capture 时挪到目标 device,消除每次 replay 的 H2D 同步);本节的修复
是 dtype-preserve(让 0-d tensor 的 dtype 准确反映 Scalar 来源)。两个修复**正交**,
都已落地。
