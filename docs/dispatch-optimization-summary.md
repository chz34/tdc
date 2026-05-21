# 算子调度优化方案综述

五种方案按缓存层次从底层（内核）到上层（Dynamo/AOT）排列。越靠近底层，单次调用收益越高但灵活性越低；越靠近上层，灵活性越高但额外开销也越大。

---

## 方案 1：ACL 内核级缓存（EXEC_NPU_CMD）

**做法**：在 torch_npu 的 `EXEC_NPU_CMD` 宏层面缓存已构造的 ACL 算子描述符（op desc / stream / context），绕过算子参数组装和 ACL 内核下发流程。

**收益**：最高。跳过整个算子适配栈（aten wrapper → aclnn 参数封装 → ACL 内核下发），仅剩内核执行时间。

**限制**：

- **完全不能处理动态 shape**。ACL 层算子描述符与 tensor 的 sizes/strides/data_ptr 强绑定，任何 shape 变化都需要重建描述符。
- 不能处理反向。反向图的算子序列和 shape 依赖前向结果，缓存的描述符无法适配。
- 内存膨胀。每个 op + shape 组合缓存一个描述符，shape 空间大时内存不可控。

---

## 方案 2：Aten Wrapper 代码生成层缓存

**做法**：在 torch_npu 的 aten wrapper 代码生成（`gen_xxx_op`）中增加缓存，对已生成的 C++ wrapper 调用路径做参数化缓存，允许有限的 metadata 变化（如 sizes 在特定范围内缓存或分桶）。

**收益**：较高。绕过代码生成产物的重复构造，减少 CPU 侧开销。

**限制**：

- **有限的动态 shape**。只能处理预定义的 shape 范围或分桶策略，不能处理任意 shape 变化。shape 超出缓存范围需要重建。
- 不能处理反向。反向的 aten wrapper 由 autograd 引擎动态触发，shape 依赖前向中间结果，无法在代码生成时静态缓存。
- 缓存管理复杂。需要分桶策略 + 淘汰机制，工程成本高。

---

## 方案 3：tdc v1 —— Dispatcher 级捕获/重放

**做法**：在 PyTorch dispatcher 层注册 `TESTING_ONLY_GenericMode` boxed fallback（优先级 #3，高于 AutogradFunctionality）。捕获期间记录每个 aten op 的调用（op handle + input refs），replay 时直接 `op.callBoxed(stack)` 重放。Tensor metadata 从捕获的 Tensor 对象实时读取，resize / in-place mutation 自动传播。

核心数据路径：

```
capture:  用户代码 → dispatcher → fallback(分类输入 → 执行op → 记录Step) → kernel
replay:  遍历 Step → 从 Tensor 对象读取当前 metadata → callBoxed → kernel
```

**收益**：中等。绕过 dispatcher 的 keyset 提取、lookup、alias 展开、redispatch 链，直接调 kernel。CPU 小 op 场景收益明显。

**动态 shape 支持**：有限支持。

| 能处理 | 不能处理 |
|--------|----------|
| in-place resize | shape-derived literal（`x.view(x.shape[0]//2, ...)` 中的 `//2` 在 Python 层已求值，dispatcher 看到的是具体 int） |
| in-place mutation | backward 中 reduction op 的 `expand`（含 shape literal） |
| dim-index ops（view/transpose/permute 沿维度变化） | `as_strided` / 带 stride 推导的 op |

**反向支持**：实验性 `allow_grad=True`，需要 warmup（让 AccumulateGrad 走上 dispatched `add_` 路径）。不使用 `torch.compile`，是纯 dispatcher 层方案。

**核心限制**：shape-derived literal 重入错误。dispatcher 在 Python 之后执行——Python 层已把 shape 算术求值成具体 int，dispatcher 无法恢复其与 `x.shape` 的关联。

---

## 方案 4：tdc v2 —— Dynamo + AOTAutograd 后端

**做法**：`torch.compile(backend=aot_autograd(fw_compiler=tdc_backend))` 跑一次获取 AOT FX graph。Python translator 将 FX graph 翻译为 C++ Trace（kTensorOp + kPyCall steps + 预计算的 ArgCoercion 表）。后续调用直接走 C++ replay 引擎，**绕过 Dynamo 和 AOTAutograd**。

核心数据路径：

```
capture:  torch.compile(dynamic=True) → Dynamo → AOTAutograd → FX graph → translator → C++ Trace
replay:  用户 args → flat_recipe(提取 Tensor/size) → C++ Trace::replay_v2 → 返回 IValue 列表
```

**关键差异**：Dynamo 做了符号化追踪，shape-derived literal 被表达为 `operator.floordiv(sym_size_0, 2)` 这种 kPyCall step，replay 时从当前输入 Tensor 的 size 重新计算 → **完整动态 shape 支持**。

**收益**：中等。纯 C++ replay 引擎，无 Python loop 开销。对 shape-polymorphic 推理和训练循环均可工作。

**反向支持**：`allow_grad=True` 自动捕获 fw + bw 双图，封装为 `torch.autograd.Function`。`loss.backward()` 驱动 bw trace replay。

**核心限制**：AOTAutograd 做 functionalization + decomposition，所有 in-place op 被拆解为 functional op + copy_。**计算退化为纯函数，副作用丢失**。

- side effect（如 `out=` buffer 写入、KV cache 更新）在 trace 中不体现
- replay 返回的是"函数返回值"，跟用户原本通过 side effect 观察结果的方式不一致
- nn.Module 参数权重在捕获时 snapshot 冻结
- HigherOrderOperator（torch.cond, torch.while_loop）不支持
- 捕获一次的成本较高（Dynamo trace + AOTAutograd compile + FX graph translation）

### 方案 4 对控制流子图（HOP）的讨论

AOTAutograd 在处理含数据依赖控制流的用户代码时，不会将控制流完全展开为线性 op 序列，而是在 graph 中保留 **HigherOrderOperator (HOP)** 节点（如 `torch.ops.higher_order.cond`、`while_loop`、`scan`、`invoke_subgraph`），每个 HOP 节点伴随 `get_attr` 节点引用其分支子图（sub-GraphModule）：

```
# 用户代码
def fn(x):
    return torch.cond(x.sum() > 0,
                      lambda x: x.sin(),    # true_fn → sub-GraphModule A
                      lambda x: x.cos(),    # false_fn → sub-GraphModule B
                      (x,))

# AOT graph 形态
#   %cond_result = call_function[higher_order.cond](
#       %predicate,                          # true/false 判断条件
#       %subgraph_true,    # get_attr → GraphModule(true_fn)
#       %subgraph_false,   # get_attr → GraphModule(false_fn)
#       %operands                              # 子图输入
#   )
```

**当前 v2 策略：显式拒绝，fail-fast 报错**。原因：

1. **子图递归翻译的工程代价**。每个子 GraphModule 需要独立的 translator → C++ Trace 翻译，且子图间可能存在嵌套（cond 内部再有 cond）。这要求：
   - 翻译器支持递归遍历子 GraphModule
   - 新增 `kSubGraph` step kind，携带子 Trace 的引用
   - replay 引擎在运行时根据 predicate 值选择执行 true/false 子 Trace
   - `get_attr` 节点解析为子 GraphModule 引用而非 Tensor 值
2. **性能收益不明确**。HOP 出现的场景（控制流密集型模型）通常 kernel 耗时本身就占比大，host 调度开销相对不显著。且子 Trace 的切换本身引入了分支判断开销。
3. **inducer 已是成熟方案**。对于控制流密集型 workload，`torch.compile(backend="inductor")` 已做了充分优化（包括子图内联、kernel fusion），v2 无需在此类场景中竞争。

**如果未来需要支持 HOP，扩展路径**：

```
翻译阶段:
  graph traversal → 遇到 HOP 节点
    → 递归遍历 true_fn / false_fn 子 GraphModule
    → 各自生成独立的 C++ Trace (sub_trace_true, sub_trace_false)
    → emit kSubGraph step:
        - 持有 sub_trace_true / sub_trace_false 的引用
        - 持有 predicate 的 StepInputRef
        - 持有 operands 的 StepInputRef list

replay 阶段:
  kSubGraph step:
    → resolve predicate (0 or 1)
    → 选择对应子 Trace
    → 将 operands 推入子 Trace 的 placeholder slots
    → 执行子 Trace::replay_v2(operands)
    → 将子 Trace 的 outputs 作为本 step 的输出
```

预估增量约 300-400 行 C++ + 200 行 Python，但基于前述收益分析，当前优先级较低。

---

## 方案 5：AOT + 副作用包装（完全依赖 Dynamo Wrapper）

**做法**：在方案 4 的基础上，不直接返回纯函数结果，而是通过 Dynamo 和 AOTAutograd 的 wrapper 层将用户代码中的所有副作用（in-place write、`out=` buffer、`.grad` update）也编译为 trace 输出的一部分，在 replay 后由 wrapper 回写到用户持有的 Tensor 对象中。

**收益**：

- 语义完全一致。用户代码的副作用行为被保留，replay 后的 buffer 状态与 eager 一致。
- 对用户透明，无需修改代码。

**限制**：

- **额外开销大，收益有限**。wrapper 层的副作用捕获 + 回写引入了 Python ↔ C++ 的多次往返和额外的 copy_/set_ 操作。kernel 耗时占比大时 wrapper 开销抵消 dispatcher 节省；小 op 场景 wrapper 开销可能超过原始 dispatcher 开销，导致负收益。
- 工程复杂度高。需要深度集成 Dynamo 的 `SideEffect` 机制，与 PyTorch 版本强耦合。
- 反向图一致性维护更难。backward 中 `.grad` accumulation、`save_for_backward` 等机制需要 wrapper 精确模拟 autograd 引擎行为。

---

## 五种方案对比总览

| 维度 | 1. ACL 内核缓存 | 2. wrapper 缓存 | 3. tdc v1 | 4. tdc v2 | 5. AOT + 副作用 |
|---|---|---|---|---|---|
| **缓存/优化层次** | ACL 驱动层 | C++ wrapper 层 | Dispatcher 层 | Dynamo/AOT 层 | Dynamo/AOT + wrapper |
| **host 开销削减** | 几乎全部 | 大部分 | dispatcher 全跳过 | Dynamo + AOT 全跳过 | Dynamo + AOT 全跳过 |
| **完整动态 shape** | 不支持 | 有限（分桶） | 有限（shape literal 不可） | 完整支持（SymInt） | 完整支持（SymInt） |
| **反向支持** | 不支持 | 不支持 | 实验性（需 warmup） | 双图捕获 | 双图 + wrapper |
| **控制流（HOP）** | — | — | — | 不支持（fail-fast） | 不支持（fail-fast） |
| **副作用保持** | — | — | 核心设计 | 不支持（纯函数） | 通过 wrapper 保持 |
| **用户代码改动** | 无 | 无 | 需遵循 capture 范式 | 需接受纯函数语义 | 无 |
| **额外开销** | 缓存查找 | 缓存管理 | replay 时 stack 重建 | 编译一次 + replay | 编译 + wrapper 回写 |
| **整体收益** | 最高 | 高 | 中 | 中 | 低，可能负收益 |
| **工程复杂度** | 低 | 中 | 中 | 中高 | 高 |
| **与 PyTorch 耦合** | 仅在 torch_npu 内 | 仅在 torch_npu 内 | out-of-tree 扩展 | out-of-tree 扩展 | 深度依赖 Dynamo |

## 各方案适用场景

```
方案 1 (ACL 缓存)
  适用：固定 shape 的推理，如静态 batch size 的图像分类
  不适用：任何有 shape 变化的场景

方案 2 (wrapper 缓存)
  适用：有限 shape 变化范围的推理，通过分桶覆盖常见 shape
  不适用：训练、shape 高度不规则的推理

方案 3 (tdc v1)
  适用：LLM decode (batch=1)、KV cache 写、以副作用为核心的推理
  不适用：有 shape-derived literal 的模型（多数 view/reshape 场景）

方案 4 (tdc v2)
  适用：训练 loop、纯函数式推理、需要完整动态 shape 的场景
  不适用：副作用密集的 serving 场景（KV cache 写等）

方案 5 (AOT + 副作用)
  适用：追求零用户改动的通用场景（理想状态）
  实际：工程成本与性能收益不匹配，当前阶段不推荐
```

## 关键 Trade-off

**灵活性 vs 开销**：越靠近底层（方案 1），单次调用节省越多，但灵活性和场景覆盖越窄。越靠近上层（方案 5），场景覆盖越广，但每次调用的额外开销累积起来可能抵消甚至超过收益。

**当前定位**：tdc v1/v2 处于中间地带——
- v1 适合副作用密集型场景（LLM serving），以 dispatcher 开销削减为核心价值
- v2 适合函数式推理/训练场景，以完整动态 shape 支持为核心价值
- 两者共享同一套 C++ Trace 数据结构和 replay 引擎