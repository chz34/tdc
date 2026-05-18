// v2 replay path and builtin PyCall dispatch.
//
// v2 traces are produced by the Python translator walking an AOT FX
// graph (DESIGN.md §17.6.3). They contain a mix of kTensorOp steps
// (regular aten / prims / custom ops, dispatched via callBoxed) and
// kPyCall steps (operator.* / torch.sym_* / getitem, dispatched here
// via the BuiltinKind switch with a py::object fallback).
//
// All five StepInputRef kinds resolve uniformly here, including
// kCapturedInt and kList which v1 never produces.

#include "capture_context.h"

#include <ATen/Functions.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <algorithm>

namespace py = pybind11;

namespace tdc {

namespace {

// Floor division matching Python's // semantics (vs C++ truncation
// toward zero for negative). PyTorch shapes are non-negative so this
// rarely matters, but we want bit-exact Python compatibility.
int64_t py_floordiv(int64_t a, int64_t b) {
    TORCH_CHECK(b != 0, "integer division or modulo by zero");
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
}

int64_t py_mod(int64_t a, int64_t b) {
    TORCH_CHECK(b != 0, "integer modulo by zero");
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

c10::IValue invoke_builtin(BuiltinKind kind,
                            const std::vector<c10::IValue>& args) {
    auto get_int = [&](size_t i) { return args[i].toInt(); };
    switch (kind) {
        case BuiltinKind::kFloorDiv:
            return c10::IValue(py_floordiv(get_int(0), get_int(1)));
        case BuiltinKind::kTrueDiv:
            return c10::IValue(static_cast<double>(get_int(0)) / get_int(1));
        case BuiltinKind::kAdd:
            return c10::IValue(get_int(0) + get_int(1));
        case BuiltinKind::kSub:
            return c10::IValue(get_int(0) - get_int(1));
        case BuiltinKind::kMul:
            return c10::IValue(get_int(0) * get_int(1));
        case BuiltinKind::kMod:
            return c10::IValue(py_mod(get_int(0), get_int(1)));
        case BuiltinKind::kNeg:
            return c10::IValue(-get_int(0));
        case BuiltinKind::kGetItem: {
            // args[0] is the container (tuple/list); args[1] is the index.
            const auto& container = args[0];
            int64_t idx = args[1].toInt();
            if (container.isTuple()) {
                const auto& elems = container.toTuple()->elements();
                TORCH_CHECK(idx >= 0 && static_cast<size_t>(idx) < elems.size(),
                            "getitem index ", idx, " out of range for tuple of size ", elems.size());
                return elems[idx];
            }
            if (container.isList()) {
                return container.toList().get(idx);
            }
            if (container.isTensorList()) {
                auto list = container.toTensorList();
                return c10::IValue(list[idx]);
            }
            TORCH_CHECK(false,
                "getitem: unsupported container IValue type: ", container.tagKind());
        }
        case BuiltinKind::kEq: return c10::IValue(get_int(0) == get_int(1));
        case BuiltinKind::kLt: return c10::IValue(get_int(0) <  get_int(1));
        case BuiltinKind::kLe: return c10::IValue(get_int(0) <= get_int(1));
        case BuiltinKind::kGt: return c10::IValue(get_int(0) >  get_int(1));
        case BuiltinKind::kGe: return c10::IValue(get_int(0) >= get_int(1));
        case BuiltinKind::kNe: return c10::IValue(get_int(0) != get_int(1));
        case BuiltinKind::kSymMax:
            return c10::IValue(std::max(get_int(0), get_int(1)));
        case BuiltinKind::kSymMin:
            return c10::IValue(std::min(get_int(0), get_int(1)));
        case BuiltinKind::kSymInt:
            return c10::IValue(args[0].toInt());
        case BuiltinKind::kSymFloat:
            return c10::IValue(args[0].toDouble());
        case BuiltinKind::kPyFallback:
        case BuiltinKind::kNumBuiltinKinds:
            TORCH_CHECK(false, "invoke_builtin called with non-builtin kind");
    }
    TORCH_CHECK(false, "unreachable: unhandled BuiltinKind ", static_cast<int>(kind));
}

c10::IValue invoke_py_fallback(void* py_fn_handle,
                                const std::vector<c10::IValue>& args) {
    TORCH_CHECK(py_fn_handle != nullptr,
                "kPyFallback step has null py_fn_handle");
    py::gil_scoped_acquire gil;
    auto fn = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject*>(py_fn_handle));
    // Convert IValue args to Python tuple. For the safe cases v2 hits
    // (rare fallback) the ints / floats / Tensors convert automatically.
    py::tuple py_args(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        py_args[i] = torch::jit::toPyObject(args[i]);
    }
    auto py_result = fn(*py_args);
    return torch::jit::toIValue(py_result, c10::AnyType::get());
}

// Recursively resolve a StepInputRef to its runtime IValue.
c10::IValue resolve_ref(
    const StepInputRef& ref,
    const std::vector<std::vector<c10::IValue>>& outputs,
    const std::vector<at::Tensor>& captured_tensors,
    const std::vector<int64_t>& captured_ints
) {
    switch (ref.kind) {
        case StepInputRef::Kind::kCapturedTensor:
            return c10::IValue(captured_tensors[ref.captured_idx]);
        case StepInputRef::Kind::kCapturedInt:
            return c10::IValue(captured_ints[ref.captured_idx]);
        case StepInputRef::Kind::kPrevStepOutput:
            return outputs[ref.prev_step][ref.prev_slot];
        case StepInputRef::Kind::kLiteral:
            return ref.literal;
        case StepInputRef::Kind::kList: {
            c10::List<c10::IValue> result(c10::AnyType::get());
            result.reserve(ref.list_elements.size());
            for (const auto& sub : ref.list_elements) {
                result.push_back(resolve_ref(sub, outputs, captured_tensors, captured_ints));
            }
            return c10::IValue(result);
        }
    }
    TORCH_CHECK(false, "unreachable: unknown ref kind");
}

}  // anonymous namespace

std::vector<c10::IValue> Trace::replay_v2(
    const std::vector<c10::IValue>& args
) {
    // Same TLS guards as v1 replay — we never want our capture key
    // re-entering at replay time, and autograd should stay below.
    c10::impl::ExcludeDispatchKeyGuard exclude_capture{
        c10::DispatchKeySet(kCaptureKey)};
    at::AutoDispatchBelowAutograd no_autograd_guard;

    TORCH_CHECK(args.size() == placeholder_routing_.size(),
        "replay_v2: expected ", placeholder_routing_.size(),
        " positional args, got ", args.size());

    // Route positional args into captured_tensors / captured_ints.
    std::vector<at::Tensor> captured_tensors(n_captured_tensors_);
    std::vector<int64_t> captured_ints(n_captured_ints_);
    for (size_t i = 0; i < args.size(); ++i) {
        const auto& [target, idx] = placeholder_routing_[i];
        if (target == PlaceholderTarget::kTensor) {
            TORCH_CHECK(args[i].isTensor(),
                "replay_v2 arg ", i, " expected Tensor, got ", args[i].tagKind());
            captured_tensors[idx] = args[i].toTensor();
        } else {
            TORCH_CHECK(args[i].isInt(),
                "replay_v2 arg ", i, " expected int, got ", args[i].tagKind());
            captured_ints[idx] = args[i].toInt();
        }
    }

    std::vector<std::vector<c10::IValue>> outputs(steps_.size());

    for (size_t i = 0; i < steps_.size(); ++i) {
        const auto& step = steps_[i];

        std::vector<c10::IValue> resolved;
        resolved.reserve(step.inputs.size());
        for (const auto& ref : step.inputs) {
            resolved.push_back(resolve_ref(ref, outputs, captured_tensors, captured_ints));
        }

        if (step.step_kind == Step::Kind::kPyCall) {
            c10::IValue result = (step.builtin_kind == BuiltinKind::kPyFallback)
                ? invoke_py_fallback(step.py_fn_handle, resolved)
                : invoke_builtin(step.builtin_kind, resolved);
            outputs[i] = {std::move(result)};
        } else {
            // kTensorOp — push resolved IValues to stack and callBoxed.
            // Schema-aware coercion: aten.X.Tensor variants accept Python
            // scalars in eager (via OpOverload.__call__'s C path) but
            // callBoxed is strict, so we wrap Scalar->0-d Tensor for slots
            // whose schema type is Tensor. This mirrors what PyTorch's
            // Python __call__ path does internally.
            TORCH_INTERNAL_ASSERT(step.op.has_value(),
                "kTensorOp step missing OperatorHandle");
            const auto& schema_args = step.op->schema().arguments();
            torch::jit::Stack stack;
            stack.reserve(resolved.size());
            for (size_t k = 0; k < resolved.size(); ++k) {
                c10::IValue iv = std::move(resolved[k]);
                if (k < schema_args.size()) {
                    const auto& schema_type = schema_args[k].type();
                    auto kind = schema_type->kind();
                    // Scalar -> 0-d Tensor wrap (e.g. aten.mul.Tensor(x, 2.0)).
                    if (kind == c10::TypeKind::TensorType
                        && !iv.isTensor()
                        && (iv.isInt() || iv.isDouble() || iv.isBool())) {
                        iv = c10::IValue(at::scalar_tensor(iv.toScalar()));
                    }
                    // GenericList -> typed list per schema (size args, cat tensors).
                    else if (kind == c10::TypeKind::ListType && iv.isList()) {
                        const auto* lt = schema_type->castRaw<c10::ListType>();
                        auto elem_kind = lt->getElementType()->kind();
                        const auto& generic = iv.toList();
                        if (elem_kind == c10::TypeKind::SymIntType
                            || elem_kind == c10::TypeKind::IntType) {
                            c10::List<int64_t> ints;
                            ints.reserve(generic.size());
                            for (const auto& e : generic) {
                                ints.push_back(c10::IValue(e).toInt());
                            }
                            iv = c10::IValue(std::move(ints));
                        } else if (elem_kind == c10::TypeKind::TensorType) {
                            c10::List<at::Tensor> tensors;
                            tensors.reserve(generic.size());
                            for (const auto& e : generic) {
                                tensors.push_back(c10::IValue(e).toTensor());
                            }
                            iv = c10::IValue(std::move(tensors));
                        }
                        // Other element types fall through with the GenericList.
                    }
                }
                stack.emplace_back(std::move(iv));
            }
            step.op->callBoxed(&stack);
            // Multi-output schemas leave N values on the stack; flat slot model
            // wraps them as a tuple in slot 0 so downstream getitem can extract.
            if (stack.size() == 1) {
                outputs[i] = {std::move(stack[0])};
            } else {
                outputs[i] = {c10::ivalue::Tuple::create(std::move(stack))};
            }
        }
    }

    // Materialize the trace's declared outputs.
    std::vector<c10::IValue> result;
    result.reserve(outputs_.size());
    for (const auto& ref : outputs_) {
        result.push_back(resolve_ref(ref, outputs, captured_tensors, captured_ints));
    }
    return result;
}

}  // namespace tdc
