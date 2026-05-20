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

// Apply the (translation-time) coercion tag to an IValue. Called per
// arg in the kTensorOp push loop; replaces what used to be a chain of
// schema().arguments()[k].type()->kind() introspection on every replay.
inline c10::IValue apply_coercion(c10::IValue iv, ArgCoercion tag) {
    switch (tag) {
        case ArgCoercion::kNone:
            return iv;
        case ArgCoercion::kScalarToTensor:
            if (iv.isTensor()) return iv;
            return c10::IValue(at::scalar_tensor(iv.toScalar()));
        case ArgCoercion::kListToIntList: {
            const auto& generic = iv.toList();
            c10::List<int64_t> ints;
            ints.reserve(generic.size());
            for (const auto& e : generic) {
                ints.push_back(c10::IValue(e).toInt());
            }
            return c10::IValue(std::move(ints));
        }
        case ArgCoercion::kListToTensorList: {
            const auto& generic = iv.toList();
            c10::List<at::Tensor> tensors;
            tensors.reserve(generic.size());
            for (const auto& e : generic) {
                tensors.push_back(c10::IValue(e).toTensor());
            }
            return c10::IValue(std::move(tensors));
        }
    }
    return iv;
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
    // Single stack reused across all steps (opt 1). For kPyCall steps
    // we re-use it as a positional-arg buffer; for kTensorOp steps it
    // doubles as the callBoxed argument stack.
    torch::jit::Stack stack;
    stack.reserve(16);

    for (size_t i = 0; i < steps_.size(); ++i) {
        const auto& step = steps_[i];
        stack.clear();

        if (step.step_kind == Step::Kind::kPyCall) {
            // PyCall doesn't need coercion — operator.* / torch.sym_*
            // builtins take raw IValues and return raw IValues.
            for (const auto& ref : step.inputs) {
                stack.emplace_back(
                    resolve_ref(ref, outputs, captured_tensors, captured_ints));
            }
            c10::IValue result = (step.builtin_kind == BuiltinKind::kPyFallback)
                ? invoke_py_fallback(step.py_fn_handle, stack)
                : invoke_builtin(step.builtin_kind, stack);
            outputs[i] = {std::move(result)};
            continue;
        }

        // kTensorOp — opt 2: fold resolve + coerce + push into a single
        // pass instead of building an intermediate `resolved` vector.
        // Opt 3: consult precomputed coercions[k] instead of querying
        // schema().arguments()[k].type()->kind() on every replay.
        TORCH_INTERNAL_ASSERT(step.op.has_value(),
            "kTensorOp step missing OperatorHandle");
        const bool has_coercions = !step.coercions.empty();
        TORCH_INTERNAL_ASSERT(
            !has_coercions || step.coercions.size() == step.inputs.size(),
            "coercions vector must match inputs in length");

        for (size_t k = 0; k < step.inputs.size(); ++k) {
            c10::IValue iv = resolve_ref(
                step.inputs[k], outputs, captured_tensors, captured_ints);
            if (has_coercions) {
                stack.emplace_back(apply_coercion(std::move(iv), step.coercions[k]));
            } else {
                stack.emplace_back(std::move(iv));
            }
        }

        step.op->callBoxed(&stack);

        // Mirror v1's flat slot layout: outputs[i] holds N IValues, one
        // per schema return. Downstream refs use kPrevStepOutput(step,
        // slot) to address individual returns directly — no per-call
        // Tuple allocation needed (the translator folds
        // operator.getitem on multi-output OpOverloads into slot
        // indices, removing the corresponding PyCall steps as well).
        // Move semantics so each IValue is transferred not copied; the
        // stack ends up in a valid empty-ish state and gets clear()'d
        // on the next loop iteration.
        outputs[i].assign(
            std::make_move_iterator(stack.begin()),
            std::make_move_iterator(stack.end()));
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
