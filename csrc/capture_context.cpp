#include "capture_context.h"

#include <ATen/core/Formatting.h>     // operator<< for at::DeprecatedTypeProperties etc.
#include <ATen/core/ivalue.h>         // operator<< for c10::IValue
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace tdc {

// ---------------- StepInputRef ----------------

StepInputRef StepInputRef::CapturedTensor(size_t idx, bool is_out) {
    StepInputRef r;
    r.kind = Kind::kCapturedTensor;
    r.captured_idx = idx;
    r.is_out = is_out;
    return r;
}

StepInputRef StepInputRef::PrevStepOutput(size_t step, size_t slot, bool is_out) {
    StepInputRef r;
    r.kind = Kind::kPrevStepOutput;
    r.prev_step = step;
    r.prev_slot = slot;
    r.is_out = is_out;
    return r;
}

StepInputRef StepInputRef::PrevStepListElement(
    size_t step, size_t slot, int sub_slot, bool is_out) {
    StepInputRef r;
    r.kind = Kind::kPrevStepOutput;
    r.prev_step = step;
    r.prev_slot = slot;
    r.prev_list_sub_slot = sub_slot;
    r.is_out = is_out;
    return r;
}

StepInputRef StepInputRef::Literal(c10::IValue v) {
    StepInputRef r;
    r.kind = Kind::kLiteral;
    r.literal = std::move(v);
    return r;
}

StepInputRef StepInputRef::CapturedInt(size_t idx) {
    StepInputRef r;
    r.kind = Kind::kCapturedInt;
    r.captured_idx = idx;
    return r;
}

StepInputRef StepInputRef::List(std::vector<StepInputRef> elements) {
    StepInputRef r;
    r.kind = Kind::kList;
    r.list_elements = std::move(elements);
    return r;
}

// ---------------- Step ----------------

Step::Step(c10::OperatorHandle h,
           c10::DispatchKey dk,
           std::vector<StepInputRef> ins,
           size_t n_out,
           std::string name)
    : op(std::move(h)),
      target_dk(dk),
      inputs(std::move(ins)),
      n_outputs(n_out),
      op_name(std::move(name)) {}

// ---------------- Trace::dump ----------------

std::string Trace::dump() const {
    std::ostringstream os;
    os << "Trace(" << steps_.size() << " ops, "
       << captured_tensors_.size() << " captured tensors)\n";
    for (auto i : c10::irange(steps_.size())) {
        const auto& step = steps_[i];
        os << "  [" << i << "] " << step.op_name
           << "  dk=" << step.target_dk
           << "  n_out=" << step.n_outputs
           << "  inputs=[";
        for (auto j : c10::irange(step.inputs.size())) {
            if (j) os << ", ";
            const auto& ref = step.inputs[j];
            switch (ref.kind) {
                case StepInputRef::Kind::kCapturedTensor:
                    os << "ext#" << ref.captured_idx;
                    break;
                case StepInputRef::Kind::kCapturedInt:
                    os << "int#" << ref.captured_idx;
                    break;
                case StepInputRef::Kind::kPrevStepOutput:
                    os << "step" << ref.prev_step << ":" << ref.prev_slot;
                    if (ref.prev_list_sub_slot >= 0) {
                        os << "[" << ref.prev_list_sub_slot << "]";
                    }
                    break;
                case StepInputRef::Kind::kLiteral:
                    os << "lit(" << ref.literal.tagKind() << ")";
                    break;
                case StepInputRef::Kind::kList:
                    os << "list[" << ref.list_elements.size() << "]";
                    break;
            }
        }
        os << "]\n";
    }
    return os.str();
}

// ---------------- debug_dump_callBoxed ----------------
//
// One-line summary of "what is about to be sent through callBoxed" so v1
// replay() and v2 replay_v2() can be diff'd side by side. Off unless
// TDC_TRACE_DEBUG=1 (env var read once and cached).
//
// Formatting policy: we lean on torch's existing operator<< overloads
// wherever possible -- IValue::operator<<, Tensor::sizes(), Tensor::dtype(),
// Tensor::device() all already have stream operators in
// ATen/core/Formatting.h and ATen/core/ivalue.h. The one place we DON'T
// just stream the IValue verbatim is for Tensors, because IValue's default
// formatting dumps the entire tensor's data -- way too noisy for a
// per-step trace. For tensors we emit a compact Tensor(dtype,sizes,device)
// summary using the same accessors print_handler would have used.

namespace {

const char* coercion_tag(ArgCoercion c) {
    switch (c) {
        case ArgCoercion::kNone:                      return "N";
        case ArgCoercion::kScalarToTensor:            return "S>T";
        case ArgCoercion::kListToIntList:             return "L>I";
        case ArgCoercion::kListToTensorList:          return "L>T";
        case ArgCoercion::kListToOptionalTensorList:  return "L>T?";
        case ArgCoercion::kListToBoolList:            return "L>B";
    }
    return "?";
}

bool debug_enabled() {
    static std::atomic<int> cached{-1};
    int v = cached.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* env = std::getenv("TDC_TRACE_DEBUG");
        v = (env != nullptr && env[0] == '1') ? 1 : 0;
        cached.store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

void format_ivalue(std::ostream& os, const c10::IValue& iv) {
    // Tensor: compact summary (avoid dumping full data via IValue<<).
    if (iv.isTensor()) {
        const auto& t = iv.toTensor();
        if (!t.defined()) {
            os << "Tensor(undefined)";
            return;
        }
        os << "Tensor(" << t.dtype() << "," << t.sizes()
           << "," << t.device() << ")";
        return;
    }
    if (iv.isTensorList()) {
        const auto lst = iv.toTensorList();
        os << "TensorList[" << lst.size() << "]";
        return;
    }
    // Everything else (Int, Double, Bool, IntList, GenericList, None,
    // String, Scalar, ...) goes through IValue::operator<< -- compact
    // enough for primitives and small lists.
    os << iv;
}

}  // namespace

void debug_dump_callBoxed(
    const char* mode,
    size_t step_idx,
    const std::string& op_name,
    c10::DispatchKey target_dk,
    const torch::jit::Stack& stack,
    const std::vector<ArgCoercion>* coercions) {
    if (!debug_enabled()) return;
    std::ostringstream os;
    os << "[" << mode << "][" << step_idx << "] " << op_name;
    if (target_dk != c10::DispatchKey::Undefined) {
        os << " dk=" << target_dk;
    }
    os << " stack=[";
    for (size_t k = 0; k < stack.size(); ++k) {
        if (k) os << ", ";
        format_ivalue(os, stack[k]);
        if (coercions && k < coercions->size()
            && (*coercions)[k] != ArgCoercion::kNone) {
            os << "<" << coercion_tag((*coercions)[k]);
        }
    }
    os << "]";
    std::cerr << os.str() << "\n";
}


// ---------------- Trace::v2_pre_bind ----------------

void Trace::v2_pre_bind(size_t arg_idx, c10::IValue value) {
    TORCH_CHECK(arg_idx < placeholder_routing_.size(),
        "v2_pre_bind: arg_idx ", arg_idx,
        " out of range (placeholders=", placeholder_routing_.size(), ")");
    const auto& [target, slot] = placeholder_routing_[arg_idx];
    if (target == PlaceholderTarget::kTensor) {
        TORCH_CHECK(value.isTensor(),
            "v2_pre_bind: arg ", arg_idx, " expected Tensor, got ", value.tagKind());
        captured_tensors_[slot] = value.toTensor();
    } else {
        TORCH_CHECK(value.isInt(),
            "v2_pre_bind: arg ", arg_idx, " expected int, got ", value.tagKind());
        captured_ints_[slot] = value.toInt();
    }
    v2_arg_pre_bound_[arg_idx] = true;
}

// ---------------- CaptureContext (TLS) ----------------

namespace {
thread_local Trace* g_active_trace = nullptr;
}  // namespace

Trace* CaptureContext::active() {
    return g_active_trace;
}

std::unique_ptr<Trace> CaptureContext::begin() {
    TORCH_CHECK(
        g_active_trace == nullptr,
        "torch_dispatch_capture: a capture is already active on this thread");
    auto t = std::make_unique<Trace>();
    g_active_trace = t.get();
    return t;
}

void CaptureContext::end() {
    g_active_trace = nullptr;
}

}  // namespace tdc
