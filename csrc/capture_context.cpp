#include "capture_context.h"

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

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

StepInputRef StepInputRef::Literal(c10::IValue v) {
    StepInputRef r;
    r.kind = Kind::kLiteral;
    r.literal = std::move(v);
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
                case StepInputRef::Kind::kPrevStepOutput:
                    os << "step" << ref.prev_step << ":" << ref.prev_slot;
                    break;
                case StepInputRef::Kind::kLiteral:
                    os << "lit(" << ref.literal.tagKind() << ")";
                    break;
            }
        }
        os << "]\n";
    }
    return os.str();
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
