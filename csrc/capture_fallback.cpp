// Boxed fallback registered on DispatchKey::TESTING_ONLY_GenericMode.
// Activated when the user enters `with capture():` which pushes
// kCaptureKey into the TLS include set.
//
// On each invocation we:
//   1. Classify each stack input as captured-tensor / prev-step-output /
//      literal-IValue.
//   2. Pre-resolve the target DispatchKey (highest priority after removing
//      our capture key) — stored on the Step for potential v1.1 fast-path
//      replay. v1 replay goes through full Dispatcher::call, so this is
//      informational only.
//   3. Execute the op by re-dispatching with the capture key excluded. The
//      dispatcher then picks the actual backend kernel.
//   4. Stash output TensorImpl identities so subsequent steps can resolve
//      their inputs as prev-step references.
//
// Position: kCaptureKey (TESTING_ONLY_GenericMode, enum #411) is one of
// the highest-priority dispatch keys — well above AutogradFunctionality
// (#333) and ADInplaceOrView (#304). So when the user calls .backward()
// inside the with-block (allow_grad=True path), the backward aten ops
// dispatched by the autograd engine also pass through this fallback.
// That's why a single trace can carry forward + backward.

#include "capture_context.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace tdc {

namespace {

StepInputRef classify_input(c10::IValue iv, Trace& trace, bool is_out) {
    if (iv.isTensor()) {
        const auto& t = iv.toTensor();
        auto* impl = t.unsafeGetTensorImpl();
        size_t step, slot;
        int sub_slot;
        if (trace.lookup_output_identity(impl, step, slot, sub_slot)) {
            if (sub_slot < 0) {
                return StepInputRef::PrevStepOutput(step, slot, is_out);
            }
            // Tensor came from a list-returning op (unbind / split / chunk
            // / meshgrid). Resolve to (step, slot)[sub_slot] at replay.
            return StepInputRef::PrevStepListElement(step, slot, sub_slot, is_out);
        }
        size_t idx = trace.append_captured_tensor(t);
        return StepInputRef::CapturedTensor(idx, is_out);
    }
    return StepInputRef::Literal(std::move(iv));
}

void capture_fallback(const c10::OperatorHandle& op,
                      c10::DispatchKeySet ks,
                      torch::jit::Stack* stack) {
    auto* trace = CaptureContext::active();
    if (trace == nullptr) {
        op.callBoxed(stack);
        return;
    }

    const auto& schema = op.schema();
    const auto n_args = schema.arguments().size();
    TORCH_CHECK(stack->size() >= n_args,
                "stack underflow during capture for ", schema.name());

    // Pre-resolve the target dispatch key for v1.1's fast-path. Functionality
    // keys like TESTING_ONLY_GenericMode can be simply subtracted from the
    // keyset (only one bit removed, no backend interaction).
    const auto effective_ks = ks - c10::DispatchKeySet(kCaptureKey);
    const auto target_dk = effective_ks.highestPriorityTypeId();

    // Classify inputs by walking the top n_args entries of the stack.
    // Read schema.arguments()[i].is_out() to mark `out=` args so replay
    // can pre-resize them and let the kernel auto-allocate.
    const auto& schema_args = schema.arguments();
    std::vector<StepInputRef> input_refs;
    input_refs.reserve(n_args);
    for (auto i : c10::irange(n_args)) {
        const auto& iv = (*stack)[stack->size() - n_args + i];
        const bool is_out = schema_args[i].is_out();
        input_refs.emplace_back(classify_input(iv, *trace, is_out));
    }

    const auto step_idx = trace->size();
    const auto n_returns = schema.returns().size();

    // Execute via explicit redispatch. We pass effective_ks (the input
    // keyset minus our capture key) so the dispatcher resumes exactly
    // where it would have without us. ExcludeDispatchKeyGuard additionally
    // guards against composite kernels that re-enter the dispatcher freshly
    // (via at::_ops::xxx::call rather than redispatch), because those
    // calls would otherwise re-pick our key from TLS.
    {
        c10::impl::ExcludeDispatchKeyGuard exclude(kCaptureKey);
        op.redispatchBoxed(effective_ks, stack);
    }

    // Record output identities.
    TORCH_CHECK(stack->size() >= n_returns,
                "stack underflow on returns for ", schema.name());
    for (auto slot : c10::irange(n_returns)) {
        const auto& iv = (*stack)[stack->size() - n_returns + slot];
        if (iv.isTensor()) {
            auto* impl = iv.toTensor().unsafeGetTensorImpl();
            trace->register_output_identity(impl, step_idx, slot);
        } else if (iv.isList()) {
            // List-returning ops (unbind / split / chunk / meshgrid /
            // tensor_split) put a c10::List<IValue> at this slot. The
            // Python-level destructure `q, k, v = ...` doesn't go
            // through the dispatcher, so v1 has no other chance to
            // register each element's TensorImpl identity. Walk the
            // list and register each tensor with its sub_slot so a
            // later op that consumes q / k / v can find them.
            auto list = iv.toList();
            for (size_t k = 0; k < list.size(); ++k) {
                c10::IValue elem = list[k];
                if (elem.isTensor()) {
                    auto* impl = elem.toTensor().unsafeGetTensorImpl();
                    trace->register_output_identity(
                        impl, step_idx, slot, static_cast<int>(k));
                }
            }
        }
    }

    trace->append_step(Step(
        op,
        target_dk,
        std::move(input_refs),
        n_returns,
        schema.name() + (schema.overload_name().empty()
            ? std::string()
            : "." + schema.overload_name())));
}

}  // namespace

TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericMode, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&capture_fallback>());
}

}  // namespace tdc
