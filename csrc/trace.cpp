// Trace::replay — the hot path.
//
// For each step:
//   1. Rebuild the stack from StepInputRef entries. Tensor inputs are
//      pushed as the *current* captured Tensor object — its metadata
//      (sizes/strides/data_ptr) is read by the kernel at call time, so
//      in-place mutation and resize automatically propagate. Prev-step
//      outputs are pulled from the per-replay outputs[] table; literals
//      are pushed verbatim.
//   2. Invoke Dispatcher::callBoxedForDispatchKey(op, target_dk, stack).
//      This is a public API that:
//        - skips key extraction from input tensors (cheap, but still cost)
//        - skips alias key resolution (uses kernelForDispatchKey(dk)
//          directly)
//        - skips reentrancy bookkeeping
//        - skips profiler RECORD_FUNCTION hooks
//      What it preserves: the kernel sees the full input keyset (computed
//      fresh from stack), so any redispatch the kernel does internally
//      works correctly. This trades a bit of theoretical max speed for
//      robustness vs. ops registered as composite kernels.

#include "capture_context.h"

#include <ATen/core/LegacyTypeDispatch.h>      // AutoDispatchBelowAutograd
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

namespace tdc {

void Trace::replay() {
    // Defensive: ensure our capture key is NOT included during replay.
    c10::impl::ExcludeDispatchKeyGuard guard{c10::DispatchKeySet(kCaptureKey)};

    // Exclude Autograd at replay so the dispatcher does NOT re-enter
    // VariableType wrappers. This serves two purposes:
    //   1. The captured trace already records both forward and backward
    //      aten ops (if any) — re-running autograd at replay would
    //      build a second backward graph that nobody traverses, wasting
    //      cycles and dirtying tensor states (attaching grad_fn to
    //      .grad, version-counter bumps, ...).
    //   2. The captured forward steps' grad_fn metadata is bound to
    //      capture-time tensor identities; rebuilding it would break
    //      observation buffers (e.g., x.grad acquiring grad_fn means
    //      it can no longer be resize_'d for the next replay).
    at::AutoDispatchBelowAutograd no_autograd_guard;

    if (steps_.empty()) {
        return;
    }

    std::vector<std::vector<c10::IValue>> outputs(steps_.size());
    torch::jit::Stack stack;
    stack.reserve(16);

    for (auto i : c10::irange(steps_.size())) {
        const auto& step = steps_[i];
        stack.clear();

        for (const auto& ref : step.inputs) {
            switch (ref.kind) {
                case StepInputRef::Kind::kCapturedTensor: {
                    const auto& t = captured_tensors_[ref.captured_idx];
                    if (ref.is_out) {
                        // Shrink to zero so the kernel will resize_output() to
                        // match the current input-derived shape. This lets
                        // users mutate input shapes without manually resizing
                        // their `out=` tensor before each replay.
                        t.unsafeGetTensorImpl()->set_sizes_contiguous({0});
                    }
                    stack.emplace_back(t);
                    break;
                }
                case StepInputRef::Kind::kPrevStepOutput: {
                    TORCH_INTERNAL_ASSERT(ref.prev_step < i);
                    TORCH_INTERNAL_ASSERT(ref.prev_slot < outputs[ref.prev_step].size());
                    const auto& iv = outputs[ref.prev_step][ref.prev_slot];
                    if (ref.is_out && iv.isTensor()) {
                        iv.toTensor().unsafeGetTensorImpl()->set_sizes_contiguous({0});
                    }
                    stack.emplace_back(iv);
                    break;
                }
                case StepInputRef::Kind::kLiteral:
                    stack.emplace_back(ref.literal);
                    break;
            }
        }

        // For v1 we go through the regular dispatcher (with our capture key
        // excluded). This is less performant than callBoxedForDispatchKey,
        // but works robustly for ops whose CPU kernels internally redispatch
        // (e.g., aten::addmm.out → aten::as_strided). Switching to a
        // pre-resolved kernel pointer is a v1.1 optimization once we have
        // correctness coverage.
        step.op.callBoxed(&stack);

        TORCH_CHECK(stack.size() == step.n_outputs,
                    "replay stack size mismatch for ", step.op_name,
                    ": expected ", step.n_outputs, ", got ", stack.size());
        outputs[i].assign(stack.begin(), stack.end());
    }
}

}  // namespace tdc
