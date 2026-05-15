// Copyright. PoC for C++ dispatcher-level capture/replay.
// See DESIGN.md at the repo root for the full design rationale.
#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/ivalue.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tdc {

// The capture DispatchKey. We use TESTING_ONLY_GenericMode because:
//   - it's a functionality key (enum 411), not a backend bit, so removing
//     it via DispatchKeySet subtraction does NOT clear shared functionality
//     bits like Dense — avoiding the empty-keyset bug we hit with PU2.
//   - it has a very high priority (#3 in the global order, only below
//     PythonDispatcher and PreDispatch), so it fires before any
//     Autograd/ADInplaceOrView processing. This means our fallback sees
//     the "raw" op as user code dispatched it, not the post-autograd
//     decomposition. Trade-off vs PU2 (which would fire below Autograd):
//     we record more ops (incl. Autograd wrappers), but the dispatch
//     model is simpler and there's no subtraction-corruption issue.
constexpr c10::DispatchKey kCaptureKey = c10::DispatchKey::TESTING_ONLY_GenericMode;

// Each captured op input is one of three things.
struct StepInputRef {
    enum class Kind {
        kCapturedTensor,  // external Tensor — trace holds a strong ref
        kPrevStepOutput,  // an output produced by an earlier step in this trace
        kLiteral,         // any non-tensor IValue (Scalar, int, list of ints, ...)
    };
    Kind kind;
    size_t captured_idx{0};   // valid iff kCapturedTensor
    size_t prev_step{0};      // valid iff kPrevStepOutput
    size_t prev_slot{0};      // valid iff kPrevStepOutput
    c10::IValue literal;      // valid iff kLiteral
    // If true, this arg is a schema-declared `out=` tensor (pure write).
    // Replay shrinks it to zero elements before the call so the kernel
    // auto-resizes to whatever shape the current inputs require — this is
    // PyTorch's recommended pattern for dynamic-shape output reuse and
    // avoids the "output was resized" deprecation warning.
    bool is_out{false};

    static StepInputRef CapturedTensor(size_t idx, bool is_out = false);
    static StepInputRef PrevStepOutput(size_t step, size_t slot, bool is_out = false);
    static StepInputRef Literal(c10::IValue v);
};

// One captured op.
struct Step {
    // The OperatorHandle for this op. Replay uses
    // Dispatcher::callBoxedForDispatchKey(handle, target_dk, stack) which
    // bypasses key extraction + alias resolution.
    c10::OperatorHandle op;
    // Dispatch key the kernel will be invoked on at replay time. Pre-resolved
    // at capture, equal to the highest-priority key the dispatcher would have
    // selected after removing our capture key.
    c10::DispatchKey target_dk;
    // Description of how to reconstruct each input on replay.
    std::vector<StepInputRef> inputs;
    // Number of return values (matches schema). Used by replay to slice
    // the stack after the kernel call.
    size_t n_outputs{0};
    // Op name for debugging / dump.
    std::string op_name;

    Step(c10::OperatorHandle h,
         c10::DispatchKey dk,
         std::vector<StepInputRef> ins,
         size_t n_out,
         std::string name);
};

// Owned by the Python side; one per `with capture(): ...` block.
class Trace {
public:
    Trace() = default;
    ~Trace() = default;
    Trace(const Trace&) = delete;
    Trace& operator=(const Trace&) = delete;

    // The hot path. See implementation in trace.cpp for details.
    //
    // Returns nothing intentionally: a trace is a recording of *side
    // effects*, not a pure function. The user observes results by reading
    // back any Tensor they themselves captured (typically by passing it as
    // `out=` or by mutating it in-place inside the captured block). A
    // captured function may write to multiple buffers (e.g., KV cache + Q +
    // attention output) and only one of those would be the "last step's
    // output" — returning that one value silently hides the rest, so we
    // don't return it at all and force the user to be explicit about which
    // tensors they care about.
    void replay();

    size_t size() const { return steps_.size(); }
    std::string dump() const;

    // Used by capture_fallback during capture; not part of public API.
    void append_step(Step&& step) { steps_.emplace_back(std::move(step)); }
    size_t append_captured_tensor(at::Tensor t) {
        captured_tensors_.emplace_back(std::move(t));
        return captured_tensors_.size() - 1;
    }
    void register_output_identity(c10::TensorImpl* impl, size_t step, size_t slot) {
        tensor_to_step_[impl] = {step, slot};
    }
    bool lookup_output_identity(c10::TensorImpl* impl, size_t& step, size_t& slot) const {
        auto it = tensor_to_step_.find(impl);
        if (it == tensor_to_step_.end()) return false;
        step = it->second.first;
        slot = it->second.second;
        return true;
    }

private:
    std::vector<Step> steps_;
    // External tensors referenced by Step inputs. Strong refs keep them alive.
    std::vector<at::Tensor> captured_tensors_;
    // Maps a TensorImpl* observed during capture to (step, output slot) so we
    // can tell whether a downstream input came from a previous step. The map
    // only needs to remain valid during the capture phase; after capture it
    // could be cleared, but we keep it for `dump()` introspection.
    std::unordered_map<c10::TensorImpl*, std::pair<size_t, size_t>> tensor_to_step_;
};

// Thread-local registry of the currently-active capture.
class CaptureContext {
public:
    // Returns the active Trace for this thread, or nullptr if not capturing.
    static Trace* active();

    // Start a new capture. Throws if another capture is already active on
    // this thread. Returns a non-owning pointer to the new Trace, which is
    // also owned by the returned unique_ptr held by the caller.
    static std::unique_ptr<Trace> begin();

    // Ends the capture. The previously-returned unique_ptr is now the user's
    // sole owner. Does nothing if no capture is active.
    static void end();

    static bool is_active() { return active() != nullptr; }
};

}  // namespace tdc
