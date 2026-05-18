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
#include <optional>
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

// Each captured op input is one of these things. v1 capture only uses
// the first three kinds; v2 translator additionally uses kCapturedInt
// and kList. DESIGN.md §17.6.2.
struct StepInputRef {
    enum class Kind {
        kCapturedTensor,  // external Tensor — trace holds a strong ref
        kPrevStepOutput,  // output produced by an earlier step in this trace
        kLiteral,         // any non-tensor IValue (int, float, scalar, ...)
        kCapturedInt,     // v2: int placeholder, extracted by Dynamo prelude
        kList,            // v2: nested list of sub-refs (matches FX immutable_list)
    };
    Kind kind;
    size_t captured_idx{0};                     // kCapturedTensor / kCapturedInt
    size_t prev_step{0};                        // kPrevStepOutput
    size_t prev_slot{0};                        // kPrevStepOutput
    c10::IValue literal;                        // kLiteral
    std::vector<StepInputRef> list_elements;    // kList
    // If true, this arg is a schema-declared `out=` tensor (pure write).
    // Replay shrinks it to zero elements before the call so the kernel
    // auto-resizes to whatever shape the current inputs require — this is
    // PyTorch's recommended pattern for dynamic-shape output reuse and
    // avoids the "output was resized" deprecation warning.
    bool is_out{false};

    static StepInputRef CapturedTensor(size_t idx, bool is_out = false);
    static StepInputRef PrevStepOutput(size_t step, size_t slot, bool is_out = false);
    static StepInputRef Literal(c10::IValue v);
    static StepInputRef CapturedInt(size_t idx);
    static StepInputRef List(std::vector<StepInputRef> elements);
};

// Tag for kPyCall steps. The few Python builtin / torch.sym helpers
// that v2 can encounter in an AOT graph all have direct C++ equivalents;
// only kPyFallback truly needs a py::object call. DESIGN §17.6.9.
enum class BuiltinKind : int32_t {
    kFloorDiv,    // operator.floordiv
    kTrueDiv,     // operator.truediv
    kAdd,         // operator.add (on ints)
    kSub,         // operator.sub
    kMul,         // operator.mul
    kMod,         // operator.mod
    kNeg,         // operator.neg
    kGetItem,     // operator.getitem on tuple/list IValue
    kEq, kLt, kLe, kGt, kGe, kNe,
    kSymMax,      // torch.sym_max
    kSymMin,      // torch.sym_min
    kSymInt,      // torch.sym_int
    kSymFloat,    // torch.sym_float
    kPyFallback,  // unrecognised callable, call via py::object
    kNumBuiltinKinds,
};

// One captured op.
struct Step {
    // Step::Kind {kTensorOp, kPyCall}. v1 capture always emits kTensorOp;
    // v2 translator emits both kinds depending on FX node target.
    enum class Kind {
        kTensorOp,    // op.callBoxed(stack)
        kPyCall,      // builtin C++ switch or py::object call
    };
    Kind step_kind{Kind::kTensorOp};

    // ---- kTensorOp fields ----
    // The OperatorHandle for this op. std::optional to allow default-
    // construction (kPyCall steps don't carry an op).
    std::optional<c10::OperatorHandle> op;
    // Dispatch key the kernel will be invoked on at replay time. v1 sets
    // this to the pre-resolved key (after removing kCaptureKey); v2 leaves
    // it as Undefined and uses callBoxed for full dispatch.
    c10::DispatchKey target_dk{c10::DispatchKey::Undefined};

    // ---- kPyCall fields ----
    BuiltinKind builtin_kind{BuiltinKind::kPyFallback};
    // Only used when builtin_kind == kPyFallback. Held opaquely so the
    // C++ replay can invoke it via pybind11 without taking a pybind
    // dependency in capture_context.h.
    void* py_fn_handle{nullptr};   // actually a PyObject*; managed by Trace

    // Description of how to reconstruct each input on replay.
    std::vector<StepInputRef> inputs;
    // Number of return values. For kTensorOp matches schema. For kPyCall
    // always 1 (a single IValue, possibly a tuple/list).
    size_t n_outputs{0};
    // Op name for debugging / dump.
    std::string op_name;

    // v1 constructor (kTensorOp by default).
    Step(c10::OperatorHandle h,
         c10::DispatchKey dk,
         std::vector<StepInputRef> ins,
         size_t n_out,
         std::string name);

    // Default constructor for v2 builder path.
    Step() = default;
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

    // v2 replay path: returns a vector of IValues (the final outputs as
    // declared by set_outputs()). `args` is the positional input list in
    // graph-placeholder order, mixing concrete ints and Tensors per
    // placeholder_routing_. Used by torch_dispatch_capture.v2.
    std::vector<c10::IValue> replay_v2(
        const std::vector<c10::IValue>& args);

    size_t size() const { return steps_.size(); }
    std::string dump() const;

    // ---- shared mutators (v1 + v2) ----
    void append_step(Step&& step) { steps_.emplace_back(std::move(step)); }
    size_t append_captured_tensor(at::Tensor t) {
        captured_tensors_.emplace_back(std::move(t));
        return captured_tensors_.size() - 1;
    }

    // ---- v1-only: capture-time TensorImpl identity tracking ----
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

    // ---- v2-only: graph-input routing + final outputs ----
    // Encode "the k-th positional arg goes into captured_tensors_[idx]
    // or captured_ints_[idx]".
    enum class PlaceholderTarget { kTensor, kInt };
    size_t append_placeholder_tensor() {
        placeholder_routing_.push_back({PlaceholderTarget::kTensor, n_captured_tensors_++});
        return n_captured_tensors_ - 1;
    }
    size_t append_placeholder_int() {
        placeholder_routing_.push_back({PlaceholderTarget::kInt, n_captured_ints_++});
        return n_captured_ints_ - 1;
    }
    void set_outputs(std::vector<StepInputRef> outs) { outputs_ = std::move(outs); }
    size_t n_captured_tensors_count() const { return n_captured_tensors_; }
    size_t n_captured_ints_count() const { return n_captured_ints_; }

private:
    std::vector<Step> steps_;
    // External tensors referenced by Step inputs. Strong refs keep them alive.
    std::vector<at::Tensor> captured_tensors_;
    // v1 capture-time bookkeeping: TensorImpl* -> (step, output slot).
    std::unordered_map<c10::TensorImpl*, std::pair<size_t, size_t>> tensor_to_step_;

    // ---- v2 fields ----
    // Captured ints come from Dynamo prelude (call_size etc.) at replay
    // time; nothing populates them during v1 dispatch capture.
    size_t n_captured_tensors_{0};
    size_t n_captured_ints_{0};
    std::vector<std::pair<PlaceholderTarget, size_t>> placeholder_routing_;
    std::vector<StepInputRef> outputs_;
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
