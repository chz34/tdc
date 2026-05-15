// Python bindings for torch_dispatch_capture.
//
// Exposed:
//   - begin_capture() -> Trace handle (opaque py::object holding a unique_ptr)
//   - end_capture(handle)
//   - is_capturing()
//   - The Trace class (wrapped) with .replay() / .size() / .dump()
//
// Note: This module also registers the boxed fallback on the capture key
// (TESTING_ONLY_GenericMode) by virtue of compiling capture_fallback.cpp's
// TORCH_LIBRARY_IMPL block into the same .so. We rely on static initializer
// order for registration to happen at import time.

#include "capture_context.h"

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>

namespace py = pybind11;

namespace {

// RAII pair that owns the TLS Include guard for the capture key. We keep
// this alive between begin() and end() by stashing it on the Trace's Python
// object via py::cpp_function-controlled keepalive — but a simpler approach
// is to store it as a thread_local unique_ptr.
thread_local std::unique_ptr<c10::impl::IncludeDispatchKeyGuard> g_include_guard;

}  // namespace

PYBIND11_MODULE(_C, m) {
    m.doc() = "torch_dispatch_capture: capture/replay at PyTorch dispatcher level";

    // Trace is opaque to Python — only methods are exposed.
    py::class_<tdc::Trace, std::shared_ptr<tdc::Trace>>(m, "Trace")
        .def("replay", &tdc::Trace::replay,
             "Replay all captured ops. Does not return anything — the "
             "trace records side effects, so observe results by reading "
             "back tensors you captured yourself (e.g., `out=` buffers, "
             "in-place mutated inputs, externally-held outputs). A captured "
             "function may write to multiple tensors, and silently "
             "returning only one of them would mislead the caller.")
        .def("size", &tdc::Trace::size,
             "Number of captured ops.")
        .def("__len__", &tdc::Trace::size)
        .def("dump", &tdc::Trace::dump,
             "String representation of the trace for debugging.")
        .def("__repr__", &tdc::Trace::dump);

    m.def("begin_capture", []() -> std::shared_ptr<tdc::Trace> {
        // Check for nested capture BEFORE touching g_include_guard. If we
        // pushed the new guard first and then CaptureContext::begin threw,
        // the unique_ptr assignment would have already destroyed the outer
        // capture's IncludeDispatchKeyGuard — leaving TLS include in an
        // inconsistent state that survives across tests.
        TORCH_CHECK(
            !tdc::CaptureContext::is_active(),
            "torch_dispatch_capture: a capture is already active on this thread");

        // Push our capture key into TLS include set so the boxed fallback
        // fires on every dispatcher call.
        g_include_guard = std::make_unique<c10::impl::IncludeDispatchKeyGuard>(
            c10::DispatchKeySet(tdc::kCaptureKey));
        auto t = tdc::CaptureContext::begin();
        return std::shared_ptr<tdc::Trace>(t.release());
    }, "Begin a capture on the current thread. Returns a Trace.");

    m.def("end_capture", []() {
        tdc::CaptureContext::end();
        g_include_guard.reset();
    }, "End the current capture. Must be paired with begin_capture().");

    m.def("is_capturing", &tdc::CaptureContext::is_active,
          "Whether a capture is currently active on this thread.");

    m.attr("capture_dispatch_key") = py::cast(
        static_cast<int>(tdc::kCaptureKey));
}
