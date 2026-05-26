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

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <memory>

namespace {

// Generic Python -> IValue conversion. The torch::jit::toIValue helper
// requires a typed TypePtr; for our trace literals and replay args we
// want a duck-typed fallback that handles ints/floats/bools/Tensors/
// lists/tuples/None directly.
c10::IValue py_to_ivalue_any(pybind11::handle obj) {
    namespace py = pybind11;
    if (obj.is_none()) return c10::IValue();
    if (THPVariable_Check(obj.ptr())) {
        return c10::IValue(py::cast<at::Tensor>(obj));
    }
    if (py::isinstance<py::bool_>(obj)) return c10::IValue(py::cast<bool>(obj));
    if (py::isinstance<py::int_>(obj)) return c10::IValue(py::cast<int64_t>(obj));
    if (py::isinstance<py::float_>(obj)) return c10::IValue(py::cast<double>(obj));
    if (py::isinstance<py::str>(obj)) return c10::IValue(py::cast<std::string>(obj));
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        c10::List<c10::IValue> result(c10::AnyType::get());
        for (auto item : obj) {
            result.push_back(py_to_ivalue_any(item));
        }
        return c10::IValue(result);
    }
    // Torch-specific enums / value types (MemoryFormat, dtype, layout,
    // device) come through as their dedicated Python types. Detect by
    // class name and use the typed toIValue with the right TypePtr.
    py::object tp = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject*>(Py_TYPE(obj.ptr())));
    std::string tname = py::cast<std::string>(tp.attr("__name__"));
    if (tname == "memory_format") {
        return torch::jit::toIValue(py::reinterpret_borrow<py::object>(obj),
                                    c10::MemoryFormatType::get());
    }
    if (tname == "dtype") {
        return torch::jit::toIValue(py::reinterpret_borrow<py::object>(obj),
                                    c10::IntType::get());
    }
    if (tname == "layout") {
        return torch::jit::toIValue(py::reinterpret_borrow<py::object>(obj),
                                    c10::LayoutType::get());
    }
    if (tname == "device") {
        return torch::jit::toIValue(py::reinterpret_borrow<py::object>(obj),
                                    c10::DeviceObjType::get());
    }
    // Fall back to typed converter; if that fails the caller sees a
    // helpful error pointing to the unconvertible object.
    return torch::jit::toIValue(py::reinterpret_borrow<py::object>(obj),
                                c10::AnyType::get());
}

}  // namespace

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

    // Trace is opaque to Python — methods exposed for v1 capture/replay
    // plus v2 build-from-FX-graph + unified replay (DESIGN.md §17.6.9).
    py::class_<tdc::Trace, std::shared_ptr<tdc::Trace>>(m, "Trace")
        // ---- v1 surface ----
        .def(py::init<>())
        .def("replay", &tdc::Trace::replay,
             "Replay all captured ops. Does not return anything — the "
             "trace records side effects, so observe results by reading "
             "back tensors you captured yourself (e.g., `out=` buffers, "
             "in-place mutated inputs, externally-held outputs). A captured "
             "function may write to multiple tensors, and silently "
             "returning only one of them would mislead the caller.")
        .def("size", &tdc::Trace::size, "Number of captured ops.")
        .def("__len__", &tdc::Trace::size)
        .def("dump", &tdc::Trace::dump,
             "String representation of the trace for debugging.")
        .def("__repr__", &tdc::Trace::dump)
        // ---- v2 builder surface ----
        .def("v2_add_placeholder_tensor", &tdc::Trace::append_placeholder_tensor)
        .def("v2_add_placeholder_int", &tdc::Trace::append_placeholder_int)
        .def("v2_add_constant_tensor", &tdc::Trace::append_captured_tensor,
             "Append a constant tensor (from an FX get_attr node) to "
             "captured_tensors_ and return its slot index. Differs from "
             "v2_add_placeholder_tensor in that no placeholder_routing_ "
             "entry is created -- the slot is frozen at the captured "
             "value, not overwritten from args at every replay.")
        .def("v2_add_tensor_op_step",
             [](tdc::Trace& self,
                const std::string& full_name,
                std::vector<tdc::StepInputRef> inputs,
                size_t n_outputs,
                std::vector<tdc::ArgCoercion> coercions) -> size_t {
                 auto dot = full_name.rfind('.');
                 TORCH_CHECK(dot != std::string::npos,
                     "v2_add_tensor_op_step: expected qualified name 'ns::name.overload', got ",
                     full_name);
                 std::string base = full_name.substr(0, dot);
                 std::string overload = full_name.substr(dot + 1);
                 auto op = c10::Dispatcher::singleton().findOp({base, overload});
                 TORCH_CHECK(op.has_value(),
                     "v2_add_tensor_op_step: op not found: ", full_name);
                 TORCH_CHECK(coercions.empty() || coercions.size() == inputs.size(),
                     "v2_add_tensor_op_step: coercions length ", coercions.size(),
                     " must equal inputs length ", inputs.size());
                 tdc::Step step;
                 step.step_kind = tdc::Step::Kind::kTensorOp;
                 step.op = op.value();
                 step.inputs = std::move(inputs);
                 step.coercions = std::move(coercions);
                 step.n_outputs = n_outputs;
                 step.op_name = full_name;
                 size_t idx = self.size();
                 self.append_step(std::move(step));
                 return idx;
             },
             py::arg("full_name"),
             py::arg("inputs"),
             py::arg("n_outputs"),
             py::arg("coercions") = std::vector<tdc::ArgCoercion>{})
        .def("v2_add_pycall_step",
             [](tdc::Trace& self,
                tdc::BuiltinKind kind,
                std::vector<tdc::StepInputRef> inputs,
                py::object py_fn,
                const std::string& name) -> size_t {
                 tdc::Step step;
                 step.step_kind = tdc::Step::Kind::kPyCall;
                 step.builtin_kind = kind;
                 step.inputs = std::move(inputs);
                 step.n_outputs = 1;
                 step.op_name = name;
                 if (kind == tdc::BuiltinKind::kPyFallback) {
                     TORCH_CHECK(!py_fn.is_none(),
                         "kPyFallback step requires py_fn");
                     py_fn.inc_ref();
                     step.py_fn_handle = py_fn.ptr();
                 }
                 size_t idx = self.size();
                 self.append_step(std::move(step));
                 return idx;
             },
             py::arg("kind"),
             py::arg("inputs"),
             py::arg("py_fn") = py::none(),
             py::arg("name") = std::string())
        .def("v2_set_outputs",
             [](tdc::Trace& self, std::vector<tdc::StepInputRef> outs) {
                 self.set_outputs(std::move(outs));
             })
        .def("v2_pre_bind",
             [](tdc::Trace& self, size_t arg_idx, py::object value) {
                 self.v2_pre_bind(arg_idx, py_to_ivalue_any(value));
             },
             "Mark placeholder arg_idx as pre-bound with this value. "
             "The slot stays filled across replays; v2_replay's args "
             "list omits the slot. Used for module parameters and "
             "Dynamo-specialised constants.")
        .def("v2_replay",
             [](tdc::Trace& self, std::vector<py::object> py_args) {
                 std::vector<c10::IValue> args;
                 args.reserve(py_args.size());
                 for (auto& obj : py_args) {
                     args.push_back(py_to_ivalue_any(obj));
                 }
                 auto outputs = self.replay_v2(args);
                 std::vector<py::object> py_outputs;
                 py_outputs.reserve(outputs.size());
                 for (auto& iv : outputs) {
                     py_outputs.push_back(torch::jit::toPyObject(iv));
                 }
                 return py_outputs;
             });

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

    // -------------------------------------------------------------
    // v2 builder / replay surface.
    //
    // The Python translator (python/v2/translator.py) walks an AOT FX
    // graph and builds a C++ Trace through this API. At call time
    // Trace.replay_v2(args) runs the unified C++ replay engine.
    // -------------------------------------------------------------

    py::enum_<tdc::ArgCoercion>(m, "ArgCoercion")
        .value("NONE",                         tdc::ArgCoercion::kNone)
        .value("SCALAR_TO_TENSOR",             tdc::ArgCoercion::kScalarToTensor)
        .value("LIST_TO_INT_LIST",             tdc::ArgCoercion::kListToIntList)
        .value("LIST_TO_TENSOR_LIST",          tdc::ArgCoercion::kListToTensorList)
        .value("LIST_TO_OPTIONAL_TENSOR_LIST", tdc::ArgCoercion::kListToOptionalTensorList)
        .value("LIST_TO_BOOL_LIST",            tdc::ArgCoercion::kListToBoolList);

    py::enum_<tdc::BuiltinKind>(m, "BuiltinKind")
        .value("FLOORDIV", tdc::BuiltinKind::kFloorDiv)
        .value("TRUEDIV",  tdc::BuiltinKind::kTrueDiv)
        .value("ADD",      tdc::BuiltinKind::kAdd)
        .value("SUB",      tdc::BuiltinKind::kSub)
        .value("MUL",      tdc::BuiltinKind::kMul)
        .value("MOD",      tdc::BuiltinKind::kMod)
        .value("NEG",      tdc::BuiltinKind::kNeg)
        .value("GETITEM",  tdc::BuiltinKind::kGetItem)
        .value("EQ", tdc::BuiltinKind::kEq).value("LT", tdc::BuiltinKind::kLt)
        .value("LE", tdc::BuiltinKind::kLe).value("GT", tdc::BuiltinKind::kGt)
        .value("GE", tdc::BuiltinKind::kGe).value("NE", tdc::BuiltinKind::kNe)
        .value("SYM_MAX",   tdc::BuiltinKind::kSymMax)
        .value("SYM_MIN",   tdc::BuiltinKind::kSymMin)
        .value("SYM_INT",   tdc::BuiltinKind::kSymInt)
        .value("SYM_FLOAT", tdc::BuiltinKind::kSymFloat)
        .value("PY_FALLBACK", tdc::BuiltinKind::kPyFallback);

    // StepInputRef is opaque to Python; only constructed via factory fns.
    py::class_<tdc::StepInputRef>(m, "StepInputRef");
    m.def("v2_ref_captured_tensor",
          [](size_t idx) { return tdc::StepInputRef::CapturedTensor(idx); });
    m.def("v2_ref_captured_int",
          [](size_t idx) { return tdc::StepInputRef::CapturedInt(idx); });
    m.def("v2_ref_prev_step",
          [](size_t step, size_t slot) {
              return tdc::StepInputRef::PrevStepOutput(step, slot);
          });
    m.def("v2_ref_literal",
          [](py::object obj) {
              return tdc::StepInputRef::Literal(
                  py_to_ivalue_any(obj));
          });
    m.def("v2_ref_list",
          [](std::vector<tdc::StepInputRef> elements) {
              return tdc::StepInputRef::List(std::move(elements));
          });
    // Optimisation: when a list arg is fully literal (e.g.
    // permute([0,2,1,3])), pre-build the typed c10::List at translation
    // time and store as one kLiteral IValue. Replay's coercion table
    // tag is then kNone — no per-call IntList re-construction.
    m.def("v2_ref_literal_int_list",
          [](std::vector<int64_t> values) {
              c10::List<int64_t> ints;
              ints.reserve(values.size());
              for (auto v : values) ints.push_back(v);
              return tdc::StepInputRef::Literal(c10::IValue(std::move(ints)));
          });
    m.def("v2_ref_literal_tensor_list",
          [](std::vector<at::Tensor> tensors) {
              c10::List<at::Tensor> ts;
              ts.reserve(tensors.size());
              for (auto& t : tensors) ts.push_back(t);
              return tdc::StepInputRef::Literal(c10::IValue(std::move(ts)));
          });
}
