"""torch_dispatch_capture.v4 - capture inductor's compiled FX host graph.

See docs/specs/2026-06-04-v4-fx-capture-design.md for the design.
"""
from .capture_fx import (
    NO_FUSION_CONFIG,
    BackendFxWrapper,
    CaptureFxWrapper,
    FxCaptureResult,
    capture_fx,
    compile_with_gm_backend,
    enable_device_via_fallback,
    force_all_fallback_lowerings,
)

__all__ = [
    "capture_fx",
    "compile_with_gm_backend",
    "enable_device_via_fallback",
    "force_all_fallback_lowerings",
    "NO_FUSION_CONFIG",
    "FxCaptureResult",
    "CaptureFxWrapper",
    "BackendFxWrapper",
]
