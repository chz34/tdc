"""torch_dispatch_capture.v4 - capture inductor's compiled FX host graph.

See docs/specs/2026-06-04-v4-fx-capture-design.md for the design.
"""
from .capture_fx import CaptureFxWrapper, FxCaptureResult, capture_fx

__all__ = [
    "capture_fx",
    "FxCaptureResult",
    "CaptureFxWrapper",
]
