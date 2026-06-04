"""torch_dispatch_capture.v3 - Inductor cpp_wrapper probe.

See docs/specs/2026-05-28-v3-design.md for the design.
"""
from .compile import (
    capture,
    capture_fallback,
    isolate_fresh_fn,
    last_capture_report,
)
from .fallback_hijack import (
    NO_FUSION_CONFIG,
    force_all_fallback,
    force_all_fallback_lowerings,
)
from .fallback_backend import (
    CppWrapperFallback,
    CppWrapperFallbackCpu,
    CppWrapperFallbackGpu,
    make_fallback_backend,
)

__all__ = [
    "capture",
    "capture_fallback",
    "isolate_fresh_fn",
    "last_capture_report",
    "force_all_fallback",
    "force_all_fallback_lowerings",
    "NO_FUSION_CONFIG",
    "make_fallback_backend",
    "CppWrapperFallback",
    "CppWrapperFallbackCpu",
    "CppWrapperFallbackGpu",
]
