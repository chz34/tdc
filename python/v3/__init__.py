"""torch_dispatch_capture.v3 - Inductor cpp_wrapper probe.

See docs/specs/2026-05-28-v3-design.md for the design.
"""
from .fallback_hijack import force_all_fallback

__all__ = ["force_all_fallback"]
