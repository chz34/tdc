"""Shared device helpers for the test suite.

Reads TDC_DEVICE env var; defaults to ``cpu``. Exposes:

  - ``DEVICE``  :  resolved ``torch.device`` to use for all tensor allocs
  - ``SYNC``    :  zero-arg callable that synchronizes the current device
                   (no-op on CPU, ``torch.cuda.synchronize`` on CUDA, etc.)

All test files import these via::

    from _device import DEVICE, SYNC

Usage in benchmark loops::

    SYNC()
    t0 = time.perf_counter_ns()
    fn()
    SYNC()
    elapsed = time.perf_counter_ns() - t0
"""
from __future__ import annotations

import os
from typing import Callable

import torch


def _resolve_device() -> torch.device:
    raw = os.environ.get("TDC_DEVICE", "cpu").lower()
    # User-friendly aliases — let folks write `npu` without worrying
    # whether the backend is registered as PrivateUse1.
    aliases = {"npu": "privateuseone"}
    raw = aliases.get(raw, raw)
    dev = torch.device(raw)
    if dev.type == "cuda":
        assert torch.cuda.is_available(), "TDC_DEVICE=cuda but no CUDA"
    elif dev.type == "xpu":
        assert hasattr(torch, "xpu") and torch.xpu.is_available(), \
            "TDC_DEVICE=xpu but no XPU"
    elif dev.type == "mps":
        assert torch.backends.mps.is_available(), "TDC_DEVICE=mps but no MPS"
    elif dev.type == "privateuseone":
        # The actual backend module (torch_npu, etc.) must already have
        # registered PrivateUse1 at import time. We don't verify here —
        # let the first kernel call surface the missing backend.
        pass
    return dev


DEVICE: torch.device = _resolve_device()


def _make_sync() -> Callable[[], None]:
    """Return a 0-arg synchronize callable matching ``DEVICE.type``."""
    if DEVICE.type == "cuda":
        return torch.cuda.synchronize
    if DEVICE.type == "xpu":
        return torch.xpu.synchronize
    if DEVICE.type == "mps":
        return torch.mps.synchronize
    if DEVICE.type == "privateuseone":
        if hasattr(torch, "npu"):
            return torch.npu.synchronize
        try:
            import torch_npu  # type: ignore
            return torch_npu.npu.synchronize
        except ImportError:
            return lambda: None
    return lambda: None


SYNC: Callable[[], None] = _make_sync()


def print_device_banner() -> None:
    """Print a one-line summary; call once per test class via setUpClass."""
    sync_name = getattr(SYNC, "__name__", str(SYNC))
    print(f"\n>>> TDC_DEVICE = {DEVICE} (sync = {sync_name})")
