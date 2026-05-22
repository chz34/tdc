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
    dev = torch.device(raw)
    if dev.type == "cuda":
        assert torch.cuda.is_available(), "TDC_DEVICE=cuda but no CUDA"
    elif dev.type == "xpu":
        assert hasattr(torch, "xpu") and torch.xpu.is_available(), \
            "TDC_DEVICE=xpu but no XPU"
    elif dev.type == "mps":
        assert torch.backends.mps.is_available(), "TDC_DEVICE=mps but no MPS"
    elif dev.type == "npu":
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
    if DEVICE.type == "npu":
        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
            return torch.npu.synchronize
        return lambda: None
    return lambda: None


SYNC: Callable[[], None] = _make_sync()


def print_device_banner() -> None:
    """Print a one-line summary; call once per test class via setUpClass."""
    sync_name = getattr(SYNC, "__name__", str(SYNC))
    print(f"\n>>> TDC_DEVICE = {DEVICE} (sync = {sync_name})")
