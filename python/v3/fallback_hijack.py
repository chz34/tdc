"""force_all_fallback(): temporarily replace every OpOverload lowering
with its fallback_handler, and enable cpp_wrapper + fusion-disabling
config flags. NOT thread-safe -- patches the module-level lowerings dict.
"""
from __future__ import annotations

import contextlib

import torch
import torch._inductor.config as inductor_config
from torch._inductor.lowering import fallback_handler, lowerings


@contextlib.contextmanager
def force_all_fallback():
    saved = dict(lowerings)
    try:
        for key in list(lowerings.keys()):
            if isinstance(key, torch._ops.OpOverload):
                lowerings[key] = fallback_handler(key, add_to_fallback_set=False)
        with inductor_config.patch({
            "cpp_wrapper": True,
            "epilogue_fusion": False,
            "max_fusion_size": 1,
            "triton.cudagraphs": False,
            "freezing": False,
        }):
            yield
    finally:
        lowerings.clear()
        lowerings.update(saved)
