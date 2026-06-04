"""All-fallback helpers, decoupled into two independent pieces:

  - force_all_fallback_lowerings(): the FALLBACK part -- replace every
    OpOverload lowering with its fallback_handler. Touches ONLY the lowerings
    dict, no config. Compose with whatever wrapper config the caller needs.
  - NO_FUSION_CONFIG: the CONFIG payload that pairs with all-fallback regardless
    of wrapper (no fusion / cudagraph / freezing).
  - force_all_fallback(): backwards-compatible bundle = lowerings + cpp_wrapper
    + NO_FUSION_CONFIG (the original v3-fallback combination).

Keeping the two decoupled lets callers pick their wrapper: v3-fallback wants
cpp_wrapper=True; an fx_wrapper capture wants fx_wrapper=True + cpp_wrapper=False
(otherwise FallbackKernel takes the cpp_wrapper runtime-dispatch path and emits a
raw Python op call that the FX converter cannot consume).

NOT thread-safe -- patches the module-level lowerings dict.
"""
from __future__ import annotations

import contextlib

import torch
import torch._inductor.config as inductor_config
from torch._inductor.lowering import fallback_handler, lowerings


# Config that pairs with all-fallback for ANY wrapper: every op becomes a
# FallbackKernel, so there is nothing to fuse, and FallbackKernels invoke the
# eager dispatcher + allocator and are not cudagraph-safe (cudaMalloc / sync in
# the capture stream). The wrapper choice (cpp_wrapper / fx_wrapper) is NOT
# included here -- callers add it.
NO_FUSION_CONFIG = {
    "epilogue_fusion": False,
    "max_fusion_size": 1,
    "triton.cudagraphs": False,
    "freezing": False,
}


@contextlib.contextmanager
def force_all_fallback_lowerings():
    """The fallback part: replace every OpOverload lowering with its
    fallback_handler so no op gets fused/compiled. Restores the lowerings dict
    on exit. No config changes."""
    saved = dict(lowerings)
    try:
        for key in list(lowerings.keys()):
            if isinstance(key, torch._ops.OpOverload):
                lowerings[key] = fallback_handler(key, add_to_fallback_set=False)
        yield
    finally:
        lowerings.clear()
        lowerings.update(saved)


@contextlib.contextmanager
def force_all_fallback():
    """Backwards-compatible bundle: all-fallback lowerings + cpp_wrapper +
    NO_FUSION_CONFIG. For a non-cpp wrapper (e.g. fx_wrapper) compose
    force_all_fallback_lowerings() with your own inductor_config.patch instead,
    so cpp_wrapper does not leak in."""
    with force_all_fallback_lowerings():
        with inductor_config.patch({"cpp_wrapper": True, **NO_FUSION_CONFIG}):
            yield
