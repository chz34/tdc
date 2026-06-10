"""Build torch_dispatch_capture.

Full build (v1/v2 C++ extension + all pure-Python versions):
    MAX_JOBS=4 pip install -e .   # respects user instruction: keep jobs <= 4

Pure-Python v4 only (no C++ compile -- fast, portable):
    TDC_PURE_PYTHON=1 pip install -e .
This skips the C++ extension and installs only the top-level namespace plus
torch_dispatch_capture.v4 (which is self-contained pure Python). v1/v2 (which
need the _C extension) are unavailable in this mode; `import
torch_dispatch_capture.v4` still works.

The extension is a single shared library `_C.so` packaged inside the
`torch_dispatch_capture` Python package (see python/__init__.py).
"""
import os
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"


def _is_truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


PURE_PYTHON = _is_truthy(os.environ.get("TDC_PURE_PYTHON", ""))

if PURE_PYTHON:
    # Pure-Python v4 only: no C++ extension, no torch.utils build machinery.
    packages = ["torch_dispatch_capture", "torch_dispatch_capture.v4"]
    package_dir = {
        "torch_dispatch_capture": "python",
        "torch_dispatch_capture.v4": "python/v4",
    }
    ext_modules = []
    cmdclass = {}
else:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    # Respect the project rule: never spawn more than 4 parallel compile jobs.
    os.environ.setdefault("MAX_JOBS", "4")
    if int(os.environ.get("MAX_JOBS", "4")) > 4:
        os.environ["MAX_JOBS"] = "4"

    sources = sorted(str(p) for p in CSRC.glob("*.cpp"))
    ext = CppExtension(
        name="torch_dispatch_capture._C",
        sources=sources,
        include_dirs=[str(CSRC)],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
                "-Wall",
                "-Wno-unused-function",
            ],
        },
    )
    packages = [
        "torch_dispatch_capture",
        "torch_dispatch_capture.v2",
        "torch_dispatch_capture.v3",
        "torch_dispatch_capture.v4",
    ]
    package_dir = {
        "torch_dispatch_capture": "python",
        "torch_dispatch_capture.v2": "python/v2",
        "torch_dispatch_capture.v3": "python/v3",
        "torch_dispatch_capture.v4": "python/v4",
    }
    ext_modules = [ext]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="torch_dispatch_capture",
    version="0.0.1",
    description="C++ dispatcher-level capture/replay for PyTorch (PoC)",
    packages=packages,
    package_dir=package_dir,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
)
