"""Build torch_dispatch_capture as a C++ extension.

Build:
    MAX_JOBS=4 pip install -e .   # respects user instruction: keep jobs <= 4

The extension is a single shared library `_C.so` packaged inside the
`torch_dispatch_capture` Python package (see python/__init__.py).
"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"

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

setup(
    name="torch_dispatch_capture",
    version="0.0.1",
    description="C++ dispatcher-level capture/replay for PyTorch (PoC)",
    packages=["torch_dispatch_capture"],
    package_dir={"torch_dispatch_capture": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
