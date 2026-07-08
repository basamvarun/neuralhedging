"""
setup.py — Build Script for QuantEdge C++ Extension

Compiles the C++ engine and pybind11 bindings into a Python-importable
shared library: `quantedge_cpp.so`

Usage (from the QuantEdge/ directory):
    pip install pybind11
    pip install -e .

After installation, Python can do:
    import quantedge_cpp
    result = quantedge_cpp.pricing.black_scholes(S=100, K=100, T=1,
                                                  r=0.05, sigma=0.2,
                                                  type=quantedge_cpp.pricing.OptionType.Call)
"""

import sys
import os
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess


class CMakeBuild(build_ext):
    """Build the extension using CMake instead of distutils."""

    def build_extension(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_dir = Path(self.build_temp).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)

        cpp_dir = Path(__file__).parent / "cpp"

        # Get pybind11 cmake dir
        import pybind11
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--config", "Release", "--", "-j4"]

        subprocess.check_call(
            ["cmake", str(cpp_dir)] + cmake_args, cwd=str(build_dir)
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=str(build_dir)
        )


setup(
    name="quantedge",
    version="1.0.0",
    description="QuantEdge: Neural Hedging with C++ Computational Engine",
    packages=["python"],
    ext_modules=[Extension("quantedge_cpp", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "pybind11>=2.11",
        "numpy",
        "pandas",
        "torch",
        "scipy",
        "pyyaml",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.10",
    zip_safe=False,
)
