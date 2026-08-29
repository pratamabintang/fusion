import os
import sys
from setuptools import setup, find_packages
import torch

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    BuildExtension = None
    CUDAExtension = None

def get_extensions():
    if not torch.cuda.is_available() or CUDAExtension is None:
        return []

    csrc_dir = os.path.join(os.path.dirname(__file__), "src", "fusion", "csrc")
    sources = [
        os.path.join(csrc_dir, "selective_scan.cpp"),
        os.path.join(csrc_dir, "selective_scan_cuda.cu"),
    ]

    ext_modules = [
        CUDAExtension(
            name="fusion.selective_scan_cuda_core",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                    "-gencode=arch=compute_100,code=sm_100",
                    "-gencode=arch=compute_120,code=sm_120",
                ],
            },
        )
    ]
    return ext_modules

cmdclass = {}
if torch.cuda.is_available() and BuildExtension is not None:
    cmdclass["build_ext"] = BuildExtension

setup(
    name="fusion",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
    python_requires=">=3.8",
)
