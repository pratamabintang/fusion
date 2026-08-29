# 11: NVIDIA RTX 5090 Blackwell CUDA Packaging & C++ Extension

**What to build:** The production C++/CUDA packaging for the high-performance selective scan operator under `src/fusion/csrc/` and `setup.py`, configured with compute capability support for NVIDIA Blackwell architectures (`sm_100` and `sm_120`), along with graceful CPU fallback.

**Blocked by:** None

**Status:** done

- [x] Place C++/CUDA kernel sources under `src/fusion/csrc/` (`selective_scan.cpp`, `selective_scan_cuda.cu`, `selective_scan.h`).
- [x] Configure `setup.py` using `torch.utils.cpp_extension.CUDAExtension` targeting `sm_100` (Blackwell Data Center) and `sm_120` (Blackwell GeForce RTX 5090).
- [x] Implement graceful fallback in `setup.py` so CPU-only systems can install in editable mode without compilation failures.
- [x] Verify `setup.py` and package build validity with PyTest.
