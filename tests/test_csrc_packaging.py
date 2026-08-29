import os
import sys
from pathlib import Path
import pytest
import ast
import torch

def test_csrc_files_exist():
    base_dir = Path(__file__).resolve().parent.parent
    csrc_dir = base_dir / "src" / "fusion" / "csrc"
    
    assert (csrc_dir / "selective_scan.h").exists(), "selective_scan.h missing"
    assert (csrc_dir / "selective_scan.cpp").exists(), "selective_scan.cpp missing"
    assert (csrc_dir / "selective_scan_cuda.cu").exists(), "selective_scan_cuda.cu missing"

def test_setup_py_exists_and_valid():
    base_dir = Path(__file__).resolve().parent.parent
    setup_file = base_dir / "setup.py"
    assert setup_file.exists(), "setup.py missing"
    
    # Check for syntax errors
    with open(setup_file, "r") as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        pytest.fail(f"setup.py has syntax errors: {e}")

def test_selective_scan_fallback_cpu():
    from fusion.ops.selective_scan import selective_scan_fn
    
    B, C, L = 2, 4, 256
    u = torch.randn(B, C, L)
    delta = torch.rand(B, C, L)
    A = torch.randn(C, 4)
    B_tensor = torch.randn(B, 4, L)
    C_tensor = torch.randn(B, 4, L)
    D = torch.randn(C)
    delta_bias = torch.randn(C)

    out = selective_scan_fn(u, delta, A, B_tensor, C_tensor, D, z=None, delta_bias=delta_bias, delta_softplus=True)
    assert out.shape == u.shape
