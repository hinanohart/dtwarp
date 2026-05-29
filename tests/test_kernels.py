"""Optional CUDA fast-path: lazy/never-vendored, guard logic, CPU fallback equals reference."""

from __future__ import annotations

import sys

import torch

import dtwarp
from dtwarp import softdtw_divergence
from dtwarp.kernels.cuda_fastpath import (
    MAX_T_CUDA,
    MIN_GAMMA_CUDA,
    _fastpath_ok,
    accelerated_softdtw_divergence,
    maghoumi_available,
    should_use_cuda_fastpath,
)


def test_kernel_never_imported_at_package_load() -> None:
    """never-vendored / lazy: importing dtwarp must not import the optional CUDA kernel."""
    assert "pytorch_softdtw_cuda" not in sys.modules
    assert isinstance(maghoumi_available(), bool)


def test_guard_predicate_branches() -> None:
    # happy path (pretend CUDA)
    assert _fastpath_ok(is_cuda=True, length=256, gamma=0.1, has_padding=False) is True
    # not cuda
    assert _fastpath_ok(is_cuda=False, length=256, gamma=0.1, has_padding=False) is False
    # padding present
    assert _fastpath_ok(is_cuda=True, length=256, gamma=0.1, has_padding=True) is False
    # too long
    assert _fastpath_ok(is_cuda=True, length=MAX_T_CUDA + 1, gamma=0.1, has_padding=False) is False
    # gamma too small
    assert _fastpath_ok(is_cuda=True, length=256, gamma=MIN_GAMMA_CUDA / 2, has_padding=False) is False


def test_cpu_tensor_never_uses_fastpath() -> None:
    x = torch.randn(2, 16, 2)
    assert should_use_cuda_fastpath(x, gamma=0.1) is False
    assert should_use_cuda_fastpath(x, gamma=0.01) is False


def test_cpu_fallback_equals_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 9, 2, dtype=torch.float64)
    y = torch.randn(3, 9, 2, dtype=torch.float64)
    fast = accelerated_softdtw_divergence(x, y, gamma=0.2)
    ref = softdtw_divergence(x, y, gamma=0.2)
    assert torch.allclose(fast, ref, atol=1e-12)


def test_not_vendored_in_source_tree() -> None:
    """The repository must not vendor the Maghoumi kernel (only an optional lazy dependency)."""
    pkg_dir = __import__("pathlib").Path(dtwarp.__file__).parent
    for path in pkg_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # the only allowed reference is the lazy import inside the try-block of cuda_fastpath.py
        if path.name != "cuda_fastpath.py":
            assert "pytorch_softdtw_cuda" not in text, f"unexpected kernel reference in {path}"
