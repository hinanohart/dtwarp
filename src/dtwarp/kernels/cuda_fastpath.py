"""Optional CUDA fast-path for soft-DTW (lazy, guarded, never vendored).

The default dtwarp path is the pure-PyTorch CPU/GPU reference in ``dtwarp.core.softdtw``. When
``pytorch_softdtw_cuda`` (Maghoumi, MIT — NOT on PyPI; the user installs it themselves) is
importable AND a CUDA tensor is used AND the guard conditions hold (T <= 1024, gamma >= 0.05, no
per-sample padding), this module can delegate the raw soft-DTW to that kernel. The kernel is
imported lazily at call time — it is never a hard dependency, never imported at package import, and
never vendored into this repository. The first successful delegation is numerically cross-checked
against the CPU reference; any mismatch permanently disables the fast-path for the session.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from dtwarp.core.softdtw import softdtw_divergence

__all__ = [
    "MAX_T_CUDA",
    "MIN_GAMMA_CUDA",
    "maghoumi_available",
    "should_use_cuda_fastpath",
    "accelerated_softdtw_divergence",
]

MAX_T_CUDA = 1024
MIN_GAMMA_CUDA = 0.05

_FASTPATH_DISABLED = False  # set True if the first numeric cross-check fails
_WARNED_FALLBACK = False


def maghoumi_available() -> bool:
    """True iff the optional CUDA kernel can be imported (lazy; never imported at module load)."""
    import importlib.util

    return importlib.util.find_spec("pytorch_softdtw_cuda") is not None


def _fastpath_ok(
    is_cuda: bool,
    length: int,
    gamma: float,
    has_padding: bool,
    max_t: int = MAX_T_CUDA,
    min_gamma: float = MIN_GAMMA_CUDA,
) -> bool:
    """Pure guard predicate (unit-testable without a GPU)."""
    if _FASTPATH_DISABLED:
        return False
    if not is_cuda:
        return False
    if has_padding:
        return False  # Maghoumi has no per-sample valid-length support; use the reference
    if length > max_t:
        return False
    return gamma >= min_gamma


def should_use_cuda_fastpath(
    x: Tensor,
    gamma: float,
    valid_lengths: Tensor | None = None,
    max_t: int = MAX_T_CUDA,
    min_gamma: float = MIN_GAMMA_CUDA,
) -> bool:
    """Guard: CUDA device, bounded length, gamma not too small, and no per-sample padding."""
    return _fastpath_ok(bool(x.is_cuda), int(x.shape[1]), gamma, valid_lengths is not None, max_t, min_gamma)


def accelerated_softdtw_divergence(
    x: Tensor, y: Tensor, gamma: float = 0.1, valid_lengths: Tensor | None = None
) -> Tensor:
    """Soft-DTW divergence, using the CUDA kernel when the guard passes, else the CPU reference.

    On CPU, when padding is present, or when the kernel is unavailable, this is exactly
    ``dtwarp.softdtw_divergence`` (warned once). The CUDA branch is exercised only on GPU hardware.
    """
    global _FASTPATH_DISABLED, _WARNED_FALLBACK
    if should_use_cuda_fastpath(x, gamma, valid_lengths) and maghoumi_available():
        try:  # pragma: no cover - requires CUDA hardware, not run in CI
            from pytorch_softdtw_cuda import SoftDTW as _MaghoumiSoftDTW

            kernel = _MaghoumiSoftDTW(use_cuda=True, gamma=gamma, normalize=True)
            fast: Tensor = kernel(x, y)
            ref = softdtw_divergence(x, y, gamma=gamma, valid_lengths=None)
            if not torch.allclose(fast.float(), ref.float(), atol=1e-2, rtol=1e-2):
                _FASTPATH_DISABLED = True
                warnings.warn(
                    "dtwarp CUDA fast-path disagreed with the CPU reference; permanently "
                    "falling back to the reference for this session.",
                    stacklevel=2,
                )
                return ref
            return fast
        except Exception as exc:  # pragma: no cover - GPU only
            _FASTPATH_DISABLED = True
            warnings.warn(f"dtwarp CUDA fast-path failed ({exc!r}); using CPU reference.", stacklevel=2)
    elif x.is_cuda and not maghoumi_available() and not _WARNED_FALLBACK:
        _WARNED_FALLBACK = True
        warnings.warn(
            "dtwarp: CUDA tensor but the optional 'pytorch_softdtw_cuda' kernel is not importable; "
            "using the pure-PyTorch reference. It is not on PyPI; install it yourself "
            "(e.g. pip install 'git+https://github.com/Maghoumi/pytorch-softdtw-cuda') for the fast-path.",
            stacklevel=2,
        )
    return softdtw_divergence(x, y, gamma=gamma, valid_lengths=valid_lengths)
