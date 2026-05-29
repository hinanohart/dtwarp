"""Core soft-DTW divergence math: identity, non-negativity, tslearn agreement, gradcheck."""

from __future__ import annotations

import pytest
import torch

from dtwarp import soft_dtw, softdtw_divergence
from dtwarp.core.cost import pairwise_cost

tslearn = pytest.importorskip("tslearn.metrics", reason="tslearn is a dev-only cross-check oracle")


def test_divergence_zero_at_identity() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 7, 3, dtype=torch.float64)
    d = softdtw_divergence(x, x, gamma=0.1)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-9), d


def test_divergence_non_negative() -> None:
    torch.manual_seed(1)
    for gamma in (0.01, 0.1, 0.5, 1.0):
        x = torch.randn(5, 9, 2, dtype=torch.float64)
        y = torch.randn(5, 9, 2, dtype=torch.float64)
        d = softdtw_divergence(x, y, gamma=gamma)
        assert float(d.min()) >= -1e-7, (gamma, d)


def test_matches_tslearn_raw_and_divergence() -> None:
    from tslearn.metrics import SoftDTWLossPyTorch

    torch.manual_seed(2)
    x = torch.randn(3, 8, 2, dtype=torch.float32)
    y = torch.randn(3, 8, 2, dtype=torch.float32)
    for gamma in (0.1, 0.5, 1.0):
        raw_mine = soft_dtw(pairwise_cost(x, y), gamma=gamma)
        raw_ts = SoftDTWLossPyTorch(gamma=gamma, normalize=False)(x, y)
        assert torch.allclose(raw_mine, raw_ts, atol=1e-4), (gamma, raw_mine, raw_ts)

        div_mine = softdtw_divergence(x, y, gamma=gamma)
        div_ts = SoftDTWLossPyTorch(gamma=gamma, normalize=True)(x, y)
        assert torch.allclose(div_mine, div_ts, atol=1e-4), (gamma, div_mine, div_ts)


def test_gradcheck_all_three_terms() -> None:
    """gradcheck the divergence (exercises SDTW(x,y), SDTW(x,x), SDTW(y,y)) and raw SDTW."""
    torch.manual_seed(3)
    x = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: softdtw_divergence(a, b, gamma=0.3).sum(), (x, y), eps=1e-6, atol=1e-4, rtol=1e-3
    )
    c = torch.randn(2, 5, 4, dtype=torch.float64).abs().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda cc: soft_dtw(cc, gamma=0.3).sum(), (c,), eps=1e-6, atol=1e-4, rtol=1e-3
    )


def test_gamma_clamp_no_overflow() -> None:
    """gamma below MIN_GAMMA is clamped; tiny gamma must not produce NaN/Inf."""
    torch.manual_seed(4)
    x = torch.randn(2, 6, 2)
    y = torch.randn(2, 6, 2)
    d = softdtw_divergence(x, y, gamma=1e-4)  # below MIN_GAMMA -> clamped to 1e-3
    assert torch.isfinite(d).all(), d
    d2 = softdtw_divergence(x, y, gamma=1e-3)
    assert torch.isfinite(d2).all(), d2


def test_raw_softdtw_can_be_negative_divergence_cannot() -> None:
    """Documents why the divergence (not raw SDTW) is the public loss."""
    torch.manual_seed(5)
    x = torch.zeros(1, 4, 1)
    raw = softdtw_divergence(x, x, gamma=1.0, raw=True)  # raw SDTW(x,x) for zero seq < 0
    div = softdtw_divergence(x, x, gamma=1.0)
    assert float(raw) < 0.0  # raw soft-DTW is not minimized at x=y
    assert torch.allclose(div, torch.zeros_like(div), atol=1e-9)  # divergence is
