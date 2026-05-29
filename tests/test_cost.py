"""Ground-cost tests: squared-euclidean correctness, w_deriv=0 bit-exactness, per_channel guard."""

from __future__ import annotations

import pytest
import torch

from dtwarp.core.cost import pairwise_cost, sq_euclidean_cost


def test_sq_euclidean_matches_cdist() -> None:
    torch.manual_seed(0)
    a = torch.randn(3, 5, 4, dtype=torch.float64)
    b = torch.randn(3, 6, 4, dtype=torch.float64)
    ours = sq_euclidean_cost(a, b)
    ref = torch.cdist(a, b, p=2.0).pow(2)
    assert torch.allclose(ours, ref, atol=1e-9), (ours - ref).abs().max()


def test_w_deriv_zero_is_bit_exact() -> None:
    torch.manual_seed(1)
    a = torch.randn(2, 7, 3, dtype=torch.float64)
    b = torch.randn(2, 7, 3, dtype=torch.float64)
    base = pairwise_cost(a, b, w_deriv=0.0)
    value_only = sq_euclidean_cost(a, b)
    assert torch.equal(base, value_only)  # exact, not allclose


def test_w_deriv_changes_cost_when_nonzero() -> None:
    torch.manual_seed(2)
    a = torch.randn(2, 7, 3, dtype=torch.float64)
    b = torch.randn(2, 7, 3, dtype=torch.float64)
    base = pairwise_cost(a, b, w_deriv=0.0)
    deriv = pairwise_cost(a, b, w_deriv=0.5)
    assert not torch.allclose(base, deriv)


def test_per_channel_not_implemented() -> None:
    a = torch.randn(2, 5, 3)
    b = torch.randn(2, 5, 3)
    with pytest.raises(NotImplementedError):
        pairwise_cost(a, b, per_channel=True)


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        pairwise_cost(torch.randn(2, 5), torch.randn(2, 5))  # not 3-D
    with pytest.raises(ValueError):
        pairwise_cost(torch.randn(2, 5, 3), torch.randn(2, 5, 4))  # action_dim mismatch
