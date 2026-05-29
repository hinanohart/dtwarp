"""Ground-cost matrices for soft-DTW.

Default = joint squared-euclidean over the action dimension. An optional first-difference
(velocity) channel sharpens alignment of motion onsets; its weight ``w_deriv`` defaults to
0.0, in which case the returned cost is bit-identical to the value-only cost. Per-channel
(independent-warp) cost is a v0.2 experimental feature and currently raises.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["pairwise_cost", "sq_euclidean_cost"]


def sq_euclidean_cost(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise squared-euclidean cost. a: (B, Ta, A), b: (B, Tb, A) -> (B, Ta, Tb)."""
    a2 = (a * a).sum(dim=-1)  # (B, Ta)
    b2 = (b * b).sum(dim=-1)  # (B, Tb)
    ab = torch.einsum("bia,bja->bij", a, b)  # (B, Ta, Tb)
    return a2[:, :, None] + b2[:, None, :] - 2.0 * ab


def _velocity(x: Tensor) -> Tensor:
    """First temporal difference with a leading zero so shape is preserved. (B, T, A)."""
    v = torch.zeros_like(x)
    v[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return v


def pairwise_cost(
    x: Tensor,
    y: Tensor,
    w_deriv: float = 0.0,
    per_channel: bool = False,
) -> Tensor:
    """Ground cost between two sequences.

    Args:
        x, y: (B, Tx, A) and (B, Ty, A).
        w_deriv: weight on a first-difference (velocity) cost channel. 0.0 (default) returns
            the value-only cost bit-for-bit.
        per_channel: experimental independent-per-dimension warping (v0.2; raises if set).

    Returns:
        (B, Tx, Ty) cost matrix.
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"x, y must be (B, T, A); got {tuple(x.shape)}, {tuple(y.shape)}")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(f"batch and action_dim must match; got {tuple(x.shape)}, {tuple(y.shape)}")
    if per_channel:
        raise NotImplementedError(
            "per_channel (independent-per-dimension warping) is a v0.2 experimental feature; "
            "v0.1 uses joint squared-euclidean cost."
        )
    cost = sq_euclidean_cost(x, y)
    if w_deriv != 0.0:
        cost = cost + w_deriv * sq_euclidean_cost(_velocity(x), _velocity(y))
    return cost
