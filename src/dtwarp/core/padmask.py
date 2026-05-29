"""Padding / variable-length handling for soft-DTW on LeRobot action chunks.

LeRobot marks padded action steps with a boolean ``action_is_pad`` of shape (B, T), where
True means "padded" (right-padding past the episode boundary). The valid length is
``(~action_is_pad).sum(dim=1)`` and the valid region is the prefix ``[0:n]``.

The runtime path (``dtwarp.core.softdtw.soft_dtw`` with ``valid_lengths``) reads the soft-DTW
value at each sample's endpoint ``(n-1, n-1)``; this module additionally provides the
per-sample *sliced* divergence used as the test oracle to prove the two agree.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dtwarp.core.softdtw import softdtw_divergence

__all__ = ["valid_lengths_from_pad", "sliced_divergence"]


def valid_lengths_from_pad(
    action_is_pad: Tensor | None, batch: int, length: int, device: torch.device
) -> Tensor:
    """Per-sample valid length from a LeRobot ``action_is_pad`` mask.

    Args:
        action_is_pad: (B, T) bool, True = padded. None -> all valid (length T).
        batch, length, device: used when ``action_is_pad`` is None.

    Returns:
        (B,) long valid lengths, each clamped to >= 1.
    """
    if action_is_pad is None:
        return torch.full((batch,), length, dtype=torch.long, device=device)
    if action_is_pad.dtype != torch.bool:
        action_is_pad = action_is_pad.bool()
    n = (~action_is_pad).sum(dim=1).to(torch.long)
    return n.clamp_min(1)


def sliced_divergence(
    x: Tensor,
    y: Tensor,
    gamma: float,
    valid_lengths: Tensor,
    w_deriv: float = 0.0,
) -> Tensor:
    """Reference oracle: compute the divergence per sample on its sliced valid prefix.

    Slower (Python loop over the batch) but transparently correct; used in tests to verify
    the batched endpoint-seeded runtime matches per-sample slicing.
    """
    out = []
    for b in range(x.shape[0]):
        n = int(valid_lengths[b])
        xb = x[b : b + 1, :n]
        yb = y[b : b + 1, :n]
        out.append(softdtw_divergence(xb, yb, gamma=gamma, valid_lengths=None, w_deriv=w_deriv))
    return torch.cat(out, dim=0)
