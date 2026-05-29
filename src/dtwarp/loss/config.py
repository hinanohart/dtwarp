"""Config dataclass + factory mapping a LeRobot policy type to the right dtwarp loss head."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from dtwarp.loss.heads import (
    AnnealedBlendSchedule,
    act_head,
    assert_continuous_policy,
    flow_matching_head,
)

__all__ = ["DTWarpLossConfig", "make_dtwarp_loss", "ACT_POLICIES", "FLOW_POLICIES"]

ACT_POLICIES = frozenset({"act"})
FLOW_POLICIES = frozenset({"smolvla", "pi0", "pi05", "pi0_fast"})
FORBIDDEN_POLICIES = frozenset({"diffusion"})

LossHead = Callable[..., tuple[Tensor, dict[str, float]]]


@dataclass
class DTWarpLossConfig:
    """Configuration for a dtwarp loss head (LeRobot draccus-style dataclass).

    ``blend == 0.0`` reproduces the native LeRobot loss bit-for-bit. ``schedule``, if set,
    overrides ``blend`` per training step (anneal soft-DTW share up from the native loss).
    """

    loss: str = "softdtw_divergence"
    blend: float = 0.5
    gamma: float = 0.1
    w_deriv: float = 0.0
    per_channel: bool = False
    pad_aware: bool = True
    schedule: AnnealedBlendSchedule | None = None

    def blend_at(self, step: int | None = None) -> float:
        if self.schedule is not None and step is not None:
            return self.schedule(step)
        return self.blend


def make_dtwarp_loss(policy_type: str, config: DTWarpLossConfig | None = None) -> LossHead:
    """Return the loss head appropriate for ``policy_type``.

    * ACT          -> ``act_head`` (drop-in for the masked L1).
    * SmolVLA/pi0  -> ``flow_matching_head`` (velocity ``u_t = noise - actions``).
    * Diffusion    -> raises (epsilon target has no temporal shape; naive swap forbidden).
    * VQ-BeT/discrete -> raises (non-smooth trajectory).
    """
    cfg = config or DTWarpLossConfig()
    assert_continuous_policy(policy_type)
    pt = policy_type.lower()

    if pt in ACT_POLICIES:

        def _act(
            actions_hat: Tensor, actions: Tensor, action_is_pad: Tensor | None = None, step: int | None = None
        ) -> tuple[Tensor, dict[str, float]]:
            return act_head(
                actions_hat,
                actions,
                action_is_pad,
                gamma=cfg.gamma,
                blend=cfg.blend_at(step),
                w_deriv=cfg.w_deriv,
            )

        return _act

    if pt in FLOW_POLICIES:
        pad_aware = pt != "pi0"  # pi0 uses a plain mean; SmolVLA-family mask

        def _flow(
            v_t: Tensor, u_t: Tensor, action_is_pad: Tensor | None = None, step: int | None = None
        ) -> tuple[Tensor, dict[str, float]]:
            return flow_matching_head(
                v_t,
                u_t,
                action_is_pad,
                gamma=cfg.gamma,
                blend=cfg.blend_at(step),
                pad_aware=pad_aware,
                w_deriv=cfg.w_deriv,
            )

        return _flow

    if pt in FORBIDDEN_POLICIES:
        raise NotImplementedError(
            f"dtwarp forbids a naive soft-DTW swap for {policy_type!r}: its loss is on the "
            "predicted noise (epsilon), which has no temporal shape. See dp_x0_aux_head (v0.2)."
        )

    raise ValueError(f"unknown / unsupported policy_type {policy_type!r} for dtwarp")
