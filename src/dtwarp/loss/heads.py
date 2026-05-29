"""LeRobot-native loss heads: the integration layer that was missing.

Each head computes a convex blend of LeRobot's own masked base loss and the soft-DTW
divergence:

    L = (1 - blend) * base_loss + blend * dtw_term

with ``blend == 0`` reproducing LeRobot's native masked reduction **bit-for-bit** (a built-in
regression guard and a risk-free adoption path). The base reductions are transcribed from the
pinned LeRobot source (see ``anchors.json``):

* ``act_head``  -> ACT masked L1 (``modeling_act.py`` 145-150): ``(|e|*valid).sum()/(valid.sum()*A)``.
* ``flow_matching_head`` -> flow-matching masked MSE on the velocity ``u_t = noise - actions``
  (pi0 ``modeling_pi0.py`` 765-811 uses a plain ``losses.mean()``; SmolVLA masks). ``pad_aware``
  selects the masked-mean (SmolVLA) vs. plain-mean (pi0) convention.

The divergence term is length-normalized by each sample's valid length so ``blend`` is a
meaningful per-step mix; it never affects the ``blend == 0`` guarantee.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from torch import Tensor

from dtwarp.core.padmask import valid_lengths_from_pad
from dtwarp.core.softdtw import softdtw_divergence

__all__ = [
    "masked_base_loss",
    "softdtw_bc_loss",
    "act_head",
    "flow_matching_head",
    "dp_x0_aux_head",
    "AnnealedBlendSchedule",
    "assert_continuous_policy",
]


def assert_continuous_policy(policy_type: str) -> None:
    """Guard: soft-DTW assumes smooth continuous trajectories. Discrete heads are unsupported."""
    pt = policy_type.lower()
    if any(k in pt for k in ("vqbet", "vq_bet", "vq-bet", "discrete", "tokenized")):
        raise NotImplementedError(
            f"dtwarp does not support discrete/tokenized policies (got {policy_type!r}); "
            "soft-DTW assumes smooth continuous action trajectories."
        )


def masked_base_loss(
    pred: Tensor,
    target: Tensor,
    action_is_pad: Tensor | None,
    base: str = "l1",
    pad_aware: bool = True,
) -> Tensor:
    """LeRobot-faithful masked base reduction (the blend=0 target).

    ``base='l1'`` matches ACT; ``base='mse'`` matches flow-matching. ``pad_aware=False`` (or
    ``action_is_pad is None``) gives a plain mean over all elements (pi0 convention).
    """
    if base == "l1":
        err = (pred - target).abs()
    elif base == "mse":
        err = (pred - target).pow(2)
    else:
        raise ValueError(f"base must be 'l1' or 'mse', got {base!r}")
    if not pad_aware or action_is_pad is None:
        return err.mean()
    valid = (~action_is_pad.bool()).unsqueeze(-1).to(err.dtype)  # (B, T, 1)
    num_valid = valid.sum() * err.shape[-1]
    return (err * valid).sum() / num_valid.clamp_min(1)


def softdtw_bc_loss(
    pred: Tensor,
    target: Tensor,
    action_is_pad: Tensor | None = None,
    gamma: float = 0.1,
    blend: float = 0.5,
    base: str = "l1",
    pad_aware: bool = True,
    w_deriv: float = 0.0,
    per_channel: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Convex blend of LeRobot's masked base loss and the soft-DTW divergence.

    Returns ``(loss, loss_dict)``. ``blend == 0`` returns exactly the base loss (bit-exact).
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend}")
    base_loss = masked_base_loss(pred, target, action_is_pad, base=base, pad_aware=pad_aware)

    loss_dict: dict[str, float] = {base: float(base_loss.detach()), "blend_lambda": float(blend)}

    if blend == 0.0:
        loss_dict["loss"] = float(base_loss.detach())
        return base_loss, loss_dict

    b, t = pred.shape[0], pred.shape[1]
    n = valid_lengths_from_pad(action_is_pad, b, t, pred.device)
    div = softdtw_divergence(
        pred, target, gamma=gamma, valid_lengths=n, w_deriv=w_deriv, per_channel=per_channel
    )
    dtw_term = (div / n.to(div.dtype)).mean()
    loss = (1.0 - blend) * base_loss + blend * dtw_term
    loss_dict["dtw_divergence"] = float(dtw_term.detach())
    loss_dict["loss"] = float(loss.detach())
    return loss, loss_dict


def act_head(
    actions_hat: Tensor,
    actions: Tensor,
    action_is_pad: Tensor | None = None,
    gamma: float = 0.1,
    blend: float = 0.5,
    w_deriv: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """ACT drop-in: blend soft-DTW divergence into the masked L1 over ``actions_hat``."""
    return softdtw_bc_loss(
        actions_hat,
        actions,
        action_is_pad,
        gamma=gamma,
        blend=blend,
        base="l1",
        pad_aware=True,
        w_deriv=w_deriv,
    )


def flow_matching_head(
    v_t: Tensor,
    u_t: Tensor,
    action_is_pad: Tensor | None = None,
    gamma: float = 0.1,
    blend: float = 0.5,
    pad_aware: bool = False,
    w_deriv: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Flow-matching (SmolVLA / pi0) building block.

    Blends soft-DTW divergence into the masked MSE between the predicted velocity ``v_t`` and the
    target velocity ``u_t = noise - actions``. ``pad_aware=True`` uses SmolVLA's masked-mean;
    ``pad_aware=False`` (default) uses pi0's plain mean.
    """
    return softdtw_bc_loss(
        v_t, u_t, action_is_pad, gamma=gamma, blend=blend, base="mse", pad_aware=pad_aware, w_deriv=w_deriv
    )


def dp_x0_aux_head(
    x0_hat: Tensor,
    x0: Tensor,
    action_is_pad: Tensor | None = None,
    gamma: float = 0.1,
    blend: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    """EXPERIMENTAL (v0.2): auxiliary soft-DTW on a reconstructed x0 for Diffusion-Policy.

    Diffusion-Policy's native loss is on the predicted noise (epsilon), which has no temporal
    shape, so a naive soft-DTW swap is meaningless and is forbidden. The only sound route is on
    the reconstructed clean action ``x0_hat = (x_t - sqrt(1-abar)*eps_hat)/sqrt(abar)``, but
    ``sqrt(abar) -> 0`` at high-noise timesteps amplifies the error. This stub does NOT apply the
    required SNR/timestep clamp; it is not validated and is NOT a claim. Use at your own risk.
    """
    warnings.warn(
        "dp_x0_aux_head is an experimental v0.2 stub WITHOUT the required SNR/timestep clamp; "
        "it is not validated and is not part of any dtwarp performance claim.",
        stacklevel=2,
    )
    return softdtw_bc_loss(
        x0_hat, x0, action_is_pad, gamma=gamma, blend=blend, base="mse", pad_aware=action_is_pad is not None
    )


@dataclass
class AnnealedBlendSchedule:
    """Monotone blend schedule: linearly anneal ``blend`` from ``start`` to ``end`` over ``warmup`` steps.

    Always clamped to [0, 1]. ``start=end`` gives a constant blend. Used so training can begin at
    (or near) the native loss and grow the soft-DTW share as the policy stabilizes.
    """

    start: float = 0.0
    end: float = 0.5
    warmup_steps: int = 1000

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.start = float(min(max(self.start, 0.0), 1.0))
        self.end = float(min(max(self.end, 0.0), 1.0))

    def __call__(self, step: int) -> float:
        if self.warmup_steps == 0:
            return self.end
        frac = min(max(step / self.warmup_steps, 0.0), 1.0)
        return float(min(max(self.start + (self.end - self.start) * frac, 0.0), 1.0))
