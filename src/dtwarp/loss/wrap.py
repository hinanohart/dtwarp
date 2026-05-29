"""``wrap_policy``: drop a dtwarp loss into a LeRobot policy without forking LeRobot.

ACT is wrapped fully: the policy's ``forward`` is replaced so the masked L1 over ``actions_hat``
becomes the convex soft-DTW-divergence blend (the VAE KLD term, if any, is preserved). Flow
policies (SmolVLA / pi0) compute their MSE deep inside the flow model on the velocity
``u_t = noise - actions``; rather than auto-monkeypatch a vision-language model, dtwarp ships
``flow_matching_head`` as a tested building block and ``wrap_policy`` points you to it. Diffusion
(epsilon) and discrete (VQ-BeT) policies are explicitly unsupported and raise.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from dtwarp.loss.config import ACT_POLICIES, FLOW_POLICIES, DTWarpLossConfig
from dtwarp.loss.heads import act_head, assert_continuous_policy

__all__ = ["wrap_policy", "detect_policy_type"]

try:  # LeRobot constant; fall back to the literal if LeRobot is not installed
    from lerobot.constants import ACTION as _ACTION
except Exception:  # pragma: no cover - exercised only without lerobot
    _ACTION = "action"


def detect_policy_type(policy: Any, override: str | None = None) -> str:
    """Best-effort policy-type detection (explicit ``override`` wins)."""
    if override is not None:
        return override.lower()
    cfg = getattr(policy, "config", None)
    for attr in ("type", "policy_type"):
        val = getattr(cfg, attr, None)
        if isinstance(val, str):
            return val.lower()
    name = type(policy).__name__.lower()
    for known in (*ACT_POLICIES, *FLOW_POLICIES, "diffusion", "vqbet"):
        if known in name:
            return known
    return name


def wrap_policy(
    policy: Any,
    loss: str = "softdtw_divergence",
    blend: float = 0.5,
    gamma: float = 0.1,
    w_deriv: float = 0.0,
    per_channel: bool = False,
    pad_aware: bool = True,
    policy_type: str | None = None,
    action_key: str | None = None,
) -> Any:
    """Replace a LeRobot policy's loss with the dtwarp soft-DTW divergence blend (in place).

    Returns the same ``policy`` object with ``forward`` patched (ACT). ``blend=0`` reproduces the
    native loss bit-for-bit. Raises for flow / diffusion / discrete policies with guidance.
    """
    if loss != "softdtw_divergence":
        raise ValueError(f"unknown loss {loss!r}; only 'softdtw_divergence' is supported")
    cfg = DTWarpLossConfig(
        loss=loss, blend=blend, gamma=gamma, w_deriv=w_deriv, per_channel=per_channel, pad_aware=pad_aware
    )
    pt = detect_policy_type(policy, policy_type)
    assert_continuous_policy(pt)
    action_key = action_key or _ACTION

    if pt in ACT_POLICIES:
        return _wrap_act(policy, cfg, action_key)

    if pt in FLOW_POLICIES:
        raise NotImplementedError(
            f"wrap_policy does not auto-patch flow policy {pt!r} (its MSE is computed inside the "
            "flow model on the velocity u_t = noise - actions, which would require monkeypatching "
            "a VLM). Use dtwarp.flow_matching_head(v_t, u_t, action_is_pad, ...) as a building "
            "block at the loss site instead."
        )

    if "diffusion" in pt:
        raise NotImplementedError(
            "Diffusion-Policy's loss is on the predicted noise (epsilon), which has no temporal "
            "shape; a naive soft-DTW swap is forbidden. See dp_x0_aux_head (experimental v0.2)."
        )

    raise ValueError(f"unsupported policy_type {pt!r} for wrap_policy")


def _wrap_act(policy: Any, cfg: DTWarpLossConfig, action_key: str) -> Any:
    original_model = policy.model

    def dtwarp_forward(batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        actions_hat, latent = original_model(batch)
        loss, loss_dict = act_head(
            actions_hat,
            batch[action_key],
            batch.get("action_is_pad"),
            gamma=cfg.gamma,
            blend=cfg.blend,
            w_deriv=cfg.w_deriv,
        )
        policy_cfg = getattr(policy, "config", None)
        use_vae = bool(getattr(policy_cfg, "use_vae", False))
        if use_vae and isinstance(latent, (tuple, list)) and latent[0] is not None:
            mu_hat, log_sigma_x2_hat = latent
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = float(mean_kld.detach())
            loss = loss + mean_kld * float(getattr(policy_cfg, "kl_weight", 1.0))
            loss_dict["loss"] = float(loss.detach())
        return loss, loss_dict

    policy.forward = dtwarp_forward
    policy._dtwarp_wrapped = True
    return policy
