"""Loss heads: blend=0 bit-exactness vs. LeRobot reductions, schedule, guards, wrap_policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from dtwarp import (
    AnnealedBlendSchedule,
    act_head,
    dp_x0_aux_head,
    flow_matching_head,
    make_dtwarp_loss,
    wrap_policy,
)


# --- independent inline transcriptions of LeRobot reductions (from anchors.json) ---
def _lerobot_act_l1(pred: torch.Tensor, target: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    abs_err = F.l1_loss(target, pred, reduction="none")
    valid_mask = ~pad.unsqueeze(-1)
    num_valid = valid_mask.sum() * abs_err.shape[-1]
    return (abs_err * valid_mask).sum() / num_valid.clamp_min(1)


def _flow_mse_plain(v_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(u_t, v_t, reduction="none").mean()


def _flow_mse_masked(v_t: torch.Tensor, u_t: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    losses = F.mse_loss(u_t, v_t, reduction="none")
    valid = (~pad).unsqueeze(-1).to(losses.dtype)
    num_valid = valid.sum() * losses.shape[-1]
    return (losses * valid).sum() / num_valid.clamp_min(1)


def test_act_head_blend0_bit_exact() -> None:
    torch.manual_seed(0)
    pred = torch.randn(4, 7, 3, dtype=torch.float64)
    target = torch.randn(4, 7, 3, dtype=torch.float64)
    pad = torch.zeros(4, 7, dtype=torch.bool)
    pad[0, 5:] = True
    pad[2, 6:] = True
    loss, d = act_head(pred, target, pad, blend=0.0)
    ref = _lerobot_act_l1(pred, target, pad)
    assert torch.allclose(loss, ref, atol=1e-12), (loss - ref).abs().max()
    assert d["blend_lambda"] == 0.0


def test_flow_head_blend0_bit_exact_both_conventions() -> None:
    torch.manual_seed(1)
    v = torch.randn(3, 8, 2, dtype=torch.float64)
    u = torch.randn(3, 8, 2, dtype=torch.float64)
    pad = torch.zeros(3, 8, dtype=torch.bool)
    pad[1, 6:] = True
    # pi0: plain mean
    loss_pi0, _ = flow_matching_head(v, u, pad, blend=0.0, pad_aware=False)
    assert torch.allclose(loss_pi0, _flow_mse_plain(v, u), atol=1e-12)
    # smolvla: masked mean
    loss_smol, _ = flow_matching_head(v, u, pad, blend=0.0, pad_aware=True)
    assert torch.allclose(loss_smol, _flow_mse_masked(v, u, pad), atol=1e-12)


def test_blend_changes_loss_and_is_convex_endpoints() -> None:
    torch.manual_seed(2)
    pred = torch.randn(2, 6, 2, dtype=torch.float64)
    target = torch.randn(2, 6, 2, dtype=torch.float64)
    l0, _ = act_head(pred, target, None, blend=0.0)
    l1, _ = act_head(pred, target, None, blend=1.0)
    lh, _ = act_head(pred, target, None, blend=0.5)
    assert not torch.allclose(l0, l1)
    # blend=0.5 lies strictly between the two pure terms
    lo, hi = sorted([float(l0), float(l1)])
    assert lo - 1e-9 <= float(lh) <= hi + 1e-9


def test_blend_out_of_range_raises() -> None:
    pred = torch.randn(1, 4, 2)
    with pytest.raises(ValueError):
        act_head(pred, pred, None, blend=1.5)


def test_annealed_blend_schedule_monotone_clamped() -> None:
    sched = AnnealedBlendSchedule(start=0.0, end=0.8, warmup_steps=100)
    vals = [sched(s) for s in range(-10, 200, 10)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:], strict=False))  # monotone non-decreasing
    assert sched(0) == pytest.approx(0.0)
    assert sched(100) == pytest.approx(0.8)
    assert sched(10_000) == pytest.approx(0.8)  # clamped
    assert AnnealedBlendSchedule(start=0.3, end=0.3, warmup_steps=0)(0) == pytest.approx(0.3)


def test_dp_x0_aux_warns() -> None:
    x0 = torch.randn(2, 5, 2)
    with pytest.warns(UserWarning, match="experimental"):
        dp_x0_aux_head(x0, x0, blend=0.5)


def test_make_dtwarp_loss_dispatch_and_guards() -> None:
    assert callable(make_dtwarp_loss("act"))
    assert callable(make_dtwarp_loss("smolvla"))
    assert callable(make_dtwarp_loss("pi0"))
    with pytest.raises(NotImplementedError):
        make_dtwarp_loss("diffusion")
    with pytest.raises(NotImplementedError):
        make_dtwarp_loss("vqbet")
    with pytest.raises(ValueError):
        make_dtwarp_loss("totally_unknown_policy")


# --- wrap_policy on a mock ACT (real ACT is exercised in the Tier-1 harness, S6) ---
class _MockACTModel:
    def __init__(self, actions_hat: torch.Tensor) -> None:
        self._ah = actions_hat

    def __call__(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, tuple[None, None]]:
        return self._ah, (None, None)


def _mock_act_policy(actions_hat: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        model=_MockACTModel(actions_hat),
        config=SimpleNamespace(type="act", use_vae=False, kl_weight=10.0),
    )


def test_wrap_policy_act_blend0_matches_native() -> None:
    torch.manual_seed(3)
    pred = torch.randn(3, 6, 2, dtype=torch.float64)
    target = torch.randn(3, 6, 2, dtype=torch.float64)
    pad = torch.zeros(3, 6, dtype=torch.bool)
    pad[0, 4:] = True
    policy = _mock_act_policy(pred)
    wrap_policy(policy, blend=0.0, action_key="action")
    assert getattr(policy, "_dtwarp_wrapped", False)
    loss, _ = policy.forward({"action": target, "action_is_pad": pad})
    assert torch.allclose(loss, _lerobot_act_l1(pred, target, pad), atol=1e-12)


def test_wrap_policy_flow_and_diffusion_raise() -> None:
    flow = SimpleNamespace(model=None, config=SimpleNamespace(type="smolvla"))
    with pytest.raises(NotImplementedError, match="flow_matching_head"):
        wrap_policy(flow)
    diff = SimpleNamespace(model=None, config=SimpleNamespace(type="diffusion"))
    with pytest.raises(NotImplementedError):
        wrap_policy(diff)
    vq = SimpleNamespace(model=None, config=SimpleNamespace(type="vqbet"))
    with pytest.raises(NotImplementedError):
        wrap_policy(vq)
