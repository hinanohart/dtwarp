"""Throughput smoke (S9): guard against a hang or a complexity regression in the soft-DTW step.

Soft-DTW is O(T^2) per pair vs. L1's O(T), so a raw dtwarp/L1 time ratio is large *and unstable*
(L1 fwd+bwd is microseconds, so its timing is noisy and the ratio flaps on shared runners). We
therefore smoke two stable things instead:

1. absolute anti-hang bound — a dtwarp fwd+bwd at chunk=100 finishes well under a few seconds;
2. self-scaling — doubling T roughly quadruples the time (O(T^2)); a large blow-up would signal an
   accidental O(T^3) / O(B^2) regression.

This replaces the literal "< K x L1" smoke from the protocol (L1 is too fast to be a stable
denominator) while serving the same intent: the loss step is not pathologically slow.
"""

from __future__ import annotations

import time

import torch

from dtwarp.loss.heads import act_head

BATCH = 16
ADIM = 2
ABS_BOUND_S = 15.0  # generous for slow CI runners; real measured ~1-2s
SCALING_BOUND = 12.0  # O(T^2) predicts ~4x for 2x T; >12x implies a complexity regression


def _time_dtwarp_step(t_len: int, repeats: int = 3) -> float:
    torch.manual_seed(0)
    target = torch.randn(BATCH, t_len, ADIM)
    best = float("inf")
    for _ in range(repeats):
        x = torch.randn(BATCH, t_len, ADIM, requires_grad=True)
        t0 = time.perf_counter()
        loss, _ = act_head(x, target, None, blend=0.5, gamma=0.1)
        loss.backward()
        best = min(best, time.perf_counter() - t0)
    return best


def test_throughput_no_hang() -> None:
    t100 = _time_dtwarp_step(100)
    assert t100 < ABS_BOUND_S, f"dtwarp step took {t100:.3f}s at chunk=100 (possible hang)"


def test_throughput_quadratic_scaling() -> None:
    t50 = _time_dtwarp_step(50)
    t100 = _time_dtwarp_step(100)
    ratio = t100 / max(t50, 1e-4)
    assert ratio < SCALING_BOUND, (
        f"chunk 50->100 time grew {ratio:.1f}x (>{SCALING_BOUND}x => complexity regression)"
    )
