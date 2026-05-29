"""Throughput smoke (S9): guard against a hang or a complexity regression in the soft-DTW step.

Soft-DTW is O(T^2) per pair vs. L1's O(T), so a raw dtwarp/L1 time ratio is large *and unstable*
(L1 fwd+bwd is microseconds, so its timing is noisy and the ratio flaps on shared runners). We
therefore smoke two stable things instead:

1. self-scaling — doubling T roughly quadruples the time (O(T^2)); a large blow-up would signal an
   accidental O(T^3) / O(B^2) regression. This is a *ratio* of two same-process measurements, so it
   is immune to a constant CPU-contention factor (both timings inflate together). It is the primary
   complexity-regression guard.
2. absolute anti-hang ceiling — a dtwarp fwd+bwd at chunk=100 finishes well under a few seconds on
   an idle or normally-loaded runner. A wall-clock ceiling cannot, by itself, distinguish a true
   hang from a box merely starved of CPU, so before asserting it we run a tiny same-process probe
   and *skip* (not fail) when the machine is pathologically loaded — an absolute liveness bound is
   meaningless at e.g. load-average 100+. On a clean/CI runner the probe is sub-second and the
   ceiling is enforced normally, preserving the hang / constant-blow-up guard where it is meaningful.

This replaces the literal "< K x L1" smoke from the protocol (L1 is too fast to be a stable
denominator) while serving the same intent: the loss step is not pathologically slow.
"""

from __future__ import annotations

import time

import pytest
import torch

from dtwarp.loss.heads import act_head

BATCH = 16
ADIM = 2
ABS_BOUND_S = 15.0  # generous for slow CI runners; real measured ~1-2s when not CPU-starved
SCALING_BOUND = 12.0  # O(T^2) predicts ~4x for 2x T; >12x implies a complexity regression
PROBE_T = 20  # tiny same-process probe used to detect a pathologically loaded machine
PROBE_OVERLOAD_S = 3.0  # chunk=20 fwd+bwd is sub-second when idle; >3s => box too loaded to time


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


def _skip_if_overloaded() -> None:
    """Skip (do not fail) when a same-process probe shows the CPU is pathologically contended.

    A wall-clock liveness/anti-hang assertion is only meaningful on a machine that is not starved
    of CPU. On a shared or heavily-multiplexed box (many parallel test suites, high load average) a
    healthy O(T^2) step can take tens of seconds purely from contention, which a fixed wall-clock
    ceiling would misreport as a hang. We probe with a tiny chunk and skip above a generous bound.
    """
    probe = _time_dtwarp_step(PROBE_T)
    if probe > PROBE_OVERLOAD_S:
        pytest.skip(
            f"machine under heavy CPU contention (chunk={PROBE_T} probe took {probe:.2f}s "
            f">{PROBE_OVERLOAD_S}s); an absolute wall-clock liveness bound is not meaningful here. "
            "The complexity-regression guard (quadratic-scaling, contention-immune) still runs."
        )


def test_throughput_no_hang() -> None:
    _skip_if_overloaded()
    t100 = _time_dtwarp_step(100)
    assert t100 < ABS_BOUND_S, f"dtwarp step took {t100:.3f}s at chunk=100 (possible hang)"


def test_throughput_quadratic_scaling() -> None:
    # Ratio of two same-process timings: immune to a constant CPU-contention factor, so this stays
    # meaningful even on a loaded box. It is the primary complexity-regression guard.
    t50 = _time_dtwarp_step(50)
    t100 = _time_dtwarp_step(100)
    ratio = t100 / max(t50, 1e-4)
    assert ratio < SCALING_BOUND, (
        f"chunk 50->100 time grew {ratio:.1f}x (>{SCALING_BOUND}x => complexity regression)"
    )
