"""Paired bootstrap confidence interval for the baseline-vs-dtwarp improvement.

Given per-unit paired deltas (``baseline_error - dtwarp_error`` on the SAME held-out unit, same
seed, same data — only the loss differs), a positive mean delta means dtwarp reduced error. The
paired design removes between-run variance so the CI is not inflated by unrelated noise (the
common failure that manufactures a spurious "improvement"). If the CI contains 0, there is no
evidence of improvement and the claim must auto-degrade.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["paired_bootstrap_ci", "BootstrapResult"]


class BootstrapResult:
    """Mean paired delta and a percentile bootstrap CI. Positive delta => dtwarp lowered error."""

    def __init__(self, mean: float, ci_low: float, ci_high: float, n: int, n_resamples: int) -> None:
        self.mean = mean
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.n = n
        self.n_resamples = n_resamples

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def favors_dtwarp(self) -> bool:
        """CI strictly above 0 (dtwarp reduced error with the whole interval positive)."""
        return self.ci_low > 0.0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "delta": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "excludes_zero": self.excludes_zero,
            "favors_dtwarp": self.favors_dtwarp,
        }


def paired_bootstrap_ci(
    deltas: NDArray[np.float64], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> BootstrapResult:
    """Percentile bootstrap CI on the mean of paired deltas.

    Args:
        deltas: 1-D array of per-unit paired deltas (baseline_error - dtwarp_error).
        n_resamples: number of bootstrap resamples (>= 1000 recommended).
        alpha: two-sided significance (0.05 -> 95% CI).
        seed: RNG seed for reproducibility.
    """
    deltas = np.asarray(deltas, dtype=np.float64).ravel()
    n = deltas.shape[0]
    if n < 2:
        raise ValueError(f"need >= 2 paired deltas for a bootstrap CI, got {n}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = deltas[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return BootstrapResult(float(deltas.mean()), lo, hi, n, n_resamples)
