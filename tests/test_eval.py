"""Empirical harness: bootstrap CI properties, auto-degrade negative fixture, paired pipeline."""

from __future__ import annotations

import numpy as np

from dtwarp.eval.bootstrap import paired_bootstrap_ci
from dtwarp.eval.harness import Tier1Config, make_synthetic_tempo_dataset, run_tier1
from dtwarp.eval.report import (
    CLAIM_ALTERNATIVE,
    CLAIM_IMPROVES,
    CLAIM_WORSE,
    build_results,
    classify_claim,
    results_to_readme_block,
)


def test_bootstrap_ci_directions() -> None:
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 0.1, size=200)  # dtwarp clearly better
    neg = rng.normal(-1.0, 0.1, size=200)  # dtwarp clearly worse
    mixed = rng.normal(0.0, 1.0, size=200)  # no effect
    assert paired_bootstrap_ci(pos).favors_dtwarp
    assert paired_bootstrap_ci(neg).ci_high < 0.0
    assert not paired_bootstrap_ci(mixed).excludes_zero


def test_autodegrade_on_ci_crossing_zero() -> None:
    """DoD #4 negative fixture: a CI that contains 0 MUST auto-degrade and forbid a 'better' claim."""
    boot = {"delta": 0.002, "ci_low": -0.4, "ci_high": 0.4, "n": 200, "n_resamples": 2000}
    c = classify_claim(boot)
    assert c["claim_tier"] == CLAIM_ALTERNATIVE
    assert c["crosses_zero"] is True
    assert c["may_say_better"] is False


def test_classify_improves_and_worse() -> None:
    improves = classify_claim({"delta": 0.5, "ci_low": 0.2, "ci_high": 0.8, "n": 200})
    assert improves["claim_tier"] == CLAIM_IMPROVES and improves["may_say_better"] is True
    worse = classify_claim({"delta": -0.5, "ci_low": -0.8, "ci_high": -0.2, "n": 200})
    assert worse["claim_tier"] == CLAIM_WORSE and worse["may_say_better"] is False


def test_underpowered_flag_blocks_better_claim() -> None:
    # strictly positive CI but very few units -> underpowered -> may_say_better False
    c = classify_claim({"delta": 0.5, "ci_low": 0.01, "ci_high": 0.99, "n": 5}, min_units=30)
    assert c["claim_tier"] == CLAIM_IMPROVES
    assert c["underpowered"] is True
    assert c["may_say_better"] is False


def test_readme_block_alternative_has_no_unsupported_claim() -> None:
    boot = {"delta": 0.0, "ci_low": -0.3, "ci_high": 0.3, "n": 200, "n_resamples": 2000}
    results = {
        "mode": "synthetic-injected-tempo",
        "blend": 0.5,
        "gamma": 0.1,
        "seeds": [0, 1],
        "n_heldout_units": 200,
        "bootstrap": boot,
        "secondary_bootstrap": {"delta": 0.1, "ci_low": 0.05, "ci_high": 0.15, "n_resamples": 2000},
        "claim": classify_claim(boot),
        "disclaimer": build_results(
            {
                "mode": "synthetic-injected-tempo",
                "blend": 0.5,
                "gamma": 0.1,
                "seeds": [0, 1],
                "n_heldout_units": 200,
                "bootstrap": boot,
            }
        )["disclaimer"],
    }
    block = results_to_readme_block(results).lower()
    assert "alternative elastic loss" in block
    assert "outperform" not in block and "state-of-the-art" not in block


def test_run_tier1_synthetic_pipeline_small() -> None:
    """End-to-end paired pipeline on a small synthetic set (fast); checks well-formed output."""
    states, chunks, pads = make_synthetic_tempo_dataset(n_samples=120, chunk=8, seed=0)
    cfg = Tier1Config(chunk=8, steps=15, seeds=(0, 1), batch=32, n_resamples=500)
    res = run_tier1("synthetic", cfg, states=states, chunks=chunks, pads=pads)
    assert res["paired"] is True
    for key in ("bootstrap", "secondary_bootstrap", "per_seed"):
        assert key in res
    b = res["bootstrap"]
    assert b["ci_low"] <= b["delta"] <= b["ci_high"]
    full = build_results(res)
    assert "claim" in full and "disclaimer" in full
    assert full["claim"]["claim_tier"] in (CLAIM_IMPROVES, CLAIM_WORSE, CLAIM_ALTERNATIVE)
