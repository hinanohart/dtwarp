"""Claim classification (auto-degrade) + results JSON + README block generation.

The machine guard against ship-and-yank: the performance claim tier is derived ONLY from the
paired bootstrap CI. If the CI for the held-out improvement contains 0, the claim auto-degrades
to ``alternative-elastic-loss`` and the word "better" must not appear. README numbers are emitted
only from the results JSON; nothing is hand-written.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

__all__ = [
    "classify_claim",
    "build_results",
    "write_results_json",
    "results_to_readme_block",
    "hardware_info",
]

CLAIM_IMPROVES = "improves-heldout-error-on-this-dataset"
CLAIM_WORSE = "increases-heldout-error-on-this-dataset"
CLAIM_ALTERNATIVE = "alternative-elastic-loss"


def classify_claim(bootstrap: dict[str, Any], min_units: int = 30) -> dict[str, Any]:
    """Derive the claim tier and an underpowered flag purely from the bootstrap CI."""
    ci_low = float(bootstrap["ci_low"])
    ci_high = float(bootstrap["ci_high"])
    delta = float(bootstrap["delta"])
    n = int(bootstrap.get("n", 0))
    crosses_zero = ci_low <= 0.0 <= ci_high
    if crosses_zero:
        tier = CLAIM_ALTERNATIVE
    elif ci_low > 0.0:
        tier = CLAIM_IMPROVES
    else:
        tier = CLAIM_WORSE
    underpowered = n < min_units or (delta != 0.0 and (ci_high - ci_low) > 4.0 * abs(delta))
    return {
        "claim_tier": tier,
        "crosses_zero": crosses_zero,
        "underpowered": underpowered,
        "may_say_better": tier == CLAIM_IMPROVES and not underpowered,
    }


def hardware_info() -> dict[str, str]:
    import torch

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
    }


def build_results(tier1: dict[str, Any]) -> dict[str, Any]:
    """Attach claim classification, hardware, and an honest disclaimer to a run result."""
    claim = classify_claim(dict(tier1["bootstrap"]))
    if claim["claim_tier"] == CLAIM_IMPROVES and not claim["underpowered"]:
        disclaimer = (
            f"On {tier1['mode']}, dtwarp lowered held-out action error (paired bootstrap 95% CI "
            f"strictly positive). This is a dataset-specific empirical result, not a general claim."
        )
    elif claim["claim_tier"] == CLAIM_WORSE:
        disclaimer = (
            f"On {tier1['mode']}, dtwarp did NOT lower held-out action error (CI strictly negative). "
            f"Reported honestly; dtwarp is an alternative elastic loss, not an improvement here."
        )
    else:
        disclaimer = (
            f"On {tier1['mode']}, the paired bootstrap CI for the improvement includes 0: there is no "
            f"evidence dtwarp is better here. dtwarp is an alternative elastic loss, not a proven gain."
            + (" (Result is underpowered.)" if claim["underpowered"] else "")
        )
    return {**tier1, "claim": claim, "hardware": hardware_info(), "disclaimer": disclaimer}


def write_results_json(results: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return p


def results_to_readme_block(results: dict[str, Any]) -> str:
    """Render the Empirical section from a results JSON (the ONLY source of README numbers)."""
    b = results["bootstrap"]
    claim = results["claim"]
    verb = {
        CLAIM_IMPROVES: "lowered",
        CLAIM_WORSE: "raised",
        CLAIM_ALTERNATIVE: "did not significantly change",
    }[claim["claim_tier"]]
    lines = [
        f"### Tier-1 ({results['mode']})",
        "",
        f"- Paired baseline (native L1) vs. dtwarp (blend={results['blend']}, gamma={results['gamma']}), "
        f"{len(results['seeds'])} seeds, {results['n_heldout_units']} held-out units, "
        f"{b['n_resamples']} bootstrap resamples.",
        f"- Held-out plain (masked) action-MSE delta (baseline - dtwarp): "
        f"**{b['delta']:.5f}** (95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]).",
        f"- dtwarp {verb} held-out error on this neutral metric. Claim tier: `{claim['claim_tier']}`"
        + ("  _(underpowered)_" if claim["underpowered"] else "")
        + ".",
    ]
    if "secondary_bootstrap" in results:
        s = results["secondary_bootstrap"]
        lines += [
            f"- Secondary (context only — the alignment-tolerant soft-DTW divergence dtwarp optimizes): "
            f"delta **{s['delta']:.5f}** (95% CI [{s['ci_low']:.5f}, {s['ci_high']:.5f}]). "
            f"A win here is expected by construction and is NOT a neutral comparison.",
        ]
    lines += ["", f"> {results['disclaimer']}"]
    return "\n".join(lines)
