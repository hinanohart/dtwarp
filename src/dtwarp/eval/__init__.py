"""dtwarp empirical harness (Tier-1 / Tier-2): paired baseline-vs-dtwarp + bootstrap CI."""

from dtwarp.eval.bootstrap import BootstrapResult, paired_bootstrap_ci
from dtwarp.eval.harness import (
    Tier1Config,
    load_pusht_chunks,
    make_synthetic_tempo_dataset,
    run_tier1,
)
from dtwarp.eval.report import (
    build_results,
    classify_claim,
    results_to_readme_block,
    write_results_json,
)

__all__ = [
    "BootstrapResult",
    "paired_bootstrap_ci",
    "Tier1Config",
    "make_synthetic_tempo_dataset",
    "load_pusht_chunks",
    "run_tier1",
    "build_results",
    "classify_claim",
    "results_to_readme_block",
    "write_results_json",
]
