"""dtwarp — soft-DTW divergence loss head for LeRobot imitation learning."""

from dtwarp.core.cost import pairwise_cost, sq_euclidean_cost
from dtwarp.core.softdtw import (
    soft_dtw,
    softdtw_divergence,
    softdtw_divergence_from_costs,
)
from dtwarp.loss.config import DTWarpLossConfig, make_dtwarp_loss
from dtwarp.loss.heads import (
    AnnealedBlendSchedule,
    act_head,
    assert_continuous_policy,
    dp_x0_aux_head,
    flow_matching_head,
    masked_base_loss,
    softdtw_bc_loss,
)
from dtwarp.loss.wrap import detect_policy_type, wrap_policy

__version__ = "0.1.0a1"

__all__ = [
    # core
    "soft_dtw",
    "softdtw_divergence",
    "softdtw_divergence_from_costs",
    "pairwise_cost",
    "sq_euclidean_cost",
    # loss heads / integration
    "softdtw_bc_loss",
    "masked_base_loss",
    "act_head",
    "flow_matching_head",
    "dp_x0_aux_head",
    "AnnealedBlendSchedule",
    "assert_continuous_policy",
    "DTWarpLossConfig",
    "make_dtwarp_loss",
    "wrap_policy",
    "detect_policy_type",
    "__version__",
]
