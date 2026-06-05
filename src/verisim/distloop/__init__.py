"""The tiered propose-verify-correct loop (SPEC-7 §8; DS5).

Model-agnostic over any :class:`DistModel`; the new axis is ``π_w`` (which oracle *tier* to spend on
a consult, :mod:`verisim.distloop.tier_policy`). Records the per-step divergence trajectory (which
defines ``H_ε``) and the cumulative oracle-dollars (the §9.4 cost H17 measures).
"""

from verisim.distloop.model import DistModel, DistNullModel, DistOracleBackedModel
from verisim.distloop.runner import budget_for_rho, ground_truth_rollout, run_dist_rollout
from verisim.distloop.tier_policy import (
    EscalatingTierPolicy,
    FixedTierPolicy,
    TierPolicy,
)

__all__ = [
    "DistModel",
    "DistNullModel",
    "DistOracleBackedModel",
    "EscalatingTierPolicy",
    "FixedTierPolicy",
    "TierPolicy",
    "budget_for_rho",
    "ground_truth_rollout",
    "run_dist_rollout",
]
