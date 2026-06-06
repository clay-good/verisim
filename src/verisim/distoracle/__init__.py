"""The distributed oracle (SPEC-7 §5). Tier-A (the reference DES) + the tiered menu + Tier-B."""

from verisim.distoracle.base import DistOracle, DistStepResult
from verisim.distoracle.differential import (
    DistDiffRecord,
    cluster_view,
    dist_differential_step,
)
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import (
    DistDeterminismReport,
    SystemDistOracle,
    SystemDistOracleUnavailable,
)
from verisim.distoracle.tiers import TIER_COSTS, TIERS, TieredOracle, TierVerdict

__all__ = [
    "TIERS",
    "TIER_COSTS",
    "DistDeterminismReport",
    "DistDiffRecord",
    "DistOracle",
    "DistStepResult",
    "ReferenceDistOracle",
    "SystemDistOracle",
    "SystemDistOracleUnavailable",
    "TierVerdict",
    "TieredOracle",
    "cluster_view",
    "dist_differential_step",
]
