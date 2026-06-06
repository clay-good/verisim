"""The distributed oracle (SPEC-7 §5). Tier-A (the reference DES) + the tiered menu + Tier-B."""

from verisim.distoracle.base import DistOracle, DistStepResult
from verisim.distoracle.differential import (
    DistDiffRecord,
    cluster_view,
    dist_differential_step,
)
from verisim.distoracle.elle import (
    AppendObservation,
    ElleReport,
    RecoveredOrder,
    TxnObservation,
    appends_to_version_history,
    build_dsg,
    check_serializable,
    check_serializable_appends,
    recover_versions,
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
    "AppendObservation",
    "DistDeterminismReport",
    "DistDiffRecord",
    "DistOracle",
    "DistStepResult",
    "ElleReport",
    "RecoveredOrder",
    "ReferenceDistOracle",
    "SystemDistOracle",
    "SystemDistOracleUnavailable",
    "TierVerdict",
    "TieredOracle",
    "TxnObservation",
    "appends_to_version_history",
    "build_dsg",
    "check_serializable",
    "check_serializable_appends",
    "cluster_view",
    "dist_differential_step",
    "recover_versions",
]
