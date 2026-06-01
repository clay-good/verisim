"""Network data generation: drivers, rollouts, and the SPEC-8 oracle-grounding factory (OG1/OG2)."""

from .drivers import NET_DRIVERS, NetDriver
from .generate import generate_net_trajectory
from .grounding import OracleTargets, fact_hosts, is_decidable, oracle_targets
from .negatives import (
    counterfactual_branches,
    enumerate_actions,
    is_hard_negative,
    one_edit_negatives,
)

__all__ = [
    "NET_DRIVERS",
    "NetDriver",
    "OracleTargets",
    "counterfactual_branches",
    "enumerate_actions",
    "fact_hosts",
    "generate_net_trajectory",
    "is_decidable",
    "is_hard_negative",
    "one_edit_negatives",
    "oracle_targets",
]
