"""The distributed world (SPEC-7): replicated services, transactions, consensus.

DS0 increment 1 -- the deterministic core: the cluster state (:mod:`verisim.dist.state`), the action
grammar (:mod:`verisim.dist.action`), the structured delta + ``apply`` (:mod:`verisim.dist.delta`),
canonical serialization (:mod:`verisim.dist.serialize`), and the curriculum config
(:mod:`verisim.dist.config`). The Tier-A reference oracle lives in :mod:`verisim.distoracle`.
"""

from verisim.dist.action import DistAction, DistParseError, parse_dist_action
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig, scaled_dist_config
from verisim.dist.delta import DistDelta, DistEdit, apply
from verisim.dist.state import DistributedState

__all__ = [
    "DEFAULT_DIST_CONFIG",
    "DistAction",
    "DistConfig",
    "DistDelta",
    "DistEdit",
    "DistParseError",
    "DistributedState",
    "apply",
    "parse_dist_action",
    "scaled_dist_config",
]
