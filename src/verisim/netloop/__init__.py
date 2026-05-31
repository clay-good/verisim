"""The NW5 partial-observation propose-verify-correct loop (SPEC-5 §8).

The network analogue of v0's :mod:`verisim.loop`: a model-agnostic rollout runner, baseline
models, the two-mode (full / probe) partial-observation oracle, probe policies ``π_o``, and
correction/belief operators. The consultation policy ``π_c`` (§8.1) is env-agnostic and is
reused directly from :mod:`verisim.loop.policy`.
"""

from __future__ import annotations

from .model import NetModel, NetNullModel, NetOracleBackedModel, NetUncertaintyModel
from .observe import HostObservation, PartialNetOracle, full_bits, observe_host
from .operator import BeliefFilter, FullCorrection, HardReset, Projection, Residual, belief_filter
from .probe import ProbePolicy, RandomProbe, RoundRobinProbe
from .runner import budget_for_rho, ground_truth_rollout, run_net_rollout

__all__ = [
    "BeliefFilter",
    "FullCorrection",
    "HardReset",
    "HostObservation",
    "NetModel",
    "NetNullModel",
    "NetOracleBackedModel",
    "NetUncertaintyModel",
    "PartialNetOracle",
    "ProbePolicy",
    "Projection",
    "RandomProbe",
    "Residual",
    "RoundRobinProbe",
    "belief_filter",
    "budget_for_rho",
    "full_bits",
    "ground_truth_rollout",
    "observe_host",
    "run_net_rollout",
]
