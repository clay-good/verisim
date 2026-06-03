"""The HC5 composed propose-verify-correct loop (SPEC-6 §8).

The host analogue of v0's :mod:`verisim.loop` and the network :mod:`verisim.netloop`: a
model-agnostic rollout runner, baseline models, the two-mode (full / per-subsystem)
partial-observation oracle, subsystem policies ``π_w``, and correction/subsystem-filter operators.
The consultation policy
``π_c`` (§8.1) is env-agnostic and is reused directly from :mod:`verisim.loop.policy`.

This module is dependency-free and GPU-free -- the deterministic core (HC0-HC3 + this loop). The
learned ``M_θ`` (HC4) drops into the runner via the :class:`~verisim.hostloop.model.HostModel`
protocol, so the loop never knows whether it holds a baseline or the neural model.
"""

from __future__ import annotations

from .model import (
    HostModel,
    HostNullModel,
    HostOracleBackedModel,
    HostSubsystemUncertaintyModel,
    HostUncertaintyModel,
)
from .observe import PartialHostOracle, SubsystemObservation, full_bits, subsystem_bits
from .operator import (
    FullCorrection,
    HardReset,
    Projection,
    Residual,
    SubsystemFilter,
    subsystem_filter,
)
from .runner import budget_for_rho, ground_truth_rollout, run_host_rollout
from .subsystem import (
    FixedSubsystem,
    RoundRobinSubsystem,
    SubsystemPolicy,
    UncertaintySubsystem,
)

__all__ = [
    "FixedSubsystem",
    "FullCorrection",
    "HardReset",
    "HostModel",
    "HostNullModel",
    "HostOracleBackedModel",
    "HostSubsystemUncertaintyModel",
    "HostUncertaintyModel",
    "PartialHostOracle",
    "Projection",
    "Residual",
    "RoundRobinSubsystem",
    "SubsystemFilter",
    "SubsystemObservation",
    "SubsystemPolicy",
    "UncertaintySubsystem",
    "budget_for_rho",
    "full_bits",
    "ground_truth_rollout",
    "run_host_rollout",
    "subsystem_bits",
    "subsystem_filter",
]
