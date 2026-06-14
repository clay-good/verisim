"""Detect-and-halt safety breaker (OpenSpec ``add-trajectory-oscillation-breaker``).

The fifth link in the Verisim ↔ OpenLore prototype chain (findings doc §4): a circuit breaker
that watches the planning loop's own trajectory for oscillation / repetitive-edit loops and, on
breach, freezes the loop and drops the in-memory speculative rollout — then *recommends* a
human-confirmed workspace rollback, never an autonomous git/file reset.

See :mod:`verisim.safety.breaker` for the metric, tiers, and the human-gated rollback contract.
"""

from __future__ import annotations

from .breaker import (
    SNAPSHOT_DIRNAME,
    TIER_CRITICAL,
    TIER_DEGRADED,
    TIER_OK,
    BreakerConfig,
    BreakerError,
    Checkpoint,
    OscillationMetric,
    PlanningTransition,
    RollbackNotConfirmed,
    RollbackRecommendation,
    SpeculativeRollout,
    TrajectoryBreaker,
    classify,
    compute_metric,
    evaluate,
)

__all__ = [
    "SNAPSHOT_DIRNAME",
    "TIER_CRITICAL",
    "TIER_DEGRADED",
    "TIER_OK",
    "BreakerConfig",
    "BreakerError",
    "Checkpoint",
    "OscillationMetric",
    "PlanningTransition",
    "RollbackNotConfirmed",
    "RollbackRecommendation",
    "SpeculativeRollout",
    "TrajectoryBreaker",
    "classify",
    "compute_metric",
    "evaluate",
]
