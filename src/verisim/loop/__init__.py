"""The propose-verify-correct loop: model protocol, policies, operators, runner."""

from __future__ import annotations

from .model import Model, NullModel, OracleBackedModel, UncertaintyModel
from .operator import CorrectionOperator, HardReset, Projection, Residual
from .policy import (
    ConsultationPolicy,
    DriftTriggered,
    FixedInterval,
    Never,
    StepContext,
    UncertaintyTriggered,
    fixed_interval_for_rho,
)
from .runner import budget_for_rho, ground_truth_rollout, run_rollout

__all__ = [
    "ConsultationPolicy",
    "CorrectionOperator",
    "DriftTriggered",
    "FixedInterval",
    "HardReset",
    "Model",
    "Never",
    "NullModel",
    "OracleBackedModel",
    "Projection",
    "Residual",
    "StepContext",
    "UncertaintyModel",
    "UncertaintyTriggered",
    "budget_for_rho",
    "fixed_interval_for_rho",
    "ground_truth_rollout",
    "run_rollout",
]
