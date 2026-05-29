"""The propose-verify-correct loop: model protocol, policies, operators, runner."""

from __future__ import annotations

from .model import Model, NullModel, OracleBackedModel
from .operator import CorrectionOperator, HardReset
from .policy import ConsultationPolicy, FixedInterval, Never, fixed_interval_for_rho
from .runner import budget_for_rho, ground_truth_rollout, run_rollout

__all__ = [
    "ConsultationPolicy",
    "CorrectionOperator",
    "FixedInterval",
    "HardReset",
    "Model",
    "Never",
    "NullModel",
    "OracleBackedModel",
    "budget_for_rho",
    "fixed_interval_for_rho",
    "ground_truth_rollout",
    "run_rollout",
]
