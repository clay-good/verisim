"""Faithfulness metrics: divergence, faithful horizon, run-record schema."""

from __future__ import annotations

from .aggregate import (
    ComparisonPoint,
    CurvePoint,
    aggregate_comparison,
    aggregate_curve,
    bootstrap_ci,
)
from .divergence import divergence, state_facts
from .horizon import faithful_horizon
from .record import RunRecord, read_records, write_records

__all__ = [
    "ComparisonPoint",
    "CurvePoint",
    "RunRecord",
    "aggregate_comparison",
    "aggregate_curve",
    "bootstrap_ci",
    "divergence",
    "faithful_horizon",
    "read_records",
    "state_facts",
    "write_records",
]
