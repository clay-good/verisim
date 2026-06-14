"""Headless CD pipeline (OpenSpec ``add-headless-cd-pipeline``).

The sixth and final link in the Verisim ↔ OpenLore prototype chain (findings doc §9.1): one local,
network-isolated headless entry point that chains the whole stack — intent → speculative simulate →
runtime trace → invariant evaluation → synthesized-edge feedback → oscillation breaker → prepared
delivery — and **stops before the one irreversible step**. The run prepares a content-sealed report
and an unpushed commit/patch on the de-fanged fixture; commit/push/deploy require explicit human
confirmation, and even confirmed cannot reach the original repository.

See :mod:`verisim.pipeline.model` for the typed run model and :mod:`verisim.pipeline.cd` for the
ordered, fail-safe orchestrator and the ``verisim-cd`` console entry point.
"""

from __future__ import annotations

from .cd import RUN_DIRNAME, PipelineError, main, run_pipeline
from .model import (
    REPORT_SCHEMA_VERSION,
    SOURCE_RUNTIME,
    SOURCE_STATIC,
    STAGE_FAILED,
    STAGE_HALTED,
    STAGE_OK,
    STAGE_SKIPPED,
    ArchInvariant,
    Intent,
    InvariantFinding,
    PreparedDelivery,
    RunReport,
    StageResult,
)

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "RUN_DIRNAME",
    "SOURCE_RUNTIME",
    "SOURCE_STATIC",
    "STAGE_FAILED",
    "STAGE_HALTED",
    "STAGE_OK",
    "STAGE_SKIPPED",
    "ArchInvariant",
    "Intent",
    "InvariantFinding",
    "PipelineError",
    "PreparedDelivery",
    "RunReport",
    "StageResult",
    "main",
    "run_pipeline",
]
