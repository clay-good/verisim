"""The faithfulness benchmark, packaged for external evaluators (SPEC-2 §15).

The framework-agnostic core (:mod:`verisim.eval.faithfulness`) is dependency-free
and scores any ``verisim.loop.Model``. The Inspect adapter
(:mod:`verisim.eval.inspect_task`) is imported lazily and needs the optional
``[eval]`` extra (``inspect_ai``); it is *not* imported here so the base benchmark
stays installable without it.
"""

from __future__ import annotations

from .faithfulness import (
    DEFAULT_SUITE,
    FaithfulnessSample,
    FaithfulnessScore,
    StepLabel,
    applied_divergence,
    grade_prediction,
    score_model,
    score_suite,
    step_labels,
)

__all__ = [
    "DEFAULT_SUITE",
    "FaithfulnessSample",
    "FaithfulnessScore",
    "StepLabel",
    "applied_divergence",
    "grade_prediction",
    "score_model",
    "score_suite",
    "step_labels",
]
