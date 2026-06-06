"""The distributed-cluster faithfulness benchmark, packaged for external evaluators (SPEC-7 §12).

The distributed analogue of :mod:`verisim.hosteval`. The framework-agnostic core
(:mod:`verisim.disteval.faithfulness`) is dependency-free and scores any
:class:`~verisim.distloop.model.DistModel`. The Inspect adapter
(:mod:`verisim.disteval.inspect_task`) is imported lazily and needs the optional ``[eval]`` extra
(``inspect_ai``); it is *not* imported here so the base benchmark stays installable without it.
"""

from __future__ import annotations

from .faithfulness import (
    DEFAULT_DIST_SUITE,
    DistFaithfulnessSample,
    DistFaithfulnessScore,
    DistStepLabel,
    applied_dist_divergence,
    dist_step_labels,
    grade_dist_prediction,
    score_dist_model,
    score_dist_suite,
)

__all__ = [
    "DEFAULT_DIST_SUITE",
    "DistFaithfulnessSample",
    "DistFaithfulnessScore",
    "DistStepLabel",
    "applied_dist_divergence",
    "dist_step_labels",
    "grade_dist_prediction",
    "score_dist_model",
    "score_dist_suite",
]
