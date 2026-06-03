"""The composed-host faithfulness benchmark, packaged for external evaluators (SPEC-6 §12 / HC8).

The host analogue of :mod:`verisim.eval`. The framework-agnostic core
(:mod:`verisim.hosteval.faithfulness`) is dependency-free and scores any
:class:`~verisim.hostloop.model.HostModel`. The Inspect adapter
(:mod:`verisim.hosteval.inspect_task`) is imported lazily and needs the optional ``[eval]`` extra
(``inspect_ai``); it is *not* imported here so the base benchmark stays installable without it.
"""

from __future__ import annotations

from .faithfulness import (
    DEFAULT_HOST_SUITE,
    HostFaithfulnessSample,
    HostFaithfulnessScore,
    HostStepLabel,
    applied_host_divergence,
    grade_host_prediction,
    host_step_labels,
    score_host_model,
    score_host_suite,
)

__all__ = [
    "DEFAULT_HOST_SUITE",
    "HostFaithfulnessSample",
    "HostFaithfulnessScore",
    "HostStepLabel",
    "applied_host_divergence",
    "grade_host_prediction",
    "host_step_labels",
    "score_host_model",
    "score_host_suite",
]
