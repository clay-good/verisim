"""Training: Stage-1 supervised + Stage-2 RLVR next-delta prediction (SPEC-2 §5.3)."""

from __future__ import annotations

from .dataset import Example, build_dataset, collate, examples_from_rollout
from .rlvr import RLVRStats, sample_delta_with_logprob, train_rlvr
from .supervised import teacher_forced_accuracy, train_supervised

__all__ = [
    "Example",
    "RLVRStats",
    "build_dataset",
    "collate",
    "examples_from_rollout",
    "sample_delta_with_logprob",
    "teacher_forced_accuracy",
    "train_rlvr",
    "train_supervised",
]
