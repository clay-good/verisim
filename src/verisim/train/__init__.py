"""Training: Stage-1 supervised next-delta prediction (SPEC-2 §5.3)."""

from __future__ import annotations

from .dataset import Example, build_dataset, collate, examples_from_rollout
from .supervised import teacher_forced_accuracy, train_supervised

__all__ = [
    "Example",
    "build_dataset",
    "collate",
    "examples_from_rollout",
    "teacher_forced_accuracy",
    "train_supervised",
]
