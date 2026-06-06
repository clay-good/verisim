"""Inspect adapter: the distributed-cluster faithfulness benchmark as an ``inspect_ai`` task (§12).

The distributed analogue of :mod:`verisim.hosteval.inspect_task`. Packages the single-step
ground-truth labels (:func:`verisim.disteval.dist_step_labels`) as an `inspect_evals`-compatible
task so the *running-cluster* faithfulness benchmark slots into the framework labs already use. The
model under test is presented a serialized ``(cluster_state, action)`` and asked for the canonical
next cluster state; the scorer grades it with the §9.1 divergence
(:func:`verisim.disteval.grade_dist_prediction`).

This module imports ``inspect_ai`` lazily and is the only part of the distributed benchmark that
needs the optional ``[eval]`` extra; the core in :mod:`verisim.disteval.faithfulness` has no such
dependency. The ``inspect_ai`` types are unresolved without the extra, so this module is relaxed
from strict typing in ``pyproject.toml`` and its public functions return ``Any``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .faithfulness import (
    DEFAULT_DIST_SUITE,
    DistFaithfulnessSample,
    dist_step_labels,
    grade_dist_prediction,
)

_INSTRUCTION = (
    "You are a faithful model of a replicated distributed key-value service: per-node object "
    "replicas, in-flight replication messages, and a partition/crash/clock medium. Given the "
    "current STATE (canonical JSON cluster) and an ACTION (a client op, fault, or clock advance), "
    "output ONLY the canonical JSON of the next cluster state."
)


def dist_faithfulness_dataset(
    suite: Sequence[DistFaithfulnessSample] = DEFAULT_DIST_SUITE,
) -> list[Any]:
    """Build the Inspect ``Sample`` list from the single-step labels of ``suite``."""
    from inspect_ai.dataset import Sample

    samples: list[Any] = []
    for spec in suite:
        for i, label in enumerate(dist_step_labels(spec)):
            samples.append(
                Sample(
                    input=f"{_INSTRUCTION}\n\nSTATE:\n{label.state}\n\nACTION: {label.action}",
                    target=label.next_state,
                    id=f"{spec.driver}-{spec.seed}-{i}",
                    metadata={"difficulty": spec.difficulty, "action": label.action},
                )
            )
    return samples


def dist_faithfulness_task(
    suite: Sequence[DistFaithfulnessSample] = DEFAULT_DIST_SUITE,
) -> Any:
    """The Verisim distributed-cluster faithfulness benchmark as an ``inspect_ai`` Task.

    Usage (with the ``[eval]`` extra installed)::

        inspect eval verisim.disteval.inspect_task:dist_faithfulness_task --model <model>
    """
    from inspect_ai import Task
    from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
    from inspect_ai.solver import TaskState, generate

    @scorer(metrics=[accuracy(), stderr()])
    def cluster_divergence_scorer() -> Any:
        async def score(state: TaskState, target: Target) -> Any:
            completion = state.output.completion.strip()
            return Score(
                value=grade_dist_prediction(completion, target.text), answer=completion
            )

        return score

    return Task(
        dataset=dist_faithfulness_dataset(suite),
        solver=generate(),
        scorer=cluster_divergence_scorer(),
    )
