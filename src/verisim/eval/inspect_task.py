"""Inspect adapter: the Verisim faithfulness benchmark as an ``inspect_ai`` task.

Packages the single-step ground-truth labels (:func:`verisim.eval.step_labels`)
as an `inspect_evals`-compatible task (SPEC-2 §15), so the benchmark slots into the
framework labs already use. The model under test is presented a serialized
`(state, action)` and asked for the canonical next state; the scorer grades it with
the §7.1 divergence (:func:`verisim.eval.grade_prediction`).

This module imports ``inspect_ai`` lazily and is the only part of the benchmark
that needs the optional ``[eval]`` extra; the core in
:mod:`verisim.eval.faithfulness` has no such dependency. The ``inspect_ai`` types
are unresolved without the extra, so this module is relaxed from strict typing in
``pyproject.toml`` and its public functions return ``Any``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .faithfulness import DEFAULT_SUITE, FaithfulnessSample, grade_prediction, step_labels

_INSTRUCTION = (
    "You are a faithful model of a POSIX shell + filesystem. Given the current "
    "STATE (canonical JSON) and an ACTION (a shell command), output ONLY the "
    "canonical JSON of the next state."
)


def faithfulness_dataset(suite: Sequence[FaithfulnessSample] = DEFAULT_SUITE) -> list[Any]:
    """Build the Inspect ``Sample`` list from the single-step labels of ``suite``."""
    from inspect_ai.dataset import Sample

    samples: list[Any] = []
    for spec in suite:
        for i, label in enumerate(step_labels(spec)):
            samples.append(
                Sample(
                    input=f"{_INSTRUCTION}\n\nSTATE:\n{label.state}\n\nACTION: {label.action}",
                    target=label.next_state,
                    id=f"{spec.driver}-{spec.seed}-{i}",
                    metadata={"difficulty": spec.difficulty, "action": label.action},
                )
            )
    return samples


def faithfulness_task(suite: Sequence[FaithfulnessSample] = DEFAULT_SUITE) -> Any:
    """The Verisim single-step faithfulness benchmark as an ``inspect_ai`` Task.

    Usage (with the ``[eval]`` extra installed)::

        inspect eval verisim.eval.inspect_task:faithfulness_task --model <model>
    """
    from inspect_ai import Task
    from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
    from inspect_ai.solver import TaskState, generate

    @scorer(metrics=[accuracy(), stderr()])
    def divergence_scorer() -> Any:
        async def score(state: TaskState, target: Target) -> Any:
            completion = state.output.completion.strip()
            return Score(value=grade_prediction(completion, target.text), answer=completion)

        return score

    return Task(
        dataset=faithfulness_dataset(suite),
        solver=generate(),
        scorer=divergence_scorer(),
    )
