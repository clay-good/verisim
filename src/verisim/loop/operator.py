"""Correction operators ``C`` (SPEC-2 §6.2).

Given a consultation that returns the truth ``s'`` (the oracle's transition from
the current coupled state) and the model's prediction ``ŝ'``, a correction
operator decides the post-consultation coupled state.

v0/M5 ships ``hard_reset`` only -- overwrite the prediction with truth (the §6.2
baseline). The ``residual`` and ``projection`` operators (RQ3 / H3) are M7.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verisim.env.state import State


@runtime_checkable
class CorrectionOperator(Protocol):
    def correct(self, predicted: State, truth: State) -> State: ...


class HardReset:
    """``s ← s'``: snap the coupled state to truth, discarding the prediction."""

    def correct(self, predicted: State, truth: State) -> State:
        return truth
