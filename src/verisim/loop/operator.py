"""Correction operators ``C`` (SPEC-2 §6.2).

Given a consultation that returns the truth ``s'`` (the oracle's transition from
the current coupled state) and the model's prediction ``ŝ'``, a correction
operator decides the post-consultation coupled state.

v0 ships all three §6.2 operators. A subtlety the spec anticipates: with a v0
oracle that returns the *full* one-step truth, ``hard_reset``, ``residual``, and
``projection`` all snap the coupled state to the same ``s'`` -- so they are
behaviorally identical on faithful horizon. Their differences are in what they
*expose*, which is where H3 will actually bite once the deferred pieces land:

  - ``hard_reset`` -- overwrite with truth; nothing exposed (the §6.2 baseline).
  - ``residual`` -- the corrected state is truth, but it records the *discrepancy*
    ``s' △ ŝ'`` (fact count). That residual is the Stage-2 online-learning signal
    (SPEC-2 §6.2): with online learning it would reduce *future* divergence; v0 has
    no Stage-2, so it is recorded as a diagnostic, not yet used.
  - ``projection`` -- the corrected state is truth (the nearest oracle-consistent
    state, since the only facts ŝ' and s' differ on are exactly the ones repaired),
    but it records the *fraction* of the state that had to be repaired (the
    pre-correction divergence) -- the per-correction cost lens of H3.

So at v0, E3's headline H_ε is identical across operators (an honest theoretical
identity, reported with CIs), and the operators are distinguished only by these
diagnostics -- which motivate the partial-verification / online-learning work that
makes H3 informative (deferred per the prime directive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from verisim.env.state import State
from verisim.metrics.divergence import divergence, state_facts


@runtime_checkable
class CorrectionOperator(Protocol):
    def correct(self, predicted: State, truth: State) -> State: ...


class HardReset:
    """``s ← s'``: snap the coupled state to truth, discarding the prediction."""

    def correct(self, predicted: State, truth: State) -> State:
        return truth


@dataclass
class Residual:
    """Corrected state is truth; records the discrepancy ``|s' △ ŝ'|`` per correction.

    The discrepancy is the Stage-2 online-learning signal (SPEC-2 §6.2). v0 logs it
    as a diagnostic (``discrepancies``) without learning from it.
    """

    discrepancies: list[int] = field(default_factory=list)

    def correct(self, predicted: State, truth: State) -> State:
        self.discrepancies.append(len(state_facts(predicted) ^ state_facts(truth)))
        return truth


@dataclass
class Projection:
    """Corrected state is truth; records the repaired *fraction* per correction.

    The fraction is the pre-correction divergence ``d(ŝ', s')`` -- how much of the
    state the consultation had to fix. Lower means the model was nearly right and
    the correction was cheap (SPEC-2 §6.2 "cheapest-faithful-horizon-per-correction").
    """

    repaired_fractions: list[float] = field(default_factory=list)

    def correct(self, predicted: State, truth: State) -> State:
        self.repaired_fractions.append(divergence(predicted, truth))
        return truth
