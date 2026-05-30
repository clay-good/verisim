"""The ``Model`` interface and baseline world models (SPEC-2 §5, §8).

The propose-verify-correct loop is generic over *any* model that predicts a
structured delta from ``(state, action)``. The learned transformer ``M_θ`` (M4)
will implement this same protocol, so it drops into the loop unchanged.

Two baseline models are provided now (SPEC-2 §8), which need no training and let
us test every loop invariant before the neural model exists:

  - ``NullModel`` (b3, trivial predictor): predicts the empty delta -- "nothing
    happens." The coupled state stays put, so it drifts the moment ground truth
    changes anything. The absolute floor; sanity that the task is nontrivial.
  - ``OracleBackedModel`` (b2, symbolic-only): predicts exactly the oracle's
    delta -- a perfect model. Used to frame what the neural model is *for* (cheap
    rollout between consultations) and as a sanity ceiling: a perfect model never
    drifts, even at ``ρ = 0``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verisim.delta.edits import Delta
from verisim.env.action import Action
from verisim.env.state import State
from verisim.oracle.base import Oracle


@runtime_checkable
class Model(Protocol):
    """Predicts the structured delta an action makes to a state (SPEC-2 §5.1)."""

    def predict_delta(self, state: State, action: Action) -> Delta: ...


@runtime_checkable
class UncertaintyModel(Protocol):
    """A :class:`Model` that also reports its per-prediction uncertainty (§7.2).

    The signal drives the ``uncertainty``/``drift``-triggered consultation policies
    (SPEC-2 §6.1). Models without one (the baselines below) simply do not implement
    this protocol, and the runner treats their signal as ``0``.
    """

    def predict_delta_with_uncertainty(
        self, state: State, action: Action
    ) -> tuple[Delta, float]: ...


class NullModel:
    """b3 trivial predictor: predicts no change (the empty delta)."""

    def predict_delta(self, state: State, action: Action) -> Delta:
        return []


class OracleBackedModel:
    """b2 symbolic-only: a perfect model that returns the oracle's own delta."""

    def __init__(self, oracle: Oracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: State, action: Action) -> Delta:
        return self._oracle.step(state, action).delta
