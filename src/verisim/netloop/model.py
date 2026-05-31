"""The network ``Model`` interface and baseline world models (SPEC-5 §6, §8).

The partial-observation propose-verify-correct loop (SPEC-5 §8) is generic over *any*
model that predicts a structured graph delta from ``(state, action)`` -- exactly as v0's
M5 loop is generic over :class:`verisim.loop.model.Model`. The learned flat ``M_θ``
(:class:`verisim.netmodel.world_model.NeuralNetworkWorldModel`, NW4) already implements
this protocol, so it drops into the loop unchanged; the message-passing/RSSM graph arm
(SPEC-5 §6.1-6.2, NW7) will too.

Two dependency-free baselines ship now (SPEC-5 §8, mirroring v0's §8 baselines), so every
loop invariant is testable with no GPU before the neural model is trained:

  - ``NetNullModel`` -- predicts the empty delta ("nothing happens"). It drifts the moment
    the oracle changes anything; the absolute floor and the proof the task is nontrivial.
  - ``NetOracleBackedModel`` -- predicts exactly the oracle's delta: a perfect model that
    never drifts even at ``ρ = 0``, framing what the neural model is *for* (cheap rollout
    between consultations) and serving as the sanity ceiling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netdelta.edits import NetDelta
from verisim.netoracle.base import NetOracle


@runtime_checkable
class NetModel(Protocol):
    """Predicts the structured graph delta an action makes to a network state (§6.1)."""

    def predict_delta(self, state: NetworkState, action: NetAction) -> NetDelta: ...


@runtime_checkable
class NetUncertaintyModel(Protocol):
    """A :class:`NetModel` that also reports its per-prediction uncertainty (§9.4).

    The signal drives the ``uncertainty``/``drift``-triggered consultation policies
    (SPEC-5 §8.1). Models without one (the baselines below) simply do not implement this
    protocol, and the runner treats their signal as ``0``.
    """

    def predict_delta_with_uncertainty(
        self, state: NetworkState, action: NetAction
    ) -> tuple[NetDelta, float]: ...


class NetNullModel:
    """Trivial predictor: predicts no change (the empty delta). The drift floor."""

    def predict_delta(self, state: NetworkState, action: NetAction) -> NetDelta:
        return []


class NetOracleBackedModel:
    """Symbolic-only: a perfect model that returns the oracle's own delta. The ceiling."""

    def __init__(self, oracle: NetOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: NetworkState, action: NetAction) -> NetDelta:
        return self._oracle.step(state, action).delta
