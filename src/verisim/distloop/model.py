"""The distributed ``Model`` interface and baseline world models (SPEC-7 §6, §8; DS5).

The tiered propose-verify-correct loop (SPEC-7 §8) is generic over *any* model that predicts a
structured ``DistDelta`` from ``(state, action)`` -- as v0's M5 loop, the network's NW5 loop,
and the host's HC5 loop are generic over their model protocols. The learned `M_θ` (DS4) will drop
into the loop unchanged via this protocol.

Two dependency-free baselines ship now (mirroring every prior world's §8 baselines), so every loop
invariant is testable with no GPU before the learned model:

  - ``DistNullModel`` -- predicts the empty delta ("nothing happens"). It drifts the moment the
    oracle changes anything; the absolute floor and the proof the task is nontrivial.
  - ``DistOracleBackedModel`` -- predicts exactly the oracle's delta: a perfect model that never
    drifts even at ``ρ = 0``, framing what the learned model is *for* (cheap rollout between
    consultations) and serving as the sanity ceiling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verisim.dist.action import DistAction
from verisim.dist.delta import DistDelta
from verisim.dist.state import DistributedState
from verisim.distoracle.base import DistOracle


@runtime_checkable
class DistModel(Protocol):
    """Predicts the structured log/replica delta an action makes to the cluster state (§6.1)."""

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta: ...


class DistNullModel:
    """Trivial predictor: predicts no change (the empty delta). The drift floor."""

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta:
        return []


class DistOracleBackedModel:
    """Symbolic-only: a perfect model that returns the oracle's own delta. The ceiling."""

    def __init__(self, oracle: DistOracle) -> None:
        self.oracle = oracle

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta:
        return self.oracle.step(state, action).delta
