"""The host ``Model`` interface and baseline world models (SPEC-6 §6, §8).

The composed propose-verify-correct loop (SPEC-6 §8) is generic over *any* model that predicts a
structured bundle delta from ``(state, action)`` -- exactly as v0's M5 loop and the network NW5 loop
are generic over their model protocols. The learned flat ``M_θ``
(:class:`verisim.hostmodel.world_model.NeuralHostWorldModel`, HC4) already implements this protocol,
so it drops into the loop unchanged; the factored interaction-graph arm (SPEC-6 §6.1) will too.

Two dependency-free baselines ship now (SPEC-6 §8, mirroring v0/NW5), so every loop invariant is
testable with no GPU before the neural model is trained:

  - ``HostNullModel`` -- predicts the empty delta ("nothing happens"). It drifts the moment the
    oracle changes anything; the absolute floor and the proof the task is nontrivial.
  - ``HostOracleBackedModel`` -- predicts exactly the oracle's bundle delta: a perfect model that
    never drifts even at ``ρ = 0``, framing what the neural model is *for* (cheap composed rollout
    between consultations) and serving as the sanity ceiling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verisim.host.action import HostAction
from verisim.host.delta import HostDelta
from verisim.host.state import HostState
from verisim.hostoracle.base import HostOracle


@runtime_checkable
class HostModel(Protocol):
    """Predicts the structured bundle delta a syscall makes to a host state (§6.1)."""

    def predict_delta(self, state: HostState, action: HostAction) -> HostDelta: ...


@runtime_checkable
class HostUncertaintyModel(HostModel, Protocol):
    """A :class:`HostModel` that *also* reports its per-prediction uncertainty (§5.4, §8.1).

    Extends :class:`HostModel` (an uncertainty model is still a model), so it carries
    ``predict_delta`` too and is accepted anywhere a :class:`HostModel` is. The signal drives the
    ``uncertainty``/``drift``-triggered consultation policies (SPEC-6 §8.1), the host analogue of
    v0's M_θ uncertainty signal. Models without one (the baselines below) do not implement this
    protocol, and the runner treats their signal as ``0``.
    """

    def predict_delta_with_uncertainty(
        self, state: HostState, action: HostAction
    ) -> tuple[HostDelta, float]: ...


@runtime_checkable
class HostSubsystemUncertaintyModel(HostModel, Protocol):
    """A :class:`HostModel` that reports its **per-subsystem** predicted-delta uncertainty (§5.4).

    The map (subsystem -> uncertainty for this step) is the signal the smart which-subsystem policy
    ``π_w`` (SPEC-6 §8.2) reads to spend a consult on the subsystem the model is least sure about.
    The factored interaction-graph arm supplies it (bucketed decode entropy); proposers without one
    do not implement it, and the runner falls the policy back to its uncertainty-free path.
    """

    def predict_delta_with_subsystem_uncertainty(
        self, state: HostState, action: HostAction
    ) -> tuple[HostDelta, dict[str, float]]: ...


class HostNullModel:
    """Trivial predictor: predicts no change (the empty delta). The drift floor."""

    def predict_delta(self, state: HostState, action: HostAction) -> HostDelta:
        return []


class HostOracleBackedModel:
    """Symbolic-only: a perfect model that returns the oracle's own bundle delta. The ceiling."""

    def __init__(self, oracle: HostOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: HostState, action: HostAction) -> HostDelta:
        return self._oracle.step(state, action).delta
