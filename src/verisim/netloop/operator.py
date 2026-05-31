"""Correction and belief operators ``C`` (SPEC-5 §8.3).

Two families, because the network oracle has two consultation modes (§5.3):

  - **Full-consult operators** -- given the complete one-step truth ``s'`` and the model's
    prediction ``ŝ'``, decide the post-consultation coupled state. ``HardReset``,
    ``Residual``, and ``Projection`` mirror v0's §6.2 operators exactly. As in v0, with a
    full-truth consultation all three snap the coupled state to ``s'`` and so are identical
    on faithful horizon; they differ only in what they *record*.
  - **The belief filter** -- given a *partial* observation of one host (the cheap probe)
    and the prediction ``ŝ'``, snap only the observed subgraph (the host's local state,
    its incident links, the flows it terminates) to truth and keep the model's belief for
    everything unobserved. This has no v0 analogue: it is the operator partial observability
    makes non-degenerate (SPEC-5 §8.3, EN3) -- a probe corrects strictly less than a full
    consult, so the two genuinely differ on horizon (no v0 identity collapse).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from verisim.net.state import HostState, NetworkState
from verisim.netmetrics.divergence import divergence, net_facts

from .observe import HostObservation


@runtime_checkable
class FullCorrection(Protocol):
    def correct(self, predicted: NetworkState, truth: NetworkState) -> NetworkState: ...


class HardReset:
    """``s ← s'``: snap the coupled state to the full truth (the §8.3 baseline)."""

    def correct(self, predicted: NetworkState, truth: NetworkState) -> NetworkState:
        return truth


@dataclass
class Residual:
    """Corrected state is truth; records the discrepancy ``|s' △ ŝ'|`` per correction.

    The discrepancy (the symmetric-difference fact count) is the online-learning signal a
    self-healing loop would step on (SPEC-5 §8.4, EN5); recorded here as a diagnostic.
    """

    discrepancies: list[int] = field(default_factory=list)

    def correct(self, predicted: NetworkState, truth: NetworkState) -> NetworkState:
        self.discrepancies.append(len(net_facts(predicted) ^ net_facts(truth)))
        return truth


@dataclass
class Projection:
    """Corrected state is truth; records the repaired *fraction* per correction.

    The fraction is the pre-correction divergence ``d(ŝ', s')`` -- how much of the graph the
    consultation had to fix (SPEC-5 §8.3 cheapest-faithful-horizon-per-correction lens).
    """

    repaired_fractions: list[float] = field(default_factory=list)

    def correct(self, predicted: NetworkState, truth: NetworkState) -> NetworkState:
        self.repaired_fractions.append(divergence(predicted, truth))
        return truth


def belief_filter(predicted: NetworkState, obs: HostObservation) -> NetworkState:
    """Snap the observed host's subgraph to truth; keep the prediction for the rest (§8.3).

    Corrects exactly what the probe reveals -- the host's local state, every link incident to
    it, and every flow it originates or terminates -- and nothing else. Returns a fresh state
    (the inputs are not mutated), so the unobserved subgraph is the model's belief verbatim.
    """
    hosts = dict(predicted.hosts)
    if obs.present:
        hosts[obs.host] = HostState(obs.up, obs.services, obs.fw_deny)
    # Replace beliefs about edges/flows incident to the observed host with the truth.
    links = {link for link in predicted.links if obs.host not in link} | set(obs.links)
    flows = {
        flow for flow in predicted.flows if flow[0] != obs.host and flow[1] != obs.host
    } | set(obs.flows)
    return NetworkState(
        hosts=hosts,
        links=links,
        flows=flows,
        clock=predicted.clock,
        last_exit=predicted.last_exit,
    )


@dataclass
class BeliefFilter:
    """The probe-mode operator: belief-filter on the observed subgraph, recording cost.

    ``repaired_fractions`` logs the pre-correction divergence of the observed subgraph each
    consultation (the EN3 diagnostic, the partial-observation analogue of :class:`Projection`).
    """

    repaired_fractions: list[float] = field(default_factory=list)

    def correct(self, predicted: NetworkState, obs: HostObservation) -> NetworkState:
        corrected = belief_filter(predicted, obs)
        self.repaired_fractions.append(divergence(predicted, corrected))
        return corrected
