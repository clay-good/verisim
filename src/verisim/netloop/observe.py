"""The partial-observation oracle (SPEC-5 §5.3): two consultation modes.

Unlike v0, whose oracle only ever returns the *full* one-step truth, the network oracle
exposes two modes -- this is what makes partial observability native rather than bolted on,
and it is what creates the new probe-selection axis (§8.2, EN2/H10):

  - **full** (expensive): the complete true next state, as in v0. Reveals every fact.
  - **probe** (cheap): a localized observation of one host -- its up/down, listening
    services, firewall block-list, the links incident to it, and the flows it originates or
    terminates. The analogue of "a single host's view of its FIB/conntrack" (§5.3).

Both consult the same underlying deterministic :class:`~verisim.netoracle.base.NetOracle`,
so the truth is identical; they differ only in *how much* of it is returned and therefore in
cost. The bit-cost of each mode (``probe.bits`` vs :func:`full_bits`) is the denominator of
probe efficiency -- faithful horizon per oracle-bit (SPEC-5 §9.4) -- the network enrichment
of ``ρ`` that EN2 measures. This module is pure and dependency-free, like the rest of the
deterministic core.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.net.action import NetAction
from verisim.net.state import Flow, Link, NetworkState
from verisim.netmetrics.divergence import net_facts
from verisim.netoracle.base import NetOracle, NetStepResult


@dataclass(frozen=True)
class HostObservation:
    """The true local state of one host -- the partial truth a host-probe returns (§5.3).

    ``links`` are the links incident to ``host`` (either endpoint); ``flows`` are the flows
    ``host`` originates or terminates. Everything else about the network stays unobserved,
    so a belief filter (:class:`~verisim.netloop.operator.BeliefFilter`) may correct only
    this subgraph and must keep the model's prediction for the rest.
    """

    host: str
    present: bool  # whether the host exists in the (true) network at all
    up: bool
    services: tuple[int, ...]
    fw_deny: tuple[str, ...]
    links: frozenset[Link]
    flows: frozenset[Flow]

    @property
    def bits(self) -> int:
        """Description length of the observation in symbols (one per revealed fact)."""
        return 1 + len(self.services) + len(self.fw_deny) + len(self.links) + len(self.flows)


def observe_host(state: NetworkState, host: str) -> HostObservation:
    """The localized, true view of ``host`` in ``state`` (the cheap probe, §5.3)."""
    hs = state.hosts.get(host)
    if hs is None:
        return HostObservation(host, False, False, (), (), frozenset(), frozenset())
    links = frozenset(link for link in state.links if host in link)
    flows = frozenset(flow for flow in state.flows if flow[0] == host or flow[1] == host)
    return HostObservation(host, True, hs.up, hs.services, hs.fw_deny, links, flows)


def full_bits(state: NetworkState) -> int:
    """Description length of a *full* consultation in symbols (one per state fact)."""
    return len(net_facts(state))


class PartialNetOracle:
    """A :class:`~verisim.netoracle.base.NetOracle` exposed in two modes (SPEC-5 §5.3)."""

    def __init__(self, oracle: NetOracle) -> None:
        self._oracle = oracle

    def full(self, state: NetworkState, action: NetAction) -> NetStepResult:
        """The complete one-step truth from ``state`` under ``action`` (expensive)."""
        return self._oracle.step(state, action)

    def probe(self, state: NetworkState, action: NetAction, host: str) -> HostObservation:
        """A localized observation of ``host`` in the true next state (cheap)."""
        truth = self._oracle.step(state, action).state
        return observe_host(truth, host)
