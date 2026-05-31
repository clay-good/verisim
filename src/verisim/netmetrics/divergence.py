"""Graph divergence and reachability-faithfulness for the network world (SPEC-5 §9.1-9.2).

``divergence(a, b)`` is the graph analogue of v0's filesystem set-difference: a normalized
symmetric difference over typed node/edge/flow facts, ``0`` iff the graphs are identical and
``∈ [0, 1]``. ``reachability_faithfulness(a, b)`` is the operationally meaningful number
(SPEC-5 §9.2): the fraction of reachability-matrix entries ``R[src][(dst,port)]`` -- can-A-reach-
service-S -- that agree, which is what a defender/SRE actually relies on. Both are pure and
dependency-free, like v0's metric core.
"""

from __future__ import annotations

from verisim.net.state import NetworkState, reachability_matrix

Fact = tuple[object, ...]


def net_facts(state: NetworkState) -> set[Fact]:
    """The set of distinguishable facts defining a network state."""
    facts: set[Fact] = set()
    for host, hs in state.hosts.items():
        facts.add(("up", host, hs.up))
        for port in hs.services:
            facts.add(("svc", host, port))
        for src in hs.fw_deny:
            facts.add(("fw", host, src))
    for a, b in state.links:
        facts.add(("link", a, b))
    for src, dst, port in state.flows:
        facts.add(("flow", src, dst, port))
    facts.add(("\x00clock", state.clock))
    facts.add(("\x00exit", state.last_exit))
    return facts


def divergence(a: NetworkState, b: NetworkState) -> float:
    """Normalized symmetric difference over typed facts. ``0.0`` iff identical."""
    fa = net_facts(a)
    fb = net_facts(b)
    denom = len(fa) + len(fb)
    return len(fa ^ fb) / denom if denom else 0.0


def reachability_faithfulness(a: NetworkState, b: NetworkState) -> float:
    """Fraction of reachability-matrix entries that agree (over the union of keys).

    ``1.0`` iff the two states induce the same can-A-reach-S relation. Keys absent from one
    matrix are treated as ``False`` (no such service => unreachable), so a state that invents
    or drops a service is correctly penalized.
    """
    ra = reachability_matrix(a)
    rb = reachability_matrix(b)
    keys = set(ra) | set(rb)
    if not keys:
        return 1.0
    matches = sum(1 for k in keys if ra.get(k, False) == rb.get(k, False))
    return matches / len(keys)
