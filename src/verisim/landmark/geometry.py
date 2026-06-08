"""Torch-free planning-geometry primitives for the SPEC-12 landmark layer (LP1, §4-§5).

LP1 asks the load-bearing question the whole L3P recipe rests on (SPEC-12 §4): *does distance in the
graph arm's ``embed()`` latent track distance in the world's transition dynamics?* If it does,
landmarks can be clustered in the latent and edges weighted by latent distance; if it does not,
SPEC-12 falls back to building the graph directly in reachability space (the free control-plane
metric). Answering it needs three distances between two network states:

  - **latent distance** - Euclidean distance between the two states' ``embed()`` vectors (torch;
    lives in the experiment, :mod:`verisim.experiments.lp1`, not here);
  - **oracle transition-distance** - the geodesic over the *action graph*: the fewest oracle steps
    that carry one state to another (:func:`bfs_geodesics`, exact within a capped ball);
  - **control-plane reachability distance** - the count of differing reachability entries
    (:func:`verisim.netoracle.reachability_bits_to_correct`, reused, not re-implemented here).

This module is the deterministic, dependency-free half (the NW0-NW3 discipline, SPEC-5 §13): the
state-identity key, the grammar action enumeration, the BFS geodesic, and pure-Python rank/linear
correlation - all property-testable with no torch and no GPU. The torch-dependent embedding and the
training live in the experiment.
"""

from __future__ import annotations

from verisim.net.action import NetAction, parse_net_action
from verisim.net.config import NetConfig
from verisim.net.state import NetworkState
from verisim.netoracle.base import NetOracle

# Canonical structural identity of a state: topology + services + firewall + flows, with the wall
# ``clock`` and ``last_exit`` observation deliberately dropped. Two states that differ only in the
# clock are the *same* planning landmark (reachability - the planning-relevant projection - ignores
# the clock, SPEC-5 §3.1), and keeping the clock would make ``advance`` an infinite BFS fan-out.
StateKey = tuple[
    tuple[tuple[str, bool, tuple[int, ...], tuple[str, ...]], ...],  # hosts
    tuple[tuple[str, str], ...],  # links
    tuple[tuple[str, str, int], ...],  # flows
]


def canon_key(state: NetworkState) -> StateKey:
    """The structural identity of ``state`` for landmark dedup (clock/last_exit dropped)."""
    hosts = tuple(
        sorted((h, hs.up, hs.services, hs.fw_deny) for h, hs in state.hosts.items())
    )
    return hosts, tuple(sorted(state.links)), tuple(sorted(state.flows))


def enumerate_actions(config: NetConfig) -> list[NetAction]:
    """Every action in the §3.2 grammar over ``config``'s closed world, in a fixed canonical order.

    The branching factor of the action-graph BFS (:func:`bfs_geodesics`). Links and firewall pairs
    are enumerated over distinct host pairs; ``connect``/``close`` over ordered host pairs × ports.
    """
    hosts, ports = config.hosts, config.ports
    raws: list[str] = ["advance"]
    for h in hosts:
        raws += [f"host_up {h}", f"host_down {h}"]
    for i, a in enumerate(hosts):
        for b in hosts[i + 1 :]:
            raws += [f"link_up {a} {b}", f"link_down {a} {b}"]
    for h in hosts:
        for p in ports:
            raws += [f"svc_up {h} {p}", f"svc_down {h} {p}"]
    for h in hosts:
        for s in hosts:
            if h != s:
                raws += [f"fw_deny {h} {s}", f"fw_allow {h} {s}"]
    for s in hosts:
        for d in hosts:
            if s != d:
                for p in ports:
                    raws += [f"connect {s} {d} {p}", f"close {s} {d} {p}"]
    return [parse_net_action(r) for r in raws]


def bfs_geodesics(
    oracle: NetOracle,
    anchor: NetworkState,
    actions: list[NetAction],
    *,
    max_depth: int,
    max_nodes: int,
) -> list[tuple[NetworkState, int]]:
    """Exact action-graph geodesics from ``anchor``, breadth-first, capped at ``max_nodes`` states.

    Returns ``(state, geodesic)`` pairs - ``anchor`` itself at distance 0, then every distinct state
    reachable within ``max_depth`` oracle steps, each tagged with its true shortest-path distance.
    Because BFS visits in distance order and dedups by :func:`canon_key`, the first time a state is
    seen *is* its geodesic - so distances are **exact** for every state inside the explored ball
    (SPEC-12 §5, H31). The cap keeps it affordable on a high-branching grammar (§10: the verify
    cost is real); states past the cap are simply unexplored, never mis-distanced.
    """
    seen: dict[StateKey, int] = {canon_key(anchor): 0}
    reps: list[tuple[NetworkState, int]] = [(anchor, 0)]
    frontier: list[NetworkState] = [anchor]
    depth = 0
    while frontier and depth < max_depth and len(seen) < max_nodes:
        depth += 1
        nxt: list[NetworkState] = []
        for s in frontier:
            for action in actions:
                t = oracle.step(s, action).state
                key = canon_key(t)
                if key not in seen:
                    seen[key] = depth
                    reps.append((t, depth))
                    nxt.append(t)
                    if len(seen) >= max_nodes:
                        return reps
            # `advance`/no-op edges keep us in place; the dedup above absorbs them.
        frontier = nxt
    return reps


# -- pure-Python correlation (no scipy; deterministic) ------------


def ranks(xs: list[float]) -> list[float]:
    """Fractional ranks of ``xs`` with ties averaged (the rank transform Spearman needs)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson linear correlation of ``xs`` and ``ys``; ``0.0`` when either is constant."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return float(sxy / (sxx * syy) ** 0.5)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation: Pearson on the rank transforms (ties averaged)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearson(ranks(xs), ranks(ys))


__all__ = [
    "StateKey",
    "bfs_geodesics",
    "canon_key",
    "enumerate_actions",
    "pearson",
    "ranks",
    "spearman",
]
