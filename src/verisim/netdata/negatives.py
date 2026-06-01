"""Oracle hard-negatives and counterfactual branches (SPEC-8 §4.3, OG2).

Contrastive / JEPA self-supervision is bottlenecked on **negatives** and on collapse, and the
field's stand-in is a *statistical* regularizer (VICReg's covariance term) for "push representations
apart." In a domain with an oracle the negatives are free and *exact*: this module is the
deterministic factory (milestone OG2) for the two kinds SPEC-8 §4.3 wants:

  - **one-edit-wrong successors** -- the true next-state ``s' = O(s, a)`` with exactly one fact
    perturbed. These are the bits-to-correct neighborhood (the near-misses a contrastive objective
    most wants and can least easily mine), each labeled "not the true successor" by construction.
  - **counterfactual branches** -- the true next-states of *alternative* actions ``a'`` from the
    same state, ``O(s, a')``. These are the very branches the model is asked to predict at
    intervention time (the EN6 / H5 interventional-fidelity set), each perfectly labeled.

This generalizes the K1 supervised hard-negative generator (SPEC-2.1 §5) to contrastive pairs and
action-branch counterfactuals. Pure and dependency-free -- no torch, no GPU; it produces *data*, and
(DD-AR2) never edits the oracle, metric, or gate.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from verisim.net.action import NetAction, parse_net_action
from verisim.net.config import NetConfig
from verisim.net.state import NetworkState, link_key
from verisim.netmetrics.divergence import divergence
from verisim.netoracle.base import NetOracle


def _one_fact_flips(true_next: NetworkState, config: NetConfig) -> list[NetworkState]:
    """Every Hamming-1 neighbor of ``true_next`` in fact space, in canonical order.

    Each neighbor toggles exactly one atomic fact -- a host's up/down, one service, one firewall
    entry, one link, or one flow -- so each differs from ``true_next`` and from the others.
    """
    hosts = config.hosts
    out: list[NetworkState] = []

    def flip_host_up(h: str) -> NetworkState:
        s = true_next.copy()
        s.hosts[h] = s.hosts[h].with_up(not s.hosts[h].up)
        return s

    def flip_service(h: str, port: int) -> NetworkState:
        s = true_next.copy()
        s.hosts[h] = s.hosts[h].with_service(port, port not in s.hosts[h].services)
        return s

    def flip_fw(h: str, src: str) -> NetworkState:
        s = true_next.copy()
        s.hosts[h] = s.hosts[h].with_fw(src, src not in s.hosts[h].fw_deny)
        return s

    def flip_link(a: str, b: str) -> NetworkState:
        s = true_next.copy()
        key = link_key(a, b)
        s.links.discard(key) if key in s.links else s.links.add(key)
        return s

    def flip_flow(src: str, dst: str, port: int) -> NetworkState:
        s = true_next.copy()
        flow = (src, dst, port)
        s.flows.discard(flow) if flow in s.flows else s.flows.add(flow)
        return s

    for h in hosts:
        if h in true_next.hosts:
            out.append(flip_host_up(h))
    for h in hosts:
        for port in config.ports:
            if h in true_next.hosts:
                out.append(flip_service(h, port))
    for h in hosts:
        for src in hosts:
            if src != h and h in true_next.hosts:
                out.append(flip_fw(h, src))
    for i, a in enumerate(hosts):
        for b in hosts[i + 1 :]:
            out.append(flip_link(a, b))
    for src in hosts:
        for dst in hosts:
            if src != dst:
                for port in config.ports:
                    out.append(flip_flow(src, dst, port))
    return out


def one_edit_negatives(
    true_next: NetworkState,
    config: NetConfig,
    *,
    limit: int | None = None,
    rng: random.Random | None = None,
) -> list[NetworkState]:
    """One-edit-wrong successors of ``true_next`` (the bits-to-correct neighborhood, §4.3).

    Every returned state differs from ``true_next`` by exactly one atomic fact, so each is ``≠`` the
    true successor by construction. With ``limit`` set and ``rng`` given, a deterministic random
    subset is drawn (the contrastive minibatch); otherwise the first ``limit`` in canonical order.
    """
    flips = _one_fact_flips(true_next, config)
    if limit is None or limit >= len(flips):
        return flips
    if rng is not None:
        return rng.sample(flips, limit)
    return flips[:limit]


def enumerate_actions(config: NetConfig) -> list[NetAction]:
    """One representative action per command name -- the grammar, for counterfactual coverage."""
    h0, h1 = config.hosts[0], config.hosts[1]
    port = config.ports[0]
    cmds = [
        f"host_up {h0}",
        f"host_down {h0}",
        f"link_up {h0} {h1}",
        f"link_down {h0} {h1}",
        f"svc_up {h0} {port}",
        f"svc_down {h0} {port}",
        f"fw_deny {h0} {h1}",
        f"fw_allow {h0} {h1}",
        f"connect {h0} {h1} {port}",
        f"close {h0} {h1} {port}",
        "advance",
    ]
    return [parse_net_action(c) for c in cmds]


def counterfactual_branches(
    state: NetworkState,
    oracle: NetOracle,
    config: NetConfig,
    *,
    alt_actions: Iterable[NetAction] | None = None,
) -> list[tuple[NetAction, NetworkState]]:
    """``(a', O(s, a'))`` for alternative actions ``a'`` -- the branches the model is asked about.

    Defaults to the full action grammar (:func:`enumerate_actions`), so coverage spans every
    command name. Each successor is the oracle's exact truth for that branch (§4.3, the EN6/H5 set).
    """
    actions = enumerate_actions(config) if alt_actions is None else list(alt_actions)
    return [(a, oracle.step(state, a).state) for a in actions]


def is_hard_negative(candidate: NetworkState, true_next: NetworkState) -> bool:
    """``True`` iff ``candidate`` is not the true successor (``divergence > 0``)."""
    return divergence(candidate, true_next) > 0.0


__all__ = [
    "counterfactual_branches",
    "enumerate_actions",
    "is_hard_negative",
    "one_edit_negatives",
]
