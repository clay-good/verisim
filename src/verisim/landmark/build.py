"""Sampling landmarks and the oracle's faithful edge set (SPEC-12 §3, §6 LP2). Torch-free.

Landmarks are sampled from seeded driver rollouts and deduplicated in **reachability space** (LP1's
§4 verdict): distinct reachability signatures become distinct landmarks. A *candidate edge* is a
reachability-changing hop the oracle actually took -- ``(src landmark, action) -> dst landmark`` --
so the oracle edge set is true by construction (the faithful graph). LP2 then asks how many of these
the *model* reproduces, and how many edges the model hallucinates to the wrong landmark.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from verisim.landmark.graph import ReachSig, reach_signature
from verisim.net.action import NetAction
from verisim.net.config import NetConfig
from verisim.net.state import NetworkState
from verisim.netoracle.base import NetOracle


@dataclass(frozen=True)
class LandmarkTransition:
    """One reachability-changing hop the oracle took: a candidate edge ``src_id -> true_dst_id``."""

    src_id: int
    src_state: NetworkState
    action: NetAction
    true_dst_id: int


@dataclass(frozen=True)
class LandmarkSample:
    """The landmark inventory of a set of rollouts: nodes, signatures, and the candidate edges."""

    nodes: tuple[NetworkState, ...]
    signatures: tuple[ReachSig, ...]
    sig_to_id: dict[ReachSig, int]
    transitions: tuple[LandmarkTransition, ...]

    def oracle_edges(self) -> frozenset[tuple[int, int]]:
        """The faithful edge set: every reachability-changing hop the oracle actually took."""
        return frozenset((t.src_id, t.true_dst_id) for t in self.transitions)


def sample_landmarks(
    oracle: NetOracle,
    config: NetConfig,
    *,
    driver: str,
    seeds: tuple[int, ...],
    n_steps: int,
) -> LandmarkSample:
    """Roll seeded trajectories; collect reachability-distinct landmarks + their changing hops."""
    from verisim.netdata import NetDriver

    sig_to_id: dict[ReachSig, int] = {}
    nodes: list[NetworkState] = []
    sigs: list[ReachSig] = []
    transitions: list[LandmarkTransition] = []

    def landmark_id(state: NetworkState) -> tuple[int, ReachSig]:
        sig = reach_signature(state)
        if sig not in sig_to_id:
            sig_to_id[sig] = len(nodes)
            nodes.append(state)
            sigs.append(sig)
        return sig_to_id[sig], sig

    for seed in seeds:
        drv = NetDriver(name=driver, config=config, rng=random.Random(seed))
        state = NetworkState.initial(config.hosts)
        src_id, src_sig = landmark_id(state)
        for _ in range(n_steps):
            action = drv.sample(state)
            nxt = oracle.step(state, action).state
            dst_id, dst_sig = landmark_id(nxt)
            if dst_sig != src_sig:  # a reachability-changing hop is a candidate edge
                transitions.append(LandmarkTransition(src_id, state, action, dst_id))
            state, src_id, src_sig = nxt, dst_id, dst_sig

    return LandmarkSample(
        nodes=tuple(nodes), signatures=tuple(sigs), sig_to_id=dict(sig_to_id),
        transitions=tuple(transitions),
    )


__all__ = ["LandmarkSample", "LandmarkTransition", "sample_landmarks"]
