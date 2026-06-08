"""Property tests for the torch-free landmark planning-geometry primitives (SPEC-12 §5, LP1).

The deterministic half of LP1: the structural state key, the grammar action enumeration, the exact
BFS geodesic over the action graph, and the pure-Python correlations. These pin the invariants
(geodesics are exact and BFS-ordered; the action set is complete and deterministic; correlation is
bounded and sign-correct) with no torch, before LP1's learned-latent measurement consumes them.
"""

from __future__ import annotations

import random

from verisim.landmark.geometry import (
    bfs_geodesics,
    canon_key,
    enumerate_actions,
    pearson,
    ranks,
    spearman,
)
from verisim.net.config import scaled_net_config
from verisim.net.state import NetworkState
from verisim.netdata import NetDriver
from verisim.netoracle import ReferenceNetworkOracle


def test_canon_key_ignores_clock_and_last_exit() -> None:
    cfg = scaled_net_config(4, 2)
    oracle = ReferenceNetworkOracle()
    drv = NetDriver(name="weighted", config=cfg, rng=random.Random(0))
    state = NetworkState.initial(cfg.hosts)
    for _ in range(10):
        state = oracle.step(state, drv.sample(state)).state
    # Same structural state but a different wall clock / observation -> same landmark.
    bumped = state.copy()
    bumped.clock = state.clock + 100
    bumped.last_exit = 1 - state.last_exit
    assert canon_key(state) == canon_key(bumped)


def test_enumerate_actions_complete_and_deterministic() -> None:
    cfg = scaled_net_config(5, 3)
    actions = enumerate_actions(cfg)
    # 1 advance + 2*5 host + 2*C(5,2) link + 2*(5*3) svc + 2*(5*4) fw + 2*(5*4*3) connect/close.
    assert len(actions) == 1 + 10 + 2 * 10 + 2 * 15 + 2 * 20 + 2 * 60
    assert [a.raw for a in actions] == [a.raw for a in enumerate_actions(cfg)]  # deterministic
    assert len({a.raw for a in actions}) == len(actions)  # no duplicates


def test_bfs_geodesics_exact_and_bfs_ordered() -> None:
    cfg = scaled_net_config(4, 2)
    oracle = ReferenceNetworkOracle()
    actions = enumerate_actions(cfg)
    anchor = NetworkState.initial(cfg.hosts)
    reps = bfs_geodesics(oracle, anchor, actions, max_depth=2, max_nodes=120)
    assert reps[0] == (anchor, 0)  # anchor itself at distance 0
    dists = [d for _, d in reps]
    assert dists == sorted(dists)  # BFS visits in non-decreasing distance order
    assert max(dists) <= 2
    # Every distance-1 state is a true one-step oracle successor of the anchor.
    one_step = {canon_key(oracle.step(anchor, a).state) for a in actions}
    for s, d in reps:
        if d == 1:
            assert canon_key(s) in one_step


def test_bfs_geodesics_respects_node_cap() -> None:
    cfg = scaled_net_config(5, 3)
    oracle = ReferenceNetworkOracle()
    actions = enumerate_actions(cfg)
    reps = bfs_geodesics(
        oracle, NetworkState.initial(cfg.hosts), actions, max_depth=4, max_nodes=30
    )
    assert len(reps) <= 30


def test_ranks_average_ties() -> None:
    assert ranks([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]
    assert ranks([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]  # tie block averaged


def test_correlations_bounded_and_signed() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert pearson(xs, xs) == 1.0
    assert spearman(xs, [5.0, 4.0, 3.0, 2.0, 1.0]) == -1.0
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0  # constant -> 0, not nan
    # Monotone-but-nonlinear: Spearman = 1, Pearson < 1.
    ys = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert spearman(xs, ys) == 1.0
    assert pearson(xs, ys) < 1.0
