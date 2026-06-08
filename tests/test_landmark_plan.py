"""Property tests for the torch-free planner: graph search + the re-grounding hop executor (LP3).

The deterministic half of LP3 (the NW0-NW3 discipline, SPEC-5 §13): ``shortest_landmark_path`` is a
BFS over verified edges (so any path it returns is real, LP2's zero-false guarantee), and
``execute_plan`` is the reachability-altitude imagine/verify loop -- generic over the ``NetModel``
protocol, so it is exercised here with the shipped dependency-free baselines (``NetNullModel`` =
drifts at once; ``NetOracleBackedModel`` = perfect) on a real seeded journey, no torch. These pin
the invariants the headline rests on: re-grounding sets the boundary steps to truth and spends
one consult per boundary, and a perfect model reaches the goal at ``ρ = 0`` while the null cannot.
"""

from __future__ import annotations

import random

from verisim.landmark.graph import LandmarkGraph, ReachSig, reach_signature
from verisim.landmark.plan import execute_plan, shortest_landmark_path
from verisim.net.action import NetAction
from verisim.net.config import scaled_net_config
from verisim.net.state import NetworkState
from verisim.netloop.model import NetNullModel, NetOracleBackedModel
from verisim.netoracle import ReferenceNetworkOracle


def _chain(n: int, edges: set[tuple[int, int]]) -> LandmarkGraph:
    nodes = (NetworkState.initial(("h0", "h1")),) * n
    sigs: tuple[ReachSig, ...] = tuple(frozenset({("h0", "h1", i)}) for i in range(n))
    return LandmarkGraph(nodes=nodes, signatures=sigs, edges=frozenset(edges))


def test_shortest_path_self_and_chain() -> None:
    g = _chain(4, {(0, 1), (1, 2), (2, 3)})
    assert shortest_landmark_path(g, 2, 2) == [2]
    assert shortest_landmark_path(g, 0, 3) == [0, 1, 2, 3]


def test_shortest_path_picks_fewest_hops_and_stitches() -> None:
    # A diamond: 0->1->4 (2 hops) vs 0->2->3->4 (3 hops). BFS must take the short route.
    g = _chain(5, {(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)})
    assert shortest_landmark_path(g, 0, 4) == [0, 1, 4]
    # Stitching: a route that exists only by composing two edges through a shared landmark.
    g2 = _chain(3, {(0, 1), (1, 2)})
    assert shortest_landmark_path(g2, 0, 2) == [0, 1, 2]


def test_shortest_path_unreachable_is_none() -> None:
    g = _chain(3, {(0, 1)})  # node 2 has no incoming edge
    assert shortest_landmark_path(g, 0, 2) is None


def _journey(n_steps: int) -> tuple[NetworkState, list[NetAction], list[NetworkState]]:
    """A real seeded journey: start, actions, and the true state after each (torch-free)."""
    from verisim.netdata import NetDriver

    net = scaled_net_config(5, 3)
    oracle = ReferenceNetworkOracle()
    drv = NetDriver(name="weighted", config=net, rng=random.Random(7))
    start = NetworkState.initial(net.hosts)
    state = start
    actions, truth = [], []
    for _ in range(n_steps):
        a = drv.sample(state)
        state = oracle.step(state, a).state
        actions.append(a)
        truth.append(state)
    return start, actions, truth


def test_perfect_model_reaches_goal_at_zero_budget() -> None:
    start, actions, truth = _journey(16)
    oracle = ReferenceNetworkOracle()
    perfect = NetOracleBackedModel(oracle)
    trace = execute_plan(perfect, start, actions, truth, frozenset(), reground=False)
    assert trace.goal_reached  # a perfect model never drifts, even free-running (ρ = 0)
    assert trace.reach_horizon == trace.n_steps
    assert trace.n_consults == 0


def test_reground_sets_boundaries_to_truth_and_spends_one_consult_each() -> None:
    start, actions, truth = _journey(16)
    boundaries = frozenset({3, 7, 11})
    trace = execute_plan(NetNullModel(), start, actions, truth, boundaries, reground=True)
    assert trace.n_consults == len(boundaries)
    # Every re-grounded boundary step is exact (state was reset to truth there).
    for t in boundaries:
        assert trace.full_correct[t]


def test_null_model_cannot_reach_a_changed_goal_free_running() -> None:
    start, actions, truth = _journey(16)
    # A journey whose goal reachability differs from the start (the null model is stuck at start).
    assert reach_signature(truth[-1]) != reach_signature(start)
    trace = execute_plan(NetNullModel(), start, actions, truth, frozenset(), reground=False)
    assert not trace.goal_reached
