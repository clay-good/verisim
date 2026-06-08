"""Unit tests for the torch-free placement helpers (SPEC-12 §6, LP5).

Betweenness, normalization, and top-budget selection are pure-Python and deterministic, so they are
tested directly against hand-computed graphs -- the dependency-free half of LP5 (the NW0-NW3
discipline, SPEC-5 §13).
"""

from __future__ import annotations

from typing import cast

from verisim.landmark.graph import LandmarkGraph
from verisim.landmark.placement import (
    betweenness_centrality,
    normalize,
    random_scores,
    select_top,
)
from verisim.net.state import NetworkState


def _graph(num_nodes: int, edges: set[tuple[int, int]]) -> LandmarkGraph:
    """A graph with ``num_nodes`` placeholder nodes and the given edges (all betweenness reads).

    Betweenness uses ``num_nodes`` (= ``len(nodes)``) and ``neighbors`` only, so the node
    representatives are immaterial here; we pass placeholder strings for the count.
    """
    placeholder_nodes = cast("tuple[NetworkState, ...]", tuple(f"n{i}" for i in range(num_nodes)))
    return LandmarkGraph(
        nodes=placeholder_nodes,
        signatures=tuple(),
        edges=frozenset(edges),
    )


def test_betweenness_path_is_a_peak_in_the_middle() -> None:
    # On a directed path, an interior node lies on every shortest path that spans it; the middle
    # node of a 5-node path is the chokepoint, the endpoints are zero.
    g = _graph(5, {(0, 1), (1, 2), (2, 3), (3, 4)})
    cb = betweenness_centrality(g)
    assert cb[0] == 0.0
    assert cb[4] == 0.0
    assert cb[2] > cb[1]  # the center is the strongest chokepoint
    assert cb[1] > cb[0]


def test_betweenness_no_edges_is_all_zero() -> None:
    assert set(betweenness_centrality(_graph(3, set())).values()) == {0.0}


def test_normalize_maps_to_unit_interval() -> None:
    out = normalize({0: 2.0, 1: 4.0, 2: 6.0})
    assert out[0] == 0.0
    assert out[2] == 1.0
    assert abs(out[1] - 0.5) < 1e-9


def test_normalize_constant_is_all_zero() -> None:
    assert normalize({0: 3.0, 1: 3.0}) == {0: 0.0, 1: 0.0}


def test_select_top_picks_highest_with_deterministic_ties() -> None:
    scores = {0: 1.0, 1: 3.0, 2: 3.0, 3: 0.5}
    assert select_top(scores, 2) == {1, 2}
    # Ties broken by id: with budget 1, id 1 wins over id 2.
    assert select_top(scores, 1) == {1}


def test_random_scores_are_deterministic() -> None:
    a = random_scores([0, 1, 2], seed=7)
    b = random_scores([0, 1, 2], seed=7)
    assert a == b
    assert set(a) == {0, 1, 2}
