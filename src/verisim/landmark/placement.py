"""Landmark placement policies: which waypoints to keep under a budget (SPEC-12 §6 LP5, H35).

A landmark graph can hold many waypoints; re-grounding at all of them is the ``ρ = 1`` ceiling. The
*placement* question (H35) is which few to keep when the budget is small -- the measured form of the
"interesting points" an analyst or a security world model flags. Two informed signals, against the
random control:

  - **reachability-betweenness (chokepoints)** -- a landmark on many shortest reachability paths is
    a chokepoint: re-grounding there fixes drift that would otherwise propagate down every route
    through it. :func:`betweenness_centrality` is Brandes' algorithm over the *verified* directed
    edges (so the centrality is of real reachability transitions -- the zero-false-paths guarantee,
    §8, carried into the placement signal).
  - **belief-variance (uncertainty)** -- a landmark where the model's RSSM posterior variance is
    high is where it is least sure, so a re-ground buys the most correction. The variance is a torch
    signal (the graph arm's §6.2 calibrated uncertainty), computed in the experiment and passed here
    as a score; this module stays torch-free (the NW0-NW3 discipline, SPEC-5 §13).

This module supplies the deterministic, dependency-free half: the betweenness centrality, min-max
normalization (for the *combined* policy), a deterministic random score, and the top-budget
selection. The model-derived belief variance and the goal-reach harness live in
:mod:`verisim.experiments.lp5`.
"""

from __future__ import annotations

import random
from collections import deque

from verisim.landmark.graph import LandmarkGraph


def betweenness_centrality(graph: LandmarkGraph) -> dict[int, float]:
    """Brandes' betweenness centrality over the verified directed edges (the chokepoint score).

    Returns ``{landmark_id: centrality}`` for every node, the sum over ordered source/target pairs
    of the fraction of shortest paths through that node. Unweighted (every verified hop costs one),
    which is the right metric for a reachability graph: a chokepoint is a landmark that lies on many
    of the fewest-hop routes between landmarks (SPEC-12 §8 -- high-value-target surfacing).
    """
    nodes = range(graph.num_nodes)
    cb: dict[int, float] = dict.fromkeys(nodes, 0.0)
    for s in nodes:
        # Single-source shortest paths (BFS) with path counts -- Brandes' accumulation.
        stack: list[int] = []
        preds: dict[int, list[int]] = {v: [] for v in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        dist = dict.fromkeys(nodes, -1)
        dist[s] = 0
        queue: deque[int] = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.neighbors(v):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    return cb


def normalize(scores: dict[int, float]) -> dict[int, float]:
    """Min-max ``scores`` to ``[0, 1]`` (all-equal maps to all-zero), for the combined policy."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo
    if span <= 0.0:
        return dict.fromkeys(scores, 0.0)
    return {k: (v - lo) / span for k, v in scores.items()}


def random_scores(ids: list[int], *, seed: int) -> dict[int, float]:
    """A deterministic random score per id (the random-placement control)."""
    rng = random.Random(seed)
    return {i: rng.random() for i in ids}


def select_top(scores: dict[int, float], budget: int) -> set[int]:
    """The ``budget`` highest-scoring ids (ties broken by id for determinism)."""
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    return set(ranked[:budget])


__all__ = [
    "betweenness_centrality",
    "normalize",
    "random_scores",
    "select_top",
]
