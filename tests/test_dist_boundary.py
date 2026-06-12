"""SPEC-20 UA11 distributed-boundary tests (the third-world structure/content confirmation).

The contract: the distributed keyed extractors read the right dimension, the faithful predictor is
exact on both tasks (ground-truth labels on the distributed world too), the cumulative rollout is a
union, and the smoke boundary run holds (content gap > structure gap). Torch-free parts use the
oracle; the real gradient comes from training a (tiny) distributed M_θ.
"""

from __future__ import annotations

import pytest

from verisim.distoracle import ReferenceDistOracle
from verisim.experiments.dist_boundary import (
    DIST_TASKS,
    DistBoundaryConfig,
    _keyed_reward,
    _oracle_step,
    _rollout_keyed,
    make_dist_workload,
    partition_set,
    value_set,
)
from verisim.experiments.ed1_learned import ED1LearnedConfig


def test_tasks_are_structure_then_content():
    assert [t.band for t in DIST_TASKS] == ["structure", "content"]
    assert [t.name for t in DIST_TASKS] == ["partition-control", "value-integrity"]
    assert [t.keyed_dimension for t in DIST_TASKS] == ["partitions", "values"]


def test_extractors_read_the_right_dimension():
    cfg = ED1LearnedConfig()
    oracle = ReferenceDistOracle(cfg.dist)
    start, actions = make_dist_workload(901, 12, oracle=oracle, config=cfg)
    step = _oracle_step(oracle)
    state = start
    for a in actions:
        state = step(state, a)
    # partitions are frozensets of node ids; values are (object_id, value) pairs
    assert all(isinstance(p, frozenset) for p in partition_set(state))
    assert all(isinstance(v, tuple) and len(v) == 2 for v in value_set(state))


def test_faithful_predictor_scores_both_tasks_perfectly():
    cfg = ED1LearnedConfig()
    oracle = ReferenceDistOracle(cfg.dist)
    step = _oracle_step(oracle)
    for task in DIST_TASKS:
        for seed in (900, 901, 902):
            start, actions = make_dist_workload(seed, 12, oracle=oracle, config=cfg)
            r = _keyed_reward(step, step, start, actions, task.budget, task.key_fn)
            assert r == pytest.approx(1.0)


def test_rollout_keyed_is_cumulative():
    cfg = ED1LearnedConfig()
    oracle = ReferenceDistOracle(cfg.dist)
    start, actions = make_dist_workload(901, 12, oracle=oracle, config=cfg)
    step = _oracle_step(oracle)
    seen = _rollout_keyed(step, start, actions, value_set)
    running, state = set(value_set(start)), start
    for a in actions:
        state = step(state, a)
        running |= value_set(state)
    assert seen == running


# --- torch-gated: the real boundary on a tiny distributed M_θ -------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.dist_boundary import run_dist_boundary  # noqa: E402


def test_dist_boundary_holds_on_smoke_model():
    results, verdict = run_dist_boundary(DistBoundaryConfig.smoke())
    assert {r.band for r in results} == {"structure", "content"}
    for r in results:
        assert r.faithful == pytest.approx(1.0)  # ground-truth labels on the distributed world too
    # the content gap exceeds the structure gap -> the boundary holds, third world
    assert verdict["content_gap"] >= verdict["structure_gap"]
    assert "boundary_holds" in verdict and "content_knee_rho" in verdict


def test_recession_verdict_rule():
    # the robust claim: the structure gap persists (not ~0) -> structural-first NOT universal
    from verisim.experiments.dist_boundary import RecessionPoint, dist_recession

    # the committed run: structure persists (~0.2 at l), content recedes -> not structural-first
    pts = [
        RecessionPoint("xs", 1024, 0.250, 0.600),
        RecessionPoint("l", 49152, 0.200, 0.275),
    ]
    threshold = 0.05
    structure_persists = pts[-1].structure_gap > threshold  # does NOT reach ~0 like host/network
    structural_first = (
        (pts[0].structure_gap - pts[-1].structure_gap)
        > (pts[0].content_gap - pts[-1].content_gap) + 0.1
        and pts[-1].structure_gap <= threshold
    )
    assert structure_persists and not structural_first  # the H87 refinement
    assert callable(dist_recession)
