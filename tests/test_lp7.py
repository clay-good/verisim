"""Smoke + invariant test for LP7 LLM-at-the-leaves boundary core (SPEC-12 §6, H37).

The committed LP7 result is dependency-free (search vs a myopic walk; the LLM arm is deferred), so
the whole core is fast and runs with no model. Checks: both strategies emit well-formed stats on
both axes; the soundness invariant the boundary rests on -- **graph search is always valid** (every
pair it is asked about is reachable by construction, so search validity is exactly 1.0 in every
bucket, the LP2 zero-false-edges guarantee); and that the LLM arm reports as deferred.
"""

from __future__ import annotations

from verisim.experiments.lp7 import (
    LP7Config,
    _greedy_traverse,
    llm_traverse_available,
    run_lp7,
)
from verisim.landmark.graph import LandmarkGraph


def _tiny() -> LP7Config:
    return LP7Config(
        n_hosts=5,
        n_ports=2,
        build_seeds=(0, 1, 2),
        build_steps=40,
        max_path_length=4,
        pairs_per_bucket=12,
        greedy_step_cap=32,
    )


def test_llm_arm_is_deferred() -> None:
    assert llm_traverse_available() is False


def test_greedy_dead_end_returns_none() -> None:
    # 0 -> 1 is a dead end that does not reach 2; the myopic walk has no path (search would).
    g = LandmarkGraph(
        nodes=("a", "b", "c"),  # type: ignore[arg-type]
        signatures=(frozenset(), frozenset({("x", "y", 1)}), frozenset({("x", "y", 2)})),
        edges=frozenset({(0, 1)}),
    )
    assert _greedy_traverse(g, 0, 2, g.signatures[2], cap=8) is None


def test_run_lp7_smoke_and_search_is_always_valid() -> None:
    stats = run_lp7(_tiny())
    assert {s.strategy for s in stats} == {"search", "greedy"}
    assert {s.axis for s in stats} == {"path_length", "degree"}
    for s in stats:
        assert s.n > 0
        assert 0.0 <= s.validity <= 1.0
        assert s.val_lo <= s.validity <= s.val_hi
        assert 0.0 <= s.optimality <= 1.0
    # Search is exact + complete: every queried pair is reachable by construction, so search
    # validity AND optimality are exactly 1.0 in every bucket (the soundness the boundary rests on).
    for s in stats:
        if s.strategy == "search":
            assert s.validity == 1.0
            assert s.optimality == 1.0


def test_run_lp7_is_deterministic() -> None:
    a = {(s.strategy, s.axis, s.bucket): s.validity for s in run_lp7(_tiny())}
    b = {(s.strategy, s.axis, s.bucket): s.validity for s in run_lp7(_tiny())}
    assert a == b
