"""Bootstrap-CI and curve-aggregation tests (SPEC-2 §9, §16)."""

from __future__ import annotations

import math
from statistics import fmean

from verisim.metrics import RunRecord, aggregate_curve, bootstrap_ci


def _record(difficulty: str, rho: float, seed: int, divergences: list[float]) -> RunRecord:
    return RunRecord(
        config={"difficulty": difficulty, "rho": rho},
        seed=seed,
        epsilon=0.0,
        divergences=divergences,
    )


def test_bootstrap_ci_collapses_for_degenerate_inputs():
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)
    assert bootstrap_ci([4.0]) == (4.0, 4.0)
    # All identical -> CI collapses to the value.
    assert bootstrap_ci([3.0, 3.0, 3.0]) == (3.0, 3.0)


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    values = [1.0, 2.0, 3.0, 10.0, 4.0, 5.0]
    a = bootstrap_ci(values, n_resamples=500, seed=7)
    b = bootstrap_ci(values, n_resamples=500, seed=7)
    assert a == b  # deterministic given seed
    lo, hi = a
    assert lo <= fmean(values) <= hi


def test_aggregate_curve_groups_and_summarizes():
    # Two difficulties x two rhos x two seeds; H_eps = prefix length of d <= 0.
    records = [
        _record("low", 0.0, 1, [0.0, 0.5]),  # H=1
        _record("low", 0.0, 2, [0.0, 0.0, 0.5]),  # H=2
        _record("low", 1.0, 1, [0.0, 0.0]),  # H=2
        _record("low", 1.0, 2, [0.0, 0.0]),  # H=2
        _record("high", 0.0, 1, [0.5]),  # H=0
        _record("high", 0.0, 2, [0.5]),  # H=0
    ]
    points = aggregate_curve(records, n_resamples=200, seed=0)
    by_key = {(p.difficulty, p.rho): p for p in points}

    assert by_key[("low", 0.0)].mean_h == 1.5
    assert by_key[("low", 0.0)].n == 2
    assert by_key[("low", 1.0)].mean_h == 2.0
    assert by_key[("high", 0.0)].mean_h == 0.0
    # Points are sorted by (difficulty, epsilon, rho).
    assert points == sorted(points, key=lambda p: (p.difficulty, p.epsilon, p.rho))
