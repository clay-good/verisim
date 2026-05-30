"""Bootstrap-CI and curve-aggregation tests (SPEC-2 §9, §16)."""

from __future__ import annotations

import math
from statistics import fmean

from verisim.metrics import (
    RunRecord,
    aggregate_comparison,
    aggregate_curve,
    aggregate_values,
    bootstrap_ci,
)


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


def _comparison_record(label: str, seed: int, divergences: list[float], schedule: list[bool]):
    return RunRecord(
        config={"policy": label},
        seed=seed,
        epsilon=0.0,
        divergences=divergences,
        consultation_schedule=schedule,
    )


def test_aggregate_comparison_groups_by_label_with_calls():
    records = [
        _comparison_record("fixed", 1, [0.0, 0.5], [True, False]),  # H=1, calls=1
        _comparison_record("fixed", 2, [0.0, 0.0, 0.5], [True, True, False]),  # H=2, calls=2
        _comparison_record("drift", 1, [0.5], [True]),  # H=0, calls=1
        _comparison_record("drift", 2, [0.5], [True]),  # H=0, calls=1
    ]
    points = aggregate_comparison(records, key="policy", n_resamples=200)
    by_label = {p.label: p for p in points}

    assert by_label["fixed"].mean_h == 1.5
    assert by_label["fixed"].mean_calls == 1.5
    assert by_label["fixed"].n == 2
    assert by_label["drift"].mean_h == 0.0
    assert by_label["drift"].mean_calls == 1.0
    # Sorted by (label, epsilon).
    assert points == sorted(points, key=lambda p: (p.label, p.epsilon))


def _acc_record(size: str, seed: int, accuracy: float) -> RunRecord:
    return RunRecord(
        config={"size": size, "step_accuracy": accuracy}, seed=seed, epsilon=0.0, divergences=[]
    )


def test_aggregate_values_groups_and_bootstraps_a_config_field():
    records = [
        _acc_record("tiny", 1, 0.2),
        _acc_record("tiny", 2, 0.4),
        _acc_record("big", 1, 0.9),
    ]
    stats = aggregate_values(records, group_keys=["size"], value="step_accuracy", n_resamples=200)
    by_key = {s.key: s for s in stats}
    assert math.isclose(by_key[("tiny",)].mean, 0.3)
    assert by_key[("tiny",)].n == 2
    assert by_key[("big",)].mean == 0.9
    assert stats == sorted(stats, key=lambda s: s.key)
