"""Uncertainty-calibration metric tests (SPEC-2 §7.2, §17.2)."""

from __future__ import annotations

import math

from verisim.metrics import calibration_report, pearson, spearman


def test_pearson_known_values():
    assert math.isclose(pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)  # perfect positive
    assert math.isclose(pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]), -1.0)  # perfect negative
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0  # no x spread -> 0
    assert pearson([1.0], [1.0]) == 0.0  # undefined for n<2


def test_spearman_is_monotone_invariant():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 4.0, 9.0, 16.0]  # monotonic but non-linear
    assert math.isclose(spearman(xs, ys), 1.0)
    assert pearson(xs, ys) < 1.0  # pearson sees the curvature; spearman doesn't


def test_calibration_report_on_well_calibrated_signal():
    # signal rises with divergence -> strong positive correlation, rising bins.
    pairs = [(s / 10.0, s / 10.0) for s in range(11)]
    report = calibration_report(pairs, n_bins=5)
    assert report.n == 11
    assert math.isclose(report.pearson, 1.0, abs_tol=1e-9)
    assert math.isclose(report.spearman, 1.0, abs_tol=1e-9)
    means = [b.mean_divergence for b in report.bins]
    assert means == sorted(means)  # monotonically rising reliability curve


def test_calibration_report_on_uninformative_signal():
    # divergence independent of signal -> correlation near zero (the E2 symptom).
    pairs = [(0.1, 0.5), (0.2, 0.5), (0.3, 0.5), (0.4, 0.5)]
    report = calibration_report(pairs)
    assert report.pearson == 0.0  # constant divergence -> no spread
    assert report.mean_divergence == 0.5


def test_calibration_report_degenerate_inputs():
    empty = calibration_report([])
    assert empty.n == 0 and empty.bins == []
    # All-equal signal collapses to one bin holding everything.
    one = calibration_report([(0.3, 0.1), (0.3, 0.9)], n_bins=4)
    assert len(one.bins) == 1
    assert one.bins[0].n == 2
