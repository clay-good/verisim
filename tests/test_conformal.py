"""Invariants for the oracle-calibrated conformal layer (SPEC-15 §2).

The conformal calibrator's guarantee is the whole point, so these tests pin it directly: on
*exchangeable* data the certified threshold keeps the empirical undetected-breach rate at/under the
target (within finite-sample slack); a calibrated score certifies at lower budget than an
uncalibrated one; the graded risk threshold bounds the graded loss; and the ACI update moves it the
right way. Structural, not magnitude — the macOS-first principle.
"""

from __future__ import annotations

import random

from verisim.conformal.calibrate import (
    calibrate_threshold,
    region_thresholds,
    undetected_rate,
)
from verisim.conformal.policy import AdaptiveConformalTriggered, ConformalTriggered
from verisim.conformal.risk import calibrate_risk_threshold, undetected_breach_loss
from verisim.loop.policy import StepContext


def _calibrated_data(n: int, seed: int) -> tuple[list[float], list[int], list[float]]:
    """Scores, breach flags, divergences where the score predicts breach (the calibrated case)."""
    rng = random.Random(seed)
    scores: list[float] = []
    breaches: list[int] = []
    divs: list[float] = []
    for _ in range(n):
        breach = rng.random() < 0.35
        # Calibrated: breach steps draw high scores, clean steps low (with overlap).
        score = rng.uniform(0.5, 1.0) if breach else rng.uniform(0.0, 0.6)
        d = rng.uniform(0.05, 0.3) if breach else rng.uniform(0.0, 0.05)
        scores.append(score)
        breaches.append(1 if breach else 0)
        divs.append(d)
    return scores, breaches, divs


def test_coverage_holds_on_exchangeable_data() -> None:
    # Calibrate on one i.i.d. split, verify undetected rate <= alpha (+ slack) on a disjoint split.
    cs, cb, _ = _calibrated_data(600, seed=0)
    ts, tb, _ = _calibrated_data(600, seed=1)
    for alpha in (0.05, 0.10, 0.20):
        th = calibrate_threshold(cs, cb, alpha)
        assert undetected_rate(ts, tb, th.tau) <= alpha + 0.05  # finite-sample slack


def test_lower_alpha_consults_more() -> None:
    cs, cb, _ = _calibrated_data(600, seed=2)
    taus = [calibrate_threshold(cs, cb, a).tau for a in (0.20, 0.10, 0.05)]
    # Lower alpha -> smaller threshold -> consult more, so tau is non-increasing as alpha falls.
    assert taus[0] >= taus[1] >= taus[2]


def test_calibrated_beats_uncalibrated_budget() -> None:
    # Calibrated signal certifies the same alpha at lower budget than an uncalibrated one.
    cs, cb, _ = _calibrated_data(800, seed=3)
    rng = random.Random(3)
    us = [rng.random() for _ in cb]  # uncalibrated: score independent of breach
    cal = calibrate_threshold(cs, cb, 0.05)
    unc = calibrate_threshold(us, cb, 0.05)
    assert cal.consult_rate < unc.consult_rate


def test_risk_threshold_bounds_graded_loss() -> None:
    cs, _, cd = _calibrated_data(700, seed=4)
    ts, _, td = _calibrated_data(700, seed=5)
    eps = 0.05
    scale = 0.25
    alpha = 0.10
    th = calibrate_risk_threshold(cs, cd, eps, alpha, overshoot_scale=scale)
    realized = sum(
        undetected_breach_loss(d, eps, consulted=(s > th.tau), overshoot_scale=scale)
        for s, d in zip(ts, td, strict=True)
    ) / len(ts)
    assert realized <= alpha + 0.05


def test_undetected_breach_loss_zero_when_caught_or_no_breach() -> None:
    assert undetected_breach_loss(0.0, 0.05, consulted=False) == 0.0  # no breach
    assert undetected_breach_loss(0.5, 0.05, consulted=True) == 0.0  # consulted -> caught
    assert undetected_breach_loss(0.5, 0.05, consulted=False) > 0.0  # undetected breach


def test_region_thresholds_fall_back_for_sparse_regions() -> None:
    cs, cb, _ = _calibrated_data(400, seed=6)
    regions = [i % 3 for i in range(len(cs))]
    th = region_thresholds(cs, cb, regions, 0.10)
    assert set(th) == {0, 1, 2}


def test_conformal_triggered_policy() -> None:
    pol = ConformalTriggered(tau=0.5)
    assert pol.should_consult(StepContext(step=0, signal=0.6)) is True
    assert pol.should_consult(StepContext(step=0, signal=0.4)) is False


def test_aci_lowers_tau_on_miss_raises_on_clean() -> None:
    cs, cb, _ = _calibrated_data(400, seed=7)
    aci = AdaptiveConformalTriggered(cs, cb, target_alpha=0.10, gamma=0.05)
    tau0 = aci.tau
    aci.update(breach=True, consulted=False)  # an undetected breach -> consult more -> tau down
    assert aci.tau < tau0
    tau1 = aci.tau
    aci.update(breach=False, consulted=False)  # a clean step -> tau drifts up
    assert aci.tau > tau1
