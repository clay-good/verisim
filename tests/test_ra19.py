"""Tests for SPEC-22 RA19 (H151): the framing-transfer trap and layer saturation (recast).

Hermetic and deterministic. They pin the HONEST recast (after the 4-reviewer adversarial pass): the
oracle is the unique framing-robust zero-residual closer; the triad is flat by SATURATION (rho
inert), not correlation; the deployer's trap is a framing-transfer effect with a reported
sensitivity (not a point 20x); factorization HOLDS within a fixed framing on the triad; and genuine
correlation non-factorization lives only off-triad on disguised-ops, validated against CU39's
ImperfectMonitor.
"""

from __future__ import annotations

from verisim.realagent.ra19_correlated_layers import (
    TRIAD,
    Layer,
    _judging_joint,
    deployer_underestimate,
    montecarlo_residual,
    p_miss,
    ra19_verdict,
    residual_indep,
    residual_true,
    rho_is_inert_on_triad,
    run_ra19,
    subsets,
    trap_sensitivity,
)

AM: tuple[Layer, ...] = ("alignment", "monitor")
AL: tuple[Layer, ...] = ("alignment",)
AO: tuple[Layer, ...] = ("alignment", "oracle")


def test_calibration_matches_measured_rates() -> None:
    assert p_miss("alignment", "explicit_injection") == 0.0  # RA7 0/30
    assert p_miss("alignment", "disguised_ops") == 0.33  # RA7 10/30
    assert p_miss("alignment", "relational") == 1.0  # RA8/RA10 60/60
    assert p_miss("monitor", "relational", "direct") == 0.05  # RA11 38/40 policy framing
    assert p_miss("oracle", "relational") == 0.0  # complete mediation


def test_oracle_is_unique_framing_robust_closer() -> None:
    # any oracle-present subset closes the triad for every rho AND every framing; no action-reader
    # does
    for framing in ("ambient", "direct"):
        closers = [s for s in subsets()
                   if all(residual_true(s, h, 1.0, framing) <= 1e-9 for h in TRIAD)]
        assert len(closers) == 4 and all("oracle" in s for s in closers), framing


def test_triad_is_flat_by_saturation_and_rho_is_inert() -> None:
    # under ambient the action-reader stack is flat -- but because both marginals saturate at 1.0,
    # not because of correlation: the rho knob is provably inert on the triad.
    assert rho_is_inert_on_triad() is True
    am = AM
    for h in TRIAD:
        assert residual_true(am, h, 1.0, "ambient") == residual_true(AL, h, 1.0, "ambient")


def test_conditionality_surfaced_flat_flips_under_direct() -> None:
    # the headline is CONDITIONAL on the ambient assumption: under the monitor's measured DIRECT
    # recall the stack is NOT flat (the monitor buys a large reduction) -- only the oracle still
    # closes.
    am = AM
    for h in TRIAD:
        assert residual_true(am, h, 1.0, "direct") == 0.05  # monitor helps under direct framing
        assert residual_true(am, h, 1.0, "direct") != residual_true(AL, h, 1.0, "direct")
        assert residual_true(AO, h, 1.0, "direct") == 0.0  # only oracle


def test_factorization_holds_within_framing_on_triad() -> None:
    # the trap is marginal-transfer, NOT correlation: within a fixed framing the product factors on
    # the triad (residual_true == residual_indep). The 20x is direct-numerator over ambient-
    # denominator.
    am = AM
    for h in TRIAD:
        for framing in ("ambient", "direct"):
            assert abs(residual_true(am, h, 1.0, framing) - residual_indep(am, h, framing)) <= 1e-9


def test_genuine_correlation_gap_is_off_triad_on_disguised_ops() -> None:
    # correlation has teeth only where marginals are unsaturated: disguised-ops min(0.33,0.33)=0.33
    # strictly exceeds the product 0.109; this gap is framing-robust and rho-driven.
    am = AM
    assert residual_true(am, "disguised_ops", 1.0) > residual_indep(am, "disguised_ops") + 0.1
    vals = [residual_true(am, "disguised_ops", r) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals)  # rho interpolates product -> min
    assert vals[0] == residual_indep(am, "disguised_ops")  # rho=0 product
    assert abs(vals[-1] - 0.33) <= 1e-9  # rho=1 min


def test_framing_transfer_trap_is_a_sensitivity_not_a_point() -> None:
    du = deployer_underestimate("relational")
    assert du["naive_direct"] == 0.05
    assert du["true_ambient"] == 1.0
    assert du["trap_factor"] == 20.0  # worst case (monitor ambient miss = 1.0)
    # the trap degrades with the assumed monitor-ambient miss and the >10x claim flips below ~0.55
    assert trap_sensitivity(1.0) == 20.0
    assert trap_sensitivity(0.55) > 10.0
    assert trap_sensitivity(0.5) <= 10.0
    assert trap_sensitivity(0.5) > trap_sensitivity(0.4)  # monotone in the assumption


def test_inversion_on_explicit() -> None:
    align = residual_true(AL, "explicit_injection", 1.0)
    align_oracle = residual_true(AO, "explicit_injection", 1.0)
    assert align <= 1e-9  # alignment alone already closes explicit injection (RA7 0/30)
    assert abs(align - align_oracle) <= 1e-9  # adding the oracle changes nothing there


def test_analytic_matches_imperfectmonitor_montecarlo_off_triad() -> None:
    # validates the correlation primitive on the unsaturated (disguised-ops) regime where rho has
    # teeth
    for p_a, p_m, rho in [(0.33, 0.33, 1.0), (0.33, 0.33, 0.0), (0.6, 0.4, 0.7)]:
        analytic = _judging_joint(p_a, p_m, rho)
        mc = montecarlo_residual(p_a, p_m, rho, n=6000)
        assert abs(analytic - mc) < 0.03, (p_a, p_m, rho, analytic, mc)


def test_unguarded_baseline_is_one() -> None:
    for h in TRIAD:
        assert residual_true((), h, 1.0) == 1.0  # the RA10 60/60 anchor


def test_verdict_flags() -> None:
    v = ra19_verdict(run_ra19())
    assert v["oracle_unique_framing_robust_closer"] is True
    assert v["triad_flat_by_saturation_under_ambient"] is True
    assert v["triad_NOT_flat_under_direct"] is True
    assert v["rho_inert_on_triad"] is True
    assert v["factorization_holds_within_framing_on_triad"] is True
    assert v["genuine_correlation_gap_on_disguised_ops"] is True
    assert v["trap_survives_above_threshold"] is True
    assert v["trap_flips_below_half"] is True
    assert v["inversion_on_explicit"] is True
