"""Smoke + structural-invariant tests for the SPEC-15 conformal experiments (CF1-CF5).

The committed CF core is the controlled signal stand-in (the trained-``M_θ`` arm is deferred, LP7
rule), so the battery is fast and GPU-free. Tests assert the *structural* claims: the coverage gate,
the conformal-beats-fixed budget, the calibrated/uncalibrated split, the static-drifts/ACI-recovers
fork, the risk-control saving, and the cross-world transfer of all three -- on tiny configs, not
exact magnitudes (the macOS-first principle).
"""

from __future__ import annotations

from verisim.experiments.cf1 import CF1Config, run_cf1
from verisim.experiments.cf2 import CF2Config, run_cf2
from verisim.experiments.cf3 import CF3Config, run_cf3
from verisim.experiments.cf4 import CF4Config, calibration_curves, run_cf4
from verisim.experiments.cf5 import CF5Config, run_cf5
from verisim.experiments.cf_common import mean


def test_cf1_gate_holds_and_beats_fixed() -> None:
    stats = run_cf1(CF1Config(n_rollouts=12, steps=40, n_seeds=4, alphas=(0.05, 0.10, 0.20)))
    for s in stats:
        # The coverage gate: empirical undetected rate at/under target (+ finite-sample slack).
        assert s.test_undetected <= s.alpha + 0.06
        # H51: conformal certifies the same α at lower budget than fixed.
        assert s.conformal_rho <= s.fixed_rho + 1e-9


def test_cf4_calibrated_beats_uncalibrated() -> None:
    stats = run_cf4(CF4Config(n_rollouts=12, steps=40, n_seeds=4))
    cal = next(s for s in stats if s.arm == "calibrated")
    unc = next(s for s in stats if s.arm == "uncalibrated")
    cal_saves = cal.fixed_rho - cal.conformal_rho
    unc_saves = unc.fixed_rho - unc.conformal_rho
    assert cal_saves > unc_saves + 0.1  # calibrated conformalizes, uncalibrated does not
    assert cal.score_div_slope > unc.score_div_slope  # the link is real for the calibrated signal
    # Both hit coverage regardless of signal (conformal validity is signal-agnostic).
    assert cal.test_undetected <= 0.05 + 0.06
    assert unc.test_undetected <= 0.05 + 0.06


def test_cf4_calibration_curve_monotone() -> None:
    curves = calibration_curves(CF4Config(n_rollouts=12, steps=40, n_seeds=4, n_score_bins=3))
    cal = sorted((p for p in curves if p.arm == "calibrated"), key=lambda p: p.score_bin)
    assert cal[-1].mean_divergence >= cal[0].mean_divergence  # higher score -> higher divergence


def test_cf2_static_drifts_aci_recovers() -> None:
    stats = run_cf2(CF2Config(
        cal_rollouts=12, cal_steps=40, test_rollouts=16, test_steps=60, n_seeds=4, n_depth_buckets=4
    ))
    static = sorted((s for s in stats if s.policy == "static"), key=lambda s: s.depth)
    aci = sorted((s for s in stats if s.policy == "aci"), key=lambda s: s.depth)
    # Static's undetected rate climbs with depth (the exchangeability violation).
    assert static[-1].undetected > static[0].undetected
    # ACI's long-run rate is below static's (it recovers coverage from the free oracle feedback).
    assert mean([c.undetected for c in aci]) < mean([c.undetected for c in static])


def test_cf3_risk_control_saves_budget() -> None:
    stats = run_cf3(CF3Config(n_rollouts=12, steps=40, n_seeds=4, alphas=(0.10, 0.20)))
    for s in stats:
        assert s.risk_rho <= s.cov_rho + 1e-9  # risk control consults no more than coverage-only
        assert s.risk_loss <= s.alpha + 0.05  # ...while certifying the graded loss


def test_cf_determinism() -> None:
    cfg = CF1Config(n_rollouts=10, steps=40, n_seeds=3, alphas=(0.1,))
    a = [(s.alpha, s.conformal_rho) for s in run_cf1(cfg)]
    b = [(s.alpha, s.conformal_rho) for s in run_cf1(cfg)]
    assert a == b


def _cf5_tiny() -> CF5Config:
    return CF5Config(n_rollouts=10, steps=40, n_seeds=4, alphas=(0.05, 0.10, 0.20),
                     headline_alpha=0.10)


def test_cf5_transfers_across_worlds() -> None:
    stats = run_cf5(_cf5_tiny())
    assert {s.world for s in stats} == {"network", "host", "distributed"}
    for w in ("network", "host", "distributed"):
        cal = [s for s in stats if s.world == w and s.arm == "calibrated"]
        unc = [s for s in stats if s.world == w and s.arm == "uncalibrated"]
        # H50 transfers: the coverage gate holds per world, every α.
        for s in cal:
            assert s.test_undetected <= s.alpha + 0.06
        # H51 transfers: calibrated certifies at lower ρ than fixed on every world.
        for s in cal:
            assert s.conformal_rho <= s.fixed_rho + 1e-9
        # H53 transfers: the calibrated signal saves ρ, the uncalibrated does not -- on every world.
        cal_h = next(s for s in cal if abs(s.alpha - 0.10) < 1e-9)
        unc_h = next(s for s in unc if abs(s.alpha - 0.10) < 1e-9)
        assert cal_h.rho_saved > unc_h.rho_saved + 0.1


def test_cf5_deterministic() -> None:
    a = [(s.world, s.arm, s.alpha, round(s.rho_saved, 4)) for s in run_cf5(_cf5_tiny())]
    b = [(s.world, s.arm, s.alpha, round(s.rho_saved, 4)) for s in run_cf5(_cf5_tiny())]
    assert a == b
