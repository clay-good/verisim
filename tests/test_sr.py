"""Smoke + structural-invariant tests for the SPEC-13 speculative-rollout experiments (SR1-SR6).

The committed SR core is the controlled stand-in drafter (the trained-``M_θ`` arm is deferred, LP7
discipline), so the whole battery is fast and GPU-free. Tests assert *structural* invariants -- the
gradual/discrete ordering, the budget crossover, the variance/bias fork, the calibration split, the
cascade negative, the g-collapse -- on tiny configs, not magnitudes (the macOS-first principle:
the same assertions pass on the macOS primary host and Linux CI).
"""

from __future__ import annotations

from verisim.experiments.sr1 import SR1Config, _crossover, run_sr1
from verisim.experiments.sr2 import SR2Config, granularity, run_sr2
from verisim.experiments.sr3 import SR3Config, run_sr3
from verisim.experiments.sr4 import SR4Config, calibration_curve, run_sr4
from verisim.experiments.sr5 import SR5Config, run_sr5
from verisim.experiments.sr6 import SR6Config, run_sr6
from verisim.experiments.sr_common import all_worlds


def test_granularity_positive_per_world() -> None:
    cfg = SR2Config(n_steps=40, gran_seeds=3)
    for world in all_worlds():
        assert granularity(world, cfg) > 0.0


def test_sr2_prefix_grows_with_g_and_fits_law() -> None:
    cfg = SR2Config(n_steps=40, n_seeds=4, g_values=(0.5, 2.0, 8.0))
    stats = run_sr2(cfg)
    for world in {s.world for s in stats}:
        cells = sorted((s for s in stats if s.world == world), key=lambda s: s.g)
        prefixes = [s.mean_prefix for s in cells]
        assert prefixes == sorted(prefixes)  # monotone in g
        # The empirical prefix tracks the i.i.d. law fed the measured alpha-hat. Skip saturated
        # cells
        # (alpha_hat ~ 1): there the law pins to k while short tail windows pull the empirical mean
        # below k (a tiling artifact, not a law failure).
        for s in cells:
            if s.alpha_hat < 0.97:
                assert abs(s.mean_prefix - s.law_prefix) < 2.5


def test_sr1_budget_crossover_exists() -> None:
    cfg = SR1Config(n_steps=60, n_seeds=4, budgets=(2, 8, 24))
    stats = run_sr1(cfg)
    for world in {s.world for s in stats}:
        cells = [s for s in stats if s.world == world]
        hi = max(cells, key=lambda s: s.budget)
        # At a generous budget speculative reaches (near) full faithfulness.
        assert hi.spec_faithful >= 0.9
        assert _crossover(cells) is not None  # speculative overtakes fixed at some budget


def test_sr3_tree_helps_under_variance_not_bias() -> None:
    cfg = SR3Config(n_steps=48, n_seeds=4, m_values=(1, 8))
    stats = run_sr3(cfg)
    for world in {s.world for s in stats}:
        var = {s.m: s.mean_prefix for s in stats if s.world == world and s.mode == "variance"}
        bias = {s.m: s.mean_prefix for s in stats if s.world == world and s.mode == "bias"}
        assert var[8] > var[1]          # a tree raises the prefix under variance
        assert bias[8] == bias[1]       # ...and does nothing under bias (identical stalls)


def test_sr4_link_transfers_policy_does_not() -> None:
    cfg = SR4Config(n_steps=48, n_seeds=4)
    stats = run_sr4(cfg)
    cal = [s for s in stats if s.arm == "calibrated"]
    unc = [s for s in stats if s.arm == "uncalibrated"]
    # The link is real with a calibrated signal, ~flat with the null.
    assert sum(s.calib_slope for s in cal) / len(cal) > 0.05
    assert sum(s.calib_slope for s in unc) / len(unc) < 0.05
    # The policy does not transfer: calibrated-k costs no fewer oracle calls than draft-long.
    for s in cal:
        assert s.calibrated_calls >= s.draftlong_calls


def test_sr4_calibration_curve_monotone_with_signal() -> None:
    cfg = SR4Config(n_steps=48, n_seeds=4, n_conf_bins=3)
    pts = calibration_curve(cfg)
    cal = sorted((p for p in pts if p.world == "network" and p.arm == "calibrated"),
                 key=lambda p: p.conf_bin)
    # Highest-confidence bin accepts at least as often as the lowest (monotone link).
    assert cal[-1].acceptance >= cal[0].acceptance


def test_sr5_cascade_does_not_beat_large_only() -> None:
    cfg = SR5Config(n_steps=48, n_seeds=4)
    stats = run_sr5(cfg)
    for world in {s.world for s in stats}:
        cells = {s.arm: s.calls_per_faithful for s in stats if s.world == world}
        # The cascade adds an oracle round without saving any -> no cheaper than the larger drafter.
        assert cells["cascade"] >= cells["large-only"] - 1e-9


def test_sr6_win_is_hump_shaped_in_g() -> None:
    cfg = SR6Config(n_steps=60, n_seeds=4, g_values=(0.5, 1.5, 4.0))
    stats = run_sr6(cfg)
    edge = [s.win for s in stats if s.g in (0.5, 4.0)]
    middle = [s.win for s in stats if s.g == 1.5]
    assert sum(middle) / len(middle) > sum(edge) / len(edge)  # peaks in the transition band


def test_sr_determinism() -> None:
    cfg = SR2Config(n_steps=40, n_seeds=3, g_values=(1.0, 2.0))
    a = [(s.world, s.g, s.mean_prefix) for s in run_sr2(cfg)]
    b = [(s.world, s.g, s.mean_prefix) for s in run_sr2(cfg)]
    assert a == b
