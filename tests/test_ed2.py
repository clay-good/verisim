"""ED2 — equal-dollar-budget when × which-tier (H17/H18, SPEC-7 §12, DS7).

The smoke instance of the equal-dollar-budget apparatus (dependency-free, GPU-free): a tiny seeded
sweep that checks the frontier is well-formed and that the central H17 verdict has the right
*mode-dependent* shape — at an equal oracle-dollar budget a cheap tier buys more horizon than
bit-exact when the proposer makes cheaply-catchable (`gross`) errors, and loses when it makes
bit-exact-only (`subtle`) errors. The committed figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed2 import (
    POLICIES,
    ED2Config,
    ED2Result,
    _interp_horizon,
    run_ed2,
    write_csv,
)


def _tiny() -> ED2Config:
    return ED2Config(eval_seeds=(100, 101, 102), n_steps=16, rhos=(0.0, 0.5, 1.0))


def test_run_ed2_is_well_formed():
    result = run_ed2(_tiny())
    # the frontier has one (dollars, horizon) cell per (mode, policy, ρ)
    keys = {(p["mode"], p["policy"], p["rho"]) for p in result.frontier}
    rhos = (0.0, 0.5, 1.0)
    expected = {(m, pol, r) for m in ("gross", "subtle") for pol in POLICIES for r in rhos}
    assert keys == expected
    for p in result.frontier:
        assert p["dollars"] >= 0.0
        assert p["ci_lo"] <= p["h_eps"] <= p["ci_hi"] or p["ci_lo"] == p["ci_hi"]
    # one verdict per mode, carrying the equal-budget H17 winner + the H18 ratio
    assert {v["mode"] for v in result.verdict} == {"gross", "subtle"}


def test_frontier_rises_with_dollars_for_bit_exact():
    result = run_ed2(_tiny())
    for mode in ("gross", "subtle"):
        cells = (p for p in result.frontier
                 if p["mode"] == mode and p["policy"] == "bit_exact")
        be = sorted(cells, key=lambda p: p["rho"])
        # ρ=0 spends nothing and sits at the floor; ρ=1 spends the most and reaches the ceiling
        assert be[0]["dollars"] == 0.0
        assert be[-1]["dollars"] > 0.0
        assert be[-1]["h_eps"] >= be[0]["h_eps"]
        assert be[-1]["h_eps"] == 16.0  # full bit-exact consultation reproduces truth over 16 steps


def test_h17_verdict_is_mode_dependent():
    """The central H17 claim: tiering wins per equal dollar on gross errors, loses on subtle."""
    result = run_ed2(_tiny())
    by_mode = {v["mode"]: v for v in result.verdict}
    # gross (cheaply-catchable) -> a non-bit-exact policy buys more horizon at the same budget
    assert by_mode["gross"]["h17_tiering_wins"] is True
    assert by_mode["gross"]["h17_winner"] != "bit_exact"
    # subtle (bit-exact-only) -> bit-exact wins at equal budget; tiering does not help
    assert by_mode["subtle"]["h17_tiering_wins"] is False
    assert by_mode["subtle"]["h17_winner"] == "bit_exact"
    # the H18 competitive ratio is a fraction of the full-truth ceiling, and the cheap-tier-useless
    # subtle mode has a strictly lower ratio at the same sub-linear budget (its honest negative)
    assert 0.0 <= by_mode["subtle"]["competitive_ratio"] <= by_mode["gross"]["competitive_ratio"]


def test_interp_horizon_uses_pareto_envelope():
    """Below cheapest -> floor; above dearest -> ceiling; non-monotone dips ignored."""
    points = [
        {"dollars": 0.0, "h_eps": 1.0},
        {"dollars": 100.0, "h_eps": 8.0},
        {"dollars": 90.0, "h_eps": 5.0},   # a cheaper-but-lower point: the envelope must not dip
        {"dollars": 200.0, "h_eps": 10.0},
    ]
    assert _interp_horizon(points, -5.0) == 1.0          # clamped to the floor
    assert _interp_horizon(points, 1000.0) == 10.0       # clamped to the ceiling
    # at $50 the envelope interpolates up from the $0 floor toward the running max (8.0 at $100)
    mid = _interp_horizon(points, 50.0)
    assert 1.0 < mid < 8.0


def test_write_csv(tmp_path):
    result = run_ed2(_tiny())
    out = write_csv(result, tmp_path / "ed2.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("frontier,") for line in lines)
    assert any(line.startswith("verdict,") for line in lines)


def test_config_round_trips():
    cfg = ED2Config.from_dict({"noise": 0.3, "driver": "uniform", "rhos": [0.0, 1.0]})
    assert cfg.noise == 0.3
    assert cfg.driver == "uniform"
    assert isinstance(run_ed2(ED2Config(eval_seeds=(100,), n_steps=8, rhos=(0.0, 1.0))), ED2Result)
