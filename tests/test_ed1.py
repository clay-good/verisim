"""ED1 — the distributed H_ε(ρ) curve + the H17 tiered-oracle measurement (SPEC-7 §0, DS6).

The smoke instance of the prime-directive apparatus (dependency-free, GPU-free): a tiny seeded sweep
that checks the curve and the H17 cells are well-formed and have the right qualitative shape — the
``H_ε(ρ)`` curve rises from a floor toward the ceiling, and the H17 tradeoff is real (a tier is
cheap-per-faithful-step on gross errors but ruinous on subtle ones). The committed figure comes from
the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed1 import CURVE_TIERS, ED1Config, ED1Result, run_ed1, write_csv


def _tiny() -> ED1Config:
    return ED1Config(eval_seeds=(100, 101), n_steps=16, rhos=(0.0, 0.5, 1.0))


def test_run_ed1_is_well_formed():
    result = run_ed1(_tiny())
    # the curve has one point per ρ, each with a horizon and a CI
    assert [p["rho"] for p in result.curve] == [0.0, 0.5, 1.0]
    for p in result.curve:
        assert p["ci_lo"] <= p["h_eps"] <= p["ci_hi"] or p["ci_lo"] == p["ci_hi"]
    # the H17 panel has every (mode, tier) cell
    cells = {(c["mode"], c["tier"]) for c in result.h17}
    assert cells == {(m, t) for m in ("gross", "subtle") for t in CURVE_TIERS}


def test_curve_rises_from_floor_to_ceiling():
    result = run_ed1(_tiny())
    floor = result.curve[0]["h_eps"]   # ρ=0
    ceil = result.curve[-1]["h_eps"]   # ρ=1
    assert floor <= ceil
    assert ceil == 16.0  # full consultation reproduces truth over all 16 steps


def test_h17_tradeoff_is_real():
    result = run_ed1(_tiny())
    by = {(c["mode"], c["tier"]): c for c in result.h17}
    # bit-exact reaches full horizon for both error classes (it catches everything)
    assert by[("gross", "bit_exact")]["h_eps"] == 16.0
    assert by[("subtle", "bit_exact")]["h_eps"] == 16.0
    # the cheap metamorphic tier is far cheaper per faithful step on GROSS errors than on SUBTLE
    # ones (it catches the gross out-of-vocab errors but misses the subtle in-flight ones -> drift)
    assert (by[("gross", "metamorphic")]["dollars_per_step"]
            < by[("subtle", "metamorphic")]["dollars_per_step"])


def test_write_csv(tmp_path):
    result = run_ed1(_tiny())
    out = write_csv(result, tmp_path / "ed1.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("curve,") for line in lines)
    assert any(line.startswith("h17,") for line in lines)


def test_config_round_trips():
    assert ED1Config.from_dict({"noise": 0.3, "driver": "uniform", "rhos": [0.0, 1.0]}).noise == 0.3
    assert isinstance(run_ed1(ED1Config(eval_seeds=(100,), n_steps=8, rhos=(1.0,))), ED1Result)
