"""ED2-smart — the π_c smart-when policy comparison on the learned M_θ (H9, SPEC-7 §8.1; DS7).

The smoke instance of the smart-π_c apparatus (torch extra): train a tiny flat `M_θ`, compare the
three §6.1 consultation policies (`fixed` / `uncertainty` / `drift`) at a couple of interior budgets
`ρ`, and check the deliverables are well-formed and structurally correct. All three policies spend
*exactly* the same budget at each `ρ` (the runner's spend-down backstop), so the comparison isolates
*where* a policy spends — the H9 question. Whether smart beats fixed is calibration-dependent and
reported, not asserted (the flat decode-entropy signal is the standing H2/H9 null); the substantive
invariants are that every policy is realized and the per-ρ verdict is computed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.ed2_smart import (
    POLICIES,
    ED2SmartConfig,
    ED2SmartResult,
    run_ed2_smart,
    write_csv,
)


def _tiny() -> ED2SmartConfig:
    return ED2SmartConfig(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=40,
        n_layer=1,
        n_embd=32,
        block_size=384,
        eval_seeds=(100, 101),
        n_steps=12,
        rhos=(0.25, 0.5),
    )


def test_run_is_well_formed():
    result = run_ed2_smart(_tiny())
    assert isinstance(result, ED2SmartResult)
    # one cell per (policy, ρ)
    keys = {(c["policy"], c["rho"]) for c in result.cells}
    expected = {(p, r) for p in POLICIES for r in (0.25, 0.5)}
    assert keys == expected
    for c in result.cells:
        assert c["h_eps"] >= 0.0
        assert c["ci_lo"] <= c["h_eps"] <= c["ci_hi"] or c["ci_lo"] == c["ci_hi"]
    # one verdict per ρ, carrying the smart-vs-fixed comparison
    assert {v["rho"] for v in result.verdict} == {0.25, 0.5}
    for v in result.verdict:
        assert v["best_smart"] in POLICIES
        assert isinstance(v["smart_wins"], bool)


def test_equal_budget_isolates_where_not_how_much():
    """All three policies spend the same budget at each ρ, so the lift is a pure where-to-spend
    comparison: the per-ρ verdict's lift = best-smart / fixed horizon, finite & non-negative."""
    result = run_ed2_smart(_tiny())
    for v in result.verdict:
        assert v["lift"] >= 0.0
        # the verdict's smart_wins flag agrees with the horizons it was computed from
        assert v["smart_wins"] == (v["best_smart_h"] > v["fixed_h"])


def test_is_deterministic():
    a = run_ed2_smart(_tiny())
    b = run_ed2_smart(_tiny())
    assert [c["h_eps"] for c in a.cells] == [c["h_eps"] for c in b.cells]
    assert [v["smart_wins"] for v in a.verdict] == [v["smart_wins"] for v in b.verdict]


def test_write_csv(tmp_path):
    result = run_ed2_smart(_tiny())
    out = write_csv(result, tmp_path / "ed2_smart.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("cell,") for line in lines)
    assert any(line.startswith("verdict,") for line in lines)


def test_config_round_trips():
    cfg = ED2SmartConfig.from_dict(
        {"train_iters": 20, "eval_driver": "uniform", "rhos": [0.5]}
    )
    assert cfg.train_iters == 20
    assert cfg.eval_driver == "uniform"
    assert cfg.rhos == (0.5,)
