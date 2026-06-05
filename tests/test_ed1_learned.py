"""ED1-learned — the real-model distributed H_ε(ρ) curve + H17 (SPEC-7 §0, DS6).

The smoke instance of the learned-model prime directive (torch extra): train a tiny flat `M_θ`,
run it through the tiered loop, and check the deliverables are well-formed and structurally correct
— the `H_ε(ρ)` curve rises from a floor to the full-horizon ceiling (which ρ=1 reaches for *any*
model, since the bit-exact tier corrects every step), and the H17 panel records a faithful horizon
and oracle-dollars for every tier/policy arm. The committed figure comes from the local run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.ed1_learned import (
    ED1LearnedConfig,
    ED1LearnedResult,
    run_ed1_learned,
    write_csv,
)


def _tiny() -> ED1LearnedConfig:
    return ED1LearnedConfig(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=40,
        n_layer=1,
        n_embd=32,
        block_size=384,
        eval_seeds=(100, 101),
        n_steps=12,
        rhos=(0.0, 0.5, 1.0),
    )


def test_run_is_well_formed():
    result = run_ed1_learned(_tiny())
    assert isinstance(result, ED1LearnedResult)
    assert [p["rho"] for p in result.curve] == [0.0, 0.5, 1.0]
    for p in result.curve:
        assert p["ci_lo"] <= p["h_eps"] <= p["ci_hi"] or p["ci_lo"] == p["ci_hi"]
    # the H17 panel has every fixed tier plus the escalate policy
    arms = {c["tier"] for c in result.h17}
    assert arms == {"metamorphic", "symbolic", "bit_exact", "escalate"}
    for c in result.h17:
        assert c["h_eps"] >= 0.0 and c["dollars"] >= 0.0


def test_curve_rises_from_floor_to_full_horizon():
    result = run_ed1_learned(_tiny())
    floor = result.curve[0]["h_eps"]   # ρ=0 (free-run; the model's real floor)
    ceil = result.curve[-1]["h_eps"]   # ρ=1 (bit-exact every step → truth)
    assert floor <= ceil
    assert ceil == 12.0  # full consultation at the bit-exact tier reproduces truth over all steps


def test_bit_exact_reaches_full_horizon_in_h17():
    """At ρ=1 the bit-exact tier corrects every refuted step, so it reaches the full horizon."""
    result = run_ed1_learned(_tiny())
    by = {c["tier"]: c for c in result.h17}
    assert by["bit_exact"]["h_eps"] == 12.0
    # every cheaper tier's horizon is bounded above by the full-truth tier (it can only miss errors)
    assert by["metamorphic"]["h_eps"] <= by["bit_exact"]["h_eps"]
    assert by["symbolic"]["h_eps"] <= by["bit_exact"]["h_eps"]


def test_is_deterministic():
    a = run_ed1_learned(_tiny())
    b = run_ed1_learned(_tiny())
    assert [p["h_eps"] for p in a.curve] == [p["h_eps"] for p in b.curve]
    assert [c["dollars"] for c in a.h17] == [c["dollars"] for c in b.h17]


def test_write_csv(tmp_path):
    result = run_ed1_learned(_tiny())
    out = write_csv(result, tmp_path / "ed1_learned.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("curve,") for line in lines)
    assert any(line.startswith("h17,") for line in lines)


def test_config_round_trips():
    cfg = ED1LearnedConfig.from_dict(
        {"train_iters": 20, "eval_driver": "uniform", "rhos": [0.0, 1.0]}
    )
    assert cfg.train_iters == 20
    assert cfg.eval_driver == "uniform"
    assert cfg.rhos == (0.0, 1.0)
