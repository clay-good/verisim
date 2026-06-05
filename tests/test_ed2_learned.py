"""ED2-learned — equal-dollar-budget on the real learned M_θ (H17/H18, SPEC-7 §12, DS7).

The smoke instance of the learned-model equal-dollar apparatus (torch extra): train a tiny flat
`M_θ`, run it through the equal-dollar frontier, and check the deliverables are well-formed and
structurally correct. The frontier has one (dollars, horizon) cell per (policy, ρ); the bit-exact
tier at ρ=1 reaches the full horizon (it corrects every refuted step, for any model); and the single
real-model verdict carries the equal-budget H17 winner + the H18 competitive ratio.

The substantive check is the **honest inverse of ED2's synthetic `gross` panel**: the constrained
decoder removes the gross error class, so a real model lives in ED2's `subtle` regime where the
cheap tiers refute nothing the bit-exact tier would not — every cheaper tier's ρ=1 horizon is
bounded above by bit-exact's, and the H18 ratio is a fraction of the full-truth ceiling. The
committed figure comes from the local run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.ed2 import POLICIES
from verisim.experiments.ed2_learned import (
    ED2LearnedConfig,
    ED2LearnedResult,
    run_ed2_learned,
    write_csv,
)


def _tiny() -> ED2LearnedConfig:
    return ED2LearnedConfig(
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
    result = run_ed2_learned(_tiny())
    assert isinstance(result, ED2LearnedResult)
    # the frontier has one (dollars, horizon) cell per (policy, ρ)
    keys = {(p["policy"], p["rho"]) for p in result.frontier}
    expected = {(pol, r) for pol in POLICIES for r in (0.0, 0.5, 1.0)}
    assert keys == expected
    for p in result.frontier:
        assert p["dollars"] >= 0.0
        assert p["ci_lo"] <= p["h_eps"] <= p["ci_hi"] or p["ci_lo"] == p["ci_hi"]
    # the single real-model verdict carries the equal-budget H17 winner + the H18 ratio
    assert result.verdict["h17_winner"] in POLICIES
    assert 0.0 <= result.verdict["competitive_ratio"] <= 1.0


def test_bit_exact_frontier_rises_to_full_horizon():
    result = run_ed2_learned(_tiny())
    be = sorted(
        (p for p in result.frontier if p["policy"] == "bit_exact"), key=lambda p: p["rho"]
    )
    # ρ=0 spends nothing and sits at the model's free-run floor; ρ=1 spends the most and reaches
    # the ceiling (bit-exact corrects every step → truth, for any model).
    assert be[0]["dollars"] == 0.0
    assert be[-1]["dollars"] > 0.0
    assert be[-1]["h_eps"] >= be[0]["h_eps"]
    assert be[-1]["h_eps"] == 12.0  # full bit-exact consultation reproduces truth over 12 steps


def test_real_model_errors_are_subtle():
    """The constrained decoder removes gross errors, so the real model lives in ED2's subtle regime:
    at ρ=1 every cheaper tier's horizon is bounded above by the full-truth bit-exact tier (it can
    only *miss* errors the bit-exact tier catches), and the H18 ratio is a bounded fraction."""
    result = run_ed2_learned(_tiny())
    rho1 = {p["policy"]: p for p in result.frontier if p["rho"] == 1.0}
    be = rho1["bit_exact"]["h_eps"]
    assert rho1["metamorphic"]["h_eps"] <= be
    assert rho1["symbolic"]["h_eps"] <= be
    assert 0.0 <= result.verdict["competitive_ratio"] <= 1.0


def test_is_deterministic():
    a = run_ed2_learned(_tiny())
    b = run_ed2_learned(_tiny())
    assert [p["h_eps"] for p in a.frontier] == [p["h_eps"] for p in b.frontier]
    assert [p["dollars"] for p in a.frontier] == [p["dollars"] for p in b.frontier]
    assert a.verdict["h17_winner"] == b.verdict["h17_winner"]


def test_write_csv(tmp_path):
    result = run_ed2_learned(_tiny())
    out = write_csv(result, tmp_path / "ed2_learned.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("frontier,") for line in lines)
    assert any(line.startswith("verdict,") for line in lines)


def test_config_round_trips():
    cfg = ED2LearnedConfig.from_dict(
        {"train_iters": 20, "eval_driver": "uniform", "rhos": [0.0, 1.0]}
    )
    assert cfg.train_iters == 20
    assert cfg.eval_driver == "uniform"
    assert cfg.rhos == (0.0, 1.0)
    assert isinstance(
        run_ed2_learned(
            ED2LearnedConfig(
                train_seeds=(0,), train_steps_per_traj=12, train_iters=20,
                n_layer=1, n_embd=32, block_size=384,
                eval_seeds=(100,), n_steps=8, rhos=(0.0, 1.0),
            )
        ),
        ED2LearnedResult,
    )
