"""ED4 consistency-level (learned arm) — the absolute-predictability H20 on the real M_θ (SPEC-7).

The smoke instance of the learned consistency-level apparatus (torch extra): train a tiny flat M_θ
per consistency level, measure its free-running horizons, and check the deliverables are well-formed
and structurally correct. The substantive magnitudes (does linearizable free-run further? does the
H19 gap collapse under strong consistency?) are a quantitative finding the committed run gives; here
we pin structure: one row per level, well-formed CIs, the consistency-faithful horizon is never
shorter than the bit-faithful one (the §9.1 view forgives the in-flight medium), and the verdict
carries the absolute-predictability comparison.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.ed4_consistency_learned import (
    ED4ConsistencyLearnedConfig,
    ED4ConsistencyLearnedResult,
    run_ed4_consistency_learned,
    write_csv,
)


def _tiny() -> ED4ConsistencyLearnedConfig:
    return ED4ConsistencyLearnedConfig(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=40,
        n_layer=1,
        n_embd=32,
        block_size=384,
        eval_seeds=(100, 101),
        n_steps=12,
    )


def test_run_is_well_formed():
    result = run_ed4_consistency_learned(_tiny())
    assert isinstance(result, ED4ConsistencyLearnedResult)
    assert {r["level"] for r in result.rows} == {"linearizable", "eventual"}
    for r in result.rows:
        assert r["bit_lo"] <= r["bit_h"] <= r["bit_hi"] or r["bit_lo"] == r["bit_hi"]
        assert r["gap_lo"] <= r["gap"] <= r["gap_hi"] or r["gap_lo"] == r["gap_hi"]


def test_consistency_horizon_at_least_bit_horizon():
    # the §9.1 consistency view forgives the consistency-invisible in-flight medium (ED5/H19), so a
    # model's consistency-faithful horizon is never shorter than its bit-faithful one, at any level.
    result = run_ed4_consistency_learned(_tiny())
    for r in result.rows:
        assert r["cons_h"] >= r["bit_h"]
        assert r["gap"] >= 0.0


def test_verdict_is_well_formed():
    result = run_ed4_consistency_learned(_tiny())
    v = result.verdict
    assert v["strong_level"] == "linearizable"
    assert v["weak_level"] == "eventual"
    assert isinstance(v["strong_more_predictable"], bool)
    assert isinstance(v["gap_collapses_under_strong"], bool)


def test_is_deterministic():
    a = run_ed4_consistency_learned(_tiny())
    b = run_ed4_consistency_learned(_tiny())
    assert [r["bit_h"] for r in a.rows] == [r["bit_h"] for r in b.rows]
    assert [r["gap"] for r in a.rows] == [r["gap"] for r in b.rows]


def test_write_csv(tmp_path):
    result = run_ed4_consistency_learned(_tiny())
    out = write_csv(result, tmp_path / "ed4_consistency_learned.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,level,bit_h,")
    assert any(line.startswith("row,linearizable,") for line in lines)
    assert any(line.startswith("row,eventual,") for line in lines)
    assert any(line.startswith("verdict,") for line in lines)


def test_config_round_trips():
    cfg = ED4ConsistencyLearnedConfig.from_dict(
        {"train_iters": 20, "eval_driver": "uniform", "levels": ["eventual"]}
    )
    assert cfg.train_iters == 20
    assert cfg.eval_driver == "uniform"
    assert cfg.levels == ("eventual",)
