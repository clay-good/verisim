"""Smoke test for the EN8 experiment driver (SPEC-8 §7, OG3): both ablations run and report.

Tiny config so it is fast; it checks the experiment wires the trainer machinery together and emits
the two pre-registered ablation tables, not the (smoke-scale) numbers themselves.
"""

from __future__ import annotations

from verisim.experiments.en8 import EN8Config, run_en8


def test_run_en8_smoke() -> None:
    cfg = EN8Config(
        train_seeds=(0,),
        train_steps_per_traj=12,
        eval_seeds=(100,),
        eval_steps=8,
        d_model=16,
        mp_rounds=1,
        decoder_iters=40,
        jepa_iters=40,
    )
    results = run_en8(cfg)
    assert {r["objective"] for r in results["objective"]} == {"likelihood", "residual"}
    assert len(results["collapse"]) == 4
    for row in results["objective"]:
        assert 0.0 <= float(row["residual_acc"]) <= 1.0
    for row in results["collapse"]:
        assert float(row["emb_std"]) >= 0.0
        assert float(row["eff_rank"]) >= 1.0
