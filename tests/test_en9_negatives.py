"""Smoke + determinism test for the EN9 k_negatives S2-recovery diagnostic (SPEC-9 S2).

Tiny grid so it is fast: checks the sweep emits the lift metrics with well-formed CIs across the
k_negatives axis and is reproducible from seeds on CPU -- not the numbers.
"""

from __future__ import annotations

from verisim.experiments.en9_negatives import EN9NegConfig, run_en9_negatives


def _tiny() -> EN9NegConfig:
    return EN9NegConfig(
        world_size=8,
        n_ports=3,
        d_model=16,
        mp_rounds=1,
        k_negatives=(4, 8),
        seeds=(0, 1),
        train_seeds=(0,),
        eval_seeds=(100,),
        train_steps_per_traj=10,
        eval_steps=6,
        contrastive_iters=20,
    )


def test_run_en9_negatives_smoke() -> None:
    stats = run_en9_negatives(_tiny())
    assert {s.k_negatives for s in stats} == {4, 8}
    assert any(s.metric == "lift_top1" for s in stats)
    for s in stats:
        assert s.n == 2
        assert s.ci_lo <= s.mean <= s.ci_hi


def test_run_en9_negatives_is_deterministic() -> None:
    a = {(s.k_negatives, s.metric): s.mean for s in run_en9_negatives(_tiny())}
    b = {(s.k_negatives, s.metric): s.mean for s in run_en9_negatives(_tiny())}
    assert a == b
