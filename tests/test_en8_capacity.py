"""Smoke + determinism test for the H24/S3 capacity-binding frontier (SPEC-9 S3).

Tiny grid so it is fast: checks the sweep emits the residual-gap metric with well-formed CIs over
the (d_model, observed_fraction) frontier and is reproducible from seeds on CPU -- not the numbers.
"""

from __future__ import annotations

from verisim.experiments.en8_capacity import EN8CapacityConfig, run_en8_capacity


def _tiny() -> EN8CapacityConfig:
    return EN8CapacityConfig(
        world_size=8,
        n_ports=3,
        d_models=(16,),
        observed_fractions=(0.25, 0.75),
        mp_rounds=1,
        seeds=(0, 1),
        train_seeds=(0,),
        eval_seeds=(100,),
        train_steps_per_traj=10,
        eval_steps=6,
        decoder_iters=20,
    )


def test_run_en8_capacity_smoke() -> None:
    stats = run_en8_capacity(_tiny())
    assert {s.observed_fraction for s in stats} == {0.25, 0.75}
    assert any(s.metric == "residual_gap" for s in stats)
    for s in stats:
        assert s.n == 2
        assert s.ci_lo <= s.mean <= s.ci_hi


def test_run_en8_capacity_is_deterministic() -> None:
    a = {(s.d_model, s.observed_fraction, s.metric): s.mean for s in run_en8_capacity(_tiny())}
    b = {(s.d_model, s.observed_fraction, s.metric): s.mean for s in run_en8_capacity(_tiny())}
    assert a == b
