"""Smoke + invariant test for EN6 counterfactual grounding (SPEC-5 §12, H5).

Tiny config so it is fast: checks the three arms train and emit well-formed intervention-prediction
metrics in [0,1] with CIs over eval seeds — not the (small-scale) learned numbers.
"""

from __future__ import annotations

from verisim.experiments.en6 import ARMS, METRICS, EN6Config, run_en6


def _tiny() -> EN6Config:
    return EN6Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=8,
        k_counterfactual=2,
        graph_iters=30,
        eval_seeds=(100, 101),
        eval_steps=6,
        m_interventions=3,
    )


def test_run_en6_smoke() -> None:
    stats = run_en6(_tiny())
    assert {s.arm for s in stats} == set(ARMS)
    assert {s.metric for s in stats} == set(METRICS)
    for s in stats:
        assert 0.0 <= s.mean <= 1.0  # both metrics are rates in [0, 1]
        assert s.ci_lo <= s.mean <= s.ci_hi
        assert s.n == 2  # two eval seeds aggregated


def test_run_en6_is_deterministic() -> None:
    a = {(s.arm, s.metric): s.mean for s in run_en6(_tiny())}
    b = {(s.arm, s.metric): s.mean for s in run_en6(_tiny())}
    assert a == b
