"""Smoke + invariant test for EN10 two-oracle grounding (SPEC-5 §12, H12).

Tiny config so it is fast: checks the measurement emits all five metrics with well-formed CIs, and
that the rate metrics are bounded in [0, 1]. Not the (small-scale) numbers.
"""

from __future__ import annotations

from verisim.experiments.en10 import METRICS, EN10Config, run_en10


def _tiny() -> EN10Config:
    return EN10Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=10,
        graph_iters=30,
        eval_difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        eval_steps=8,
    )


def test_run_en10_smoke() -> None:
    stats = run_en10(_tiny())
    assert {s.metric for s in stats} == set(METRICS)
    for s in stats:
        assert s.ci_lo <= s.mean <= s.ci_hi
        assert s.n == 2  # two eval seeds aggregated (one difficulty)
    by = {s.metric: s for s in stats}
    for rate in ("nonredundant_rate", "cp_sufficient_rate"):
        assert 0.0 <= by[rate].mean <= 1.0


def test_run_en10_is_deterministic() -> None:
    a = {s.metric: s.mean for s in run_en10(_tiny())}
    b = {s.metric: s.mean for s in run_en10(_tiny())}
    assert a == b
