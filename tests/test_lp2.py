"""Smoke + invariant test for LP2 the faithful landmark graph (SPEC-12 §6, H32).

Tiny config so it is fast: checks the measurement emits all five metrics with well-formed CIs, that
the rates are bounded in [0, 1], that the verified residual false-edge rate is exactly 0 (the
zero-false-paths guarantee), and that the run is deterministic. Not the (small-scale) magnitudes.
"""

from __future__ import annotations

from verisim.experiments.lp2 import METRICS, LP2Config, run_lp2


def _tiny() -> LP2Config:
    return LP2Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=12,
        graph_d_model=24,
        graph_iters=40,
        eval_difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        eval_steps=16,
    )


def test_run_lp2_smoke() -> None:
    stats = run_lp2(_tiny())
    assert {s.metric for s in stats} == set(METRICS)
    by = {s.metric: s for s in stats}
    for s in stats:
        assert s.n > 0
        assert s.ci_lo <= s.mean <= s.ci_hi
    for rate in (
        "edge_precision", "edge_recall", "false_edge_rate",
        "verified_residual_false_rate", "consult_bits_ratio",
    ):
        assert 0.0 <= by[rate].mean <= 1.0
    # The SPEC-12 §8 guarantee: the verified graph has zero false edges, by construction.
    assert by["verified_residual_false_rate"].mean == 0.0


def test_run_lp2_is_deterministic() -> None:
    a = {s.metric: s.mean for s in run_lp2(_tiny())}
    b = {s.metric: s.mean for s in run_lp2(_tiny())}
    assert a == b
