"""Smoke + invariant test for LP1 latent planning-geometry (SPEC-12 §5-§6, H31).

Tiny config so it is fast: checks the measurement emits all five correlations with well-formed CIs,
each bounded in [-1, 1], and that the run is deterministic. Not the (gate) magnitudes - the ρ≥0.6
verdict is the empirical finding, reported in the spec, never asserted by CI (SPEC-12 §9.2).
"""

from __future__ import annotations

import math

from verisim.experiments.lp1 import METRICS, LP1Config, run_lp1


def _tiny() -> LP1Config:
    return LP1Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=10,
        graph_d_model=24,
        graph_iters=30,
        anchor_seeds=(100, 101),
        anchor_stride=4,
        anchors_per_seed=2,
        bfs_max_depth=2,
        bfs_max_nodes=40,
    )


def test_run_lp1_smoke() -> None:
    stats = run_lp1(_tiny())
    assert {s.metric for s in stats} == set(METRICS)
    for s in stats:
        assert s.n > 0
        assert not math.isnan(s.mean)
        assert s.ci_lo <= s.mean <= s.ci_hi
        assert -1.0 <= s.mean <= 1.0  # a correlation is bounded


def test_run_lp1_is_deterministic() -> None:
    a = {s.metric: s.mean for s in run_lp1(_tiny())}
    b = {s.metric: s.mean for s in run_lp1(_tiny())}
    assert a == b
