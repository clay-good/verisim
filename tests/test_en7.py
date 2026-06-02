"""Smoke + invariant test for the EN7 model-invariance experiment (SPEC-5 §12, H22).

Tiny config so it is fast: checks the sweep runs all four proposers through the same loop and emits
a well-formed H_ε(ρ) curve per proposer, and that the reference baselines bound the plot
(oracle-backed = ceiling everywhere) -- not the (small-scale) learned numbers.
"""

from __future__ import annotations

from verisim.experiments.en7 import PROPOSERS, EN7Config, run_en7


def _tiny() -> EN7Config:
    return EN7Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=10,
        flat_iters=30,
        graph_iters=30,
        difficulties={"low": "weighted"},
        eval_seeds=(100,),
        eval_steps=8,
        rhos=(0.0, 1.0),
        epsilon=0.05,
    )


def test_run_en7_smoke() -> None:
    points = run_en7(_tiny())
    assert {p.proposer for p in points} == set(PROPOSERS)
    for p in points:
        assert p.ci_lo <= p.mean <= p.ci_hi
        assert 0.0 <= p.mean <= 8.0  # within [0, ceiling T]


def test_en7_baselines_bound_the_plot() -> None:
    pts = {(p.proposer, p.rho): p.mean for p in run_en7(_tiny())}
    # The oracle-backed proposer is perfect: full horizon at every ρ (including ρ=0).
    assert pts[("oracle", 0.0)] == 8.0
    assert pts[("oracle", 1.0)] == 8.0
    # ρ=1 consults every step, so every proposer reaches the ceiling.
    for name in PROPOSERS:
        assert pts[(name, 1.0)] == 8.0
