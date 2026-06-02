"""Smoke + invariant test for EN5 self-healing / online-TTT (SPEC-5 §12, H7).

Tiny config so it is fast: checks the sweep runs both arms through the same loop and emits a
well-formed H_ε(ρ) curve, and the loop invariant that ρ=1 (consult every step) reaches the ceiling
for both arms. Not the (small-scale) learned numbers.
"""

from __future__ import annotations

from verisim.experiments.en5 import ARMS, EN5Config, run_en5


def _tiny() -> EN5Config:
    return EN5Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=10,
        graph_iters=30,
        ttt_steps=2,
        difficulties={"low": "weighted"},
        eval_seeds=(100,),
        eval_steps=8,
        rhos=(0.0, 1.0),
        epsilon=0.05,
    )


def test_run_en5_smoke() -> None:
    points = run_en5(_tiny())
    assert {p.arm for p in points} == set(ARMS)
    for p in points:
        assert p.ci_lo <= p.mean <= p.ci_hi
        assert 0.0 <= p.mean <= 8.0  # within [0, ceiling T]


def test_en5_full_consult_reaches_ceiling() -> None:
    pts = {(p.arm, p.rho): p.mean for p in run_en5(_tiny())}
    # ρ=1 consults (and hard-resets) every step, so both arms stay bit-exact the whole rollout.
    for arm in ARMS:
        assert pts[(arm, 1.0)] == 8.0
