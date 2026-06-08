"""Smoke + invariant test for LP5 placement policy (SPEC-12 §6, H35).

Tiny, undertrained config so it is fast: checks all four policies emit well-formed stats with
bounded goal-reach and CIs over the budget sweep, and that the run is deterministic. The pure
betweenness + selection helpers are unit-tested separately (``test_placement``); here we run the
end-to-end harness. Not the (small-scale) magnitudes -- whether informed placement beats random is
the committed full-config run's verdict (and is pre-registered both ways, SPEC-12 §6).
"""

from __future__ import annotations

from verisim.experiments.lp5 import POLICIES, LP5Config, run_lp5


def _tiny() -> LP5Config:
    return LP5Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=14,
        graph_d_model=24,
        graph_iters=50,
        landmark_seeds=(0, 1),
        landmark_steps=20,
        n_uncertainty_actions=3,
        eval_difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        goal_distance=10,
        hop_length=2,
        budgets=(1, 2, 3),
    )


def test_run_lp5_smoke() -> None:
    stats = run_lp5(_tiny())
    assert {s.policy for s in stats} == set(POLICIES)
    assert {s.budget for s in stats} == {1.0, 2.0, 3.0}
    for s in stats:
        assert s.n > 0
        assert 0.0 <= s.goal_reach <= 1.0
        assert s.gr_lo <= s.goal_reach <= s.gr_hi


def test_run_lp5_is_deterministic() -> None:
    a = {(s.policy, s.budget): s.goal_reach for s in run_lp5(_tiny())}
    b = {(s.policy, s.budget): s.goal_reach for s in run_lp5(_tiny())}
    assert a == b
