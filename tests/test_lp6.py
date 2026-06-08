"""Smoke + invariant test for LP6 replanning policy (SPEC-12 §6, H36).

Tiny, undertrained config so it is fast: checks all three triggers emit well-formed stats over the
budget sweep, the equal-budget invariant (each policy re-grounds at exactly ``min(B, interior)``
steps, so goal reach is bounded), and determinism. Not the (small-scale) magnitudes -- whether a
trigger beats fixed-interval is the committed full-config run's verdict (pre-registered both ways).
"""

from __future__ import annotations

from verisim.experiments.lp6 import POLICIES, LP6Config, _fixed_steps, _select, run_lp6


def _tiny() -> LP6Config:
    return LP6Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=14,
        graph_d_model=24,
        graph_iters=50,
        eval_difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        goal_distance=10,
        budgets=(1, 2, 3),
    )


def test_fixed_steps_are_interior_and_budgeted() -> None:
    steps = _fixed_steps(goal_dist=12, budget=3)
    assert len(steps) == 3
    assert all(0 <= s <= 10 for s in steps)  # interior: never the goal step (11)


def test_select_spends_equal_budget_across_policies() -> None:
    signals = ([1.0] * 9, [0.5] * 9)  # 10-step journey, flat triggers
    for policy in POLICIES:
        assert len(_select(policy, signals, goal_dist=10, budget=3)) == 3


def test_run_lp6_smoke() -> None:
    stats = run_lp6(_tiny())
    assert {s.policy for s in stats} == set(POLICIES)
    assert {s.budget for s in stats} == {1.0, 2.0, 3.0}
    for s in stats:
        assert s.n > 0
        assert 0.0 <= s.goal_reach <= 1.0
        assert s.gr_lo <= s.goal_reach <= s.gr_hi


def test_run_lp6_is_deterministic() -> None:
    a = {(s.policy, s.budget): s.goal_reach for s in run_lp6(_tiny())}
    b = {(s.policy, s.budget): s.goal_reach for s in run_lp6(_tiny())}
    assert a == b
