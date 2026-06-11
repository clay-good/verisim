"""FL3 structured-arm goal-horizon tests (SPEC-19 §4, H71).

The contract: the structured flagship arm trains, the wall (`H_free`) and the goal-reach battery are
co-measured on the *same* model, and the H71 verdict combines them. The smoke model is trivial; the
committed co-report comes from the local frontier run -- CI guarantees the apparatus is correct and
deterministic, not what the verdict is.
"""

from __future__ import annotations

import pytest

from verisim.experiments.flagship_goal import GoalStat, h71_verdict


def test_h71_verdict_combines_wall_and_lift():
    stats = [
        GoalStat("flat", 16, 0.2, 0.1, 0.3, 0.0, 6),
        GoalStat("landmark", 16, 0.8, 0.7, 0.9, 0.25, 6),
    ]
    # wall survives (H_free small) AND landmark beats flat at the far goal -> supported
    v = h71_verdict(0.0, stats, wall_tol=1.0)
    assert v["wall_survives"] and v["goal_lift"] == pytest.approx(0.6) and v["h71_supported"]
    # wall broken (H_free large) -> not the HS3 regime -> not supported even with a lift
    v2 = h71_verdict(10.0, stats, wall_tol=1.0)
    assert not v2["wall_survives"] and not v2["h71_supported"]
    # no lift -> not supported
    flat_stats = [
        GoalStat("flat", 16, 0.8, 0.7, 0.9, 0.0, 6),
        GoalStat("landmark", 16, 0.5, 0.4, 0.6, 0.25, 6),
    ]
    assert not h71_verdict(0.0, flat_stats)["h71_supported"]


# --- torch-gated: the real structured-arm run -----------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.flagship_goal import (  # noqa: E402
    FlagshipGoalConfig,
    run_flagship_goal,
)


def test_run_flagship_goal_co_reports_wall_and_goal_reach():
    h_free, stats = run_flagship_goal(FlagshipGoalConfig.smoke())
    assert h_free >= 0.0
    arms = {(s.arm, s.goal_distance) for s in stats}
    assert ("flat", 8) in arms and ("landmark", 8) in arms
    for s in stats:
        assert 0.0 <= s.goal_reach <= 1.0
        assert s.gr_lo <= s.goal_reach <= s.gr_hi or s.n == 1
        # landmark arm spends a positive re-grounding budget; flat spends none
        if s.arm == "flat":
            assert s.rho == 0.0


def test_run_flagship_goal_is_deterministic():
    cfg = FlagshipGoalConfig.smoke()
    a_h, a = run_flagship_goal(cfg)
    b_h, b = run_flagship_goal(cfg)
    assert a_h == b_h
    assert [(s.arm, s.goal_distance, s.goal_reach) for s in a] == [
        (s.arm, s.goal_distance, s.goal_reach) for s in b
    ]
