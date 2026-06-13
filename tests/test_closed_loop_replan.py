"""SPEC-22 CU6 / H98 -- the replanning agent.

The contract: routes carry oracle-grounded danger labels, replanning lifts capability (recovers the
goals a one-shot agent abandons), a *free* replanner amplifies harm (its retry loop searches the
model's blind spots) while the oracle replanner is capable AND safe (success 1.0, harm 0), and the
oracle truly aborts every dangerous route regardless of retries. Torch-free, fast (a risk stand-in).
"""

from __future__ import annotations

from verisim.acd.closed_loop_replan import (
    CU6Config,
    build_goal,
    cu6_verdict,
    run_cu6,
    run_goal,
)
from verisim.oracle.reference import ReferenceOracle


def test_routes_are_oracle_grounded_with_a_real_choice():
    # every goal has at least one safe route (achievable) and one dangerous shortcut (temptation)
    ref = ReferenceOracle()
    config = CU6Config.smoke()
    for g in range(config.n_goals):
        routes = build_goal(config, g, ref)
        assert any(r.dangerous for r in routes) and any(not r.dangerous for r in routes)
        for r in routes:
            assert r.dangerous == r.action.args[0].startswith(config.protected_prefix)


def test_oracle_replanner_is_capable_and_safe():
    # ρ=1: every route verified -> no dangerous route is ever executed, and every goal is reached
    ref = ReferenceOracle()
    config = CU6Config.smoke()
    harms = successes = 0
    for g in range(config.n_goals):
        routes = build_goal(config, g, ref)
        out = run_goal(config, g, routes, retry_budget=config.n_routes, rho=1.0)
        harms += out == "harm"
        successes += out == "success"
    assert harms == 0  # the oracle aborts every dangerous route, no matter how many retries
    assert successes == config.n_goals  # every goal reached via a safe route


def test_free_replanning_amplifies_harm():
    # H98: replanning lifts success but a FREE replanner is more dangerous than a one-shot agent
    result = run_cu6(CU6Config())
    verdict = cu6_verdict(result)
    assert verdict["replanning_lifts_capability"] is True
    assert verdict["free_replanning_amplifies_harm"] is True
    from verisim.acd.closed_loop_replan import _cell

    amp = _cell(result, "replanner", 0.0).harm_rate - _cell(result, "one_shot", 0.0).harm_rate
    assert amp > 0.0  # the free replanner is strictly more dangerous than the one-shot agent


def test_only_verified_is_capable_and_safe():
    result = run_cu6(CU6Config())
    verdict = cu6_verdict(result)
    assert verdict["grounded_replanning_is_pure_benefit"] is True
    assert verdict["only_verified_is_capable_and_safe"] is True


def test_one_shot_never_outperforms_replanner_on_success():
    # replanning can only help capability: at every ρ the replanner reaches at least as many goals
    result = run_cu6(CU6Config())
    from verisim.acd.closed_loop_replan import cells_for

    one = {c.rho: c.success_rate for c in cells_for(result, "one_shot")}
    rep = {c.rho: c.success_rate for c in cells_for(result, "replanner")}
    for rho, s in one.items():
        assert rep[rho] >= s - 1e-9
