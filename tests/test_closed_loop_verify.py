"""SPEC-22 CU7 / H99 -- verify-before-commit.

The contract: a free agent is unsafe, verify-before-commit reaches the zero-harm guarantee *by
construction* (every executed route is oracle-verified) at strictly lower oracle cost than verifying
everything, and most of a full-verification agent's calls are wasted on routes it would have aborted
anyway. Torch-free, fast (a controlled risk stand-in, no training).
"""

from __future__ import annotations

from verisim.acd.closed_loop_replan import build_goal
from verisim.acd.closed_loop_verify import (
    CU7Config,
    cu7_verdict,
    run_cu7,
    run_goal_verify_before_commit,
)
from verisim.oracle.reference import ReferenceOracle


def test_verify_before_commit_never_executes_a_dangerous_route():
    # the zero-harm guarantee is structural: every executed route is verified, so harm is impossible
    ref = ReferenceOracle()
    config = CU7Config.smoke()
    harms = 0
    for g in range(config.n_goals):
        routes = build_goal(config.world(), g, ref)
        outcome, _ = run_goal_verify_before_commit(config, g, routes)
        harms += outcome == "harm"
    assert harms == 0


def test_free_agent_is_unsafe_but_both_verified_strategies_are_safe():
    result = run_cu7(CU7Config())
    verdict = cu7_verdict(result)
    assert verdict["free_agent_unsafe"] is True
    assert verdict["vbc_zero_harm"] is True
    assert verdict["full_verify_zero_harm"] is True


def test_vbc_reaches_zero_harm_cheaper_than_full_verification():
    # H99: the same zero-harm guarantee, strictly fewer oracle calls
    result = run_cu7(CU7Config())
    verdict = cu7_verdict(result)
    assert verdict["vbc_cheaper_than_full_verify"] is True
    assert result.vbc.calls < result.full_verify.calls
    assert result.vbc.harm_rate <= result.full_verify.harm_rate


def test_most_of_full_verification_is_wasted():
    # the mechanism: a full-verify agent spends most of its calls on routes it would abort anyway
    result = run_cu7(CU7Config())
    assert result.full_verify.wasted_calls > 0.5 * result.full_verify.calls


def test_vbc_lies_below_the_budgeted_frontier():
    # Pareto: at a comparable cost the budgeted replanner still has nonzero harm; VBC has zero
    result = run_cu7(CU7Config())
    # the cheapest budgeted rung that costs at least as much as VBC still carries harm above VBC's
    comparable = [c for c in result.budgeted if c.calls <= result.vbc.calls]
    assert comparable  # there is a budgeted rung no more expensive than VBC
    assert max(c.harm_rate for c in comparable) > result.vbc.harm_rate
