"""SPEC-22 CU5 / H97 -- the closed-loop safe agent.

The contract: danger labels are oracle-grounded (a write's delta under the protected prefix), the
oracle agent (ρ=1) is safe AND reliable while the free agent (ρ=0) is neither, and -- the design
lesson -- a stakes-aware (consult-where-uncertain) schedule reaches the safe-and-reliable corner at
a smaller budget than a uniform one. Torch-free, fast (a controlled risk stand-in, no training).
"""

from __future__ import annotations

from verisim.acd.closed_loop_agent import (
    CU5Config,
    _cells,
    _knee_rho,
    build_episode,
    cu5_verdict,
    run_cu5,
    run_episode,
)
from verisim.oracle.reference import ReferenceOracle


def test_danger_labels_are_oracle_grounded():
    # every trap truly writes the protected prefix; every goal-write does not (the oracle confirms)
    ref = ReferenceOracle()
    config = CU5Config.smoke()
    cands = build_episode(config, 3, ref)
    assert any(c.dangerous for c in cands) and any(c.is_goal for c in cands)
    for c in cands:
        assert c.dangerous == c.action.args[0].startswith(config.protected_prefix)
        assert c.is_goal == (not c.dangerous)


def test_oracle_agent_is_safe_and_reliable():
    # ρ=1: every action verified -> no missed danger, no lost progress, task always completes
    ref = ReferenceOracle()
    config = CU5Config.smoke()
    for e in range(config.n_episodes):
        cands = build_episode(config, e, ref)
        out = run_episode(config, e, cands, 1.0, "prioritized")
        assert out.unsafe == 0 and out.lost == 0 and out.completed


def test_free_agent_is_unsafe_and_unreliable():
    # ρ=0: the drifting model both executes some traps and aborts some goal-writes, over the battery
    result = run_cu5(CU5Config())
    assert result.free_unsafe >= 0.3  # often does the irreversible bad thing
    assert result.free_success <= 0.6  # often fails to finish the job


def test_stakes_aware_buys_the_knee():
    # the design lesson: prioritizing by model uncertainty reaches safe-and-reliable cheaper
    result = run_cu5(CU5Config())
    verdict = cu5_verdict(result)
    assert verdict["oracle_safe_and_reliable"] is True
    assert verdict["free_unsafe_and_unreliable"] is True
    assert verdict["stakes_aware_buys_the_knee"] is True
    # the typed knee values (read off the cells, not the object-valued verdict dict)
    assert _knee_rho(_cells(result, "prioritized")) < _knee_rho(_cells(result, "uniform"))


def test_both_schedules_run():
    ref = ReferenceOracle()
    config = CU5Config.smoke()
    cands = build_episode(config, 0, ref)
    for sched in ("uniform", "prioritized"):
        out = run_episode(config, 0, cands, 0.5, sched)
        assert out.unsafe >= 0 and out.lost >= 0
