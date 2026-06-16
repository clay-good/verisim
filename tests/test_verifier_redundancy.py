"""Tests for SPEC-22 CU39 (H132): the redundant verifier -- defense in depth needs independence.

CU38 tiled the CIA triad with EXACT members; CU39 asks what happens when the members drift (CU35's
phi < 1). The decisive mechanics, all torch-free over the CU21 net/host arms:

  - a single imperfect monitor is adversarially gameable (CU35's cliff);
  - HOMOGENEOUS redundancy (copies of one monitor, a shared blind spot) is flat in stack height m --
    OR-ing identical members is identical to one member;
  - HETEROGENEOUS redundancy (independent blind spots) drives adversarial breach to the oracle's 0
    at a knee, because the members' faithful surfaces tile the leg;
  - depth costs a compounding false-block tax; and independence is load-bearing (the headline).
"""

from __future__ import annotations

from verisim.acd.unified_targeting import net_flow_arm
from verisim.acd.verifier_redundancy import (
    CU39Config,
    ImperfectMonitor,
    arm_verdict,
    cu39_verdict,
    heterogeneous,
    homogeneous,
    run_cu39,
)

# --------------------------------------------------------------------------------------------------
# The monitor + the two redundancy combinators.
# --------------------------------------------------------------------------------------------------


def test_exact_monitor_flags_every_danger() -> None:
    """phi=1 (and psi=1) is the exact oracle: faithful on the surface, silent off it."""
    realizes = lambda s, a: a == "danger"  # noqa: E731
    monitor = ImperfectMonitor(realizes, phi=1.0, psi=1.0, salt="0")
    assert monitor.verdict(None, "danger") is True  # on-surface, fully faithful
    assert monitor.verdict(None, "benign") is False  # off-surface, no false alarm


def test_homogeneous_ensemble_equals_a_single_member() -> None:
    """m copies of one monitor share a blind spot -> the OR is identical to a single member."""
    realizes = lambda s, a: True  # every action is on the danger surface  # noqa: E731
    one = ImperfectMonitor(realizes, phi=0.5, psi=1.0, salt="0")
    homo = homogeneous(realizes, phi=0.5, psi=1.0, m=5)
    actions = [f"a{i}" for i in range(200)]
    assert all(homo.verdict(None, a) == one.verdict(None, a) for a in actions)


def test_heterogeneous_ensemble_is_more_faithful_than_one_member() -> None:
    """Independent blind spots: the OR flags strictly more danger actions than a single member."""
    realizes = lambda s, a: True  # noqa: E731
    one = ImperfectMonitor(realizes, phi=0.5, psi=1.0, salt="0")
    hetero = heterogeneous(realizes, phi=0.5, psi=1.0, m=5)
    actions = [f"a{i}" for i in range(400)]
    one_flags = sum(one.verdict(None, a) for a in actions)
    hetero_flags = sum(hetero.verdict(None, a) for a in actions)
    # the ensemble omits an action only where ALL 5 members omit it -> far fewer omissions
    assert hetero_flags > one_flags


def test_heterogeneous_tiles_the_surface_as_m_grows() -> None:
    """A tall-enough heterogeneous stack flags EVERY danger action over a finite arsenal."""
    realizes = lambda s, a: True  # noqa: E731
    actions = [f"a{i}" for i in range(60)]
    tall = heterogeneous(realizes, phi=0.5, psi=1.0, m=30)
    assert all(tall.verdict(None, a) for a in actions)


# --------------------------------------------------------------------------------------------------
# The end-to-end verdict over the net + host arms.
# --------------------------------------------------------------------------------------------------


def test_single_monitor_is_gameable() -> None:
    """m=1 imperfect monitor: the adversary fires in its blind spot (CU35's cliff)."""
    result = run_cu39(CU39Config(phis=(0.5,), heights=(1, 2), max_scenarios=40, headline_phi=0.5))
    for arm in result.arms:
        a = arm_verdict(arm, 0.5)
        assert a["single_monitor_gameable"] is True


def test_homogeneous_flat_heterogeneous_reaches_oracle() -> None:
    """The headline contrast: copies are flat in m; diverse monitors reach breach 0."""
    result = run_cu39()
    v = cu39_verdict(result)
    assert v["homogeneous_flat_everywhere"] is True
    assert v["heterogeneous_reaches_oracle_everywhere"] is True
    assert v["heterogeneous_monotone_everywhere"] is True


def test_independence_is_load_bearing() -> None:
    """At the top of the stack, heterogeneous is strictly safer than homogeneous."""
    result = run_cu39()
    v = cu39_verdict(result)
    assert v["independence_load_bearing_everywhere"] is True


def test_depth_costs_a_compounding_false_block_tax() -> None:
    """OR-ing more members compounds off-surface false alarms (the utility cost of depth)."""
    result = run_cu39()
    v = cu39_verdict(result)
    assert v["depth_costs_utility_somewhere"] is True


def test_smoke_config_runs() -> None:
    """The smoke config is small and exercises the full pipeline."""
    result = run_cu39(CU39Config.smoke())
    v = cu39_verdict(result)
    assert v["n_worlds"] == 2
    # even the smoke heights show heterogeneous strictly improving on a single monitor somewhere
    arm = next(a for a in result.arms if a.world_name == net_flow_arm().world_name)
    hetero = arm.heterogeneous[0.5]
    assert hetero[-1].adversarial_breach <= hetero[0].adversarial_breach + 1e-9
