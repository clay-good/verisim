"""Tests for SPEC-22 CU33 (H126): the value of the oracle -- the cost-optimal verification policy.

Two layers. Pure-function unit tests on tiny synthetic policy points make the economic mechanism
crisp and fast: the structure target Pareto-dominates the full oracle, the critical ratio equals
``calls_structure``, and no non-covering policy is ever cost-optimal under an adversary. Then the
real arms are run on the smoke battery to confirm the dial collapses (sloped vs nature, flat vs an
adversary) and the decision rule holds in the grounded worlds. Torch-free; deterministic.
"""

from __future__ import annotations

from verisim.acd.oracle_value import (
    CU33Config,
    PolicyPoint,
    critical_ratio,
    cu33_verdict,
    dominates,
    expected_loss,
    optimal_policy,
    run_cu33,
)
from verisim.acd.unified_targeting import host_arm, net_flow_arm

# --- synthetic policy points (the canonical arc shape) -------------------------------------------

_FREE = PolicyPoint("free", "free", adversarial_breach=1.0, random_breach=1.0, calls=0.0)
_UNIFORM = PolicyPoint(
    "uniform_knee", "knee", adversarial_breach=1.0, random_breach=0.66, calls=24.0
)
_MODEL = PolicyPoint("model", "model", adversarial_breach=1.0, random_breach=1.0, calls=0.07)
_STRUCT = PolicyPoint("structure", "struct", adversarial_breach=0.0, random_breach=0.0, calls=4.0)
_FULL = PolicyPoint("full_oracle", "full", adversarial_breach=0.0, random_breach=0.0, calls=48.0)
_ALL = [_FREE, _UNIFORM, _MODEL, _STRUCT, _FULL]


def test_expected_loss_is_breach_cost_times_breach_plus_call_cost_times_calls() -> None:
    # L = C*p + c*calls; worst-case (adversarial) by default
    assert expected_loss(_STRUCT, 100.0, 1.0) == 4.0  # 100*0 + 1*4
    assert expected_loss(_FREE, 100.0, 1.0) == 100.0  # 100*1 + 1*0
    # the random axis differs from the adversarial axis for a gameable policy
    assert expected_loss(_UNIFORM, 100.0, 1.0, adversarial=True) == 100.0 + 24.0
    assert expected_loss(_UNIFORM, 100.0, 1.0, adversarial=False) == 66.0 + 24.0


def test_structure_pareto_dominates_full_oracle() -> None:
    # same zero breach, strictly fewer calls -> the full oracle is never cost-optimal
    assert dominates(_STRUCT, _FULL)
    assert not dominates(_FULL, _STRUCT)


def test_critical_ratio_equals_structure_calls_against_the_free_baseline() -> None:
    # free breach 1, structure breach 0, free calls 0 -> threshold = structure calls
    assert critical_ratio(_STRUCT, _FREE) == 4.0


def test_optimal_policy_switches_free_to_structure_at_the_threshold() -> None:
    # below the threshold accept the loss; above it cover the surface; full never wins
    assert optimal_policy(_ALL, 2.0, 1.0).name == "free"
    assert optimal_policy(_ALL, 10.0, 1.0).name == "structure"
    assert optimal_policy(_ALL, 1000.0, 1.0).name == "structure"  # full oracle still dominated


def test_noncovering_policies_are_never_optimal_under_an_adversary() -> None:
    # uniform/model have adversarial breach 1 -> dominated by free (cheaper, same breach)
    for ratio in (0.5, 4.0, 50.0, 500.0):
        assert optimal_policy(_ALL, ratio, 1.0).name in ("free", "structure")


# --- the real arms on the smoke battery ----------------------------------------------------------

def _smoke_config() -> CU33Config:
    return CU33Config(arm_factory=lambda: [net_flow_arm(), host_arm()])


def test_real_arms_structure_dominates_full_and_threshold_is_small() -> None:
    result = run_cu33(_smoke_config())
    v = cu33_verdict(result)
    assert v["structure_dominates_full_everywhere"] is True
    assert v["full_oracle_never_optimal_everywhere"] is True
    assert v["noncovering_never_optimal_everywhere"] is True
    # the deploy threshold is a handful of oracle calls, far below the full oracle
    assert isinstance(v["max_critical_ratio"], float)
    assert 0.0 < v["max_critical_ratio"] < 12.0


def test_real_arms_dial_is_sloped_vs_nature_and_flat_vs_adversary() -> None:
    result = run_cu33(_smoke_config())
    v = cu33_verdict(result)
    assert v["nature_dial_is_sloped"] is True
    assert v["adversary_dial_is_flat"] is True
    assert v["dial_collapses_under_adversary"] is True


def test_real_arms_savings_match_the_arc() -> None:
    # structure is ~12x cheaper (net) / ~14x (host) than the full oracle
    result = run_cu33(_smoke_config())
    savings = {a.world_name: a.structure_call_saving for a in result.arms}
    assert savings["network"] > 5.0
    assert savings["host"] > 5.0
