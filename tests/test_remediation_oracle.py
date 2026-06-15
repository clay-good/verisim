"""CU30 / H123 -- the remediation oracle: the recovery dual of the forensic oracle (SPEC-22 §5).

Torch-free. A synthetic "fire-once" world with a *redundant consumer* exercises the recovery
mechanics in isolation -- the key non-obvious finding that undoing the realizing action does not
undo the breach (a second consumer re-triggers it), so only removing the genesis (or the oracle's
minimal certified set) averts. The real unified arms (smoke) prove the four-world integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from verisim.acd.forensic_oracle import ExecutedStep, breach_steps, replay
from verisim.acd.remediation_oracle import (
    CU30Config,
    averts,
    collateral,
    cu30_verdict,
    min_certified,
    model_remediation,
    root_cause_remediation,
    run_cu30,
    sledgehammer,
    surgical,
)
from verisim.acd.unified_targeting import Arm, Danger, OmitterDefender, Scenario, World

# --------------------------------------------------------------------------------------------------
# A synthetic "fire-once" world: a plant arms the danger; a trigger fires it ONCE (the danger state
# changes after the first consumption, so a second trigger reads realizes=False in the factual trace
# but RE-realizes once the first is removed). This reproduces the host redundant-consumer structure
# (two writes corrupt one file; deleting the first lets the second corrupt it) in isolation.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _A:
    name: str


def _fire_once_arm(actions_per_scenario: list[tuple[_A, ...]]) -> Arm:
    def advance(state: frozenset[str], a: _A) -> frozenset[str]:
        if a.name == "plant":
            return state | {"armed"}
        if a.name in ("trigger", "armed_trigger"):
            return state | {"fired"}
        return state

    def realizes(state: frozenset[str], a: _A) -> bool:
        if a.name == "armed_trigger":
            return "fired" not in state
        return a.name == "trigger" and "armed" in state and "fired" not in state

    def attacks(state: frozenset[str]) -> list[_A]:
        atk = [_A("armed_trigger")]
        if "armed" in state and "fired" not in state:
            atk.append(_A("trigger"))
        return atk

    danger = Danger(realizes=realizes, attacks=attacks)
    world = World(advance=advance)
    target = danger.realizes
    scenarios = [Scenario(frozenset(), acts, danger, target, None) for acts in actions_per_scenario]
    return Arm("synthetic", "fire-once danger", "realizes", None, world, scenarios, 8)


_PLANT, _TRIG, _NOOP = _A("plant"), _A("trigger"), _A("noop")
_ARMED_TRIG = _A("armed_trigger")


def _redundant_scenario() -> tuple[Arm, Scenario, list[ExecutedStep]]:
    arm = _fire_once_arm([(_PLANT, _TRIG, _TRIG)])
    sc = arm.scenarios[0]
    return arm, sc, replay(arm, sc)


# --------------------------------------------------------------------------------------------------
# Certification: the oracle re-runs the counterfactual and scores both axes.
# --------------------------------------------------------------------------------------------------


def test_averts_and_collateral_basics():
    arm, sc, trace = _redundant_scenario()
    assert breach_steps(trace) == [1]  # only the FIRST trigger realizes in the factual trace
    assert averts(arm, sc, frozenset()) is False  # no remediation -> the breach stands
    assert averts(arm, sc, frozenset({0})) is True  # do(remove plant) averts
    # collateral counts only benign (non-realizing) blocked actions
    assert collateral(trace, frozenset({0})) == 1  # plant is benign -> 1
    assert collateral(trace, frozenset({1})) == 0  # the trigger realized -> 0


# --------------------------------------------------------------------------------------------------
# THE HEADLINE: undoing the realizing action does not undo the breach (redundant consumer).
# --------------------------------------------------------------------------------------------------


def test_surgical_undo_fails_under_redundant_consumer():
    arm, sc, trace = _redundant_scenario()
    surg = surgical(trace)
    assert surg == frozenset({1})  # the naive undo removes exactly the realizing action
    assert averts(arm, sc, surg) is False  # ...but the SECOND trigger re-realizes the danger


def test_root_cause_and_min_certified_avert_the_redundant_breach():
    arm, sc, trace = _redundant_scenario()
    rc = root_cause_remediation(arm, sc)
    assert rc == frozenset({0})  # the genesis (plant) is the robust fix
    assert averts(arm, sc, rc) is True
    mc = min_certified(arm, sc, trace)
    assert averts(arm, sc, mc) is True  # the certified fix always averts
    assert collateral(trace, mc) <= collateral(trace, sledgehammer(trace, sc))  # cheaper


def test_surgical_suffices_when_genesis_equals_consumption():
    """A single-action danger (armed_trigger) has no upstream genesis: the surgical undo averts."""
    arm = _fire_once_arm([(_NOOP, _ARMED_TRIG)])
    sc = arm.scenarios[0]
    trace = replay(arm, sc)
    surg = surgical(trace)
    assert surg == frozenset({1})
    assert averts(arm, sc, surg) is True  # removing the one realizing action suffices
    assert collateral(trace, min_certified(arm, sc, trace)) == 0  # zero-collateral certified fix


# --------------------------------------------------------------------------------------------------
# The other policies.
# --------------------------------------------------------------------------------------------------


def test_sledgehammer_averts_but_pays_collateral():
    arm, sc, trace = _redundant_scenario()
    sl = sledgehammer(trace, sc)
    assert sl == frozenset({1, 2})  # disable the whole "trigger" class
    assert averts(arm, sc, sl) is True
    assert collateral(trace, sl) == 1  # the second (benign) trigger is collateral


def test_model_remediation_is_empty_for_the_omitter():
    arm, sc, trace = _redundant_scenario()
    blocked = model_remediation(trace, OmitterDefender())
    assert blocked == frozenset()  # the omitter sees no incident -> the empty fix
    assert averts(arm, sc, blocked) is False  # recovery indistinguishable from doing nothing


# --------------------------------------------------------------------------------------------------
# Integration on the real unified arms (smoke): the four-world recovery verdict.
# --------------------------------------------------------------------------------------------------


def test_verdict_min_certified_is_the_all_good_corner():
    v = cu30_verdict(run_cu30(CU30Config.smoke()))
    assert v["min_certified_averts_every_incident"] is True
    assert v["min_certified_cheaper_than_sledgehammer"] is True
    assert v["sledgehammer_averts_every_incident"] is True
    assert v["model_remediation_is_empty"] is True
    assert v["min_certified_dominates_surgical"] is True


def test_verdict_surgical_undo_fails_and_collateral_is_the_redundancy_tax():
    v = cu30_verdict(run_cu30(CU30Config.smoke()))
    assert v["surgical_undo_fails_in_separated_world"] is True
    assert v["collateral_is_redundancy_tax"] is True
    # the host world: deleting the realizing write does not avert (a redundant write re-corrupts).
    arms = {a["world"]: a for a in cast(list[dict[str, Any]], v["arms"])}
    assert arms["host"]["policies"]["surgical"]["avert_rate"] < 1.0
    assert arms["host"]["policies"]["min_certified"]["avert_rate"] >= 1.0 - 1e-9
