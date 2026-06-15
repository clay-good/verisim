"""CU29 / H122 -- the forensic oracle: the posterior dual of the targeting arc (SPEC-22 §5).

Torch-free. Synthetic worlds exercise the forensic mechanics in isolation (localization,
counterfactual ``do``-removal, root-cause attribution); the real unified arms (smoke) prove the
integration verdict -- the exact oracle attributes every breach, the omitter is blind, the
forensic and prevention surfaces converge, and the cause precedes the breach in the
genesis-separated worlds.
"""

from __future__ import annotations

from verisim.acd.forensic_oracle import (
    CU29Config,
    breach_steps,
    counterfactual_realizes,
    cu29_verdict,
    detection_rate,
    first_breach_step,
    localization_accuracy,
    model_localize,
    oracle_localize,
    replay,
    root_cause_step,
    run_cu29,
)
from verisim.acd.unified_targeting import (
    Arm,
    Danger,
    OmitterDefender,
    OracleDefender,
    Scenario,
    World,
)

# --------------------------------------------------------------------------------------------------
# A tiny synthetic world: a "plant" action arms the danger; a "trigger" realizes it iff armed.
# This makes the genesis (plant) separable from the consumption (trigger) by construction.
# --------------------------------------------------------------------------------------------------


def _plant_world() -> World:
    def advance(state: frozenset[str], action: str) -> frozenset[str]:
        return state | {"armed"} if action == "plant" else state

    return World(advance=advance)


def _plant_danger() -> Danger:
    def realizes(state: frozenset[str], action: str) -> bool:
        # "trigger" needs a prior plant; "armed_trigger" both arms and fires in one action.
        return (action == "trigger" and "armed" in state) or action == "armed_trigger"

    def attacks(state: frozenset[str]) -> list[str]:
        atk = ["armed_trigger"]
        if "armed" in state:
            atk.append("trigger")
        return atk

    return Danger(realizes=realizes, attacks=attacks)


def _arm(actions_per_scenario: list[tuple[str, ...]]) -> Arm:
    world, danger = _plant_world(), _plant_danger()
    target = danger.realizes  # the covering target = the realizing event (CU18-style)
    scenarios = [
        Scenario(frozenset(), acts, danger, target, None) for acts in actions_per_scenario
    ]
    return Arm("synthetic", "armed trigger", "realizes", None, world, scenarios, 8)


def test_replay_flags_exactly_the_realizing_steps():
    arm = _arm([("plant", "noop", "trigger")])
    trace = replay(arm, arm.scenarios[0])
    assert [s.realizes for s in trace] == [False, False, True]
    assert breach_steps(trace) == [2]
    assert first_breach_step(trace) == 2


def test_oracle_localizes_every_breach_exactly():
    arm = _arm([("plant", "trigger"), ("armed_trigger",), ("noop", "plant", "noop", "trigger")])
    assert localization_accuracy(arm, None) == 1.0  # the exact oracle is always right
    for sc in arm.scenarios:
        trace = replay(arm, sc)
        assert oracle_localize(trace) == first_breach_step(trace)


def test_omitter_is_forensically_blind():
    arm = _arm([("plant", "trigger"), ("armed_trigger",)])
    omitter = OmitterDefender()
    assert localization_accuracy(arm, omitter) == 0.0  # cannot locate the step it omitted
    assert detection_rate(arm, omitter) == 0.0  # cannot even tell an incident happened
    for sc in arm.scenarios:
        assert model_localize(replay(arm, sc), omitter) is None


def test_perfect_model_localizes_exactly():
    arm = _arm([("plant", "trigger"), ("armed_trigger",)])
    perfect = OracleDefender(arm.scenarios[0].danger.realizes)
    assert localization_accuracy(arm, perfect) == 1.0  # the recall-1 control attributes exactly
    assert detection_rate(arm, perfect) == 1.0


def test_counterfactual_removal_eliminates_the_breach():
    arm = _arm([("plant", "noop", "trigger")])
    sc = arm.scenarios[0]
    assert counterfactual_realizes(arm, sc, frozenset()) is True  # the factual trace breaches
    assert counterfactual_realizes(arm, sc, frozenset({0})) is False  # do(remove plant) averts it
    assert counterfactual_realizes(arm, sc, frozenset({2})) is False  # do(remove trigger) too


def test_root_cause_precedes_breach_when_genesis_is_separated():
    """plant (idx 0) -> trigger (idx 2): the cause is the upstream plant, not the realizing step."""
    arm = _arm([("plant", "noop", "trigger")])
    sc = arm.scenarios[0]
    assert first_breach_step(replay(arm, sc)) == 2
    assert root_cause_step(arm, sc) == 0  # the earliest averting intervention is the genesis


def test_root_cause_equals_breach_when_single_action():
    """armed_trigger arms and fires in one step: genesis == consumption, the cause is the breach."""
    arm = _arm([("armed_trigger",)])
    sc = arm.scenarios[0]
    assert first_breach_step(replay(arm, sc)) == 0
    assert root_cause_step(arm, sc) == 0  # no upstream cause -> lag 0


def test_root_cause_never_after_the_breach():
    arm = _arm([("plant", "trigger"), ("armed_trigger",), ("plant", "noop", "noop", "trigger")])
    for sc in arm.scenarios:
        breach = first_breach_step(replay(arm, sc))
        cause = root_cause_step(arm, sc)
        assert cause is not None and breach is not None and cause <= breach


# --------------------------------------------------------------------------------------------------
# Integration on the real unified arms (smoke): the four-world forensic verdict.
# --------------------------------------------------------------------------------------------------


def test_verdict_oracle_attributes_model_blind_surfaces_converge():
    result = run_cu29(CU29Config.smoke())
    v = cu29_verdict(result)
    assert v["oracle_attributes_every_breach"] is True
    assert v["model_is_forensically_blind"] is True
    assert v["model_cannot_detect_incident"] is True
    assert v["forensic_and_prevention_surfaces_converge"] is True


def test_verdict_genesis_separation_reappears_as_root_cause():
    """At full battery the cause precedes the breach in the host/dist worlds; on the smoke battery
    at least one world shows the upstream cause (the genesis-grammar boundary, read backward)."""
    result = run_cu29(CU29Config.smoke())
    v = cu29_verdict(result)
    assert v["genesis_separation_reappears_as_root_cause"] is True
    # host and distributed each have a temporally separated genesis -> some upstream cause.
    by_world = {a.world_name: a for a in result.arms}
    assert by_world["host"].fraction_cause_precedes_breach > 0.0
