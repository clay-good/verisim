"""Tests for SPEC-22 CU21 (H114): the unified target -- one model-free rule across all worlds.

Two layers. A tiny synthetic world makes the un-gameability *theorem* crisp and fast: a covering,
model-free target blocks every adversarial placement regardless of the defender model, while a
non-covering shortcut leaks. Then the four real arms (network exfil, host corruption, segmentation,
distributed staleness) are run on the smoke battery under the worst-case omitter, confirming the
same structural result holds in every grounded world. Torch-free; deterministic.
"""

from __future__ import annotations

from verisim.acd.dist_targeting import CU18Config
from verisim.acd.host_targeting import CU16Config
from verisim.acd.segmentation_targeting import CU17Config
from verisim.acd.targeted_verification import CU10Config
from verisim.acd.unified_targeting import (
    Arm,
    Danger,
    OmitterDefender,
    OracleDefender,
    Scenario,
    World,
    adversarial_scenario,
    arm_verdict,
    covers,
    cu21_verdict,
    dist_arm,
    host_arm,
    net_flow_arm,
    net_reach_arm,
    run_arm,
    run_cu21,
    run_scenario,
)

# --- a tiny synthetic world: the danger action is literally "danger" -----------------------------

_WORLD = World(advance=lambda s, a: s)  # state is irrelevant to the toy danger


def _danger() -> Danger:
    return Danger(realizes=lambda s, a: a == "danger", attacks=lambda s: ["danger"])


def _cover(s: object, a: object) -> bool:
    return a == "danger"  # covering: fires on exactly the danger action


def _no_cover(s: object, a: object) -> bool:
    return False  # a non-covering shortcut: never consults -> leaks


def _benign_scenario() -> Scenario:
    return Scenario(0, ("noop", "noop"), _danger(), _cover, _no_cover)


def _danger_scenario() -> Scenario:
    return Scenario(0, ("noop", "danger", "noop"), _danger(), _cover, _no_cover)


def test_coverage_invariant_holds_for_target_breaks_for_shortcut() -> None:
    """The theorem's hypothesis: the covering target fires on every attack; the shortcut won't."""
    sc = _benign_scenario()
    assert covers(_WORLD, sc) is True
    assert covers(_WORLD, sc, use_shortcut=True) is False


def test_covering_target_is_ungameable_for_any_defender() -> None:
    """THE THEOREM: a covering, model-free target blocks every attack -- even a blind omitter."""
    sc = _benign_scenario()
    assert adversarial_scenario(_WORLD, sc, OmitterDefender(), "target") is False
    # and the result is model-independent: it also holds when the model is the perfect oracle
    assert adversarial_scenario(_WORLD, sc, OracleDefender(sc.danger.realizes), "target") is False


def test_non_covering_shortcut_leaks() -> None:
    """Break coverage and the theorem's conclusion fails: the attack executes uncaught."""
    sc = _benign_scenario()
    assert adversarial_scenario(_WORLD, sc, OmitterDefender(), "target", use_shortcut=True) is True


def test_free_agent_is_gameable() -> None:
    """An unverified agent (uniform ρ=0 + omitter) is gameable: the attacker always wins."""
    sc = _benign_scenario()
    assert adversarial_scenario(_WORLD, sc, OmitterDefender(), "uniform", 0.0) is True


def test_run_scenario_target_blocks_danger_cheaply() -> None:
    """On a mission with a real danger action: free breaches; the target blocks it at one call."""
    sc = _danger_scenario()
    assert run_scenario(_WORLD, sc, OmitterDefender(), "uniform", 0.0) == (True, 0)
    assert run_scenario(_WORLD, sc, OmitterDefender(), "target") == (False, 1)


def test_perfect_model_self_governs() -> None:
    """A faithful model (recall 1) needs no oracle: it gates the danger itself at zero calls."""
    sc = _danger_scenario()
    out = run_scenario(_WORLD, sc, OracleDefender(sc.danger.realizes), "uniform", 0.0)
    assert out == (False, 0)


# --- the four real arms on the smoke battery -----------------------------------------------------

_RHOS = (0.0, 0.5, 1.0)


def _smoke_arms() -> list[Arm]:
    return [
        net_flow_arm(CU10Config.smoke()),
        host_arm(CU16Config.smoke()),
        dist_arm(CU18Config.smoke()),
        net_reach_arm(CU17Config.smoke()),
    ]


def test_every_arm_has_a_covering_target_that_is_safe_cheap_ungameable() -> None:
    """In every grounded world: the covering target is safe, cheap, and un-gameable."""
    for arm in _smoke_arms():
        result = run_arm(arm, _RHOS)
        v = arm_verdict(result)
        assert result.target_covers is True, arm.world_name
        assert v["target_is_safe"] is True, arm.world_name
        assert v["target_is_ungameable"] is True, arm.world_name
        assert v["target_cheaper_than_full"] is True, arm.world_name
        assert float(v["target_call_saving"]) > 1.0, arm.world_name  # type: ignore[arg-type]


def test_every_arm_model_self_targeting_fails_oracle_self_governs() -> None:
    """The omitter cannot flag its blind spots; the perfect model needs no oracle -- everywhere."""
    for arm in _smoke_arms():
        result = run_arm(arm, _RHOS)
        v = arm_verdict(result)
        assert v["model_self_targeting_fails"] is True, arm.world_name
        assert v["oracle_self_governs"] is True, arm.world_name


def test_every_arm_uniform_knee_is_gameable() -> None:
    """The blind clock's apparent knee is a mirage under adversarial timing -- in every world."""
    for arm in _smoke_arms():
        result = run_arm(arm, _RHOS)
        v = arm_verdict(result)
        assert v["uniform_is_gameable"] is True, arm.world_name


def test_shortcut_arms_break_coverage_and_leak() -> None:
    """The CU17/CU18 boundary, unified: a target from another world breaks coverage and leaks."""
    for arm in (dist_arm(CU18Config.smoke()), net_reach_arm(CU17Config.smoke())):
        result = run_arm(arm, _RHOS)
        v = arm_verdict(result)
        assert result.shortcut is not None, arm.world_name
        assert v["shortcut_covers"] is False, arm.world_name
        assert v["shortcut_leaks"] is True, arm.world_name


def test_unified_verdict_headline() -> None:
    """H114: one covering rule is safe + un-gameable + cheap in EVERY world; shortcuts leak."""
    result = run_cu21(_smoke_arms(), _RHOS)
    verdict = cu21_verdict(result)
    assert verdict["n_worlds"] == 4
    assert verdict["unified_target_safe_everywhere"] is True
    assert verdict["unified_target_ungameable_everywhere"] is True
    assert verdict["unified_target_cheaper_everywhere"] is True
    assert verdict["unified_target_covers_everywhere"] is True
    assert verdict["shortcuts_leak_everywhere"] is True
    assert verdict["shortcuts_break_coverage_everywhere"] is True
