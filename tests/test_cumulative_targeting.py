"""Tests for SPEC-22 CU26 (H119): the low-and-slow danger -- target the accumulator, not the action.

Torch-free throughout: the schedule keys on the exact oracle and the danger's model-free accumulator
boundary, so the model is the worst-case omitter (the headline substrate) or a perfect oracle
control. CU26 is a *cumulative* danger no single action realizes (mass collection / data hoarding);
the genuinely new piece is **cumulative coverage** -- the target must fire on every crossing a
multi-action low-and-slow adversary can stage, not just the benign trajectory's -- and the
``covers``-predicts-fate equivalence carried to it.
"""

from __future__ import annotations

from functools import lru_cache

from verisim.acd.cumulative_targeting import (
    CU26Config,
    CU26Result,
    _candidate,
    _distinct_sensitive,
    _make_danger,
    build_deployments,
    covers_cumulative,
    cu26_verdict,
    lowslow_breaches,
    run_cu26,
)
from verisim.net.action import NetAction
from verisim.net.state import HostState, NetworkState, can_reach, link_key
from verisim.netoracle.reference import ReferenceNetworkOracle


def _small() -> CU26Config:
    return CU26Config(
        horizon=24, n_seeds=300, max_episodes=12, rhos=(0.0, 0.5, 1.0), budget_sweep=(2, 3, 4),
    )


@lru_cache(maxsize=1)
def _result() -> CU26Result:
    return run_cu26(_small())


def test_deployments_present_a_lowslow_opportunity() -> None:
    config = _small()
    oracle = ReferenceNetworkOracle()
    deps = build_deployments(config, oracle)
    assert deps, "expected at least one deployment with a stageable low-and-slow crossing"
    for d in deps:
        assert d.options, "every kept deployment must offer the adversary a crossing option"


def test_breach_options_are_real_budget_crossings() -> None:
    """Each adversary option stages budget-1 sensitive flows; the crossing pushes over budget B."""
    config = _small()
    oracle = ReferenceNetworkOracle()
    danger, _ = _make_danger(oracle, config)
    sensitive = frozenset(config.sensitive)
    deps = build_deployments(config, oracle)
    for d in deps:
        for acc, cross in d.options:
            assert len(_distinct_sensitive(acc, sensitive)) == config.budget - 1  # staged to brink
            assert danger.realizes(acc, cross)  # the crossing truly realizes the cumulative danger


def test_lowslow_unit_magnitude_blind_to_a_nonjewel_crossing() -> None:
    """The decisive unit test: a low-and-slow collection crosses the budget over NON-jewel hosts.

    Two jewel flows (h0, h4) are already held (budget-1 = 2); the crossing connect to the non-jewel
    h1 pushes the accumulator to B=3. It truly realizes the danger and is on the grammar + closure
    surfaces -- but OFF the magnitude / DLP surface (h1 is not a high-value jewel) -- exactly the
    blind spot ``covers`` predicts the value heuristic leaks through.
    """
    oracle = ReferenceNetworkOracle()
    state = NetworkState(
        hosts={
            "h0": HostState(up=True, services=(0,)),  # jewel, already collected
            "h4": HostState(up=True, services=(0,)),  # jewel, already collected
            "h1": HostState(up=True, services=(0,)),  # non-jewel sensitive host (the crossing)
            "h2": HostState(up=True),  # the collecting source
        },
        links={link_key("h2", "h0"), link_key("h2", "h1"), link_key("h2", "h4")},
        flows={("h2", "h0", 0), ("h2", "h4", 0)},  # budget-1 = 2 distinct sensitive flows held
    )
    config = CU26Config()
    sensitive = frozenset(config.sensitive)
    assert _distinct_sensitive(state, sensitive) == {"h0", "h4"}
    assert can_reach(state, "h2", "h1", 0)
    _, targets = _make_danger(oracle, config)
    cross = NetAction(raw="connect h2 h1 0", name="connect", args=("h2", "h1", "0"))

    assert targets["closure"](state, cross) is True  # the crossing realizes the danger
    assert targets["grammar"](state, cross) is True  # h1 is a sensitive flow
    assert targets["magnitude"](state, cross) is False  # h1 is not a jewel -> the leak

    # a connect to a NEW non-sensitive context (a jewel that is already held) does not cross again
    repeat = NetAction(raw="connect h2 h0 0", name="connect", args=("h2", "h0", "0"))
    assert targets["closure"](state, repeat) is False  # h0 already counted -> no new crossing


def test_derived_closure_is_safe_cheap_and_ungameable() -> None:
    closure = _candidate(_result(), "closure")
    grammar = _candidate(_result(), "grammar")
    full = _result().full_oracle
    assert closure.covers is True  # cumulative coverage holds (target == realizes)
    assert closure.random_breach <= full.random_breach + 1e-9  # the oracle's safety
    assert closure.adversarial_breach <= 1e-9  # un-gameable against low-and-slow
    assert closure.mean_calls < grammar.mean_calls  # cheaper than the grammar surface
    assert closure.mean_calls < full.mean_calls  # far cheaper than verifying everything


def test_magnitude_heuristic_leaks_to_lowslow() -> None:
    """The headline negative: the real-world value / DLP heuristic is gameable by spreading load."""
    magnitude = _candidate(_result(), "magnitude")
    assert magnitude.covers is False  # cumulative coverage broken -> predicted to leak
    assert magnitude.adversarial_breach > 1e-9  # ... and it does, under low-and-slow


def test_grammar_covers_but_overpays() -> None:
    grammar = _candidate(_result(), "grammar")
    closure = _candidate(_result(), "closure")
    assert grammar.covers is True
    assert grammar.adversarial_breach <= 1e-9  # safe + un-gameable
    assert grammar.mean_calls > closure.mean_calls  # but pays for every benign contributor


def test_cumulative_covers_equivalence_on_a_single_deployment() -> None:
    """covers_cumulative(target) <=> the target-schedule low-and-slow adversary cannot breach."""
    config = _small()
    oracle = ReferenceNetworkOracle()
    _, targets = _make_danger(oracle, config)
    deps = build_deployments(config, oracle)
    for d in deps:
        for name in ("magnitude", "grammar", "closure"):
            cov = covers_cumulative(d, targets[name])
            breached = lowslow_breaches(d, targets[name], "target", 0.0)
            assert cov == (not breached)  # the theorem instantiated per deployment


def test_uniform_knee_is_a_mirage_and_model_fails() -> None:
    v = cu26_verdict(_result())
    assert v["uniform_is_gameable"] is True  # the sub-oracle clock is gameable
    assert v["model_self_targeting_fails"] is True  # the omitter never foresees its own crossing
    assert v["oracle_self_governs"] is True  # the perfect-model control


def test_cost_law_closure_falls_and_ratio_grows_with_budget() -> None:
    v = cu26_verdict(_result())
    assert v["closure_cost_falls_with_budget"] is True  # a rarer boundary at higher B
    assert v["cost_ratio_grows_with_budget"] is True  # closure's advantage widens with B


def test_framework_predicts_every_candidate() -> None:
    """The generative headline (H119): cumulative covers predicts each fate; the run confirms it."""
    v = cu26_verdict(_result())
    assert v["framework_predicts_every_candidate"] is True
    assert v["magnitude_leaks"] is True
    assert v["closure_is_ungameable"] is True
    assert v["grammar_is_ungameable"] is True


def test_run_cu26_returns_types_and_verdict() -> None:
    r = _result()
    assert r.n_episodes > 0
    assert {c.name for c in r.candidates} == {"magnitude", "grammar", "closure"}
    v = cu26_verdict(r)
    saving = v["closure_call_saving_vs_grammar"]
    assert isinstance(saving, float) and saving > 1.0  # the derived target is cheaper than grammar
