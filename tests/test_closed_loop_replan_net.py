"""SPEC-22 CU13 / H106 -- capability under real drift (torch-free core).

Validated with cheap stand-in models on the net replanning world: a *perfect* oracle model never
executes a dangerous route (so one-shot and replanner are both safe and verify-before-commit costs
one call); a *pure-omitter* (the real ``M_θ``'s profile -- says "yes" to everything) shows no
harm-amplification and no verify-before-commit saving, yet stays unsafe and keeps the zero-harm
guarantee; and the two dials separate the mechanism -- raising the false-alarm rate (recall 0) makes
replanning amplify harm, raising the recall (false-alarm 0) makes verify-before-commit cheaper. The
real trained transformer arm is the experiment's committed run (torch-gated, not exercised in CI).
"""

from __future__ import annotations

from verisim.acd.closed_loop_replan_net import (
    CU13Config,
    DialNetModel,
    Goal,
    build_goals,
    cu13_verdict,
    run_cu13,
    run_goal_full_verify,
    run_goal_replan,
    run_goal_verify_before_commit,
)
from verisim.netoracle.reference import ReferenceNetworkOracle


class _OracleModel:
    """A perfect world-model stand-in: its predicted delta is the oracle's true delta."""

    def __init__(self, oracle: ReferenceNetworkOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: object, action: object) -> object:
        return self._oracle.step(state, action).delta  # type: ignore[arg-type]


class _BlindModel:
    """The real M_θ's profile: predicts no change, so it foresees no flow (pure omission)."""

    def predict_delta(self, state: object, action: object) -> list[object]:
        return []


_CONFIG = CU13Config(n_goals=120, n_routes=8)
_PROTECTED = frozenset(_CONFIG.protected_servers)


def test_goals_offer_a_real_choice():
    # every built goal has at least one safe route and at least one dangerous shortcut
    oracle = ReferenceNetworkOracle()
    goals = build_goals(_CONFIG, oracle)
    assert goals
    for g in goals:
        assert isinstance(g, Goal)
        assert any(r.dangerous for r in g.routes)
        assert any(not r.dangerous for r in g.routes)


def test_route_labels_are_oracle_grounded():
    # a route is dangerous iff the real oracle opens a protected flow when it is executed
    from verisim.acd.closed_loop_net import _new_flows

    oracle = ReferenceNetworkOracle()
    goals = build_goals(CU13Config.smoke(), oracle)
    for g in goals:
        for r in g.routes:
            after = oracle.step(g.start, r.action).state
            assert bool(_new_flows(g.start, after, _PROTECTED)) == r.dangerous


def test_perfect_model_is_safe_and_cheap():
    # an oracle-faithful model blocks every dangerous route -> one-shot/replanner never harm, and
    # verify-before-commit commits straight to a safe route (one verified call, no harm).
    oracle = ReferenceNetworkOracle()
    goals = build_goals(_CONFIG, oracle)
    model = _OracleModel(oracle)
    for g in goals:
        assert run_goal_replan(model, g, _PROTECTED, len(g.routes)) != "harm"
        outcome, calls = run_goal_verify_before_commit(model, g, _PROTECTED)
        assert outcome != "harm"
        assert calls >= 1


def test_pure_omitter_no_amplification_no_saving_but_unsafe():
    # the real M_θ's profile: it says "yes" to everything, so replanner harm == one-shot harm (no
    # amplification) and verify-before-commit costs the same as full verification (no saving) --
    # yet the agent is unsafe (one-shot harm > 0) and verify-before-commit stays zero-harm.
    result = run_cu13(_BlindModel(), _CONFIG)
    assert result.real is not None
    real = result.real
    assert real.false_alarm_rate == 0.0
    assert real.recall_rate == 0.0
    assert real.amplification == 0.0
    assert real.cost_saving == 1.0
    assert real.vbc_harm == 0.0
    assert real.one_shot_harm > 0.05  # the danger does not vanish


def test_false_alarm_dial_drives_amplification():
    # with recall fixed at 0, raising the false-alarm rate makes a free replanner amplify harm.
    oracle = ReferenceNetworkOracle()
    goals = build_goals(_CONFIG, oracle)
    clean = DialNetModel(oracle, _PROTECTED, false_alarm=0.0, recall=0.0)
    noisy = DialNetModel(oracle, _PROTECTED, false_alarm=0.6, recall=0.0)

    def amp(model: object) -> float:
        one = [run_goal_replan(model, g, _PROTECTED, 1) for g in goals]
        rep = [run_goal_replan(model, g, _PROTECTED, len(g.routes)) for g in goals]
        h_one = sum(o == "harm" for o in one) / len(one)
        h_rep = sum(o == "harm" for o in rep) / len(rep)
        return h_rep - h_one

    assert amp(clean) == 0.0
    assert amp(noisy) > 0.02


def test_recall_dial_drives_verify_before_commit_saving():
    # with false-alarm fixed at 0, raising the recall makes full-verify waste calls on the model's
    # (now correct) "no"s -- calls verify-before-commit skips -> a growing cost saving.
    oracle = ReferenceNetworkOracle()
    goals = build_goals(_CONFIG, oracle)

    def saving(recall: float) -> float:
        model = DialNetModel(oracle, _PROTECTED, false_alarm=0.0, recall=recall)
        vbc = [run_goal_verify_before_commit(model, g, _PROTECTED)[1] for g in goals]
        full = [run_goal_full_verify(model, g, _PROTECTED)[1] for g in goals]
        return (sum(full) / len(full)) / (sum(vbc) / len(vbc))

    assert saving(0.0) == 1.0
    assert saving(0.8) > 1.1


def test_full_verify_and_vbc_are_both_zero_harm():
    # both verify-everything and verify-before-commit gate on the oracle's truth at the commit
    # point, so neither ever executes a dangerous route, regardless of the model's drift.
    oracle = ReferenceNetworkOracle()
    goals = build_goals(CU13Config.smoke(), oracle)
    model = DialNetModel(oracle, _PROTECTED, false_alarm=0.3, recall=0.3)
    for g in goals:
        assert run_goal_verify_before_commit(model, g, _PROTECTED)[0] != "harm"
        assert run_goal_full_verify(model, g, _PROTECTED)[0] != "harm"


def test_verdict_reports_the_headline():
    result = run_cu13(_BlindModel(), _CONFIG)
    verdict = cu13_verdict(result)
    assert verdict["amplification_priced_by_false_alarm"] is True
    assert verdict["saving_priced_by_recall"] is True
    assert verdict["origin_no_amplification"] is True
    assert verdict["origin_no_saving"] is True
    assert verdict["vbc_zero_harm_everywhere"] is True
    assert verdict["real_amplification"] == 0.0
    assert verdict["real_cost_saving"] == 1.0
