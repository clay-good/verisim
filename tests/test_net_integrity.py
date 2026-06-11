"""UA10 network flow-integrity tests (SPEC-20 §7, H82).

The contract: the predictive flow-defense reward is exact for a faithful predictor, the ρ-grounded
predictor's extremes recover the faithful/free predictors, grounding is monotone under drift, and
the H82 verdict logic is correct. The torch-free parts use the oracle as predictor + truth and small
fake `M_θ`s; the real faithful-vs-free gap comes from the local run on the network flagship.
"""

from __future__ import annotations

import pytest

from verisim.acd.net_integrity import (
    established_flows,
    grounded_flow_defense_reward,
    grounded_flow_rollout,
    make_net_workload,
    oracle_step,
    predictive_flow_defense_reward,
    rollout_flows,
)
from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netoracle.reference import ReferenceNetworkOracle


class _FaithfulNetModel:
    """A torch-free 'perfect' M_θ: its predicted delta is the oracle's, so it never drifts."""

    def __init__(self, oracle: ReferenceNetworkOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: NetworkState, action: NetAction) -> object:
        return self._oracle.step(state, action).delta


class _BlindNetModel:
    """A torch-free 'maximally drifted' M_θ: predicts the identity delta (never opens a flow)."""

    def predict_delta(self, state: NetworkState, action: NetAction) -> list[object]:
        return []


def test_make_workload_establishes_flows():
    oracle = ReferenceNetworkOracle()
    start, actions = make_net_workload(seed=801, n_steps=20, oracle=oracle)
    assert len(actions) == 20
    # the connected seed has links + services, so the cumulative flow set is non-trivial
    true_flows = rollout_flows(oracle_step(oracle), start, actions)
    assert isinstance(true_flows, set)
    # the cumulative set is the union over the rollout (a flow that drops still counts)
    state, seen = start, set(established_flows(start))
    step = oracle_step(oracle)
    for a in actions:
        state = step(state, a)
        seen |= established_flows(state)
    assert true_flows == seen


def test_faithful_predictor_scores_perfectly():
    oracle = ReferenceNetworkOracle()
    step = oracle_step(oracle)
    for seed in (800, 801, 802, 803):
        start, actions = make_net_workload(seed=seed, n_steps=20, oracle=oracle)
        r = predictive_flow_defense_reward(step, step, start, actions, budget=2)
        assert r == pytest.approx(1.0)


def test_grounded_extremes_match_faithful_and_free():
    oracle = ReferenceNetworkOracle()
    start, actions = make_net_workload(seed=801, n_steps=20, oracle=oracle)
    blind = _BlindNetModel()
    # ρ=1: re-anchor every step -> predicted cumulative ≡ true cumulative, full oracle cost.
    pred, true, calls = grounded_flow_rollout(blind, oracle, start, actions, rho=1.0)
    assert pred == true and calls == len(actions)
    # ρ=0: never re-anchor -> the blind model opens no flow, 0 calls.
    pred0, true0, calls0 = grounded_flow_rollout(blind, oracle, start, actions, rho=0.0)
    assert calls0 == 0 and pred0 == set() and true0 == true


def test_grounded_faithful_model_perfect_at_every_rho():
    oracle = ReferenceNetworkOracle()
    start, actions = make_net_workload(seed=802, n_steps=20, oracle=oracle)
    perfect = _FaithfulNetModel(oracle)
    for rho in (0.0, 0.2, 0.5, 1.0):
        r, _ = grounded_flow_defense_reward(perfect, oracle, start, actions, budget=2, rho=rho)
        assert r == pytest.approx(1.0)


def test_grounded_catch_monotone_in_rho_with_drift():
    oracle = ReferenceNetworkOracle()
    blind = _BlindNetModel()
    rewards = []
    for rho in (0.0, 0.5, 1.0):
        rs = [
            grounded_flow_defense_reward(blind, oracle, *make_net_workload(s, 20, oracle=oracle),
                                         budget=2, rho=rho)[0]
            for s in (800, 801, 802, 803)
        ]
        rewards.append(sum(rs) / len(rs))
    assert all(rewards[i + 1] >= rewards[i] for i in range(len(rewards) - 1))
    assert rewards[-1] == pytest.approx(1.0) and rewards[0] < 1.0  # ceiling at ρ=1, gap at ρ=0


def test_h82_verdict_logic():
    from verisim.experiments.ua_net_integrity import IntegrityPoint, KneePoint, h82_verdict

    def hp(pred, h, r):
        return IntegrityPoint(pred, h, r, r, r, 24)

    def kp(rho, r, calls):
        return KneePoint(rho, r, r, r, calls, 24)

    # content positive (free degrades, gap widens) + monotone sub-linear knee -> supported
    hpts = [hp("faithful", 8, 1.0), hp("free", 8, 0.58),
            hp("faithful", 28, 1.0), hp("free", 28, 0.08)]
    kpts = [kp(0.0, 0.08, 0.0), kp(0.2, 1.0, 4.0), kp(1.0, 1.0, 20.0)]
    v = h82_verdict(hpts, kpts, knee_frac=0.9)
    assert v["h82_supported"] and v["content_positive"] and v["useful_knee"]
    assert v["knee_rho"] == 0.2 and v["gap_at_max_horizon"] == pytest.approx(0.92)
    # a faithful free predictor (no content gap) -> not the positive -> not supported
    flat = [hp("faithful", 8, 1.0), hp("free", 8, 1.0),
            hp("faithful", 28, 1.0), hp("free", 28, 1.0)]
    assert not h82_verdict(flat, kpts)["h82_supported"]


# --- torch-gated: the real faithful-vs-free + knee sweep -----------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.flagship import FlagshipConfig, train_flagship  # noqa: E402
from verisim.experiments.ua_net_integrity import (  # noqa: E402
    NetIntegrityConfig,
    h82_verdict,
    run_horizon_sweep,
    run_knee_sweep,
)


def test_net_integrity_sweeps_well_formed():
    model, _ = train_flagship(FlagshipConfig.smoke())
    config = NetIntegrityConfig.smoke()
    oracle = ReferenceNetworkOracle()
    hpts = run_horizon_sweep(model, config, oracle=oracle)
    kpts = run_knee_sweep(model, config, oracle=oracle)
    assert {p.predictor for p in hpts} == {"faithful", "free"}
    for p in hpts:
        assert 0.0 <= p.reward <= 1.0
        if p.predictor == "faithful":
            assert p.reward == pytest.approx(1.0)  # the oracle predicting itself catches every flow
    assert [p.rho for p in kpts] == [0.0, 0.5, 1.0]
    assert kpts[-1].reward == pytest.approx(1.0) and kpts[-1].oracle_calls == pytest.approx(12.0)
    assert kpts[0].oracle_calls == pytest.approx(0.0)
    v = h82_verdict(hpts, kpts, config.knee_frac)
    assert set(v) >= {"content_positive", "useful_knee", "h82_supported", "knee_rho"}
