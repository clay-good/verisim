"""UA8 file-integrity tests (SPEC-20 §7, H80).

The contract: the predictive-defense reward is exact for a faithful predictor and the H80 verdict
logic is correct. The torch-free parts use the oracle as both predictor and truth (a faithful
predictor scores 1.0); the real faithful-vs-free gap comes from the local run on the host flagship.
"""

from __future__ import annotations

import pytest

from verisim.acd.host_integrity import (
    grounded_defense_reward,
    grounded_rollout_writes,
    make_workload,
    oracle_step,
    predictive_defense_reward,
    rollout_writes,
    written_files,
)
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle


class _FaithfulModel:
    """A torch-free 'perfect' M_θ: its predicted delta is the oracle's, so it never drifts."""

    def __init__(self, oracle: ReferenceHostOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: HostState, action: HostAction) -> object:
        return self._oracle.step(state, action).delta


class _BlindModel:
    """A torch-free 'maximally drifted' M_θ: predicts the identity delta (no writes at all)."""

    def predict_delta(self, state: HostState, action: HostAction) -> list[object]:
        return []


def test_written_files_and_workload():
    oracle = ReferenceHostOracle()
    start, actions = make_workload(seed=701, n_steps=14, oracle=oracle)
    assert len(actions) == 14
    true_corrupted = rollout_writes(oracle_step(oracle), start, actions)
    assert isinstance(true_corrupted, set)
    # the written set is exactly the files with content in the final true state
    final = start
    step = oracle_step(oracle)
    for a in actions:
        final = step(final, a)
    assert true_corrupted == written_files(final)


def test_faithful_predictor_scores_perfectly():
    oracle = ReferenceHostOracle()
    step = oracle_step(oracle)
    for seed in (701, 702, 703):
        start, actions = make_workload(seed=seed, n_steps=14, oracle=oracle)
        # the oracle predicting itself catches every true corruption -> reward 1.0 (or 1.0 if none)
        r = predictive_defense_reward(step, step, start, actions, budget=2)
        assert r == pytest.approx(1.0)


def test_h80_verdict_logic():
    from verisim.experiments.ua_host_integrity import IntegrityPoint, h80_verdict

    def pt(pred, h, r):
        return IntegrityPoint(pred, h, r, r, r, 4)

    # faithful flat at 1.0; free degrades with horizon -> gap widens -> supported
    pts = [
        pt("faithful", 6, 1.0), pt("free", 6, 0.95),
        pt("faithful", 18, 1.0), pt("free", 18, 0.60),
    ]
    v = h80_verdict(pts)
    assert v["h80_supported"] and v["gap_at_max_horizon"] == pytest.approx(0.40)
    # no widening -> not supported
    flat = [pt("faithful", 6, 1.0), pt("free", 6, 1.0),
            pt("faithful", 18, 1.0), pt("free", 18, 1.0)]
    assert not h80_verdict(flat)["h80_supported"]


# --- UA9 / H81: the grounded predictor (torch-free contract) -------------------------------------


def test_grounded_extremes_match_faithful_and_free():
    """ρ=1 ≡ faithful (predicted=true, |actions| calls); ρ=0 ≡ free (pure model, 0 calls)."""
    oracle = ReferenceHostOracle()
    start, actions = make_workload(seed=701, n_steps=14, oracle=oracle)
    blind = _BlindModel()
    # ρ=1: re-anchor every step -> predicted ≡ true, full oracle cost, perfect catch.
    pred, true, calls = grounded_rollout_writes(blind, oracle, start, actions, rho=1.0)
    assert pred == true and calls == len(actions)
    # ρ=0: never re-anchor -> pure (blind) model, no oracle calls; the blind model writes nothing.
    pred0, true0, calls0 = grounded_rollout_writes(blind, oracle, start, actions, rho=0.0)
    assert calls0 == 0 and pred0 == set() and true0 == true


def test_grounded_faithful_model_perfect_at_every_rho():
    """A perfect model needs no grounding: catch 1.0 at every ρ, with calls scaling in ρ."""
    oracle = ReferenceHostOracle()
    start, actions = make_workload(seed=702, n_steps=14, oracle=oracle)
    perfect = _FaithfulModel(oracle)
    for rho in (0.0, 0.2, 0.5, 1.0):
        r, _ = grounded_defense_reward(perfect, oracle, start, actions, budget=2, rho=rho)
        assert r == pytest.approx(1.0)
    # the consultation cost is monotone in ρ (every-other-step at 0.5, every step at 1.0)
    _, c_half = grounded_defense_reward(perfect, oracle, start, actions, budget=2, rho=0.5)
    _, c_full = grounded_defense_reward(perfect, oracle, start, actions, budget=2, rho=1.0)
    assert 0 < c_half < c_full == len(actions)


def test_grounded_catch_monotone_in_rho_with_drift():
    """With the blind (drifted) model, more grounding never lowers the catch rate."""
    oracle = ReferenceHostOracle()
    blind = _BlindModel()
    rewards = []
    for rho in (0.0, 0.25, 0.5, 1.0):
        seeds_r = [
            grounded_defense_reward(blind, oracle, *make_workload(s, 14, oracle=oracle),
                                    budget=2, rho=rho)[0]
            for s in (701, 702, 703, 704)
        ]
        rewards.append(sum(seeds_r) / len(seeds_r))
    assert all(rewards[i + 1] >= rewards[i] for i in range(len(rewards) - 1))
    assert rewards[-1] == pytest.approx(1.0) and rewards[0] < 1.0  # ceiling at ρ=1, gap at ρ=0


def test_h81_verdict_logic():
    from verisim.experiments.ua_host_grounded import KneePoint, h81_verdict

    def pt(rho, r, calls):
        return KneePoint(rho, r, r, r, calls, 24)

    # monotone recovery from a free floor to a faithful ceiling, knee reached sub-linearly
    pts = [pt(0.0, 0.50, 0.0), pt(0.3, 0.85, 4.0), pt(0.5, 1.00, 7.0), pt(1.0, 1.00, 14.0)]
    v = h81_verdict(pts, knee_frac=0.9)
    assert v["h81_supported"] and v["knee_rho"] == 0.5 and v["sublinear_knee"]
    assert v["recoverable_gap"] == pytest.approx(0.5)
    # a flat curve (no recoverable gap, nothing for ρ to buy) -> not supported (the H81 null)
    flat = [pt(0.0, 1.0, 0.0), pt(0.5, 1.0, 7.0), pt(1.0, 1.0, 14.0)]
    assert not h81_verdict(flat)["h81_supported"]


# --- torch-gated: the real faithful-vs-free sweep ------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.host_flagship import HostFlagshipConfig, train_host_flagship  # noqa: E402
from verisim.experiments.ua_host_integrity import (  # noqa: E402
    HostIntegrityConfig,
    run_host_integrity,
)


def test_run_host_integrity_well_formed():
    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    points = run_host_integrity(model, HostIntegrityConfig.smoke())
    assert {p.predictor for p in points} == {"faithful", "free"}
    for p in points:
        assert 0.0 <= p.reward <= 1.0
    # the faithful predictor (oracle) catches every true corruption -> reward 1.0 at every horizon
    for p in points:
        if p.predictor == "faithful":
            assert p.reward == pytest.approx(1.0)


def test_grounded_knee_sweep_well_formed():
    from verisim.experiments.ua_host_grounded import (
        GroundedKneeConfig,
        h81_verdict,
        run_grounded_knee,
    )

    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    points = run_grounded_knee(model, GroundedKneeConfig.smoke())
    assert [p.rho for p in points] == [0.0, 0.5, 1.0]
    for p in points:
        assert 0.0 <= p.reward <= 1.0 and p.oracle_calls >= 0.0
    # ρ=1 is the every-step faithful predictor: perfect catch, full oracle cost.
    assert points[-1].reward == pytest.approx(1.0) and points[-1].oracle_calls == pytest.approx(8.0)
    assert points[0].oracle_calls == pytest.approx(0.0)  # ρ=0 is the free predictor
    v = h81_verdict(points, GroundedKneeConfig.smoke().knee_frac)
    assert set(v) >= {"knee_rho", "monotone_in_rho", "recoverable_gap", "h81_supported"}
