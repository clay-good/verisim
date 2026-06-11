"""UA8 file-integrity tests (SPEC-20 §7, H80).

The contract: the predictive-defense reward is exact for a faithful predictor and the H80 verdict
logic is correct. The torch-free parts use the oracle as both predictor and truth (a faithful
predictor scores 1.0); the real faithful-vs-free gap comes from the local run on the host flagship.
"""

from __future__ import annotations

import pytest

from verisim.acd.host_integrity import (
    make_workload,
    oracle_step,
    predictive_defense_reward,
    rollout_writes,
    written_files,
)
from verisim.hostoracle.reference import ReferenceHostOracle


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
