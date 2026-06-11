"""UA7 predictive-defender tests (SPEC-20 §7, H79).

Torch-free (the oracle/perfect-model planners need no torch). The contract:

  - `deterministic_spread` is deterministic and only takes exposed beachheads;
  - `plan_isolation` returns a legal isolate-or-noop and respects the candidate set;
  - a predictive episode against true dynamics returns a containment in [0, 1];
  - with a *perfect* (oracle-backed) model the free planner equals the faithful planner (no drift ->
    no gap, the control); `h79_verdict` only fires when the gap is positive and widens with k.

The committed verdict comes from the local run with the real drifting flagship model.
"""

from __future__ import annotations

import random

import pytest

from verisim.acd.containment import ContainmentConfig, seed_topology
from verisim.acd.predictive import (
    deterministic_spread,
    model_step_fn,
    oracle_step_fn,
    plan_isolation,
    run_predictive_episode,
    run_reactive_episode,
)
from verisim.net.state import NetworkState, link_key
from verisim.netoracle import ReferenceNetworkOracle


def test_deterministic_spread_takes_only_beachheads_and_is_deterministic():
    net = NetworkState.initial(("h0", "h1", "h2"))
    net.links = {link_key("h0", "h1")}  # h2 isolated
    net.hosts["h1"] = net.hosts["h1"].with_service(22, True)
    comp = frozenset({"h0"})
    a = deterministic_spread(net, comp)
    b = deterministic_spread(net, comp)
    assert a == b  # deterministic
    assert a == frozenset({"h0", "h1"})  # h1 (connected + service) falls; h2 (no link) safe


def test_plan_isolation_returns_legal_isolate_or_noop():
    oracle = ReferenceNetworkOracle()
    cfg = ContainmentConfig(cut_budget=2)
    net, comp = seed_topology(cfg, random.Random(1))
    action = plan_isolation(oracle_step_fn(oracle), net, comp, cfg, k=3)
    assert action.kind in ("isolate", "noop")
    if action.kind == "isolate":
        assert net.hosts[action.host].up


def test_predictive_episode_returns_valid_containment():
    oracle = ReferenceNetworkOracle()
    cfg = ContainmentConfig(episode_steps=8, cut_budget=2)
    step = oracle_step_fn(oracle)
    c = run_predictive_episode(step, step, cfg, seed=5, k=3)
    assert 0.0 <= c <= 1.0


def test_reactive_episode_returns_valid_containment():
    oracle = ReferenceNetworkOracle()
    cfg = ContainmentConfig(episode_steps=8, cut_budget=2)
    c = run_reactive_episode(oracle_step_fn(oracle), cfg, seed=5)
    assert 0.0 <= c <= 1.0


def test_perfect_model_planner_equals_faithful_planner():
    # a perfect (oracle-backed) model -> the free planner's lookahead == the faithful planner's.
    from verisim.netloop.model import NetOracleBackedModel

    oracle = ReferenceNetworkOracle()
    cfg = ContainmentConfig(episode_steps=10, cut_budget=2)
    true_step = oracle_step_fn(oracle)
    faithful = oracle_step_fn(oracle)
    perfect_free = model_step_fn(NetOracleBackedModel(oracle))
    for seed in (600, 601, 602):
        cf = run_predictive_episode(true_step, faithful, cfg, seed, k=3)
        ce = run_predictive_episode(true_step, perfect_free, cfg, seed, k=3)
        assert cf == ce  # no drift -> identical planning -> identical containment


def test_h79_verdict_logic():
    from verisim.experiments.ua_predictive import PredictivePoint, h79_verdict

    def pt(planner, k, c):
        return PredictivePoint(planner, k, c, c, c, 4)

    # faithful flat, free degrades with k -> gap widens -> supported
    pts = [
        pt("faithful", 1, 0.6), pt("free", 1, 0.59), pt("reactive", 1, 0.5),
        pt("faithful", 8, 0.6), pt("free", 8, 0.40), pt("reactive", 8, 0.5),
    ]
    v = h79_verdict(pts)
    assert v["h79_supported"] and v["closed_gap_at_max_k"] == pytest.approx(0.20)
    # no widening -> not supported
    flat = [
        pt("faithful", 1, 0.6), pt("free", 1, 0.6), pt("reactive", 1, 0.5),
        pt("faithful", 8, 0.6), pt("free", 8, 0.6), pt("reactive", 8, 0.5),
    ]
    assert not h79_verdict(flat)["h79_supported"]
