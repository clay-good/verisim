"""UA1/UA2 transfer tests (SPEC-20 §5, H73/H74).

The contract is the three-environment orchestration and the verdict logic:

  - the cost axis is right (oracle = 1 call/step, grounded = ρ/step, free = 0);
  - `run_ua_transfer` trains in each backend and evaluates EVERY arm in `E_oracle` (reality), with a
    perfect oracle-backed stand-in model so the orchestration runs torch-free in CI;
  - `ua_verdict` computes the H73 (cheap transfer) and H74 (grounding load-bearing) fields.

With a *perfect* model the three backends are identical, so the smoke run is an orchestration
not a science result -- the H74 signal comes from the local run with a real (drifting) trained
model, where `E_free`'s drift is expected to hurt transfer.
"""

from __future__ import annotations

import pytest

from verisim.experiments.ua_transfer import (
    ArmResult,
    UATransferConfig,
    _train_oracle_calls,
    run_ua_transfer,
    ua_verdict,
)
from verisim.netloop.model import NetOracleBackedModel
from verisim.netoracle import ReferenceNetworkOracle


def test_train_oracle_cost_axis():
    cfg = UATransferConfig.smoke()
    steps = cfg.train.episodes * cfg.containment.episode_steps
    assert _train_oracle_calls("oracle", cfg) == steps
    assert _train_oracle_calls("grounded", cfg) == round(steps * cfg.rho)
    assert _train_oracle_calls("free", cfg) == 0


def test_run_ua_transfer_orchestration_torch_free():
    model = NetOracleBackedModel(ReferenceNetworkOracle())  # a perfect (non-drifting) stand-in
    results = run_ua_transfer(model, UATransferConfig.smoke())
    assert set(results) == {"oracle", "grounded", "free"}
    for r in results.values():
        assert 0.0 <= r.reality_containment <= 1.0
    # all three evaluated in reality; with a perfect model they should match closely
    assert results["grounded"].reality_containment == results["oracle"].reality_containment
    assert results["free"].reality_containment == results["oracle"].reality_containment


def test_ua_verdict_fields_and_logic():
    res = {
        "oracle": ArmResult("oracle", 0.80, 1000),
        "grounded": ArmResult("grounded", 0.78, 200),
        "free": ArmResult("free", 0.60, 0),
    }
    v = ua_verdict(res)
    assert v["cost_ratio_oracle_over_grounded"] == 5.0
    assert v["h73_supported"]  # 0.78 >= 0.9*0.80? no -> check: 0.72 threshold, 0.78>=0.72 and 5x
    assert v["h74_supported"] and v["grounding_advantage"] == pytest.approx(0.18)
    # grounding null -> H74 refuted (the bankable negative)
    res2 = dict(res, free=ArmResult("free", 0.78, 0))
    assert not ua_verdict(res2)["h74_supported"]


def test_transfer_gap_zero_for_perfect_model():
    # a perfect model makes E_grounded == E_oracle, so the policy transfer gap is exactly 0 (UA3)
    from verisim.experiments.ua_transfer import transfer_gap

    model = NetOracleBackedModel(ReferenceNetworkOracle())
    g = transfer_gap(model, UATransferConfig.smoke())
    assert g["transfer_gap"] == pytest.approx(0.0)
    assert g["abs_transfer_gap"] == pytest.approx(0.0)


def test_budget_sweep_reports_advantage_per_rho():
    from verisim.experiments.ua_transfer import budget_sweep

    model = NetOracleBackedModel(ReferenceNetworkOracle())  # perfect -> advantage 0 at every ρ
    rows = budget_sweep(model, (0.25, 0.5), UATransferConfig.smoke())
    assert [r["rho"] for r in rows] == [0.25, 0.5]
    for r in rows:
        assert r["grounding_advantage"] == pytest.approx(0.0)  # perfect model -> no drift, no gap


def test_run_ua_transfer_is_deterministic():
    model = NetOracleBackedModel(ReferenceNetworkOracle())
    cfg = UATransferConfig.smoke()
    a = run_ua_transfer(model, cfg)
    b = run_ua_transfer(model, cfg)
    assert {k: v.reality_containment for k, v in a.items()} == {
        k: v.reality_containment for k, v in b.items()
    }
