"""SPEC-22 CU10 / H103 -- targeted verification (torch-free core).

Validated with cheap stand-in models (no training): a *blind* no-op model (the worst omitter) lets
the structure schedule reach zero breach at a small call count while the model self-targeting
schedule fails (it never sees activity, so it never consults), and a *perfect* oracle model is safe
under every schedule. The real trained transformer arm is the experiment's committed run
(torch-gated, not exercised in CI).
"""

from __future__ import annotations

from verisim.acd.targeted_verification import (
    CU10Config,
    build_deployments,
    cu10_verdict,
    run_cu10,
    run_deployment,
)
from verisim.netoracle.reference import ReferenceNetworkOracle


class _OracleModel:
    """A perfect world-model stand-in: its predicted delta is the oracle's true delta."""

    def __init__(self, oracle: ReferenceNetworkOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: object, action: object) -> object:
        return self._oracle.step(state, action).delta  # type: ignore[arg-type]


class _BlindModel:
    """A maximally-drifting stand-in: predicts no change, so it foresees no flow (pure omission)."""

    def predict_delta(self, state: object, action: object) -> list[object]:
        return []


_CONFIG = CU10Config(horizon=48, n_seeds=400, max_episodes=120, rhos=(0.0, 0.5, 1.0))


def test_deployments_present_an_exfil_opportunity():
    oracle = ReferenceNetworkOracle()
    deployments = build_deployments(CU10Config.smoke(), oracle)
    assert deployments
    for d in deployments:
        assert d.n_opportunities > 0


def test_structure_targeting_reaches_zero_breach_cheaply():
    # The blind model omits every exfil; structure targeting verifies exactly the connect-to-
    # protected actions, so the oracle's verdict gates every real breach -> zero breach, few calls.
    result = run_cu10(_BlindModel(), _CONFIG)
    assert result.structure.breach_rate == 0.0
    assert result.structure.mean_calls > 0.0
    # ... and it is cheaper than reaching zero breach the blind way (the full oracle every step)
    full = result.uniform[-1]
    assert full.breach_rate == 0.0
    assert result.structure.mean_calls < full.mean_calls


def test_model_self_targeting_fails_to_see_its_own_omissions():
    # The blind model never predicts a flow, so the self-targeting schedule never consults and
    # breaches like the free agent -- the CU8 lesson: you can't ask the omitter where it omits.
    result = run_cu10(_BlindModel(), _CONFIG)
    free = result.uniform[0]
    assert free.breach_rate >= 0.5
    assert result.model.mean_calls < 1.0  # it spends almost no budget
    assert result.model.breach_rate >= 0.5 * free.breach_rate  # and stays unsafe


def test_perfect_model_is_safe_under_every_schedule():
    oracle = ReferenceNetworkOracle()
    result = run_cu10(_OracleModel(oracle), CU10Config.smoke())
    for c in (*result.uniform, result.model, result.structure):
        assert c.breach_rate == 0.0  # a faithful preview foresees every exfil; the gate blocks it


def test_run_deployment_returns_breach_and_calls():
    oracle = ReferenceNetworkOracle()
    deployments = build_deployments(CU10Config.smoke(), oracle)
    breached, calls = run_deployment(
        _BlindModel(), oracle, deployments[0], CU10Config.smoke(), "structure"
    )
    assert isinstance(breached, bool)
    assert isinstance(calls, int) and calls >= 0


def test_verdict_reports_the_headline():
    result = run_cu10(_BlindModel(), _CONFIG)
    verdict = cu10_verdict(result)
    assert verdict["structure_is_safe"] is True
    assert verdict["structure_cheaper_than_full"] is True
    assert verdict["model_self_targeting_fails"] is True
