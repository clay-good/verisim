"""SPEC-22 CU9 / H102 -- the agent-safety horizon (torch-free core).

Validated with stand-ins: a perfect (oracle) model never breaches, so survival is flat at 1.0 at
every budget; a blind no-op model breaches over the deployment (survival decays) and verification
extends the safe runtime. The real trained M_θ is the experiment's committed run.
"""

from __future__ import annotations

from verisim.acd.safety_horizon import (
    CU9Config,
    build_deployments,
    cu9_verdict,
    first_breach_step,
    run_cu9,
)
from verisim.netoracle.reference import ReferenceNetworkOracle


class _OracleModel:
    def __init__(self, oracle: ReferenceNetworkOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: object, action: object) -> object:
        return self._oracle.step(state, action).delta  # type: ignore[arg-type]


class _BlindModel:
    def predict_delta(self, state: object, action: object) -> list[object]:
        return []


def test_deployments_present_an_exfil_opportunity():
    oracle = ReferenceNetworkOracle()
    config = CU9Config.smoke()
    deployments = build_deployments(config, oracle)
    assert deployments
    for d in deployments:
        assert d.n_opportunities > 0


def test_perfect_model_never_breaches():
    oracle = ReferenceNetworkOracle()
    config = CU9Config.smoke()
    model = _OracleModel(oracle)
    for d in build_deployments(config, oracle):
        assert first_breach_step(model, oracle, d, config, 0.0) is None
    result = run_cu9(model, config)
    for curve in result.curves:
        assert curve.breach_rate == 0.0
        assert all(s == 1.0 for s in curve.survival)


def test_blind_agent_breaches_and_verification_extends_the_horizon():
    config = CU9Config(horizon=48, n_seeds=400, max_episodes=200, rhos=(0.0, 0.3, 1.0))
    result = run_cu9(_BlindModel(), config)
    free = result.curves[0]
    oracle = result.curves[-1]
    assert free.breach_rate >= 0.5  # the unverified agent is unsafe over a deployment
    assert free.survival[-1] < free.survival[0]  # its survival decays with deployment length
    assert oracle.breach_rate == 0.0  # the oracle agent stays safe
    verdict = cu9_verdict(result)
    assert verdict["free_unsafe_over_deployment"] is True
    assert verdict["oracle_stays_safe"] is True
    assert verdict["verification_extends_horizon"] is True
