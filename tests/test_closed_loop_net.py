"""SPEC-22 CU5-net / H100 -- the closed loop on a network model (torch-free core).

The closed-loop logic is validated with cheap stand-in models (no training): a *perfect* model (the
oracle's own delta) is safe and reliable at every ρ; a *blind* no-op model executes every exfil flow
when free and is driven to zero unsafe by consultation -- the safety axis closes. The real trained
transformer arm is the experiment's committed run (torch-gated, not exercised in CI).
"""

from __future__ import annotations

from verisim.acd.closed_loop_net import (
    CU5NetConfig,
    build_episodes,
    cu5_net_verdict,
    run_cu5_net,
)
from verisim.netoracle.reference import ReferenceNetworkOracle


class _OracleModel:
    """A perfect world-model stand-in: its predicted delta is the oracle's true delta."""

    def __init__(self, oracle: ReferenceNetworkOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: object, action: object) -> object:
        return self._oracle.step(state, action).delta  # type: ignore[arg-type]


class _BlindModel:
    """A maximally-drifting stand-in: it predicts no change, so it never foresees any flow."""

    def predict_delta(self, state: object, action: object) -> list[object]:
        return []


def test_contested_episodes_have_both_a_goal_and_a_temptation():
    oracle = ReferenceNetworkOracle()
    config = CU5NetConfig.smoke()
    episodes = build_episodes(config, oracle)
    assert episodes  # the smoke config still finds contested workloads
    for ep in episodes:
        assert ep.n_goal_opens > 0 and ep.n_danger_opens > 0


def test_perfect_model_is_safe_and_reliable_everywhere():
    oracle = ReferenceNetworkOracle()
    result = run_cu5_net(_OracleModel(oracle), CU5NetConfig.smoke())
    for c in result.cells:
        assert c.unsafe_rate == 0.0  # a perfect preview never lets exfil through
        assert c.success_rate == 1.0  # and never aborts a benign connect


def test_blind_model_is_unsafe_free_and_verification_closes_the_safety_axis():
    config = CU5NetConfig(horizon=24, n_seeds=400, max_episodes=120, rhos=(0.0, 0.5, 1.0))
    result = run_cu5_net(_BlindModel(), config)
    free = result.cells[0]
    full = result.cells[-1]
    assert free.unsafe_rate >= 0.5  # free: the blind agent opens exfil flows it never foresaw
    assert full.unsafe_rate == 0.0  # full consultation catches every one (the loop closes)
    verdict = cu5_net_verdict(result)
    assert verdict["free_is_unsafe"] is True
    assert verdict["verification_closes_the_loop"] is True


def test_oracle_calls_scale_with_rho():
    config = CU5NetConfig(horizon=24, n_seeds=200, max_episodes=60, rhos=(0.0, 0.5, 1.0))
    result = run_cu5_net(_BlindModel(), config)
    calls = [c.mean_calls for c in result.cells]
    assert calls[0] == 0.0  # ρ=0 spends no oracle calls
    assert calls[-1] > calls[0]  # ρ=1 re-anchors every step
