"""SPEC-22 CU25 (H118): the composite under real drift -- torch-free tests via stand-in models.

The module takes any ``predict_delta`` model; the real ``M_theta`` is loaded only by the
torch-gated experiment. Here two pure-python stand-ins pin the recall-0 / recall-1 endpoints and
the headline structure:

  - ``_NoopDelta`` (predict_delta = []) is the worst-case omitter -- it foresees no danger, so it
    must reproduce CU24's omitter composite (recall 0 every leg, model-self-targeting breached
    everywhere, the union target still 0.000): the consistency bridge to CU24.
  - ``_OracleDelta`` (predict_delta = the oracle's delta) is the perfect model -- recall 1 every
    leg, model self-targeting self-governs to 0.000: the recall-1 endpoint.
"""

from __future__ import annotations

from verisim.acd.availability_targeting import CU22Config
from verisim.acd.composite_targeting import LEG_NAMES
from verisim.acd.composite_trained import (
    CU25Result,
    cu25_verdict,
    run_cu25,
    trace_deployment,
)
from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netdelta.edits import NetDelta
from verisim.netoracle.reference import ReferenceNetworkOracle


class _NoopDelta:
    """The worst-case omitter: every action is previewed as a no-op (foresees no danger)."""

    def predict_delta(self, state: NetworkState, action: NetAction) -> NetDelta:
        return []


class _OracleDelta:
    """The perfect model: previews each action with the oracle's exact delta (recall 1)."""

    def __init__(self) -> None:
        self._oracle = ReferenceNetworkOracle()

    def predict_delta(self, state: NetworkState, action: NetAction) -> NetDelta:
        return self._oracle.step(state, action).delta


def _smoke() -> CU22Config:
    return CU22Config(horizon=20, n_seeds=160, max_episodes=24, seed0=11000)


def test_run_cu25_returns_result() -> None:
    result = run_cu25(_NoopDelta(), _smoke())
    assert isinstance(result, CU25Result)
    assert result.n_episodes > 0
    assert set(result.self_gov_recall) == set(LEG_NAMES)


def test_omitter_reproduces_cu24_composite() -> None:
    """The no-op delta == the CU24 omitter: it leaks exactly every leg it has an opportunity on.

    The omitter foresees nothing, so under model self-targeting it never consults and a deployment
    is breached on a leg iff that leg has >=1 attack opportunity along the trajectory -- the exact
    CU24 omitter per-leg adversarial number (the consistency bridge). The CU22 battery guarantees an
    outage opportunity, so the outage leg leaks on every deployment.
    """
    config = _smoke()
    oracle = ReferenceNetworkOracle()
    from verisim.acd.availability_targeting import build_deployments
    from verisim.acd.composite_trained import trace_deployment

    deps = build_deployments(config, oracle)
    traces = [trace_deployment(_NoopDelta(), oracle, dep, config) for dep in deps]
    result = run_cu25(_NoopDelta(), config)
    for leg in LEG_NAMES:
        assert result.self_gov_recall[leg] == 0.0  # foresees nothing
        chances = sum(any(st.n_attacks[leg] > 0 for st in tr.steps) for tr in traces)
        opportunity = chances / len(traces)
        assert result.model_adv_breach[leg] == opportunity  # leaks exactly where it has a chance
    assert result.model_adv_breach["outage"] == 1.0  # the battery guarantees an outage opportunity
    assert result.model_composite_adv == 1.0
    # the model-free union target is safe and un-gameable regardless of the (omitter) model.
    assert result.union_target_random_breach == 0.0
    assert result.union_target_adv_breach == 0.0
    assert result.union_covers is True


def test_oracle_model_self_governs() -> None:
    """The oracle delta == the perfect model: recall 1, model self-targeting reaches 0 breach."""
    result = run_cu25(_OracleDelta(), _smoke())
    for leg in LEG_NAMES:
        assert result.self_gov_recall[leg] == 1.0
        assert result.model_adv_breach[leg] == 0.0
    assert result.model_composite_adv == 0.0


def test_union_target_cheaper_than_full_oracle() -> None:
    """The union surface is a small fraction of verifying every step (defense in depth is cheap)."""
    result = run_cu25(_NoopDelta(), _smoke())
    assert 0.0 < result.union_target_calls < result.full_oracle_calls


def test_union_covers_on_the_real_battery() -> None:
    """Coverage (realizes => union target) holds a priori on every attack -- model-free."""
    result = run_cu25(_NoopDelta(), _smoke())
    assert result.union_covers is True


def test_trace_caches_per_step_foresight() -> None:
    """trace_deployment records, per step, each leg's arsenal size + the model's foreseen count."""
    config = _smoke()
    oracle = ReferenceNetworkOracle()
    from verisim.acd.availability_targeting import build_deployments

    deps = build_deployments(config, oracle)
    trace = trace_deployment(_OracleDelta(), oracle, deps[0], config)
    assert len(trace.steps) == config.horizon
    for st in trace.steps:
        for leg in LEG_NAMES:
            assert st.n_foreseen[leg] == st.n_attacks[leg]  # the oracle model foresees all


def test_verdict_headline() -> None:
    """The verdict exposes the boundary-split + high-foresight-still-breached structure."""
    v = cu25_verdict(run_cu25(_NoopDelta(), _smoke()))
    # the omitter has recall 0 everywhere, so model self-targeting fails every leg.
    assert v["model_self_targeting_fails_every_leg"] is True
    assert v["union_target_safe_on_every_leg"] is True
    assert v["union_covers"] is True
    assert float(v["composite_call_saving"]) > 1.0  # type: ignore[arg-type]


def test_high_foresight_leg_still_breached_is_model_specific() -> None:
    """The boundary-split flags are False for the degenerate stand-ins (they need the real M_theta).

    The omitter sees nothing (recall 0 on outage) and the oracle sees everything (no breach), so
    neither triggers ``high_foresight_leg_still_breached`` -- that headline is a property of the
    *real* partially-faithful model, asserted in the torch-gated experiment, not here.
    """
    omitter_v = cu25_verdict(run_cu25(_NoopDelta(), _smoke()))
    oracle_v = cu25_verdict(run_cu25(_OracleDelta(), _smoke()))
    assert omitter_v["high_foresight_leg_still_breached"] is False
    assert oracle_v["high_foresight_leg_still_breached"] is False
    assert omitter_v["foresight_heterogeneous"] is False
