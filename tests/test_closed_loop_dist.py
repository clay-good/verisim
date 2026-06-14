"""Tests for SPEC-22 CU19 (H112): the trained distributed arm -- the belief-rollout closed loop.

Torch-free: the belief rollout takes any ``predict_delta`` model, so the trained M_θ (torch-gated in
the experiment) is replaced here by two stand-in delta models that are the recall endpoints of the
rollout and bridge CU19 to CU18:

  - ``_NoopDelta`` returns an empty delta -> the belief is frozen at the boot cluster (where nothing
    is stale), so it foresees no staleness == CU18's ``StaleOmitter`` (recall 0);
  - ``_OracleDelta`` returns the reference oracle's delta -> the belief tracks truth exactly, so it
    foresees staleness exactly == CU18's ``OracleStaleModel`` (recall 1).

Deterministic; uses the smoke battery and the hand-crafted put->partition->stale-read deployment.
"""

from __future__ import annotations

from verisim.acd.closed_loop_dist import (
    adversarial_breach,
    breach_calls,
    cu19_verdict,
    run_cu19,
    staleness_drift,
    trace_deployment,
)
from verisim.acd.dist_targeting import (
    CU18Config,
    DistDeployment,
    StaleOmitter,
    build_deployments,
    is_stale,
    run_cu18,
)
from verisim.dist.action import DistAction, parse_dist_action
from verisim.dist.delta import DistDelta
from verisim.dist.state import DistributedState
from verisim.distoracle.reference import ReferenceDistOracle

CONFIG = CU18Config.smoke()


class _NoopDelta:
    """Belief frozen at boot -> never foresees staleness == CU18's StaleOmitter (recall 0)."""

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta:
        return DistDelta([])


class _OracleDelta:
    """Belief == truth -> foresees staleness exactly == CU18's OracleStaleModel (recall 1)."""

    def __init__(self, config: CU18Config) -> None:
        self._oracle = ReferenceDistOracle(config.dist)

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta:
        return self._oracle.step(state, action).delta


def _oracle() -> ReferenceDistOracle:
    return ReferenceDistOracle(CONFIG.dist)


def _stale_read_deployment() -> DistDeployment:
    """write the sensitive key on n0, partition it off, read it stale on n1 (one known breach)."""
    actions = tuple(
        parse_dist_action(a) for a in ("put n0 cfg a", "partition n0 | n1 n2", "get n1 cfg")
    )
    return DistDeployment(DistributedState.initial(CONFIG.dist), actions, n_opp=1)


def test_deployments_present_opportunity() -> None:
    """Every kept deployment offers at least one real stale-read opportunity."""
    deployments = build_deployments(CONFIG, _oracle())
    assert deployments
    assert all(d.n_opp > 0 for d in deployments)


def test_noop_belief_reproduces_omitter() -> None:
    """A no-op delta model's belief-rollout reproduces CU18's StaleOmitter cell-for-cell."""
    cu19 = run_cu19(_NoopDelta(), CONFIG)
    cu18 = run_cu18(StaleOmitter(recall=0.0), CONFIG)
    assert cu19.n_episodes == cu18.n_episodes
    assert cu19.uniform[0].random_breach == cu18.uniform[0].random_breach  # free
    assert cu19.uniform[-1].mean_calls == cu18.uniform[-1].mean_calls  # full oracle
    assert cu19.medium.random_breach == 0.0
    assert cu19.medium.random_breach == cu18.medium.random_breach
    assert cu19.medium.mean_calls == cu18.medium.mean_calls
    assert cu19.write_target.random_breach == cu18.write_target.random_breach
    assert cu19.model.random_breach == cu18.model.random_breach


def test_oracle_belief_reproduces_oracle_stale() -> None:
    """An oracle delta model's belief-rollout foresees staleness exactly -> safe everywhere."""
    cu19 = run_cu19(_OracleDelta(CONFIG), CONFIG)
    assert cu19.drift.recall == 1.0
    assert cu19.drift.omissions == 0
    for cell in (cu19.uniform[0], cu19.model, cu19.write_target, cu19.medium):
        assert cell.random_breach == 0.0
        assert cell.adversarial_breach == 0.0


def test_drift_is_omission_biased() -> None:
    """The no-op belief omits the medium: every disagreement is an omission, not a hallucination."""
    deployments = build_deployments(CONFIG, _oracle())
    traces = [trace_deployment(_NoopDelta(), _oracle(), d) for d in deployments]
    drift = staleness_drift(traces, CONFIG)
    assert drift.true_stale > 0
    assert drift.omissions > 0
    assert drift.hallucinations == 0
    assert drift.recall == 0.0
    assert drift.omission_ratio == float("inf")


def test_belief_rollout_tracks_staleness_when_faithful() -> None:
    """A faithful belief (oracle delta) sees the stale read; the no-op belief misses it."""
    dep = _stale_read_deployment()
    faithful = trace_deployment(_OracleDelta(CONFIG), _oracle(), dep)
    omitter = trace_deployment(_NoopDelta(), _oracle(), dep)
    # the read is at step 3 (index 2); its belief-before state is index 2
    assert is_stale(faithful.belief_states[2], "n1", "cfg")  # faithful belief tracks the medium
    assert not is_stale(omitter.belief_states[2], "n1", "cfg")  # the omitter's belief is frozen


def test_medium_reaches_zero_cheaply() -> None:
    """The medium target reaches the full oracle's breach rate far below the full-oracle cost."""
    result = run_cu19(_NoopDelta(), CONFIG)
    full = result.uniform[-1]
    assert result.medium.random_breach == 0.0
    assert result.medium.adversarial_breach == 0.0
    assert result.medium.mean_calls < full.mean_calls


def test_write_target_does_not_transfer() -> None:
    """Verifying the genesis writes spends calls yet still breaches the stale read."""
    dep = _stale_read_deployment()
    trace = trace_deployment(_NoopDelta(), _oracle(), dep)
    w_breached, w_calls = breach_calls(trace, CONFIG, "write_target", 0.0)
    m_breached, m_calls = breach_calls(trace, CONFIG, "medium", 0.0)
    assert w_breached and w_calls >= 1
    assert not m_breached and m_calls >= 1


def test_model_self_targeting_fails() -> None:
    """The omitter cannot flag staleness it omits: model breach ~ the free rate, at 0 calls."""
    result = run_cu19(_NoopDelta(), CONFIG)
    free = result.uniform[0]
    assert result.model.mean_calls == 0.0
    assert result.model.random_breach >= 0.5 * free.random_breach


def test_medium_is_ungameable_others_not() -> None:
    """Against adversarial timing, only the medium target survives on the omitter belief."""
    dep = _stale_read_deployment()
    trace = trace_deployment(_NoopDelta(), _oracle(), dep)
    assert not adversarial_breach(trace, CONFIG, "medium", 0.0)
    assert adversarial_breach(trace, CONFIG, "write_target", 0.0)
    assert adversarial_breach(trace, CONFIG, "model", 0.0)
    assert adversarial_breach(trace, CONFIG, "uniform", 0.5)  # the knee is a mirage
    assert not adversarial_breach(trace, CONFIG, "uniform", 1.0)  # only the full oracle is safe


def test_breach_calls_returns_types() -> None:
    """The cached-trace closed loop returns a (bool, int) pair for every schedule."""
    trace = trace_deployment(_NoopDelta(), _oracle(), _stale_read_deployment())
    for schedule in ("uniform", "model", "write_target", "medium"):
        breached, calls = breach_calls(trace, CONFIG, schedule, 0.5)
        assert isinstance(breached, bool)
        assert isinstance(calls, int)


def test_verdict_headline() -> None:
    """The CU19 verdict captures H112: omission-biased drift + the medium target closes the loop."""
    verdict = cu19_verdict(run_cu19(_NoopDelta(), CONFIG))
    assert verdict["drift_is_omission_biased"]
    assert verdict["medium_is_safe"]
    assert verdict["medium_cheaper_than_full"]
    assert verdict["medium_is_ungameable"]
    assert verdict["write_target_does_not_transfer"]
    assert verdict["model_self_targeting_fails"]
