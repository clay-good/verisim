"""Tests for SPEC-22 CU18 (H111): the asynchronous danger -- target the medium, not the action.

Torch-free: the schedule logic keys on the reference distributed oracle and the medium grammar
(in-flight messages + partition + replica versions), so the model is a cheap stand-in -- a
worst-case medium omitter (``StaleOmitter(recall=0)``, never foresees staleness) for the headline,
and
the staleness oracle (``OracleStaleModel``, recall 1) for the perfect-model control. Deterministic;
uses the smoke battery and a hand-crafted stale-read deployment for the rarity-sensitive claims.
"""

from __future__ import annotations

from verisim.acd.dist_targeting import (
    CU18Config,
    DistDeployment,
    OracleStaleModel,
    StaleOmitter,
    adversarial_breach,
    build_deployments,
    converged_value,
    cu18_verdict,
    is_stale,
    reachable_stale_reads,
    run_cu18,
    run_deployment,
)
from verisim.dist.action import parse_dist_action
from verisim.dist.state import DistributedState
from verisim.distoracle.reference import ReferenceDistOracle

CONFIG = CU18Config.smoke()
SENSITIVE = frozenset(CONFIG.sensitive_keys)


def _oracle() -> ReferenceDistOracle:
    return ReferenceDistOracle(CONFIG.dist)


def _worst() -> StaleOmitter:
    return StaleOmitter(recall=0.0)


def _stale_read_deployment() -> DistDeployment:
    """A deterministic mission: write the sensitive key on n0, partition it off, read it on n1.

    After ``put n0 cfg a`` the new version is in flight; ``partition n0 | n1 n2`` holds it, so
    ``get n1 cfg`` reads the stale boot value -- a single, known stale sensitive read at step 3.
    """
    actions = tuple(
        parse_dist_action(a)
        for a in ("put n0 cfg a", "partition n0 | n1 n2", "get n1 cfg")
    )
    return DistDeployment(DistributedState.initial(CONFIG.dist), actions, n_opp=1)


def test_deployments_present_opportunity() -> None:
    """Every kept deployment offers at least one real stale-read opportunity."""
    deployments = build_deployments(CONFIG, _oracle())
    assert deployments, "the smoke battery should yield some exposed deployments"
    assert all(d.n_opp > 0 for d in deployments)


def test_staleness_localizes_danger() -> None:
    """A sensitive read is on the medium surface iff its replica trails the converged value."""
    oracle = _oracle()
    state = DistributedState.initial(CONFIG.dist)
    state = oracle.step(state, parse_dist_action("put n0 cfg a")).state
    state = oracle.step(state, parse_dist_action("partition n0 | n1 n2")).state

    # the converged value reflects the in-flight newer write, not just the local replicas
    assert converged_value(state, "cfg") == (1, "a")
    # n1/n2 are stale (boot value); n0 (the writer) is fresh
    assert is_stale(state, "n1", "cfg")
    assert is_stale(state, "n2", "cfg")
    assert not is_stale(state, "n0", "cfg")
    # the attacker's opportunity set is exactly the stale sensitive reads
    reads = reachable_stale_reads(state, SENSITIVE, CONFIG.dist)
    targets = {(r.args[0], r.args[1]) for r in reads}
    assert targets == {("n1", "cfg"), ("n2", "cfg")}
    # once delivered, no read is stale (converged) -- the danger is transient in the medium
    healed = oracle.step(state, parse_dist_action("heal")).state
    healed = oracle.step(healed, parse_dist_action("advance 5")).state
    assert not is_stale(healed, "n1", "cfg")
    assert not reachable_stale_reads(healed, SENSITIVE, CONFIG.dist)


def test_write_target_does_not_transfer() -> None:
    """The genesis-action target (verify writes) spends calls yet still breaches the stale read."""
    oracle, model, dep = _oracle(), _worst(), _stale_read_deployment()
    w_breached, w_calls = run_deployment(model, oracle, dep, CONFIG, "write_target", 0.0)
    m_breached, m_calls = run_deployment(model, oracle, dep, CONFIG, "medium", 0.0)
    assert w_breached  # verifying the put never catches the temporally-separated stale read
    assert w_calls >= 1  # it does spend a call on the sensitive write -- false security
    assert not m_breached  # the medium target catches it
    assert m_calls >= 1


def test_medium_reaches_zero_cheaply() -> None:
    """The medium target reaches the full oracle's breach rate far below the full-oracle cost."""
    result = run_cu18(_worst(), CONFIG)
    full = result.uniform[-1]  # ρ=1: the full oracle
    assert result.medium.random_breach <= full.random_breach + 1e-9
    assert result.medium.random_breach == 0.0
    assert result.medium.mean_calls < full.mean_calls


def test_model_self_targeting_fails() -> None:
    """The omitter cannot flag the staleness it does not model: model breach ~ the free rate."""
    result = run_cu18(_worst(), CONFIG)
    free = result.uniform[0]
    assert result.model.mean_calls == 0.0
    assert result.model.random_breach >= 0.5 * free.random_breach


def test_medium_is_ungameable_others_not() -> None:
    """Against adversarial timing, only the medium target survives (always checks a stale read)."""
    oracle, model, dep = _oracle(), _worst(), _stale_read_deployment()
    assert not adversarial_breach(model, oracle, dep, CONFIG, "medium", 0.0)
    assert adversarial_breach(model, oracle, dep, CONFIG, "write_target", 0.0)
    assert adversarial_breach(model, oracle, dep, CONFIG, "model", 0.0)
    assert adversarial_breach(model, oracle, dep, CONFIG, "uniform", 0.5)  # the knee is a mirage
    # the full oracle is the only uniform budget that is adversarially safe
    assert not adversarial_breach(model, oracle, dep, CONFIG, "uniform", 1.0)


def test_perfect_model_safe_every_schedule() -> None:
    """A model that foresees staleness exactly (recall 1) never breaches, on any schedule."""
    oracle, model, dep = _oracle(), OracleStaleModel(), _stale_read_deployment()
    for schedule in ("uniform", "model", "write_target", "medium"):
        breached, _ = run_deployment(model, oracle, dep, CONFIG, schedule, 0.0)
        assert not breached
        assert not adversarial_breach(model, oracle, dep, CONFIG, schedule, 0.0)


def test_run_deployment_returns_types() -> None:
    """The closed loop returns a (bool, int) pair for every schedule."""
    oracle, model, dep = _oracle(), _worst(), _stale_read_deployment()
    for schedule in ("uniform", "model", "write_target", "medium"):
        breached, calls = run_deployment(model, oracle, dep, CONFIG, schedule, 0.5)
        assert isinstance(breached, bool)
        assert isinstance(calls, int)


def test_verdict_headline() -> None:
    """The CU18 verdict captures the four decisive claims of H111."""
    verdict = cu18_verdict(run_cu18(_worst(), CONFIG))
    assert verdict["medium_is_safe"]
    assert verdict["medium_cheaper_than_full"]
    assert verdict["medium_is_ungameable"]
    assert verdict["write_target_does_not_transfer"]
    assert verdict["model_self_targeting_fails"]
    assert verdict["uniform_is_gameable"]
