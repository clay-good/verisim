"""Tests for SPEC-22 CU22 (H115): the generative test of the unified target framework.

Torch-free throughout: the schedule result keys on the exact oracle and the danger's model-free
surface, so the model is the worst-case omitter (the headline substrate) or a perfect oracle
control. CU22 applies the CU21 ``unified_targeting`` engine to a danger the arc never saw
(availability / a self-inflicted outage) and checks that the ``covers`` invariant predicts every
candidate target's fate a priori.
"""

from __future__ import annotations

from functools import lru_cache

from verisim.acd.availability_targeting import (
    CU22Config,
    CU22Result,
    _breaks,
    _candidate,
    _names_work,
    build_deployments,
    cu22_verdict,
    disconnect_attacks,
    provision_work,
    required_pairs,
    run_cu22,
)
from verisim.net.action import NetAction
from verisim.net.state import HostState, NetworkState, can_reach, link_key
from verisim.netoracle.reference import ReferenceNetworkOracle


def _small() -> CU22Config:
    return CU22Config(horizon=16, n_seeds=240, max_episodes=10, rhos=(0.0, 0.5, 1.0))


@lru_cache(maxsize=1)
def _result() -> CU22Result:
    return run_cu22(_small())


def test_deployments_present_an_outage_opportunity() -> None:
    config = _small()
    oracle = ReferenceNetworkOracle()
    deps = build_deployments(config, oracle)
    assert deps, "expected at least one deployment with a disconnection opportunity"
    for d in deps:
        assert d.pairs, "every deployment must have a required-connectivity SLA to protect"
        assert d.n_opportunities > 0


def test_provision_work_creates_required_pairs() -> None:
    config = _small()
    oracle = ReferenceNetworkOracle()
    work = frozenset(config.work)
    ports = tuple(config.net.ports)
    from verisim.acd.net_integrity import make_net_workload

    raw, _ = make_net_workload(config.seed0, config.horizon, oracle=oracle, config=config.net)
    start = provision_work(raw, work, ports)
    # every provisioned work host is up + listening on the critical port
    for w in work:
        assert start.hosts[w].up
        assert ports[0] in start.hosts[w].services
    # the provisioned start exposes a non-empty connectivity SLA among the work hosts
    assert required_pairs(start, work, ports)


def test_disconnect_attacks_truly_break_a_required_pair() -> None:
    config = _small()
    oracle = ReferenceNetworkOracle()
    deps = build_deployments(config, oracle)
    saw_arsenal = False
    for d in deps:
        attacks = disconnect_attacks(d.start, d.pairs, oracle, config)
        for a in attacks:  # every action in the arsenal must truly disconnect a required pair
            assert _breaks(d.start, oracle.step(d.start, a).state, d.pairs)
        saw_arsenal = saw_arsenal or bool(attacks)
    assert saw_arsenal, "expected a reachable disconnection somewhere in the battery"


def test_closure_localizes_a_multi_hop_disconnection() -> None:
    """The decisive unit test: a ``link_down`` between two non-work relays severs a work pair.

    ``h1 (work) -- h0 (relay) -- h4 (relay) -- h3 (work)``: the only path between the work hosts
    runs through two NON-work relays. Cutting the ``h0--h4`` link disconnects the work pair -- an
    outage born by a ``link_down`` that names *neither* work host. It is on the semantic disconnect
    *closure* surface (``_breaks``) but off the *syntactic* target (it names no work host) -- the
    leak the framework predicts via ``covers``.
    """
    oracle = ReferenceNetworkOracle()
    state = NetworkState(
        hosts={
            "h1": HostState(up=True, services=(0,)),  # work endpoint, listening
            "h3": HostState(up=True, services=(0,)),  # work endpoint, listening
            "h0": HostState(up=True),  # non-work relay
            "h4": HostState(up=True),  # non-work relay
        },
        links={link_key("h1", "h0"), link_key("h0", "h4"), link_key("h4", "h3")},
    )
    work = frozenset({"h1", "h2", "h3"})
    pairs = frozenset({("h1", "h3", 0), ("h3", "h1", 0)})
    assert can_reach(state, "h1", "h3", 0) and can_reach(state, "h3", "h1", 0)

    cut = NetAction(raw="link_down h0 h4", name="link_down", args=("h0", "h4"))
    nxt = oracle.step(state, cut).state
    # closure (semantic): the cut disconnects a work pair -> on the surface
    assert _breaks(state, nxt, pairs)
    # syntactic: link_down h0 h4 names neither work host -> OFF the syntactic surface (the leak)
    assert not _names_work(cut, work)

    # a benign action that adds a service does not disconnect a work pair -> off the closure surface
    benign = NetAction(raw="svc_up h1 80", name="svc_up", args=("h1", "80"))
    assert not _breaks(state, oracle.step(state, benign).state, pairs)


def test_derived_closure_is_safe_cheap_and_ungameable() -> None:
    closure = _candidate(_result(), "closure")
    full = _result().full_oracle
    assert closure.covers is True  # coverage holds by construction (target == realizes)
    assert closure.random_breach <= full.random_breach + 1e-9  # the oracle's safety
    assert closure.adversarial_breach <= 1e-9  # un-gameable
    assert closure.mean_calls < full.mean_calls  # cheaper than verifying everything


def test_connect_carry_over_leaks() -> None:
    """The CU10 connect-to-jewel target: a disconnect is not a connect -> never fires -> leaks."""
    connect = _candidate(_result(), "connect")
    assert connect.covers is False
    assert connect.adversarial_breach > 1e-9


def test_exposure_carry_over_leaks() -> None:
    """The CU17 exposure-closure target (the seductive cousin): wrong polarity -> never fires."""
    exposure = _candidate(_result(), "exposure")
    assert exposure.covers is False
    assert exposure.adversarial_breach > 1e-9


def test_model_self_targeting_fails() -> None:
    v = cu22_verdict(_result())
    assert v["model_self_targeting_fails"] is True


def test_perfect_model_self_governs() -> None:
    v = cu22_verdict(_result())
    assert v["oracle_self_governs"] is True


def test_framework_predicts_every_candidate() -> None:
    """The headline (H115): covers() predicted each candidate's fate; the run confirms it."""
    v = cu22_verdict(_result())
    assert v["framework_predicts_every_candidate"] is True
    assert v["carried_over_all_break_coverage"] is True
    assert v["carried_over_all_leak"] is True
    assert v["closure_covers"] is True


def test_run_cu22_returns_types_and_verdict() -> None:
    r = _result()
    assert r.n_episodes > 0
    assert {c.name for c in r.candidates} == {"connect", "exposure", "syntactic", "closure"}
    v = cu22_verdict(r)
    saving = v["closure_call_saving"]
    assert isinstance(saving, float) and saving > 1.0  # the derived target is cheaper than full
    assert v["uniform_is_gameable"] is True
