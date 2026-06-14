"""Tests for SPEC-22 CU17 (H110): the genesis-grammar boundary (segmentation exposure).

Torch-free throughout: the schedule result keys on the oracle and the grammar, so the model is a
worst-case content omitter (the headline substrate) or a perfect oracle stand-in (the control). The
trained network/host ``M_theta`` is the deferred GPU arm (LP7).
"""

from __future__ import annotations

from verisim.acd.segmentation_targeting import (
    CU17Config,
    NetOmitter,
    Segment,
    _flips_reachability,
    _genesis_grammar,
    adversarial_breach,
    build_segments,
    cu17_verdict,
    isolate_jewels,
    run_cu17,
    run_segment,
)
from verisim.net.action import NetAction
from verisim.net.state import HostState, NetworkState, can_reach, link_key
from verisim.netdelta.edits import NetEdit
from verisim.netoracle.reference import ReferenceNetworkOracle


class _OracleModel:
    """A perfect model: its preview is the oracle's true step (foresees every exposure)."""

    def __init__(self) -> None:
        self.oracle = ReferenceNetworkOracle()

    def predict_delta(self, state: NetworkState, action: NetAction) -> list[NetEdit]:
        return self.oracle.step(state, action).delta


def _small_config() -> CU17Config:
    return CU17Config(horizon=16, n_seeds=160, max_episodes=10, rhos=(0.0, 0.5, 1.0))


def test_isolate_jewels_segments_the_crown_jewels() -> None:
    oracle = ReferenceNetworkOracle()
    config = _small_config()
    jewels = frozenset(config.jewels)
    ports = tuple(config.net.ports)
    from verisim.acd.net_integrity import make_net_workload

    raw, _ = make_net_workload(config.seed0, config.horizon, oracle=oracle, config=config.net)
    start = isolate_jewels(raw, jewels, ports)
    untrusted = tuple(h for h in sorted(start.hosts) if h not in jewels)
    # after isolation no jewel is reachable from any untrusted host
    assert not any(
        can_reach(start, u, j, p) for j in jewels for u in untrusted for p in ports
    )


def test_segments_present_an_exposure_opportunity() -> None:
    oracle = ReferenceNetworkOracle()
    segments = build_segments(_small_config(), oracle)
    assert segments, "expected at least one segmented deployment with an exposure opportunity"
    assert all(s.n_opportunities > 0 for s in segments)


def test_closure_localizes_a_multi_hop_intermediate_exposure() -> None:
    """The decisive unit test: a ``host_up`` of an intermediate relay exposes a jewel.

    ``h2 (untrusted) -- h1 (relay, DOWN) -- h0 (jewel)``: the path is broken while ``h1`` is down,
    so the jewel is segmented. Bringing ``h1`` up completes the path -- an exposure born by a
    ``host_up`` that names *neither* the jewel nor a connect. It is on the semantic reachability
    *closure* surface but off both the ``connect`` target and the *syntactic* genesis grammar.
    """
    oracle = ReferenceNetworkOracle()
    state = NetworkState(
        hosts={
            "h0": HostState(up=True, services=(0,)),  # the jewel: up, listening
            "h1": HostState(up=False),  # the relay: DOWN (breaks the path)
            "h2": HostState(up=True),  # the untrusted host
        },
        links={link_key("h0", "h1"), link_key("h1", "h2")},
    )
    jewels, untrusted, ports = frozenset({"h0"}), ("h2",), (0,)
    assert not can_reach(state, "h2", "h0", 0)  # segmented while the relay is down

    host_up_relay = NetAction(raw="host_up h1", name="host_up", args=("h1",))
    # closure (semantic): bringing the relay up exposes the jewel -> on the surface
    assert _flips_reachability(state, host_up_relay, oracle, untrusted, jewels, ports)
    # grammar (syntactic): the action names h1, not a jewel -> OFF the grammar surface (the leak)
    assert not _genesis_grammar(host_up_relay, jewels)
    # connect target: a host_up is not a connect -> also off it
    assert host_up_relay.name != "connect"

    # a benign config action that does NOT expose the jewel is off the closure surface
    benign = NetAction(raw="svc_up h2 0", name="svc_up", args=("h2", "0"))
    assert not _flips_reachability(state, benign, oracle, untrusted, jewels, ports)


def test_connect_target_does_not_transfer() -> None:
    """The CU10-CU16 connect target is blind to the config genesis -> breach at the free rate."""
    oracle = ReferenceNetworkOracle()
    config = _small_config()
    segments = build_segments(config, oracle)
    omitter = NetOmitter()
    free = [run_segment(omitter, oracle, s, config, "uniform", 0.0)[0] for s in segments]
    connect = [run_segment(omitter, oracle, s, config, "connect", 0.0)[0] for s in segments]
    free_rate = sum(free) / len(free)
    connect_rate = sum(connect) / len(connect)
    assert free_rate > 0.5  # the omitter exposes the jewels when unverified
    assert connect_rate >= 0.5 * free_rate  # connect targeting buys ~nothing against this danger


def test_closure_reaches_zero_breach_cheaply() -> None:
    oracle = ReferenceNetworkOracle()
    config = _small_config()
    segments = build_segments(config, oracle)
    omitter = NetOmitter()
    breaches = [run_segment(omitter, oracle, s, config, "closure", 0.0)[0] for s in segments]
    calls = [run_segment(omitter, oracle, s, config, "closure", 0.0)[1] for s in segments]
    full_calls = [run_segment(omitter, oracle, s, config, "uniform", 1.0)[1] for s in segments]
    assert not any(breaches)  # the closure target reaches zero breach
    assert sum(calls) < sum(full_calls)  # at fewer calls than the full oracle


def test_grammar_is_gameable_closure_is_not() -> None:
    """Syntactic grammar leaks through a multi-hop intermediate; the reachability closure does not.

    A hand-crafted segment whose only exposure is a ``host_up`` of an intermediate relay (the leak
    of the multi-hop probe): the syntactic grammar does not name it, so an adversary slips it past
    grammar targeting, while the semantic closure target covers it.
    """
    oracle = ReferenceNetworkOracle()
    config = CU17Config(jewels=("h0",))
    start = NetworkState(
        hosts={
            "h0": HostState(up=True, services=(22,)),  # the jewel (listening on a real config port)
            "h1": HostState(up=False),  # the relay: down -> the path is broken
            "h2": HostState(up=True),  # the untrusted host
        },
        links={link_key("h0", "h1"), link_key("h1", "h2")},
    )
    segment = Segment(start=start, actions=(NetAction(raw="advance", name="advance", args=()),),
                      untrusted=("h2",), n_opportunities=1)
    omitter = NetOmitter()
    assert adversarial_breach(omitter, oracle, segment, config, "grammar", 0.0)  # grammar leaks
    assert not adversarial_breach(omitter, oracle, segment, config, "closure", 0.0)  # closure ok


def test_perfect_model_safe_every_schedule() -> None:
    """A perfect model foresees the exposure, so even the free agent never breaches."""
    oracle = ReferenceNetworkOracle()
    config = _small_config()
    segments = build_segments(config, oracle)
    model = _OracleModel()
    for schedule in ("uniform", "connect", "grammar", "closure"):
        breaches = [
            run_segment(model, oracle, s, config, schedule, 0.0)[0]
            for s in segments
        ]
        assert not any(breaches)


def test_run_segment_returns_breach_and_calls() -> None:
    oracle = ReferenceNetworkOracle()
    config = _small_config()
    segments = build_segments(config, oracle)
    assert isinstance(segments[0], Segment)
    breached, calls = run_segment(NetOmitter(), oracle, segments[0], config, "closure", 0.0)
    assert isinstance(breached, bool)
    assert isinstance(calls, int) and calls >= 0


def test_verdict_headline() -> None:
    result = run_cu17(NetOmitter(), _small_config())
    verdict = cu17_verdict(result)
    assert verdict["connect_fails_to_transfer"] is True
    assert verdict["closure_is_safe"] is True
    assert verdict["closure_is_ungameable"] is True
    assert verdict["closure_cheaper_than_full"] is True
