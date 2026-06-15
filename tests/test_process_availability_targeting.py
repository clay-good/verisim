"""Tests for SPEC-22 CU23 (H116): the second generative test of the unified target framework.

Torch-free throughout: the schedule result keys on the exact host oracle and the danger's model-free
process-liveness surface, so the model is the worst-case omitter (the headline substrate) or a
perfect oracle control. CU23 applies the CU21 ``unified_targeting`` engine to a second danger the
arc never saw -- host availability (terminating a critical defensive daemon) -- and checks that the
``covers`` invariant predicts every candidate's fate a priori, including the cross-world contrast
that the *syntactic* class covers here (no cascade) though it leaked in CU22 (multi-hop).
"""

from __future__ import annotations

from functools import lru_cache

from verisim.acd.process_availability_targeting import (
    CU23Config,
    CU23Result,
    _candidate,
    _daemon_dies,
    _kills_running_daemon,
    build_deployments,
    cu23_verdict,
    provision_daemons,
    run_cu23,
    terminate_attacks,
)
from verisim.host.action import parse_host_action
from verisim.host.state import RUNNING, HostState, Process
from verisim.hostoracle.reference import ReferenceHostOracle


def _small() -> CU23Config:
    return CU23Config(horizon=24, n_seeds=300, max_episodes=10, rhos=(0.0, 0.5, 1.0))


@lru_cache(maxsize=1)
def _result() -> CU23Result:
    return run_cu23(_small())


def test_deployments_present_an_outage_opportunity() -> None:
    config = _small()
    oracle = ReferenceHostOracle()
    deps = build_deployments(config, oracle)
    assert deps, "expected at least one deployment with a daemon-termination opportunity"
    for d in deps:
        assert d.daemons, "every deployment must have a critical-daemon fleet to protect"
        assert d.n_opportunities > 0


def test_provision_daemons_creates_a_running_root_fleet() -> None:
    oracle = ReferenceHostOracle()
    start, daemons = provision_daemons(3, oracle)
    assert daemons == frozenset({2, 3, 4})
    for d in daemons:
        proc = start.procs[d]
        assert proc.state == RUNNING
        assert proc.uid == 0  # forked off root init, so the daemon is privileged


def test_terminate_attacks_truly_kill_a_daemon() -> None:
    config = _small()
    oracle = ReferenceHostOracle()
    deps = build_deployments(config, oracle)
    saw_arsenal = False
    for d in deps:
        attacks = terminate_attacks(d.start, d.daemons, oracle)
        for a in attacks:  # every action in the arsenal must truly terminate a running daemon
            assert _daemon_dies(d.start, oracle.step(d.start, a).state, d.daemons)
        saw_arsenal = saw_arsenal or bool(attacks)
    assert saw_arsenal, "expected a reachable daemon termination somewhere in the battery"


def test_no_cascade_a_daemon_dies_only_by_naming_it() -> None:
    """The decisive unit test (the CU22 contrast): process death has no cascade.

    A daemon (pid 2) is a child of init (pid 1). Killing its *parent* does NOT kill the daemon -- it
    orphans but keeps running. So the daemon dies only by an action that NAMES it (``kill 1 2`` or
    ``exit 2``), which is why the *syntactic* terminate target covers here, unlike CU22's syntactic
    disconnect rule, where a multi-hop ``link_down`` could sever a pair without naming an endpoint.
    """
    oracle = ReferenceHostOracle()
    state = HostState(
        procs={
            1: Process(pid=1, ppid=0, state=RUNNING, uid=0),  # init / the agent (root)
            2: Process(pid=2, ppid=1, state=RUNNING, uid=0),  # the critical daemon
            5: Process(pid=5, ppid=2, state=RUNNING, uid=0),  # an unrelated child of the daemon
        },
        next_pid=6,
    )
    daemons = frozenset({2})

    # killing the daemon's CHILD (pid 5) does not kill the daemon -> off the liveness surface
    kill_child = parse_host_action("kill 1 5")
    assert not _daemon_dies(state, oracle.step(state, kill_child).state, daemons)
    assert not _kills_running_daemon(state, kill_child, daemons)

    # killing the daemon directly (naming it) terminates it -> on the liveness surface
    kill_daemon = parse_host_action("kill 1 2")
    assert _daemon_dies(state, oracle.step(state, kill_daemon).state, daemons)
    assert _kills_running_daemon(state, kill_daemon, daemons)

    # the daemon exiting itself also names it -> on the surface
    exit_daemon = parse_host_action("exit 2 0")
    assert _daemon_dies(state, oracle.step(state, exit_daemon).state, daemons)
    assert _kills_running_daemon(state, exit_daemon, daemons)

    # a benign open of a file does not touch daemon liveness -> off the surface
    benign = parse_host_action("open 1 /passwd")
    assert not _daemon_dies(state, oracle.step(state, benign).state, daemons)
    assert not _kills_running_daemon(state, benign, daemons)


def test_derived_liveness_closure_is_safe_cheap_and_ungameable() -> None:
    liveness = _candidate(_result(), "liveness")
    full = _result().full_oracle
    assert liveness.covers is True  # coverage holds by construction (liveness surface = realizes)
    assert liveness.random_breach <= full.random_breach + 1e-9  # the oracle's safety
    assert liveness.adversarial_breach <= 1e-9  # un-gameable
    assert liveness.mean_calls < full.mean_calls  # cheaper than verifying everything


def test_cu16_integrity_carry_over_leaks() -> None:
    """The host's own CU16 target (write-to-fd): a termination is not a write -> false security."""
    write = _candidate(_result(), "write")
    assert write.covers is False
    assert write.adversarial_breach > 1e-9


def test_syntactic_covers_here_the_opposite_of_cu22() -> None:
    """The cross-world contrast: the syntactic class COVERS here (no cascade), unlike CU22.

    It is safe and un-gameable, but overpays vs the derived closure (it consults every benign
    process exit, not only the daemon-terminating ones).
    """
    syntactic = _candidate(_result(), "syntactic")
    liveness = _candidate(_result(), "liveness")
    assert syntactic.covers is True
    assert syntactic.adversarial_breach <= 1e-9
    assert syntactic.mean_calls > liveness.mean_calls  # overpays vs the precise closure


def test_model_self_targeting_fails() -> None:
    v = cu23_verdict(_result())
    assert v["model_self_targeting_fails"] is True


def test_perfect_model_self_governs() -> None:
    v = cu23_verdict(_result())
    assert v["oracle_self_governs"] is True


def test_framework_predicts_every_candidate() -> None:
    """The headline (H116): covers() predicted each candidate's fate; the run confirms it."""
    v = cu23_verdict(_result())
    assert v["framework_predicts_every_candidate"] is True
    assert v["write_covers"] is False and v["write_leaks"] is True
    assert v["syntactic_covers"] is True
    assert v["liveness_covers"] is True


def test_run_cu23_returns_types_and_verdict() -> None:
    r = _result()
    assert r.n_episodes > 0
    assert {c.name for c in r.candidates} == {"write", "syntactic", "liveness"}
    v = cu23_verdict(r)
    saving = v["liveness_call_saving"]
    assert isinstance(saving, float) and saving > 1.0  # the derived target is cheaper than full
    assert v["uniform_is_gameable"] is True
