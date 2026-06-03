"""Host scheduler + interleaving-entropy tests (SPEC-6 §3.2-3.4, HC7 H14 apparatus).

Torch-free unit tests of the concurrency dial:

  - schedules are deterministic given ``(workload, interleave, chaos_seed)`` and replay from boot to
    a fixed trajectory (the §3.3 determinism contract);
  - every worker thread completes its ``fork→open→write→close→exit`` program (a valid schedule);
  - the realized interleaving entropy (thread context-switch rate) rises monotonically with the
    chaos knob -- ``interleave=0`` is near-sequential, ``interleave=1`` is maximally interleaved.
"""

from __future__ import annotations

import hashlib
import json

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.state import HostState, to_canonical_host
from verisim.hostdata import HostScheduler, interleaving_entropy, make_workload
from verisim.hostoracle.reference import ReferenceHostOracle

CONFIG = DEFAULT_HOST_CONFIG


def _replay_hash(actions) -> str:
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    for action in actions:
        state = oracle.step(state, action).state
    return hashlib.sha256(json.dumps(to_canonical_host(state), sort_keys=True).encode()).hexdigest()


def test_schedule_is_deterministic_and_replayable():
    wl = make_workload(CONFIG, n_threads=4, seed=0)
    a = HostScheduler(CONFIG, interleave=0.5).schedule(wl, chaos_seed=3)
    b = HostScheduler(CONFIG, interleave=0.5).schedule(wl, chaos_seed=3)
    assert [x.raw for x in a.actions] == [x.raw for x in b.actions]
    assert a.thread_ids == b.thread_ids
    # the emitted sequence replays from boot without error (concrete, deterministic)
    assert len(_replay_hash(a.actions)) == 64


def test_every_thread_completes_its_program():
    n = 4
    wl = make_workload(CONFIG, n_threads=n, seed=1)
    sched = HostScheduler(CONFIG, interleave=1.0).schedule(wl, chaos_seed=0)
    # 5 ops per thread (fork/open/write/close/exit)
    assert len(sched.actions) == 5 * n
    assert len(sched.thread_ids) == len(sched.actions)
    assert set(sched.thread_ids) == set(range(n))  # every thread ran
    # each thread forks exactly once (acting pid 1) and exits exactly once
    forks = sum(1 for x in sched.actions if x.name == "fork")
    exits = sum(1 for x in sched.actions if x.name == "exit")
    assert forks == exits == n


def test_interleaving_entropy_rises_with_the_chaos_knob():
    wl = make_workload(CONFIG, n_threads=5, seed=0)
    rates = {
        il: interleaving_entropy(HostScheduler(CONFIG, interleave=il).schedule(wl, 0).thread_ids)
        for il in (0.0, 0.5, 1.0)
    }
    # near-sequential at 0 (switches only at thread boundaries), much higher at 1
    assert rates[0.0] < rates[0.5] < rates[1.0]
    assert rates[0.0] < 0.3 and rates[1.0] > 0.5


def test_interleaving_entropy_edge_cases():
    assert interleaving_entropy([]) == 0.0
    assert interleaving_entropy([2]) == 0.0
    assert interleaving_entropy([0, 0, 0]) == 0.0  # no switches
    assert interleaving_entropy([0, 1, 0, 1]) == 1.0  # switch every step
