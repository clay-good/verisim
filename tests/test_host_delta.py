"""The HC1 invariant tests: compositional ``apply`` and ``apply == oracle`` (SPEC-6 §4).

The contract that keeps the host loop model-agnostic, the analogue of v0's M1 and the network's NW1:
for every transition, ``apply(state, oracle.delta) == oracle.next_state`` by construction, and the
bundle delta round-trips through serialization. Pins the deterministic core before any model claim.
"""

from __future__ import annotations

from verisim.host import (
    HostState,
    delta_from_list,
    delta_to_list,
    parse_host_action,
    to_canonical_host,
)
from verisim.host.delta import apply
from verisim.hostoracle import ReferenceHostOracle

# A mixed trajectory exercising every implemented edit type + the FS-composition delta.
_TRAJECTORY = [
    "fork 1",          # ProcSpawn
    "mkdir 1 /etc",    # FsDelta wrapping a v0 Create(Dir) (so the nested write below has a parent)
    "open 2 /etc/cfg", # FdOpen
    "write 2 0 alpha", # FsDelta (delegated to the v0 FS sub-oracle)
    "dup 2 0",         # FdOpen via dup (alias fd 1 onto fd 0's path -- no new edit type)
    "setuid 1 1000",   # CredChange
    "close 2 0",       # FdClose
    "fork 1",          # ProcSpawn (another child)
    "exit 2 0",        # ProcExit (+ implied fd release)
    "wait 1 2",        # ProcReap (collect the zombie, free the table entry)
    "fork 1",          # ProcSpawn (pid 4 -- pids are not reused after reaping)
    "kill 1 4",        # ProcExit via kill (zombify pid 4)
    "write 1 9 x",     # a failing syscall (EBADF) -> SetExit only
]


def test_apply_equals_oracle_over_a_trajectory() -> None:
    """``apply(state, result.delta)`` reproduces ``result.state`` each step (the M1-analogue)."""
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    for cmd in _TRAJECTORY:
        result = oracle.step(state, parse_host_action(cmd))
        rebuilt = apply(state, result.delta)
        assert to_canonical_host(rebuilt) == to_canonical_host(result.state)
        state = result.state


def test_apply_does_not_mutate_input() -> None:
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    result = oracle.step(state, parse_host_action("fork 1"))
    before = to_canonical_host(state)
    apply(state, result.delta)
    assert to_canonical_host(state) == before  # apply built a fresh state


def test_delta_serialization_round_trips() -> None:
    """The bundle delta (incl. the embedded FS delta) survives to-list / from-list unchanged."""
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    for cmd in _TRAJECTORY:
        result = oracle.step(state, parse_host_action(cmd))
        round_tripped = delta_from_list(delta_to_list(result.delta))
        # The round-tripped delta applies to the same next state — the operational equivalence.
        assert to_canonical_host(apply(state, round_tripped)) == to_canonical_host(result.state)
        state = result.state
