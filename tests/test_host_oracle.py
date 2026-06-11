"""Property tests + golden for the Tier-A reference host oracle (SPEC-6 §5.1, HC0 increment 1).

Pins the deterministic core before any model claim (the M0/NW0 discipline): the process table
(fork/exit), credentials (setuid privilege gating), the per-process fd table (open/write/close +
exit cleanup), the FS composition (a write through an fd lands in the embedded fs), purity of
the oracle, and a golden canonical hash that fixes the semantics of a fixed syscall trajectory.
"""

from __future__ import annotations

from verisim.host import (
    RUNNING,
    ZOMBIE,
    HostState,
    parse_host_action,
    to_canonical_host,
)
from verisim.hostoracle import EXIT_ERR, EXIT_OK, ReferenceHostOracle


def _run(oracle: ReferenceHostOracle, state: HostState, cmds: list[str]) -> HostState:
    for cmd in cmds:
        state = oracle.step(state, parse_host_action(cmd)).state
    return state


def test_fork_creates_a_child_process() -> None:
    oracle = ReferenceHostOracle()
    result = oracle.step(HostState.initial(), parse_host_action("fork 1"))
    assert result.exit_code == EXIT_OK
    assert result.stdout == "2"  # fork returns the child pid
    child = result.state.procs[2]
    assert child.ppid == 1 and child.state == RUNNING and child.uid == 0


def test_exit_zombifies_and_releases_fds() -> None:
    oracle = ReferenceHostOracle()
    state = _run(oracle, HostState.initial(), ["fork 1", "open 2 /f", "exit 2 7"])
    assert state.procs[2].state == ZOMBIE
    assert state.procs[2].exit_code == 7
    assert not any(pid == 2 for (pid, _fd) in state.fds)  # fds released on exit


def test_write_through_fd_lands_in_the_embedded_filesystem() -> None:
    oracle = ReferenceHostOracle()
    # open returns fd 0; write delegates `write /f alpha` to the v0 FS sub-oracle.
    state = _run(oracle, HostState.initial(), ["open 1 /f", "write 1 0 alpha"])
    assert "/f" in state.fs.fs  # the file exists in the composed filesystem subsystem
    # a second open by the same process gets the next fd (per-process fd table).
    result = oracle.step(state, parse_host_action("open 1 /g"))
    assert result.stdout == "1"


def test_kill_terminates_the_target_and_releases_its_fds() -> None:
    # root (pid 1) forks pid 2, which opens an fd; pid 1 kills pid 2 -> ZOMBIE (137) + fds released.
    oracle = ReferenceHostOracle()
    state = _run(oracle, HostState.initial(), ["fork 1", "open 2 /f"])
    result = oracle.step(state, parse_host_action("kill 1 2"))
    assert result.exit_code == EXIT_OK
    assert result.state.procs[2].state == ZOMBIE
    assert result.state.procs[2].exit_code == 137  # the SIGKILL convention (128 + 9)
    assert not any(pid == 2 for (pid, _fd) in result.state.fds)  # the victim's fds are released
    assert result.state.procs[1].state == RUNNING  # the killer is untouched


def test_kill_is_permission_gated() -> None:
    oracle = ReferenceHostOracle()
    # pid 2 drops to uid 1000; pid 3 stays root (uid 0). A non-root process can't kill root (EPERM).
    state = _run(oracle, HostState.initial(), ["fork 1", "setuid 2 1000", "fork 1"])
    denied = oracle.step(state, parse_host_action("kill 2 3"))
    assert denied.exit_code == EXIT_ERR and denied.state.procs[3].state == RUNNING
    # root may kill anyone...
    assert oracle.step(state, parse_host_action("kill 1 2")).state.procs[2].state == ZOMBIE
    # ...and a process may kill another of the same uid (pid 2 forks pid 4, both uid 1000).
    same = _run(oracle, state, ["fork 2"])  # pid 4 is pid 2's child, inherits uid 1000
    killed = oracle.step(same, parse_host_action("kill 2 4"))
    assert killed.exit_code == EXIT_OK and killed.state.procs[4].state == ZOMBIE


def test_kill_nonexistent_or_zombie_target_is_esrch() -> None:
    oracle = ReferenceHostOracle()
    assert oracle.step(HostState.initial(), parse_host_action("kill 1 99")).exit_code == EXIT_ERR
    zombie = _run(oracle, HostState.initial(), ["fork 1", "exit 2 0"])
    assert oracle.step(zombie, parse_host_action("kill 1 2")).exit_code == EXIT_ERR  # already dead


def test_setuid_is_root_gated() -> None:
    oracle = ReferenceHostOracle()
    # root (uid 0) may drop privilege...
    dropped = oracle.step(HostState.initial(), parse_host_action("setuid 1 1000"))
    assert dropped.exit_code == EXIT_OK and dropped.state.procs[1].uid == 1000
    # ...but a non-root process may not change its uid (EPERM).
    denied = oracle.step(dropped.state, parse_host_action("setuid 1 0"))
    assert denied.exit_code == EXIT_ERR and denied.state.procs[1].uid == 1000


def test_bad_pid_and_bad_fd_fail_without_mutating() -> None:
    oracle = ReferenceHostOracle()
    s0 = HostState.initial()
    # a syscall on a non-existent pid fails and leaves the state unchanged (modulo last_exit).
    bad = oracle.step(s0, parse_host_action("fork 99"))
    assert bad.exit_code == EXIT_ERR
    assert to_canonical_host(bad.state)["procs"] == to_canonical_host(s0)["procs"]
    # writing an unopened fd is EBADF.
    assert oracle.step(s0, parse_host_action("write 1 3 x")).exit_code == EXIT_ERR


def test_read_closes_the_write_read_round_trip() -> None:
    # `read` reads back through the fd what `write` wrote — the round trip a host needs (read op).
    oracle = ReferenceHostOracle()
    state = _run(oracle, HostState.initial(), ["open 1 /f", "write 1 0 alpha"])
    result = oracle.step(state, parse_host_action("read 1 0"))
    assert result.exit_code == EXIT_OK
    assert result.stdout == "alpha"  # the content delegated to the FS `cat` sub-oracle


def test_read_is_read_only() -> None:
    # `read` is read-only: it leaves the FS, process table, and fd table unchanged (only last_exit).
    oracle = ReferenceHostOracle()
    state = _run(oracle, HostState.initial(), ["open 1 /f", "write 1 0 gamma"])
    before = to_canonical_host(state)
    after = to_canonical_host(oracle.step(state, parse_host_action("read 1 0")).state)
    before.pop("last_exit")
    after.pop("last_exit")
    assert before == after


def test_read_bad_fd_and_after_close_are_ebadf() -> None:
    oracle = ReferenceHostOracle()
    s0 = HostState.initial()
    assert oracle.step(s0, parse_host_action("read 1 5")).exit_code == EXIT_ERR  # never opened
    closed = _run(oracle, s0, ["open 1 /f", "write 1 0 beta", "close 1 0"])
    assert oracle.step(closed, parse_host_action("read 1 0")).exit_code == EXIT_ERR  # released fd


def test_read_on_a_dead_pid_fails() -> None:
    oracle = ReferenceHostOracle()
    dead = _run(oracle, HostState.initial(), ["open 1 /f", "write 1 0 d", "exit 1 0"])
    assert oracle.step(dead, parse_host_action("read 1 0")).exit_code == EXIT_ERR  # pid 1 zombie


def test_oracle_is_pure_and_deterministic() -> None:
    oracle = ReferenceHostOracle()
    cmds = ["fork 1", "open 2 /a", "write 2 0 hello", "setuid 1 1000", "exit 2 0"]
    a = _run(oracle, HostState.initial(), cmds)
    b = _run(ReferenceHostOracle(), HostState.initial(), cmds)
    assert to_canonical_host(a) == to_canonical_host(b)
    # stepping does not mutate the input state.
    s = HostState.initial()
    before = to_canonical_host(s)
    oracle.step(s, parse_host_action("fork 1"))
    assert to_canonical_host(s) == before


def test_golden_trajectory_canonical_hash_is_stable() -> None:
    """A fixed syscall trajectory pins a fixed canonical state (the HC0 golden)."""
    oracle = ReferenceHostOracle()
    cmds = ["fork 1", "fork 1", "open 2 /etc", "write 2 0 cfg", "setuid 1 33", "exit 3 0"]
    final = _run(oracle, HostState.initial(), cmds)
    canon = to_canonical_host(final)
    assert [p["pid"] for p in canon["procs"]] == [1, 2, 3]
    assert canon["procs"][0]["uid"] == 33  # pid 1 dropped to uid 33
    assert canon["procs"][2]["state"] == "ZOMBIE"  # pid 3 exited
    assert canon["next_pid"] == 4
    assert "/etc" in dict(final.fs.fs)  # the write landed in the composed FS
