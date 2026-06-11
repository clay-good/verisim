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


def test_wait_reaps_a_zombie_child_and_returns_its_exit_code() -> None:
    # The reaping half of the lifecycle: pid 1 forks pid 2, which exits; `wait` collects the status
    # and removes pid 2 from the table (so zombies do not accumulate forever).
    oracle = ReferenceHostOracle()
    state = _run(oracle, HostState.initial(), ["fork 1", "exit 2 7"])
    assert state.procs[2].state == ZOMBIE  # waiting to be reaped
    result = oracle.step(state, parse_host_action("wait 1 2"))
    assert (result.exit_code, result.stdout) == (EXIT_OK, "7")  # the reaped child's exit code
    assert 2 not in result.state.procs  # the table entry is freed
    # a killed child (exit 137) reaps the same way.
    killed = _run(oracle, HostState.initial(), ["fork 1", "kill 1 2"])
    assert oracle.step(killed, parse_host_action("wait 1 2")).stdout == "137"


def test_wait_requires_a_zombie_child_of_the_caller() -> None:
    oracle = ReferenceHostOracle()
    # waiting on a still-RUNNING child fails (non-blocking: nothing to reap yet, not a hang).
    running = _run(oracle, HostState.initial(), ["fork 1"])
    assert oracle.step(running, parse_host_action("wait 1 2")).exit_code == EXIT_ERR
    # waiting on a zombie that is not the caller's child is ECHILD (pid 3 is pid 2's child).
    other = _run(oracle, HostState.initial(), ["fork 1", "fork 2", "exit 3 0"])
    assert oracle.step(other, parse_host_action("wait 1 3")).exit_code == EXIT_ERR  # not pid 1's
    assert oracle.step(other, parse_host_action("wait 2 3")).exit_code == EXIT_OK  # the true parent
    # waiting on a nonexistent process is ECHILD.
    assert oracle.step(HostState.initial(), parse_host_action("wait 1 99")).exit_code == EXIT_ERR


def test_reaping_does_not_reuse_the_pid() -> None:
    # The monotone allocator never reuses a pid: reaping frees the table entry but not the number.
    oracle = ReferenceHostOracle()
    final = _run(oracle, HostState.initial(), ["fork 1", "exit 2 0", "wait 1 2", "fork 1"])
    assert 2 not in final.procs  # reaped
    assert 3 in final.procs  # the next fork is pid 3, not a recycled pid 2
    assert final.next_pid == 4


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


def test_dup_aliases_an_fd_onto_the_same_path() -> None:
    # `dup` duplicates an open fd to the smallest free fd, both pointing at the same path: the two
    # fds alias one file (the shared-file coupling). The new fd reads back what the original wrote.
    oracle = ReferenceHostOracle()
    state = _run(oracle, HostState.initial(), ["open 1 /f", "write 1 0 alpha"])
    result = oracle.step(state, parse_host_action("dup 1 0"))
    assert result.exit_code == EXIT_OK
    assert result.stdout == "1"  # the smallest free fd (0 is taken)
    state = result.state
    assert state.fds[(1, 0)].path == state.fds[(1, 1)].path == "/f"  # both fds alias /f
    assert oracle.step(state, parse_host_action("read 1 1")).stdout == "alpha"  # the alias reads it


def test_dup_picks_the_smallest_free_fd_after_a_gap() -> None:
    # Opening 0,1,2 then closing 1 leaves a gap; `dup` fills the smallest free slot (1) like `open`.
    oracle = ReferenceHostOracle()
    state = _run(oracle, HostState.initial(), ["open 1 /a", "open 1 /b", "open 1 /c", "close 1 1"])
    result = oracle.step(state, parse_host_action("dup 1 2"))
    assert result.stdout == "1"  # the freed slot, not fd 3
    assert result.state.fds[(1, 1)].path == "/c"  # aliases fd 2's path


def test_dup_bad_fd_and_dead_pid_are_ebadf() -> None:
    oracle = ReferenceHostOracle()
    s0 = HostState.initial()
    assert oracle.step(s0, parse_host_action("dup 1 7")).exit_code == EXIT_ERR  # never opened
    dead = _run(oracle, HostState.initial(), ["open 1 /f", "exit 1 0"])
    assert oracle.step(dead, parse_host_action("dup 1 0")).exit_code == EXIT_ERR  # pid 1 zombie


def test_mkdir_creates_a_directory_and_enables_nested_writes() -> None:
    # `mkdir` adds directory structure to the embedded fs: before it a write into /d/ fails (no
    # parent dir); after it the nested write lands — the composition that makes `chdir` worthwhile.
    oracle = ReferenceHostOracle()
    s0 = HostState.initial()
    # without the directory, opening then writing a nested path fails (the parent /d is not a dir)
    blocked = _run(oracle, s0, ["open 1 /d/f"])
    assert oracle.step(blocked, parse_host_action("write 1 0 hi")).exit_code == EXIT_ERR
    # mkdir creates /d; the same nested write now succeeds and reads back through the fd
    s = _run(oracle, s0, ["mkdir 1 /d", "open 1 /d/f", "write 1 0 hi"])
    assert "/d" in s.fs.fs  # the directory node exists in the embedded v0 filesystem
    assert oracle.step(s, parse_host_action("read 1 0")).stdout == "hi"


def test_mkdir_existing_path_and_bad_parent_fail() -> None:
    oracle = ReferenceHostOracle()
    made = _run(oracle, HostState.initial(), ["mkdir 1 /d"])
    assert oracle.step(made, parse_host_action("mkdir 1 /d")).exit_code == EXIT_ERR  # EEXIST
    assert oracle.step(made, parse_host_action("mkdir 1 /x/y")).exit_code == EXIT_ERR  # ENOENT


def test_mkdir_on_a_dead_pid_fails() -> None:
    oracle = ReferenceHostOracle()
    dead = _run(oracle, HostState.initial(), ["exit 1 0"])
    assert oracle.step(dead, parse_host_action("mkdir 1 /d")).exit_code == EXIT_ERR  # pid 1 zombie


def test_chdir_makes_relative_paths_resolve_against_the_cwd() -> None:
    # `chdir` moves a process's cwd so a subsequent *relative* open/mkdir resolves against it: after
    # `chdir /d` an `open f` opens `/d/f`, not `/f` — the navigation `mkdir` made meaningful.
    oracle = ReferenceHostOracle()
    s = _run(oracle, HostState.initial(), ["mkdir 1 /d", "chdir 1 /d"])
    assert s.procs[1].cwd == "/d"
    # a relative open now lands in /d; write+read round-trips through the relative path
    s = _run(oracle, s, ["open 1 f", "write 1 0 hi"])
    assert "/d/f" in s.fs.fs  # the relative `open f` resolved against cwd /d
    assert oracle.step(s, parse_host_action("read 1 0")).stdout == "hi"
    # chdir is relative too: `chdir e` from /d (after making /d/e) lands at /d/e, not /e
    s = _run(oracle, s, ["mkdir 1 e", "chdir 1 e"])
    assert s.procs[1].cwd == "/d/e"


def test_chdir_to_a_file_or_missing_path_fails() -> None:
    oracle = ReferenceHostOracle()
    made = _run(oracle, HostState.initial(), ["open 1 /f", "write 1 0 x"])  # /f is a file
    assert oracle.step(made, parse_host_action("chdir 1 /f")).exit_code == EXIT_ERR  # ENOTDIR
    assert oracle.step(made, parse_host_action("chdir 1 /nope")).exit_code == EXIT_ERR  # ENOENT
    assert oracle.step(made, parse_host_action("chdir 1 /")).exit_code == EXIT_OK  # root always ok


def test_fork_child_inherits_parent_cwd() -> None:
    # Standard fork semantics: a child starts in the parent's cwd (computed in `apply`, so the
    # ProcSpawn edit — and the learned model's token for it — is unchanged).
    oracle = ReferenceHostOracle()
    s = _run(oracle, HostState.initial(), ["mkdir 1 /d", "chdir 1 /d", "fork 1"])
    assert s.procs[2].cwd == "/d"  # the child inherits /d
    # the child's later chdir does not move the parent (per-process cwd)
    s = _run(oracle, s, ["mkdir 2 e", "chdir 2 e"])
    assert s.procs[2].cwd == "/d/e" and s.procs[1].cwd == "/d"


def test_chdir_on_a_dead_pid_fails() -> None:
    oracle = ReferenceHostOracle()
    dead = _run(oracle, HostState.initial(), ["exit 1 0"])
    assert oracle.step(dead, parse_host_action("chdir 1 /")).exit_code == EXIT_ERR  # pid 1 zombie


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
