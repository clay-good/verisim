"""Tier-A reference host oracle (SPEC-6 §5.1, HC0-HC1).

A from-scratch deterministic model of the pinned syscall semantics: a process table, per-process fd
tables, and the **embedded v0 filesystem**, with file effects *delegated* to the v0
:class:`~verisim.oracle.reference.ReferenceOracle` (the FS sub-oracle, reused verbatim). The host
oracle owns only the process/fd/credential glue — it does not reimplement filesystem semantics. This
is the structural point of SPEC-6: **the host oracle is a composition of sub-oracles**, and its
correctness is the correctness of the parts plus the glue. Pure, deterministic, no runtime deps, no
GPU. The normative semantics are in ``docs/host-semantics.md``.

Each syscall computes a **bundle delta** (HC1); the oracle returns it alongside the next state, with
``apply(state, delta) == next_state`` by construction (the M1-analogue invariant). A file syscall's
delta wraps the v0 FS sub-oracle's own delta in an :class:`~verisim.host.delta.FsDelta`.
"""

from __future__ import annotations

from verisim.env.action import parse_action
from verisim.env.state import resolve
from verisim.host.action import HostAction
from verisim.host.delta import (
    CredChange,
    FdClose,
    FdOpen,
    FsDelta,
    HostDelta,
    ProcExit,
    ProcReap,
    ProcSpawn,
    SetExit,
    apply,
)
from verisim.host.state import RUNNING, ZOMBIE, HostState
from verisim.oracle.reference import ReferenceOracle

from .base import EXIT_ERR, EXIT_OK, HostOracle, HostStepResult

# A handler computes (bundle delta, exit code, stdout) from the current state + action.
_Outcome = tuple[HostDelta, int, str]


class ReferenceHostOracle(HostOracle):
    """The deterministic host oracle: process/fd/cred glue over the embedded v0 FS sub-oracle."""

    def __init__(self) -> None:
        self._fs = ReferenceOracle()

    def step(self, state: HostState, action: HostAction) -> HostStepResult:
        handler = {
            "fork": self._fork, "exit": self._exit, "kill": self._kill, "wait": self._wait,
            "setuid": self._setuid,
            "open": self._open, "write": self._write, "read": self._read, "close": self._close,
            "dup": self._dup, "mkdir": self._mkdir,
        }[action.name]
        delta, exit_code, stdout = handler(state, action)
        return HostStepResult(
            state=apply(state, delta), delta=delta, exit_code=exit_code, stdout=stdout
        )

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _running(state: HostState, pid: int) -> bool:
        proc = state.procs.get(pid)
        return proc is not None and proc.state == RUNNING

    @staticmethod
    def _fail() -> _Outcome:
        """A syscall that does not apply (bad pid/fd, EPERM): only the exit code changes."""
        return [SetExit(EXIT_ERR)], EXIT_ERR, ""

    # -- process table --------------------------------------------------------

    def _fork(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        new_pid = state.next_pid
        uid = state.procs[action.pid].uid
        delta: HostDelta = [ProcSpawn(pid=new_pid, ppid=action.pid, uid=uid), SetExit(EXIT_OK)]
        return delta, EXIT_OK, str(new_pid)

    def _exit(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        try:
            code = int(action.args[0])
        except ValueError:
            return self._fail()
        return [ProcExit(pid=action.pid, code=code), SetExit(EXIT_OK)], EXIT_OK, ""

    def _kill(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        try:
            target = int(action.args[0])
        except ValueError:
            return self._fail()
        victim = state.procs.get(target)
        if victim is None or victim.state != RUNNING:  # ESRCH: no such running process
            return self._fail()
        # PERMISSION (the privilege axis, EPERM): a process may terminate another only if it is root
        # (uid 0) or shares the target's uid -- the standard Unix rule, the inter-process echo of
        # ``setuid``'s root gate. A non-root process cannot kill another user's process.
        killer_uid = state.procs[action.pid].uid
        if killer_uid != 0 and killer_uid != victim.uid:
            return self._fail()
        # The target becomes a ZOMBIE and its fds are released -- reusing ``ProcExit`` (so no new
        # delta type), with the SIGKILL convention for the exit status (128 + signal 9 = 137).
        return [ProcExit(pid=target, code=137), SetExit(EXIT_OK)], EXIT_OK, ""

    def _wait(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        try:
            child = int(action.args[0])
        except ValueError:
            return self._fail()
        victim = state.procs.get(child)
        # The reaping half of the lifecycle: a parent collects a dead child's exit status and frees
        # the table entry. Non-blocking and parent-only: ``child`` must exist, be a ZOMBIE, and have
        # ``ppid == pid`` (you reap only your own dead children) -- else ECHILD. Waiting on a
        # running child is *not* blocking here (no scheduler): it fails (nothing to reap yet).
        if victim is None or victim.state != ZOMBIE or victim.ppid != action.pid:
            return self._fail()
        # Remove the zombie (``ProcReap``) and report its exit status; pids are not reused.
        return [ProcReap(pid=child), SetExit(EXIT_OK)], EXIT_OK, str(victim.exit_code)

    def _setuid(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        try:
            uid = int(action.args[0])
        except ValueError:
            return self._fail()
        if state.procs[action.pid].uid != 0:  # only root may change credentials (EPERM otherwise)
            return self._fail()
        return [CredChange(pid=action.pid, uid=uid), SetExit(EXIT_OK)], EXIT_OK, ""

    # -- per-process fd table over the embedded filesystem --------------------

    def _open(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        path = resolve("/", action.args[0])  # absolute resolution (per-process cwd is a later step)
        used = {fd for (pid, fd) in state.fds if pid == action.pid}
        fd = next(i for i in range(len(used) + 1) if i not in used)  # smallest free fd
        return [FdOpen(pid=action.pid, fd=fd, path=path), SetExit(EXIT_OK)], EXIT_OK, str(fd)

    def _write(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        try:
            fd = int(action.args[0])
        except ValueError:
            return self._fail()
        entry = state.fds.get((action.pid, fd))
        if entry is None:  # EBADF
            return self._fail()
        token = action.args[1]
        # DELEGATE the file effect to the v0 FS sub-oracle (the composition, SPEC-6 §5.1); wrap its
        # delta in an FsDelta so the host apply reproduces the embedded fs through the v0 apply.
        fs_result = self._fs.step(state.fs, parse_action(f"write {entry.path} {token}"))
        delta: HostDelta = [FsDelta(edits=fs_result.delta), SetExit(fs_result.exit_code)]
        return delta, fs_result.exit_code, fs_result.stdout

    def _read(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        try:
            fd = int(action.args[0])
        except ValueError:
            return self._fail()
        entry = state.fds.get((action.pid, fd))
        if entry is None:  # EBADF
            return self._fail()
        # DELEGATE the read to the v0 FS sub-oracle (the composition, SPEC-6 §5.1): ``cat`` returns
        # the file's content as stdout with an empty (read-only) delta, so the only host effect
        # is the exit code. A read of an fd whose path is not a readable file (removed/a dir)
        # inherits the FS oracle's failure (EXIT_ERR, ""). No per-fd offset yet, so a read returns
        # the whole content each time. Reports the content read.
        fs_result = self._fs.step(state.fs, parse_action(f"cat {entry.path}"))
        return [SetExit(fs_result.exit_code)], fs_result.exit_code, fs_result.stdout

    def _close(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        try:
            fd = int(action.args[0])
        except ValueError:
            return self._fail()
        if (action.pid, fd) not in state.fds:  # EBADF
            return self._fail()
        return [FdClose(pid=action.pid, fd=fd), SetExit(EXIT_OK)], EXIT_OK, ""

    def _dup(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        try:
            fd = int(action.args[0])
        except ValueError:
            return self._fail()
        entry = state.fds.get((action.pid, fd))
        if entry is None:  # EBADF: cannot duplicate an fd that is not open
            return self._fail()
        # Allocate the smallest free fd (the standard ``dup`` contract, same rule as ``open``) and
        # bind it to the *source fd's path* -- the new fd aliases the same file. Reuses ``FdOpen``
        # (no new delta type, the ``kill``-reuses-``ProcExit`` pattern): two fds onto one path is
        # the shared-file coupling the factored model's edges fold onto the process spine (§6.2).
        used = {f for (pid, f) in state.fds if pid == action.pid}
        new_fd = next(i for i in range(len(used) + 1) if i not in used)
        delta: HostDelta = [FdOpen(pid=action.pid, fd=new_fd, path=entry.path), SetExit(EXIT_OK)]
        return delta, EXIT_OK, str(new_fd)

    def _mkdir(self, state: HostState, action: HostAction) -> _Outcome:
        if not self._running(state, action.pid):
            return self._fail()
        # DELEGATE directory creation to the v0 FS sub-oracle (the composition, SPEC-6 §5.1), the
        # ``write`` pattern: v0 ``mkdir`` yields a ``[Create(path, Dir())]`` we wrap in an
        # ``FsDelta`` (no new edit type). It fails (EEXIST / ENOENT) iff the path exists or its
        # parent is not a directory -- inheriting the FS oracle's verdict. Adds directory structure
        # to the host so a later ``write`` into the subdir works (the prerequisite for ``chdir``).
        path = resolve("/", action.args[0])  # absolute resolution (per-process cwd is a later step)
        fs_result = self._fs.step(state.fs, parse_action(f"mkdir {path}"))
        delta: HostDelta = [FsDelta(edits=fs_result.delta), SetExit(fs_result.exit_code)]
        return delta, fs_result.exit_code, fs_result.stdout
