"""Tier-A reference host oracle (SPEC-6 §5.1, HC0 increment 1).

A from-scratch deterministic model of the pinned syscall semantics: a process table, per-process fd
tables, and the **embedded v0 filesystem**, with file effects *delegated* to the v0
:class:`~verisim.oracle.reference.ReferenceOracle` (the FS sub-oracle, reused verbatim). The host
oracle owns only the process/fd/credential glue — it does not reimplement filesystem semantics. This
is the structural point of SPEC-6: **the host oracle is a composition of sub-oracles**, and its
correctness is the correctness of the parts plus the glue. Pure, deterministic, no runtime deps, no
GPU. The normative semantics are in ``docs/host-semantics.md``.
"""

from __future__ import annotations

from dataclasses import replace

from verisim.env.action import parse_action
from verisim.env.state import resolve
from verisim.host.action import HostAction
from verisim.host.state import RUNNING, ZOMBIE, FdEntry, HostState, Process
from verisim.oracle.reference import ReferenceOracle

from .base import EXIT_ERR, EXIT_OK, HostOracle, HostStepResult


class ReferenceHostOracle(HostOracle):
    """The deterministic host oracle: process/fd/cred glue over the embedded v0 FS sub-oracle."""

    def __init__(self) -> None:
        self._fs = ReferenceOracle()

    def step(self, state: HostState, action: HostAction) -> HostStepResult:
        handler = {
            "fork": self._fork, "exit": self._exit, "setuid": self._setuid,
            "open": self._open, "write": self._write, "close": self._close,
        }[action.name]
        return handler(state.copy(), action)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _running(state: HostState, pid: int) -> bool:
        proc = state.procs.get(pid)
        return proc is not None and proc.state == RUNNING

    @staticmethod
    def _fail(state: HostState) -> HostStepResult:
        """A syscall that does not apply (bad pid/fd, EPERM): unchanged state, error exit."""
        return HostStepResult(state=state.with_last(EXIT_ERR), exit_code=EXIT_ERR, stdout="")

    @staticmethod
    def _ok(state: HostState, stdout: str = "") -> HostStepResult:
        return HostStepResult(state=state.with_last(EXIT_OK), exit_code=EXIT_OK, stdout=stdout)

    # -- process table --------------------------------------------------------

    def _fork(self, state: HostState, action: HostAction) -> HostStepResult:
        if not self._running(state, action.pid):
            return self._fail(state)
        parent = state.procs[action.pid]
        new_pid = state.next_pid
        state.procs[new_pid] = Process(pid=new_pid, ppid=action.pid, state=RUNNING, uid=parent.uid)
        state.next_pid += 1
        return self._ok(state, stdout=str(new_pid))

    def _exit(self, state: HostState, action: HostAction) -> HostStepResult:
        if not self._running(state, action.pid):
            return self._fail(state)
        try:
            code = int(action.args[0])
        except ValueError:
            return self._fail(state)
        state.procs[action.pid] = replace(
            state.procs[action.pid], state=ZOMBIE, exit_code=code
        )
        # a zombie releases its file descriptors
        for key in [k for k in state.fds if k[0] == action.pid]:
            del state.fds[key]
        return self._ok(state)

    def _setuid(self, state: HostState, action: HostAction) -> HostStepResult:
        if not self._running(state, action.pid):
            return self._fail(state)
        try:
            uid = int(action.args[0])
        except ValueError:
            return self._fail(state)
        if state.procs[action.pid].uid != 0:  # only root may change credentials (EPERM otherwise)
            return self._fail(state)
        state.procs[action.pid] = replace(state.procs[action.pid], uid=uid)
        return self._ok(state)

    # -- per-process fd table over the embedded filesystem --------------------

    def _open(self, state: HostState, action: HostAction) -> HostStepResult:
        if not self._running(state, action.pid):
            return self._fail(state)
        path = resolve("/", action.args[0])  # absolute resolution (per-process cwd is a later step)
        used = {fd for (pid, fd) in state.fds if pid == action.pid}
        fd = next(i for i in range(len(used) + 1) if i not in used)  # smallest free fd
        state.fds[(action.pid, fd)] = FdEntry(path=path)
        return self._ok(state, stdout=str(fd))

    def _write(self, state: HostState, action: HostAction) -> HostStepResult:
        if not self._running(state, action.pid):
            return self._fail(state)
        try:
            fd = int(action.args[0])
        except ValueError:
            return self._fail(state)
        entry = state.fds.get((action.pid, fd))
        if entry is None:  # EBADF
            return self._fail(state)
        token = action.args[1]
        # DELEGATE the file effect to the v0 FS sub-oracle (the composition, SPEC-6 §5.1).
        fs_result = self._fs.step(state.fs, parse_action(f"write {entry.path} {token}"))
        state.fs = fs_result.state
        return HostStepResult(
            state=state.with_last(fs_result.exit_code),
            exit_code=fs_result.exit_code,
            stdout=fs_result.stdout,
        )

    def _close(self, state: HostState, action: HostAction) -> HostStepResult:
        if not self._running(state, action.pid):
            return self._fail(state)
        try:
            fd = int(action.args[0])
        except ValueError:
            return self._fail(state)
        if (action.pid, fd) not in state.fds:  # EBADF
            return self._fail(state)
        del state.fds[(action.pid, fd)]
        return self._ok(state)
