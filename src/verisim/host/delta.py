"""The host bundle delta + compositional ``apply`` (SPEC-6 §4, HC1).

``M_θ`` predicts a structured **bundle delta**, not a full state (§2.1, the v0 E4 result). The delta
vocabulary composes the process/fd/credential edits with the **embedded v0 filesystem delta**
verbatim: an ``FsDelta`` wraps a v0 :data:`~verisim.delta.edits.Delta` and is applied by the v0
:func:`~verisim.delta.apply.apply`. This is SPEC-6's compositional structure -- a bundle delta
applies to each subsystem independently, and the embedded subsystem reuses its own ``apply``.

The **M1-analogue invariant** keeps the loop model-agnostic (tested in
:mod:`tests.test_host_delta`): ``apply(state, oracle.step(s, a).delta) == oracle.step(s, a).state``
for every transition, by construction. Delta-to-serialization round-trips. No runtime deps, no GPU.

Scope (HC1, matching HC0 increment 1): process/fd/credential/cwd edits + the embedded FS delta.
Socket/IPC/scheduler edits arrive with their syscalls in later HC increments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verisim.delta import Delta as FsEdits
from verisim.delta import apply as fs_apply
from verisim.delta import delta_from_list as fs_delta_from_list
from verisim.delta import delta_to_list as fs_delta_to_list

from .state import RUNNING, ZOMBIE, FdEntry, HostState, Process


@dataclass(frozen=True)
class ProcSpawn:
    """Create a child process (``fork``)."""

    pid: int
    ppid: int
    uid: int


@dataclass(frozen=True)
class ProcExit:
    """A process becomes a ``ZOMBIE``; ``apply`` releases its file descriptors (``exit``)."""

    pid: int
    code: int


@dataclass(frozen=True)
class ProcReap:
    """Remove a ``ZOMBIE`` from the process table (``wait``).

    The reaping half of the process lifecycle: a parent collects its dead child's exit status and
    frees the table entry, so zombies do not accumulate forever. ``apply`` removes the pid from
    ``procs`` (its fds were already released when it became a zombie). Pids are not reused (the
    monotone allocator), so reaping frees the table slot but never the pid number.
    """

    pid: int


@dataclass(frozen=True)
class FdOpen:
    """Bind a file descriptor to a path (``open``)."""

    pid: int
    fd: int
    path: str


@dataclass(frozen=True)
class FdClose:
    """Release a file descriptor (``close``, or implied by ``ProcExit``)."""

    pid: int
    fd: int


@dataclass(frozen=True)
class CredChange:
    """Change a process's uid (``setuid``)."""

    pid: int
    uid: int


@dataclass(frozen=True)
class CwdChange:
    """Change a process's current working directory (``chdir``).

    Parallel to :class:`CredChange` (the other per-process scalar): it rewrites one field of one
    ``Process`` and touches nothing else. The cwd resolves a process's relative path arguments, so
    after ``chdir /d`` an ``open f`` opens ``/d/f``. A ``fork`` child inherits its parent's cwd in
    :func:`apply` (no field on :class:`ProcSpawn`), so the learned-model delta vocabulary is
    unchanged -- the ``mkdir``/``dup`` "reuse, no new model token" pattern.
    """

    pid: int
    cwd: str


@dataclass(frozen=True)
class FsDelta:
    """The embedded v0 filesystem delta, applied verbatim by the v0 ``apply`` (the composition)."""

    edits: FsEdits


@dataclass(frozen=True)
class SetExit:
    """Record the host-level exit code of the syscall (the observation)."""

    exit_code: int


HostEdit = (
    ProcSpawn | ProcExit | ProcReap | FdOpen | FdClose | CredChange | CwdChange | FsDelta | SetExit
)
HostDelta = list[HostEdit]


def apply(state: HostState, delta: HostDelta) -> HostState:
    """Return a fresh :class:`HostState` with ``delta`` applied; does not mutate ``state``.

    Compositional: each edit touches one subsystem, and ``FsDelta`` re-applies the v0 delta through
    the v0 ``apply``. This is the function the ``apply == oracle`` invariant pins (HC1).
    """
    procs = dict(state.procs)
    fds = dict(state.fds)
    fs = state.fs
    next_pid = state.next_pid
    last_exit = state.last_exit
    for edit in delta:
        if isinstance(edit, ProcSpawn):
            # A child inherits the parent's cwd (the standard fork semantics) -- read here from the
            # parent already in ``procs`` rather than carried on ``ProcSpawn``, so the spawn edit
            # (and the learned model's token for it) stays unchanged. ppid 0 (init) has no parent.
            parent = procs.get(edit.ppid)
            procs[edit.pid] = Process(
                pid=edit.pid, ppid=edit.ppid, state=RUNNING, uid=edit.uid,
                cwd=parent.cwd if parent is not None else "/",
            )
            next_pid = max(next_pid, edit.pid + 1)
        elif isinstance(edit, ProcExit):
            if edit.pid in procs:
                procs[edit.pid] = Process(
                    pid=edit.pid, ppid=procs[edit.pid].ppid, state=ZOMBIE,
                    uid=procs[edit.pid].uid, exit_code=edit.code, cwd=procs[edit.pid].cwd,
                )
                fds = {k: v for k, v in fds.items() if k[0] != edit.pid}
        elif isinstance(edit, ProcReap):
            procs.pop(edit.pid, None)  # remove the reaped zombie from the table (fds already freed)
        elif isinstance(edit, FdOpen):
            fds[(edit.pid, edit.fd)] = FdEntry(path=edit.path)
        elif isinstance(edit, FdClose):
            fds.pop((edit.pid, edit.fd), None)
        elif isinstance(edit, CredChange):
            if edit.pid in procs:
                p = procs[edit.pid]
                procs[edit.pid] = Process(p.pid, p.ppid, p.state, edit.uid, p.exit_code, p.cwd)
        elif isinstance(edit, CwdChange):
            if edit.pid in procs:
                p = procs[edit.pid]
                procs[edit.pid] = Process(p.pid, p.ppid, p.state, p.uid, p.exit_code, edit.cwd)
        elif isinstance(edit, FsDelta):
            fs = fs_apply(fs, edit.edits)
        elif isinstance(edit, SetExit):
            last_exit = edit.exit_code
    return HostState(procs=procs, fds=fds, fs=fs, next_pid=next_pid, last_exit=last_exit)


# --- serialization (delta <-> list of dicts; the embedded FS delta reuses v0's) ----------


def edit_to_dict(edit: HostEdit) -> dict[str, Any]:
    """Serialize one host edit to a JSON-able dict."""
    if isinstance(edit, ProcSpawn):
        return {"op": "ProcSpawn", "pid": edit.pid, "ppid": edit.ppid, "uid": edit.uid}
    if isinstance(edit, ProcExit):
        return {"op": "ProcExit", "pid": edit.pid, "code": edit.code}
    if isinstance(edit, ProcReap):
        return {"op": "ProcReap", "pid": edit.pid}
    if isinstance(edit, FdOpen):
        return {"op": "FdOpen", "pid": edit.pid, "fd": edit.fd, "path": edit.path}
    if isinstance(edit, FdClose):
        return {"op": "FdClose", "pid": edit.pid, "fd": edit.fd}
    if isinstance(edit, CredChange):
        return {"op": "CredChange", "pid": edit.pid, "uid": edit.uid}
    if isinstance(edit, CwdChange):
        return {"op": "CwdChange", "pid": edit.pid, "cwd": edit.cwd}
    if isinstance(edit, FsDelta):
        return {"op": "FsDelta", "edits": fs_delta_to_list(edit.edits)}
    return {"op": "SetExit", "exit_code": edit.exit_code}


def edit_from_dict(d: dict[str, Any]) -> HostEdit:
    """Inverse of :func:`edit_to_dict`."""
    op = d["op"]
    if op == "ProcSpawn":
        return ProcSpawn(pid=d["pid"], ppid=d["ppid"], uid=d["uid"])
    if op == "ProcExit":
        return ProcExit(pid=d["pid"], code=d["code"])
    if op == "ProcReap":
        return ProcReap(pid=d["pid"])
    if op == "FdOpen":
        return FdOpen(pid=d["pid"], fd=d["fd"], path=d["path"])
    if op == "FdClose":
        return FdClose(pid=d["pid"], fd=d["fd"])
    if op == "CredChange":
        return CredChange(pid=d["pid"], uid=d["uid"])
    if op == "CwdChange":
        return CwdChange(pid=d["pid"], cwd=d["cwd"])
    if op == "FsDelta":
        return FsDelta(edits=fs_delta_from_list(d["edits"]))
    if op == "SetExit":
        return SetExit(exit_code=d["exit_code"])
    raise ValueError(f"unknown host edit op {op!r}")


def delta_to_list(delta: HostDelta) -> list[dict[str, Any]]:
    """Serialize a host bundle delta to a JSON-able list (inverse of :func:`delta_from_list`)."""
    return [edit_to_dict(e) for e in delta]


def delta_from_list(items: list[dict[str, Any]]) -> HostDelta:
    """Inverse of :func:`delta_to_list`."""
    return [edit_from_dict(d) for d in items]
