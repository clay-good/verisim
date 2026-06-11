"""The host world's bundle state (SPEC-6 §3.1, HC0 increment 1).

The host state is **not** a single tree (SPEC-2) or graph (SPEC-5) but a *bundle* of coupled
subsystems sharing references: the **process table** (the new core subsystem), per-process
**file-descriptor tables**, and the **embedded v0 filesystem** (SPEC-2's
:class:`~verisim.env.state.State`, composed verbatim — SPEC-6 does not refork it). An ``FdEntry``
points *into* the FS subsystem, so the process table is the spine and the FS hangs off it through
fds — the factored/object-centric structure SPEC-6 §2.3 prescribes.

This is the HC0-increment-1 subset: processes (``fork``/``exit``), credentials (``setuid``), and a
per-process fd table over files (``open``/``write``/``close``), file effects delegated to the v0
FS sub-oracle. Each process also carries a **current working directory** (``chdir``), inherited by
a child at ``fork`` and used to resolve relative path arguments. Sockets, pipes/signals, the
scheduler, and a per-fd offset are later HC increments
(:mod:`verisim.hostoracle.reference`; ``docs/host-semantics.md``). Canonicalization (sorted maps,
v0-canonical fs) is mandatory so the divergence metric measures competence, not identifier churn
(SPEC-6 §3.1, SPEC-3 DD-1). No runtime dependencies, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from verisim.env.serialize import from_canonical, to_canonical
from verisim.env.state import State

RUNNING = "RUNNING"
ZOMBIE = "ZOMBIE"


@dataclass(frozen=True)
class Process:
    """One entry of the process table. ``exit_code`` is set iff the process is a ``ZOMBIE``."""

    pid: int
    ppid: int
    state: str = RUNNING
    uid: int = 0  # 0 == root; the privilege axis (SPEC-6 §3.2)
    exit_code: int | None = None
    cwd: str = "/"  # per-process working directory (``chdir``); a fork child inherits the parent's


@dataclass(frozen=True)
class FdEntry:
    """A file descriptor pointing into the FS subsystem: an absolute path it was opened at.

    (Pipe/Socket/Dev fd targets are later HC increments — SPEC-6 §3.1.)
    """

    path: str


@dataclass
class HostState:
    """The coupled bundle: process table + per-process fd tables + the embedded v0 filesystem.

    ``fds`` is keyed by ``(pid, fd)``. ``fs`` *is* the v0 :class:`~verisim.env.state.State`.
    ``next_pid`` is the monotonic pid allocator (canonical, so identical trajectories get identical
    pids). Mutation is by convention done through the oracle, which builds a fresh state.
    """

    procs: dict[int, Process]
    fds: dict[tuple[int, int], FdEntry] = field(default_factory=dict)
    fs: State = field(default_factory=State.empty)
    next_pid: int = 2
    last_exit: int = 0

    @staticmethod
    def initial() -> HostState:
        """The boot state: one running root process (pid 1, the init/shell), empty filesystem."""
        return HostState(procs={1: Process(pid=1, ppid=0, state=RUNNING, uid=0)})

    def copy(self) -> HostState:
        """A fresh-container copy; ``Process``/``FdEntry`` are immutable, ``fs`` is copied."""
        return HostState(
            procs=dict(self.procs),
            fds=dict(self.fds),
            fs=self.fs.copy(),
            next_pid=self.next_pid,
            last_exit=self.last_exit,
        )

    def with_last(self, exit_code: int) -> HostState:
        return replace(self, last_exit=exit_code)


def to_canonical_host(state: HostState) -> dict[str, Any]:
    """Canonical, order-stable dict for hashing/serialization (SPEC-6 §3.1).

    Process table and fd table are emitted in sorted key order; the filesystem reuses v0's canonical
    form verbatim — the composition is visible right down to the serialization.
    """
    return {
        "procs": [
            {
                "pid": p.pid, "ppid": p.ppid, "state": p.state, "uid": p.uid,
                "exit_code": p.exit_code, "cwd": p.cwd,
            }
            for p in sorted(state.procs.values(), key=lambda p: p.pid)
        ],
        "fds": [
            {"pid": pid, "fd": fd, "path": entry.path}
            for (pid, fd), entry in sorted(state.fds.items())
        ],
        "fs": to_canonical(state.fs),
        "next_pid": state.next_pid,
        "last_exit": state.last_exit,
    }


def from_canonical_host(d: dict[str, Any]) -> HostState:
    """Reconstruct a :class:`HostState` from its canonical dict — the inverse of
    :func:`to_canonical_host` (round-trippable, SPEC-6 §3.1).

    Needed by the verified-contribution protocol (SPEC-6 §16): to re-execute the oracle on a
    contributed ``(state, action)`` and check the claimed result bit-for-bit, the verifier must
    first rebuild the contributed state. The embedded filesystem reuses v0's
    :func:`~verisim.env.serialize.from_canonical` verbatim — the composition is invertible too.
    """
    procs = {
        p["pid"]: Process(
            pid=p["pid"], ppid=p["ppid"], state=p["state"], uid=p["uid"],
            exit_code=p["exit_code"], cwd=p.get("cwd", "/"),
        )
        for p in d["procs"]
    }
    fds = {(e["pid"], e["fd"]): FdEntry(path=e["path"]) for e in d["fds"]}
    return HostState(
        procs=procs,
        fds=fds,
        fs=from_canonical(d["fs"]),
        next_pid=d["next_pid"],
        last_exit=d["last_exit"],
    )
