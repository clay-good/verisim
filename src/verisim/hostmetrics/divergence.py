"""Composed host divergence + the per-subsystem decomposition (SPEC-6 §9.1, HC3).

The host state is a *bundle* of coupled subsystems (process table + fd table + the embedded v0
filesystem), so its divergence is the v0 FS set-difference **summed with** the new process/fd facts,
canonicalized (SPEC-6 §9.1). Two views are reported, because the per-subsystem breakdown is what the
composition law (H13, :mod:`verisim.hostmetrics.composition`) needs and what a single scalar hides
(§5.4):

- :func:`divergence` -- the **composed** scalar over the union of all subsystem facts, ``0.0`` iff
  every subsystem agrees, ``∈ [0, 1]``. The v0/SPEC-5 formula, now over the bundle.
- :func:`divergence_by_subsystem` -- the divergence *within* each subsystem (``proc``/``fd``/``fs``/
  ``global``), each normalized independently. Note the composed scalar is **not** the mean of these
  (the denominators differ); both views are first-class (§9.1).

The embedded filesystem reuses v0's :func:`~verisim.metrics.divergence.state_facts` verbatim -- the
composition is visible right down to the metric. Pure and dependency-free, like every metric core in
the program.
"""

from __future__ import annotations

from collections.abc import Mapping

from verisim.host.state import HostState
from verisim.metrics.divergence import state_facts as fs_state_facts

Fact = tuple[object, ...]

#: The host subsystems, in canonical order (SPEC-6 §3.1, §5.4). ``proc`` carries credentials (the
#: privilege axis is part of process state); ``global`` carries the last syscall's exit code.
SUBSYSTEMS: tuple[str, ...] = ("proc", "fd", "fs", "global")


def facts_by_subsystem(state: HostState) -> dict[str, set[Fact]]:
    """The distinguishable facts of ``state``, partitioned by subsystem (SPEC-6 §9.1).

    ``next_pid`` is deliberately omitted: it is a derived monotonic allocator, fully determined by
    the process facts, so including it would double-penalize a mispredicted ``fork``.
    """
    proc: set[Fact] = {
        ("proc", p.pid, p.ppid, p.state, p.uid, p.exit_code, p.cwd) for p in state.procs.values()
    }
    fd: set[Fact] = {("fd", pid, fd, entry.path) for (pid, fd), entry in state.fds.items()}
    fs: set[Fact] = {("fs", *fact) for fact in fs_state_facts(state.fs)}
    glob: set[Fact] = {("global", "last_exit", state.last_exit)}
    return {"proc": proc, "fd": fd, "fs": fs, "global": glob}


def host_facts(state: HostState) -> set[Fact]:
    """The flat union of all subsystem facts -- the set the composed divergence is computed over."""
    out: set[Fact] = set()
    for facts in facts_by_subsystem(state).values():
        out |= facts
    return out


def _normalized_symdiff(fa: set[Fact], fb: set[Fact]) -> float:
    denom = len(fa) + len(fb)
    return len(fa ^ fb) / denom if denom else 0.0


def divergence(a: HostState, b: HostState) -> float:
    """Composed normalized symmetric difference over the union of all subsystem facts.

    ``0.0`` iff every subsystem of ``a`` and ``b`` agrees; ``∈ [0, 1]`` otherwise (SPEC-6 §9.1).
    """
    return _normalized_symdiff(host_facts(a), host_facts(b))


def divergence_by_subsystem(a: HostState, b: HostState) -> dict[str, float]:
    """Per-subsystem divergence -- each subsystem's facts normalized independently (§5.4, §9.1)."""
    fa = facts_by_subsystem(a)
    fb = facts_by_subsystem(b)
    return {sub: _normalized_symdiff(fa[sub], fb[sub]) for sub in SUBSYSTEMS}


def step_faithful_by_subsystem(
    true: HostState, predicted: HostState, epsilon: float
) -> dict[str, bool]:
    """Per-subsystem faithfulness at tolerance ``ε`` -- ``divergence_i ≤ ε`` for each subsystem.

    The per-step input the composition-law diagnostic (H13, :mod:`.composition`) consumes: a step is
    composed-faithful iff *every* subsystem is faithful (SPEC-6 §9.2).
    """
    return {sub: d <= epsilon for sub, d in divergence_by_subsystem(true, predicted).items()}


def composed_faithful(per_subsystem: Mapping[str, bool]) -> bool:
    """A step is composed-faithful iff every subsystem is faithful (the §9.2 conjunction)."""
    return all(per_subsystem.values())
