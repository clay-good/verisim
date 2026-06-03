"""Task-level goals -- the "third oracle" an agent scores a plan against (SPEC-6 §7).

The state oracle (§5) answers *"is the predicted host state faithful?"*; a **task oracle** answers
*"would the task have succeeded?"*. OSWorld-style change-safety / incident-response tasks (§2.6) are
exactly this: a predicate over the final host state ("the rogue process is dead", "``/passwd`` is
unchanged", "the service runs as a non-root uid"). A :class:`Goal` is that predicate, named so a
plan report reads cleanly. The simulator (:mod:`.simulator`) evaluates a goal on **both** the
model's predicted final state and the oracle's true final state, and their agreement is the
task-level faithfulness signal the agent (and a verifiable-reward trainer, §2.7) can score on -- the
composition of the state oracle and the task oracle §7 specifies. Pure and dependency-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from verisim.env.state import File
from verisim.host.state import RUNNING, ZOMBIE, HostState


@dataclass(frozen=True)
class Goal:
    """A named predicate over a final :class:`HostState` -- the task-success check (§7)."""

    description: str
    predicate: Callable[[HostState], bool]

    def holds(self, state: HostState) -> bool:
        return self.predicate(state)


def file_content(path: str, content: str) -> Goal:
    """The embedded fs has a file at ``path`` whose content is exactly ``content``."""

    def pred(s: HostState) -> bool:
        node = s.fs.fs.get(path)
        return isinstance(node, File) and node.content == content

    return Goal(f"file {path!r} has content {content!r}", pred)


def file_absent(path: str) -> Goal:
    """No file exists at ``path`` in the embedded fs (the incident-response "scrubbed" check)."""
    return Goal(f"no file at {path!r}", lambda s: not isinstance(s.fs.fs.get(path), File))


def proc_running(pid: int) -> Goal:
    """Process ``pid`` exists and is ``RUNNING``."""

    def pred(s: HostState) -> bool:
        p = s.procs.get(pid)
        return p is not None and p.state == RUNNING

    return Goal(f"pid {pid} is running", pred)


def proc_killed(pid: int) -> Goal:
    """Process ``pid`` is a ``ZOMBIE`` (the "rogue process is dead" check)."""

    def pred(s: HostState) -> bool:
        p = s.procs.get(pid)
        return p is not None and p.state == ZOMBIE

    return Goal(f"pid {pid} is killed (zombie)", pred)


def proc_uid(pid: int, uid: int) -> Goal:
    """Process ``pid`` runs as ``uid`` (the privilege check -- e.g. dropped to non-root)."""

    def pred(s: HostState) -> bool:
        p = s.procs.get(pid)
        return p is not None and p.uid == uid

    return Goal(f"pid {pid} runs as uid {uid}", pred)


def all_of(*goals: Goal) -> Goal:
    """Conjunction: every sub-goal must hold (a multi-condition task)."""
    desc = " AND ".join(g.description for g in goals) if goals else "(trivially true)"
    return Goal(desc, lambda s: all(g.holds(s) for g in goals))
