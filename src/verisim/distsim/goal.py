"""Task-level goals — the "third oracle" an agent scores a distributed plan against (SPEC-7 §7).

The state oracle (§5) answers *"is the predicted cluster state faithful?"*; a **task oracle**
answers *"would the admin task have succeeded?"*. The plans an SRE/defender agent runs against a
cluster are exactly this: a predicate over the final cluster state ("object ``x`` converged to ``v``
everywhere", "node ``n1`` is back up", "no object is split-brained"). A :class:`DistGoal` is that,
named so a plan report reads cleanly. The simulator (:mod:`.simulator`) evaluates a goal on **both**
the model's predicted final state and the oracle's true final state, and their agreement is the
task-level faithfulness signal — the composition of the state oracle and the task oracle §7
specifies, the distributed analogue of the host :mod:`verisim.hostsim.goal`. Pure, dependency-free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.state import DistributedState
from verisim.distmetrics import object_consistency_view


@dataclass(frozen=True)
class DistGoal:
    """A named predicate over a final :class:`DistributedState` — the task-success check (§7)."""

    description: str
    predicate: Callable[[DistributedState], bool]

    def holds(self, state: DistributedState) -> bool:
        return self.predicate(state)


def _converged_to(state: DistributedState, obj: str, value: str) -> bool:
    view = object_consistency_view(state, obj)
    return len(view) == 1 and next(iter(view))[1] == value


def object_converged_to(obj: str, value: str) -> DistGoal:
    """Every replica of ``obj`` agrees on ``value`` (the "write propagated everywhere" check)."""
    return DistGoal(
        f"object {obj!r} converged to {value!r}", lambda s: _converged_to(s, obj, value)
    )


def object_converged(obj: str) -> DistGoal:
    """Object ``obj`` is converged (all replicas agree — not split-brained), to any value."""
    return DistGoal(
        f"object {obj!r} is converged", lambda s: len(object_consistency_view(s, obj)) == 1
    )


def no_split_brain(config: DistConfig = DEFAULT_DIST_CONFIG) -> DistGoal:
    """No object is split-brained — every object's replicas agree (the cluster is consistent)."""
    return DistGoal(
        "no object is split-brained",
        lambda s: all(len(object_consistency_view(s, o)) == 1 for o in config.objects),
    )


def node_up(node: str) -> DistGoal:
    """Node ``node`` is up (not crashed) — the "the failover/restart succeeded" check."""
    return DistGoal(f"node {node!r} is up", lambda s: s.is_up(node))


def node_down(node: str) -> DistGoal:
    """Node ``node`` is down (crashed) — the "the drain/fence succeeded" check."""
    return DistGoal(f"node {node!r} is down", lambda s: node in s.down)


def all_of(*goals: DistGoal) -> DistGoal:
    """Conjunction: every sub-goal must hold (a multi-condition admin task)."""
    desc = " AND ".join(g.description for g in goals) if goals else "(trivially true)"
    return DistGoal(desc, lambda s: all(g.holds(s) for g in goals))
