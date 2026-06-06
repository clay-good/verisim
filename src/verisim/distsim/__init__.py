"""The LLM-callable cluster-simulator protocol (SPEC-7 §7, DS8).

Packages the distributed world as the verified "what-if" cluster a computer-use / cyber-defense
agent calls: a :class:`~verisim.distsim.simulator.DistSimulator` that both *predicts the next
cluster state* (the loop interface, reused) and *simulates a plan* (Dreamer-style imagination +
budgeted oracle verification + the task "third oracle"), with two distributed-specific readouts —
a consistency-faithful plan horizon distinct from the bit-exact one (ED5/H19 at the plan level) and
**change-safety as differential consistency-faithfulness** (the securifine pattern). Dependency-free
except the loop's ``DistModel``; the host analogue is :mod:`verisim.hostsim`.
"""

from __future__ import annotations

from .goal import (
    DistGoal,
    all_of,
    no_split_brain,
    node_down,
    node_up,
    object_converged,
    object_converged_to,
)
from .simulator import (
    DistPlanReport,
    DistSimulator,
    Plan,
    PlanRollout,
    consistency_health,
)

__all__ = [
    "DistGoal",
    "DistPlanReport",
    "DistSimulator",
    "Plan",
    "PlanRollout",
    "all_of",
    "consistency_health",
    "no_split_brain",
    "node_down",
    "node_up",
    "object_converged",
    "object_converged_to",
]
