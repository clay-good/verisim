"""The LLM-callable whole-machine simulator protocol (SPEC-6 §7, HC8).

Packages the host world as the verified "what-if" machine a computer-use / cyber-defense agent
calls: a :class:`~verisim.hostsim.simulator.HostSimulator` that both *predicts the next host state*
(the loop interface, reused) and *simulates a plan* (Dreamer-style imagination + budgeted oracle
verification + the task-level "third oracle"). Dependency-free except the loop's ``HostModel``.
"""

from __future__ import annotations

from .goal import (
    Goal,
    all_of,
    file_absent,
    file_content,
    proc_killed,
    proc_running,
    proc_uid,
)
from .simulator import HostSimulator, Plan, PlanReport, PlanRollout

__all__ = [
    "Goal",
    "HostSimulator",
    "Plan",
    "PlanReport",
    "PlanRollout",
    "all_of",
    "file_absent",
    "file_content",
    "proc_killed",
    "proc_running",
    "proc_uid",
]
