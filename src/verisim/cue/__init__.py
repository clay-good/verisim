"""verisim-cue -- the verifiable computer-use environment (SPEC-21 §2).

The SPEC-6 host world (shell / filesystem / process) positioned and hardened as a *verifiable
computer-use environment*: the CLI slice of computer use that admits a ground-truth oracle (unlike
GUI/browser computer use, which has none). It exposes an ordered structure->content task suite
(:mod:`verisim.cue.tasks`) -- each a SPEC-20-style faithful-vs-free predictive-defense whose gap
is the load-bearing signal -- which the SPEC-21 scale law sweeps across model capacity to measure
*where the faithfulness-for-control frontier sits, and how it moves with scale*.
"""

from __future__ import annotations

from .tasks import (
    TASK_SUITE,
    CueTask,
    TaskGap,
    TaskGapConfig,
    alive_procs,
    file_contents,
    grounded_keyed_reward,
    keyed_defense_reward,
    open_fds,
    rollout_keyed,
    task_gap,
    task_knee_rho,
    written_files,
)

__all__ = [
    "TASK_SUITE",
    "CueTask",
    "TaskGap",
    "TaskGapConfig",
    "alive_procs",
    "file_contents",
    "grounded_keyed_reward",
    "keyed_defense_reward",
    "open_fds",
    "rollout_keyed",
    "task_gap",
    "task_knee_rho",
    "written_files",
]
