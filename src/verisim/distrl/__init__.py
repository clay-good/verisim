"""Distributed oracle-as-reward RL environment (SPEC-7 §12 / DS8).

The ``verifiers``-spec env whose reward is the distributed tiered oracle's faithfulness verdict --
the distributed analogue of :mod:`verisim.hostrl`. See :mod:`verisim.distrl.environment`.
"""

from verisim.distrl.environment import (
    REWARD_MODES,
    DistWorldModelEnv,
    Observation,
    Transition,
    load_environment,
)

__all__ = [
    "REWARD_MODES",
    "DistWorldModelEnv",
    "Observation",
    "Transition",
    "load_environment",
]
