"""Oracle-as-reward RL environment (SPEC-2 §15), dependency-free.

Wraps the v0 env + reference oracle as a ``verifiers``-spec RL environment for
training a faithful world model against a verifiable reward. Conforms to the
reset/step protocol and exposes the :func:`load_environment` discovery entrypoint;
needs no RL framework installed to construct or test.
"""

from __future__ import annotations

from .environment import Observation, Transition, WorldModelEnv, load_environment

__all__ = [
    "Observation",
    "Transition",
    "WorldModelEnv",
    "load_environment",
]
