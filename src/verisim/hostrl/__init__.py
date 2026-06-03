"""Oracle-as-reward RL environment for the host world (SPEC-6 / HC8), dependency-free.

Wraps the host env + reference oracle as a ``verifiers``-spec RL environment (the v0 §15 shape) for
training a faithful host world model against a verifiable composed-faithfulness reward. Conforms to
the reset/step protocol and exposes the :func:`load_environment` discovery entrypoint; needs no RL
framework installed to construct or test.
"""

from __future__ import annotations

from .environment import HostWorldModelEnv, Observation, Transition, load_environment

__all__ = [
    "HostWorldModelEnv",
    "Observation",
    "Transition",
    "load_environment",
]
