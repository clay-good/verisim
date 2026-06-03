"""Open, decentralized, oracle-verified contribution (SPEC-6 §16, HC8).

The concrete form of the program's "open, freely-available, decentralized" intent: the
deterministic oracle makes contributed host data **trustless by construction**. Where Prime
Intellect's INTELLECT-2 must *verify* untrusted rollouts heuristically (TOPLOC's
locality-sensitive hashing, SPEC-6 §2.9), verisim re-runs the oracle and checks the contributed
``(state, action, next_state)`` / ``(state, action, delta)`` **bit-for-bit**. A contribution is
accepted iff it reproduces. There is no trust to establish and no tampering to detect
probabilistically — the oracle settles it, freely and certainly.

This module ships the *verification* protocol (the part that is free and certain, §16's v1
scope); DiLoCo-style low-communication merging is deferred and gated behind the same
no-runtime-network posture (§15). No runtime dependencies, no GPU.
"""

from __future__ import annotations

from .protocol import (
    VerificationReport,
    content_address,
    verify_trajectory,
    verify_transition,
)

__all__ = [
    "VerificationReport",
    "content_address",
    "verify_trajectory",
    "verify_transition",
]
