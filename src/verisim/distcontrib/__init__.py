"""Trustless verified contribution for distributed traces (SPEC-7 §16; the distributed `contrib`).

The oracle makes a contributed cluster trace trustless *by construction*: it is accepted iff
re-running the deterministic DES reproduces it bit-for-bit — *or*, where bit-exact is intractable
(W7), iff the cheap consistency tier admits it under the declared model (the **tiered acceptance**
the host/network protocols could not need). Pure, dependency-free; the host analogue is
:mod:`verisim.contrib`.
"""

from __future__ import annotations

from verisim.distcontrib.protocol import (
    VerificationReport,
    content_address,
    transition_record,
    verify_trajectory,
    verify_transition,
)

__all__ = [
    "VerificationReport",
    "content_address",
    "transition_record",
    "verify_trajectory",
    "verify_transition",
]
