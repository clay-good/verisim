"""Privilege-faithfulness -- the security-relevant metric (SPEC-6 §9.4, HC3).

A defender's trust in the host simulator hinges on it getting *failures* right, not just successes:
does the model correctly predict that a ``setuid`` by a non-root process is **denied** (EPERM), that
a write to a closed fd is **EBADF**, that an operation a process lacks privilege for *fails*? This
metric grades exactly that, first-class (SPEC-6 §9.4, §3.2).

Operationally it is the agreement rate on the **denied/allowed** outcome over a set of
privilege-relevant transitions: each is a ``(predicted_exit_code, true_exit_code)`` pair, and the
model is faithful on that transition iff it predicts failure exactly when truth is failure
(``EXIT_OK`` ↔ success). ``1.0`` over no privilege-relevant transitions. Pure, dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from verisim.hostoracle.base import EXIT_OK


def _denied(exit_code: int) -> bool:
    """A non-``EXIT_OK`` code is a denial/failure (EPERM, EBADF, ...)."""
    return exit_code != EXIT_OK


def privilege_faithfulness(
    predicted_codes: Sequence[int], true_codes: Sequence[int]
) -> float:
    """Fraction of privilege-relevant transitions whose denied/allowed outcome agrees (§9.4).

    ``predicted_codes`` and ``true_codes`` must be aligned (same length, same transitions). ``1.0``
    iff the model predicts failure exactly when the oracle does; ``1.0`` over no transitions.
    """
    if len(predicted_codes) != len(true_codes):
        raise ValueError("predicted_codes and true_codes must be the same length")
    if not true_codes:
        return 1.0
    agree = sum(
        1 for p, t in zip(predicted_codes, true_codes, strict=True) if _denied(p) == _denied(t)
    )
    return agree / len(true_codes)


__all__ = ["privilege_faithfulness"]
