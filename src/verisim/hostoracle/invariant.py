"""A symbolic second-oracle: the privilege-safety invariant (SPEC-6 §2.7, §5.3; the EH6/H12 axis).

The state oracle (§5) answers the *full* question — "is the predicted bundle state faithful, bit for
bit?". For many security decisions an agent does not need all of that; it needs one cheap,
formally-checkable property. This module is that **symbolic second-oracle**: a privilege invariant
over the host bundle, the host analogue of SPEC-5's Batfish-style control-plane oracle (EN10/H12).

The invariant: **no non-root process holds an open file descriptor to a protected path.** A non-root
(uid != 0) process with ``/passwd`` open is a privilege-safety violation — exactly the
incident-response property a defender cares about, and a pure function of the (proc, fd) subsystems,
so it is far cheaper to *describe* than the whole state. The "answer" the oracle returns is the
**security-status matrix** (per process × protected path, does that process hold it open) -- the
analogue of the control-plane reachability matrix ``R``. Its size (``invariant_bits``) is the
consult cost; the cells the predicted state gets wrong (``invariant_bits_to_correct``) are the
second-oracle error. Pure, deterministic, dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from verisim.host.state import HostState

#: The security-sensitive paths the invariant guards (the cyber-defense framing, §2.6).
PROTECTED_PATHS: tuple[str, ...] = ("/passwd",)


def privilege_violations(
    state: HostState, protected: Sequence[str] = PROTECTED_PATHS
) -> frozenset[tuple[int, str]]:
    """The ``(pid, path)`` pairs that violate it: a non-root process holding a protected fd."""
    guarded = set(protected)
    out: set[tuple[int, str]] = set()
    for (pid, _fd), entry in state.fds.items():
        proc = state.procs.get(pid)
        if proc is not None and proc.uid != 0 and entry.path in guarded:
            out.add((pid, entry.path))
    return frozenset(out)


def is_privilege_safe(state: HostState, protected: Sequence[str] = PROTECTED_PATHS) -> bool:
    """The one-bit security decision: ``True`` iff no non-root process holds a protected fd."""
    return not privilege_violations(state, protected)


def _status_matrix(
    state: HostState, protected: Sequence[str]
) -> set[tuple[int, str]]:
    """The security-status cells that are *unsafe* in ``state`` -- the answer the oracle returns."""
    return set(privilege_violations(state, protected))


def invariant_bits(state: HostState, protected: Sequence[str] = PROTECTED_PATHS) -> int:
    """Consult cost: the size of the security-status answer (procs × protected-paths), like |R|.

    The symbolic oracle reports, per process × protected path, whether that process holds it -- a
    ``|procs| × |protected|`` matrix, far smaller than the full bundle (which carries every fd
    target, file content, and credential). That gap is the whole point (H12).
    """
    return max(1, len(state.procs) * len(protected))


def invariant_bits_to_correct(
    predicted: HostState, true: HostState, protected: Sequence[str] = PROTECTED_PATHS
) -> int:
    """Security-status cells the predicted state gets wrong -- ``0`` iff the decision agrees.

    The symmetric difference of the unsafe-cell sets: how many ``(process, path)`` verdicts the
    model's predicted state disagrees with the truth on. ``0`` iff the model gets the
    privilege-safety status exactly right (even if its full bundle prediction is wrong).
    """
    return len(_status_matrix(predicted, protected) ^ _status_matrix(true, protected))


__all__ = [
    "PROTECTED_PATHS",
    "invariant_bits",
    "invariant_bits_to_correct",
    "is_privilege_safe",
    "privilege_violations",
]
