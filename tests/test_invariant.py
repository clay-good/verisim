"""Privilege-invariant second-oracle tests (SPEC-6 §2.7, §5.3; the EH6/H12 axis). Torch-free.

  - ``privilege_violations`` flags exactly the (non-root process, protected fd) pairs;
  - ``is_privilege_safe`` is the one-bit decision;
  - ``invariant_bits_to_correct`` is 0 iff predicted/true agree on the security status, else the
    symmetric difference of the unsafe-cell sets;
  - ``invariant_bits`` (the answer size) is far smaller than the full bundle's fact count.
"""

from __future__ import annotations

from verisim.host.delta import CredChange, FdOpen, apply
from verisim.host.state import HostState, Process
from verisim.hostloop.observe import full_bits
from verisim.hostoracle.invariant import (
    invariant_bits,
    invariant_bits_to_correct,
    is_privilege_safe,
    privilege_violations,
)


def _state_with(pid: int, uid: int, path: str) -> HostState:
    """A small fixture: a process ``pid`` (child of init) at ``uid`` holding fd 0 -> ``path``."""
    s = HostState.initial()
    s.procs[pid] = Process(pid=pid, ppid=1, uid=uid)
    return apply(s, [FdOpen(pid=pid, fd=0, path=path)])


def test_violation_is_non_root_holding_protected_path():
    # a non-root (uid 1000) process holding /passwd is a violation
    bad = _state_with(2, uid=1000, path="/passwd")
    assert privilege_violations(bad) == frozenset({(2, "/passwd")})
    assert not is_privilege_safe(bad)
    # root holding /passwd is fine; non-root holding a non-protected path is fine
    assert is_privilege_safe(_state_with(2, uid=0, path="/passwd"))
    assert is_privilege_safe(_state_with(2, uid=1000, path="/log"))


def test_bits_to_correct_zero_iff_security_status_agrees():
    bad = _state_with(2, uid=1000, path="/passwd")
    safe = _state_with(2, uid=0, path="/passwd")
    assert invariant_bits_to_correct(bad, bad) == 0  # agree (both unsafe)
    assert invariant_bits_to_correct(safe, safe) == 0  # agree (both safe)
    assert invariant_bits_to_correct(safe, bad) == 1  # disagree on the one cell
    # if the model predicted root but truth is the non-root escalation, the invariant catches it
    escalate = apply(safe, [CredChange(pid=2, uid=1000)])
    assert not is_privilege_safe(escalate)
    assert invariant_bits_to_correct(safe, escalate) == 1


def test_invariant_answer_is_far_cheaper_than_the_full_state():
    bad = _state_with(2, uid=1000, path="/passwd")
    # the security answer (procs × protected) is tiny; the full bundle carries every fact
    assert invariant_bits(bad) <= len(bad.procs)  # 1 protected path
    assert invariant_bits(bad) < full_bits(bad)
