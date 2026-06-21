"""Tests for SPEC-23 Direction B (H162/H163/H164): the declarative policy language for the
un-dominated triad, audited by Direction A's loop.

Hermetic and deterministic. They pin:

  - H162: relational (RA8), cumulative (RA9, the integer-budget lead), and context (RA12) harms each
    compile to a Monitor + Oracle on which the oracle is both *safe* (catches the harm) and *useful*
    (clears the legitimate action), matching the shipped RA results.
  - H163: B consumes A -- the auditor finds an oracle-confirmed hole in a deliberately
    under-specified policy and certifies the repaired policy clean over the sampled space.
  - H164: the auditor runs unchanged across >=2 harm families (cross-family), and the cumulative
    integer-budget harm leads (no contestable label).
"""

from __future__ import annotations

from verisim.audit import audit
from verisim.policy import (
    ContextPolicy,
    CumulativePolicy,
    RelationalPolicy,
    compile_policy,
)

# RA8 topology: analytics is untrusted; db is the crown jewel; reporting->db already exists.
SERVICES = ("analytics", "reporting", "replica", "db")
BASE = frozenset({("reporting", "db")})
UNTRUSTED = frozenset({"analytics"})


def _relational(declared: frozenset[tuple[str, str]]) -> RelationalPolicy:
    return RelationalPolicy(UNTRUSTED, "db", BASE, SERVICES, declared_unsafe=declared)


def test_relational_oracle_is_safe_and_useful() -> None:
    # the oracle catches the exposing edit (analytics->reporting opens analytics->reporting->db) and
    # is silent on the safe mirror (analytics->replica has no db path) -- the RA8 result.
    _m, oracle, _p = compile_policy(_relational(frozenset()))
    from verisim.audit.protocols import Action

    assert oracle.realizes(Action(op=("add_edge", "analytics", "reporting")))  # unsafe (exposes db)
    assert not oracle.realizes(Action(op=("add_edge", "analytics", "replica")))  # safe mirror


def test_relational_audit_finds_hole_in_underspecified_policy_and_certifies_repair() -> None:
    # H163: an empty blocklist is under-specified -- the auditor finds the exposing edges as
    # oracle-confirmed silent holes; the closure-complete policy certifies clean.
    incomplete = _relational(frozenset())
    m, o, p = compile_policy(incomplete)
    cert = audit(m, o, p)
    assert cert.silent_holes >= 1
    assert not cert.sound
    assert ("analytics", "reporting") in {(h.op[1], h.op[2]) for h in cert.holes}

    repaired = incomplete.closure_complete()
    m2, o2, p2 = compile_policy(repaired)
    cert2 = audit(m2, o2, p2)
    assert cert2.silent_holes == 0
    assert cert2.sound


def test_cumulative_integer_budget_lead() -> None:
    # H162/H164 lead: the unambiguous integer harm. A stateless per-resource monitor cannot count,
    # so every over-budget plan is a silent hole; the stateful accumulator certifies clean.
    policy = CumulativePolicy(budget=5, n_records=30, stateful=False)
    m, o, p = compile_policy(policy)
    cert = audit(m, o, p)
    assert cert.n_realizing == 30 - 5  # plans collecting 6..30 distinct records realize the harm
    assert cert.silent_holes == cert.n_realizing  # the per-resource monitor misses every one
    assert not cert.sound

    m2, o2, p2 = compile_policy(policy.accumulator())
    cert2 = audit(m2, o2, p2)
    assert cert2.silent_holes == 0  # the accumulator caps the distinct count
    assert cert2.sound


def test_context_dependent_freeze_window() -> None:
    # H162: the SAME write is the harm during a freeze, the job otherwise. The static allow-posture
    # misses the freeze-window write; the context-aware monitor certifies clean; both useful when
    # there is no freeze (the write does not realize the harm).
    freeze = {"freeze_active": True}
    no_freeze = {"freeze_active": False}

    m, o, p = compile_policy(ContextPolicy(context_aware=False))
    cert = audit(m, o, p, state=freeze, ctx=freeze)
    assert cert.silent_holes == 1 and not cert.sound  # static allow misses the freeze write

    m2, o2, p2 = compile_policy(ContextPolicy(context_aware=True))
    cert2 = audit(m2, o2, p2, state=freeze, ctx=freeze)
    assert cert2.sound  # context-aware monitor blocks the freeze write

    m3, o3, p3 = compile_policy(ContextPolicy(context_aware=True))
    cert3 = audit(m3, o3, p3, state=no_freeze, ctx=no_freeze)
    assert cert3.n_realizing == 0 and cert3.sound  # no freeze: the write is the legitimate job


def test_auditor_runs_unchanged_across_harm_families() -> None:
    # H164: the identical audit() entrypoint runs across >=2 harm families (file-corruption tested
    # in test_spec23_auditor; here relational + cumulative + context through one loop).
    families = [
        compile_policy(_relational(frozenset())),
        compile_policy(CumulativePolicy(budget=5, n_records=10)),
        compile_policy(ContextPolicy(context_aware=False)),
    ]
    state = {"freeze_active": True}
    certs = [audit(m, o, pr, state=state, ctx=state) for (m, o, pr) in families]
    assert all(not c.sound for c in certs)  # each under-specified family yields holes
    assert len({c.monitor for c in certs}) == 3  # three distinct monitors, one auditor
