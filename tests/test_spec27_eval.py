"""SPEC-27 step 1 -- the logged harness agrees with the trusted ``audit`` certificate.

The harness (:func:`verisim.audit.eval.run_audit_logged`) processes a proposer's stream in
emission order to record *when* each hole is found; ``audit`` materializes-then-scores. They must
measure the *same quantity* over the same deterministic sample -- same realizing count, same
in-contract silent holes -- or the discovery curve is measuring something ``audit`` is not. This is
the step-1 gate from ``plans/SPEC-27-honest-evaluation.md``: prove the refactor didn't change the
measured quantity before trusting any head-to-head built on it.

Target under audit: the RA18 resolver with the printf-format-escape hole left OPEN
(``sound_printf=False``) -- the RA24 silent-miss class, so there is a real bug to discover.
"""

from __future__ import annotations

from verisim.audit.auditor import audit
from verisim.audit.eval import run_audit_logged
from verisim.audit.monitors import ResolverMonitor
from verisim.audit.oracles import ShellPathOracle
from verisim.audit.proposers import GrammarProposer

PREFIX = "/etc"


def _monitor() -> ResolverMonitor:
    # printf hole OPEN: a string-resolvable realizing command the resolver declares CLEAR -> a
    # genuine in-contract silent miss the harness must surface.
    return ResolverMonitor(PREFIX, sound_printf=False)


def test_harness_matches_audit_on_enumerate() -> None:
    """The deterministic enumerate arm: harness counts == certificate counts."""
    oracle = ShellPathOracle(PREFIX)
    cert = audit(_monitor(), oracle, GrammarProposer(work="/home/work", mode="enumerate"))
    log = run_audit_logged(
        _monitor(), oracle, GrammarProposer(work="/home/work", mode="enumerate"),
        budget=10_000, arm="enumerate",
    )
    assert log.n_realizing == cert.n_realizing
    assert log.silent_holes == cert.silent_holes
    # the enumerate set contains the printf-format-escape form, so the open hole is found.
    assert cert.silent_holes >= 1
    assert log.first_silent_call is not None


def test_harness_blind_is_deterministic_under_seed() -> None:
    """Same seed -> identical run, so sweep seed-variance is real signal, not nondeterminism."""
    oracle = ShellPathOracle(PREFIX)
    a = run_audit_logged(_monitor(), oracle,
                         GrammarProposer(PREFIX + "/shadow", "/home/work", mode="blind", seed=7),
                         budget=400, arm="blind", seed=7)
    b = run_audit_logged(_monitor(), oracle,
                         GrammarProposer(PREFIX + "/shadow", "/home/work", mode="blind", seed=7),
                         budget=400, arm="blind", seed=7)
    assert (a.n_realizing, a.silent_holes, a.first_silent_call) == \
           (b.n_realizing, b.silent_holes, b.first_silent_call)
    assert a.discovery_curve == b.discovery_curve


def test_blind_seeds_differ() -> None:
    """Different seeds give different streams (the variance the >=20-seed CIs will quantify)."""
    oracle = ShellPathOracle(PREFIX)
    runs = [
        run_audit_logged(_monitor(), oracle,
                         GrammarProposer(PREFIX + "/shadow", "/home/work", mode="blind", seed=s),
                         budget=400, arm="blind", seed=s)
        for s in range(5)
    ]
    # at least some variation in time-to-first-bug across seeds (else "blind" is secretly fixed).
    firsts = {r.first_silent_call for r in runs}
    assert len(firsts) >= 2
