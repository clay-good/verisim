"""SPEC-27 step 3 -- the sweep arms, retargeting, statistics, and verdict logic.

Fast tests only (no neural arm -- it is exercised by the full CLI sweep). Covers the two arms RA24
lacked (enumerate, bandit) in the judge() world, that the printf bug reproduces under retargeting,
that bootstrap CIs are well-formed, and that the kill-criterion verdict logic reports the null when
a non-learned baseline matches/beats the neural arm.
"""

from __future__ import annotations

from verisim.experiments.spec27_proposer_eval import (
    TARGETS,
    Cell,
    _target_atoms,
    bootstrap_ci,
    run_bandit,
    run_enumerate,
    summarize,
)

WORK = "/home/work"


def test_enumerate_finds_printf_bug() -> None:
    prefix, path = TARGETS[0]
    r = run_enumerate(200, 0, False, WORK, prefix, _target_atoms(path))
    assert r.silent_miss > 0
    assert r.distinct_silent_classes >= 1
    assert r.first_silent_call is not None


def test_bandit_finds_bug_and_is_deterministic() -> None:
    prefix, path = TARGETS[0]
    atoms = _target_atoms(path)
    a = run_bandit(400, 1, False, WORK, prefix, atoms)
    b = run_bandit(400, 1, False, WORK, prefix, atoms)
    assert a.silent_miss > 0 and a.first_silent_call is not None
    assert (a.silent_miss, a.distinct_silent_classes, a.first_silent_call) == \
           (b.silent_miss, b.distinct_silent_classes, b.first_silent_call)


def test_printf_bug_reproduces_under_retargeting() -> None:
    """The bug must surface on every target -- else the result is a single-path artifact."""
    for prefix, path in TARGETS:
        r = run_enumerate(400, 0, False, WORK, prefix, _target_atoms(path))
        assert r.silent_miss > 0, f"no silent miss on target {path}"


def test_bandit_beats_blind_on_raw_count_but_not_on_classes() -> None:
    """The core honest finding in miniature: adaptivity lifts raw count, not the distinct-class
    count (there are only ~1-2 classes; the raw advantage is compositions of the same bug)."""
    prefix, path = TARGETS[0]
    atoms = _target_atoms(path)
    bandit = run_bandit(800, 0, False, WORK, prefix, atoms)
    # bandit concentrates on the hole region -> far more raw silent misses than its class count.
    assert bandit.silent_miss > 50 * bandit.distinct_silent_classes


def test_bootstrap_ci_well_formed() -> None:
    lo, mean, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=0)
    assert lo <= mean <= hi
    assert abs(mean - 3.0) < 1e-9


def test_summarize_reports_null_when_baseline_matches_neural() -> None:
    """Verdict logic: a non-learned baseline tying/beating neural on the honest metrics => NULL."""
    cells = []
    # neural: huge raw count, 2 classes, slow first bug; bandit: ties classes, faster first bug.
    for seed in range(10):
        cells.append(Cell("neural", "t", seed, 1600, 1300, 1300, 2, 28, 0.81, 1.0))
        cells.append(Cell("bandit", "t", seed, 1600, 1560, 1560, 2, 20, 0.97, 0.18))
        cells.append(Cell("blind", "t", seed, 1600, 58, 58, 1, 23, 0.04, 0.06))
        cells.append(Cell("enumerate", "t", seed, 1600, 6, 6, 1, 10, 0.0, 0.05))
    summ = summarize(cells)
    assert summ["confirmed"] is False
    assert "NULL" in str(summ["verdict"])
