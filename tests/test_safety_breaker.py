"""Trajectory-oscillation circuit breaker tests (OpenSpec ``add-trajectory-oscillation-breaker``).

Pin the three requirements of the safety-breaker spec delta on a real, locally-built git fixture
(no network, no external repo): OscillationMetric (oscillating trips, converging stays ok,
repetitive-edit path, determinism), DetectAndHaltBreaker (the trip fires freeze + drop and *no*
git/fs mutation), and HumanGatedRollback (recommended-not-executed without confirmation; confirmed
rollback restores the baseline, snapshots first, leaves the original source repo byte-identical).
Every test materializes its own fixture from a source repo inside the test's allowlist, so the
suite is hermetic and cross-POSIX (macOS-first, Linux CI for free).
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from verisim.fixture import Fixture, FixtureConfig, materialize, teardown, tree_hash
from verisim.safety import (
    TIER_CRITICAL,
    TIER_OK,
    BreakerConfig,
    PlanningTransition,
    RollbackNotConfirmed,
    SpeculativeRollout,
    TrajectoryBreaker,
    classify,
    compute_metric,
    evaluate,
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout


def _make_source_repo(root: Path) -> Path:
    repo = root / "sample-proj"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (repo / "README.md").write_text("# sample\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.name", "Source Author")
    _git(repo, "config", "user.email", "author@example.com")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def fixture(tmp_path: Path) -> Iterator[Fixture]:
    source = _make_source_repo(tmp_path / "src-root")
    cfg = FixtureConfig(source_roots=(tmp_path / "src-root",), scratch_root=tmp_path / "scratch")
    fx = materialize(source, cfg)
    yield fx
    teardown(fx)


def _osc_stream(states: str) -> list[PlanningTransition]:
    """A transition stream from a string of single-char states, e.g. ``"ABABA"``."""
    return [PlanningTransition(states[i], states[i + 1]) for i in range(len(states) - 1)]


# --- OscillationMetric --------------------------------------------------------------------------


def test_oscillating_stream_reaches_critical() -> None:
    metric, tier = evaluate(_osc_stream("ABABA"))
    assert metric.oscillation == pytest.approx(0.5)
    assert tier == TIER_CRITICAL


def test_converging_stream_stays_ok() -> None:
    metric, tier = evaluate(_osc_stream("ABCDE"))
    assert metric.oscillation == pytest.approx(0.0)
    assert tier == TIER_OK


def test_repetitive_file_modification_trips_critical() -> None:
    # Distinct states (oscillation stays 0) but the same file edited five times → file path trips.
    stream = [
        PlanningTransition(f"s{i}", f"s{i + 1}", files_modified=("src/main.py",)) for i in range(5)
    ]
    metric, tier = evaluate(stream)
    assert metric.oscillation == pytest.approx(0.0)
    assert metric.worst_file == "src/main.py"
    assert metric.file_repeat == 5
    assert tier == TIER_CRITICAL


def test_partial_oscillation_is_degraded() -> None:
    # A stream that is partly flip-flopping and partly progressing lands in the middle tier:
    # 7 transitions, 4 distinct bigrams → oscillation ≈ 0.43, between the two thresholds.
    stream = [
        PlanningTransition("A", "B"),
        PlanningTransition("B", "A"),
        PlanningTransition("A", "B"),
        PlanningTransition("B", "A"),
        PlanningTransition("A", "B"),
        PlanningTransition("C", "D"),
        PlanningTransition("D", "E"),
    ]
    metric, tier = evaluate(stream)
    assert 0.34 <= metric.oscillation < 0.5
    assert tier == "degraded"


def test_sliding_window_forgets_old_oscillation() -> None:
    # A long converging tail past the window should wash out an early flip-flop.
    cfg = BreakerConfig(window=4)
    stream = _osc_stream("ABAB") + _osc_stream("WXYZ")
    _, tier = evaluate(stream, cfg)
    assert tier == TIER_OK


def test_metric_is_deterministic() -> None:
    stream = _osc_stream("ABABABAB")
    a = compute_metric(stream)
    b = compute_metric(stream)
    assert a == b
    assert classify(a) == classify(b) == TIER_CRITICAL


def test_empty_stream_is_ok() -> None:
    metric, tier = evaluate([])
    assert metric.n_transitions == 0
    assert tier == TIER_OK


# --- DetectAndHaltBreaker -----------------------------------------------------------------------


def test_critical_trips_safe_automatic_actions_only(fixture: Fixture) -> None:
    rollout = SpeculativeRollout(states=["imagined-1", "imagined-2"])
    breaker = TrajectoryBreaker(fixture, rollout=rollout)
    before = tree_hash(fixture.repo_path)

    final_tier = TIER_OK
    for t in _osc_stream("ABABA"):
        final_tier = breaker.observe(t)

    assert final_tier == TIER_CRITICAL
    assert breaker.tripped and breaker.frozen
    # Reversible automatic action: the in-memory rollout is dropped.
    assert rollout.dropped and rollout.states == []
    # No irreversible action: the fixture working tree is byte-for-byte unchanged by the trip.
    assert tree_hash(fixture.repo_path) == before


def test_frozen_loop_ignores_further_transitions(fixture: Fixture) -> None:
    breaker = TrajectoryBreaker(fixture)
    for t in _osc_stream("ABABA"):
        breaker.observe(t)
    n_at_trip = len(breaker.transitions)
    breaker.observe(PlanningTransition("C", "D"))
    assert len(breaker.transitions) == n_at_trip  # halted: not recorded


# --- HumanGatedRollback -------------------------------------------------------------------------


def test_recommendation_has_target_and_diff(fixture: Fixture) -> None:
    breaker = TrajectoryBreaker(fixture)  # auto-baseline taken at construction
    baseline_hash = breaker.checkpoints[-1].tree_hash

    # Mutate the working tree, then trip.
    (fixture.repo_path / "src" / "main.py").write_text("def main():\n    return 99\n")
    for t in _osc_stream("ABABA"):
        breaker.observe(t)

    rec = breaker.recommendation
    assert rec is not None
    assert rec.target_label == "baseline"
    assert rec.target_tree_hash == baseline_hash
    assert rec.current_tree_hash != baseline_hash
    assert "M src/main.py" in rec.diff_preview
    assert "critical" in rec.reason


def test_rollback_is_recommended_not_executed(fixture: Fixture) -> None:
    breaker = TrajectoryBreaker(fixture)
    (fixture.repo_path / "src" / "main.py").write_text("def main():\n    return 99\n")
    mutated_hash = tree_hash(fixture.repo_path)
    for t in _osc_stream("ABABA"):
        breaker.observe(t)

    # A recommendation exists, but without confirmation the working tree is untouched.
    assert breaker.recommendation is not None
    assert tree_hash(fixture.repo_path) == mutated_hash
    with pytest.raises(RollbackNotConfirmed):
        breaker.confirm_and_rollback(confirmed=False)
    assert tree_hash(fixture.repo_path) == mutated_hash


def test_confirmed_rollback_is_safe_and_scoped(tmp_path: Path) -> None:
    source = _make_source_repo(tmp_path / "src-root")
    source_before = tree_hash(source)
    cfg = FixtureConfig(source_roots=(tmp_path / "src-root",), scratch_root=tmp_path / "scratch")
    fx = materialize(source, cfg)
    try:
        breaker = TrajectoryBreaker(fx)
        baseline_hash = breaker.checkpoints[-1].tree_hash

        # Mutate: change a file and add a new one (rollback must revert both).
        (fx.repo_path / "src" / "main.py").write_text("def main():\n    return 99\n")
        (fx.repo_path / "scratch_note.txt").write_text("stray\n", encoding="utf-8")
        assert tree_hash(fx.repo_path) != baseline_hash

        for t in _osc_stream("ABABA"):
            breaker.observe(t)

        pre = breaker.confirm_and_rollback(confirmed=True)

        # Restored to baseline, and the pre-rollback snapshot retains the mutated content.
        assert tree_hash(fx.repo_path) == baseline_hash
        assert not (fx.repo_path / "scratch_note.txt").exists()
        assert (pre.snapshot_path / "scratch_note.txt").exists()

        # The original source repository is byte-identical to before the run.
        assert tree_hash(source) == source_before
    finally:
        teardown(fx)
