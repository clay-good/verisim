"""ED8 — the OCC transaction commit/abort frontier under contention (SPEC-7 §3.2, DS0 increment 2).

The smoke instance: a tiny seeded sweep checking the OCC semantics produce the right structural
shape — the measured commit rate tracks the balls-in-bins occupancy law, commit rate rises
monotonically as objects multiply (contention drops), and the transaction layer composes with
Tier-B (the autonomous-actor system oracle agrees with Tier-A on every scenario). The committed
figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed8 import ED8Config, run_ed8, write_csv


def _tiny() -> ED8Config:
    return ED8Config(n_txns=8, object_counts=(1, 2, 4, 8), seeds=(0, 1, 2, 3, 4))


def test_run_ed8_is_well_formed():
    result = run_ed8(_tiny())
    assert [r["objects"] for r in result.rows] == [1, 2, 4, 8]
    for r in result.rows:
        assert 0.0 <= r["commit_rate"] <= 1.0


def test_commit_rate_tracks_the_occupancy_law():
    result = run_ed8(_tiny())
    for r in result.rows:
        # measured OCC commit rate matches the closed-form balls-in-bins prediction (the semantics
        # are exactly right, not merely plausible); the small gap is sampling noise over few seeds
        assert abs(r["commit_rate"] - r["occupancy_rate"]) < 0.1, r


def test_commit_rate_rises_as_contention_drops():
    result = run_ed8(_tiny())
    rates = [r["commit_rate"] for r in result.rows]
    assert rates == sorted(rates)  # more objects -> fewer read-set collisions -> more commits
    assert rates[0] < rates[-1]


def test_single_hot_object_aborts_all_but_one():
    # M=1: every transaction reads+writes the one object, so exactly one commits per batch (the
    # first committer) and the rest abort — commit rate = 1/K.
    result = run_ed8(ED8Config(n_txns=8, object_counts=(1,), seeds=(0, 1, 2, 3, 4)))
    assert abs(result.rows[0]["commit_rate"] - 1.0 / 8) < 1e-9


def test_tier_b_agrees_on_every_scenario():
    result = run_ed8(_tiny())
    assert all(r["tier_b_agrees"] for r in result.rows)


def test_write_csv_round_trips(tmp_path):
    result = run_ed8(_tiny())
    path = write_csv(result, tmp_path / "ed8.csv")
    text = path.read_text()
    assert "objects,commit_rate" in text
    assert "occupancy_rate" in text
