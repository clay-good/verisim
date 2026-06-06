"""ED9 — transaction isolation: write-skew + the price of serializability (SPEC-7 §3.2, DS0 incr 3).

The smoke instance: snapshot isolation admits the write-skew anomaly (both disjoint-write
transactions commit) while serializable forbids it (one aborts), and serializable pays a strictly
higher abort rate under read-heavy contention to buy that guarantee. Both compose with Tier-B.
The committed figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed9 import ED9Config, run_ed9, write_csv


def _tiny() -> ED9Config:
    return ED9Config(
        skew_object_pairs=((0, 1), (1, 2)), n_txns=8, n_objects=4, seeds=(0, 1, 2, 3, 4)
    )


def test_run_ed9_is_well_formed():
    result = run_ed9(_tiny())
    assert {r["isolation"] for r in result.write_skew} == {"serializable", "snapshot"}
    assert {r["isolation"] for r in result.abort_rate} == {"serializable", "snapshot"}


def test_snapshot_admits_write_skew_serializable_forbids_it():
    result = run_ed9(_tiny())
    by_level = {r["isolation"]: r for r in result.write_skew}
    assert by_level["snapshot"]["anomaly_rate"] == 1.0  # both txns commit -> write skew
    assert by_level["serializable"]["anomaly_rate"] == 0.0  # one aborts -> no write skew


def test_serializable_aborts_more_than_snapshot():
    result = run_ed9(_tiny())
    by_level = {r["isolation"]: r for r in result.abort_rate}
    # the price of serializability: read-set validation aborts strictly more than write-set-only
    assert by_level["serializable"]["abort_rate"] > by_level["snapshot"]["abort_rate"]


def test_both_levels_compose_with_tier_b():
    result = run_ed9(_tiny())
    assert all(r["tier_b_agrees"] for r in result.write_skew)


def test_write_csv_round_trips(tmp_path):
    result = run_ed9(_tiny())
    path = write_csv(result, tmp_path / "ed9.csv")
    text = path.read_text()
    assert "write_skew,serializable" in text
    assert "abort_rate,snapshot" in text
