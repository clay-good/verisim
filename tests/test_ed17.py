"""ED17 — read-uncommitted isolation: the dirty read + its black-box recovery (SPEC-7 §3.2, inc10).

The smoke instance: read_uncommitted admits the dirty-read anomaly (a transaction observes another
active transaction's uncommitted write, which is then rolled back) while read_committed, snapshot,
and serializable forbid it (the MVCC ``tget`` gives only committed data); and Elle's value oracle
recovers the same dirty read black-box from the client history alone, matching the oracle on every
scenario. All four levels compose with Tier-B. The committed figure is from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed17 import ED17Config, run_ed17, write_csv


def _tiny() -> ED17Config:
    return ED17Config(dirty_objects=(0, 1), n_objects=4)


def test_run_ed17_is_well_formed():
    result = run_ed17(_tiny())
    expected = {"serializable", "snapshot", "read_committed", "read_uncommitted"}
    assert {r["isolation"] for r in result.dirty_read} == expected
    assert {r["isolation"] for r in result.recovery} == expected


def test_read_uncommitted_admits_dirty_read_stronger_levels_forbid_it():
    result = run_ed17(_tiny())
    by_level = {r["isolation"]: r for r in result.dirty_read}
    assert by_level["read_uncommitted"]["anomaly_rate"] == 1.0  # B sees A's uncommitted write
    assert by_level["read_committed"]["anomaly_rate"] == 0.0  # MVCC tget -> committed data only
    assert by_level["snapshot"]["anomaly_rate"] == 0.0
    assert by_level["serializable"]["anomaly_rate"] == 0.0


def test_elle_recovers_the_dirty_read_black_box_and_matches_the_oracle():
    result = run_ed17(_tiny())
    by_level = {r["isolation"]: r for r in result.recovery}
    # the value oracle flags `dirty-read` exactly where the oracle admits it, reference-free
    assert by_level["read_uncommitted"]["recovery_rate"] == 1.0
    assert by_level["read_committed"]["recovery_rate"] == 0.0
    assert all(r["matches_oracle"] for r in result.recovery)


def test_all_levels_compose_with_tier_b():
    result = run_ed17(_tiny())
    assert all(r["tier_b_agrees"] for r in result.dirty_read)


def test_write_csv_round_trips(tmp_path):
    result = run_ed17(_tiny())
    path = write_csv(result, tmp_path / "ed17.csv")
    text = path.read_text()
    assert "dirty_read,read_uncommitted" in text
    assert "recovery,serializable" in text
