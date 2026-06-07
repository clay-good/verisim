"""ED16 — read-committed isolation: lost update + the cost of preventing it (SPEC-7 §3.2, DS0 inc9).

The smoke instance: read_committed admits the lost-update anomaly (both same-key read-modify-write
transactions commit, the earlier write silently overwritten) while snapshot and serializable forbid
it (the second committer aborts on the write-write conflict), and read_committed pays strictly fewer
aborts under contention precisely because it admits those lost updates. All three levels compose
with Tier-B. The committed figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed16 import ED16Config, run_ed16, write_csv


def _tiny() -> ED16Config:
    return ED16Config(lost_update_objects=(0, 1), n_txns=8, n_objects=4, seeds=(0, 1, 2, 3, 4))


def test_run_ed16_is_well_formed():
    result = run_ed16(_tiny())
    expected = {"serializable", "snapshot", "read_committed"}
    assert {r["isolation"] for r in result.lost_update} == expected
    assert {r["isolation"] for r in result.abort_rate} == expected


def test_read_committed_admits_lost_update_stronger_levels_forbid_it():
    result = run_ed16(_tiny())
    by_level = {r["isolation"]: r for r in result.lost_update}
    assert by_level["read_committed"]["anomaly_rate"] == 1.0  # both commit -> lost update
    assert by_level["snapshot"]["anomaly_rate"] == 0.0  # second aborts -> no lost update
    assert by_level["serializable"]["anomaly_rate"] == 0.0


def test_read_committed_aborts_less_than_the_validating_levels():
    result = run_ed16(_tiny())
    by_level = {r["isolation"]: r for r in result.abort_rate}
    # the price it sells correctness for: read_committed never aborts (no validation), the others do
    assert by_level["read_committed"]["abort_rate"] == 0.0
    assert by_level["snapshot"]["abort_rate"] > 0.0
    assert by_level["serializable"]["abort_rate"] > 0.0


def test_all_levels_compose_with_tier_b():
    result = run_ed16(_tiny())
    assert all(r["tier_b_agrees"] for r in result.lost_update)


def test_write_csv_round_trips(tmp_path):
    result = run_ed16(_tiny())
    path = write_csv(result, tmp_path / "ed16.csv")
    text = path.read_text()
    assert "lost_update,read_committed" in text
    assert "abort_rate,serializable" in text
