"""ED10 — Elle: black-box serializability checking (SPEC-7 §5, §9.1; DS3 incr 2).

The smoke instance + the golden the figure pins: Elle recovers the write-skew anomaly the oracle
sees (a G2 cycle under snapshot, none under serializable) reading only the observable history, and
it certifies the serializable level (zero non-serializable contended histories) while catching the
anomalies snapshot admits. The committed figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed10 import ED10Config, run_ed10, write_csv


def _tiny() -> ED10Config:
    return ED10Config(
        skew_object_pairs=((0, 1), (1, 2)), n_txns=8, n_objects=4, seeds=(0, 1, 2, 3, 4)
    )


def test_run_ed10_is_well_formed():
    result = run_ed10(_tiny())
    assert {r["isolation"] for r in result.write_skew} == {"serializable", "snapshot"}
    assert {r["isolation"] for r in result.contention} == {"serializable", "snapshot"}


def test_elle_recovers_write_skew_black_box():
    result = run_ed10(_tiny())
    by_level = {r["isolation"]: r for r in result.write_skew}
    # snapshot histories are non-serializable via a G2 anti-dependency cycle; serializable are not
    assert by_level["snapshot"]["elle_g2_rate"] == 1.0
    assert by_level["serializable"]["elle_g2_rate"] == 0.0


def test_elle_agrees_with_the_oracle_on_every_scenario():
    result = run_ed10(_tiny())
    # the black-box checker's verdict matches the omniscient oracle's commit-count (ED9) exactly
    assert all(r["elle_matches_oracle"] for r in result.write_skew)


def test_elle_certifies_serializable_and_catches_snapshot():
    result = run_ed10(_tiny())
    by_level = {r["isolation"]: r for r in result.contention}
    assert by_level["serializable"]["nonserializable_rate"] == 0.0  # certified, zero cycles
    assert by_level["snapshot"]["nonserializable_rate"] > 0.0  # SI admits anomalies Elle catches


def test_write_csv_round_trips(tmp_path):
    result = run_ed10(_tiny())
    path = write_csv(result, tmp_path / "ed10.csv")
    text = path.read_text()
    assert "write_skew,snapshot,elle_g2_rate" in text
    assert "contention,serializable,nonserializable_rate" in text
