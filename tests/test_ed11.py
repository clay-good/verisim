"""ED11 — Elle's version oracle: serializability from values + the split-brain fork (DS3 incr 3).

The smoke instance + the goldens the figure pins: the version oracle is *sound* (recovering
versions from list-append values reproduces the store's exact version history, so the G2 write-skew
rate is ED10's, with zero store cooperation), and value-recovery catches the **split-brain fork**
(``incompatible-order``) the integer-version mode structurally cannot represent. The committed
figure comes from the local run, not CI.
"""

from __future__ import annotations

from verisim.experiments.ed11 import ED11Config, run_ed11, to_append_history, write_csv


def _tiny() -> ED11Config:
    return ED11Config(skew_object_pairs=((0, 1), (1, 2)), n_objects=4, n_forks=4)


def test_run_ed11_is_well_formed():
    result = run_ed11(_tiny())
    assert {r["isolation"] for r in result.recovery} == {"serializable", "snapshot"}
    assert "incompatible_order_rate" in result.fork


def test_version_oracle_is_sound_and_recovers_write_skew():
    result = run_ed11(_tiny())
    by_level = {r["isolation"]: r for r in result.recovery}
    # recovering versions from values alone reproduces the store-supplied history exactly...
    assert all(r["recovery_sound"] and r["agrees_supplied"] for r in result.recovery)
    # ...so the black-box verdict is ED10's: G2 write skew under snapshot, none under serializable.
    assert by_level["snapshot"]["elle_g2_rate"] == 1.0
    assert by_level["serializable"]["elle_g2_rate"] == 0.0


def test_split_brain_fork_is_caught_and_control_is_clean():
    result = run_ed11(_tiny())
    assert result.fork["incompatible_order_rate"] == 1.0  # every fork caught as incompatible-order
    assert result.fork["clean_control_flag_rate"] == 0.0  # the un-forked control is serializable


def test_to_append_history_round_trips_versions():
    from verisim.distoracle.elle import (
        TxnObservation,
        appends_to_version_history,
        recover_versions,
    )

    supplied = [
        TxnObservation("A", reads=(("x", 0), ("y", 0)), writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 1),), writes=(("y", 1),)),
    ]
    appends = to_append_history(supplied)
    recovered = recover_versions(appends)
    assert recovered.ok
    assert appends_to_version_history(appends, recovered) == supplied


def test_write_csv_round_trips(tmp_path):
    result = run_ed11(_tiny())
    path = write_csv(result, tmp_path / "ed11.csv")
    text = path.read_text()
    assert "recovery,snapshot,elle_g2_rate" in text
    assert "fork,split_brain,incompatible_order_rate" in text
