"""Elle-style serializability checking (SPEC-7 §5, §9.1; DS3 incr 2).

The checker reconstructs Adya's Direct Serialization Graph from an observable transaction history
and reports a cycle iff the schedule was non-serializable. These tests pin the three things that
make it a faithful Elle: write skew is a G2 anti-dependency cycle, a clean serial history is
acyclic, and the canonical anomaly classes (G0/G1c/G2) are distinguished.
"""

from __future__ import annotations

from verisim.distoracle.elle import (
    AppendObservation,
    Edge,
    TxnObservation,
    appends_to_version_history,
    build_dsg,
    check_serializable,
    check_serializable_appends,
    recover_versions,
)


def test_write_skew_is_a_g2_cycle():
    # A reads x@0,y@0 then writes x@1; B reads x@0,y@0 then writes y@1 (disjoint write-sets).
    history = [
        TxnObservation("A", reads=(("x", 0), ("y", 0)), writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 0), ("y", 0)), writes=(("y", 1),)),
    ]
    report = check_serializable(history)
    assert not report.serializable
    assert report.anomaly == "G2"
    assert set(report.cycle) == {"A", "B"}
    assert "rw" in report.cycle_kinds


def test_serial_history_is_serializable():
    # A writes x@1; B reads x@1 (sees A) then writes y@1 — a clean serial order A then B, acyclic.
    history = [
        TxnObservation("A", reads=(("x", 0),), writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 1),), writes=(("y", 1),)),
    ]
    report = check_serializable(history)
    assert report.serializable
    assert report.anomaly == ""
    assert report.cycle == ()


def test_empty_and_single_txn_histories_are_serializable():
    assert check_serializable([]).serializable
    assert check_serializable([TxnObservation("only", writes=(("x", 1),))]).serializable


def test_wr_edge_emitted_for_a_read_of_a_written_version():
    history = [
        TxnObservation("A", writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 1),)),
    ]
    edges = build_dsg(history)
    assert Edge("A", "B", "wr", "x") in edges


def test_rw_edge_targets_the_immediate_successor_version():
    # A reads x@0; B installs x@1 (the immediate successor) -> rw A->B.
    history = [
        TxnObservation("A", reads=(("x", 0),)),
        TxnObservation("B", writes=(("x", 1),)),
    ]
    edges = build_dsg(history)
    assert Edge("A", "B", "rw", "x") in edges


def test_ww_edge_between_consecutive_writers():
    history = [
        TxnObservation("A", writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 1),), writes=(("x", 2),)),
    ]
    edges = build_dsg(history)
    assert Edge("A", "B", "ww", "x") in edges


def test_g1c_circular_information_flow_without_rw():
    # A reads B's write of x and B reads A's write of y: a wr/wr cycle (no anti-dependency) = G1c.
    history = [
        TxnObservation("A", reads=(("x", 1),), writes=(("y", 1),)),
        TxnObservation("B", reads=(("y", 1),), writes=(("x", 1),)),
    ]
    report = check_serializable(history)
    assert not report.serializable
    assert report.anomaly == "G1c"
    assert "rw" not in report.cycle_kinds


def test_g0_dirty_write_cycle_uses_only_ww():
    # A and B interleave writes so x orders A<B but y orders B<A: a pure ww cycle = G0.
    history = [
        TxnObservation("A", writes=(("x", 1), ("y", 2))),
        TxnObservation("B", writes=(("x", 2), ("y", 1))),
    ]
    report = check_serializable(history)
    assert not report.serializable
    assert report.anomaly == "G0"
    assert set(report.cycle_kinds) == {"ww"}


def test_build_dsg_is_deterministic():
    history = [
        TxnObservation("A", reads=(("x", 0), ("y", 0)), writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 0), ("y", 0)), writes=(("y", 1),)),
    ]
    assert build_dsg(history) == build_dsg(list(reversed(history)))


# --- the version oracle: list-append / value-recoverable histories (DS3 increment 3) --------------


def test_version_oracle_recovers_order_from_read_prefixes():
    # Three reads of one key, all prefixes of the same growing append log -> the order is [a,b,c].
    history = [
        AppendObservation("A", appends=(("x", "a"),)),
        AppendObservation("B", appends=(("x", "b"),), list_reads=(("x", ("a",)),)),
        AppendObservation("C", appends=(("x", "c"),), list_reads=(("x", ("a", "b")),)),
        AppendObservation("R", list_reads=(("x", ("a", "b", "c")),)),
    ]
    recovered = recover_versions(history)
    assert recovered.ok
    assert recovered.order == {"x": ["a", "b", "c"]}


def test_version_oracle_recovers_write_skew_as_a_g2_cycle_from_values():
    # The list-append form of write skew: both read empty {x,y}, append disjoint halves.
    history = [
        AppendObservation("A", appends=(("x", "ax"),), list_reads=(("x", ()), ("y", ()))),
        AppendObservation("B", appends=(("y", "by"),), list_reads=(("x", ()), ("y", ()))),
    ]
    report = check_serializable_appends(history)
    assert not report.serializable
    assert report.anomaly == "G2"
    assert set(report.cycle) == {"A", "B"}
    assert set(report.cycle_kinds) == {"rw"}


def test_value_recovery_matches_supplied_versions_on_a_clean_schedule():
    # The same schedule, two ways: the value-recovery path and the integer-version path agree.
    supplied = [
        TxnObservation("A", reads=(("x", 0),), writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 1),), writes=(("y", 1),)),
    ]
    appends = [
        AppendObservation("A", appends=(("x", "x#1"),), list_reads=(("x", ()),)),
        AppendObservation("B", appends=(("y", "y#1"),), list_reads=(("x", ("x#1",)),)),
    ]
    recovered = recover_versions(appends)
    assert recovered.ok
    # the version oracle reproduces the store's exact version history (soundness)
    assert appends_to_version_history(appends, recovered) == supplied
    val, ver = check_serializable_appends(appends), check_serializable(supplied)
    assert val.serializable == ver.serializable


def test_split_brain_fork_is_incompatible_order():
    # Two reads of x disagree on append order (neither a prefix of the other) -> a fork.
    history = [
        AppendObservation("A", appends=(("x", "a"),)),
        AppendObservation("B", appends=(("x", "b"),)),
        AppendObservation("R1", list_reads=(("x", ("a", "b")),)),
        AppendObservation("R2", list_reads=(("x", ("b", "a")),)),
    ]
    report = check_serializable_appends(history)
    assert not report.serializable
    assert report.anomaly == "incompatible-order"
    assert report.cycle == ()  # a recovery anomaly, not a DSG cycle
    assert "incompatible-order (recovery anomaly" in report.detail


def test_dirty_read_of_an_uncommitted_value():
    # A read observes a value no committed transaction ever appended (Adya G1a).
    history = [AppendObservation("A", list_reads=(("x", ("ghost",)),))]
    report = check_serializable_appends(history)
    assert not report.serializable
    assert report.anomaly == "dirty-read"


def test_duplicate_write_is_caught():
    history = [
        AppendObservation("A", appends=(("x", "v"),)),
        AppendObservation("B", appends=(("x", "v"),)),  # same value appended twice
    ]
    report = check_serializable_appends(history)
    assert not report.serializable
    assert report.anomaly == "duplicate-write"


def test_recover_versions_places_unread_appends_after_the_read_prefix():
    # 'a' is read (pinned first), 'b' and 'c' are appended but never read -> after, value-sorted.
    history = [
        AppendObservation("A", appends=(("x", "a"),)),
        AppendObservation("R", list_reads=(("x", ("a",)),)),
        AppendObservation("C", appends=(("x", "c"),)),
        AppendObservation("B", appends=(("x", "b"),)),
    ]
    recovered = recover_versions(history)
    assert recovered.order == {"x": ["a", "b", "c"]}


def test_recover_versions_is_deterministic_under_reordering():
    history = [
        AppendObservation("A", appends=(("x", "a"),)),
        AppendObservation("B", appends=(("x", "b"),), list_reads=(("x", ("a",)),)),
        AppendObservation("R", list_reads=(("x", ("a", "b")),)),
    ]
    assert recover_versions(history).order == recover_versions(list(reversed(history))).order
