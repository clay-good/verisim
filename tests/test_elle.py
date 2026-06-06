"""Elle-style serializability checking (SPEC-7 §5, §9.1; DS3 incr 2).

The checker reconstructs Adya's Direct Serialization Graph from an observable transaction history
and reports a cycle iff the schedule was non-serializable. These tests pin the three things that
make it a faithful Elle: write skew is a G2 anti-dependency cycle, a clean serial history is
acyclic, and the canonical anomaly classes (G0/G1c/G2) are distinguished.
"""

from __future__ import annotations

from verisim.distoracle.elle import (
    Edge,
    TxnObservation,
    build_dsg,
    check_serializable,
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
