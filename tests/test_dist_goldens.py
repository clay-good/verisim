"""DS0 golden trajectories: committed scripts -> exact canonical states (SPEC-7 §5.1, §13).

These pin the Tier-A reference oracle's distributed semantics; CI fails on any drift, exactly as
v0's ``test_goldens`` pins the filesystem and ``test_net_goldens`` pins the network. A 3-node config
keeps the expected canonical states readable, and the scenarios exercise the DS0-increment-1 core:
async replication + convergence, and the stale-read-under-partition dynamic that is the whole point
of the distributed world.
"""

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.serialize import to_canonical
from verisim.distoracle import ReferenceDistOracle

CONFIG = DistConfig(name="golden", nodes=("n0", "n1", "n2"), objects=("x", "y"))
ORACLE = ReferenceDistOracle(CONFIG)

# The strong-consistency counterpart, pinning the linearizable (synchronous, CP-under-partition)
# semantics added for the H20 consistency-level sweep (SPEC-7 §3.4, §5.1).
LIN_CONFIG = DistConfig(
    name="golden-lin", nodes=("n0", "n1", "n2"), objects=("x", "y"),
    consistency_model="linearizable",
)
LIN_ORACLE = ReferenceDistOracle(LIN_CONFIG)


def _final(cmds: list[str]) -> dict[str, object]:
    state = DistributedState.initial(CONFIG)
    for cmd in cmds:
        state = ORACLE.step(state, parse_dist_action(cmd)).state
    return to_canonical(state)


def _final_lin(cmds: list[str]) -> dict[str, object]:
    state = DistributedState.initial(LIN_CONFIG)
    for cmd in cmds:
        state = LIN_ORACLE.step(state, parse_dist_action(cmd)).state
    return to_canonical(state)


def _rep(obj: str, node: str, version: int, value: str) -> dict[str, object]:
    return {"object_id": obj, "node_id": node, "version": version, "value": value}


def _boot_y() -> list[dict[str, object]]:
    return [_rep("y", "n0", 0, "nil"), _rep("y", "n1", 0, "nil"), _rep("y", "n2", 0, "nil")]


def test_golden_put_replicates_and_converges_on_advance():
    final = _final(["put n0 x b", "advance 2"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"), _rep("x", "n1", 1, "b"), _rep("x", "n2", 1, "b"), *_boot_y()
        ],
        "log": [{"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []}],
        "inflight": [],  # both replication messages delivered
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 2,
        "next_event_id": 1,
        "next_msg_id": 2,
        "last_result": ["advanced", "2"],
    }


def test_golden_partition_leaves_isolated_replica_stale():
    final = _final(
        ["put n0 x b", "advance 2", "partition n0 n1 | n2", "put n0 x c", "advance 2"]
    )
    assert final == {
        "replicas": [
            _rep("x", "n0", 2, "c"),   # writer
            _rep("x", "n1", 2, "c"),   # same partition side: converged
            _rep("x", "n2", 1, "b"),   # isolated: stale at the pre-partition value
            *_boot_y(),
        ],
        "log": [
            {"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []},
            {"id": 1, "node": "n0", "op": "put n0 x c", "clock": 2, "happens_before": [0]},
        ],
        # the replication message to the isolated n2 is stuck in-flight (cannot cross the partition)
        "inflight": [
            {"id": 3, "src": "n0", "dst": "n2", "object_id": "x", "version": 2, "value": "c",
             "deliver_after": 3}
        ],
        "partitions": [["n0", "n1"], ["n2"]],
        "down": [],
        "clock": 4,
        "next_event_id": 2,
        "next_msg_id": 4,
        "last_result": ["advanced", "1"],
    }


def test_golden_linearizable_replicates_synchronously():
    # A single put commits to *every* replica in the same step — no in-flight, no advance needed,
    # the strong-consistency counterpart of the eventual-consistency async-then-converge golden.
    final = _final_lin(["put n0 x b"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"), _rep("x", "n1", 1, "b"), _rep("x", "n2", 1, "b"), *_boot_y()
        ],
        "log": [{"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []}],
        "inflight": [],          # synchronous: nothing left in flight
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 0,              # no advance was needed to converge
        "next_event_id": 1,
        "next_msg_id": 0,        # no replication messages were ever sent
        "last_result": ["ok", "b"],
    }


def test_golden_transaction_commit_applies_atomically_and_replicates():
    # A multi-key OCC transaction (DS0 incr 2): begin, buffer two writes, commit -> both keys bump
    # to version 1 at the coordinator and replicate async; advance converges every replica. The txn
    # is removed on commit (no ``txns`` key in the canonical form).
    final = _final(
        ["begin n0 t0", "tput n0 t0 x a", "tput n0 t0 y b", "commit n0 t0", "advance 2"]
    )
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "a"), _rep("x", "n1", 1, "a"), _rep("x", "n2", 1, "a"),
            _rep("y", "n0", 1, "b"), _rep("y", "n1", 1, "b"), _rep("y", "n2", 1, "b"),
        ],
        "log": [
            {"id": 0, "node": "n0", "op": "begin n0 t0", "clock": 0, "happens_before": []},
            {"id": 1, "node": "n0", "op": "tput n0 t0 x a", "clock": 0, "happens_before": [0]},
            {"id": 2, "node": "n0", "op": "tput n0 t0 y b", "clock": 0, "happens_before": [0, 1]},
            {"id": 3, "node": "n0", "op": "commit n0 t0", "clock": 0, "happens_before": [0, 1, 2]},
        ],
        "inflight": [],  # all four replication messages (x,y -> n1,n2) delivered on advance
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 2,
        "next_event_id": 4,
        "next_msg_id": 4,
        "last_result": ["advanced", "4"],
    }


def test_golden_transaction_conflict_aborts_first_committer_wins():
    # OCC first-committer-wins: t0 reads x (version 0), a concurrent ``put`` bumps x to version 1,
    # then t0's commit validates its read-set, finds x changed, and ABORTS (``conflict``) — none of
    # t0's writes apply, and the txn is discarded (no ``txns`` key). x holds the concurrent put's
    # value; its replication is still in flight (no advance yet).
    final = _final(["begin n0 t0", "tget n0 t0 x", "put n0 x c", "commit n0 t0"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "c"), _rep("x", "n1", 0, "nil"), _rep("x", "n2", 0, "nil"),
            *_boot_y(),
        ],
        "log": [
            {"id": 0, "node": "n0", "op": "begin n0 t0", "clock": 0, "happens_before": []},
            {"id": 1, "node": "n0", "op": "tget n0 t0 x", "clock": 0, "happens_before": [0]},
            {"id": 2, "node": "n0", "op": "put n0 x c", "clock": 0, "happens_before": [0, 1]},
            {"id": 3, "node": "n0", "op": "commit n0 t0", "clock": 0, "happens_before": [0, 1, 2]},
        ],
        "inflight": [
            {"id": 0, "src": "n0", "dst": "n1", "object_id": "x", "version": 1, "value": "c",
             "deliver_after": 1},
            {"id": 1, "src": "n0", "dst": "n2", "object_id": "x", "version": 1, "value": "c",
             "deliver_after": 1},
        ],
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 0,
        "next_event_id": 4,
        "next_msg_id": 2,
        "last_result": ["conflict", ""],
    }


def test_golden_write_skew_admitted_under_snapshot_forbidden_under_serializable():
    # The canonical write-skew scenario: A and B both read {x, y}; A writes x, B writes y. Under
    # snapshot isolation (write-write validation only) both disjoint-write txns commit — write skew.
    # Under serializable (read-set validation) A's commit invalidates B's read of x, so B aborts.
    skew = ["begin n0 A", "begin n0 B", "tget n0 A x", "tget n0 A y", "tget n0 B x", "tget n0 B y",
            "tput n0 A x a", "tput n0 B y b", "commit n0 A", "commit n0 B"]

    def outcomes(isolation: str) -> tuple[list[str], str, str]:
        config = DistConfig(name="golden-iso", nodes=("n0", "n1", "n2"), objects=("x", "y"),
                            txn_isolation=isolation)
        oracle = ReferenceDistOracle(config)
        state = DistributedState.initial(config)
        statuses: list[str] = []
        for cmd in skew:
            r = oracle.step(state, parse_dist_action(cmd))
            statuses.append(r.status)
            state = r.state
        return statuses[-2:], state.replicas[("x", "n0")].value, state.replicas[("y", "n0")].value

    snap_commits, snap_x, snap_y = outcomes("snapshot")
    assert snap_commits == ["committed", "committed"]  # both commit — write skew
    assert (snap_x, snap_y) == ("a", "b")  # both writes landed (the anomaly)

    ser_commits, ser_x, ser_y = outcomes("serializable")
    assert ser_commits == ["committed", "conflict"]  # A commits, B aborts — no write skew
    assert (ser_x, ser_y) == ("a", "nil")  # only A's write landed


def test_golden_elle_recovers_write_skew_as_a_g2_cycle_black_box():
    # The Elle (DS3 incr 2) counterpart of the write-skew golden: a checker that sees only the
    # observable transaction history reconstructs Adya's DSG and reports the anomaly the oracle
    # admits — a G2 anti-dependency cycle A->B->A — with no oracle and no cluster state consulted.
    from verisim.distoracle.elle import TxnObservation, check_serializable

    # The observable footprint of the snapshot write-skew run: both txns read {x@0, y@0}; A installs
    # x@1, B installs y@1 (both committed, the anomaly).
    snapshot_history = [
        TxnObservation("A", reads=(("x", 0), ("y", 0)), writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 0), ("y", 0)), writes=(("y", 1),)),
    ]
    report = check_serializable(snapshot_history)
    assert not report.serializable
    assert report.anomaly == "G2"
    assert set(report.cycle) == {"A", "B"}
    assert set(report.cycle_kinds) == {"rw"}  # write skew is a pure anti-dependency cycle

    # Under serializable, B aborts, so the history is the single committed txn A — acyclic.
    serializable_history = [
        TxnObservation("A", reads=(("x", 0), ("y", 0)), writes=(("x", 1),)),
    ]
    assert check_serializable(serializable_history).serializable


def test_golden_version_oracle_recovers_write_skew_from_values_alone():
    # The DS3-incr-3 counterpart: the same write skew, but Elle is handed only list-append *values*
    # (no store-supplied MVCC versions) and must recover the order itself (the version oracle).
    from verisim.distoracle.elle import AppendObservation, check_serializable_appends

    # Both txns read empty {x, y} then append disjoint halves; only the values reach Elle.
    snapshot_appends = [
        AppendObservation("A", appends=(("x", "ax"),), list_reads=(("x", ()), ("y", ()))),
        AppendObservation("B", appends=(("y", "by"),), list_reads=(("x", ()), ("y", ()))),
    ]
    report = check_serializable_appends(snapshot_appends)
    assert not report.serializable
    assert report.anomaly == "G2"  # recovered from values, identical to the supplied-version golden
    assert set(report.cycle) == {"A", "B"}
    assert set(report.cycle_kinds) == {"rw"}


def test_golden_version_oracle_catches_split_brain_fork():
    # The split-brain anomaly the integer-version mode cannot represent: two reads of x fork.
    from verisim.distoracle.elle import AppendObservation, check_serializable_appends

    fork = [
        AppendObservation("A", appends=(("x", "a"),)),
        AppendObservation("B", appends=(("x", "b"),)),
        AppendObservation("R1", list_reads=(("x", ("a", "b")),)),
        AppendObservation("R2", list_reads=(("x", ("b", "a")),)),
    ]
    report = check_serializable_appends(fork)
    assert not report.serializable
    assert report.anomaly == "incompatible-order"


def test_golden_linearizable_rejects_write_under_partition():
    # CP under partition: a synchronous write that cannot reach all replicas is rejected
    # (``unavailable``) rather than committed locally — so no replica is ever stale (vs the
    # eventual golden above, where the same script leaves n2 stale).
    final = _final_lin(["put n0 x b", "partition n0 n1 | n2", "put n0 x c"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"), _rep("x", "n1", 1, "b"), _rep("x", "n2", 1, "b"), *_boot_y()
        ],
        "log": [
            {"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []},
            {"id": 1, "node": "n0", "op": "put n0 x c", "clock": 0, "happens_before": [0]},
        ],
        "inflight": [],          # the rejected write enqueued nothing
        "partitions": [["n0", "n1"], ["n2"]],
        "down": [],
        "clock": 0,
        "next_event_id": 2,
        "next_msg_id": 0,
        "last_result": ["unavailable", ""],  # partitioned write rejected, not stale-committed
    }
