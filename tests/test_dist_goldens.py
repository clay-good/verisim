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

# The Raft-subset consensus counterpart (DS0 increment 7): a 3-node cluster, so a strict majority is
# 2 — a write from the 2-node side commits (sync to the majority, async to the stale minority) where
# the same partition makes linearizable reject.
QUORUM_CONFIG = DistConfig(
    name="golden-quorum", nodes=("n0", "n1", "n2"), objects=("x", "y"),
    consistency_model="quorum",
)
QUORUM_ORACLE = ReferenceDistOracle(QUORUM_CONFIG)


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


def _final_quorum(cmds: list[str]) -> dict[str, object]:
    state = DistributedState.initial(QUORUM_CONFIG)
    for cmd in cmds:
        state = QUORUM_ORACLE.step(state, parse_dist_action(cmd)).state
    return to_canonical(state)


# The pessimistic concurrency-control counterpart (DS0 increment 8): lock-based 2PL with wound-wait.
TPL_CONFIG = DistConfig(
    name="golden-2pl", nodes=("n0", "n1", "n2"), objects=("x", "y"),
    concurrency_control="2pl",
)
TPL_ORACLE = ReferenceDistOracle(TPL_CONFIG)


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


def test_golden_drop_loses_the_write_and_heal_cannot_recover_it():
    # The message-loss golden (SPEC-7 §3.2, DS0 incr 11): a write, then drop the n0->n1 replication
    # message, then advance + heal + advance. Unlike the partition golden above (whose held message
    # is delivered on heal), the *dropped* message is gone, so n1 stays permanently stale at the
    # boot value — the eventual-consistency convergence guarantee broken by an unreliable network.
    final = _final(["put n0 x b", "drop n0 n1", "advance 2", "heal", "advance 2"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"),   # writer
            _rep("x", "n1", 0, "nil"),  # dropped: never received the write, heal cannot repair it
            _rep("x", "n2", 1, "b"),   # the surviving message was delivered
            *_boot_y(),
        ],
        "log": [{"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []}],
        "inflight": [],  # the dropped message left nothing in flight (destroyed, not held)
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 4,
        "next_event_id": 1,
        "next_msg_id": 2,  # the dropped message id was still allocated (the write enqueued two)
        "last_result": ["advanced", "0"],  # the final advance delivered nothing — nothing remained
    }
    # contrast: a *newer* write reaches n1 where heal could not — the only repair for a lost write.
    state = DistributedState.initial(CONFIG)
    for cmd in ["put n0 x b", "drop n0 n1", "advance 2", "put n0 x c", "advance 2"]:
        state = ORACLE.step(state, parse_dist_action(cmd)).state
    r = state.replicas[("x", "n1")]
    assert (r.version, r.value) == (2, "c")  # the overwrite reached n1; the lost "b" never did


def test_golden_anti_entropy_repairs_a_dropped_write_without_a_new_write():
    # The read-repair golden (SPEC-7 §5.1, DS0 incr 12): drop the write to n1, advance, heal — n1 is
    # stale (the ED18 leftover) — then `anti_entropy n1` pulls the latest from its reachable peers
    # and repairs it WITHOUT a fresh write, where `advance` never could. The repair is a
    # ReplicaWrite (no new edit type), so the canonical form is unchanged in shape.
    final = _final(["put n0 x b", "drop n0 n1", "advance 2", "heal", "anti_entropy n1"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"),
            _rep("x", "n1", 1, "b"),   # read-repaired from a reachable peer (was 0/nil post-drop)
            _rep("x", "n2", 1, "b"),
            *_boot_y(),
        ],
        "log": [{"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []}],
        "inflight": [],
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 2,  # the single advance; anti_entropy does not move the clock (no event, no time)
        "next_event_id": 1,
        "next_msg_id": 2,
        "last_result": ["repaired", "1"],  # one replica reconciled
    }
    # bounded by reachability: while n1 is partitioned away, the same op repairs nothing.
    state = DistributedState.initial(CONFIG)
    for cmd in ["put n0 x b", "drop n0 n1", "advance 2", "partition n1 | n0 n2", "anti_entropy n1"]:
        state = ORACLE.step(state, parse_dist_action(cmd)).state
    r = state.replicas[("x", "n1")]
    assert (r.version, r.value) == (0, "nil")  # nothing reachable held the new value


def test_golden_delay_defers_a_message_but_holds_it_in_flight():
    # The recoverable-delay golden (SPEC-7 §3.4, DS0 incr 13): a write, then delay the n0->n1
    # replication message by 5 (deliver_after 1 -> 6), then advance to clock 2. Unlike drop, the
    # message is *held*, not destroyed — n1 is stale now but the message is still in flight (due at
    # clock 6), so it will arrive on a later advance (the recoverable half of the medium).
    final = _final(["put n0 x b", "delay n0 n1 5", "advance 2"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"),    # writer
            _rep("x", "n1", 0, "nil"),  # deferred: not yet delivered at clock 2 (due at 6)
            _rep("x", "n2", 1, "b"),    # the un-delayed peer's message was delivered
            *_boot_y(),
        ],
        "log": [{"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []}],
        "inflight": [
            # the delayed message is held (not lost): same payload, deliver_after pushed 1 -> 6.
            {"id": 0, "src": "n0", "dst": "n1", "object_id": "x", "version": 1, "value": "b",
             "deliver_after": 6}
        ],
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 2,
        "next_event_id": 1,
        "next_msg_id": 2,
        "last_result": ["advanced", "1"],  # only the un-delayed n0->n2 message delivered
    }


def test_golden_reorder_flips_which_write_arrives_first():
    # The reorder golden (SPEC-7 §3.4, DS0 incr 13): two writes to x (b then c); delay the first so
    # the newer write (c) is scheduled to arrive first; reorder reverses the schedule, so the older
    # write (b) arrives first instead. advance 2 then delivers b to n1 (the transit flip) while c is
    # held to deliver_after 101 — yet n2 (un-reordered) already shows the converged c. Last-writer-
    # wins keeps the eventual value at c regardless; only the in-transit observation moved.
    final = _final(["put n0 x b", "delay n0 n1 100", "put n0 x c", "reorder n0 n1", "advance 2"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 2, "c"),   # writer: the newer value
            _rep("x", "n1", 1, "b"),   # transit flip: reorder made the OLDER write arrive first
            _rep("x", "n2", 2, "c"),   # un-reordered peer: already converged to the newer value
            *_boot_y(),
        ],
        "log": [
            {"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []},
            {"id": 1, "node": "n0", "op": "put n0 x c", "clock": 0, "happens_before": [0]},
        ],
        "inflight": [
            # the newer write's message to n1 is held (reorder pushed it to deliver_after 101).
            {"id": 2, "src": "n0", "dst": "n1", "object_id": "x", "version": 2, "value": "c",
             "deliver_after": 101}
        ],
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 2,
        "next_event_id": 2,
        "next_msg_id": 4,
        "last_result": ["advanced", "3"],  # b->n1, b->n2, then c->n2 (c->n1 still held)
    }


def test_golden_clock_skew_defers_a_nodes_sends():
    # The clock-skew golden (SPEC-7 §3.4, DS0 incr 14): skew n0's clock +3, so the messages it sends
    # are stamped with deliver_after = clock + 3 + 1 = 4. advance 2 cannot deliver them (4 > 2), so
    # both peers stay stale and the messages are held in flight — clock skew deferring delivery,
    # with the per-node offset recorded in the (otherwise-omitted) `skew` map.
    final = _final(["clock_skew n0 3", "put n0 x b", "advance 2"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"),    # writer
            _rep("x", "n1", 0, "nil"),  # deferred: the +3-skewed send is not yet due at clock 2
            _rep("x", "n2", 0, "nil"),
            *_boot_y(),
        ],
        "log": [{"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []}],
        "inflight": [
            {"id": 0, "src": "n0", "dst": "n1", "object_id": "x", "version": 1, "value": "b",
             "deliver_after": 4},
            {"id": 1, "src": "n0", "dst": "n2", "object_id": "x", "version": 1, "value": "b",
             "deliver_after": 4},
        ],
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 2,
        "next_event_id": 1,
        "next_msg_id": 2,
        "last_result": ["advanced", "0"],  # nothing due at clock 2 (both sends deferred to 4)
        "skew": {"n0": 3},  # the per-node offset, omitted-when-empty (here non-empty)
    }


def test_golden_clock_skew_zero_clears_the_offset():
    # A 0 offset clears the skew (no residue): re-syncing n0 returns it to the global clock, so its
    # send is stamped with the normal deliver_after = 1, delivered by advance 2 — and the canonical
    # form carries no `skew` key, byte-identical to the pre-increment-14 normal form.
    final = _final(["clock_skew n0 3", "clock_skew n0 0", "put n0 x b", "advance 2"])
    assert "skew" not in final  # the 0 offset cleared it — no residue
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"), _rep("x", "n1", 1, "b"), _rep("x", "n2", 1, "b"), *_boot_y()
        ],
        "log": [{"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []}],
        "inflight": [],  # both sends delivered (deliver_after 1 ≤ clock 2) — re-synced
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 2,
        "next_event_id": 1,
        "next_msg_id": 2,
        "last_result": ["advanced", "2"],
    }


def test_golden_gossip_pairwise_reconciles_a_dropped_write():
    # The pairwise-gossip golden (SPEC-7 §4, DS0 incr 15): a write is dropped to n1 (n1 stays stale
    # the way ED18 showed), then `gossip n0 n1` reconciles the pair — n1 adopts the winning
    # (version, value) of the two replicas with no in-flight message, so the cluster converges (the
    # Dynamo/Cassandra pairwise anti-entropy, vs anti_entropy's one-directional pull-to-one-node).
    final = _final(["put n0 x b", "drop n0 n1", "advance 2", "gossip n0 n1"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"),   # writer
            _rep("x", "n1", 1, "b"),   # gossip pulled it up to the winner (was stale post-drop)
            _rep("x", "n2", 1, "b"),   # the un-dropped peer already had it
            *_boot_y(),
        ],
        "log": [{"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []}],
        "inflight": [],  # gossip reads the replicas directly — no in-flight message left or needed
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 2,
        "next_event_id": 1,
        "next_msg_id": 2,
        "last_result": ["gossiped", "1"],  # one replica (n1) moved
    }


def test_golden_consensus_elect_then_propose_under_partition():
    # The consensus golden (SPEC-7 §3.2, DS0 incr 16): elect n1 (term 1, leader n1), partition n0
    # off, then `propose n1 x b` — the leader commits to the reachable majority and queues
    # async catch-up to the stranded n0; heal + advance delivers it, so it converges and the
    # leader/term metadata appears in the canonical form (omitted until the first election).
    final = _final(["elect n1", "partition n0 | n1 n2", "propose n1 x b", "heal", "advance 5"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 1, "b"),   # minority replica, caught up after heal+advance
            _rep("x", "n1", 1, "b"),   # leader: synchronous majority write
            _rep("x", "n2", 1, "b"),   # majority side: synchronous
            *_boot_y(),
        ],
        "log": [],  # consensus ops (elect/propose) are protocol-layer; they append no causal event
        "inflight": [],  # the async catch-up to n0 was delivered on advance
        "partitions": [["n0", "n1", "n2"]],
        "down": [],
        "clock": 5,
        "next_event_id": 0,
        "next_msg_id": 1,
        "last_result": ["advanced", "1"],
        "term": 1,        # one election happened
        "leader": "n1",   # ...installing n1 as the cluster leader
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


def test_golden_lost_update_admitted_under_read_committed_forbidden_under_snapshot():
    # The canonical lost-update scenario (DS0 increment 9): A and B both read x at version 0, then
    # both write x (a read-modify-write). Under read_committed (no commit-time validation) both
    # commit and B's write overwrites A's — A's update is silently lost. Under snapshot (and
    # serializable) B's write-write validation sees x bumped by A, so B aborts — no update lost.
    lost = ["begin n0 A", "begin n0 B", "tget n0 A x", "tget n0 B x",
            "tput n0 A x a", "tput n0 B x b", "commit n0 A", "commit n0 B"]

    def outcomes(isolation: str) -> tuple[list[str], str]:
        config = DistConfig(name="golden-iso", nodes=("n0", "n1", "n2"), objects=("x", "y"),
                            txn_isolation=isolation)
        oracle = ReferenceDistOracle(config)
        state = DistributedState.initial(config)
        statuses: list[str] = []
        for cmd in lost:
            r = oracle.step(state, parse_dist_action(cmd))
            statuses.append(r.status)
            state = r.state
        return statuses[-2:], state.replicas[("x", "n0")].value

    rc_commits, rc_x = outcomes("read_committed")
    assert rc_commits == ["committed", "committed"]  # both commit — lost update
    assert rc_x == "b"  # only B's (later) write survives; A's "a" is lost

    snap_commits, snap_x = outcomes("snapshot")
    assert snap_commits == ["committed", "conflict"]  # B aborts on the write-write conflict
    assert snap_x == "a"  # A's write preserved — no update lost


def test_golden_elle_recovers_lost_update_as_a_cycle_black_box():
    # The Elle (DS3 incr 2) counterpart of the lost-update golden: from the observable history alone
    # (no oracle, no cluster state) the checker reconstructs Adya's DSG and reports the cycle the
    # read_committed level admits. Lost update's cycle carries both a ww edge (same-key overwrite)
    # and an rw anti-dependency, distinguishing it from write skew's pure rw cycle.
    from verisim.distoracle.elle import TxnObservation, check_serializable

    # Both txns read x@0; A installs x@1, B installs x@2 (both committed — the anomaly).
    lost_update_history = [
        TxnObservation("A", reads=(("x", 0),), writes=(("x", 1),)),
        TxnObservation("B", reads=(("x", 0),), writes=(("x", 2),)),
    ]
    report = check_serializable(lost_update_history)
    assert not report.serializable
    assert report.anomaly == "G2"
    assert set(report.cycle) == {"A", "B"}
    assert set(report.cycle_kinds) == {"rw", "ww"}  # overwrite (ww) + stale read (rw)

    # Under snapshot, B aborts, so the history is the single committed txn A — acyclic.
    serializable_history = [TxnObservation("A", reads=(("x", 0),), writes=(("x", 1),))]
    assert check_serializable(serializable_history).serializable


def test_golden_dirty_read_admitted_under_read_uncommitted_forbidden_under_read_committed():
    # The canonical dirty-read scenario (DS0 increment 10): A writes x (uncommitted), B reads x,
    # then A aborts. Under read_uncommitted B observes A's uncommitted value (the dirty read) — a
    # value that, after A's rollback, never committed. Under read_committed (and every stronger
    # level) the MVCC tget gives B only the committed boot value: no dirty read.
    script = ["begin n0 A", "begin n0 B", "tput n0 A x b", "tget n0 B x",
              "abort n0 A", "commit n0 B"]

    def b_read(isolation: str) -> tuple[str, str]:
        config = DistConfig(name="golden-ru", nodes=("n0", "n1", "n2"), objects=("x", "y"),
                            txn_isolation=isolation)
        oracle = ReferenceDistOracle(config)
        state = DistributedState.initial(config)
        observed = ""
        for cmd in script:
            r = oracle.step(state, parse_dist_action(cmd))
            if cmd == "tget n0 B x":
                observed = r.value
            state = r.state
        return observed, state.replicas[("x", "n0")].value

    ru_read, ru_final = b_read("read_uncommitted")
    assert ru_read == "b"  # B observed A's uncommitted write — the dirty read
    assert ru_final == "nil"  # ...but A aborted, so the read value never committed (the anomaly)

    rc_read, rc_final = b_read("read_committed")
    assert rc_read == "nil"  # committed data only — no dirty read
    assert rc_final == "nil"


def test_golden_elle_recovers_dirty_read_from_values_alone():
    # The value-oracle (DS3 incr 3) counterpart of the dirty-read golden: from the client-visible
    # list-append history alone — no oracle, no cluster state — Elle reports the dirty read. The
    # aborted writer A contributes NO committed append; B's observed read of A's value becomes a
    # list-read of a value no committed txn appended, which recover_versions flags as `dirty-read`
    # (Adya G1a) before any DSG cycle search.
    from verisim.distoracle.elle import AppendObservation, check_serializable_appends

    # read_uncommitted: B read "b" (A's uncommitted value); A aborted so it appended nothing.
    dirty = [
        AppendObservation("A", appends=(), list_reads=()),
        AppendObservation("B", appends=(), list_reads=(("x", ("b",)),)),
    ]
    report = check_serializable_appends(dirty)
    assert not report.serializable
    assert report.anomaly == "dirty-read"

    # read_committed: B read the committed empty log; no uncommitted value observed — serializable.
    clean = [
        AppendObservation("A", appends=(), list_reads=()),
        AppendObservation("B", appends=(), list_reads=(("x", ()),)),
    ]
    assert check_serializable_appends(clean).serializable


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


def test_golden_causal_holds_effect_until_cause_arrives():
    # The causal-consistency golden (SPEC-7 §3.4, DS0 incr 5): a write that causally depends on a
    # blocked one is held, so no replica sees the effect before its cause — where `eventual` admits
    # exactly that anomaly. The scenario routes y to n2 while x is still partitioned away from n2.
    causal_cfg = DistConfig(
        name="golden-causal", nodes=("n0", "n1", "n2"), objects=("x", "y"),
        consistency_model="causal",
    )
    causal = ReferenceDistOracle(causal_cfg)
    script = ["put n0 x a", "partition n0 n1 | n2", "advance 1",
              "put n1 y b", "partition n0 | n1 n2", "advance 1"]

    s_ev = DistributedState.initial(CONFIG)  # CONFIG is eventual
    s_ca = DistributedState.initial(causal_cfg)
    for cmd in script:
        s_ev = ORACLE.step(s_ev, parse_dist_action(cmd)).state
        s_ca = causal.step(s_ca, parse_dist_action(cmd)).state

    # eventual: n2 adopted the effect (y=b) but not its cause (x=nil) — the anomaly
    assert s_ev.replicas[("y", "n2")].value == "b"
    assert s_ev.replicas[("x", "n2")].value == "nil"
    # causal: the y message is held (deps={x@1} unmet at n2), so n2 sees neither yet — no anomaly
    assert s_ca.replicas[("y", "n2")].value == "nil"
    assert s_ca.replicas[("x", "n2")].value == "nil"
    held = [m for m in s_ca.inflight.values() if m.object_id == "y" and m.dst == "n2"]
    assert len(held) == 1 and held[0].deps == (("x", 1),)


def test_golden_partial_observation_crash_equals_partition_from_one_vantage():
    # The partial-observation golden (SPEC-7 §5.4): from a single external vantage, a crashed node
    # and a node partitioned away project to byte-identical Observations — the failure-detector
    # indistinguishability — yet a paired vantage that reaches the node's side separates them.
    from verisim.dist.observe import observe

    base = DistributedState.initial(CONFIG)
    for cmd in ["put n0 x b", "advance 2"]:
        base = ORACLE.step(base, parse_dist_action(cmd)).state
    crashed = ORACLE.step(base, parse_dist_action("crash n2")).state
    partitioned = ORACLE.step(base, parse_dist_action("partition n0 n1 | n2")).state

    # the committed projection an external client at n0 obtains in *both* worlds (n2 dark)
    expected = {
        "vantage": ["n0"],
        "reachable": ["n0", "n1"],
        "unreachable": ["n2"],
        "replicas": [_rep("x", "n0", 1, "b"), _rep("x", "n1", 1, "b"),
                     _rep("y", "n0", 0, "nil"), _rep("y", "n1", 0, "nil")],
        "clock": 2,
    }

    def _obs_dict(state: DistributedState, vantage: tuple[str, ...]) -> dict[str, object]:
        o = observe(state, vantage)
        return {
            "vantage": sorted(o.vantage),
            "reachable": sorted(o.reachable),
            "unreachable": sorted(o.unreachable),
            "replicas": [_rep(obj, node, ver, val) for obj, node, ver, val in sorted(o.replicas)],
            "clock": o.clock,
        }

    assert _obs_dict(crashed, ("n0",)) == expected
    assert _obs_dict(partitioned, ("n0",)) == expected  # indistinguishable from one vantage
    # the paired vantage {n0, n2} sees n2's live replica only in the partition world
    assert "n2" in observe(partitioned, ("n0", "n2")).reachable
    assert "n2" in observe(crashed, ("n0", "n2")).unreachable


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


def test_golden_quorum_commits_on_majority_side_rejects_on_minority():
    # The Raft-subset consensus golden (SPEC-7 §3.4, DS0 incr 7): with a strict majority of 2 (of
    # 3), a write from the 2-node side commits synchronously to that majority and enqueues an async
    # catch-up to the stale minority — where linearizable (above) rejects the partitioned write.
    final = _final_quorum(["put n0 x b", "partition n0 n1 | n2", "put n0 x c"])
    assert final == {
        "replicas": [
            _rep("x", "n0", 2, "c"),   # majority side: synchronously committed at version 2
            _rep("x", "n1", 2, "c"),   # majority side: synchronously committed
            _rep("x", "n2", 1, "b"),   # minority: stale at the pre-partition value (catch-up due)
            *_boot_y(),
        ],
        "log": [
            {"id": 0, "node": "n0", "op": "put n0 x b", "clock": 0, "happens_before": []},
            {"id": 1, "node": "n0", "op": "put n0 x c", "clock": 0, "happens_before": [0]},
        ],
        # the catch-up to the partitioned n2 is in flight (no `deps`: quorum is unordered delivery)
        "inflight": [
            {"id": 0, "src": "n0", "dst": "n2", "object_id": "x", "version": 2, "value": "c",
             "deliver_after": 1}
        ],
        "partitions": [["n0", "n1"], ["n2"]],
        "down": [],
        "clock": 0,
        "next_event_id": 2,
        "next_msg_id": 1,
        "last_result": ["ok", "c"],  # committed on the majority side (vs linearizable's reject)
    }
    # and a write from the 1-node minority side IS rejected (no majority reachable)
    rejected = QUORUM_ORACLE.step(
        DistributedState.initial(QUORUM_CONFIG), parse_dist_action("partition n0 n1 | n2")
    ).state
    assert QUORUM_ORACLE.step(rejected, parse_dist_action("put n2 x z")).status == "unavailable"


def test_golden_2pl_wound_wait_older_txn_wins():
    # The 2PL concurrency-control golden (SPEC-7 §3.2, DS0 incr 8): A (older) and B (younger) both
    # write x. B holds the X lock first, then A requests it — wound-wait lets the *older* A preempt
    # (wound) B, so A commits its value and B is gone. The lock table is held mid-txn, then freed.
    state = DistributedState.initial(TPL_CONFIG)
    statuses = []
    for cmd in ["begin n0 A", "begin n0 B", "tput n0 B x b", "tput n0 A x a",
                "commit n0 B", "commit n0 A"]:
        r = TPL_ORACLE.step(state, parse_dist_action(cmd))
        statuses.append(r.status)
        state = r.state
    # B holds X first (ok), A (older) wounds B and acquires (ok), B's commit finds it gone (no_txn),
    # A commits — the older transaction wins the wound-wait race deterministically.
    assert statuses == ["ok", "ok", "ok", "ok", "no_txn", "committed"]

    final = to_canonical(state)
    assert final["replicas"][0] == _rep("x", "n0", 1, "a")  # A's value committed
    assert "locks" not in final  # every lock released after the commit (the shrinking phase)
    assert "txns" not in final   # no open transactions
    assert final["last_result"] == ["committed", ""]

    # Mid-transaction, the X lock IS held (additive lock-table state, omitted only when empty).
    held = DistributedState.initial(TPL_CONFIG)
    for cmd in ["begin n0 A", "tput n0 A x a"]:
        held = TPL_ORACLE.step(held, parse_dist_action(cmd)).state
    assert to_canonical(held)["locks"] == {"x": [["A", "X"]]}
