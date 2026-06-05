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


def _final(cmds: list[str]) -> dict[str, object]:
    state = DistributedState.initial(CONFIG)
    for cmd in cmds:
        state = ORACLE.step(state, parse_dist_action(cmd)).state
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
