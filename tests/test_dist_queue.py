"""DS0 increment 21 — the distributed FIFO queue: ``enqueue`` / ``dequeue``.

A second client data type beside the KV store (SPEC-7 §3.2). The property under test: delivery
semantics follow the consistency model — ``eventual`` admits duplicate (at-least-once) delivery
under partition, while ``linearizable``/``quorum`` gate availability for exactly-once. Queue
replicas are omitted from the canonical form until the first ``enqueue`` (purely additive).
"""

import pytest

from verisim.dist import DistConfig, DistributedState, apply, parse_dist_action
from verisim.dist.action import CLIENT_OPS, QUEUE_OPS, DistParseError
from verisim.dist.delta import QueueSet
from verisim.dist.serialize import from_canonical, state_hash, to_canonical
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


def _config(model: str = "eventual") -> DistConfig:
    return DistConfig(name="queue", nodes=("n0", "n1", "n2"), objects=("x",),
                      values=("a", "b", "c", "d"), replication_factor=3, consistency_model=model)


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_enqueue_and_dequeue() -> None:
    assert parse_dist_action("enqueue n0 q a").args == ("n0", "q", "a")
    assert parse_dist_action("dequeue n0 q").args == ("n0", "q")
    assert {"enqueue", "dequeue"} == QUEUE_OPS
    assert QUEUE_OPS <= CLIENT_OPS


@pytest.mark.parametrize("bad", ["enqueue n0 q", "enqueue n0 q a b", "dequeue n0",
                                 "dequeue n0 q x"])
def test_grammar_rejects_bad_arity(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- FIFO semantics on the connected path -------------------------------------------------------

def test_enqueue_then_dequeue_is_fifo_and_exactly_once() -> None:
    config = _config()
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["enqueue n0 q a", "enqueue n0 q b", "enqueue n0 q c"])
    # the queue replicates to every node (full connectivity)
    assert all(s.queues[("q", n)] == ("a", "b", "c") for n in config.nodes)
    got = []
    for _ in range(3):
        r = oracle.step(s, parse_dist_action("dequeue n0 q"))
        s, r_status, r_value = r.state, r.status, r.value
        got.append((r_status, r_value))
    assert got == [("dequeued", "a"), ("dequeued", "b"), ("dequeued", "c")]  # FIFO
    assert oracle.step(s, parse_dist_action("dequeue n0 q")).status == "empty"  # drained


def test_dequeue_empty_queue_is_empty() -> None:
    config = _config()
    r = ReferenceDistOracle(config).step(DistributedState.initial(config),
                                         parse_dist_action("dequeue n0 q"))
    assert r.status == "empty"


def test_enqueue_on_crashed_node_is_unavailable() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("enqueue n0 q a")).status \
        == "unavailable"


# --- delivery semantics follow the consistency model --------------------------------------------

def test_eventual_admits_duplicate_delivery_under_partition() -> None:
    # The headline: an item enqueued before a partition is dequeued on BOTH sides — delivered twice.
    config = _config("eventual")
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["enqueue n0 q a", "partition n0 | n1 n2"])
    r_min = oracle.step(s, parse_dist_action("dequeue n0 q"))
    r_maj = oracle.step(s, parse_dist_action("dequeue n1 q"))
    assert (r_min.status, r_min.value) == ("dequeued", "a")  # minority side delivers it...
    assert (r_maj.status, r_maj.value) == ("dequeued", "a")  # ...and so does the majority side


def test_linearizable_blocks_the_partitioned_dequeue() -> None:
    config = _config("linearizable")
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["enqueue n0 q a", "partition n0 | n1 n2"])
    # neither side has all-replica reachability -> both unavailable (exactly-once preserved)
    assert oracle.step(s, parse_dist_action("dequeue n0 q")).status == "unavailable"
    assert oracle.step(s, parse_dist_action("dequeue n1 q")).status == "unavailable"


def test_quorum_delivers_exactly_once_on_the_majority_side() -> None:
    config = _config("quorum")
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["enqueue n0 q a", "partition n0 | n1 n2"])
    assert oracle.step(s, parse_dist_action("dequeue n0 q")).status == "unavailable"  # minority
    r_maj = oracle.step(s, parse_dist_action("dequeue n1 q"))
    assert (r_maj.status, r_maj.value) == ("dequeued", "a")  # majority side: exactly once


def test_linearizable_enqueue_under_partition_is_unavailable() -> None:
    config = _config("linearizable")
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("enqueue n0 q a")).status \
        == "unavailable"


# --- delta + serialization ----------------------------------------------------------------------

def test_queue_set_applies_and_round_trips() -> None:
    config = _config()
    s = apply(DistributedState.initial(config), [QueueSet("q", "n0", ("a", "b"))])
    assert s.queues[("q", "n0")] == ("a", "b")
    assert from_canonical(to_canonical(s)) == s


def test_empty_queue_leaves_no_canonical_residue() -> None:
    config = _config()
    s = apply(DistributedState.initial(config), [QueueSet("q", "n0", ())])
    assert "queues" not in to_canonical(s)  # an empty queue is indistinguishable from never-used


def test_canonical_form_omits_queues_until_first_enqueue() -> None:
    config = _config()
    s = DistributedState.initial(config)
    assert "queues" not in to_canonical(s)  # KV-only cluster: purely additive
    enq = ReferenceDistOracle(config).step(s, parse_dist_action("enqueue n0 q a")).state
    assert any(e["queue"] == "q" for e in to_canonical(enq)["queues"])
    # and the KV-only hash is unchanged by the (omitted) empty queue field
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_queue_trajectory() -> None:
    config = _config("eventual")
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "enqueue n0 q a", "enqueue n1 q b",          # two enqueues from different coordinators
        "dequeue n0 q",                               # FIFO head
        "partition n0 | n1 n2", "dequeue n0 q", "dequeue n1 q",  # duplicate delivery, partitioned
        "heal", "enqueue n2 q c", "dequeue n2 q",
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
