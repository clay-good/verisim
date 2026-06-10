"""Pairwise gossip — the `gossip` protocol op (SPEC-7 §4, DS0 increment 15).

Pins the **pairwise, bidirectional** anti-entropy primitive — the Merkle-tree sync real
eventually-consistent stores (Dynamo, Cassandra) run between *pairs* of nodes. `gossip a b`
reconciles **both** `a` and `b` to the per-object winner of their two replicas, vs `anti_entropy`'s
one-directional pull-to-one-node. It reuses the `ReplicaWrite` edit (no new state field) and is a
pure coordinator-level reconciliation (reads both replicas directly, no in-flight message), so it
composes with every consistency model and Tier-A ≡ Tier-B bit-for-bit. Dependency-free, GPU-free.
"""

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import PROTOCOL_OPS, DistParseError
from verisim.dist.delta import ReplicaWrite, apply
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle

CONFIG = DistConfig(name="gossip-test", nodes=("n0", "n1", "n2"), objects=("x", "y"))


def _run(
    cmds: list[str], cfg: DistConfig = CONFIG
) -> tuple[DistributedState, list[tuple[str, str]]]:
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    results: list[tuple[str, str]] = []
    for cmd in cmds:
        r = oracle.step(state, parse_dist_action(cmd))
        results.append((r.status, r.value))
        state = r.state
    return state, results


def _v(state: DistributedState, key: str, node: str) -> tuple[int, str]:
    r = state.replicas[(key, node)]
    return (r.version, r.value)


# --- the grammar ---------------------------------------------------------------------------------


def test_gossip_parses_with_two_args_and_is_a_protocol_op():
    action = parse_dist_action("gossip n0 n1")
    assert action.name == "gossip"
    assert action.args == ("n0", "n1")
    assert "gossip" in PROTOCOL_OPS


def test_gossip_requires_exactly_two_args():
    for bad in ("gossip", "gossip n0", "gossip n0 n1 n2"):
        try:
            parse_dist_action(bad)
        except DistParseError:
            continue
        raise AssertionError(f"expected {bad!r} to fail parsing")


# --- the semantics -------------------------------------------------------------------------------


def test_gossip_reconciles_a_dropped_write_without_a_message():
    state, results = _run(["put n0 x b", "drop n0 n1", "advance 2", "gossip n0 n1"])
    assert results[3] == ("gossiped", "1")  # one replica (n1) moved
    assert _v(state, "x", "n1") == (1, "b")  # n1 pulled up to the winner, no in-flight message
    assert not state.inflight


def test_gossip_is_bidirectional_either_node_can_be_the_one_behind():
    # b ahead (a stale): gossip a b moves the *first* arg up — anti_entropy(a) would do this too,
    # but gossip does it symmetrically regardless of which node is behind.
    a_behind, _ = _run(["put n1 x a", "drop n1 n0", "advance 2", "gossip n0 n1"])
    assert _v(a_behind, "x", "n0") == (1, "a")  # n0 (first arg) moved up to n1's value
    # a ahead (b stale): the same op moves the *second* arg up.
    b_behind, _ = _run(["put n0 x b", "drop n0 n1", "advance 2", "gossip n0 n1"])
    assert _v(b_behind, "x", "n1") == (1, "b")  # n1 (second arg) moved up to n0's value


def test_one_gossip_fills_complementary_holes_in_both_nodes():
    """The bidirectional headline: a stale on x, b stale on y — one gossip fixes both."""
    # n0 writes x (dropped to n1); n1 writes y (dropped to n0).
    state, _ = _run([
        "put n0 x b", "drop n0 n1", "put n1 y c", "drop n1 n0", "advance 2", "gossip n0 n1",
    ])
    assert _v(state, "x", "n0") == (1, "b") and _v(state, "y", "n0") == (1, "c")  # n0 fully synced
    assert _v(state, "x", "n1") == (1, "b") and _v(state, "y", "n1") == (1, "c")  # n1 fully synced


def test_gossip_needs_a_live_link():
    # crashed endpoint
    _, results = _run(["crash n1", "gossip n0 n1"])
    assert results[1] == ("unavailable", "")
    # partitioned apart
    _, results = _run(["partition n0 | n1 n2", "gossip n0 n1"])
    assert results[1] == ("unavailable", "")


def test_gossip_of_already_synced_pair_is_a_no_op():
    state, results = _run(["put n0 x b", "advance 2", "gossip n0 n1"])  # both already have b
    assert results[2] == ("gossiped", "0")
    assert _v(state, "x", "n0") == _v(state, "x", "n1") == (1, "b")


# --- additive: apply == oracle, no new edit type, round-trip ------------------------------------


def test_apply_equals_oracle_on_a_gossip_step():
    oracle = ReferenceDistOracle(CONFIG)
    state = oracle.step(DistributedState.initial(CONFIG), parse_dist_action("put n0 x b")).state
    state = oracle.step(state, parse_dist_action("drop n0 n1")).state
    state = oracle.step(state, parse_dist_action("advance 2")).state
    r = oracle.step(state, parse_dist_action("gossip n0 n1"))
    assert apply(state, r.delta) == r.state
    assert any(isinstance(e, ReplicaWrite) for e in r.delta)  # reuses ReplicaWrite (no new edit)


def test_gossip_adds_no_state_field_so_canonical_form_round_trips():
    state, _ = _run(["put n0 x b", "drop n0 n1", "advance 2", "gossip n0 n1"])
    canon = to_canonical(state)
    assert from_canonical(canon) == state
    assert set(canon) == set(to_canonical(DistributedState.initial(CONFIG)))  # no new top-level key


# --- Tier-B agreement ----------------------------------------------------------------------------


def test_gossip_tier_b_agrees_bit_for_bit():
    """Pairwise gossip is a coordinator-level reconciliation, so Tier-B computes the byte-identical
    sync and reproduces the bidirectional / epidemic convergence on its own scheduler."""
    ref, sysb = ReferenceDistOracle(CONFIG), SystemDistOracle(CONFIG)
    scripts = [
        ["put n0 x b", "drop n0 n1", "advance 2", "gossip n0 n1"],
        ["put n1 x a", "put n0 x b", "gossip n0 n1", "advance 3"],
        ["put n0 x b", "drop n0 n1", "put n1 y c", "drop n1 n0", "advance 2", "gossip n0 n1"],
        ["put n0 x b", "drop n0 n1", "drop n0 n2", "advance 2", "gossip n0 n1", "gossip n1 n2"],
        ["put n0 x b", "partition n0 n1 | n2", "advance 2", "gossip n0 n2"],  # cross-cut: unavail
    ]
    for script in scripts:
        sa = sb = DistributedState.initial(CONFIG)
        for cmd in script:
            action = parse_dist_action(cmd)
            ra, rb = ref.step(sa, action), sysb.step(sb, action)
            assert cluster_view(ra.state) == cluster_view(rb.state), (cmd, script)
            assert ra.status == rb.status and ra.value == rb.value
            sa, sb = ra.state, rb.state
