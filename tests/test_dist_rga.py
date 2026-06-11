"""DS0 increment 34 — the CRDT RGA: ``rins``/``rdel``/``rget`` + the ordered sequence.

A *state-based* replicated growable array (SPEC-7 §3.2), the first **ordered** CRDT — a sequence,
the basis of collaborative text. Each element has a unique id ``(seq, owner)`` and a ``parent`` (the
element it was inserted after, or ``RGA_ROOT`` for the head); the visible order is a DFS where sibs
are ordered by id **descending**. ``rins n list i v`` inserts after the i-th visible element (i=0 =
head); ``rdel n list i`` tombstones the i-th visible element (deletes preserve structure); ``rget``
reads the visible values concatenated. The join is **set union** of elements + tombstones; because
the order is a pure function of the element set, concurrent inserts converge to one deterministic
order. Tier-A ≡ Tier-B, purely additive over increment 33 (two omitted-when-empty ``rga_*`` maps +
two edits).
"""

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import CLIENT_OPS, DistParseError
from verisim.dist.serialize import from_canonical, state_hash, to_canonical
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


def _config(n: int = 5) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name="rg", nodes=nodes, objects=("l",), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_rga_ops() -> None:
    assert parse_dist_action("rins n0 l 0 a").args == ("n0", "l", "0", "a")
    assert parse_dist_action("rdel n0 l 1").args == ("n0", "l", "1")
    assert parse_dist_action("rget n0 l").args == ("n0", "l")
    assert {"rins", "rdel", "rget"} <= CLIENT_OPS


@pytest.mark.parametrize("bad", ["rins n0 l 0", "rdel n0 l", "rget n0 l x", "rins n0 l 0 a b"])
def test_grammar_rejects_bad(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- node-local sequence operations -------------------------------------------------------------

def test_sequential_insert_builds_the_string() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["rins n0 l 0 a", "rins n0 l 1 b", "rins n0 l 2 c"])
    assert oracle.step(s, parse_dist_action("rget n0 l")).value == "abc"


def test_insert_at_head_and_middle() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["rins n0 l 0 b", "rins n0 l 0 a", "rins n0 l 2 c"])
    # rins 0 b -> "b"; rins 0 a (head) -> "ab"; rins 2 c (after 2nd 'b') -> "abc".
    assert oracle.step(s, parse_dist_action("rget n0 l")).value == "abc"


def test_delete_tombstones_the_element() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["rins n0 l 0 a", "rins n0 l 1 b", "rins n0 l 2 c", "rdel n0 l 2"])
    assert oracle.step(s, parse_dist_action("rget n0 l")).value == "ac"  # 'b' removed


def test_delete_out_of_range_is_not_found() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["rins n0 l 0 a"])
    assert oracle.step(s, parse_dist_action("rdel n0 l 5")).status == "not_found"


def test_delete_preserves_structure_as_anchor() -> None:
    # delete 'b' then insert after it — the tombstone keeps its position as an anchor for the ins.
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["rins n0 l 0 a", "rins n0 l 1 b", "rdel n0 l 2"])
    # visible is "a"; insert after pos 1 (a) -> "ac" (c anchored after a, before b's tombstone).
    s = oracle.step(s, parse_dist_action("rins n0 l 1 c")).state
    assert oracle.step(s, parse_dist_action("rget n0 l")).value == "ac"


def test_rins_is_always_available_even_partitioned_alone() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("rins n0 l 0 z"))
    assert (r.status, r.value) == ("ok", "z")


def test_crashed_node_rins_is_unavailable() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("rins n0 l 0 a")).status == \
        "unavailable"


# --- the defining property: concurrent inserts converge deterministically -----------------------

def test_concurrent_inserts_converge_to_one_order() -> None:
    # 'a' present cluster-wide; partition; n0 inserts 'b' after 'a', n3 inserts 'c' after 'a'. After
    # heal+gossip BOTH nodes derive the SAME string, both inserts present, no duplication.
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["rins n0 l 0 a", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4", "partition n0 n1 n2 | n3 n4", "rins n0 l 1 b", "rins n3 l 1 c",
              "heal", "gossip n0 n3"])
    v0 = oracle.step(s, parse_dist_action("rget n0 l")).value
    v3 = oracle.step(s, parse_dist_action("rget n3 l")).value
    assert v0 == v3  # the SAME order on both nodes
    assert len(v0) == 3 and set(v0) == {"a", "b", "c"}  # both inserts present, no duplication


def test_anti_entropy_converges_all_nodes() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "rins n0 l 0 x", "rins n3 l 0 y", "heal",
              "anti_entropy n0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4"])
    seqs = {oracle.step(s, parse_dist_action(f"rget {nd} l")).value for nd in config.nodes}
    assert len(seqs) == 1 and len(seqs.pop()) == 2  # all nodes agree, both elements present


def test_union_join_is_idempotent() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 | n2", "rins n0 l 0 x", "rins n2 l 0 y", "heal",
              "gossip n0 n2", "gossip n0 n2"])
    assert len(oracle.step(s, parse_dist_action("rget n0 l")).value) == 2  # stable under re-merge


# --- delta + serialization ----------------------------------------------------------------------

def test_canonical_form_omits_rga_until_first_op() -> None:
    config = _config(n=3)
    s = DistributedState.initial(config)
    assert not any(k.startswith("rga") for k in to_canonical(s))  # purely additive
    written = _run(ReferenceDistOracle(config), config, ["rins n0 l 0 a"])
    assert to_canonical(written)["rga_elems"][0] == {
        "list": "l", "holder": "n0", "seq": 1, "owner": "n0", "value": "a", "pseq": 0, "powner": ""
    }
    deleted = _run(ReferenceDistOracle(config), config, ["rins n0 l 0 a", "rdel n0 l 1"])
    assert to_canonical(deleted)["rga_tombs"][0] == {
        "list": "l", "holder": "n0", "seq": 1, "owner": "n0"
    }
    assert from_canonical(to_canonical(deleted)) == deleted
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))  # empty form unchanged


def test_rga_in_cluster_view() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["rins n0 l 0 a"])
    assert "'rga_elems'" in cluster_view(s)


# --- the metamorphic tier -----------------------------------------------------------------------

def test_metamorphic_tier_refutes_a_phantom_owner() -> None:
    from verisim.dist.delta import RGAInsert, apply
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["rins n0 l 0 a"])
    bogus = apply(s, [RGAInsert("l", "n0", 2, "ghost", "z", 0, "")])  # owner not a cluster node
    act = parse_dist_action("rins n0 l 0 a")
    verdict = TieredOracle(config).check("metamorphic", s, act, bogus)
    assert verdict.refuted and "rga" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_an_rga_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "rins n0 l 0 a", "rins n0 l 1 b", "rget n0 l",
        "anti_entropy n1", "anti_entropy n2", "anti_entropy n3", "anti_entropy n4",
        "partition n0 n1 n2 | n3 n4",
        "rins n0 l 1 X", "rins n3 l 1 Y", "rdel n3 l 2",   # concurrent inserts + a delete
        "rget n0 l", "rget n3 l",                          # local views diverge
        "heal", "gossip n0 n3", "anti_entropy n1", "rget n4 l",  # converge to one order
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
