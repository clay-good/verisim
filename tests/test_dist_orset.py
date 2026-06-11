"""DS0 increment 30 — the CRDT OR-Set: ``sadd``/``srem``/``smembers`` + convergence.

A *state-based* add-wins observed-remove set (SPEC-7 §3.2), the canonical *interesting* CRDT — the
one a naive replicated set gets wrong. Each ``sadd n key elem`` tags the element with a **unique
dot** ``(owner=n, seq)`` and stores it in ``n``'s observed add-set; ``srem n key elem`` tombstones
only the dots ``n`` has *observed*; ``smembers`` is the elements with a non-tombstoned dot. The join
is **set union** of both halves (commutative, associative, idempotent). The dot mechanism buys
**add-wins** (a concurrent add survives a concurrent remove) and **re-addability** (a removed item
can be added again) — the two properties an element-level 2P-Set lacks. Tier-A ≡ Tier-B, purely
additive (two omitted-when-empty ``orset_adds``/``orset_tombs`` + ``ORSetAdd``/``ORSetTomb``
edits).
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
    return DistConfig(name="or", nodes=nodes, objects=("s",), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


def _members(value: str) -> set[str]:
    inner = value.strip("{}")
    return set(inner.split(",")) if inner else set()


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_orset_ops() -> None:
    assert parse_dist_action("sadd n0 s x").args == ("n0", "s", "x")
    assert parse_dist_action("srem n0 s x").args == ("n0", "s", "x")
    assert parse_dist_action("smembers n0 s").args == ("n0", "s")
    assert {"sadd", "srem", "smembers"} <= CLIENT_OPS


@pytest.mark.parametrize("bad", ["sadd n0 s", "srem n0 s", "smembers n0 s x", "sadd n0 s x y"])
def test_grammar_rejects_bad(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- node-local add / remove / read -------------------------------------------------------------

def test_sadd_then_smembers_reads_the_set() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["sadd n0 s x", "sadd n0 s y", "sadd n0 s z"])
    assert _members(oracle.step(s, parse_dist_action("smembers n0 s")).value) == {"x", "y", "z"}


def test_srem_is_observed_remove() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["sadd n0 s x", "sadd n0 s y", "srem n0 s x"])
    assert _members(oracle.step(s, parse_dist_action("smembers n0 s")).value) == {"y"}


def test_removed_element_is_re_addable() -> None:
    # The property an element-level 2P-Set lacks: a removed element can be added again (a fresh
    # dot).
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["sadd n0 s x", "srem n0 s x", "sadd n0 s x"])
    assert _members(oracle.step(s, parse_dist_action("smembers n0 s")).value) == {"x"}


def test_sadd_is_always_available_even_partitioned_alone() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("sadd n0 s z"))
    assert (r.status, r.value) == ("ok", "z")


def test_crashed_node_sadd_is_unavailable() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("sadd n0 s x")).status == \
        "unavailable"


def test_unique_dots_per_node() -> None:
    # Each node is the single writer of its own dots; re-adding after a remove uses a fresh seq.
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["sadd n0 s x", "srem n0 s x", "sadd n0 s x"])
    seqs = sorted(seq for (e, o, seq) in s.orset_adds[("s", "n0")] if e == "x")
    assert seqs == [1, 2]  # the re-add did not reuse the removed dot's seq


# --- add-wins + convergence (the resolution a 2P-Set lacks) -------------------------------------

def test_add_wins_over_concurrent_remove() -> None:
    # x present cluster-wide; partition; n0 re-adds (fresh dot) while n3 removes the dot it saw.
    # After heal+gossip the element SURVIVES (a 2P-Set would drop it — remove-wins).
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["sadd n0 s x", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4", "partition n0 n1 n2 | n3 n4", "sadd n0 s x", "srem n3 s x",
              "heal", "gossip n0 n3"])
    assert "x" in _members(oracle.step(s, parse_dist_action("smembers n0 s")).value)
    assert "x" in _members(oracle.step(s, parse_dist_action("smembers n3 s")).value)


def test_anti_entropy_converges_all_nodes() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "sadd n0 s a", "sadd n3 s b", "heal",
              "anti_entropy n0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4"])
    for nd in config.nodes:
        assert _members(oracle.step(s, parse_dist_action(f"smembers {nd} s")).value) == {"a", "b"}


def test_gossip_union_join_is_idempotent() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["sadd n0 s a", "sadd n1 s b", "gossip n0 n1", "gossip n0 n1"])
    assert _members(oracle.step(s, parse_dist_action("smembers n0 s")).value) == {"a", "b"}


# --- delta + serialization ----------------------------------------------------------------------

def test_canonical_form_omits_orset_until_first_op() -> None:
    config = _config(n=3)
    s = DistributedState.initial(config)
    assert "orset_adds" not in to_canonical(s)  # never-set cluster: purely additive
    assert "orset_tombs" not in to_canonical(s)
    added = _run(ReferenceDistOracle(config), config, ["sadd n0 s x"])
    assert to_canonical(added)["orset_adds"][0] == {
        "key": "s", "holder": "n0", "elem": "x", "owner": "n0", "seq": 1
    }
    removed = _run(ReferenceDistOracle(config), config, ["sadd n0 s x", "srem n0 s x"])
    assert to_canonical(removed)["orset_tombs"][0] == {
        "key": "s", "holder": "n0", "owner": "n0", "seq": 1
    }
    assert from_canonical(to_canonical(removed)) == removed
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))  # empty form unchanged


def test_orset_in_cluster_view() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["sadd n0 s x"])
    assert "'orset_adds'" in cluster_view(s)


# --- the metamorphic tier -----------------------------------------------------------------------

def test_metamorphic_tier_refutes_a_phantom_dot() -> None:
    from verisim.dist.delta import ORSetAdd, apply
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["sadd n0 s x"])
    bogus = apply(s, [ORSetAdd("s", "n0", "x", "ghost", 1)])  # an owner that is not a cluster node
    verdict = TieredOracle(config).check("metamorphic", s, parse_dist_action("sadd n0 s x"), bogus)
    assert verdict.refuted and "orset" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_an_orset_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "sadd n0 s x", "sadd n1 s y", "smembers n0 s",
        "anti_entropy n1", "anti_entropy n2", "anti_entropy n3", "anti_entropy n4",
        "partition n0 n1 n2 | n3 n4",
        "sadd n0 s x", "srem n3 s x",               # concurrent add/remove, both ack (AP)
        "smembers n0 s", "smembers n3 s",           # local views diverge, nothing lost
        "heal", "gossip n0 n3", "anti_entropy n1", "smembers n4 s",  # converge to the union
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
