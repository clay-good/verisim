"""DS0 increment 26 — the tombstone delete: ``delete`` + the resurrection problem.

`delete node key` is the fundamental KV remove (SPEC-7 §3.2). It is a **versioned write of a
tombstone** (it reuses the `put` replication path with the `TOMBSTONE` value), not a removal of the
replica, so last-writer-wins orders the delete against concurrent/stale writes by version — which is
what prevents the **resurrection problem** (a deleted key reappearing because a stale replica's old
value out-versions an absence). A `get` on a tombstoned key reads `deleted`. Tier-A ≡ Tier-B, and
the op is purely additive (the tombstone is just a replica value — no new state field, no new edit).
"""

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import CLIENT_OPS, DistParseError
from verisim.dist.state import TOMBSTONE
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


def _config(n: int = 5, model: str = "linearizable") -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name="del", nodes=nodes, objects=("x",), values=("a", "b"),
                      replication_factor=n, consistency_model=model)


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_delete() -> None:
    assert parse_dist_action("delete n0 x").args == ("n0", "x")
    assert "delete" in CLIENT_OPS


@pytest.mark.parametrize("bad", ["delete n0", "delete n0 x y"])
def test_grammar_rejects_bad_delete(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- tombstone semantics ------------------------------------------------------------------------

def test_delete_tombstones_the_key_and_get_reads_deleted() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["put n0 x a", "delete n0 x"])
    r = oracle.step(s, parse_dist_action("delete n0 x"))  # the delete itself returns ("deleted","")
    s = _run(oracle, config, ["put n0 x a", "delete n0 x"])
    assert oracle.step(s, parse_dist_action("get n0 x")).status == "deleted"
    assert s.replicas[("x", "n0")].value == TOMBSTONE  # the tombstone is the replicated value
    assert r.status == "deleted" and r.value == ""


def test_delete_is_a_versioned_write() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["put n0 x a"])
    put_ver = s.replicas[("x", "n0")].version
    s2 = oracle.step(s, parse_dist_action("delete n0 x")).state
    assert s2.replicas[("x", "n0")].version == put_ver + 1  # the tombstone out-versions the put


def test_newer_put_resurrects_legitimately() -> None:
    # a put after a delete (a strictly higher version) brings the key back — a new write, not a bug.
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["put n0 x a", "delete n0 x", "put n0 x b"])
    r = oracle.step(s, parse_dist_action("get n0 x"))
    assert (r.status, r.value) == ("ok", "b")


def test_delete_of_a_boot_key_tombstones_at_version_1() -> None:
    config = _config(n=3)
    s = ReferenceDistOracle(config).step(DistributedState.initial(config),
                                         parse_dist_action("delete n0 x"))
    assert s.status == "deleted" and s.state.replicas[("x", "n0")].version == 1


def test_delete_respects_the_consistency_model_under_partition() -> None:
    # linearizable: a delete that cannot reach all replicas is rejected (CP), like any write.
    config = _config(n=3, model="linearizable")
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("delete n0 x"))
    assert r.status == "unavailable"


# --- the resurrection problem (eventual) --------------------------------------------------------

def test_partitioned_minority_still_reads_the_deleted_item() -> None:
    # The danger: a delete on the majority side does not reach the partitioned minority, which keeps
    # reading the old value (the deleted item is "still there").
    config = _config(n=5, model="eventual")
    s = _run(ReferenceDistOracle(config), config,
             ["put n0 x a", "advance 5", "partition n0 n1 n2 | n3 n4", "delete n0 x", "advance 5"])
    oracle = ReferenceDistOracle(config)
    assert oracle.step(s, parse_dist_action("get n0 x")).status == "deleted"  # majority: deleted
    assert oracle.step(s, parse_dist_action("get n3 x")).value == "a"  # minority: stale, still here


def test_anti_entropy_converges_to_deleted_no_resurrection() -> None:
    # The repair: after heal, anti_entropy carries the tombstone, whose higher version wins the
    # merge, so the minority converges to deleted rather than resurrecting the stale value.
    config = _config(n=5, model="eventual")
    s = _run(ReferenceDistOracle(config), config,
             ["put n0 x a", "advance 5", "partition n0 n1 n2 | n3 n4", "delete n0 x", "advance 5",
              "heal", "anti_entropy n3"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("get n3 x")).status == "deleted"
    assert s.replicas[("x", "n3")].value == TOMBSTONE  # the tombstone won, not the stale "a"


def test_gossip_also_converges_to_deleted() -> None:
    config = _config(n=5, model="eventual")
    s = _run(ReferenceDistOracle(config), config,
             ["put n0 x a", "advance 5", "partition n0 n1 n2 | n3 n4", "delete n0 x", "advance 5",
              "heal", "gossip n0 n3"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("get n3 x")).status == "deleted"


# --- the metamorphic tier admits the tombstone --------------------------------------------------

def test_metamorphic_tier_admits_a_tombstoned_replica() -> None:
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["put n0 x a"])
    act = parse_dist_action("delete n0 x")
    predicted = oracle.step(s, act).state  # a legal tombstone state
    verdict = TieredOracle(config).check("metamorphic", s, act, predicted)
    assert not verdict.refuted  # the tombstone is a legal value, not out-of-vocab


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_delete_trajectory() -> None:
    config = _config(n=5, model="eventual")
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "put n0 x a", "advance 5",                  # replicate a everywhere
        "partition n0 n1 n2 | n3 n4", "delete n0 x", "advance 5",  # delete majority; minority lags
        "get n3 x",                                  # the minority still reads a
        "heal", "anti_entropy n3", "get n3 x",       # repair -> deleted, no resurrection
        "put n0 x b", "advance 5", "get n3 x",       # a genuinely newer write brings it back
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
