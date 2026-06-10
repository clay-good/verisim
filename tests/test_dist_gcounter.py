"""DS0 increment 28 — the CRDT G-counter: ``cincr``/``cget`` + convergence.

A *state-based* grow-only counter (SPEC-7 §3.2), the loss-free, always-available resolution to
``incr``'s LWW lost-update negative (ED34). Each node keeps a per-owner vector of monotone
sub-counts; ``cincr n key`` bumps **only ``n``'s own** sub-count (node-local, always available), and
the CRDT join is the per-(key, owner) **max**, applied by ``anti_entropy``/``gossip`` — commutative,
associative, idempotent. Concurrent ``cincr``s never conflict and never lose an update; the counter
converges to the exact total. Tier-A ≡ Tier-B, purely additive (one omitted-when-empty ``gcounters``
map + one ``GCounterSet`` edit).
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
    return DistConfig(name="gc", nodes=nodes, objects=("c",), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_cincr_cget() -> None:
    assert parse_dist_action("cincr n0 c").args == ("n0", "c")
    assert parse_dist_action("cget n0 c").args == ("n0", "c")
    assert {"cincr", "cget"} <= CLIENT_OPS


@pytest.mark.parametrize("bad", ["cincr n0", "cget n0 c v", "cincr n0 c v"])
def test_grammar_rejects_bad(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- node-local counting ------------------------------------------------------------------------

def test_cincr_counts_and_cget_reads_the_sum() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cincr n0 c", "cincr n0 c", "cincr n0 c"])
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "3"


def test_cincr_bumps_only_its_own_subcount() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cincr n0 c", "cincr n1 c"])
    # n0's own copy holds only n0's sub-count (1) until a merge brings n1's in.
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "1"
    assert s.gcounters == {("c", "n0", "n0"): 1, ("c", "n1", "n1"): 1}


def test_cincr_is_always_available_even_partitioned_alone() -> None:
    # The AP property: a partitioned-alone node still counts (LWW incr under quorum/lin would fail).
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("cincr n0 c"))
    assert (r.status, r.value) == ("ok", "1")


def test_crashed_node_cincr_is_unavailable() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("cincr n0 c")).status == \
        "unavailable"


# --- no lost update + convergence (the resolution to ED34) --------------------------------------

def test_no_lost_update_under_partition() -> None:
    # The direct contrast with ED34: two concurrent cincrs on opposite sides BOTH count, and after
    # heal+gossip the total is the sum (where ED34's LWW counter lost one).
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "cincr n0 c", "cincr n0 c", "cincr n3 c",
              "heal", "gossip n0 n3"])
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "3"  # 2 (n0) + 1 (n3)


def test_anti_entropy_converges_all_nodes() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "cincr n0 c", "cincr n3 c", "heal",
              "anti_entropy n0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4"])
    for nd in config.nodes:
        assert oracle.step(s, parse_dist_action(f"cget {nd} c")).value == "2"


def test_gossip_join_is_idempotent() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cincr n0 c", "cincr n1 c", "gossip n0 n1", "gossip n0 n1"])
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "2"  # stable under re-merge


# --- delta + serialization ----------------------------------------------------------------------

def test_canonical_form_omits_gcounters_until_first_cincr() -> None:
    config = _config(n=3)
    s = DistributedState.initial(config)
    assert "gcounters" not in to_canonical(s)  # never-counted cluster: purely additive
    counted = _run(ReferenceDistOracle(config), config, ["cincr n0 c"])
    assert to_canonical(counted)["gcounters"][0] == {
        "key": "c", "holder": "n0", "owner": "n0", "count": 1
    }
    assert from_canonical(to_canonical(counted)) == counted
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))  # empty form unchanged


def test_gcounter_in_cluster_view() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["cincr n0 c"])
    assert "'gcounters'" in cluster_view(s)


# --- the metamorphic tier -----------------------------------------------------------------------

def test_metamorphic_tier_refutes_a_negative_subcount() -> None:
    from verisim.dist.delta import GCounterSet, apply
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cincr n0 c"])
    bogus = apply(s, [GCounterSet("c", "n0", "n0", -1)])  # a negative sub-count is impossible
    verdict = TieredOracle(config).check("metamorphic", s, parse_dist_action("cincr n0 c"), bogus)
    assert verdict.refuted and "gcounter" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_gcounter_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "cincr n0 c", "cincr n0 c", "cget n0 c",
        "partition n0 n1 n2 | n3 n4",
        "cincr n0 c", "cincr n3 c",                 # concurrent, both ack (AP)
        "cget n0 c", "cget n3 c",                   # local views diverge, nothing lost
        "heal", "gossip n0 n3", "anti_entropy n1", "cget n0 c",  # converge to the full total
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
