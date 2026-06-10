"""DS0 increment 29 — the CRDT PN-counter: ``cdecr`` + the decrementable counter.

A *state-based* PN-counter (SPEC-7 §3.2), the decrementable extension of ED35's grow-only G-counter.
A PN-counter pairs two G-counters, P (the ``cincr`` half) and N (the ``cdecr`` half), and reads
**P − N**. ``cdecr n key`` bumps **only ``n``'s own** N sub-count (node-local, always available),
and the same per-(key, owner) **max** join in ``anti_entropy``/``gossip`` merges *both* halves —
commutative, associative, idempotent. Concurrent ops never conflict and never lose an update; the
counter converges to the exact net, and its value may go **negative** (the property the G-counter
lacks). Tier-A ≡ Tier-B, purely additive over increment 28 (one omitted-when-empty ``ncounters``
map + one ``NCounterSet`` edit).
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
    return DistConfig(name="pn", nodes=nodes, objects=("c",), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_cdecr() -> None:
    assert parse_dist_action("cdecr n0 c").args == ("n0", "c")
    assert "cdecr" in CLIENT_OPS


@pytest.mark.parametrize("bad", ["cdecr n0", "cdecr n0 c v"])
def test_grammar_rejects_bad(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- node-local decrement -----------------------------------------------------------------------

def test_cincr_then_cdecr_nets() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cincr n0 c", "cincr n0 c", "cincr n0 c", "cdecr n0 c"])
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "2"  # 3 up, 1 down


def test_cdecr_goes_below_zero() -> None:
    # The PN-counter property a G-counter lacks: cget may be negative.
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cdecr n0 c", "cdecr n0 c"])
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "-2"


def test_cdecr_bumps_only_its_own_decrement_subcount() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cincr n0 c", "cdecr n1 c"])
    # n0's local view holds only n0's P sub-count (1); n1's decrement is unseen until a merge.
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "1"
    assert s.gcounters == {("c", "n0", "n0"): 1}
    assert s.ncounters == {("c", "n1", "n1"): 1}


def test_cdecr_is_always_available_even_partitioned_alone() -> None:
    # The AP property: a partitioned-alone node still counts down.
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("cdecr n0 c"))
    assert (r.status, r.value) == ("ok", "1")  # reports the node's own new decrement sub-count


def test_crashed_node_cdecr_is_unavailable() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("cdecr n0 c")).status == \
        "unavailable"


# --- no lost update + convergence over both halves ----------------------------------------------

def test_no_lost_update_inc_and_dec_under_partition() -> None:
    # +2 (majority) and -1 (minority) across a partition BOTH count, and after heal+gossip the net
    # is the merge of both halves (where the LWW incr/a single value would have lost one side).
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "cincr n0 c", "cincr n0 c", "cdecr n3 c",
              "heal", "gossip n0 n3"])
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "1"  # +2 - 1


def test_anti_entropy_converges_all_nodes_both_halves() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "cincr n0 c", "cincr n0 c", "cdecr n3 c", "heal",
              "anti_entropy n0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4"])
    for nd in config.nodes:
        assert oracle.step(s, parse_dist_action(f"cget {nd} c")).value == "1"


def test_gossip_join_over_both_halves_is_idempotent() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["cincr n0 c", "cdecr n1 c", "gossip n0 n1", "gossip n0 n1"])
    assert oracle.step(s, parse_dist_action("cget n0 c")).value == "0"  # +1 - 1, stable on re-merge


# --- delta + serialization ----------------------------------------------------------------------

def test_canonical_form_omits_ncounters_until_first_cdecr() -> None:
    config = _config(n=3)
    s = DistributedState.initial(config)
    assert "ncounters" not in to_canonical(s)  # never-decremented cluster: purely additive
    # a cluster that only cincr-s is byte-identical to the pre-increment-29 form.
    gc_only = _run(ReferenceDistOracle(config), config, ["cincr n0 c"])
    assert "ncounters" not in to_canonical(gc_only)
    decremented = _run(ReferenceDistOracle(config), config, ["cdecr n0 c"])
    assert to_canonical(decremented)["ncounters"][0] == {
        "key": "c", "holder": "n0", "owner": "n0", "count": 1
    }
    assert from_canonical(to_canonical(decremented)) == decremented
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))  # empty form unchanged


def test_pncounter_in_cluster_view() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["cdecr n0 c"])
    assert "'ncounters'" in cluster_view(s)


# --- the metamorphic tier -----------------------------------------------------------------------

def test_metamorphic_tier_refutes_a_negative_decrement_subcount() -> None:
    from verisim.dist.delta import NCounterSet, apply
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["cdecr n0 c"])
    bogus = apply(s, [NCounterSet("c", "n0", "n0", -1)])  # a negative sub-count is impossible
    verdict = TieredOracle(config).check("metamorphic", s, parse_dist_action("cdecr n0 c"), bogus)
    assert verdict.refuted and "pn-counter" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_pncounter_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "cincr n0 c", "cdecr n0 c", "cget n0 c",
        "partition n0 n1 n2 | n3 n4",
        "cincr n0 c", "cincr n0 c", "cdecr n3 c",   # concurrent inc/dec, all ack (AP)
        "cget n0 c", "cget n3 c",                   # local views diverge, nothing lost
        "heal", "gossip n0 n3", "anti_entropy n1", "cget n0 c",  # converge to the net
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
