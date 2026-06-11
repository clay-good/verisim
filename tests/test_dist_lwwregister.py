"""DS0 increment 32 — the CRDT LWW-register: ``lwwput``/``lwwget`` + Lamport-ordered resolution.

A *state-based* last-writer-wins register (SPEC-7 §3.2), the policy-opposite of the MV-register:
the MV-register surfaces a conflict as siblings, the LWW-register **deterministically picks one
winner** by a **Lamport-timestamp total order**. ``lwwput n key val`` stamps ``val`` with ``(ts,
owner=n)`` (``ts = lamport[n] + 1``), and the join keeps the **max** by ``(ts, owner, value)``: a
write that happened-after another (higher ts) wins regardless of node, and concurrent writes (equal
ts) break the tie by node id. Tier-A ≡ Tier-B, purely additive over increment 31 (two
omitted-when-empty ``lwwreg``/``lamport`` maps + the ``LWWRegSet``/``LamportSet`` edits).
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
    return DistConfig(name="lww", nodes=nodes, objects=("w",), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_lwwreg_ops() -> None:
    assert parse_dist_action("lwwput n0 w a").args == ("n0", "w", "a")
    assert parse_dist_action("lwwget n0 w").args == ("n0", "w")
    assert {"lwwput", "lwwget"} <= CLIENT_OPS


@pytest.mark.parametrize("bad", ["lwwput n0 w", "lwwget n0 w v", "lwwput n0 w a b"])
def test_grammar_rejects_bad(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- node-local write / read + the Lamport clock ------------------------------------------------

def test_lwwput_then_lwwget_reads_the_value() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["lwwput n0 w a"])
    assert oracle.step(s, parse_dist_action("lwwget n0 w")).value == "a"


def test_lamport_clock_advances_on_write() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["lwwput n0 w a", "lwwput n0 w b"])
    assert s.lamport["n0"] == 2  # two writes -> ts 1 then 2
    assert s.lwwreg[("w", "n0")] == ("b", 2, "n0")  # the later write is the local winner


def test_lwwput_is_always_available_even_partitioned_alone() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["partition n0 | n1 n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("lwwput n0 w z"))
    assert (r.status, r.value) == ("ok", "z")


def test_crashed_node_lwwput_is_unavailable() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("lwwput n0 w a")).status == \
        "unavailable"


# --- the Lamport-timestamp total order (happens-after wins) -------------------------------------

def test_happens_after_wins_regardless_of_node_id() -> None:
    # The high-id node n2 writes first; the low-id node n0, having observed it, writes later. The
    # later write has a higher Lamport ts, so it wins — proving ts (causality) beats node id.
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["lwwput n2 w a", "anti_entropy n0", "anti_entropy n1", "lwwput n0 w b",
              "anti_entropy n1", "anti_entropy n2"])
    for nd in config.nodes:
        assert oracle.step(s, parse_dist_action(f"lwwget {nd} w")).value == "b"


def test_concurrent_writes_resolve_by_node_id_tiebreak() -> None:
    # Both writes are concurrent (equal Lamport ts 1, neither saw the other) — the tie breaks by
    # node id, so n2 > n0 wins, deterministically and the same on every node.
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 | n2", "lwwput n0 w a", "lwwput n2 w b", "heal", "gossip n0 n2"])
    assert oracle.step(s, parse_dist_action("lwwget n0 w")).value == "b"  # owner n2 > n0
    assert oracle.step(s, parse_dist_action("lwwget n2 w")).value == "b"


def test_anti_entropy_converges_all_nodes() -> None:
    config = _config(n=5)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 n2 | n3 n4", "lwwput n0 w a", "lwwput n3 w b", "heal",
              "anti_entropy n0", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
              "anti_entropy n4"])
    for nd in config.nodes:
        assert oracle.step(s, parse_dist_action(f"lwwget {nd} w")).value == "b"  # n3 > n0


def test_max_by_timestamp_join_is_idempotent() -> None:
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config,
             ["partition n0 n1 | n2", "lwwput n0 w a", "lwwput n2 w b", "heal",
              "gossip n0 n2", "gossip n0 n2"])
    assert oracle.step(s, parse_dist_action("lwwget n0 w")).value == "b"  # stable under re-merge


# --- delta + serialization ----------------------------------------------------------------------

def test_canonical_form_omits_lwwreg_until_first_op() -> None:
    config = _config(n=3)
    s = DistributedState.initial(config)
    assert "lwwreg" not in to_canonical(s)  # never-written cluster: purely additive
    assert "lamport" not in to_canonical(s)
    written = _run(ReferenceDistOracle(config), config, ["lwwput n0 w a"])
    assert to_canonical(written)["lwwreg"][0] == {
        "key": "w", "holder": "n0", "value": "a", "ts": 1, "owner": "n0"
    }
    assert to_canonical(written)["lamport"][0] == {"holder": "n0", "value": 1}
    assert from_canonical(to_canonical(written)) == written
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))  # empty form unchanged


def test_lwwreg_in_cluster_view() -> None:
    config = _config(n=3)
    s = _run(ReferenceDistOracle(config), config, ["lwwput n0 w a"])
    view = cluster_view(s)
    assert "'lwwreg'" in view and "'lamport'" in view


# --- the metamorphic tier -----------------------------------------------------------------------

def test_metamorphic_tier_refutes_a_phantom_owner() -> None:
    from verisim.dist.delta import LWWRegSet, apply
    from verisim.distoracle.tiers import TieredOracle
    config = _config(n=3)
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["lwwput n0 w a"])
    bogus = apply(s, [LWWRegSet("w", "n0", "a", 1, "ghost")])  # an owner that is not a cluster node
    act = parse_dist_action("lwwput n0 w a")
    verdict = TieredOracle(config).check("metamorphic", s, act, bogus)
    assert verdict.refuted and "lwwreg" in verdict.reason


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_lwwreg_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "lwwput n0 w a", "anti_entropy n4", "lwwput n4 w b", "lwwget n0 w",  # happens-after
        "partition n0 n1 n2 | n3 n4",
        "lwwput n0 w x", "lwwput n3 w y",           # concurrent writes, both ack (AP)
        "lwwget n0 w", "lwwget n3 w",               # local views diverge, no wrong resolution
        "heal", "gossip n0 n3", "anti_entropy n1", "lwwget n2 w",  # converge to the winner
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
