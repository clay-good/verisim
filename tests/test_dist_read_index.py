"""DS0 increment 25 — the quorum-confirmed linearizable read: ``read_index`` (Raft ReadIndex).

`read_index node key` is the partner to the lease read `lread` (incr 18) — the *other* way Raft
serves a linearizable read (SPEC-7 §5.1). Where `lread` skips the quorum round-trip via a lease,
`read_index` confirms leadership with a majority before serving the read. The two have opposite
availability profiles (a minority leader with a live lease serves `lread` where `read_index` is
`no_quorum`), and `read_index` refuses the stale read a plain `get` from a deposed leader would
serve. It touches no state (a pure read) and Tier-A ≡ Tier-B.
"""

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import CONSENSUS_OPS, DistParseError
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


def _config(n: int = 5) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name="ri", nodes=nodes, objects=("x",), values=("a", "b"),
                      replication_factor=n, consistency_model="quorum")


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_read_index() -> None:
    assert parse_dist_action("read_index n0 x").args == ("n0", "x")
    assert "read_index" in CONSENSUS_OPS


@pytest.mark.parametrize("bad", ["read_index n0", "read_index n0 x y"])
def test_grammar_rejects_bad_read_index(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- the leader fence + quorum confirmation -----------------------------------------------------

def test_read_index_at_leader_with_quorum_serves_the_committed_value() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "append n0 x b"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("read_index n0 x"))
    assert (r.status, r.value) == ("ok", "b")


def test_read_index_by_a_non_leader_is_fenced() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["elect n0"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("read_index n1 x"))
    assert r.status == "not_leader" and r.value == "n0"  # carries the current leader


def test_read_index_with_no_leader_is_fenced() -> None:
    config = _config()
    r = ReferenceDistOracle(config).step(DistributedState.initial(config),
                                         parse_dist_action("read_index n0 x"))
    assert (r.status, r.value) == ("not_leader", "")


def test_minority_stranded_leader_cannot_confirm_leadership() -> None:
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "partition n0 n1 | n2 n3 n4"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("read_index n0 x"))
    assert r.status == "no_quorum"  # cannot confirm it is still leader -> refuses the read


def test_crashed_leader_read_index_is_unavailable() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "crash n0"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("read_index n0 x"))
    assert r.status == "unavailable"


def test_single_node_cluster_confirms_trivially() -> None:
    config = _config(n=1)
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "append n0 x b"])
    r = ReferenceDistOracle(config).step(s, parse_dist_action("read_index n0 x"))
    assert (r.status, r.value) == ("ok", "b")


# --- the lease/quorum contrast (the two linearizable reads) -------------------------------------

def test_lease_read_serves_in_minority_where_quorum_read_refuses() -> None:
    # The headline contrast: a minority-stranded leader holding a *live lease* serves lread locally
    # (the read-availability the lease buys) where read_index is no_quorum (the safety it declines).
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config,
             ["elect n0", "partition n0 n1 | n2 n3 n4", "lease n0 100"])
    oracle = ReferenceDistOracle(config)
    assert oracle.step(s, parse_dist_action("lread n0 x")).status == "ok"
    assert oracle.step(s, parse_dist_action("read_index n0 x")).status == "no_quorum"


# --- linearizable safety (no stale read from a deposed leader) ----------------------------------

def test_deposed_leader_read_index_is_fenced_where_get_serves_stale() -> None:
    # n0 leads, is partitioned ALONE while the majority elects n2 and commits a newer value n0 never
    # sees; after heal, n0's local replica is stale. read_index refuses (not_leader), but a plain
    # get serves the stale value — exactly the read read_index exists to prevent.
    config = _config(n=5)
    s = _run(ReferenceDistOracle(config), config,
             ["elect n0", "partition n0 | n1 n2 n3 n4", "elect n2", "append n2 x b", "heal"])
    oracle = ReferenceDistOracle(config)
    assert oracle.step(s, parse_dist_action("read_index n0 x")).status == "not_leader"
    assert oracle.step(s, parse_dist_action("get n0 x")).value != "b"  # stale local replica served
    assert oracle.step(s, parse_dist_action("read_index n2 x")).value == "b"  # fresh from leader


# --- a pure read touches no state ---------------------------------------------------------------

def test_read_index_touches_no_state() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["elect n0", "append n0 x b"])
    before = cluster_view(s)
    after = ReferenceDistOracle(config).step(s, parse_dist_action("read_index n0 x")).state
    # only last_result changes (the read verdict); replicas/log/medium are untouched.
    assert after.replicas == s.replicas and after.logs == s.logs
    assert after.commit_index == s.commit_index
    assert before != cluster_view(after)  # last_result reflects the read


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_read_index_trajectory() -> None:
    config = _config(n=5)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "elect n0", "append n0 x a", "read_index n0 x",   # ok a
        "read_index n1 x",                                 # not_leader
        "partition n0 n1 | n2 n3 n4", "read_index n0 x",   # no_quorum
        "lease n0 100", "lread n0 x", "read_index n0 x",   # lread ok, read_index no_quorum
        "heal", "elect n2", "append n2 x b",               # n2 leads, commits b
        "read_index n2 x",                                  # ok b
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
