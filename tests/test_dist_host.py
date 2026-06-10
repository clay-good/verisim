"""DS0 increment 23 — the embedded host: each cluster node runs a real SPEC-6 host.

`host node <syscall>` delegates to the SPEC-6 `ReferenceHostOracle` on that node's own embedded host
(process table + per-process fd tables + an embedded v0 filesystem) — the compositional vision
SPEC-7 §3.1/§4 names. Per-node isolated; host ops respect the node's up/down status; the `hosts` map
is omitted from the canonical form until the first host op (purely additive), and Tier-A ≡ Tier-B.
"""

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.action import HOST_OPS, DistParseError
from verisim.dist.serialize import from_canonical, state_hash, to_canonical
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


def _config(n: int = 3) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name="host", nodes=nodes, objects=("x",), replication_factor=n)


def _run(oracle: ReferenceDistOracle, config: DistConfig, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(config)
    for cmd in cmds:
        s = oracle.step(s, parse_dist_action(cmd)).state
    return s


# --- grammar ------------------------------------------------------------------------------------

def test_grammar_parses_host_syscalls() -> None:
    assert parse_dist_action("host n0 fork 1").args == ("n0", "fork", "1")
    assert parse_dist_action("host n0 open 1 /f").args == ("n0", "open", "1", "/f")
    assert {"host"} == HOST_OPS


@pytest.mark.parametrize("bad", ["host", "host n0", "host n0 bogus 1", "host n0 fork",
                                 "host n0 fork x"])
def test_grammar_rejects_bad_host_syscall(bad: str) -> None:
    with pytest.raises(DistParseError):
        parse_dist_action(bad)


# --- composition + per-node isolation -----------------------------------------------------------

def test_fork_runs_on_the_named_nodes_host() -> None:
    config = _config()
    r = ReferenceDistOracle(config).step(DistributedState.initial(config),
                                         parse_dist_action("host n0 fork 1"))
    assert (r.status, r.value) == ("ok", "2")  # the new pid
    assert 2 in r.state.hosts["n0"].procs


def test_per_node_host_isolation() -> None:
    # A fork on n0 must not create a process (or even a host) on n1.
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["host n0 fork 1"])
    assert 2 in s.hosts["n0"].procs
    assert "n1" not in s.hosts and "n2" not in s.hosts  # other nodes' hosts are untouched


def test_kv_and_host_coexist_on_one_node() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["put n0 x b", "host n0 fork 1"])
    assert s.replicas[("x", "n0")].value == "b"  # the KV subsystem
    assert 2 in s.hosts["n0"].procs  # the host subsystem, independent, same node


def test_embedded_filesystem_open_write() -> None:
    # open + write through an fd materializes the file in the node's embedded v0 filesystem.
    config = _config()
    oracle = ReferenceDistOracle(config)
    s = DistributedState.initial(config)
    r_open = oracle.step(s, parse_dist_action("host n0 open 1 /f"))
    s = r_open.state
    fd = r_open.value
    s = oracle.step(s, parse_dist_action(f"host n0 write 1 {fd} a")).state
    fs_node = s.hosts["n0"].fs.fs["/f"]  # the file in n0's embedded FS (a File node)
    assert getattr(fs_node, "content", None) == "a"


def test_setuid_privilege_is_enforced_per_host() -> None:
    # The embedded host enforces SPEC-6 privilege: a non-root process cannot setuid (EPERM).
    config = _config()
    oracle = ReferenceDistOracle(config)
    # root (pid 1, uid 0) forks pid 2, drops it to uid 1000, then pid 2 tries to setuid -> host_err
    s = _run(oracle, config, ["host n0 fork 1", "host n0 setuid 2 1000"])
    r = oracle.step(s, parse_dist_action("host n0 setuid 2 0"))
    assert r.status == "host_err"  # EPERM: a non-root process may not change credentials


# --- the cross-layer crash linkage --------------------------------------------------------------

def test_host_op_on_crashed_node_is_unavailable() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config, ["crash n0"])
    assert ReferenceDistOracle(config).step(s, parse_dist_action("host n0 fork 1")).status \
        == "unavailable"


def test_host_state_survives_crash_and_restart() -> None:
    config = _config()
    oracle = ReferenceDistOracle(config)
    s = _run(oracle, config, ["host n0 fork 1", "crash n0", "restart n0"])
    assert 2 in s.hosts["n0"].procs  # the pre-crash process persists
    r = oracle.step(s, parse_dist_action("host n0 fork 1"))
    assert (r.status, r.value) == ("ok", "3")  # pids keep counting — the host was paused, not wiped


def test_host_on_unknown_node_is_rejected() -> None:
    config = _config()
    r = ReferenceDistOracle(config).step(DistributedState.initial(config),
                                         parse_dist_action("host ghost fork 1"))
    assert r.status == "unknown_node"


# --- serialization ------------------------------------------------------------------------------

def test_canonical_form_omits_hosts_until_first_host_op() -> None:
    config = _config()
    s = DistributedState.initial(config)
    assert "hosts" not in to_canonical(s)  # host-free cluster: purely additive
    forked = ReferenceDistOracle(config).step(s, parse_dist_action("host n0 fork 1")).state
    assert any(h["node"] == "n0" for h in to_canonical(forked)["hosts"])
    # the host-free hash is unchanged by the (omitted) hosts field
    assert state_hash(s) == state_hash(from_canonical(to_canonical(s)))


def test_host_state_round_trips_through_canonical() -> None:
    config = _config()
    s = _run(ReferenceDistOracle(config), config,
             ["host n0 fork 1", "host n0 open 1 /f", "host n0 write 1 0 a", "put n0 x b"])
    assert from_canonical(to_canonical(s)) == s  # the embedded host (incl. its v0 FS) round-trips


# --- Tier-A ≡ Tier-B ----------------------------------------------------------------------------

def test_tier_a_equals_tier_b_over_a_host_trajectory() -> None:
    config = _config()
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    script = [
        "host n0 fork 1", "host n1 fork 1",       # per-node forks
        "put n0 x b",                              # KV + host coexist
        "host n0 open 1 /f", "host n0 write 1 0 a",  # embedded FS
        "crash n0", "host n0 fork 1",              # crashed -> unavailable
        "restart n0", "host n0 fork 1",            # restored; state survived
        "host n0 setuid 2 1000", "host n0 setuid 2 0",  # privilege (the second is EPERM)
    ]
    for cmd in script:
        a = parse_dist_action(cmd)
        ra, rb = ref.step(sa, a), sysb.step(sb, a)
        assert cluster_view(ra.state) == cluster_view(rb.state), cmd
        assert (ra.status, ra.value) == (rb.status, rb.value), cmd
        sa, sb = ra.state, rb.state
