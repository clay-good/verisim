"""The quorum (Raft-subset) consensus model (SPEC-7 §3.4, DS0 increment 7).

Pins the realistic CP middle of the consistency curriculum: ``quorum`` commits synchronously to a
reachable majority and rejects a write that cannot reach one — strictly more available than
``linearizable`` (which needs every replica) while still divergence-free (no split-brain). Tier-A
and Tier-B agree bit-for-bit. The additive ``quorum`` value leaves every prior golden/hash intact.
Dependency-free, GPU-free.
"""

import random

from verisim.dist import DistributedState, parse_dist_action
from verisim.dist.config import scaled_dist_config
from verisim.distdata import DistDriver
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle


def _cfg(model: str):
    # 5 nodes, full replication → a strict majority is 3.
    return scaled_dist_config(5, n_objects=1, replication_factor=5, consistency_model=model)


def _run(model: str, cmds: list[str]) -> DistributedState:
    cfg = _cfg(model)
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    for cmd in cmds:
        state = oracle.step(state, parse_dist_action(cmd)).state
    return state


def _status(model: str, cmds: list[str]) -> str:
    cfg = _cfg(model)
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    status = "ok"
    for cmd in cmds:
        r = oracle.step(state, parse_dist_action(cmd))
        status = r.status
        state = r.state
    return status


def test_quorum_commits_on_the_majority_side():
    """A write from a 3-node (majority) side commits; the majority replicas hold the new value."""
    state = _run("quorum", ["partition n0 n1 | n2 n3 n4", "put n2 o0 b"])
    # the majority side {n2,n3,n4} is synchronously written
    for n in ("n2", "n3", "n4"):
        assert state.replicas[("o0", n)].value == "b"
    # the minority side {n0,n1} is stale (catch-up message in flight), not divergent
    for n in ("n0", "n1"):
        assert state.replicas[("o0", n)].value == "nil"
    assert state.inflight, "the minority catch-up messages should be in flight"


def test_quorum_rejects_on_the_minority_side():
    """A write from a 2-node (minority) side cannot reach a majority, so it is rejected (CP)."""
    assert _status("quorum", ["partition n0 n1 | n2 n3 n4", "put n0 o0 a"]) == "unavailable"
    state = _run("quorum", ["partition n0 n1 | n2 n3 n4", "put n0 o0 a"])
    # nothing was committed anywhere
    assert all(r.value == "nil" for r in state.replicas.values())


def test_quorum_availability_frontier_steps_at_the_majority():
    """Swept over partition-side size k, a quorum write commits iff k >= majority (= 3 of 5)."""
    nodes = scaled_dist_config(5, n_objects=1).nodes
    for k in (1, 2, 3, 4):
        part = f"partition {' '.join(nodes[:k])} | {' '.join(nodes[k:])}"
        status = _status("quorum", [part, f"put {nodes[0]} o0 a"])
        expected = "ok" if k >= 3 else "unavailable"
        assert status == expected, f"k={k}: got {status}, expected {expected}"


def test_quorum_prevents_split_brain_where_eventual_forks():
    """Both sides write the same key. Eventual forks (two version-1 values); quorum does not."""
    script = ["partition n0 n1 | n2 n3 n4", "put n0 o0 a", "put n2 o0 b"]

    def _forked(state: DistributedState) -> bool:
        seen: dict[int, str] = {}
        for (_obj, _node), r in state.replicas.items():
            if r.version in seen and seen[r.version] != r.value:
                return True
            seen[r.version] = r.value
        return False

    assert _forked(_run("eventual", script))       # both sides commit → divergent version-1 values
    assert not _forked(_run("quorum", script))      # only the majority side commits → single value


def test_quorum_converges_after_heal():
    """The stale minority catches up on heal+advance — quorum is divergence-free, available."""
    state = _run("quorum", ["partition n0 n1 | n2 n3 n4", "put n2 o0 b", "heal", "advance 5"])
    assert all(r.value == "b" for r in state.replicas.values())
    assert not state.inflight


def test_quorum_healthy_cluster_writes_all_replicas():
    """With no partition the whole cluster is reachable, so a write reaches every node."""
    state = _run("quorum", ["put n0 o0 a"])
    assert all(r.value == "a" for r in state.replicas.values())
    assert not state.inflight  # everyone reachable → synchronous, nothing left in flight


# --- Tier-A == Tier-B under quorum, and backward-compat -------------------------------------------


def test_quorum_tier_b_agrees_bit_for_bit():
    """The W1 retirement for quorum: the autonomous-actor Tier-B reproduces the quorum decision
    bit-for-bit across the grammar battery and all drivers."""
    cfg = scaled_dist_config(5, n_objects=3, replication_factor=5, consistency_model="quorum")
    ref, sysb = ReferenceDistOracle(cfg), SystemDistOracle(cfg)
    for driver in ("uniform", "contention", "adversarial"):
        for seed in range(6):
            drv = DistDriver(driver, cfg, random.Random(seed))
            s = DistributedState.initial(cfg)
            for _ in range(30):
                a = drv.sample(s)
                assert cluster_view(ref.step(s, a).state) == cluster_view(sysb.step(s, a).state)
                s = ref.step(s, a).state


def test_quorum_is_additive_eventual_unchanged():
    """The new model leaves the eventual default's behavior identical (additive enum value)."""
    base = scaled_dist_config(3, n_objects=2)  # default eventual
    assert base.consistency_model == "eventual"
    # an eventual put still leaves async messages in flight (unchanged DS0-incr-1 dynamic)
    cfg = base
    oracle = ReferenceDistOracle(cfg)
    s = oracle.step(DistributedState.initial(cfg), parse_dist_action("put n0 o0 a")).state
    assert s.inflight  # async replication, exactly as before quorum existed
