"""DS0 increment 2 — multi-key transactions over the replicated KV (SPEC-7 §3.2).

These pin the OCC (optimistic concurrency control, first-committer-wins) transaction semantics: a
coordinator buffers a transaction's reads and writes and validates at commit. The tests cover the
M1-analogue invariant (``apply(state, delta) == next_state``) on every transition, atomic multi-key
commit + async replication, read-your-writes, the OCC conflict→abort path, the linearizable
CP-commit rejection, Tier-A↔Tier-B agreement on transaction workloads, and the backward-compatible
serialization (an empty transaction set is omitted from the canonical form, so DS0-increment-1
hashes are unchanged). Dependency-free, GPU-free.
"""

from __future__ import annotations

import random

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.delta import apply
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.dist.state import TxnState
from verisim.distdata.drivers import ALL_DIST_DRIVERS, DistDriver
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle

CFG = DistConfig()
REF = ReferenceDistOracle(CFG)


def _run(oracle: ReferenceDistOracle | SystemDistOracle, cmds: list[str]) -> DistributedState:
    s = DistributedState.initial(CFG)
    for cmd in cmds:
        r = oracle.step(s, parse_dist_action(cmd))
        assert apply(s, r.delta) == r.state, f"apply != oracle at {cmd!r}"
        s = r.state
    return s


def _status(oracle: ReferenceDistOracle, s: DistributedState, cmd: str) -> str:
    return oracle.step(s, parse_dist_action(cmd)).status


def test_commit_applies_all_buffered_writes_atomically():
    s = _run(REF, ["begin n0 t0", "tput n0 t0 x a", "tput n0 t0 y b", "commit n0 t0", "advance 2"])
    assert s.replicas[("x", "n0")].value == "a"
    assert s.replicas[("y", "n0")].value == "b"
    # both keys replicated and converged
    assert {s.replicas[("x", n)].value for n in CFG.nodes} == {"a"}
    assert {s.replicas[("y", n)].value for n in CFG.nodes} == {"b"}
    assert s.txns == {}  # committed txn is removed


def test_read_your_writes_within_a_transaction():
    s = DistributedState.initial(CFG)
    for cmd in ["begin n0 t0", "tput n0 t0 x q"]:
        s = REF.step(s, parse_dist_action(cmd)).state
    r = REF.step(s, parse_dist_action("tget n0 t0 x"))
    assert r.status == "ok" and r.value == "q"  # sees its own buffered write, not the committed nil


def test_occ_conflict_aborts_first_committer_wins():
    # t0 reads x; a concurrent committer (t1) changes x; t0's commit must abort (conflict).
    s = DistributedState.initial(CFG)
    cmds = ["begin n0 t0", "tget n0 t0 x", "begin n0 t1", "tput n0 t1 x c", "commit n0 t1",
            "tput n0 t0 x z", "commit n0 t0"]
    statuses = []
    for cmd in cmds:
        r = REF.step(s, parse_dist_action(cmd))
        statuses.append(r.status)
        s = r.state
    assert statuses[4] == "committed"  # t1 wins
    assert statuses[-1] == "conflict"  # t0 loses (its read of x was invalidated)
    assert s.replicas[("x", "n0")].value == "c"  # t0's write z never applied
    assert s.txns == {}  # both transactions resolved


def test_commit_with_no_read_conflict_succeeds_even_after_concurrent_other_key_write():
    # t0 reads x; a concurrent write to a DIFFERENT key (y) does not invalidate t0's read-set.
    s = DistributedState.initial(CFG)
    cmds = ["begin n0 t0", "tget n0 t0 x", "put n0 y c", "tput n0 t0 x z", "commit n0 t0"]
    last = "n/a"
    for cmd in cmds:
        r = REF.step(s, parse_dist_action(cmd))
        last = r.status
        s = r.state
    assert last == "committed"
    assert s.replicas[("x", "n0")].value == "z"


def test_abort_discards_all_buffered_writes():
    s = _run(REF, ["begin n0 t0", "tput n0 t0 x z", "abort n0 t0", "get n0 x"])
    assert s.replicas[("x", "n0")].value == "nil"
    assert s.txns == {}


def test_ops_on_unknown_transaction_are_rejected():
    s = DistributedState.initial(CFG)
    for cmd in ["tput n0 ghost x a", "tget n0 ghost x", "commit n0 ghost", "abort n0 ghost"]:
        assert _status(REF, s, cmd) == "no_txn"


def test_begin_on_existing_txn_is_idempotent_noop():
    s = REF.step(DistributedState.initial(CFG), parse_dist_action("begin n0 t0")).state
    r = REF.step(s, parse_dist_action("begin n0 t0"))
    assert r.status == "exists"
    assert apply(s, r.delta) == r.state


def test_linearizable_commit_under_partition_is_rejected_and_keeps_txn_open():
    lin = DistConfig(consistency_model="linearizable")
    ref = ReferenceDistOracle(lin)
    s = DistributedState.initial(lin)
    cmds = ["begin n0 t0", "tput n0 t0 x a", "partition n0 | n1 n2", "commit n0 t0"]
    last = "n/a"
    for cmd in cmds:
        r = ref.step(s, parse_dist_action(cmd))
        last = r.status
        s = r.state
    assert last == "unavailable"  # CP: cannot reach all replicas synchronously
    assert "t0" in s.txns  # the txn stays open for retry after heal (state not aborted)
    assert s.replicas[("x", "n0")].value == "nil"  # nothing committed


def test_apply_equals_oracle_on_transactional_driver():
    for seed in range(6):
        drv = DistDriver("transactional", CFG, random.Random(seed))
        s = DistributedState.initial(CFG)
        for _ in range(50):
            a = drv.sample(s)
            r = REF.step(s, a)
            assert apply(s, r.delta) == r.state
            s = r.state


def test_tier_a_and_tier_b_agree_on_transaction_workload():
    sys_oracle = SystemDistOracle(CFG)
    for seed in range(6):
        drv = DistDriver("transactional", CFG, random.Random(seed))
        s = DistributedState.initial(CFG)
        for _ in range(50):
            a = drv.sample(s)
            assert cluster_view(REF.step(s, a).state) == cluster_view(sys_oracle.step(s, a).state)
            s = REF.step(s, a).state


def test_empty_txns_omitted_from_canonical_but_active_txns_roundtrip():
    # backward-compat: a cluster with no open txns serializes to the exact DS0-incr-1 normal form
    assert "txns" not in to_canonical(DistributedState.initial(CFG))
    # an active txn is included and round-trips exactly
    s = _run(REF, ["begin n0 t0", "tput n0 t0 x a", "tget n0 t0 y"])
    canon = to_canonical(s)
    assert "txns" in canon
    assert from_canonical(canon) == s
    # reads pin the read version (y@0); writes buffer the value (x="a") AND pin the write version
    # (x@0) for snapshot-isolation validation
    assert s.txns["t0"] == TxnState("t0", "n0", (("y", 0),), (("x", "a"),), (("x", 0),))


def test_transactional_preset_is_named_but_not_in_default_sweep():
    # the transaction workload is reachable by name but kept out of the default driver sweep so the
    # DS0-increment-1 experiments/figures (which iterate DIST_DRIVERS) are unchanged
    from verisim.distdata.drivers import DIST_DRIVERS

    assert "transactional" not in DIST_DRIVERS
    assert "transactional" in ALL_DIST_DRIVERS
