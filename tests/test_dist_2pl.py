"""Lock-based 2PL with deterministic wound-wait (SPEC-7 §3.2, DS0 increment 8).

Pins the pessimistic concurrency-control alternative to OCC: ``tget``/``tput`` acquire shared/
exclusive locks held to commit, conflicts resolved by **wound-wait** (the older txn — smaller id —
preempts; the younger aborts, never waits), so it is deterministic and deadlock-free. 2PL gives
serializability (forbids write skew). The lock table is purely additive (omitted-when-empty), so
every prior golden/hash is unchanged, and Tier-B reproduces it bit-for-bit. Dependency-free.
"""

import random

from verisim.dist import DistributedState, parse_dist_action
from verisim.dist.config import scaled_dist_config
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distdata import DistDriver
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle


def _cfg(cc: str = "2pl"):
    return scaled_dist_config(3, n_objects=3, concurrency_control=cc)


def _run(cmds: list[str], cc: str = "2pl") -> tuple[DistributedState, list[str]]:
    cfg = _cfg(cc)
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    statuses: list[str] = []
    for cmd in cmds:
        r = oracle.step(state, parse_dist_action(cmd))
        statuses.append(r.status)
        state = r.state
    return state, statuses


# --- wound-wait: the older transaction always wins -----------------------------------------------


def test_younger_is_wounded_when_it_requests_an_older_txns_lock():
    """A (older) holds X on o0; B (younger) requesting X is wounded (aborts), A commits."""
    state, st = _run([
        "begin n0 A", "begin n0 B", "tput n0 A o0 a", "tput n0 B o0 b",
        "commit n0 A", "commit n0 B",
    ])
    assert st[3] == "wounded"   # B's tput on A's locked key
    assert st[4] == "committed"  # A commits
    assert st[5] == "no_txn"     # B was already removed by the wound
    assert state.replicas[("o0", "n0")].value == "a"
    assert not state.locks  # all locks released after commit


def test_older_preempts_a_younger_holder():
    """B (younger) holds X on o0; A (older) requesting X wounds B and wins (the older wins)."""
    state, st = _run([
        "begin n0 A", "begin n0 B", "tput n0 B o0 b", "tput n0 A o0 a",
        "commit n0 B", "commit n0 A",
    ])
    assert st[3] == "ok"          # A's tput preempts B
    assert st[4] == "no_txn"      # B was wounded by A
    assert st[5] == "committed"   # A commits
    assert state.replicas[("o0", "n0")].value == "a"  # the older txn's write wins


def test_shared_read_locks_are_compatible():
    """Two txns can both hold a shared lock on the same key (concurrent reads don't conflict)."""
    state, st = _run([
        "begin n0 A", "begin n0 B", "tget n0 A o0", "tget n0 B o0",
    ])
    assert st[2] == "ok" and st[3] == "ok"  # both reads acquire compatible S locks
    holders = dict(state.locks)["o0"]
    assert ("A", "S") in holders and ("B", "S") in holders


def test_2pl_forbids_write_skew():
    """Both read {x, y}; A writes x, B writes y. 2PL's read S-locks block the cross-write."""
    _, st = _run([
        "begin n0 A", "begin n0 B",
        "tget n0 A o0", "tget n0 A o1", "tget n0 B o0", "tget n0 B o1",
        "tput n0 A o0 a", "tput n0 B o1 b", "commit n0 A", "commit n0 B",
    ])
    assert st.count("committed") < 2  # write skew forbidden (OCC-serializable forbids it too)


def test_abort_releases_locks():
    """An explicit abort releases the txn's locks so a later txn can acquire them."""
    state, st = _run([
        "begin n0 A", "tput n0 A o0 a", "abort n0 A", "begin n0 B", "tput n0 B o0 b",
    ])
    assert st[2] == "aborted"
    assert st[4] == "ok"  # B acquires the lock A released
    assert dict(state.locks)["o0"] == (("B", "X"),)


def test_commit_releases_all_locks():
    """A committed txn releases every lock it held (the shrinking phase)."""
    state, _ = _run([
        "begin n0 A", "tget n0 A o0", "tput n0 A o1 a", "commit n0 A",
    ])
    assert not state.locks


# --- the lock table is additive and round-trips --------------------------------------------------


def test_occ_default_has_no_locks_and_is_unchanged():
    """The default OCC path acquires no locks, so an OCC state serializes with no ``locks`` key."""
    state, _ = _run(["begin n0 A", "tget n0 A o0", "tput n0 A o1 a"], cc="occ")
    assert not state.locks
    assert "locks" not in to_canonical(state)


def test_locks_round_trip_through_canonical_form():
    """A 2PL state with held locks round-trips exactly through the canonical serialization."""
    state, _ = _run(["begin n0 A", "tget n0 A o0", "tput n0 A o1 a"])
    assert state.locks  # locks are held mid-transaction
    canon = to_canonical(state)
    assert "locks" in canon
    assert from_canonical(canon) == state


# --- Tier-B agreement ----------------------------------------------------------------------------


def test_2pl_tier_b_agrees_bit_for_bit():
    """Transaction bookkeeping is coordinator-local, so Tier-B (delegating to the same txn_step)
    reproduces 2PL — the lock table and wound-wait — bit-for-bit on the cluster channel."""
    cfg = _cfg("2pl")
    ref, sysb = ReferenceDistOracle(cfg), SystemDistOracle(cfg)
    for seed in range(8):
        drv = DistDriver("transactional", cfg, random.Random(seed))
        s = DistributedState.initial(cfg)
        for _ in range(40):
            a = drv.sample(s)
            assert cluster_view(ref.step(s, a).state) == cluster_view(sysb.step(s, a).state)
            s = ref.step(s, a).state
