"""Read-uncommitted isolation: the dirty-read anomaly (SPEC-7 §3.2, DS0 increment 10).

Pins the weakest OCC isolation level — the bottom of the standard SQL hierarchy
(``read_uncommitted ⊂ read_committed ⊂ snapshot ⊂ serializable``). It drops even read-committed's
last guarantee: a ``tget`` may observe another active transaction's **uncommitted** buffered write,
so if that writer aborts the reader saw a value that never committed — the classic **dirty-read**
anomaly (Adya G1a). Read-uncommitted is purely additive (a new ``txn_isolation`` value), so every
prior golden/hash is unchanged, and Tier-B reproduces it bit-for-bit (transaction bookkeeping is
coordinator-local). The dirty read applies only under OCC: 2PL's locks block it. Dependency-free.
"""

from __future__ import annotations

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.config import TXN_ISOLATION_LEVELS, scaled_dist_config
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle

# The canonical dirty-read scenario: A writes x (uncommitted), B reads x, then A aborts (rollback).
_DIRTY_READ = [
    "begin n0 A", "begin n0 B",
    "tput n0 A o0 b",  # A buffers an uncommitted write
    "tget n0 B o0",    # B reads x -- the dirty read under read_uncommitted
    "abort n0 A",      # A rolls back -> the value B may have read never commits
    "commit n0 B",
]


def _b_read(isolation: str, concurrency_control: str = "occ") -> tuple[str, str]:
    """Return (B's observed read value, the committed final value of o0) for one config."""
    cfg = scaled_dist_config(3, n_objects=2, txn_isolation=isolation,
                             concurrency_control=concurrency_control)
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    b_read = ""
    for cmd in _DIRTY_READ:
        r = oracle.step(state, parse_dist_action(cmd))
        if cmd.startswith("tget n0 B"):
            b_read = r.value
        state = r.state
    return b_read, state.replicas[("o0", "n0")].value


# --- the dirty-read anomaly -----------------------------------------------------------------------


def test_read_uncommitted_admits_dirty_read():
    b_read, final = _b_read("read_uncommitted")
    assert b_read == "b"  # B observed A's UNcommitted write -- the dirty read
    assert final == "nil"  # ...but A aborted, so the value B read never committed (the anomaly)


@pytest.mark.parametrize("isolation", ["read_committed", "snapshot", "serializable"])
def test_stronger_levels_forbid_dirty_read(isolation):
    b_read, final = _b_read(isolation)
    assert b_read == "nil"  # the MVCC tget gives only committed data -- no dirty read
    assert final == "nil"


def test_2pl_blocks_the_dirty_read_even_under_read_uncommitted():
    # Under 2PL the writer A holds an exclusive lock, so B can never observe the uncommitted write
    # (2PL gives serializability regardless of the declared level). B is wounded (younger) and reads
    # nothing dirty; the declared read_uncommitted level is inert under locking.
    b_read, _ = _b_read("read_uncommitted", concurrency_control="2pl")
    assert b_read != "b"  # no dirty read: locks dominate the declared isolation level


# --- Tier-B reproduces it bit-for-bit -------------------------------------------------------------


def test_tier_b_reproduces_read_uncommitted_bit_for_bit():
    cfg = scaled_dist_config(3, n_objects=2, txn_isolation="read_uncommitted")
    ref = ReferenceDistOracle(cfg)
    sys_oracle = SystemDistOracle(cfg)
    s, s_sys = DistributedState.initial(cfg), DistributedState.initial(cfg)
    for cmd in _DIRTY_READ:
        action = parse_dist_action(cmd)
        r = ref.step(s, action)
        r_sys = sys_oracle.step(s_sys, action)
        assert cluster_view(r.state) == cluster_view(r_sys.state)
        assert r.status == r_sys.status
        assert r.value == r_sys.value
        s, s_sys = r.state, r_sys.state


# --- additivity: the new level does not perturb the existing configs ------------------------------


def test_read_uncommitted_is_a_known_isolation_level():
    assert "read_uncommitted" in TXN_ISOLATION_LEVELS
    assert TXN_ISOLATION_LEVELS[-1] == "read_uncommitted"  # ordered last (the weakest)


def test_default_hash_is_unchanged_by_the_new_level():
    # The default config still serializes txn_isolation="serializable"; adding read_uncommitted to
    # the vocabulary must not change the default config's identity (additive guarantee).
    cfg = DistConfig()
    assert cfg.txn_isolation == "serializable"
    assert cfg.to_dict()["txn_isolation"] == "serializable"


def test_read_uncommitted_state_round_trips_through_canonical():
    cfg = scaled_dist_config(3, n_objects=2, txn_isolation="read_uncommitted")
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    for cmd in _DIRTY_READ:
        state = oracle.step(state, parse_dist_action(cmd)).state
    assert from_canonical(to_canonical(state)) == state
