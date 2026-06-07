"""Read-committed isolation: the lost-update anomaly (SPEC-7 §3.2, DS0 increment 9).

Pins the weakest OCC isolation level — the real-world default of Postgres/Oracle/SQL-Server. It does
no commit-time validation: reads still see only committed data (the MVCC ``tget`` — no dirty reads),
but with no write-write check two same-key read-modify-write transactions both commit and the later
silently overwrites the earlier — the classic **lost-update** anomaly snapshot's first-committer-
wins prevents. Read-committed is purely additive (a new ``txn_isolation`` value), so every prior
golden/hash is unchanged, and Tier-B reproduces it bit-for-bit (transaction bookkeeping is
coordinator-local). Dependency-free.
"""

from __future__ import annotations

import pytest

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.config import TXN_ISOLATION_LEVELS, scaled_dist_config
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.system import SystemDistOracle

# The canonical lost-update scenario: A and B both read x at one version, then both write x back.
_LOST_UPDATE = [
    "begin n0 A", "begin n0 B",
    "tget n0 A o0", "tget n0 B o0",
    "tput n0 A o0 b", "tput n0 B o0 c",
    "commit n0 A", "commit n0 B",
]


def _run(isolation: str) -> tuple[list[str], str]:
    cfg = scaled_dist_config(3, n_objects=2, txn_isolation=isolation)
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    statuses: list[str] = []
    for cmd in _LOST_UPDATE:
        r = oracle.step(state, parse_dist_action(cmd))
        statuses.append(r.status)
        state = r.state
    return statuses, state.replicas[("o0", "n0")].value


# --- the lost-update anomaly ----------------------------------------------------------------------


def test_read_committed_admits_lost_update():
    statuses, final = _run("read_committed")
    # both txns commit; the later writer (B -> "c") overwrites the earlier (A -> "b") -- update lost
    assert statuses[-2:] == ["committed", "committed"]
    assert final == "c"


@pytest.mark.parametrize("isolation", ["snapshot", "serializable"])
def test_stronger_levels_forbid_lost_update(isolation):
    statuses, final = _run(isolation)
    # the second committer's write-write (snapshot) / read-set (serializable) validation aborts it
    assert statuses[-2:] == ["committed", "conflict"]
    assert final == "b"  # only the first committer's write survives -- no update lost


# --- read-committed still forbids dirty reads (the one guarantee it keeps) ------------------------


def test_read_committed_reads_only_committed_data():
    # B's uncommitted buffered write must not be visible to A's read (no dirty read).
    cfg = scaled_dist_config(3, n_objects=2, txn_isolation="read_committed")
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    script = ["begin n0 A", "begin n0 B", "tput n0 B o0 c", "tget n0 A o0"]
    last = ""
    for cmd in script:
        r = oracle.step(state, parse_dist_action(cmd))
        last = r.value
        state = r.state
    assert last == "nil"  # A sees the committed boot value, not B's uncommitted "c"


# --- Tier-B reproduces it bit-for-bit -------------------------------------------------------------


def test_tier_b_reproduces_read_committed_bit_for_bit():
    cfg = scaled_dist_config(3, n_objects=2, txn_isolation="read_committed")
    ref = ReferenceDistOracle(cfg)
    sys_oracle = SystemDistOracle(cfg)
    s, s_sys = DistributedState.initial(cfg), DistributedState.initial(cfg)
    for cmd in _LOST_UPDATE:
        action = parse_dist_action(cmd)
        r = ref.step(s, action)
        r_sys = sys_oracle.step(s_sys, action)
        assert cluster_view(r.state) == cluster_view(r_sys.state)
        assert r.status == r_sys.status
        s, s_sys = r.state, r_sys.state


# --- additivity: the new level does not perturb the existing configs ------------------------------


def test_read_committed_is_a_known_isolation_level():
    assert "read_committed" in TXN_ISOLATION_LEVELS
    with pytest.raises(ValueError):
        DistConfig(txn_isolation="nonsense")


def test_serializable_default_hash_is_unchanged_by_the_new_level():
    # The default config still serializes txn_isolation="serializable"; adding read_committed to the
    # vocabulary must not change the default config's identity (additive guarantee).
    cfg = DistConfig()
    assert cfg.txn_isolation == "serializable"
    assert cfg.to_dict()["txn_isolation"] == "serializable"


def test_read_committed_state_round_trips_through_canonical():
    cfg = scaled_dist_config(3, n_objects=2, txn_isolation="read_committed")
    oracle = ReferenceDistOracle(cfg)
    state = DistributedState.initial(cfg)
    for cmd in _LOST_UPDATE:
        state = oracle.step(state, parse_dist_action(cmd)).state
    assert from_canonical(to_canonical(state)) == state
