"""Multi-key transactions over the replicated KV (SPEC-7 §3.2, DS0 increment 2).

The transaction family (``begin``/``tget``/``tput``/``commit``/``abort``) under **optimistic
concurrency control** (OCC, first-committer-wins): a coordinator buffers a transaction's reads and
writes locally and validates at ``commit``. This is the *shared, local* transaction logic both the
Tier-A reference oracle and the Tier-B system oracle delegate to — transaction bookkeeping is a
*coordinator-local, deterministic* concern, not a distributed one, so it is computed identically by
both; the genuinely-distributed part (a committed write's asynchronous **replication** to peer
replicas) flows through the same in-flight message medium as a plain ``put`` and is delivered later
by ``advance``, where Tier-B's autonomous-actor delivery independently validates it.

**Why OCC, not 2PL (design decision DD-D3).** OCC is *deterministic and deadlock-free* (no lock
table, no lock-acquisition order, no deadlock detection / victim selection — all of which inject
nondeterminism or require a scheduler), so it is the discipline the deterministic core pins first,
exactly as the KV core pinned async-replication LWW before consensus. The semantics:

  - ``begin node txn``           -- open a transaction at the coordinator ``node``.
  - ``tget node txn key``        -- read ``key``'s local replica; **pin its version** on first read
                                    (the read-set the commit validates), with read-your-writes for a
                                    value the txn has already buffered.
  - ``tput node txn key val``    -- buffer a write to ``key`` (no replica changes until commit).
  - ``commit node txn``          -- **validate**: if any read key's local version changed since the
                                    read, **abort** (``conflict``); else apply every buffered write
                                    atomically (each an MVCC bump + async/synchronous replication),
                                    end the txn (``committed``). First-committer-wins.
  - ``abort node txn``           -- discard the txn (``aborted``).

The commit's replication obeys the declared consistency model exactly as ``put`` does: ``eventual``
enqueues async messages (peers converge on ``advance``); ``linearizable`` writes every replica
synchronously and **rejects** (``unavailable``) a commit it cannot fully replicate (the CP choice).
Pure and dependency-free; ``apply(state, delta) == next_state`` holds by construction.
"""

from __future__ import annotations

from verisim.dist.action import DistAction
from verisim.dist.config import DistConfig
from verisim.dist.delta import (
    DistDelta,
    DistEdit,
    EventAppend,
    LockSet,
    MsgSend,
    ReplicaWrite,
    SetResult,
    TxnDel,
    TxnSet,
)
from verisim.dist.state import DistributedState, TxnState

LockTable = dict[str, tuple[tuple[str, str], ...]]


def txn_event(state: DistributedState, node: str, raw: str) -> EventAppend:
    """A causal-log event for a transaction client op (program-order on the same node)."""
    prior = tuple(e.id for e in state.log if e.node == node)
    return EventAppend(state.next_event_id, node, raw, state.clock, prior)


def txn_step(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """Compute the (delta, status, value) for one transaction action. Pure, deterministic."""
    name = action.name
    node, txn_id = action.args[0], action.args[1]
    ev = txn_event(state, node, action.raw)

    if name == "begin":
        if not state.is_up(node):
            return [ev, SetResult("unavailable", "")], "unavailable", ""
        if txn_id in state.txns:
            return [ev, SetResult("exists", "")], "exists", ""
        return [ev, TxnSet(TxnState(txn_id, node)), SetResult("ok", "")], "ok", ""

    txn = state.txns.get(txn_id)
    if txn is None or txn.node != node:
        return [ev, SetResult("no_txn", "")], "no_txn", ""
    if not state.is_up(node):
        return [ev, SetResult("unavailable", "")], "unavailable", ""

    two_pl = config.concurrency_control == "2pl"
    if name == "tget":
        return _tget_2pl(state, ev, txn, action.args[2]) if two_pl \
            else _tget(state, ev, txn, action.args[2])
    if name == "tput":
        return _tput_2pl(state, ev, txn, action.args[2], action.args[3]) if two_pl \
            else _tput(state, ev, txn, action.args[2], action.args[3])
    if name == "commit":
        return _commit(state, ev, txn, config)
    if name == "abort":
        # 2PL holds locks to the end, so an abort must release them; OCC holds none.
        releases = _release_edits(state.locks, [txn_id]) if two_pl else []
        return [ev, *releases, TxnDel(txn_id), SetResult("aborted", "")], "aborted", ""
    raise ValueError(f"not a transaction action: {name!r}")  # pragma: no cover - grammar is closed


def _tget(
    state: DistributedState, ev: EventAppend, txn: TxnState, key: str
) -> tuple[DistDelta, str, str]:
    buffered = txn.buffered_write(key)
    if buffered is not None:  # read-your-writes: the txn's own buffered value, not the snapshot
        return [ev, SetResult("ok", buffered)], "ok", buffered
    replica = state.replicas.get((key, txn.node))
    if replica is None:
        return [ev, SetResult("no_replica", "")], "no_replica", ""
    if txn.read_version(key) is None:  # first read pins the version the commit validates against
        updated = TxnState(txn.txn_id, txn.node, (*txn.reads, (key, replica.version)),
                           txn.writes, txn.write_versions)
        return [ev, TxnSet(updated), SetResult("ok", replica.value)], "ok", replica.value
    return [ev, SetResult("ok", replica.value)], "ok", replica.value  # re-read: version pinned


def _tput(
    state: DistributedState, ev: EventAppend, txn: TxnState, key: str, value: str
) -> tuple[DistDelta, str, str]:
    writes = (*txn.writes, (key, value))
    write_versions = txn.write_versions
    if txn.write_version(key) is None:  # pin the version on first write (snapshot-isolation check)
        replica = state.replicas.get((key, txn.node))
        write_versions = (*write_versions, (key, replica.version if replica is not None else 0))
    updated = TxnState(txn.txn_id, txn.node, txn.reads, writes, write_versions)
    return [ev, TxnSet(updated), SetResult("ok", value)], "ok", value


# --- 2PL: strict two-phase locking with deterministic wound-wait (DS0 increment 8) ---------------

def _release_table(locks: LockTable, txn_ids: set[str]) -> LockTable:
    """A copy of ``locks`` with every ``txn_id`` removed from every object (empty objects dropped)."""
    out: LockTable = {}
    for obj, holders in locks.items():
        kept = tuple((t, m) for (t, m) in holders if t not in txn_ids)
        if kept:
            out[obj] = kept
    return out


def _lock_edits(old: LockTable, new: LockTable) -> list[DistEdit]:
    """The minimal ``LockSet`` edits (sorted by object) that turn ``old`` into ``new``."""
    edits: list[DistEdit] = []
    for obj in sorted(set(old) | set(new)):
        if old.get(obj, ()) != new.get(obj, ()):
            edits.append(LockSet(obj, new.get(obj, ())))
    return edits


def _release_edits(locks: LockTable, txn_ids: list[str]) -> list[DistEdit]:
    """``LockSet`` edits releasing every lock held by ``txn_ids`` (commit/abort/wound)."""
    return _lock_edits(locks, _release_table(locks, set(txn_ids)))


def _acquire(
    locks: LockTable, txn_id: str, key: str, mode: str
) -> tuple[LockTable | None, list[str]]:
    """Acquire a ``mode`` (``"S"``/``"X"``) lock on ``key`` for ``txn_id`` under **wound-wait**.

    Returns ``(new_locks, wounded_ids)`` if granted, or ``(None, [])`` if the requester must abort.
    Conflict rule: ``X`` conflicts with any other holder; ``S`` conflicts only with an ``X`` holder.
    Wound-wait (deterministic, deadlock-free): the **older** txn — the lexicographically smaller id —
    preempts; so the requester acquires by *wounding* (aborting) every conflicting holder it is older
    than, and is itself aborted iff any conflicting holder is older than it (the younger never waits).
    """
    holders = locks.get(key, ())
    conflicting = [
        t for (t, m) in holders if t != txn_id and (mode == "X" or m == "X")
    ]
    if not conflicting:
        others = [(t, m) for (t, m) in holders if t != txn_id]
        granted = {**locks, key: tuple(sorted([*others, (txn_id, mode)]))}
        return granted, []
    if all(txn_id < t for t in conflicting):  # older than every conflict -> wound them, then grant
        wounded = sorted(set(conflicting))
        table = _release_table(locks, set(wounded))
        others = [(t, m) for (t, m) in table.get(key, ()) if t != txn_id]
        table[key] = tuple(sorted([*others, (txn_id, mode)]))
        return table, wounded
    return None, []  # a conflicting holder is older -> the requester aborts (no-wait)


def _tget_2pl(
    state: DistributedState, ev: EventAppend, txn: TxnState, key: str
) -> tuple[DistDelta, str, str]:
    """``tget`` under 2PL: a **shared** lock, then read. No version pinning (locks give serializability)."""
    buffered = txn.buffered_write(key)
    if buffered is not None:  # read-your-writes: already holds the X lock from its own tput
        return [ev, SetResult("ok", buffered)], "ok", buffered
    replica = state.replicas.get((key, txn.node))
    if replica is None:
        return [ev, SetResult("no_replica", "")], "no_replica", ""
    new_locks, wounded = _acquire(state.locks, txn.txn_id, key, "S")
    if new_locks is None:  # wounded by an older lock holder -> abort, releasing our own locks
        return _wound_self(state, ev, txn)
    edits: list[DistEdit] = [ev, *(TxnDel(w) for w in wounded),
                             *_lock_edits(state.locks, new_locks),
                             SetResult("ok", replica.value)]
    return edits, "ok", replica.value


def _tput_2pl(
    state: DistributedState, ev: EventAppend, txn: TxnState, key: str, value: str
) -> tuple[DistDelta, str, str]:
    """``tput`` under 2PL: an **exclusive** lock, then buffer the write (applied atomically at commit)."""
    new_locks, wounded = _acquire(state.locks, txn.txn_id, key, "X")
    if new_locks is None:
        return _wound_self(state, ev, txn)
    updated = TxnState(txn.txn_id, txn.node, txn.reads, (*txn.writes, (key, value)), txn.write_versions)
    edits: list[DistEdit] = [ev, *(TxnDel(w) for w in wounded),
                             *_lock_edits(state.locks, new_locks),
                             TxnSet(updated), SetResult("ok", value)]
    return edits, "ok", value


def _wound_self(
    state: DistributedState, ev: EventAppend, txn: TxnState
) -> tuple[DistDelta, str, str]:
    """The requesting txn loses the wound-wait race (younger): it aborts and releases its own locks."""
    releases = _release_edits(state.locks, [txn.txn_id])
    return [ev, *releases, TxnDel(txn.txn_id), SetResult("wounded", "")], "wounded", ""


def _commit(
    state: DistributedState, ev: EventAppend, txn: TxnState, config: DistConfig
) -> tuple[DistDelta, str, str]:
    two_pl = config.concurrency_control == "2pl"
    if not two_pl:
        # OCC validation — the isolation level decides *which* set is checked (SPEC-7 §3.2, incr 3):
        #   serializable: every **read** key's version must be unchanged since it was read (backward
        #                 validation). This catches the read another txn's write invalidated, so it
        #                 forbids write skew (A reads y, B writes y -> A aborts).
        #   snapshot:     only every **written** key's version must be unchanged since it was first
        #                 written (write-write conflict, first-committer-wins). A read another txn
        #                 wrote is *not* checked, so disjoint-write-set txns both commit -> write skew.
        # Under 2PL there is nothing to validate: the txn held its locks to here, so no concurrent
        # txn could have written a key it read or wrote (the locks already guaranteed serializability).
        validation_set = txn.reads if config.txn_isolation == "serializable" else txn.write_versions
        for key, pinned_version in validation_set:
            replica = state.replicas.get((key, txn.node))
            current = replica.version if replica is not None else 0
            if current != pinned_version:  # a concurrent committer won the race -> abort (first wins)
                return [ev, TxnDel(txn.txn_id), SetResult("conflict", "")], "conflict", ""

    # The set of keys to write, last buffered value per key, in deterministic (sorted) order.
    write_values: dict[str, str] = {}
    for key, value in txn.writes:
        write_values[key] = value
    keys = sorted(write_values)
    # 2PL holds locks until here; the commit releases every lock this txn holds (the shrinking phase).
    lock_releases = _release_edits(state.locks, [txn.txn_id]) if two_pl else []

    # A write to a key the coordinator does not replicate cannot be applied -> abort the txn.
    if any(state.replicas.get((key, txn.node)) is None for key in keys):
        return [ev, TxnDel(txn.txn_id), *lock_releases,
                SetResult("no_replica", "")], "no_replica", ""

    model = config.consistency_model
    # CP write-rejection: a committed write replicates under the *same* discipline as a plain ``put``
    # (§5.1) — synchronous to *all* replicas under ``linearizable``, to a reachable *majority* under
    # ``quorum``. A commit that cannot meet that quorum is rejected (``unavailable``) and the txn stays
    # **open** (locks held), so the caller may retry once the partition heals — it is not aborted.
    reach = {key: [p for p in config.replicas_of(key)
                   if state.connected(txn.node, p) and state.is_up(p)] for key in keys}
    if model == "linearizable" and any(len(reach[k]) < len(config.replicas_of(k)) for k in keys):
        return [ev, SetResult("unavailable", "")], "unavailable", ""
    if model == "quorum" and any(len(reach[k]) < len(config.replicas_of(k)) // 2 + 1 for k in keys):
        return [ev, SetResult("unavailable", "")], "unavailable", ""

    edits: list[DistEdit] = [ev]
    msg_id = state.next_msg_id
    for key in keys:
        replica = state.replicas[(key, txn.node)]
        new_version = replica.version + 1
        value = write_values[key]
        if model == "linearizable":
            edits.extend(
                ReplicaWrite(key, peer, new_version, value) for peer in config.replicas_of(key)
            )
        elif model == "quorum":
            # synchronous to the reachable majority; async catch-up to the unreachable minority.
            edits.extend(ReplicaWrite(key, peer, new_version, value) for peer in reach[key])
            for peer in config.replicas_of(key):
                if peer in reach[key]:
                    continue
                edits.append(
                    MsgSend(msg_id, txn.node, peer, key, new_version, value, state.clock + 1)
                )
                msg_id += 1
        else:  # eventual / causal: local write + async replication to the other replicas
            edits.append(ReplicaWrite(key, txn.node, new_version, value))
            for peer in config.replicas_of(key):
                if peer == txn.node:
                    continue
                edits.append(
                    MsgSend(msg_id, txn.node, peer, key, new_version, value, state.clock + 1)
                )
                msg_id += 1
    edits.extend(lock_releases)
    edits.append(TxnDel(txn.txn_id))
    edits.append(SetResult("committed", ""))
    return edits, "committed", ""
