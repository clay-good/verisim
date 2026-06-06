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
    MsgSend,
    ReplicaWrite,
    SetResult,
    TxnDel,
    TxnSet,
)
from verisim.dist.state import DistributedState, TxnState


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

    if name == "tget":
        return _tget(state, ev, txn, action.args[2])
    if name == "tput":
        return _tput(ev, txn, action.args[2], action.args[3])
    if name == "commit":
        return _commit(state, ev, txn, config)
    if name == "abort":
        return [ev, TxnDel(txn_id), SetResult("aborted", "")], "aborted", ""
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
        updated = TxnState(txn.txn_id, txn.node, (*txn.reads, (key, replica.version)), txn.writes)
        return [ev, TxnSet(updated), SetResult("ok", replica.value)], "ok", replica.value
    return [ev, SetResult("ok", replica.value)], "ok", replica.value  # re-read: version pinned


def _tput(
    ev: EventAppend, txn: TxnState, key: str, value: str
) -> tuple[DistDelta, str, str]:
    updated = TxnState(txn.txn_id, txn.node, txn.reads, (*txn.writes, (key, value)))
    return [ev, TxnSet(updated), SetResult("ok", value)], "ok", value


def _commit(
    state: DistributedState, ev: EventAppend, txn: TxnState, config: DistConfig
) -> tuple[DistDelta, str, str]:
    # OCC validation: every read key's local version must be unchanged since it was read.
    for key, read_version in txn.reads:
        replica = state.replicas.get((key, txn.node))
        current = replica.version if replica is not None else 0
        if current != read_version:  # a concurrent committer won the race -> abort (first wins)
            return [ev, TxnDel(txn.txn_id), SetResult("conflict", "")], "conflict", ""

    # The set of keys to write, last buffered value per key, in deterministic (sorted) order.
    write_values: dict[str, str] = {}
    for key, value in txn.writes:
        write_values[key] = value
    keys = sorted(write_values)

    # A write to a key the coordinator does not replicate cannot be applied -> abort the txn.
    if any(state.replicas.get((key, txn.node)) is None for key in keys):
        return [ev, TxnDel(txn.txn_id), SetResult("no_replica", "")], "no_replica", ""

    linearizable = config.consistency_model == "linearizable"
    if linearizable and any(
        not (state.connected(txn.node, peer) and state.is_up(peer))
        for key in keys
        for peer in config.replicas_of(key)
    ):
        # CP under partition: a synchronous commit that cannot reach all replicas is rejected; the
        # transaction stays open so the caller may retry once the partition heals (not aborted).
        return [ev, SetResult("unavailable", "")], "unavailable", ""

    edits: list[DistEdit] = [ev]
    msg_id = state.next_msg_id
    for key in keys:
        replica = state.replicas[(key, txn.node)]
        new_version = replica.version + 1
        value = write_values[key]
        if linearizable:
            edits.extend(
                ReplicaWrite(key, peer, new_version, value) for peer in config.replicas_of(key)
            )
        else:
            edits.append(ReplicaWrite(key, txn.node, new_version, value))
            for peer in config.replicas_of(key):
                if peer == txn.node:
                    continue
                edits.append(
                    MsgSend(msg_id, txn.node, peer, key, new_version, value, state.clock + 1)
                )
                msg_id += 1
    edits.append(TxnDel(txn.txn_id))
    edits.append(SetResult("committed", ""))
    return edits, "committed", ""
