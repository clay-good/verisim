"""The distributed state delta `Δ` and its `apply` (SPEC-7 §4, DS0 increment 1).

`M_θ` predicts a structured **log/replica delta**, not a full global state and not raw wire bytes
(SPEC-7 §2.7). DS0 increment 1 ships the replication/log/fault edit types; the embedded SPEC-6
``HostDelta`` / SPEC-5 ``NetDelta`` (applied verbatim to the host/net inside each node) and the
consensus/transaction edits are later increments.

The **M1-analogue invariant** is required and tested (DS1, but already by construction here):
``apply(state, oracle.delta) == oracle.next_state`` for every transition. ``apply`` is a pure
function over a fresh copy, and delta<->serialization round-trips, keeping the loop model-agnostic.
The allocator bumps (``next_event_id`` / ``next_msg_id``) are folded into the ``EventAppend`` /
``MsgSend`` edits, so the delta carries everything needed to reconstruct the next state exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from verisim.dist.state import (
    DistributedState,
    Event,
    LogEntry,
    Message,
    ReplicaState,
    TxnState,
)


@dataclass(frozen=True)
class ReplicaWrite:
    """Set node ``node_id``'s replica of ``object_id`` to ``(version, value)`` (an MVCC bump)."""

    object_id: str
    node_id: str
    version: int
    value: str


@dataclass(frozen=True)
class MsgSend:
    """Enqueue an in-flight replication message (a write the coordinator propagates to a peer).

    ``deps`` is the message's causal context (DS0 increment 5, the ``causal`` model) -- empty under
    ``eventual`` / ``linearizable``; see :class:`verisim.dist.state.Message`.
    """

    msg_id: int
    src: str
    dst: str
    object_id: str
    version: int
    value: str
    deliver_after: int
    deps: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class MsgDeliver:
    """Remove a delivered message from the in-flight set (its effect is a paired ReplicaWrite)."""

    msg_id: int


@dataclass(frozen=True)
class MsgDrop:
    """Remove a dropped message from the in-flight set (no effect applied)."""

    msg_id: int


@dataclass(frozen=True)
class MsgReschedule:
    """Change an in-flight message's ``deliver_after`` (the ``delay`` / ``reorder`` faults, incr 13).

    The message keeps its identity and payload; only *when* it becomes deliverable moves. ``delay``
    pushes it later (a recoverable delay -- the counterpart to ``drop``'s unrecoverable loss);
    ``reorder`` reverses the delivery schedule of a channel's messages (the multiset of times is
    preserved, the order flipped). It edits the ``Message.deliver_after`` field that already exists,
    so it adds **no new state field** and every prior golden/hash/tokenization is byte-for-byte
    unchanged. Applying it to a missing ``msg_id`` (already delivered/dropped) is a no-op.
    """

    msg_id: int
    deliver_after: int


@dataclass(frozen=True)
class EventAppend:
    """Append a causal-log event; bumps ``next_event_id`` to ``id + 1``."""

    id: int
    node: str
    op: str
    clock: int
    happens_before: tuple[int, ...] = ()


@dataclass(frozen=True)
class PartitionSet:
    """Replace the partition groups (``heal`` sets one all-nodes group). Groups are node tuples."""

    groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class NodeDown:
    """Mark ``node`` crashed (it stops delivering/applying until ``NodeUp``)."""

    node: str


@dataclass(frozen=True)
class NodeUp:
    """Mark ``node`` back up."""

    node: str


@dataclass(frozen=True)
class ClockSet:
    """Set the simulation clock to ``clock`` (``advance`` sets the new absolute time)."""

    clock: int


@dataclass(frozen=True)
class ClockSkewSet:
    """Set node ``node``'s clock offset (the ``clock_skew`` fault, DS0 increment 14).

    A positive ``offset`` is a clock running *ahead* (its sends are stamped later); a negative one is
    *behind* (stamped earlier). An ``offset`` of 0 **clears** the node's skew, so a synchronized
    cluster carries no ``skew`` residue and serializes to the pre-increment-14 form.
    """

    node: str
    offset: int


@dataclass(frozen=True)
class TxnSet:
    """Upsert an active transaction's buffered state (``begin``/``tget``/``tput`` produce this)."""

    txn: TxnState


@dataclass(frozen=True)
class TxnDel:
    """Remove an active transaction (``commit``/``abort`` produce this)."""

    txn_id: str


@dataclass(frozen=True)
class LockSet:
    """Set the 2PL lock holders for ``object_id`` (DS0 increment 8); empty ``holders`` removes the key.

    ``holders`` is the sorted tuple of ``(txn_id, mode)`` with ``mode ∈ {"S", "X"}``. Replacing the
    whole holder set per key keeps ``apply`` a pure function and the delta round-trippable.
    """

    object_id: str
    holders: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProtocolStep:
    """A consensus protocol step: set the cluster ``leader`` + ``term`` (DS0 increment 16, §4).

    The SPEC-7 §4 ``ProtocolStep(kind, term, index, leader?)`` edit, specialized to the Raft-subset
    leader-election core: ``kind == "elect"`` installs ``leader`` at the new ``term`` (a strict
    monotone bump); ``kind == "step_down"`` (DS0 increment 17) clears ``leader`` (``→ None``) at the
    *same* ``term`` — voluntary relinquishment leaving the cluster leaderless until a fresh election.
    It changes no replica — leadership is cluster metadata, separate from the data plane — so it
    composes with every consistency model and applies to the protocol layer alone. The fields
    ``term``/``leader`` are at their boot defaults (``0``/``None``) until the first election, so a
    cluster that never elects serializes to the exact pre-increment-16 form.
    """

    kind: str
    term: int
    leader: str | None


@dataclass(frozen=True)
class LeaseSet:
    """Set the cluster's leader lease deadline (DS0 increment 18, the `ProtocolStep` lease form, §4).

    The §4 `ProtocolStep(kind, ..., lease)` lease/view-change form, specialized to the Raft leader-
    lease: ``until`` is the global-clock deadline through which the current leader may serve **local
    linearizable reads without a quorum** (`lread`), and through which a new election is **blocked**
    (a successor must wait out the incumbent's lease — the leader-lease safety property). ``lease``
    grants/extends it; ``elect`` and ``step_down`` clear it (``until == 0``) so a new leader starts
    with no lease and a leaderless cluster holds none. ``until == 0`` is the boot/no-lease default and
    is **omitted from the canonical form**, so a cluster that never leases serializes to the exact
    pre-increment-18 form and every prior golden/hash is unchanged.
    """

    until: int


@dataclass(frozen=True)
class LogSet:
    """Set one node's replicated log to ``entries`` (DS0 increment 19, the Raft log, `append`).

    Setting the whole log (rather than an incremental append) makes a follower **adopt the leader's
    prefix in one edit** — appending the new entry and, when the follower's tail diverged, truncating
    the conflicting uncommitted entries in the same step (the log-matching reconciliation). ``entries``
    is the node's full log after the step. Empty logs are omitted from the canonical form, so a
    cluster that never appends is byte-identical to the pre-increment-19 form.
    """

    node: str
    entries: tuple[LogEntry, ...]


@dataclass(frozen=True)
class CommitIndexSet:
    """Advance the committed-prefix length (DS0 increment 19): entries ``0..index-1`` are committed.

    The commit index is **monotone** — committed entries are permanent (a Raft safety invariant the
    metamorphic tier enforces). It advances only when a log entry reaches a majority of nodes. ``0``
    (nothing committed) is the default and is omitted from the canonical form.
    """

    index: int


@dataclass(frozen=True)
class SetResult:
    """The client-visible result of the step: ``(status, value_token)``."""

    status: str
    value: str


DistEdit = (
    ReplicaWrite
    | MsgSend
    | MsgDeliver
    | MsgDrop
    | MsgReschedule
    | EventAppend
    | PartitionSet
    | NodeDown
    | NodeUp
    | ClockSet
    | ClockSkewSet
    | TxnSet
    | TxnDel
    | LockSet
    | ProtocolStep
    | LeaseSet
    | LogSet
    | CommitIndexSet
    | SetResult
)
DistDelta = list[DistEdit]


def apply(state: DistributedState, delta: DistDelta) -> DistributedState:
    """Apply ``delta`` to a fresh copy of ``state`` and return the next state (the M1 invariant)."""
    s = state.copy()
    for edit in delta:
        if isinstance(edit, ReplicaWrite):
            s.replicas[(edit.object_id, edit.node_id)] = ReplicaState(
                edit.object_id, edit.node_id, edit.version, edit.value
            )
        elif isinstance(edit, MsgSend):
            s.inflight[edit.msg_id] = Message(
                edit.msg_id, edit.src, edit.dst, edit.object_id,
                edit.version, edit.value, edit.deliver_after, edit.deps,
            )
            s.next_msg_id = max(s.next_msg_id, edit.msg_id + 1)
        elif isinstance(edit, (MsgDeliver, MsgDrop)):
            s.inflight.pop(edit.msg_id, None)
        elif isinstance(edit, MsgReschedule):
            msg = s.inflight.get(edit.msg_id)
            if msg is not None:  # no-op if already delivered/dropped
                s.inflight[edit.msg_id] = replace(msg, deliver_after=edit.deliver_after)
        elif isinstance(edit, EventAppend):
            s.log = (*s.log, Event(edit.id, edit.node, edit.op, edit.clock, edit.happens_before))
            s.next_event_id = max(s.next_event_id, edit.id + 1)
        elif isinstance(edit, PartitionSet):
            # canonical (sorted) group order: partitions is conceptually a set, and this keeps
            # equality + the to_canonical/from_canonical round-trip exact (§16, the verified-
            # contribution protocol re-executes through from_canonical and compares bit-for-bit).
            s.partitions = tuple(sorted((frozenset(g) for g in edit.groups), key=sorted))
        elif isinstance(edit, NodeDown):
            s.down = s.down | {edit.node}
        elif isinstance(edit, NodeUp):
            s.down = s.down - {edit.node}
        elif isinstance(edit, ClockSet):
            s.clock = edit.clock
        elif isinstance(edit, ClockSkewSet):
            if edit.offset == 0:
                s.skew.pop(edit.node, None)  # a 0 offset clears the skew (no residue)
            else:
                s.skew[edit.node] = edit.offset
        elif isinstance(edit, TxnSet):
            s.txns[edit.txn.txn_id] = edit.txn
        elif isinstance(edit, TxnDel):
            s.txns.pop(edit.txn_id, None)
        elif isinstance(edit, LockSet):
            if edit.holders:
                s.locks[edit.object_id] = edit.holders
            else:
                s.locks.pop(edit.object_id, None)
        elif isinstance(edit, ProtocolStep):
            s.term = edit.term
            s.leader = edit.leader
        elif isinstance(edit, LeaseSet):
            s.lease_until = edit.until
        elif isinstance(edit, LogSet):
            if edit.entries:
                s.logs[edit.node] = edit.entries
            else:
                s.logs.pop(edit.node, None)  # an empty log leaves no residue in the canonical form
        elif isinstance(edit, CommitIndexSet):
            s.commit_index = edit.index
        else:
            assert isinstance(edit, SetResult)
            s.last_result = (edit.status, edit.value)
    return s


# --- serialization (delta <-> JSON-able list of dicts; round-trips by construction) --------------

def edit_to_dict(edit: DistEdit) -> dict[str, Any]:
    """Serialize one edit to a JSON-able dict tagged by ``op`` (its class name)."""
    if isinstance(edit, ReplicaWrite):
        return {"op": "ReplicaWrite", "object_id": edit.object_id, "node_id": edit.node_id,
                "version": edit.version, "value": edit.value}
    if isinstance(edit, MsgSend):
        d: dict[str, Any] = {"op": "MsgSend", "msg_id": edit.msg_id, "src": edit.src,
                             "dst": edit.dst, "object_id": edit.object_id, "version": edit.version,
                             "value": edit.value, "deliver_after": edit.deliver_after}
        if edit.deps:  # omitted when empty so eventual/linearizable deltas keep their prior form
            d["deps"] = [list(dep) for dep in edit.deps]
        return d
    if isinstance(edit, MsgDeliver):
        return {"op": "MsgDeliver", "msg_id": edit.msg_id}
    if isinstance(edit, MsgDrop):
        return {"op": "MsgDrop", "msg_id": edit.msg_id}
    if isinstance(edit, MsgReschedule):
        return {"op": "MsgReschedule", "msg_id": edit.msg_id, "deliver_after": edit.deliver_after}
    if isinstance(edit, EventAppend):
        return {"op": "EventAppend", "id": edit.id, "node": edit.node, "op_str": edit.op,
                "clock": edit.clock, "happens_before": list(edit.happens_before)}
    if isinstance(edit, PartitionSet):
        return {"op": "PartitionSet", "groups": [list(g) for g in edit.groups]}
    if isinstance(edit, NodeDown):
        return {"op": "NodeDown", "node": edit.node}
    if isinstance(edit, NodeUp):
        return {"op": "NodeUp", "node": edit.node}
    if isinstance(edit, ClockSet):
        return {"op": "ClockSet", "clock": edit.clock}
    if isinstance(edit, ClockSkewSet):
        return {"op": "ClockSkewSet", "node": edit.node, "offset": edit.offset}
    if isinstance(edit, TxnSet):
        return {"op": "TxnSet", "txn_id": edit.txn.txn_id, "node": edit.txn.node,
                "reads": [list(r) for r in edit.txn.reads],
                "writes": [list(w) for w in edit.txn.writes],
                "write_versions": [list(w) for w in edit.txn.write_versions]}
    if isinstance(edit, TxnDel):
        return {"op": "TxnDel", "txn_id": edit.txn_id}
    if isinstance(edit, LockSet):
        return {"op": "LockSet", "object_id": edit.object_id,
                "holders": [list(h) for h in edit.holders]}
    if isinstance(edit, ProtocolStep):
        return {"op": "ProtocolStep", "kind": edit.kind, "term": edit.term, "leader": edit.leader}
    if isinstance(edit, LeaseSet):
        return {"op": "LeaseSet", "until": edit.until}
    if isinstance(edit, LogSet):
        return {"op": "LogSet", "node": edit.node,
                "entries": [[e.term, e.index, e.key, e.value] for e in edit.entries]}
    if isinstance(edit, CommitIndexSet):
        return {"op": "CommitIndexSet", "index": edit.index}
    assert isinstance(edit, SetResult)
    return {"op": "SetResult", "status": edit.status, "value": edit.value}


def edit_from_dict(d: dict[str, Any]) -> DistEdit:
    """Inverse of :func:`edit_to_dict`."""
    op = d["op"]
    if op == "ReplicaWrite":
        return ReplicaWrite(d["object_id"], d["node_id"], d["version"], d["value"])
    if op == "MsgSend":
        return MsgSend(d["msg_id"], d["src"], d["dst"], d["object_id"], d["version"],
                       d["value"], d["deliver_after"],
                       tuple((o, v) for o, v in d.get("deps", [])))
    if op == "MsgDeliver":
        return MsgDeliver(d["msg_id"])
    if op == "MsgDrop":
        return MsgDrop(d["msg_id"])
    if op == "MsgReschedule":
        return MsgReschedule(d["msg_id"], d["deliver_after"])
    if op == "EventAppend":
        return EventAppend(d["id"], d["node"], d["op_str"], d["clock"],
                           tuple(d["happens_before"]))
    if op == "PartitionSet":
        return PartitionSet(tuple(tuple(g) for g in d["groups"]))
    if op == "NodeDown":
        return NodeDown(d["node"])
    if op == "NodeUp":
        return NodeUp(d["node"])
    if op == "ClockSet":
        return ClockSet(d["clock"])
    if op == "ClockSkewSet":
        return ClockSkewSet(d["node"], d["offset"])
    if op == "TxnSet":
        return TxnSet(TxnState(
            d["txn_id"], d["node"],
            tuple((k, v) for k, v in d["reads"]),
            tuple((k, val) for k, val in d["writes"]),
            tuple((k, v) for k, v in d.get("write_versions", [])),
        ))
    if op == "TxnDel":
        return TxnDel(d["txn_id"])
    if op == "LockSet":
        return LockSet(d["object_id"], tuple((t, m) for t, m in d["holders"]))
    if op == "ProtocolStep":
        return ProtocolStep(d["kind"], d["term"], d["leader"])
    if op == "LeaseSet":
        return LeaseSet(d["until"])
    if op == "LogSet":
        return LogSet(d["node"], tuple(LogEntry(t, i, k, v) for t, i, k, v in d["entries"]))
    if op == "CommitIndexSet":
        return CommitIndexSet(d["index"])
    if op == "SetResult":
        return SetResult(d["status"], d["value"])
    raise ValueError(f"unknown edit op {op!r}")


def delta_to_list(delta: DistDelta) -> list[dict[str, Any]]:
    return [edit_to_dict(e) for e in delta]


def delta_from_list(items: list[dict[str, Any]]) -> DistDelta:
    return [edit_from_dict(d) for d in items]
