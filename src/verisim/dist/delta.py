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

from dataclasses import dataclass
from typing import Any

from verisim.dist.state import DistributedState, Event, Message, ReplicaState, TxnState


@dataclass(frozen=True)
class ReplicaWrite:
    """Set node ``node_id``'s replica of ``object_id`` to ``(version, value)`` (an MVCC bump)."""

    object_id: str
    node_id: str
    version: int
    value: str


@dataclass(frozen=True)
class MsgSend:
    """Enqueue an in-flight replication message (a write the coordinator propagates to a peer)."""

    msg_id: int
    src: str
    dst: str
    object_id: str
    version: int
    value: str
    deliver_after: int


@dataclass(frozen=True)
class MsgDeliver:
    """Remove a delivered message from the in-flight set (its effect is a paired ReplicaWrite)."""

    msg_id: int


@dataclass(frozen=True)
class MsgDrop:
    """Remove a dropped message from the in-flight set (no effect applied)."""

    msg_id: int


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
class TxnSet:
    """Upsert an active transaction's state (``begin``/``tget``/``tput`` produce this; DS0 incr2)."""

    txn: TxnState


@dataclass(frozen=True)
class TxnDel:
    """Remove an active transaction (``commit``/``abort`` produce this)."""

    txn_id: str


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
    | EventAppend
    | PartitionSet
    | NodeDown
    | NodeUp
    | ClockSet
    | TxnSet
    | TxnDel
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
                edit.version, edit.value, edit.deliver_after,
            )
            s.next_msg_id = max(s.next_msg_id, edit.msg_id + 1)
        elif isinstance(edit, (MsgDeliver, MsgDrop)):
            s.inflight.pop(edit.msg_id, None)
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
        elif isinstance(edit, TxnSet):
            s.txns[edit.txn.txn_id] = edit.txn
        elif isinstance(edit, TxnDel):
            s.txns.pop(edit.txn_id, None)
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
        return {"op": "MsgSend", "msg_id": edit.msg_id, "src": edit.src, "dst": edit.dst,
                "object_id": edit.object_id, "version": edit.version, "value": edit.value,
                "deliver_after": edit.deliver_after}
    if isinstance(edit, MsgDeliver):
        return {"op": "MsgDeliver", "msg_id": edit.msg_id}
    if isinstance(edit, MsgDrop):
        return {"op": "MsgDrop", "msg_id": edit.msg_id}
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
    if isinstance(edit, TxnSet):
        return {"op": "TxnSet", "txn_id": edit.txn.txn_id, "node": edit.txn.node,
                "reads": [list(r) for r in edit.txn.reads],
                "writes": [list(w) for w in edit.txn.writes]}
    if isinstance(edit, TxnDel):
        return {"op": "TxnDel", "txn_id": edit.txn_id}
    assert isinstance(edit, SetResult)
    return {"op": "SetResult", "status": edit.status, "value": edit.value}


def edit_from_dict(d: dict[str, Any]) -> DistEdit:
    """Inverse of :func:`edit_to_dict`."""
    op = d["op"]
    if op == "ReplicaWrite":
        return ReplicaWrite(d["object_id"], d["node_id"], d["version"], d["value"])
    if op == "MsgSend":
        return MsgSend(d["msg_id"], d["src"], d["dst"], d["object_id"], d["version"],
                       d["value"], d["deliver_after"])
    if op == "MsgDeliver":
        return MsgDeliver(d["msg_id"])
    if op == "MsgDrop":
        return MsgDrop(d["msg_id"])
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
    if op == "TxnSet":
        return TxnSet(TxnState(
            d["txn_id"], d["node"],
            tuple((k, v) for k, v in d["reads"]),
            tuple((k, val) for k, val in d["writes"]),
        ))
    if op == "TxnDel":
        return TxnDel(d["txn_id"])
    if op == "SetResult":
        return SetResult(d["status"], d["value"])
    raise ValueError(f"unknown edit op {op!r}")


def delta_to_list(delta: DistDelta) -> list[dict[str, Any]]:
    return [edit_to_dict(e) for e in delta]


def delta_from_list(items: list[dict[str, Any]]) -> DistDelta:
    return [edit_from_dict(d) for d in items]
