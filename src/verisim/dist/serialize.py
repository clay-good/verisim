"""Canonical serialization of the distributed state (SPEC-7 §3.1, DS0 increment 1).

Canonicalization is mandatory in every world (SPEC-3 DD-1): a state serializes to a **sorted,
deterministic** structure so the divergence metric and goldens measure protocol competence, not map
ordering or identifier churn. ``to_canonical`` is the JSON normal form; ``from_canonical`` is its
exact inverse (round-trips by construction); ``state_hash`` is the content address for goldens and
the verified-contribution protocol. Pure and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from verisim.dist.state import DistributedState, Event, Message, ReplicaState, TxnState


def _msg_canonical(m: Message) -> dict[str, Any]:
    """One in-flight message's normal form. ``deps`` is omitted when empty so that an ``eventual`` /
    ``linearizable`` message serializes to the exact pre-DS0-incr-5 form (goldens/hashes unchanged)."""
    out: dict[str, Any] = {
        "id": m.id, "src": m.src, "dst": m.dst, "object_id": m.object_id, "version": m.version,
        "value": m.value, "deliver_after": m.deliver_after,
    }
    if m.deps:
        out["deps"] = [list(dep) for dep in m.deps]
    return out


def to_canonical(state: DistributedState) -> dict[str, Any]:
    """The sorted, deterministic normal form of a :class:`DistributedState`."""
    replicas = [
        {"object_id": r.object_id, "node_id": r.node_id, "version": r.version, "value": r.value}
        for r in sorted(state.replicas.values(), key=lambda r: (r.object_id, r.node_id))
    ]
    log = [
        {"id": e.id, "node": e.node, "op": e.op, "clock": e.clock,
         "happens_before": list(e.happens_before)}
        for e in sorted(state.log, key=lambda e: e.id)
    ]
    inflight = [
        _msg_canonical(m)
        for m in sorted(state.inflight.values(), key=lambda m: m.id)
    ]
    partitions = sorted(sorted(g) for g in state.partitions)
    out: dict[str, Any] = {
        "replicas": replicas,
        "log": log,
        "inflight": inflight,
        "partitions": partitions,
        "down": sorted(state.down),
        "clock": state.clock,
        "next_event_id": state.next_event_id,
        "next_msg_id": state.next_msg_id,
        "last_result": list(state.last_result) if state.last_result is not None else None,
    }
    # ``txns`` is included only when non-empty so a cluster with no open transactions serializes to
    # the exact DS0-increment-1 normal form (the goldens and contributed hashes predating DS0 incr 2
    # stay valid; the transaction substrate is purely additive).
    if state.txns:
        out["txns"] = [
            {"txn_id": t.txn_id, "node": t.node,
             "reads": [list(r) for r in t.reads], "writes": [list(w) for w in t.writes],
             "write_versions": [list(w) for w in t.write_versions]}
            for t in sorted(state.txns.values(), key=lambda t: t.txn_id)
        ]
    # ``locks`` is included only when non-empty, so an ``occ`` (lock-free) cluster serializes to the
    # exact pre-DS0-incr-8 normal form (the 2PL lock table is purely additive, like ``txns``).
    if state.locks:
        out["locks"] = {
            obj: [list(h) for h in holders] for obj, holders in sorted(state.locks.items())
        }
    return out


def from_canonical(d: dict[str, Any]) -> DistributedState:
    """Inverse of :func:`to_canonical` (exact round-trip)."""
    replicas = {
        (r["object_id"], r["node_id"]): ReplicaState(
            r["object_id"], r["node_id"], r["version"], r["value"]
        )
        for r in d["replicas"]
    }
    log = tuple(
        Event(e["id"], e["node"], e["op"], e["clock"], tuple(e["happens_before"]))
        for e in d["log"]
    )
    inflight = {
        m["id"]: Message(m["id"], m["src"], m["dst"], m["object_id"], m["version"],
                         m["value"], m["deliver_after"],
                         tuple((o, v) for o, v in m.get("deps", [])))
        for m in d["inflight"]
    }
    partitions = tuple(frozenset(g) for g in d["partitions"])
    last = d["last_result"]
    txns = {
        t["txn_id"]: TxnState(
            t["txn_id"], t["node"],
            tuple((k, v) for k, v in t["reads"]),
            tuple((k, val) for k, val in t["writes"]),
            tuple((k, v) for k, v in t.get("write_versions", [])),
        )
        for t in d.get("txns", [])
    }
    locks = {
        obj: tuple((t, m) for t, m in holders)
        for obj, holders in d.get("locks", {}).items()
    }
    return DistributedState(
        replicas=replicas,
        partitions=partitions,
        log=log,
        inflight=inflight,
        down=frozenset(d["down"]),
        clock=d["clock"],
        next_event_id=d["next_event_id"],
        next_msg_id=d["next_msg_id"],
        last_result=(last[0], last[1]) if last is not None else None,
        txns=txns,
        locks=locks,
    )


def to_json(state: DistributedState) -> str:
    """Compact canonical JSON (sorted keys) -- the stable wire/golden form."""
    return json.dumps(to_canonical(state), sort_keys=True, separators=(",", ":"))


def state_hash(state: DistributedState) -> str:
    """A content-address hash of the canonical state (goldens, §16 verified contribution)."""
    return hashlib.sha256(to_json(state).encode("utf-8")).hexdigest()[:16]
