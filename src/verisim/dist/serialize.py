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

from verisim.dist.state import (
    DistributedState,
    Event,
    LogEntry,
    Message,
    ReplicaState,
    TxnState,
)
from verisim.host.state import from_canonical_host, to_canonical_host


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
    # ``skew`` is included only when non-empty, so a cluster with synchronized clocks serializes to
    # the exact pre-DS0-incr-14 normal form (per-node clock offsets are purely additive, like
    # ``locks``/``txns``); a 0 offset is never stored (it is cleared on apply).
    if state.skew:
        out["skew"] = {node: off for node, off in sorted(state.skew.items())}
    # ``term``/``leader`` are included only once an election has happened, so a cluster that never
    # runs consensus serializes to the exact pre-DS0-incr-16 normal form (the Raft-subset
    # leader/term metadata is purely additive, like ``skew``/``locks``/``txns``).
    if state.term != 0 or state.leader is not None:
        out["term"] = state.term
        out["leader"] = state.leader
    # ``lease_until`` (DS0 incr 18, the leader lease) is included only once a lease has been granted;
    # ``0`` (the boot/no-lease default) is omitted, so a cluster that never leases serializes to the
    # exact pre-increment-18 form (purely additive, like ``skew``/``term``/``leader``).
    if state.lease_until != 0:
        out["lease_until"] = state.lease_until
    # The Raft log (DS0 incr 19): nodes with a non-empty log, sorted, each as an ordered entry list;
    # ``commit_index`` when non-zero. Both omitted at their empty/0 defaults, so a cluster that never
    # appends serializes to the exact pre-increment-19 form (purely additive, like term/leader/lease).
    logs = {node: entries for node, entries in state.logs.items() if entries}
    if logs:
        out["logs"] = [
            {"node": node, "entries": [
                {"term": e.term, "index": e.index, "key": e.key, "value": e.value}
                for e in logs[node]
            ]}
            for node in sorted(logs)
        ]
    if state.commit_index != 0:
        out["commit_index"] = state.commit_index
    # The consensus voting membership (DS0 incr 20): the empty frozenset (the "all config nodes vote"
    # sentinel / boot default) is omitted, so a cluster that never reconfigures serializes to the
    # exact pre-increment-20 form (purely additive). A non-empty (reconfigured) set serializes sorted.
    if state.members:
        out["members"] = sorted(state.members)
    # Distributed queues (DS0 incr 21): non-empty (queue, node) replicas, sorted, each an ordered
    # item list. Omitted when no queue has been used, so a KV-only cluster serializes to the exact
    # pre-increment-21 form (purely additive, like members/logs/lease).
    queues = {k: v for k, v in state.queues.items() if v}
    if queues:
        out["queues"] = [
            {"queue": q, "node": n, "items": list(queues[(q, n)])}
            for q, n in sorted(queues)
        ]
    # Per-node running versions (DS0 incr 22): non-base versions, sorted by node. Omitted when every
    # node is at the base version 0, so a cluster that never deploys serializes to the exact
    # pre-increment-22 form (purely additive, like queues/members/lease).
    versions = {node: v for node, v in state.versions.items() if v != 0}
    if versions:
        out["versions"] = {node: versions[node] for node in sorted(versions)}
    # The cluster configuration (DS0 incr 24, `config_push`): pushed per-(node, key) values, sorted.
    # Omitted when empty, so a cluster that never pushes config serializes to the exact
    # pre-increment-24 form (purely additive, like versions/queues/members/lease).
    if state.config:
        out["config"] = [
            {"node": node, "key": key, "value": state.config[(node, key)]}
            for node, key in sorted(state.config)
        ]
    # The CRDT G-counters (DS0 incr 28, `cincr`/`cget`): non-zero (key, holder, owner) sub-counts,
    # sorted. Omitted when empty, so a cluster that never uses a CRDT counter serializes to the exact
    # pre-increment-28 form (purely additive, like config/versions/queues).
    gcounters = {k: c for k, c in state.gcounters.items() if c != 0}
    if gcounters:
        out["gcounters"] = [
            {"key": key, "holder": holder, "owner": owner, "count": gcounters[(key, holder, owner)]}
            for key, holder, owner in sorted(gcounters)
        ]
    # The embedded SPEC-6 hosts (DS0 incr 23): per-node host canonical form (the v0 FS reuses its own
    # canonical verbatim — the composition is visible in serialization). Omitted when no node runs a
    # host op, so a host-free cluster serializes to the exact pre-increment-23 form (purely additive).
    if state.hosts:
        out["hosts"] = [
            {"node": node, "host": to_canonical_host(state.hosts[node])}
            for node in sorted(state.hosts)
        ]
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
    skew = {node: off for node, off in d.get("skew", {}).items()}
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
        skew=skew,
        term=d.get("term", 0),
        leader=d.get("leader"),
        logs={
            entry["node"]: tuple(
                LogEntry(e["term"], e["index"], e["key"], e["value"]) for e in entry["entries"]
            )
            for entry in d.get("logs", [])
        },
        commit_index=d.get("commit_index", 0),
        members=frozenset(d.get("members", [])),
        queues={
            (q["queue"], q["node"]): tuple(q["items"]) for q in d.get("queues", [])
        },
        versions=dict(d.get("versions", {})),
        config={(c["node"], c["key"]): c["value"] for c in d.get("config", [])},
        gcounters={
            (g["key"], g["holder"], g["owner"]): g["count"] for g in d.get("gcounters", [])
        },
        hosts={h["node"]: from_canonical_host(h["host"]) for h in d.get("hosts", [])},
        lease_until=d.get("lease_until", 0),
    )


def to_json(state: DistributedState) -> str:
    """Compact canonical JSON (sorted keys) -- the stable wire/golden form."""
    return json.dumps(to_canonical(state), sort_keys=True, separators=(",", ":"))


def state_hash(state: DistributedState) -> str:
    """A content-address hash of the canonical state (goldens, §16 verified contribution)."""
    return hashlib.sha256(to_json(state).encode("utf-8")).hexdigest()[:16]
