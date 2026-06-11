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
    # The PN-counter decrement half (DS0 incr 29, `cdecr`): the N G-counter, same shape as `gcounters`.
    # Omitted when empty, so a cluster that never `cdecr`-s serializes to the exact pre-increment-29
    # form (purely additive over increment 28).
    ncounters = {k: c for k, c in state.ncounters.items() if c != 0}
    if ncounters:
        out["ncounters"] = [
            {"key": key, "holder": holder, "owner": owner, "count": ncounters[(key, holder, owner)]}
            for key, holder, owner in sorted(ncounters)
        ]
    # The CRDT OR-Set (DS0 incr 30, `sadd`/`srem`/`smembers`): per (key, holder), the observed add-dots
    # and tombstoned dots, flattened and sorted id-independently. Omitted when empty, so a cluster that
    # never uses a set serializes to the exact pre-increment-30 form (purely additive).
    adds = sorted((key, holder, elem, owner, seq)
                  for (key, holder), dots in state.orset_adds.items() if dots
                  for (elem, owner, seq) in dots)
    if adds:
        out["orset_adds"] = [
            {"key": key, "holder": holder, "elem": elem, "owner": owner, "seq": seq}
            for key, holder, elem, owner, seq in adds
        ]
    tombs = sorted((key, holder, owner, seq)
                   for (key, holder), dots in state.orset_tombs.items() if dots
                   for (owner, seq) in dots)
    if tombs:
        out["orset_tombs"] = [
            {"key": key, "holder": holder, "owner": owner, "seq": seq}
            for key, holder, owner, seq in tombs
        ]
    # The CRDT MV-register (DS0 incr 31, `mvput`/`mvget`): per (key, holder), the surviving write-dots
    # and superseded dots, same shape as the OR-Set. Omitted when empty (pre-increment-31 form).
    mvvals = sorted((key, holder, value, owner, seq)
                    for (key, holder), dots in state.mvreg_vals.items() if dots
                    for (value, owner, seq) in dots)
    if mvvals:
        out["mvreg_vals"] = [
            {"key": key, "holder": holder, "value": value, "owner": owner, "seq": seq}
            for key, holder, value, owner, seq in mvvals
        ]
    mvtombs = sorted((key, holder, owner, seq)
                     for (key, holder), dots in state.mvreg_tombs.items() if dots
                     for (owner, seq) in dots)
    if mvtombs:
        out["mvreg_tombs"] = [
            {"key": key, "holder": holder, "owner": owner, "seq": seq}
            for key, holder, owner, seq in mvtombs
        ]
    # The CRDT LWW-register (DS0 incr 32, `lwwput`/`lwwget`): each (key, holder) winning entry and each
    # node's Lamport clock, sorted. Omitted when empty (pre-increment-32 form, purely additive).
    if state.lwwreg:
        out["lwwreg"] = [
            {"key": key, "holder": holder, "value": v, "ts": ts, "owner": owner}
            for (key, holder), (v, ts, owner) in sorted(state.lwwreg.items())
        ]
    lamport = {n: t for n, t in state.lamport.items() if t != 0}
    if lamport:
        out["lamport"] = [{"holder": n, "value": lamport[n]} for n in sorted(lamport)]
    # The CRDT OR-Map (DS0 incr 33, `mput`/`mget`/`mdel`/`mkeys`): the field-presence dots + tombstones
    # (OR-Set half) and each field's LWW value (LWW-register half). Omitted when empty (pre-incr-33).
    omf = sorted((m, h, fld, o, s) for (m, h), dots in state.ormap_fields.items() if dots
                 for (fld, o, s) in dots)
    if omf:
        out["ormap_fields"] = [
            {"map": m, "holder": h, "field": fld, "owner": o, "seq": s} for m, h, fld, o, s in omf
        ]
    omt = sorted((m, h, o, s) for (m, h), dots in state.ormap_tombs.items() if dots
                 for (o, s) in dots)
    if omt:
        out["ormap_tombs"] = [
            {"map": m, "holder": h, "owner": o, "seq": s} for m, h, o, s in omt
        ]
    if state.ormap_vals:
        out["ormap_vals"] = [
            {"map": m, "field": fld, "holder": h, "value": v, "ts": ts, "owner": o}
            for ((m, fld), h), (v, ts, o) in sorted(state.ormap_vals.items())
        ]
    # The CRDT RGA sequence (DS0 incr 34, `rins`/`rdel`/`rget`): the elements (with parent ids) + the
    # tombstoned element ids, flattened and sorted. Omitted when empty (pre-increment-34 form).
    rels = sorted((ln, h, seq, owner, value, pseq, powner)
                  for (ln, h), es in state.rga_elems.items() if es
                  for (seq, owner, value, pseq, powner) in es)
    if rels:
        out["rga_elems"] = [
            {"list": ln, "holder": h, "seq": seq, "owner": owner, "value": value,
             "pseq": pseq, "powner": powner}
            for ln, h, seq, owner, value, pseq, powner in rels
        ]
    rtombs = sorted((ln, h, seq, owner)
                    for (ln, h), ts in state.rga_tombs.items() if ts
                    for (seq, owner) in ts)
    if rtombs:
        out["rga_tombs"] = [
            {"list": ln, "holder": h, "seq": seq, "owner": owner} for ln, h, seq, owner in rtombs
        ]
    # The CRDT counter-map (DS0 incr 35, `cminc`/`cmget`/`cmdel`/`cmkeys`): the OR-Set field-presence
    # dots + tombstones, and the per-field G-counter sub-counts. Omitted when empty (pre-incr-35 form).
    cmf = sorted((m, h, fld, o, s) for (m, h), dots in state.cmap_fields.items() if dots
                 for (fld, o, s) in dots)
    if cmf:
        out["cmap_fields"] = [
            {"map": m, "holder": h, "field": fld, "owner": o, "seq": s} for m, h, fld, o, s in cmf
        ]
    cmt = sorted((m, h, o, s) for (m, h), dots in state.cmap_tombs.items() if dots
                 for (o, s) in dots)
    if cmt:
        out["cmap_tombs"] = [
            {"map": m, "holder": h, "owner": o, "seq": s} for m, h, o, s in cmt
        ]
    cmc = sorted((m, fld, h, o, c) for ((m, fld), h, o), c in state.cmap_counts.items() if c != 0)
    if cmc:
        out["cmap_counts"] = [
            {"map": m, "field": fld, "holder": h, "owner": o, "count": c} for m, fld, h, o, c in cmc
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


def _group_orset_adds(rows: list[dict[str, Any]]) -> dict[tuple[str, str], frozenset[tuple[str, str, int]]]:
    """Re-group the flattened OR-Set add-dot rows (DS0 incr 30) into the per-(key, holder) frozensets."""
    grouped: dict[tuple[str, str], set[tuple[str, str, int]]] = {}
    for r in rows:
        grouped.setdefault((r["key"], r["holder"]), set()).add((r["elem"], r["owner"], r["seq"]))
    return {k: frozenset(v) for k, v in grouped.items()}


def _group_orset_tombs(rows: list[dict[str, Any]]) -> dict[tuple[str, str], frozenset[tuple[str, int]]]:
    """Re-group flattened tombstone rows (DS0 incr 30/31) into the per-(key, holder) frozensets.

    Shared by the OR-Set and MV-register tombstone halves (both are ``(owner, seq)`` dots)."""
    grouped: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for r in rows:
        grouped.setdefault((r["key"], r["holder"]), set()).add((r["owner"], r["seq"]))
    return {k: frozenset(v) for k, v in grouped.items()}


def _group_mvreg_vals(rows: list[dict[str, Any]]) -> dict[tuple[str, str], frozenset[tuple[str, str, int]]]:
    """Re-group the flattened MV-register write-dot rows (DS0 incr 31) into per-(key, holder) sets."""
    grouped: dict[tuple[str, str], set[tuple[str, str, int]]] = {}
    for r in rows:
        grouped.setdefault((r["key"], r["holder"]), set()).add((r["value"], r["owner"], r["seq"]))
    return {k: frozenset(v) for k, v in grouped.items()}


def _group_ormap_fields(rows: list[dict[str, Any]]) -> dict[tuple[str, str], frozenset[tuple[str, str, int]]]:
    """Re-group the flattened OR-Map presence-dot rows (DS0 incr 33) into per-(map, holder) sets."""
    grouped: dict[tuple[str, str], set[tuple[str, str, int]]] = {}
    for r in rows:
        grouped.setdefault((r["map"], r["holder"]), set()).add((r["field"], r["owner"], r["seq"]))
    return {k: frozenset(v) for k, v in grouped.items()}


def _group_ormap_tombs(rows: list[dict[str, Any]]) -> dict[tuple[str, str], frozenset[tuple[str, int]]]:
    """Re-group the flattened OR-Map tombstone rows (DS0 incr 33) into per-(map, holder) sets."""
    grouped: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for r in rows:
        grouped.setdefault((r["map"], r["holder"]), set()).add((r["owner"], r["seq"]))
    return {k: frozenset(v) for k, v in grouped.items()}


def _group_rga_elems(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], frozenset[tuple[int, str, str, int, str]]]:
    """Re-group the flattened RGA element rows (DS0 incr 34) into per-(list, holder) sets."""
    grouped: dict[tuple[str, str], set[tuple[int, str, str, int, str]]] = {}
    for r in rows:
        grouped.setdefault((r["list"], r["holder"]), set()).add(
            (r["seq"], r["owner"], r["value"], r["pseq"], r["powner"])
        )
    return {k: frozenset(v) for k, v in grouped.items()}


def _group_rga_tombs(rows: list[dict[str, Any]]) -> dict[tuple[str, str], frozenset[tuple[int, str]]]:
    """Re-group the flattened RGA tombstone rows (DS0 incr 34) into per-(list, holder) sets."""
    grouped: dict[tuple[str, str], set[tuple[int, str]]] = {}
    for r in rows:
        grouped.setdefault((r["list"], r["holder"]), set()).add((r["seq"], r["owner"]))
    return {k: frozenset(v) for k, v in grouped.items()}


def _group_cmap_fields(rows: list[dict[str, Any]]) -> dict[tuple[str, str], frozenset[tuple[str, str, int]]]:
    """Re-group the flattened counter-map presence-dot rows (DS0 incr 35) into per-(map, holder) sets."""
    grouped: dict[tuple[str, str], set[tuple[str, str, int]]] = {}
    for r in rows:
        grouped.setdefault((r["map"], r["holder"]), set()).add((r["field"], r["owner"], r["seq"]))
    return {k: frozenset(v) for k, v in grouped.items()}


def _group_cmap_tombs(rows: list[dict[str, Any]]) -> dict[tuple[str, str], frozenset[tuple[str, int]]]:
    """Re-group the flattened counter-map tombstone rows (DS0 incr 35) into per-(map, holder) sets."""
    grouped: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for r in rows:
        grouped.setdefault((r["map"], r["holder"]), set()).add((r["owner"], r["seq"]))
    return {k: frozenset(v) for k, v in grouped.items()}


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
        ncounters={
            (g["key"], g["holder"], g["owner"]): g["count"] for g in d.get("ncounters", [])
        },
        orset_adds=_group_orset_adds(d.get("orset_adds", [])),
        orset_tombs=_group_orset_tombs(d.get("orset_tombs", [])),
        mvreg_vals=_group_mvreg_vals(d.get("mvreg_vals", [])),
        mvreg_tombs=_group_orset_tombs(d.get("mvreg_tombs", [])),
        lwwreg={
            (r["key"], r["holder"]): (r["value"], r["ts"], r["owner"])
            for r in d.get("lwwreg", [])
        },
        lamport={r["holder"]: r["value"] for r in d.get("lamport", [])},
        ormap_fields=_group_ormap_fields(d.get("ormap_fields", [])),
        ormap_tombs=_group_ormap_tombs(d.get("ormap_tombs", [])),
        ormap_vals={
            ((r["map"], r["field"]), r["holder"]): (r["value"], r["ts"], r["owner"])
            for r in d.get("ormap_vals", [])
        },
        rga_elems=_group_rga_elems(d.get("rga_elems", [])),
        rga_tombs=_group_rga_tombs(d.get("rga_tombs", [])),
        cmap_fields=_group_cmap_fields(d.get("cmap_fields", [])),
        cmap_tombs=_group_cmap_tombs(d.get("cmap_tombs", [])),
        cmap_counts={
            ((r["map"], r["field"]), r["holder"], r["owner"]): r["count"]
            for r in d.get("cmap_counts", [])
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
