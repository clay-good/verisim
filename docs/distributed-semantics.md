# Distributed semantics (SPEC-7, DS0)

The normative semantics of the **Tier-A reference distributed oracle**
([`verisim.distoracle.reference`](../src/verisim/distoracle/reference.py)) — the from-scratch
deterministic discrete-event simulator that is the executable ground truth for the distributed world.
This document is to SPEC-7 what [`network-semantics.md`](./network-semantics.md) is to SPEC-5 and
[`host-semantics.md`](./host-semantics.md) is to SPEC-6: it pins, in prose, exactly what the oracle
computes, so the golden trajectories ([`tests/test_dist_goldens.py`](../tests/test_dist_goldens.py))
and the `apply == oracle` invariant ([`tests/test_dist_core.py`](../tests/test_dist_core.py)) have a
specification to be checked against.

> **Scope — DS0 increment 1.** This covers the shipped slice: a **fully-replicated key-value store
> under asynchronous replication and the fault/time medium** (the eventual-consistency core that
> makes stale-reads-under-partition the central dynamic). The Raft-subset consensus group,
> transactions/locks, the embedded SPEC-6 host inside each node, and the cheap/Tier-B oracle tiers
> are later DS increments (SPEC-7 §5, §13); each will extend this document.

## 1. State

A `DistributedState` ([`verisim.dist.state`](../src/verisim/dist/state.py)) is **not** a single tree
(SPEC-2), graph (SPEC-5), or bundle (SPEC-6). It is:

| Field | Meaning |
|---|---|
| `replicas: {(object_id, node_id) → ReplicaState}` | each node's MVCC `(version, value)` copy of a logical object |
| `log: (Event, …)` | the append-only causal event log (one event per client op; `happens_before` = program order on the same node) |
| `inflight: {msg_id → Message}` | replication messages sent but not yet delivered |
| `partitions: (frozenset[node], …)` | disjoint groups that cover every node; two nodes can exchange messages iff they share a group (one all-nodes group = healed) |
| `down: frozenset[node]` | crashed nodes |
| `clock: int` | the simulation clock (a logical timestamp advanced by `advance`) |
| `next_event_id`, `next_msg_id` | monotone canonical id allocators |
| `last_result: (status, value)?` | the client-visible result of the last action |

**There is no `global` state field** (W7, SPEC-7 §3.1): a consistent global snapshot is a *derived,
coordinated* read, never stored — which is exactly why under partition two replicas legitimately
disagree. The boot state (`DistributedState.initial(config)`) places every object on the first
`replication_factor` nodes at `version 0` holding `config.default_value` ("nil"), with no faults.

## 2. The MVCC version & last-writer-wins

Each replica carries a per-object **version** (a Lamport-style counter the coordinator bumps `+1` on
each local write). On convergence, a replica adopts an incoming `(version, value)` iff it is
**greater by `(version, value)` lexicographically** than what it holds — *last-writer-wins by version,
ties broken deterministically by the value token*. This makes the store **eventually consistent**:
once every replication message is delivered, all replicas of an object hold the same `(version,
value)`, regardless of the order messages arrived in.

## 3. Actions

### Client ops (append a causal-log event; set `last_result`)

| Action | Semantics |
|---|---|
| `put <node> <key> <val>` | write `node`'s local replica to `(local_version + 1, val)` immediately; enqueue an async `MsgSend` to every other replica of `key` (`deliver_after = clock + 1`). `("ok", val)`. |
| `get <node> <key>` | return `node`'s **local** replica value — which under partition may be **stale**. `("ok", value)`. |
| `cas <node> <key> <old> <new>` | if `node`'s local value `== old`, behave as `put node key new`; else no write, `("conflict", local_value)`. |

If the coordinator node is **down**, a client op returns `("unavailable", "")` and makes no state
change beyond logging the attempt; if the node holds no replica of the key, `("no_replica", "")`.

### Fault / time ops (the medium — no causal-log event)

| Action | Semantics |
|---|---|
| `advance <dt>` | set `clock += dt`; deliver every in-flight message that is now **due** (`deliver_after ≤ clock`) **and reachable** (`src`/`dst` share a partition group, `dst` is up), in `msg_id` order, applying last-writer-wins; messages not due/reachable stay in-flight. `("advanced", str(num_delivered))`. |
| `partition <nodes> \| <nodes> [\| …]` | split the network into the named groups; any **unmentioned** node forms its own isolated group. `("ok", "")`. |
| `heal` | one all-nodes group (fully connected). `("ok", "")`. |
| `crash <node>` | `node` goes down: it stops delivering/applying messages until restarted. `("ok", "")`. |
| `restart <node>` | `node` comes back up (its replicas are whatever they were when it crashed; pending messages can now deliver). `("ok", "")`. |

`advance` is the engine of the distributed dynamics — it is where replication actually happens and
where partition/crash make their effect felt. Delivery is simulated **sequentially within one
`advance`**, so a later message's last-writer-wins comparison sees the effect of earlier deliveries in
the same step.

## 4. The delta and the `apply == oracle` invariant

The oracle returns the next state **and** the structured delta that produces it
([`verisim.dist.delta`](../src/verisim/dist/delta.py)): `ReplicaWrite`, `MsgSend`/`MsgDeliver`/
`MsgDrop`, `EventAppend`, `PartitionSet`, `NodeDown`/`NodeUp`, `ClockSet`, `SetResult`. The
**M1-analogue invariant** holds by construction and is tested on every transition:

```
apply(state, oracle.step(state, action).delta) == oracle.step(state, action).state
```

`apply` is a pure function over a fresh copy; the id-allocator bumps are folded into `EventAppend`
(`next_event_id → id+1`) and `MsgSend` (`next_msg_id → msg_id+1`), so the delta alone reconstructs the
next state exactly. This is what keeps the loop (a later DS increment) model-agnostic: a learned
`M_θ` predicts the same delta vocabulary, and the oracle's job is to verify/correct it.

## 5. Determinism

The world is a **pure function of `(state, action)`** (SPEC-7 §3.3): there is no RNG inside the
oracle — all nondeterminism (which messages exist, when they deliver, when/where the network splits)
is made an explicit, seeded *input* via the action stream (the `BUGGIFY` of deterministic-simulation
testing, SPEC-7 §2.1). A given initial config + action sequence replays bit-for-bit, which is what
makes the distributed oracle a free, reproducible ground-truth factory despite asynchrony being the
one nondeterminism source that cannot be sealed cheaply (HW-5).

## 6. Canonicalization

`to_canonical` ([`verisim.dist.serialize`](../src/verisim/dist/serialize.py)) sorts every map
(replicas by `(object, node)`, log by id, inflight by id, partition groups lexicographically) so the
divergence metric and goldens measure **protocol competence, not map ordering or identifier churn**
(SPEC-3 DD-1). `from_canonical` is its exact inverse; `state_hash` is the content address used by the
goldens and the §16 verified-contribution protocol.

## 7. Worked example (golden B)

`put n0 x b · advance 2 · partition n0 n1 | n2 · put n0 x c · advance 2`:

1. `put n0 x b` → `x@n0 = (1, b)`; messages to `n1`, `n2` enqueued.
2. `advance 2` → both deliver; `x@n1 = x@n2 = (1, b)` (converged).
3. `partition n0 n1 | n2` → groups `{n0,n1}`, `{n2}`.
4. `put n0 x c` → `x@n0 = (2, c)`; messages to `n1`, `n2` enqueued.
5. `advance 2` → the `n1` message delivers (`x@n1 = (2, c)`); the `n2` message **cannot cross the
   partition** and stays in-flight, so `x@n2` is **stale at `(1, b)`**.

A subsequent `heal · advance` delivers the stuck message and `n2` converges to `(2, c)`. This is the
distributed world's defining dynamic — a free, exact, reproducible stale-read-under-partition — pinned
as [`test_golden_partition_leaves_isolated_replica_stale`](../tests/test_dist_goldens.py).
