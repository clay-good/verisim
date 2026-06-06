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
> makes stale-reads-under-partition the central dynamic). **Tier-B (the system oracle) now ships**
> (§8 below). The Raft-subset consensus group, transactions/locks, and the embedded SPEC-6 host
> inside each node are later DS increments (SPEC-7 §5, §13); each will extend this document.

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

### 2.1 The declared consistency model (`DistConfig.consistency_model`)

The `eventual` model above is the default. A second model, **`linearizable`**, ships for the H20
consistency-level sweep (SPEC-7 §3.4, §10.2) — the same `put`/`cas` grammar with a different
replication discipline:

| | `eventual` (default) | `linearizable` |
|---|---|---|
| replication | **async**: write local replica now, enqueue `MsgSend`s; peers converge on `advance` | **synchronous**: write **every** replica in the same step — no `MsgSend`, no in-flight window |
| staleness | a `get` may read a stale replica under partition | no replica is ever stale (all writes are all-replica) |
| under partition | the write commits locally and propagates after `heal`+`advance` | the write is **rejected** `("unavailable","")` if any replica is unreachable/down — a **CP** system that trades availability for the absence of divergence (CAP, HW-5) |
| in-flight medium | present (the hidden state a partial correction/cheap tier can miss) | **absent** |

The two models are the strong/weak ends of the §3.4 consistency curriculum. The contrast is the H20
mechanism made concrete: weak consistency adds a *consistency-invisible in-flight medium* (the source
of stale reads, the `subtle` error class of ED1/ED3/ED5, and the slack that lets the consistency-
faithful horizon outlast the bit-faithful one in H19); strong consistency removes it, so every error
is immediately consistency-visible and that slack collapses.

## 3. Actions

### Client ops (append a causal-log event; set `last_result`)

| Action | Semantics |
|---|---|
| `put <node> <key> <val>` | write `node`'s local replica to `(local_version + 1, val)` immediately; enqueue an async `MsgSend` to every other replica of `key` (`deliver_after = clock + 1`). `("ok", val)`. |
| `get <node> <key>` | return `node`'s **local** replica value — which under partition may be **stale**. `("ok", value)`. |
| `cas <node> <key> <old> <new>` | if `node`'s local value `== old`, behave as `put node key new`; else no write, `("conflict", local_value)`. |

If the coordinator node is **down**, a client op returns `("unavailable", "")` and makes no state
change beyond logging the attempt; if the node holds no replica of the key, `("no_replica", "")`.

### Transaction ops (DS0 increment 2 — multi-key OCC; append a causal-log event)

| Action | Semantics |
|---|---|
| `begin <node> <txn>` | open a transaction `txn` at coordinator `node`. `("ok", "")`; `("exists", "")` if already open; `("unavailable", "")` if the node is down. |
| `tget <node> <txn> <key>` | read `key`'s local replica within the txn; **pin its version** on first read (the read-set the commit validates), with read-your-writes for a value the txn has already buffered. `("ok", value)`. |
| `tput <node> <txn> <key> <val>` | buffer a write to `key` in the txn (no replica changes yet). `("ok", val)`. |
| `commit <node> <txn>` | **validate**: if any read key's local version changed since it was read → discard the txn, `("conflict", "")`; else apply every buffered write atomically (each an MVCC bump + replication, exactly as `put`) and end the txn, `("committed", "")`. |
| `abort <node> <txn>` | discard the txn. `("aborted", "")`. |

A `tget`/`tput`/`commit`/`abort` on an unknown txn (or one opened at a different node) returns
`("no_txn", "")`. See §9 for the concurrency-control discipline.

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

## 8. Tier-B — the system oracle (the reality check, SPEC-7 §5.2)

Everything above is **Tier-A**: a *single-threaded analytic discrete-event simulator* that computes
the next cluster state in closed form. It is the executable truth, but — like every reference oracle
in the program — it is a *model* of a distributed system, not one. **Tier-B**
([`verisim.distoracle.system`](../src/verisim/distoracle/system.py)) closes that gap (SPEC-3 wall
**W1**, "the oracle is a model, not reality") for the distributed world, exactly as the host
`SandboxOracle` (SPEC-11) does by running a real `/bin/sh`.

Tier-B **runs the replicated-KV protocol as a real distributed system**: a set of autonomous
**node actors** (`_NodeActor`) that each hold *only their own replicas and an inbox*, exchange real
replication **messages**, and have **no access to any global state**. The cluster state is
*emergent*, reconstructed by polling the actors — never stored in one place, exactly as W7 demands.

- **Determinism via a seeded scheduler (the DST thesis, SPEC-7 §2.1).** A real cluster's delivery
  order is nondeterministic, which is what makes a real cluster un-replayable. Tier-B does what
  madsim / turmoil / FoundationDB's simulator do: it keeps the real message-passing structure but
  drives it with a **seeded scheduler** whose seed is a pure function of `(state, action)`. The
  order it picks is **not** Tier-A's fixed sorted-by-`msg_id` order but a seed-**shuffled** one, so
  agreement certifies the property Tier-A quietly assumes: the eventual-consistency convergence is
  **delivery-order-independent** (LWW by `(version, value)` is a commutative join).
- **Two isolation tiers, disclosed never assumed** (the SPEC-11 `process`/`namespaced` split):
  `simulated` (the always-on default — actors single-stepped by the scheduler, the madsim model) and
  `threaded` (each actor in a *real OS thread* blocking on a real `queue.Queue`, the scheduler
  dispatching one message at a time and awaiting its ack — genuine kernel concurrency, deadlock-free
  by the strictly-sequential protocol). A requested tier that cannot run raises
  `SystemDistOracleUnavailable` — a first-class, disclosed skip, never a silent pass.

The **differential** ([`verisim.distoracle.differential`](../src/verisim/distoracle/differential.py))
compares Tier-A and Tier-B on the **observable-cluster channel** — replicas + in-flight (compared
id-independently) + partitions + down + clock + result — and deliberately *excludes* the causal log
and the monotone id counters, which are bookkeeping of our representation (exactly as the host
differential excludes the `last` observation). The only modeling boundary the KV semantics admits is
`delivery_order`: a converged replica whose value depends on the delivery order (a *non-commutative*
join), which a correct LWW actor never produces and a deliberately-broken **arrival-order** actor
always does. That broken actor is the teeth-bearing negative control (the SY3 analog): the
differential **catches** it, proving the harness can detect a faithfulness break, not merely
rubber-stamp an identical reimplementation. The result is **ED7**
([`verisim.experiments.ed7`](../src/verisim/experiments/ed7.py), [`ed7.png`](../../figures/ed7.png)):
across the exhaustive grammar battery and all three workload drivers (including the fault-heavy
adversarial one) Tier-A and Tier-B agree **bit-for-bit (1.000, residual 0)**, the `H_ε(ρ)` curve is
oracle-invariant (gap 0 at every ρ), the broken control is caught, and the real-OS-thread tier agrees
too — the distributed W1 retirement.

## 9. Transactions — optimistic concurrency control (DS0 increment 2, SPEC-7 §3.2)

The transaction family (`begin`/`tget`/`tput`/`commit`/`abort`, §3) is a multi-key atomic unit over
the replicated KV, implemented in [`verisim.dist.txn`](../src/verisim/dist/txn.py) and shared by
both oracles. The coordinator buffers a transaction's reads and writes locally; the read-set pins
the `(key, version)` each key was first read at, and the write-buffer holds the `(key, value)` pairs
to apply atomically.

**Concurrency control: OCC, first-committer-wins (design decision `DD-D3`).** On `commit`, the
read-set is **validated** — for each read key, the coordinator's *current* local version must equal
the version the txn read. If any changed (a concurrent transaction committed it first), the commit
**aborts** (`conflict`) and applies nothing; otherwise every buffered write is applied atomically
(an MVCC version bump per key) and replicated through the same in-flight medium as a plain `put`.
*Rationale:* OCC is **deterministic and deadlock-free** — no lock table, no lock-acquisition order,
no deadlock detection / victim selection (all of which would inject nondeterminism or require a
scheduler) — so it is the discipline the deterministic core pins first, exactly as the KV core
pinned async-replication LWW before consensus. *Lock-based 2PL is a later refinement; the two OCC
isolation levels (`serializable`/`snapshot`) ship now — see §9.1.*

**Composition.** A committed transaction's writes flow through the existing replication medium, so
they inherit the consistency model unchanged: under `eventual` the peers converge on a later
`advance`; under `linearizable` the commit replicates synchronously and is **rejected**
(`unavailable`, the txn staying open for retry) if it cannot reach all replicas under partition. The
transaction state is purely additive: an empty `txns` set is **omitted** from the canonical form
(§6), so every DS0-increment-1 golden and content-addressed hash is unchanged, and Tier-B reproduces
every transaction trajectory bit-for-bit (the commit replication is delivered by its autonomous
actors on `advance`, where its independence does its work). The OCC commit/abort frontier is pinned
by **ED8** ([`verisim.experiments.ed8`](../src/verisim/experiments/ed8.py),
[`ed8.png`](../../figures/ed8.png)): at concurrency `K` over `M` objects the measured commit rate
tracks the balls-in-bins occupancy law `M·(1−(1−1/M)^K)/K` (the semantics are exactly right, not
merely plausible), with Tier-B agreeing on every scenario.

### 9.1 Isolation levels — `serializable` vs `snapshot` (DS0 increment 3)

The `txn_isolation` config dial selects *which set* `commit` validates (design decision `DD-D4`):

| Level | Validates at commit | Forbids write skew? |
|---|---|---|
| `serializable` (default) | the **read-set** — every read key's local version must be unchanged since it was read (OCC backward validation, Kung–Robinson) | **yes** — a read another txn wrote is caught |
| `snapshot` | only the **write-set** — every written key's version must be unchanged since the txn first wrote it (write-write conflict, first-committer-wins) | **no** — disjoint write-sets both commit |

The distinction is the classic **write-skew** anomaly. Two transactions both read `{x, y}`, then
`A` writes `x` and `B` writes `y`:

- under `snapshot`, `A`'s write-set `{x}` and `B`'s write-set `{y}` are disjoint, so neither write-write
  check fails and **both commit** — a pair of outcomes no serial schedule produces, silently breaking
  any cross-object invariant they each verified;
- under `serializable`, when `A` commits (bumping `x`), `B`'s pinned read of `x` is now stale, so `B`'s
  read-set validation fails and it **aborts** — the anomaly cannot occur.

Both levels remain OCC (deterministic, deadlock-free) and share every other rule above; only the
validation set differs, so the write version is pinned at first `tput` (the `write_versions` field on
`TxnState`) exactly as the read version is pinned at first `tget`. The anomaly and the cost of
forbidding it are pinned by **ED9** ([`verisim.experiments.ed9`](../src/verisim/experiments/ed9.py),
[`ed9.png`](../../figures/ed9.png)): the write-skew anomaly rate is **1.0 under `snapshot`, 0.0 under
`serializable`**, and under a read-heavy contended workload `serializable` aborts strictly more
(`0.70` vs `0.55`, disjoint CIs) — the extra aborts are precisely the price of the stronger
guarantee. Both compose with Tier-B (the autonomous-actor system oracle agrees on every scenario).

### 9.2 Elle-style serializability checking — the black-box history verifier (DS3 increment 2)

ED9 detected write skew the way an omniscient observer would: by counting which transactions the
oracle let commit. **Elle** ([`verisim.distoracle.elle`](../src/verisim/distoracle/elle.py)) detects
it the way a real operator must — from the **observable transaction history alone**, with no oracle
and no cluster state. It is the distributed analogue of Jepsen's Elle (Kingsbury & Alvaro, VLDB
2020), and the *stronger-consistency, over-a-history* sibling of the per-step `cycle` oracle tier
(§8 / SPEC-7 §5), which is the eventual-consistency form.

The theory (Adya 1999). A history over a multi-version store is **serializable iff its Direct
Serialization Graph is acyclic**. The DSG has one node per committed transaction and three edge kinds,
all read off the per-object MVCC **version order**:

| Edge | When | Meaning |
|---|---|---|
| `ww` (write-depends) | `Ti` installs version `v` of `k`, `Tj` installs the next installed version of `k` | the write order |
| `wr` (read-depends) | `Ti` installs version `v` of `k`, `Tj` reads exactly `v` | `Tj` saw `Ti`'s write |
| `rw` (anti-dependency) | `Ti` reads version `v` of `k`, `Tj` installs the version immediately after `v` | `Tj` overwrote what `Ti` read |

A cycle is classified by Adya's G-hierarchy: **G0** (a `ww`-only cycle, dirty write), **G1c** (a
`ww`/`wr` cycle, circular information flow), **G2** (any cycle with an `rw` edge, the general
non-serializable form). **Write skew is the canonical G2**: `A` reads `{x@0, y@0}` and installs
`x@1`; `B` reads `{x@0, y@0}` and installs `y@1`; `B` read `x@0` that `A` overwrote (`rw B→A` on `x`)
and `A` read `y@0` that `B` overwrote (`rw A→B` on `y`) — a two-cycle of anti-dependency edges.

**ED10** ([`verisim.experiments.ed10`](../src/verisim/experiments/ed10.py),
[`ed10.png`](../../figures/ed10.png)) records only each committed transaction's read/installed
versions and hands that history to Elle:

- **the write-skew anomaly, recovered black-box.** Elle's G2-cycle rate is **1.0 under `snapshot`,
  0.0 under `serializable`** — identical to ED9's oracle-side anomaly rate, and it agrees with the
  oracle on every single scenario (`elle_matches_oracle = True`). The free, reference-free checker
  recovers exactly the anomaly the expensive bit-exact oracle sees.
- **Elle certifies the serializable level.** Under a read-heavy contended workload, Elle flags
  **0.60 [0.30, 0.90]** of `snapshot` histories non-serializable (the G2 anomalies that level admits)
  and **0.0** of `serializable` histories (the guarantee that level enforces — certified
  independently of the oracle that enforces it).

ED10 supplies the store's MVCC version order to Elle (its *version-oracle* mode); recovering the
version order from observed values alone (Elle's list-append / unique-write recoverability) is
deferred. Pure standard library, dependency-free, GPU-free.

## 10. Partial observation — the probe projection (DS3 increment 4, SPEC-7 §5.4)

Every section above describes the *full* cluster state — what the oracle computes and what a
bit-exact divergence compares. But **W7 says there is no consistent global snapshot**, and no real
observer ever has one: a client, an SRE, or a monitoring probe sees only the part of the cluster it
can *reach*. [`verisim.dist.observe`](../src/verisim/dist/observe.py) makes that epistemic limit a
deterministic object. `observe(state, vantage)` projects a `DistributedState` onto the `Observation`
an observer connected to a set of `vantage` nodes can obtain:

| `Observation` field | Meaning |
|---|---|
| `vantage` | the nodes the observer can query directly (a client's connections, an SRE's hosts) |
| `reachable` | the **up** nodes the observer can talk to: up vantage nodes plus their co-partitioned up peers |
| `unreachable` | every other node, **with no reason attached** — crashed and partitioned-away are not distinguished |
| `replicas` | the `(object, node, version, value)` tuples on reachable nodes only — the rest of the cluster is dark |
| `clock` | observable via any reachable node; `None` if the whole vantage is dark |

There is deliberately **no in-flight field**. Three properties follow, all pinned by tests
([`test_dist_observe.py`](../tests/test_dist_observe.py)) and a golden
([`test_dist_goldens.py`](../tests/test_dist_goldens.py)):

1. **The in-flight medium is unobservable.** The replication messages are in the network, not in any
   node's memory, so no probe — even the maximal whole-cluster vantage — can read them. A model that
   mispredicts only a message-in-transit is *observably faithful* until `advance` delivers it and
   writes a replica. This is the partial-observation form of H19/ED5: where the consistency view
   forgives the in-flight medium by *abstraction*, the probe forgives it by *physical
   unobservability*.
2. **Crash and partition are indistinguishable from one vantage.** A `down` node and a
   partitioned-away node both project to the same `unreachable` fact — the failure-detector limit
   behind FLP. `observe(crashed, ("n0",)) == observe(partitioned, ("n0",))` exactly. A **second**
   vantage that reaches the node's side of the split separates them (the partition case exposes the
   live isolated replica; the crash case does not). One probe cannot localize a fault to
   crash-vs-partition; a quorum of probes can.
3. **A probe is the §5.4 cheap-localized oracle mode.**
   [`observable_divergence(truth, pred, vantage)`](../src/verisim/distmetrics/observe.py) is the
   probe-mode divergence: identical to the bit-exact `divergence` when the vantage reaches the whole
   cluster, smaller (and in-flight-forgiving) under partition. Because a bit-faithful step is
   necessarily observably faithful, the **observable-faithful horizon dominates the bit-faithful
   horizon** — `H_ε^bit ≤ H_ε^observable`, structurally.

**ED12** ([`verisim.experiments.ed12`](../src/verisim/experiments/ed12.py),
[`ed12.png`](../../figures/ed12.png)) measures both, dependency-free: free-running, the observable
horizon outlasts the bit horizon for `subtle` (in-flight) errors (**probe gap +9.0 steps**, disjoint
CI) and coincides for `gross` (durable-replica) errors (the control); and across a battery a single
external vantage cannot tell a crash from a partition (**indistinguishable rate 1.0**) while a paired
vantage always can (**0.0**). The `Observation` is canonical by construction (frozensets), so the
indistinguishability is a testable byte-equality. This is the deterministic substrate the (deferred)
RSSM belief (§6.2) must roll forward under partition — the belief's task is to predict the full state
from the observable one, and that task is undefined until "observable" is. Pure standard library,
dependency-free, GPU-free.

