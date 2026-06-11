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
> (§8 below). Multi-key transactions with the four SQL isolation levels (§9), the OCC/2PL split, the
> complete §3.4 fault grammar (`drop`/`delay`/`reorder`/`clock_skew`), the `anti_entropy`/`gossip`
> convergence ops, the Raft-subset consensus core (`elect`/`propose`/`step_down`/`lease`/`lread`/`read_index`/`append`/membership, §3.3),
> the `deploy`/`config_push` admin ops, and the embedded SPEC-6 `host` inside each node have all since
> shipped as later DS increments. The external real-binary DST runtime and the SPEC-5 net embedded
> *between* nodes remain later increments (SPEC-7 §5, §13); each will extend this document.

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

### 2.2 The middle: `causal` consistency (DS0 increment 5)

A third model, **`causal`**, ships as the *middle* of the curriculum — strictly stronger than
`eventual`, strictly weaker than `linearizable`. It keeps `eventual`'s async, available-under-partition
replication (writes commit locally, peers converge on `advance`) but adds **one guarantee**: *if write
`B` causally depends on write `A`, no replica ever observes `B` before `A`*. This is the cross-object
delivery ordering a defender/SRE relies on — you never see an effect whose cause is still invisible.

It is implemented as a **delivery-order refinement**, not a new write path. Each replication
`Message` carries a `deps` field — the **causal context**: the `(object, version)` pairs the writing
node had already *observed* (applied to its own replicas, at a non-boot `version > 0`) for objects
other than the one being written. That is a slice of the node's version vector. On `advance`, a
message is delivered only when (the existing conditions hold **and**) the destination has already
applied at least those dependency versions; otherwise it **waits in flight** — the message is held,
not lost, and is delivered once its cause arrives. `deps` is empty under `eventual` / `linearizable`
(those models do not order delivery) and is **omitted from the canonical form when empty**, so every
pre-DS0-incr-5 golden, hash, and tokenization is byte-for-byte unchanged (a purely additive field).

| | `eventual` | **`causal`** | `linearizable` |
|---|---|---|---|
| replication | async, greedy delivery | async, **dependency-ordered** delivery | synchronous, all-replica |
| effect-before-cause | **admitted** (a replica can see `y` before its cause `x`) | **forbidden** (the `y` message holds for `x@1`) | impossible (no in-flight) |
| concurrency | full | full — only *causally-linked* writes are ordered, independent ones stay free | n/a (no in-flight) |
| under partition | available | available | rejected (CP) |
| in-flight medium | present | present (with deps) | absent |

The mechanism (and the scenario the golden + **ED13** pin) routes the *effect* `y` to an observer
`n2` while its *cause* `x` is still partitioned away — the only way to manufacture out-of-causal-order
delivery in a group-partition model, since disjoint groups are transitive at any single instant:

```
put n0 x a               # cause: x=a@1 at n0
partition n0 n1 | n2     # isolate the observer n2
advance 1                # x@1 -> n1 delivers; x@1 -> n2 is blocked
put n1 y b               # n1 has now observed x@1, so y=b@1 carries deps={x@1}
partition n0 | n1 n2     # re-route so y can reach n2 while x cannot
advance 1                # eventual: y -> n2 (effect before cause!); causal: y held (dep unmet)
```

Under `eventual`, `n2` reads `y=b, x=nil` (the anomaly). Under `causal`, the `y` message's
`deps={x@1}` is unmet at `n2`, so it is held: `n2` reads `y=nil, x=nil`, and after `heal`+`advance`
both arrive in causal order and the cluster converges to the *identical* durable state `eventual`
reaches. ED13 ([`ed13.py`](../src/verisim/experiments/ed13.py),
[`ed13.png`](../../figures/ed13.png)) reports the anomaly rate (**eventual 1.0, causal 0.0**), that
`causal` holds the *dependent* message but never the *independent* one (it orders only causally-linked
writes), and that convergence is preserved (eventual ≡ causal final state, in-flight drains to 0).
**Tier-B (the autonomous-actor system oracle, §8) reproduces causal delivery bit-for-bit** — see §8.1.

### 2.3 Consensus: `quorum` (the Raft-subset model, DS0 increment 7)

A fourth replication model, **`quorum`**, is the realistic CP middle real consensus protocols (Raft,
Paxos) occupy — strictly more available than `linearizable` while still divergence-free. A `quorum`
write commits **synchronously to the reachable majority** of an object's replicas and **rejects**
(`unavailable`) only when a *majority is not reachable*; the unreachable minority catches up
**asynchronously** (one `MsgSend` each, delivered on `heal`+`advance`). Concretely, with a coordinator
on a partition side that can reach `m` of the `n` replicas, the write commits iff `m >= n//2 + 1`.

The contrast with the other CP model is the whole point:

| | `eventual` | `quorum` (Raft-subset) | `linearizable` |
|---|---|---|---|
| commit needs | the local node only | a reachable **majority** | **every** replica |
| under a *minority* partition | commits (locally) | **rejected** (CP) | rejected |
| under a *majority* partition (coordinator on the majority side) | commits | **commits** (available!) | **rejected** |
| split-brain (both sides write) | **forks** (diverges) | **never** (only the majority commits) | never (neither commits) |
| stale replicas | yes (until `advance`) | only the minority (until it rejoins) | none |

So `quorum` is the only model that is **both available on the majority side and divergence-free** —
the reason real systems use majority quorums rather than all-replica synchrony. Where `linearizable`
goes completely dark under *any* partition (it needs all `n`), `quorum` keeps serving the side that
holds a majority, and because only one side can ever hold the majority, the object never forks (the
split-brain ED11's version oracle catches in the `eventual` case). **ED14**
([`ed14.py`](../src/verisim/experiments/ed14.py), [`ed14.png`](../../figures/ed14.png)) plots the
availability frontier (the quorum step at the majority threshold) and the split-brain rates
(`eventual` 1.0, `quorum`/`linearizable` 0.0). The `quorum` enum value is purely additive (no new
state), so every prior golden/hash is unchanged, and the autonomous-actor **Tier-B reproduces the
quorum decision bit-for-bit** (the W1 retirement, §8) — the availability/safety behavior is a property
of a real message-passing execution, not just the analytic DES.

## 3. Actions

### Client ops (append a causal-log event; set `last_result`)

| Action | Semantics |
|---|---|
| `put <node> <key> <val>` | write `node`'s local replica to `(local_version + 1, val)` immediately; enqueue an async `MsgSend` to every other replica of `key` (`deliver_after = clock + 1`). `("ok", val)`. |
| `get <node> <key>` | return `node`'s **local** replica value — which under partition may be **stale**. `("ok", value)`. |
| `cas <node> <key> <old> <new>` | if `node`'s local value `== old`, behave as `put node key new`; else no write, `("conflict", local_value)`. |
| `delete <node> <key>` (DS0 increment 26) | **tombstone delete**: behave as `put node key TOMBSTONE` — a *versioned write of a tombstone*, not a removal of the replica, so it inherits every consistency model and is ordered by version (the resurrection-safe discipline). A `get` on a tombstoned replica reports `("deleted", "")`. The client result is `("deleted", "")`; the replicated value is the `__deleted__` sentinel. Purely additive (no new state field, no new edit). |
| `incr <node> <key>` (DS0 increment 27) | **atomic counter** (read-modify-write): read `node`'s local counter (a non-numeric/absent value is `0`) and behave as `put node key str(count+1)`. Sequentially correct, but under a partition the consistency model decides whether concurrent increments survive: `eventual` **silently loses** a concurrent increment (two same-version writes, LWW keeps one), `quorum` makes the minority `("unavailable", "")` (no silent loss), `linearizable` rejects under any partition. `("ok", new_count)`. Purely additive (the counter is a digit-valued replica). |
| `cincr <node> <key>` (DS0 increment 28) | **CRDT G-counter increment**: bump **only `node`'s own** sub-count for `key` (`GCounterSet(key, node, node, +1)`) — purely node-local, so **always available** (a partitioned-alone node still counts — the AP property `incr` lacks under quorum/lin) and **never loses** a concurrent increment (disjoint owners). The CRDT join (per-(key, owner) max) is applied by `anti_entropy`/`gossip`. `("ok", new_own_subcount)`; `("unavailable", "")` if down. Purely additive (one `gcounters` map + one `GCounterSet` edit). |
| `cdecr <node> <key>` (DS0 increment 29) | **CRDT PN-counter decrement**: bump **only `node`'s own** *decrement* sub-count for `key` (`NCounterSet(key, node, node, +1)`) — the twin of `cincr` over the PN-counter's N half. Purely node-local (**always available**), concurrent decrements never conflict, and the same per-(key, owner) max join merges it. The counter's value (`cget` = P − N) may now go **negative**, the property the grow-only G-counter lacks. `("ok", new_own_decrement_subcount)`; `("unavailable", "")` if down. Purely additive (one `ncounters` map + one `NCounterSet` edit). |
| `cget <node> <key>` (DS0 increment 28/29) | **CRDT counter read**: return `node`'s G-counter sum **minus** its decrement sum over owners (a PN-counter; for a grow-only counter never `cdecr`-ed, N is empty and this is just the G-counter sum) — `node`'s local view (may lag, but never loses; a later join adds the rest). `("ok", total)`; `("unavailable", "")` if down. A pure read. |
| `sadd <node> <key> <elem>` (DS0 increment 30) | **CRDT OR-Set add**: tag `elem` with a **unique dot** `(owner=node, seq)` — `node`'s next monotone sequence for `key` — and store it in `node`'s observed add-set (`ORSetAdd`). Purely node-local (**always available**), and because the dot is fresh it **survives** a concurrent `srem` (add-wins) and a removed element is **re-addable**. The union join applies in `anti_entropy`/`gossip`. `("ok", elem)`; `("unavailable", "")` if down. Purely additive (two `orset_adds`/`orset_tombs` maps + the `ORSetAdd`/`ORSetTomb` edits). |
| `srem <node> <key> <elem>` (DS0 increment 30) | **CRDT OR-Set observed-remove**: tombstone **only the dots of `elem` that `node` has observed** in its own add-set (`ORSetTomb` per dot) — the *observed*-remove that makes add-wins work. The dot stays in the add-set (union semantics); membership no longer counts it. Node-local (always available); removing an absent element is a no-op. `("ok", elem)`; `("unavailable", "")` if down. |
| `smembers <node> <key>` (DS0 increment 30) | **CRDT OR-Set read**: return `node`'s elements with at least one non-tombstoned dot, rendered `{a,b,c}` (`{}` empty) — `node`'s local view (may lag, never loses). `("ok", members)`; `("unavailable", "")` if down. A pure read. |
| `mvput <node> <key> <val>` (DS0 increment 31) | **CRDT MV-register write**: tag `val` with a fresh dot `(owner=node, seq)`, **tombstone every dot `node` currently observes** (a write supersedes the values it saw, `MVRegTomb` per dot), and store its own write-dot (`MVRegWrite`). A *sequential* overwrite collapses to one value; *concurrent* writes (neither observing the other) **both survive** as siblings. Purely node-local (**always available**); the union join applies in `anti_entropy`/`gossip`. `("ok", val)`; `("unavailable", "")` if down. Purely additive (two `mvreg_vals`/`mvreg_tombs` maps + the `MVRegWrite`/`MVRegTomb` edits). |
| `mvget <node> <key>` (DS0 increment 31) | **CRDT MV-register read**: return `node`'s surviving (non-tombstoned) sibling values, rendered `{a,b}` (`{}` empty, one value if resolved) — `node`'s local view (may lag, never loses). `("ok", siblings)`; `("unavailable", "")` if down. A pure read. |
| `lwwput <node> <key> <val>` (DS0 increment 32) | **CRDT LWW-register write**: stamp `val` with `(ts, owner=node)` where `ts = lamport[node] + 1` (advancing `node`'s Lamport clock), and store it as the current register value (`LWWRegSet` + `LamportSet`). The join keeps the **max** copy by `(ts, owner, value)` — a write that happened-after (higher ts) wins regardless of node; concurrent (equal-ts) writes break the tie by node id. Purely node-local (**always available**). `("ok", val)`; `("unavailable", "")` if down. Purely additive (two `lwwreg`/`lamport` maps + the `LWWRegSet`/`LamportSet` edits). |
| `lwwget <node> <key>` (DS0 increment 32) | **CRDT LWW-register read**: return `node`'s current winning value (`` `` if never written) — `node`'s local view (may lag, never resolves wrongly). `("ok", value)`; `("unavailable", "")` if down. A pure read. |
| `enqueue <node> <queue> <val>` (DS0 incr 21) | append `val` to each reachable replica's FIFO `queue` list (the relative append op). Availability follows the consistency model (linearizable needs every replica, quorum a majority, eventual/causal proceed on the reachable set); else `("unavailable", "")`. `("enqueued", val)`. |
| `dequeue <node> <queue>` (DS0 incr 21) | return the head of `node`'s local `queue` replica and pop the head from each reachable replica. The delivery semantics follow the model: `eventual` admits **duplicate delivery** under partition (the removal does not cross the split), `linearizable`/`quorum` gate availability for **exactly-once**. `("dequeued", head)`, or `("empty", "")` if the local replica has no items; `("unavailable", "")` if the model's quorum is unreachable. |

If the coordinator node is **down**, a client op returns `("unavailable", "")` and makes no state
change beyond logging the attempt; if the node holds no replica of the key, `("no_replica", "")`.
(The KV ops `put`/`get`/`cas` append a causal-log event; the queue ops `enqueue`/`dequeue` do not
yet — they are a separate data plane, DS0 increment 21.)

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
| `drop <src> <dst>` (DS0 increment 11) | **lose** every in-flight message from `src` to `dst` (a `MsgDrop` per lost message). Unconditional — the drop does not require the link to be currently connected (a message can be lost whether or not it would have been delivered). `("dropped", str(num_dropped))`; a channel with no in-flight message is a no-op `("dropped", "0")`. |
| `delay <src> <dst> <dt>` (DS0 increment 13) | defer every in-flight `src`→`dst` message by `dt` (`deliver_after += dt`, a `MsgReschedule` per message) — a *recoverable* delay (the message still arrives, just later), the counterpart to `drop`'s loss. `("delayed", str(num_moved))`. |
| `reorder <src> <dst>` (DS0 increment 13) | reverse the channel's delivery schedule (reassign the sorted delivery times in reverse, a `MsgReschedule` per moved message). Last-writer-wins makes the *converged* state invariant, but it flips which write a peer sees *in transit*. `("reordered", str(num_moved))`. |
| `clock_skew <node> <delta>` (DS0 increment 14) | set `node`'s signed clock offset (a `ClockSkewSet`); it shifts only the `deliver_after` the node stamps on its future sends. Because LWW resolves by `(version, value)` and never by timestamp, the converged state is clock-independent. `delta == 0` clears the skew. `("skewed", str(delta))`. |

### Protocol ops (the convergence + consensus machinery — no causal-log event)

| Action | Semantics |
|---|---|
| `anti_entropy <node>` (DS0 increment 12) | **read-repair** `node`: for every object it replicates, adopt the winning `(version, value)` (last-writer-wins) among its **reachable** replicas (itself + co-partitioned, up peers), emitting a `ReplicaWrite` per object that moves. A crashed node is `("unavailable", "")`. `("repaired", str(num_repaired))`; nothing to repair is `("repaired", "0")`. |
| `gossip <a> <b>` (DS0 increment 15) | **pairwise, bidirectional** anti-entropy: for every object both `a` and `b` replicate, *both* adopt their mutual per-object winner (a `ReplicaWrite` for whichever is behind). Needs a live link (both up + connected), else `("unavailable", "")`. One gossip reconciles both endpoints; a chain spreads a write epidemically. `("gossiped", str(num_moved))`. |
| `elect <node>` (DS0 increment 16) | **leader election**: `node` becomes the cluster leader iff the *live* nodes in its partition group are a strict majority of all nodes (`> n//2`), bumping the monotone `term` and installing the global `leader` (a `ProtocolStep`). A crashed candidate is `("unavailable", "")`; one without a live majority is `("no_quorum", "")`. On success `("elected", str(new_term))`. Touches no replica. |
| `propose <node> <key> <val>` (DS0 increment 16) | **leader-fenced write**: commit `val` to `key` iff `node` is the current `leader` and can reach a majority of `key`'s replicas (the consensus quorum, regardless of `consistency_model`) — synchronous to the reachable majority, async catch-up to the minority, exactly like a `quorum` `put`. Rejected `("not_leader", current_leader)` if `node` is not the leader (a deposed leader stays fenced even after `heal`); `("no_quorum", "")` if no reachable majority. `("ok", val)` on commit. |
| `step_down <node>` (DS0 increment 17) | **voluntary relinquishment**: clear `leader` (`→ None`) **at the same `term`** iff `node` is the current leader — the graceful counterpart to `elect`'s higher-term deposition, leaving the cluster leaderless until a fresh `elect`. Rejected `("not_leader", current_leader)` if `node` is not the leader (so a non-leader or a second `step_down` is a no-op reject — idempotently safe); `("unavailable", "")` if crashed. Needs **no quorum** (it reads only the node's own leadership), so a minority-stranded leader can relinquish where its `propose` is `no_quorum`. On success `("stepped_down", str(term))`. Also **releases the lease** (DS0 incr 18). Touches no replica. |
| `lease <node> <dt>` (DS0 increment 18) | **leader read lease**: iff `node` is the current leader, set the global-clock lease deadline `lease_until = clock + dt` (a `LeaseSet`). Through that deadline the leader may serve `lread` without a quorum and a new `elect` is fenced. Rejected `("not_leader", current_leader)` / `("unavailable", "")`. On success `("leased", str(until))`. Touches no replica. |
| `lread <node> <key>` (DS0 increment 18) | **leader-lease local read**: iff `node` is the current leader **and** `clock < lease_until` and up, return `node`'s local replica value with **no quorum round-trip** — linearizable w.r.t. consensus (`propose`) writes (the leader is always in a propose's commit majority, and a live lease guarantees its term is uncontested). So a minority-stranded leader can still `lread` where its `propose` is `no_quorum`. Rejected `("lease_expired", "")` if no live lease; `("not_leader", current_leader)` if not the leader (a deposed leader cannot read off a stale lease); `("unavailable"/"no_replica", "")` otherwise. On success `("ok", value)`. A pure read. |
| `read_index <node> <key>` (DS0 increment 25) | **quorum-confirmed linearizable read** (Raft ReadIndex, the partner to `lread`): iff `node` is the current leader **and** a **majority** of the voting members is reachable (the leadership-confirmation round), return `node`'s local replica value — linearizable with no clock/lease assumption. The two reads' availability is opposite: a minority-stranded leader is `("no_quorum", "")` here (it cannot confirm leadership) where a live-lease `lread` still serves locally; a deposed leader is `("not_leader", current_leader)` even after `heal` (no stale read off a stale leader, where a plain `get` would serve it). `("unavailable"/"no_replica", "")` otherwise. On success `("ok", value)`. A pure read — no state field, no edit. |
| `append <node> <key> <val>` (DS0 increment 19) | **replicated-log append**: iff `node` is the current leader, append a `LogEntry(term, index, key, value)` to its log and replicate to the reachable followers, who **adopt the leader's prefix** (a `LogSet` per node — overwriting any divergent *uncommitted* tail, the log-matching reconciliation). It **commits iff a majority holds it** (advancing the monotone `commit_index` and folding the committed prefix into the KV — a key's version is the count of its committed writes — which backfills a rejoined follower): `("appended", str(index))`. A minority-stranded leader still appends locally but `("uncommitted", str(index))` — not applied to the KV, overwritable by a higher-term leader. Rejected `("not_leader", current_leader)` / `("unavailable", "")`. |
| `add_replica <node>` / `remove_replica <node>` (DS0 increment 20) | **membership change**: reconfigure the consensus voting set `members` (a leader-committed `MemberSet`), so the quorum threshold (`elect`/`propose`/`append`) tracks it. `add_replica` adds a node (restoring the full cluster collapses to the empty "all vote" sentinel); `remove_replica` removes one — shrinking the majority, the lever that **restores availability** after failures. Rejected `("no_leader", "")` (a leader must commit it), `("is_leader", node)` (the active leader cannot be removed), `("last_member", node)`, `("unknown_node", "")`; a no-op is `("already_member"/"not_member", node)`. On success `("added"/"removed", node)`. All config nodes still store replicas; `members` is the voting overlay. Touches no replica. |
| `deploy <node> <version>` (DS0 increment 22) | **rolling upgrade**: set `node`'s running software `version` (a `VersionSet`). Two nodes share a consensus quorum (`elect`/`propose`/`append`) only if their versions are within `DistConfig.max_version_skew` (default `1`, the N-1 window), so a deploy that creates an incompatible split with no compatible majority loses quorum — *the deploy broke the cluster*. `("deployed", str(version))`; `("unknown_node", "")` for a non-config node. Gates *consensus* only (the KV/queue data plane is version-agnostic). Touches no replica. |
| `host <node> <syscall...>` (DS0 increment 23) | **embedded host**: run a SPEC-6 syscall (`fork`/`exit`/`setuid`/`open`/`write`/`close`) on `node`'s own embedded host — a process table + per-process fd tables + an embedded v0 filesystem — by delegating to the SPEC-6 `ReferenceHostOracle` and wrapping its bundle delta in a `HostStep`. Per-node isolated; `("unavailable", "")` if the node is down (the cross-layer crash gate); `("ok", stdout)` on a successful syscall (e.g. the new pid / fd), `("host_err", "")` on a syscall failure (EPERM/EBADF/bad pid). The node's host is created lazily on its first host op. Touches no replica (the host is a separate per-node subsystem). |
| `config_push <node> <key> <val>` (DS0 increment 24) | **config push**: a leader-committed, majority-replicated cluster config value (a `ConfigSet` per reachable voting member). Unlike `deploy` (a node-local version *label* gating consensus *compatibility*), this is a Raft-style config *entry* — leader-fenced like `propose`/`append`: `("not_leader", current_leader)` for a non-leader, `("no_quorum", "")` for a minority-stranded leader (and **no node's config changes** — all-or-nothing at commit). On commit `("committed", val)`, writing the value to the reachable majority — so a push under partition leaves the **partitioned minority with stale config** (config divergence), repaired by a re-push after `heal`. `("unavailable", "")` if the leader is down. Gates nothing in the data plane; the `config` map is observable cluster state. Touches no replica. |

`advance` is the engine of the distributed dynamics — it is where replication actually happens and
where partition/crash make their effect felt. Delivery is simulated **sequentially within one
`advance`**, so a later message's last-writer-wins comparison sees the effect of earlier deliveries in
the same step.

### 3.1 `drop` vs `partition` — lost vs delayed (DS0 increment 11)

`drop` is the unreliable-network fault (the `BUGGIFY` message-loss primitive of deterministic
simulation testing, SPEC-7 §2.1). It is the mirror of `partition`, and the contrast is the whole
point: `partition` **holds** a replication message (it stays in-flight, blocked, and is delivered once
the link `heal`s and time `advance`s), whereas `drop` **destroys** it (the message is gone, and
`heal`+`advance` has nothing to deliver). So the two media produce the *same* symptom — a stale
replica — for opposite reasons: a **recoverable delay** versus an **unrecoverable loss**. Concretely,
`drop` breaks the eventual-consistency convergence guarantee, which silently assumes delivery is
*reliable, if delayed*:

```
put n0 x b               # n0.x = (1,b); enqueue MsgSend n0->n1 and n0->n2
drop n0 n1               # destroy the n0->n1 message  →  ("dropped","1")
advance 2                # only the surviving n0->n2 message delivers  →  ("advanced","1")
heal                     # restore the network — but there is nothing left to deliver
advance 2                # n1 stays at the boot value (0,nil): the write is lost, not delayed
put n0 x c               # a *newer* write (version 2) — its fresh message DOES reach n1
advance 2                # n1.x = (2,c): the only repair for a dropped write is being superseded
```

The dropped value (`b`) is **never observed** by `n1` — its replica goes boot → `c`, skipping the
lost write entirely (a lost update at the network layer). `drop` adds no state field (it only removes
in-flight messages), so it composes with every consistency model and leaves every prior golden, hash,
and tokenization unchanged. ED18 ([`experiments/ed18.py`](../src/verisim/experiments/ed18.py)) pins
both findings; Tier-B reproduces the drop and the broken/repaired convergence bit-for-bit.

### 3.2 `anti_entropy` — read-repair, the convergence `drop` broke (DS0 increment 12)

`anti_entropy` is the first **protocol** op (SPEC-7 §3.2) and the §4 `ReplicaConverge` the spec named:
the **read-repair / anti-entropy** mechanism real eventually-consistent stores (Dynamo, Cassandra) use
to converge *despite* lost messages. Where `drop` *breaks* the convergence guarantee, anti-entropy
*restores* it — and, crucially, it needs **no in-flight message**: `anti_entropy node` reads the
*current* replicas of `node`'s reachable peers and pulls each object to the winning `(version, value)`,
so it repairs a write `advance` can never deliver because the message is gone. Continuing the `drop`
example above (after `heal`, `n1` permanently stale at the boot value):

```
anti_entropy n1          # n1 pulls x from its reachable peers (n0, n2 hold the latest)
                         #   → n1.x = (1,b), ("repaired","1") — read-repair, with no new write
```

Two properties make it faithful to real gossip rather than magic. First, it **adopts the latest
reachable version, skipping intermediates** — a replica stuck at `v0` that missed `v1` and `v2`
read-repairs straight to `v2` (it never saw `v1`), so a single step can jump a version by more than
one (the reason the cheap `cycle`/`symbolic` oracle tiers defer `anti_entropy` to bit-exact rather
than applying the per-step "version moves by ≤1" rule). Second, it is **bounded by reachability** —
under partition it reconciles only within `node`'s group, so it cannot pull a value held across the
split; full convergence still needs `heal`. A crashed node is `("unavailable","")`. `anti_entropy`
reuses the `ReplicaWrite` edit and adds no state field, so it composes with every consistency model
and leaves every prior golden, hash, and tokenization unchanged. ED19
([`experiments/ed19.py`](../src/verisim/experiments/ed19.py)) pins both findings; Tier-B reproduces the
read-repair bit-for-bit.

### 3.3 `elect` / `propose` / `step_down` / `lease` / `lread` / `read_index` / `append` / membership — the Raft-subset consensus core (DS0 increments 16–20, 25)

`elect` and `propose` are the **consensus** family — the third action family (SPEC-7 §3.2) and the
§4 `ProtocolStep` the spec named. They add the one safety property a leaderless `quorum` write
*cannot* provide: a single, fenced writer. Two state fields appear (`term: int`, `leader: str | None`),
both at their boot defaults (`0` / `None`) until the first election, so a cluster that never runs
consensus serializes to the exact pre-increment-16 form — every prior golden and hash is unchanged.

**`elect node` — the majority rule.** A candidate becomes leader iff the **live** nodes in its
partition group are a strict majority of the whole cluster (`> n // 2`). Because two disjoint groups
can never each hold a majority, **at most one leader can be elected** — split-brain at the leadership
level is structurally impossible, not merely unlikely. The even-split edge is the sharpest case: a
`2 | 2` in a 4-node cluster leaves *neither* side a strict majority, so neither can elect — the
cluster is **leaderless rather than forked** (the CAP-availability price consensus pays to never
fork). A successful election bumps the monotone `term` and installs the global `leader`:

```
elect n0                       # full connectivity → ("elected","1"), leader n0, term 1
partition n0 | n1 n2           # n0 into the minority
elect n0                       # ("no_quorum","") — a minority side cannot elect (no split-brain)
elect n1                       # {n1,n2} is 2 of 3 → ("elected","2"), leader n1, term 2
```

**`propose node key val` — the fence.** A leader-fenced majority write: it commits only if `node` is
the *current* `leader` and can reach a majority of `key`'s replicas (synchronous to the reachable
majority, async catch-up to the minority — exactly a `quorum` `put`), regardless of the configured KV
`consistency_model`, because consensus *is* majority-quorum. The two rejections encode the safety:

- `("not_leader", current_leader)` — `node` is not the leader. The headline property: a leader
  deposed by a higher-term election is fenced **even after the partition heals**, because the global
  `leader` already moved on — the Raft *leader-completeness* guarantee. A leaderless `quorum` put,
  available to any coordinator that reaches a majority, would happily commit that stale write.
- `("no_quorum", "")` — the leader cannot reach a majority (a leader stranded in the minority cannot
  commit), so a write never proceeds on a side that cannot durably hold it.

```
elect n0; partition n0 | n1 n2; elect n1; heal   # n0 deposed by n1 (term 2), then the network heals
propose n0 x d                                    # ("not_leader","n1") — fenced after heal
propose n1 x d                                    # ("ok","d") — the legitimate leader commits
```

Both ops are **coordinator-level decisions** (the quorum is read from the partition/down medium, not
an actor's local view, exactly like a `quorum` write's reachability), so Tier-A and Tier-B compute
byte-identical leader/term/replica deltas via the shared `elect_edits` / `propose_edits` helpers —
the W1-retirement guarantee holds for consensus too. The cheap oracle tiers defer the election logic
to bit-exact (the `metamorphic` tier adds two reference-free invariants — *term is monotone* and
*leader is a known node* — so a backward-term or bogus-leader prediction is still refuted cheaply).
ED23 ([`experiments/ed23.py`](../src/verisim/experiments/ed23.py)) pins both panels (no split-brain;
term-fencing vs the unfenced-`put` control); Tier-B reproduces every transition bit-for-bit.

**`step_down node` — the voluntary handoff (DS0 increment 17).** Where `elect` *deposes* a leader by
a higher term, `step_down` lets the *current* leader hand back power on its own, leaving the cluster
**leaderless at the same `term`** (`leader → None`, `term` held). The term machinery then closes the
gap the same way it fences a deposed leader: until a fresh `elect` installs a successor at a strictly
higher term, every `propose` is `("not_leader", "")` — so a clean handoff is `step_down` then
`elect <successor>`, and **no leaderless window ever commits a consensus write**. Only the current
leader may relinquish — a non-leader (or a second `step_down` on an already-leaderless cluster) is a
no-op reject `("not_leader", current_leader)`, so it is idempotently safe; a crashed leader is
`("unavailable", "")`. The asymmetry it exposes: relinquishing power reads only the node's own
leadership, never the medium, so a **minority-stranded leader can still step down** where its `propose`
there is `no_quorum` — *giving up* authority is always safe, *exercising* it needs a quorum.

```
elect n0; propose n0 x b                          # ("ok","b") — n0 leads (term 1) and commits
step_down n0                                       # ("stepped_down","1") — leaderless, term held at 1
propose n0 x c                                     # ("not_leader","") — no leaderless commit window
elect n1; propose n1 x c                           # successor at term 2 commits — the clean handoff
partition n0 | n1 n2; step_down n0                 # a minority leader still relinquishes (no quorum needed)
```

Like `elect`/`propose`, `step_down` is a coordinator-level decision touching no replica, computed
byte-identically by Tier-A and Tier-B via the shared `step_down_edits` helper and reusing the
`ProtocolStep` edit (`leader → None`). ED24 ([`experiments/ed24.py`](../src/verisim/experiments/ed24.py))
pins both panels (the handoff lifecycle; authority + partition-independence) with Tier-B agreeing on
every transition.

**`lease node dt` / `lread node key` — the leader lease (DS0 increment 18).** The Raft read
optimization: `lease` lets the *current* leader take a read lease through the global-clock deadline
`clock + dt` (a `LeaseSet(until)`), and while that lease holds `lread` serves the leader's local
replica value with **no quorum round-trip**. It is linearizable w.r.t. consensus (`propose`) writes
because (i) the leader is always in a `propose`'s commit majority, so its local replica holds every
committed value, and (ii) the live lease guarantees no other leader has been elected. So a leader
**partitioned into the minority can still `lread`** locally where its `propose` is `no_quorum` — the
read-availability the lease buys. The safety mechanism is a coupling with `elect`: a new election is
**fenced** (`lease_held`) until the incumbent's lease expires, so leadership cannot change hands
under a live lease (which is what makes the local read safe) — but a voluntary `step_down` **releases
the lease immediately**, so a graceful handoff needs no wait where a *crashed* leader forces the
cluster to outlast the lease. The lease is global cluster metadata (like `term`/`leader`), so `lease`
and `lread` are coordinator-level decisions computed byte-identically by Tier-A and Tier-B; the
deadline is omitted from the canonical form until the first `lease`.

```
elect n0; propose n0 x b; lease n0 5               # n0 leads, commits x=b, takes a lease through clock 5
lread n0 x                                          # ("ok","b") — local read, no quorum contacted
elect n1                                            # ("lease_held","5") — a successor waits out the lease
partition n0 | n1 n2; lread n0 x                    # ("ok","b") — minority leader still reads locally
advance 6; lread n0 x                               # ("lease_expired","") — past the deadline, must renew
```

(Honest scope: the lease deadline is a global-clock instant read as cluster metadata, and `lread` is
linearizable with respect to consensus writes. The real per-node lease timer under bounded clock
*drift* is a Tier-B refinement; our `clock_skew` fault is a constant offset, not a rate, so it shifts
neither the grant nor the expiry here.) ED25 ([`experiments/ed25.py`](../src/verisim/experiments/ed25.py))
pins both panels with Tier-B agreeing on every transition.

**`read_index node key` — the quorum-confirmed linearizable read (DS0 increment 25).** The *other* way
Raft serves a linearizable read, and the deliberate counterpoint to `lread`. Where `lread` trades a
quorum round-trip for a **time lease**, `read_index` keeps **no clock assumption** and instead
**confirms leadership with a majority** (the Raft ReadIndex heartbeat round) before serving the
leader's local replica. The two reads' availability profiles are opposite, and that is the lesson: a
leader **stranded in a minority** is `no_quorum` on `read_index` (it cannot confirm it is still leader)
exactly where a live-lease `lread` *does* serve locally — the lease buys minority read-availability,
the quorum read buys freedom from the clock. Both refuse the stale read a deposed leader would
otherwise serve: a leader fenced by a higher-term election is `not_leader` even after `heal`, where a
plain `get` from that node returns its now-stale local replica. `read_index` touches no replica (a pure
read) and reads its majority confirmation from the medium, so Tier-A ≡ Tier-B compute the identical
verdict; it adds no state field and no edit type (purely additive).

```
elect n0; append n0 x a; read_index n0 x           # ("ok","a") — leadership confirmed by a majority
read_index n1 x                                     # ("not_leader","n0") — only the leader serves it
partition n0 n1 | n2 n3 n4; read_index n0 x         # ("no_quorum","") — minority leader cannot confirm
lease n0 9; lread n0 x; read_index n0 x             # lread ("ok","a") | read_index ("no_quorum","")
```

ED32 ([`experiments/ed32.py`](../src/verisim/experiments/ed32.py)) pins both panels (the two reads'
opposite availability; the no-stale-read safety) with Tier-B agreeing on every read verdict.

**`append node key val` — the replicated log (DS0 increment 19).** Where increment 16's `propose`
was a *one-shot* leader-fenced commit, `append` adds the explicit **Raft log** underneath: the
leader appends a `LogEntry(term, index, key, value)` to its log and replicates it to the reachable
followers, who **adopt the leader's prefix** in one step (overwriting any divergent *uncommitted*
tail — the log-matching reconciliation). The entry **commits iff a majority holds it**, advancing the
monotone `commit_index` and folding the committed prefix into the KV state machine (a key's version
is the count of its committed writes, its value the last) — which also **backfills a rejoined
follower** that missed committed entries while partitioned, so the KV never diverges from the
committed log. Two safety properties the one-shot `propose` could not express: a committed entry is
**permanent** (the commit index is monotone — the metamorphic tier refutes any backward move), and an
**uncommitted** entry (a minority leader appended it but could not reach a majority) is never applied
to the KV and may be **overwritten** by a higher-term leader's entry at the same index.

```
elect n0; append n0 x a                            # committed a@0 (majority): commit_index 1, KV x=a
partition n0 | n1 n2; append n0 x b                # ("uncommitted","1") — b@1 on n0's log, NOT in KV
elect n1; append n1 x c                            # term 2 commits c@1 on {n1,n2}: commit_index 2
heal; append n1 x d                                # n0 reconciles: b@1 overwritten by c@1, then d@2
```

`append` reads the majority from the partition/down medium (a coordinator-level decision, like
`propose`), so Tier-A and Tier-B compute byte-identical log/commit/KV deltas via the shared
`append_edits` helper; the `logs`/`commit_index` are omitted from the canonical form until the first
`append`. ED26 ([`experiments/ed26.py`](../src/verisim/experiments/ed26.py)) pins both panels
(commit-requires-a-majority; log-matching reconciliation) with Tier-B agreeing on every transition.

**`add_replica node` / `remove_replica node` — membership change (DS0 increment 20).** The voting
set that defines a quorum is itself reconfigurable. `add_replica`/`remove_replica` install a new
`members` set (a leader-committed `MemberSet`), and every quorum computation — election votes,
`propose`/`append` majority — routes through it (`active_members` resolves the empty "all config
nodes vote" sentinel). So the **majority threshold tracks the membership**: shrinking the voting set
lowers it (a minority becomes a majority — the lever an operator pulls to **restore availability**
after losing nodes: a lone survivor of a 3-node cluster commits again once the two dead are removed),
and growing it raises the threshold back. The change is fenced for safety — the *active leader*
cannot be removed (`is_leader`), the *last* member cannot be removed, and a change needs a current
leader to commit it. All config nodes still physically store replicas; `members` is the voting
overlay, omitted from the canonical form until the first change.

```
elect n0; partition n0 | n1 n2; propose n0 x a    # ("no_quorum","") — n0 alone is 1 of 3 members
remove_replica n1; remove_replica n2              # members → {n0}; the threshold drops to 1
propose n0 x a                                    # ("ok","a") — a majority of one, availability restored
add_replica n1                                    # members → {n0,n1}; the threshold rises to 2
```

Membership is coordinator-level cluster metadata, so `add_replica`/`remove_replica` are computed
byte-identically by Tier-A and Tier-B. ED27 ([`experiments/ed27.py`](../src/verisim/experiments/ed27.py))
pins both panels (the threshold tracks the votes; restore-availability) with Tier-B agreeing on every
transition. (Honest scope: membership is the *voting* overlay and the change is committed by leader
fiat; real Raft commits a config change as a log entry under joint consensus to make *concurrent*
reconfigurations safe — deferred.)

### 3.4 `enqueue` / `dequeue` — the distributed FIFO queue (DS0 increment 21)

A **second client data type** beside the KV store: a replicated FIFO queue. `enqueue node queue val`
appends `val` to each reachable replica's queue list; `dequeue node queue` returns the head of the
coordinator's local replica and pops the head from each reachable replica. Queues are fully
replicated (every node), and a queue replica is created lazily on first `enqueue` (omitted from the
canonical form until then). The headline is that **the queue's delivery guarantee is the consistency
model's, not the queue's** — the same `_queue_available` gate that decides a KV write's availability
decides the queue's:

- **`eventual` / `causal`** — proceed on the reachable set. Under a partition a `dequeue`'s
  head-removal reaches only one side, so a peer on the other side still holds the item and can
  `dequeue` it **again**: **at-least-once / duplicate delivery**. Available, not exactly-once.
- **`quorum`** — needs a reachable majority. The majority side serves the `dequeue` (exactly-once);
  the minority is `unavailable`. The realistic CP middle.
- **`linearizable`** — needs *every* replica. Under any partition both sides are `unavailable`:
  never duplicated, but no progress.

So the delivery count of one item, dequeued from both sides of a partition, steps `2 → 1 → 0` as the
model strengthens — the KV fork-vs-availability tradeoff (§2.3, ED14) restated for a queue. On the
connected path the queue is a correct FIFO: `enqueue a, b, c` then three `dequeue`s return `a, b, c`
in order, each exactly once, then `empty`.

```
enqueue n0 q a (full conn → [a] on every node); partition n0 | n1 n2
dequeue n0 q → ("dequeued","a")     # eventual: n0 pops its head...
dequeue n1 q → ("dequeued","a")     # ...and n1 still has [a] → the item is delivered TWICE
```

Queue ops touch no replica (a separate data plane) and the reachable set is read from the medium (a
coordinator-level decision), so Tier-A and Tier-B compute byte-identical queue deltas via the shared
`enqueue_edits`/`dequeue_edits` helpers, and `cluster_view` now includes the queue replicas so the
differential catches a queue divergence. ED28 ([`experiments/ed28.py`](../src/verisim/experiments/ed28.py))
pins both panels with Tier-B agreeing on every transition. (Honest scope: queue replication is
synchronous to the reachable set — no async in-flight medium or anti-entropy for queue ops yet, so a
divergent replica is reconciled only by a fresh op that reaches it; the consistency model gates
availability, which is the lever the delivery semantics turn on.)

### 3.5 `deploy` — the rolling upgrade (DS0 increment 22)

The op that answers SPEC-7's headline operational question, *"will this deploy break the cluster?"*
`deploy node version` sets a node's running software `version` (a node-local change — restarting a
node with new code needs no consensus). The consequence lives in the quorum: every consensus
computation (`elect`/`propose`/`append`) routes its reachable/voter set through `_compatible`, which
admits a peer only if its version is within `DistConfig.max_version_skew` of the coordinator's
(default `1` — the N-1 rolling-upgrade window every real system guarantees). So:

- A **rolling upgrade** that advances nodes one at a time keeps the version spread inside the window
  (`v0` and `v1` coexisting is spread `1 ≤ 1`), so a compatible majority always exists and consensus
  never stalls — the deploy is safe.
- A deploy that creates an **incompatible split with no compatible majority** (e.g. half the cluster
  at `v0`, half at `v2`, spread `2 > 1`) leaves no version cohort able to form a quorum, so the next
  `elect`/`propose` is `no_quorum` — *the deploy broke the cluster.*

The diagnostic that pins the cause: the *same* version assignment is safe at a smaller spread, or
under a wider configured `max_version_skew` — it is the spread **exceeding the compatibility window**
that breaks consensus, not the mere presence of mixed versions.

```
elect n0; deploy n0 1; propose n0 x a    # spread 1 ≤ skew 1 → ("ok","a"), the rolling step is safe
deploy n0 2; deploy n1 2                  # 2×v2 | 2×v0 in a 4-node cluster: spread 2 > skew 1
propose n0 x b                            # ("no_quorum","") — no compatible majority: the deploy broke it
```

Compatibility gates *consensus* only — the best-effort KV/queue data plane (`put`/`get`/`enqueue`)
ignores versions. A node's version is observable cluster metadata read coordinator-side, so Tier-A
and Tier-B compute byte-identical version deltas via the shared `deploy_edits` helper, and
`cluster_view` includes the versions; the version map is omitted from the canonical form until the
first `deploy`. ED29 ([`experiments/ed29.py`](../src/verisim/experiments/ed29.py)) pins both panels
(the safe rolling upgrade; the break + the spread-vs-window diagnostic) with Tier-B agreeing on every
transition. (Honest scope: version compatibility is a symmetric within-skew relation and the deploy
is a node-local label; a real upgrade's per-feature compatibility matrix and staged migrations are a
deeper refinement.)

### 3.6 `host` — the embedded SPEC-6 host (DS0 increment 23)

The compositional vision SPEC-7 names since increment 1 (§3.1/§4: a `HostDelta` on an embedded
subsystem), realized: each cluster node runs a real **SPEC-6 host** — a process table, per-process
file-descriptor tables, and an embedded **v0 filesystem** (composed verbatim, as SPEC-6 itself
composes the v0 FS). `host node <syscall...>` runs a SPEC-6 syscall on `node`'s own host by
delegating to the SPEC-6 `ReferenceHostOracle`, and wraps its bundle delta in a `HostStep(node,
edits)` — so the dist `apply` reproduces the embedded host through the SPEC-6 `apply` verbatim, which
in turn reproduces the embedded filesystem through the v0 `apply`. The composition is three layers
deep and visible right down to serialization (a node's host canonical contains the v0 FS canonical).

Three properties:

- **Composition** — a node is no longer just a bag of KV replicas: it serves a KV `put` *and* a host
  `fork` independently, and a `host open` + `write` materializes a file in that node's embedded FS.
- **Per-node isolation** — a node's host is created lazily on its first host op and is entirely its
  own; a `fork` on `n0` never appears on `n1`'s host.
- **The cross-layer crash linkage** — host ops obey the *same* up/down gate the KV client ops do: a
  `host` syscall on a crashed node is `unavailable`, and `restart` resumes it. The host state
  **survives** the crash (a crash pauses the node, it does not wipe its process table or FS).

```
host n0 fork 1            # ("ok","2") — pid 2 on n0's host; n1's host is untouched (isolation)
put n0 x b               # the KV subsystem, independent, on the same node
host n0 open 1 /f; host n0 write 1 0 a   # the file /f lands in n0's embedded v0 filesystem
crash n0; host n0 fork 1  # ("unavailable","") — the crash gate reaches the host
restart n0; host n0 fork 1  # ("ok","3") — host ops resume; the process table survived the crash
```

`host` delegates to the SPEC-6 oracle on the node's own state (a node-local computation, no medium
interaction), so Tier-A and Tier-B compute byte-identical host deltas via the shared `host_op_edits`
helper; the embedded hosts join `cluster_view`, and the `hosts` map is omitted from the canonical
form until the first `host` op. ED30 ([`experiments/ed30.py`](../src/verisim/experiments/ed30.py))
pins both panels (composition + isolation; the crash linkage) with Tier-B agreeing on every
transition. (Honest scope: only the SPEC-6 HC0 syscall subset — process/fd/cred/FS — is wired here;
sockets, IPC, the scheduler, and the SPEC-5 network embedded *between* hosts are later increments.)

### 3.7 `config_push` — the leader-committed config change (DS0 increment 24)

The op that answers SPEC-7's *other* headline operational question — *"will this config push break
the cluster?"* — the sibling of `deploy` (§3.5). The two are deliberately different in *mechanism*,
which is the point:

- **`deploy`** sets a **node-local** software `version` (no consensus — you restart a node with new
  code without asking anyone), and the version gates consensus *compatibility* (the N-1 skew window).
- **`config_push node key val`** is a **leader-committed, majority-replicated** cluster setting — a
  Raft-style config *entry*. It shares the leader-fence + majority-reachability rule of
  `propose`/`append`: only the current leader may push, and it commits only on a majority.

The semantics:

- A push by a **non-leader** is `("not_leader", current_leader)`; a push with **no leader** elected
  is `("not_leader", "")` — config changes go through consensus, not any node that asks.
- A leader **stranded in the minority** is `("no_quorum", "")`, and **not a single node's config
  changes** — a config rollout is all-or-nothing at commit (a minority side never installs a value it
  cannot durably hold). A crashed leader is `("unavailable", "")`.
- On commit `("committed", val)`, the value is written (`ConfigSet`) to **every reachable voting
  member**. Under a partition that leaves the leader on the majority side, this reaches only the
  majority — so the **partitioned minority retains its stale config** (config divergence, the
  broken-cluster outcome). The repair is a **re-push after `heal`**, which converges every node.

```
elect n0; config_push n0 feature on        # ("committed","on") — reaches every voting member
config_push n1 feature off                 # ("not_leader","n0") — only the leader may push
partition n0 n1 | n2 n3 n4                  # strand the leader n0 in the 2-of-5 minority
config_push n0 feature off                 # ("no_quorum","") — and no node's config changes
heal; partition n0 n1 n2 | n3 n4           # n0 now on the 3-of-5 majority
config_push n0 feature v2                   # ("committed","v2") on n0/n1/n2; n3/n4 keep stale "on"
heal; config_push n0 feature v2            # the re-push converges all five nodes
```

The commit quorum is read from the partition/down medium (a coordinator-level decision, not an
actor's local view), so Tier-A and Tier-B compute byte-identical config deltas via the shared
`config_push_edits` helper — the divergence-under-partition is reproduced on the autonomous actors
too. The `config` map joins `cluster_view`, is omitted from the canonical form until the first push
(purely additive), and gates **nothing** in the data plane (it is observable cluster metadata a model
must learn to predict, not a control on the KV/queue). ED31
([`experiments/ed31.py`](../src/verisim/experiments/ed31.py)) pins both panels (the leader-committed
rollout + fence; the partition divergence + repair) with Tier-B agreeing on every transition.

### 3.8 `delete` — the tombstone delete and the resurrection problem (DS0 increment 26)

The fundamental KV **remove** — and a canonical distributed hazard the design must get right. The key
decision: a `delete node key` is a **versioned write of a tombstone**, *not* a removal of the replica.
It reuses the `put` write path with the sentinel `TOMBSTONE` value (`__deleted__`), so it inherits
every consistency model (eventual/causal/quorum/linearizable), bumps the version like any write, and a
`get` on a tombstoned replica reports `("deleted", "")`.

Why a tombstone and not an erasure? Because erasure causes the **resurrection problem**. If a delete
simply removed the replica, then under a partition the deleted key still exists on the unreachable
side, and when the network heals the merge has no way to tell "deleted" from "never seen" — the stale
replica's old value wins and the key **comes back from the dead** (the classic Dynamo/Cassandra bug
tombstones exist to prevent). With a *versioned* tombstone, the delete is just another write ordered
by version: its higher version **wins the last-writer-wins merge** over any older value, so
`anti_entropy`/`gossip` converge a lagging replica to `deleted`, not to the resurrected value.

```
put n0 x a; advance 5                       # x=a (version 1) on every replica
partition n0 n1 n2 | n3 n4; delete n0 x      # ("deleted","") — tombstone (version 2) on the majority
advance 5; get n3 x                          # ("ok","a") — the partitioned minority still reads it!
heal; anti_entropy n3; get n3 x              # ("deleted","") — the tombstone's higher version wins
put n0 x b; advance 5; get n3 x              # ("ok","b") — a genuinely newer write legitimately returns
```

`delete` reuses the `put` write path, so Tier-A and Tier-B compute byte-identical deltas (the
divergence-under-partition and the version-ordered convergence are reproduced on the autonomous actors
too). The tombstone is just a replica value — it adds no state field and no edit type, so the op is
purely additive (every prior golden/hash holds). ED33
([`experiments/ed33.py`](../src/verisim/experiments/ed33.py)) pins both panels (the versioned
tombstone; resurrection under partition + the anti_entropy/gossip repair) with Tier-B agreeing on
every transition. (Honest scope: the tombstone is never garbage-collected here — real systems expire
tombstones after a grace period, a Tier-B/operational refinement; keeping them is the conservative,
always-correct choice for the reference oracle.)

### 3.9 `incr` — the atomic counter and the lost-update problem (DS0 increment 27)

The first **read-modify-write** client op (`put`/`cas`/`delete` are blind or compare writes), and the
canonical demonstration that **you cannot build a correct counter on last-writer-wins**. An `incr node
key` reads the coordinator's local counter (a non-numeric/absent value is `0`) and writes `count + 1`
at a bumped version, reusing the `put` replication path — so it inherits every consistency model.

With no concurrency it is exactly correct (`incr` `k` times → `k`, under every model). The hazard is
*concurrent* increments, and it is **strictly worse than the blind-write CAP tradeoff** (ED14): a
blind `put` that loses a race merely leaves a replica *stale* (a value that a newer write or
anti_entropy will fix), but a lost `incr` is *gone* — the count is permanently short. Under a
partition, two `incr`s on opposite sides both read the same count `N` and both write `N+1` at the same
version; the eventual-consistency merge keeps one of the two same-version writes (LWW + tiebreak), so
**one increment is silently lost** even though both were acknowledged. `quorum` avoids the silent loss
by making the minority side `unavailable` (only the accepted increment counts), and `linearizable`
rejects the write under any partition.

```
incr n0 c; advance 5                          # c = 1 on every replica
partition n0 n1 n2 | n3 n4
incr n0 c                                      # ("ok","2") on the majority side
incr n3 c                                      # ("ok","2") on the minority — both acknowledged
heal; advance 5; anti_entropy n0; get n0 c     # ("ok","2") — expected 3: one increment was lost
```

`incr` reuses the `put` write path, so Tier-A and Tier-B compute byte-identical deltas (the lost
update is reproduced on the autonomous actors too). The counter is just a digit-valued replica — no
new state field, no new edit type — so the op is purely additive, and the metamorphic tier admits any
digit string as a legal counter value. ED34
([`experiments/ed34.py`](../src/verisim/experiments/ed34.py)) pins both panels (sequential
correctness; the read-modify-write CAP tradeoff across the three models) with Tier-B agreeing on every
transition. (A loss-free eventual counter is a **CRDT/PN-counter** — per-node sub-counters merged by
max — built next in increment 28; the point here is the negative the simple model exhibits.)

### 3.10 `cincr` / `cget` — the CRDT G-counter (DS0 increment 28)

The **loss-free, always-available resolution** to `incr`'s negative — a *state-based* grow-only
counter (the canonical CRDT). The state is a vector: per `(key, holder, owner)`, node `holder`'s copy
of `owner`'s monotone sub-count; the counter's value at a node is the **sum over owners**. The
discipline that makes it work is *single-writer-per-entry*: `cincr n key` bumps **only `n`'s own**
sub-count (`GCounterSet(key, n, n, +1)`), so two concurrent `cincr`s at different nodes touch
*disjoint* vector entries — there is no conflict and **nothing to lose**. The CRDT **join is the
per-(key, owner) max**, a commutative, associative, idempotent merge applied by `anti_entropy` (pull
into one node) and `gossip` (merge a pair); applying it in any order, any number of times, converges
every node to the same vector — the exact total.

Two properties distinguish it sharply from the LWW `incr`:

- **Always available (AP).** `cincr` is purely node-local — no replication, no in-flight message — so
  it succeeds whenever the node is up, *including a partitioned-alone node*, where the LWW `incr`
  under `quorum`/`linearizable` is `unavailable`. Availability with no coordination.
- **No lost update.** The three-increment partition that lost one under `incr` (ED34) keeps all three
  here: each side's increments land on different owners' sub-counts, and the join sums them.

```
partition n0 n1 n2 | n3 n4
cincr n0 c; cincr n0 c          # ("ok","1"), ("ok","2") — n0's own sub-count
cincr n3 c                      # ("ok","1") — the minority node still counts (AP)
heal; gossip n0 n3; cget n0 c   # ("ok","3") — 2 (n0) + 1 (n3), no lost update
```

`cincr` is node-local and the merge is a coordinator-level read of the medium, so Tier-A and Tier-B
compute byte-identical deltas. The `gcounters` map joins `cluster_view`, is omitted from the canonical
form until the first `cincr` (purely additive), and the metamorphic tier checks sub-counts are
non-negative and held/owned by known nodes. ED35
([`experiments/ed35.py`](../src/verisim/experiments/ed35.py)) pins both panels (loss-free +
always-available; gossip/anti_entropy convergence + idempotence) with Tier-B agreeing on every
transition. (Honest scope: this is a **grow-only** G-counter — a decrementable **PN-counter** pairs
two G-counters, built next in increment 29.)

### 3.11 `cdecr` — the CRDT PN-counter (DS0 increment 29)

A G-counter only goes up. The **PN-counter** is the standard CRDT way to make a counter
*decrementable*: pair **two** G-counters — `P` (the `cincr` half, `gcounters`) and `N` (the `cdecr`
half, `ncounters`) — and read the value as **`P − N`**. `cdecr n key` is the exact twin of `cincr`
over the N half: it bumps **only `n`'s own** decrement sub-count (`NCounterSet(key, n, n, +1)`), so it
inherits every property that made the G-counter work — purely node-local (**always available**, the AP
property), single-writer-per-entry (concurrent decrements touch *disjoint* entries, **nothing to
lose**), and merged by the same per-(key, owner) **max** join. The join over the full PN-counter is
just the join over each half independently, so `anti_entropy`/`gossip` reconcile `P` and `N` together
and every node converges to the exact net.

The one thing the PN-counter adds is the property the G-counter *lacked*: the value may go
**negative**. (The sub-counts are still individually monotone and non-negative — only their
*difference* can dip below zero. The metamorphic tier enforces exactly this: a negative *sub-count* is
impossible, a negative *value* is fine.)

```
cdecr n0 c                      # ("ok","1") — n0's own decrement sub-count; cget now reads -1
partition n0 n1 n2 | n3 n4
cincr n0 c; cincr n0 c          # ("ok","1"), ("ok","2") — majority side, P[n0]=2
cdecr n3 c                      # ("ok","1") — the minority node still counts down (AP)
heal; gossip n0 n3; cget n0 c   # ("ok","1") — +2 (n0) − 1 (n3), no lost update across either half
```

`cdecr` is node-local and the merge is a coordinator-level read of the medium, so Tier-A and Tier-B
compute byte-identical deltas. The `ncounters` map joins `cluster_view`, is omitted from the canonical
form until the first `cdecr` (purely additive over increment 28), and a cluster that only ever
`cincr`-s is byte-identical to the pre-increment-29 form. ED36
([`experiments/ed36.py`](../src/verisim/experiments/ed36.py)) pins both panels (decrement works +
loss-free + goes-negative + always-available; gossip/anti_entropy convergence + idempotence over both
halves) with Tier-B agreeing on every transition.

### 3.12 `sadd` / `srem` / `smembers` — the CRDT OR-Set (DS0 increment 30)

The CRDT counter was a number; the OR-Set is the canonical *interesting* CRDT — a replicated **set**,
the data type a naive implementation gets wrong. The tempting design, an element-level **2P-Set** (a
grow-only add-set plus a grow-only remove-set, `member iff in adds and not in removes`), has two
defects: it is **remove-wins** (a concurrent add and remove of the same element resolve to *absent*),
and once removed an element can **never be re-added** (its identity is permanently in the remove-set).
Both are unacceptable for a real set.

The **observed-remove set** fixes both with a **unique dot**. `sadd n key elem` does not just record
"elem present" — it tags the element with a fresh dot `(owner=n, seq)`, where `seq` is `n`'s next
monotone sequence for `key` (the same single-writer-per-owner discipline as the G-counter), and stores
`(elem, n, seq)` in `n`'s observed add-set. `srem n key elem` is the *observed*-remove: it tombstones
**only the dots of `elem` that `n` has actually observed**, not the element itself. `smembers` is the
elements with at least one add-dot whose `(owner, seq)` is **not** tombstoned. The CRDT join is **set
union** of both halves, applied by `anti_entropy`/`gossip` — commutative, associative, idempotent.

The dot mechanism buys exactly the two properties the 2P-Set lacks:

- **Add wins.** A concurrent `sadd` mints a *fresh* dot that the concurrent `srem` never observed, so
  the union keeps it un-tombstoned and the element survives. (A `srem` can only remove what it has
  seen; a brand-new add is, by construction, unseen.)
- **Re-addable.** Re-adding a removed element mints a *new* dot, which is not in the tombstone-set, so
  the element returns — the identity is per-*dot*, not per-*element*.

```
sadd n0 s x; anti_entropy n2            # x present cluster-wide (dot (n0,1) at n0 and n2)
partition n0 n1 | n2
sadd n0 s x                             # ("ok","x") — concurrent re-add, a FRESH dot (n0,2)
srem n2 s x                             # ("ok","x") — n2 tombstones only the dot it saw, (n0,1)
heal; gossip n0 n2; smembers n0 s       # ("ok","{x}") — dot (n0,2) survives: ADD WINS
```

`sadd`/`srem` are node-local and the merge is a coordinator-level read of the medium, so Tier-A and
Tier-B compute byte-identical deltas. The `orset_adds`/`orset_tombs` maps join `cluster_view`, are
omitted from the canonical form until the first set op (purely additive), and the metamorphic tier
checks every dot is held/owned by a known node with a positive sequence. ED37
([`experiments/ed37.py`](../src/verisim/experiments/ed37.py)) pins both panels (reads-back + re-addable
+ add-wins + always-available; gossip/anti_entropy convergence + idempotence) with Tier-B agreeing on
every transition.

### 3.13 `mvput` / `mvget` — the CRDT MV-register (DS0 increment 31)

The counters and the KV `put` resolve a write conflict by **last-writer-wins**: one value survives, the
other is silently dropped (ED14, ED34). The OR-Set unions *disjoint* adds. The **multi-value register**
(MV-register) is the third option, the one Dynamo/Riak chose for the shopping cart: when two writes
conflict, **keep both** as *siblings* and surface the conflict to a later reader, who resolves it. It
is the data type that makes a conflict *visible and resolvable* rather than *lost*.

It reuses the OR-Set's dot/union machinery exactly. `mvput n key val` tags `val` with a fresh dot
`(owner=n, seq)`, **tombstones every dot it currently observes** (the surviving values it can see — a
write supersedes what it saw), and stores its own write-dot. `mvget n key` is the values whose
write-dot is not tombstoned. The CRDT join is **set union** of both halves, applied by
`anti_entropy`/`gossip`. The "supersede what you observed" rule is the whole design:

- **Sequential overwrite resolves.** A second write by the same node *observed* the first, so it
  tombstones it — one value survives. The common case is a plain register.
- **Concurrent writes become siblings.** Two writes on opposite sides of a partition each observed
  only their own side, so neither tombstones the other; the union keeps **both**. `mvget` returns
  `{a,b}` — the conflict, made visible, where a LWW `put` would have dropped one.
- **A later write resolves the conflict.** Once a node has *seen* both siblings (after convergence), a
  fresh `mvput` tombstones both and installs one value — the Dynamo read-and-resolve.

```
partition n0 n1 n2 | n3 n4
mvput n0 r a                            # ("ok","a") — majority side, dot (n0,1)
mvput n3 r b                            # ("ok","b") — minority side, concurrent dot (n3,1)
heal; gossip n0 n3; mvget n0 r          # ("ok","{a,b}") — BOTH survive: the conflict surfaced
mvput n0 r c; gossip n0 n3; mvget n3 r  # ("ok","{c}") — n0 saw both, so c supersedes them
```

`mvput` is node-local and the merge is a coordinator-level read of the medium, so Tier-A and Tier-B
compute byte-identical deltas. The `mvreg_vals`/`mvreg_tombs` maps join `cluster_view`, are omitted
from the canonical form until the first register op (purely additive over increment 30), and the
metamorphic tier checks every dot is held/owned by a known node with a positive sequence. ED38
([`experiments/ed38.py`](../src/verisim/experiments/ed38.py)) pins both panels (basic-read +
sequential-resolve + siblings-preserved + always-available; gossip/anti_entropy convergence +
idempotence + context-resolve) with Tier-B agreeing on every transition.

### 3.14 `lwwput` / `lwwget` — the CRDT LWW-register (DS0 increment 32)

The MV-register *surfaces* a write conflict; the **LWW-register** is its policy-opposite — it
**deterministically picks one winner**. It is the register CRDT for the common case where you want a
single value and an automatic resolution rule, not a sibling set to merge by hand. The mechanism is a
**Lamport clock** — a per-node logical counter that makes "happens-after" a comparable order without a
real clock (which a partitioned cluster cannot share, HW-5).

`lwwput n key val` stamps `val` with `(ts, owner=n)` where `ts = lamport[n] + 1` (advancing `n`'s
Lamport clock), and stores it as `n`'s current register value. The CRDT join keeps the **max** copy by
`(ts, owner, value)`, and each node advances its Lamport clock to the max timestamp it observes (so a
later local write always out-stamps what it overwrote — the invariant that keeps the join convergent).
`lwwget n key` reads the single winning value. Two properties follow:

- **Happens-after wins, regardless of node.** A write that *observed* another has a strictly higher
  Lamport ts, so it wins — even if a *lower*-id node made it. The timestamp encodes causality; the
  node id is only the tie-break, not the deciding factor. (A naive "highest node id wins" gets this
  backwards.)
- **Concurrent writes resolve deterministically.** Two causally-unrelated writes have the same ts, so
  the tie breaks by node id (then value) — a single, stable winner the whole cluster agrees on. The
  price the MV-register avoids: the concurrent *loser* is dropped (one value, not siblings).

```
lwwput n2 w a; anti_entropy n0          # a@(ts1, n2); n0 observes it, its clock -> 1
lwwput n0 w b                           # b@(ts2, n0) — later, so HIGHER ts, even though n0 < n2
gossip n0 n2; lwwget n2 w               # ("ok","b") — happens-after wins, regardless of node id
```

`lwwput` is node-local and the merge is a coordinator-level read of the medium, so Tier-A and Tier-B
compute byte-identical deltas. The `lwwreg`/`lamport` maps join `cluster_view`, are omitted from the
canonical form until the first register op (purely additive over increment 31), and the metamorphic
tier checks every entry is held/owned by a known node with a positive timestamp and a non-negative
clock. ED39 ([`experiments/ed39.py`](../src/verisim/experiments/ed39.py)) pins both panels (basic-read
+ causal-LWW + deterministic-resolution + always-available; gossip/anti_entropy convergence +
idempotence + loser-dropped) with Tier-B agreeing on every transition.

## 4. The delta and the `apply == oracle` invariant

The oracle returns the next state **and** the structured delta that produces it
([`verisim.dist.delta`](../src/verisim/dist/delta.py)): `ReplicaWrite`, `MsgSend`/`MsgDeliver`/
`MsgDrop`, `EventAppend`, `PartitionSet`, `NodeDown`/`NodeUp`, `ClockSet`, the consensus
`ProtocolStep`/`LeaseSet`, the replicated-log `LogSet`/`CommitIndexSet`, the membership `MemberSet`,
the FIFO-queue `QueueSet`, the per-node `VersionSet`, the embedded-host `HostStep`, the cluster-config
`ConfigSet`, the CRDT-counter `GCounterSet` (DS0 incr 16–28), and `SetResult`. The **M1-analogue invariant** holds by construction and is tested on every
transition:

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

### 8.1 Tier-B under causal consistency (DS0 increment 6)

Tier-B honors the `causal` model too — the W1 retirement extended to the third consistency end, and a
**stronger** test than `eventual`'s. Under `eventual` a correct actor may deliver in any order (LWW is
a commutative join), so the seed-shuffle is the whole challenge. Under `causal` the actor must *also*
hold a message until its `deps` are applied locally — and because the shuffle may try a message
*before* its cause, Tier-B's `_advance` runs delivery to a **fixed point**: it repeatedly scans the
not-yet-delivered messages, delivering any whose deps are now satisfied at the destination actor (read
from the actor's *own* replicas — the no-global-state guarantee), until a pass delivers nothing. A
message whose deps never arrive stays in flight. The fixed point delivers exactly the causally-ready
closure, **independent of the shuffle**, so it reproduces Tier-A's sorted-order result (message ids
are topologically ordered: a causally-later write always has a higher id, so Tier-A's single sorted
pass is already a valid causal order). Both oracles attach deps via the *shared* `causal_deps` helper,
and the differential's observable channel now includes `deps`, so a Tier-B that mis-computed or
mis-ordered them would be caught. Across the exhaustive grammar battery and all three drivers (1080+
steps) Tier-A and Tier-B agree **bit-for-bit under `causal`**, the broken-arrival control is still
caught, and the held-message anomaly (ED13) is reproduced exactly — a causal `M_θ` would be graded
against a real autonomous-actor execution, not only the analytic DES.

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
pinned async-replication LWW before consensus. *The two OCC isolation levels
(`serializable`/`snapshot`) ship in §9.1; the pessimistic **2PL** alternative — deterministic via
wound-wait — ships in §9.1.1.*

**Composition.** A committed transaction's writes flow through the existing replication medium, so
they inherit the consistency model unchanged, **exactly as a plain `put` does** (§2.1–2.3): under
`eventual`/`causal` the peers converge on a later `advance`; under `quorum` the commit replicates
synchronously to a reachable majority (async catch-up to the minority) and is rejected if no majority
is reachable; under `linearizable` it replicates to *every* replica synchronously and is **rejected**
(`unavailable`, the txn staying open for retry) if it cannot reach all under partition. The
transaction state is purely additive: an empty `txns` set is **omitted** from the canonical form
(§6), so every DS0-increment-1 golden and content-addressed hash is unchanged, and Tier-B reproduces
every transaction trajectory bit-for-bit (the commit replication is delivered by its autonomous
actors on `advance`, where its independence does its work). The OCC commit/abort frontier is pinned
by **ED8** ([`verisim.experiments.ed8`](../src/verisim/experiments/ed8.py),
[`ed8.png`](../../figures/ed8.png)): at concurrency `K` over `M` objects the measured commit rate
tracks the balls-in-bins occupancy law `M·(1−(1−1/M)^K)/K` (the semantics are exactly right, not
merely plausible), with Tier-B agreeing on every scenario.

### 9.1 Isolation levels — `serializable` / `snapshot` / `read_committed` / `read_uncommitted` (DS0 increments 3, 9, 10)

The `txn_isolation` config dial selects *what a read sees* and *which set* `commit` validates (design
decision `DD-D4`). The four are the standard SQL isolation hierarchy, ordered strong → weak (weaker
admits more anomalies and so is *harder to predict*, SPEC-7 §3.4 — `read_uncommitted ⊂ read_committed
⊂ snapshot ⊂ serializable`):

| Level | Validates at commit | Reads see | Forbids write skew? | Forbids lost update? | Forbids dirty read? |
|---|---|---|---|---|---|
| `serializable` (default) | the **read-set** — every read key's local version must be unchanged since it was read (OCC backward validation, Kung–Robinson) | committed data (MVCC) | **yes** | **yes** | **yes** |
| `snapshot` | only the **write-set** — every written key's version must be unchanged since the txn first wrote it (write-write conflict, first-committer-wins) | committed data (MVCC) | **no** — disjoint write-sets both commit | **yes** — a same-key write-write conflict still aborts | **yes** |
| `read_committed` | **nothing** — no concurrency validation at all | committed data (MVCC) | **no** | **no** — two same-key RMW txns both commit, the later overwrites the earlier | **yes** — the MVCC `tget` gives no dirty reads |
| `read_uncommitted` | **nothing** (the weakest level) | **uncommitted** data — a `tget` may observe another active txn's buffered write | **no** | **no** | **no** — if the observed writer aborts, the reader saw a value that never committed |

The `serializable`/`snapshot` distinction is the classic **write-skew** anomaly (two transactions
both read `{x, y}`, then `A` writes `x` and `B` writes `y`): under `snapshot` the disjoint write-sets
both pass write-write validation and **both commit** — a pair of outcomes no serial schedule
produces; under `serializable`, `A`'s commit bumps `x`, so `B`'s pinned read of `x` is stale and `B`
**aborts**. Pinned by **ED9** ([`ed9.png`](../../figures/ed9.png)): write-skew rate **1.0 under
`snapshot`, 0.0 under `serializable`**, and under read-heavy contention `serializable` aborts
strictly more (`0.70` vs `0.55`, disjoint CIs) — the price of the stronger guarantee.

The `snapshot`/`read_committed` distinction is the classic **lost-update** anomaly — the real-world
default of Postgres/Oracle/SQL-Server made measurable. Two transactions both read `x` at the same
version and both write it back (a read-modify-write):

- under `read_committed`, the commit validates *nothing*, so **both commit** and only the later
  write survives — the earlier transaction's update is silently lost despite committing successfully;
- under `snapshot` (and `serializable`), the second committer's write-set validation sees `x`'s
  version bumped by the first, so it **aborts** — the update is preserved.

All three levels remain OCC (deterministic, deadlock-free) and share every other rule above; only the
validation set differs, so the write version is pinned at first `tput` (the `write_versions` field on
`TxnState`) exactly as the read version is pinned at first `tget`. The lost-update anomaly and its
price are pinned by **ED16** ([`verisim.experiments.ed16`](../src/verisim/experiments/ed16.py),
[`ed16.png`](../../figures/ed16.png)): the lost-update anomaly rate is **1.0 under `read_committed`,
0.0 under `snapshot` and `serializable`**, and under read-modify-write contention `read_committed`
**never aborts** (`0.00` vs `~0.53`) — the apparent throughput it buys by selling the correctness of
the first panel. All these levels compose with Tier-B (the autonomous-actor system oracle agrees on
every scenario, transaction bookkeeping being coordinator-local).

The `read_committed`/`read_uncommitted` distinction is the last rung — the classic **dirty-read**
anomaly (Adya G1a). `read_committed` keeps one guarantee `read_uncommitted` drops: that a read sees
only *committed* data. `read_uncommitted`'s `tget` may instead observe another active transaction's
**uncommitted** buffered write (when several have, the lexicographically-greatest txn id wins — a
deterministic stand-in for "the latest uncommitted writer"; the canonical two-transaction scenario
has exactly one other writer). So if `A` writes `x` (uncommitted), `B` reads `x`, and then `A`
**aborts**, under `read_uncommitted` `B` saw a value that never committed; under every stronger level
the MVCC `tget` gave `B` only the committed boot value. The dirty read applies **only under OCC** —
2PL's exclusive lock blocks any reader from ever seeing an uncommitted write (locking gives
serializability regardless of the declared level, as in real systems). Read-uncommitted is purely
additive (a new `txn_isolation` value; the commit path is identical to `read_committed`), so every
prior golden/hash is unchanged. Pinned by **ED17**
([`verisim.experiments.ed17`](../src/verisim/experiments/ed17.py), [`ed17.png`](../../figures/ed17.png)):
the dirty-read anomaly rate is **1.0 under `read_uncommitted`, 0.0 under the three stronger levels**,
Tier-B agrees on every scenario, and Elle's value oracle (§9.2) recovers the dirty read black-box at
exactly the oracle's rate — a `dirty-read` recovery anomaly from the client history alone.

### 9.1.1 Concurrency control: `occ` vs `2pl` (DS0 increment 8)

`DD-D3` chose **OCC** (optimistic) for the deterministic core because the *blocking* form of two-phase
locking injects nondeterminism (lock-acquisition order, deadlock detection, victim selection — all
need a scheduler). The `concurrency_control` dial adds the alternative the core *can* pin: **`2pl`**,
**strict two-phase locking with deterministic wound-wait**. `tget` acquires a **shared (S)** lock and
`tput` an **exclusive (X)** lock on the key; locks are held to `commit`/`abort` (the two phases:
growing, then shrinking). A conflict (S vs X, or X vs anything, held by another txn) is resolved by
**wound-wait**: the **older** transaction — the lexicographically smaller id, a deterministic proxy
for start order — preempts (wounds → aborts) the younger holder it conflicts with, and a requester
that is *younger* than any conflicting holder aborts itself rather than waiting. Because the older
always wins and no one ever blocks, it is **deterministic and deadlock-free without a scheduler** —
the deterministic 2PL the core pins. The lock table lives in `DistributedState.locks` (keyed by
object, the sorted `(txn_id, mode)` holders); it is empty and **omitted from the canonical form**
under the `occ` default, so the field is purely additive and every prior golden/hash is unchanged.
Under `2pl` the commit performs **no validation** — the locks already guaranteed serializability — it
just applies the buffered writes and releases the locks.

The two mechanisms reach the *same* serializable guarantee by opposite routes, so **ED15**
([`verisim.experiments.ed15`](../src/verisim/experiments/ed15.py), [`ed15.png`](../../figures/ed15.png))
measures *when each pays for a conflict*. Both forbid write skew (anomaly rate 0.0), but their **wasted
work** differs: OCC validates at commit, so an aborted txn completed *all* its operations (**3.0**
data ops wasted per abort), while 2PL fails at the conflicting lock-acquisition (**2.0** ops) — the
classic optimistic/pessimistic tradeoff, made measurable. Transaction bookkeeping (including the lock
table and wound-wait) is coordinator-local, so **Tier-B reproduces 2PL bit-for-bit** by delegating to
the same `txn_step` — the W1 retirement covers it for free.

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
**Lost update (the `read_committed` anomaly, §9.1) is a G2 of a different shape**: both write the
*same* key `x`, so the cycle carries a `ww` edge (`A` installs `x@1` before `B` installs `x@2`)
*and* an `rw` anti-dependency (`B` read `x@0`, which `A` overwrote) — a `{ww, rw}` cycle, whereas
write skew is `{rw}`-only. Both are recovered black-box (a lost-update golden pins it in
[`test_dist_goldens`](../../tests/test_dist_goldens.py)).

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

ED10 supplies the store's MVCC version order to Elle (its *version-oracle* mode). Recovering the
version order from observed **values alone** (Elle's list-append / unique-write recoverability,
Kingsbury & Alvaro 2020) shipped in **ED11** ([`recover_versions`](../src/verisim/distoracle/elle.py)):
over a list-append register a read of `[x, y, z]` is direct testimony that `x` preceded `y` preceded
`z`, so the per-key order is recovered with **zero** store cooperation, soundly reproducing the
store's exact versions. The value mode also surfaces anomalies the integer-version mode cannot even
represent, *before* any cycle search: **`incompatible-order`** (two reads fork on order — the
black-box split-brain signature), **`dirty-read`** (Adya G1a — a read of a value no committed
transaction installed; the **ED17** read-uncommitted recovery, §9.1), and **`duplicate-write`**.
Pure standard library, dependency-free, GPU-free.

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

