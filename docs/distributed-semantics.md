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
> convergence ops, and the Raft-subset consensus core (`elect`/`propose`/`step_down`, §3.3) have all since shipped
> as later DS increments. The embedded SPEC-6 host inside each node and the external real-binary DST
> runtime remain later increments (SPEC-7 §5, §13); each will extend this document.

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
| `step_down <node>` (DS0 increment 17) | **voluntary relinquishment**: clear `leader` (`→ None`) **at the same `term`** iff `node` is the current leader — the graceful counterpart to `elect`'s higher-term deposition, leaving the cluster leaderless until a fresh `elect`. Rejected `("not_leader", current_leader)` if `node` is not the leader (so a non-leader or a second `step_down` is a no-op reject — idempotently safe); `("unavailable", "")` if crashed. Needs **no quorum** (it reads only the node's own leadership), so a minority-stranded leader can relinquish where its `propose` is `no_quorum`. On success `("stepped_down", str(term))`. Touches no replica. |

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

### 3.3 `elect` / `propose` / `step_down` — the Raft-subset consensus core (DS0 increments 16–17)

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

