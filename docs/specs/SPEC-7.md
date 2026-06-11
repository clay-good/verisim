# SPEC-7 — The Distributed World

**Engineering & experiment specification for replicated services, transactions, and
consensus — the world that composes the others into a running *system*, where the
bit-exact global oracle becomes *provably intractable*, so faithfulness must be
verified at *tiered* cost, and where the deterministic-simulation-testing tradition
supplies both the oracle and the training data for free.**

This is to the *distributed system* what [SPEC-2](./SPEC-2.md) is to the filesystem,
[SPEC-5](./SPEC-5.md) is to the network, and [SPEC-6](./SPEC-6.md) is to the host.
SPEC-2 proved the *method* (propose-verify-correct against a free deterministic oracle)
on one filesystem; SPEC-5 lifted it to a multi-host network; SPEC-6 composed FS + net +
process into one running machine. SPEC-7 is the layer *above the host*: many machines
running **replicated services** that talk over the SPEC-5 network and execute on SPEC-6
hosts — databases, key-value stores, message queues, consensus groups, and the APIs in
front of them. It is the layer a computer-use or cyber-defense agent actually reasons
about ("will this config push break the cluster?"), and it is the first world where the
program's central asset — *a free, bit-exact, full-state oracle* — **stops being free**,
which is exactly why it is the world that proves the thesis matters.

It does not restate the science. Read [SPEC.md](./SPEC.md) first for *why* oracle-grounded
world models of computer environments are the one domain where long-horizon faithfulness
is measurable. This document is *how* — for a distributed system.

> **Reading order.** Prereqs: [SPEC.md](./SPEC.md) (science), [SPEC-2.md](./SPEC-2.md)
> (filesystem v0), [SPEC-5.md](./SPEC-5.md) (network world), [SPEC-6.md](./SPEC-6.md)
> (host world). Companions: [SPEC-3.md](./SPEC-3.md) (the depth roadmap whose walls
> **W1/W3/W4** and the speculative-execution framing this spec extends) and
> [SPEC-4.md](./SPEC-4.md) (the autonomous research engine that builds this with the
> human at the boundary).

> **▶ ACTIVE — DS0–DS6 + the equal-dollar-budget ED2 shipped (2026-06).** The deterministic core
> is up and the prime directive is measured: the **replicated KV under partition** slice
> ([`dist/`](../../src/verisim/dist/), [`distoracle/`](../../src/verisim/distoracle/),
> [`docs/distributed-semantics.md`](../distributed-semantics.md)) ships dependency-free and GPU-free
> with the `apply == oracle` invariant and golden trajectories (DS0–DS3); the **tiered
> propose-verify-correct loop** (DS5, [`distloop/`](../../src/verisim/distloop/)) and the flat
> learned `M_θ` (DS4) drop into it; and the distributed **`H_ε(ρ)` curve + the tiered-oracle H17
> measurement** is plotted on both the synthetic (ED1) and the real-model (ED1-learned) error
> distributions (DS6), with the **equal-*dollar*-budget H17/H18 frontier** (ED2) and the
> **fault-injection H21 data-factory result** (ED4) now in (DS7), and the **DS8 ED5 consistency-vs-bit
> horizon (H19) + competitive-ratio fit (H18)** and **ED6 counterfactual lift (H5)** in. The distributed knee + tiered
> oracle (**H17**) — the gate SPEC-7 was built around — reads *mode-dependent*: cheap tiers win per
> dollar exactly when the model's errors are cheaply catchable. ED5 sharpens the same throughline:
> consistency-faithful horizon *outlasts* bit-faithful where the error hides in the
> consistency-invisible in-flight medium (H19), and the loop is learning-augmented in the error axis
> but floor→cliff in the budget axis (H18). The **`linearizable` consistency model + the ED4
> consistency-level sweep (H20)** then show that the H19 gap is *exclusively* a weak-consistency
> phenomenon — it tracks the in-flight medium, present under `eventual` and structurally absent under
> `linearizable`. **ED6 then finds the distributed world is where counterfactual replay finally pays
> (H5):** off-policy oracle fault-branch training beats equal-volume on-policy data on held-out
> interventions (intervention-exact 0.51 vs 0.25), the honest inverse of the network/host supervision
> null — because the medium is the one hidden state on-policy volume cannot reach. **ED6's two-oracle
> slice (H12)** then closes the loop: the cheap consistency oracle is *redundant* for verification
> (non-redundant rate 0) but **decision-sufficient** for the split-brain question (1.00 on in-flight
> errors, 0.00 on durable — the in-flight medium again) at ~3.6× lower consult cost. Canonical build
> order: [SPEC §12](./SPEC.md#12-research-roadmap).

**License:** MIT. **Status:** DS0–DS8 shipped (DS0–DS6 core + the DS7 ED2/ED3/ED4 H21/H20 sweeps + the DS8 ED5 H18/H19 + ED6 H5 + ED6 two-oracle H12 + the learned-`M_θ` H12 re-pointing + the §16 verified-contribution protocol + the §7 LLM-callable simulator + the `verifiers`-spec RL env + the Inspect benchmark + the DS8 technical report + **the §5.2 Tier-B system oracle (ED7, the distributed W1 retirement)** + **the DS0-increment-2 multi-key transaction core (OCC first-committer-wins; ED8 commit/abort frontier)** + **the DS0-increment-3 transaction isolation levels (`serializable`/`snapshot`; ED9 the write-skew anomaly + the price of serializability)** + **the DS3-increment-2 Elle-style serializability checker (ED10 — the write-skew anomaly recovered black-box from the observable history as a `G2` anti-dependency cycle, certifying the `serializable` level reference-free)** + **the DS3-increment-3 version oracle (ED11 — Elle's list-append value recovery: the per-key version order recovered from the read values alone, sound against the store's exact versions, and the split-brain `incompatible-order` fork the integer-version mode cannot represent)** + **the DS0-increment-9 `read_committed` isolation level (ED16 — the classic lost-update anomaly, rate 1.0 under `read_committed` vs 0.0 under `snapshot`/`serializable`, the real-world default that sells correctness for never aborting; recovered black-box by Elle as a `{ww, rw}` G2 cycle)** + **the DS0-increment-10 `read_uncommitted` isolation level (ED17 — the classic dirty-read anomaly, rate 1.0 under `read_uncommitted` vs 0.0 under the three stronger levels, the weakest rung of the SQL hierarchy where a read observes another txn's uncommitted write; recovered black-box by Elle as a `dirty-read` recovery anomaly)** + **the DS0-increment-11 `drop` fault (ED18 — message loss, the unreliable-network `BUGGIFY` primitive the `MsgDrop` delta already anticipated: `drop src dst` destroys in-flight replication so the peer permanently misses the write — the convergence guarantee broken where `partition` recovers, post-heal convergence rate 0.0 under `drop` vs 1.0 under `partition`, repaired only by a newer overwriting write)** + **the DS0-increment-12 `anti_entropy` protocol op (ED19 — read-repair, the §4 `ReplicaConverge` the spec named: a node pulls each object to the latest reachable replica, so it *restores* the convergence `drop` broke without a fresh write (rate 1.0 vs 0.0 advance-only), bounded only by reachability — the Dynamo/Cassandra mechanism that makes eventual consistency eventual under an unreliable network)** + **the DS0-increment-13 `delay`/`reorder` message-timing faults (ED20 — the message-timing half of the §3.4 medium: `delay` defers an in-flight message (a *recoverable* delay, convergence rate 1.0 where `drop` is 0.0 — completing ED18's two-media contrast), `reorder` reverses a channel's delivery schedule (the in-transit observation flips at rate 1.0 while last-writer-wins keeps the converged value invariant at rate 1.0 — delivery-order independence made a controllable input); both via a new `MsgReschedule` edit over the existing `deliver_after`, no new state, Tier-A ≡ Tier-B bit-for-bit through the shared `timing_fault_edits` helper)** + **the DS0-increment-14 `clock_skew` fault (ED21 — the *last* of the §3.4 medium faults, the fault grammar now complete: a signed per-node clock offset shifts the `deliver_after` a node stamps on its sends, a persistent timing shift, yet convergence is *clock-independent* — sweeping skew leaves the converged state byte-identical at invariance rate 1.0, because LWW is by `(version, value)` not by timestamp, the property DST injects skew to verify; one omitted-when-empty `skew` map, no per-message state, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-15 `gossip` protocol op (ED22 — pairwise bidirectional anti-entropy, the §4 `ReplicaConverge`: `gossip a b` reconciles *both* nodes to the per-object winner, vs `anti_entropy`'s one-directional pull-to-one-node; one pairwise gossip fills complementary holes in both endpoints, and a chain of pairwise gossips converges the whole reachable component epidemically — bounded by reachability; reuses `ReplicaWrite`, no new state, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-16 `elect`/`propose` consensus core (ED23 — the Raft-subset third action family, the `ProtocolStep`/`ProtocolState` the spec named since increment 1: `elect node` makes a node leader iff its partition side holds a strict majority of the *live* cluster — so two sides can never both elect (**no split-brain**, and an even split is leaderless rather than forked) — bumping a monotone `term`; `propose node key val` is a **leader-fenced** majority write that commits only for the current leader, so a leader deposed by a higher-term election cannot commit **even after the partition heals** — the Raft **leader-completeness** safety property a leaderless `quorum` write lacks (a plain `put` by the same stale coordinator still commits — the control); two omitted-when-default state fields (`term`/`leader`), one `ProtocolStep` edit, shared `elect_edits`/`propose_edits`, Tier-A ≡ Tier-B bit-for-bit, and metamorphic term-monotone/known-leader invariants)** + **the DS0-increment-17 `step_down` op (ED24 — voluntary leadership relinquishment, the leadership lifecycle's graceful close: `step_down node` lets the *current* leader hand back power on its own, leaving the cluster **leaderless at the same term** (the voluntary counterpart to ED23's higher-term deposition), so the same node's next `propose` is `not_leader` — **no leaderless commit window** — and a clean handoff is `step_down` then `elect <successor>` at a strictly higher term; only the current leader may relinquish (a non-leader/second `step_down` is a no-op reject — idempotently safe), and a **minority-stranded leader can still step down** where its `propose` is `no_quorum` — relinquishing needs no quorum, exercising power does; reuses the `ProtocolStep` edit with `leader → None`, no new state, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-18 `lease`/`lread` leader-lease (ED25 — the Raft read optimization): `lease node dt` lets the current leader take a read lease through global clock `+ dt`, and `lread node key` then serves a **local linearizable read with no quorum round-trip** while it holds — so a leader partitioned into the minority can still `lread` locally where its `propose` is `no_quorum` (the read-availability the lease buys); the safety tension is that a fresh `elect` is fenced `lease_held` until the incumbent's lease expires (leadership cannot change hands under a live lease — what makes the local read safe), while a voluntary `step_down` **releases** the lease for an immediate handoff (vs a crashed leader the cluster must outlast); one omitted-when-default `lease_until` field + one `LeaseSet` edit, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-19 `append` replicated log (ED26 — the Raft log the spec named since increment 1, what the one-shot `propose` elided): `append node key val` appends a `(term, index, key, value)` entry to the leader's log and replicates it to the reachable followers (who adopt the leader's prefix, **overwriting any divergent uncommitted tail** — the log-matching reconciliation), committing it (and folding the committed prefix into the KV state machine, backfilling a rejoined follower) **iff a majority holds it**; a minority-stranded leader still appends locally (`uncommitted`) but does not commit, so its entry is never applied to the KV and is **overwritten by a higher-term leader's entry at the same index** — the log-matching safety `propose` could not express; per-node `logs` + a monotone `commit_index` (both omitted until the first `append`) and a `LogSet`/`CommitIndexSet` edit pair, with a metamorphic commit-index-monotone invariant, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-20 `add_replica`/`remove_replica` membership change (ED27 — the §3.2 admin ops): they reconfigure the consensus *voting set* (a leader-committed change), so the **majority threshold tracks the membership** — `remove_replica` shrinks the cluster (a smaller majority suffices, the standard way to restore availability after nodes fail: a lone survivor of a 3-node cluster commits again once the 2 dead are removed) and `add_replica` grows it; the active leader cannot be removed (`is_leader` fence) and the last member is protected; one omitted-when-default `members` voting set (empty = the "all vote" sentinel) + one `MemberSet` edit, with a metamorphic members-subset invariant, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-21 `enqueue`/`dequeue` distributed FIFO queue (ED28 — the §3.2 client ops, a *second data type* beside the KV store): a queue's delivery guarantee follows the `consistency_model` — under `eventual` a dequeue's head-removal reaches only the reachable side, so a partitioned peer re-delivers the same item (**at-least-once / duplicate**), where `quorum` (majority) / `linearizable` (all replicas) gate availability for **exactly-once** (the delivery count steps `2 → 1 → 0` across eventual/quorum/linearizable under a partition — the KV CAP tradeoff in delivery-semantics form), while the connected happy path is a correct FIFO; one omitted-when-empty `queues` map + one `QueueSet` edit, with queues now part of the observable `cluster_view`, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-22 `deploy` rolling-upgrade op (ED29 — the §3.2 admin op answering SPEC-7's headline *"will this deploy break the cluster?"*): `deploy node version` sets a node's running software version, and two nodes share a consensus quorum only if their versions are within `max_version_skew` (the N-1 window, default 1) — so a rolling upgrade that stays inside the window keeps quorum, but a deploy that creates an **incompatible version split with no compatible majority** turns the next `elect`/`propose` into `no_quorum` (the deploy broke the cluster); one omitted-when-default `versions` map + a `max_version_skew` config dial + one `VersionSet` edit, versions in the observable `cluster_view`, compatibility gating *consensus* only, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-23 `host` embedded-host op (ED30 — the compositional vision §3.1/§4 names since increment 1): each cluster node runs a real **SPEC-6 host** (process table + per-process fd tables + an embedded v0 filesystem), and `host node <syscall>` delegates to the SPEC-6 `ReferenceHostOracle` on that node's own host — so a node is not just a bag of KV replicas but a process host; per-node **isolated** (a `fork` on one node never touches another's host), host ops respect the node's up/down status (a crashed node's host is `unavailable` — the cross-layer crash linkage) and the host state survives a crash; one omitted-when-empty per-node `hosts` map + one `HostStep` edit (wrapping the SPEC-6 bundle delta), hosts in the observable `cluster_view`, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-24 `config_push` config-management admin op (ED31 — SPEC-7's *other* headline operational question, *"will this config push break the cluster?"*, the sibling of ED29's `deploy`): unlike `deploy` (a node-local version *label* gating consensus *compatibility*), a `config_push node key val` is a **leader-committed, majority-replicated** cluster setting — a Raft-style config entry — so it shares the leader-fence + majority-reachability rule of `propose`/`append` (a non-leader push is `not_leader`, a minority-stranded leader is `no_quorum` and changes nothing); a push that commits under a partition reaches only the majority side, so the **partitioned minority retains its stale config** (config divergence, the broken-cluster outcome), repaired by a re-push after `heal` that converges every node; one omitted-when-empty per-(node, key) `config` map + one `ConfigSet` edit, config in the observable `cluster_view`, gating nothing in the data plane, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-25 `read_index` quorum-confirmed linearizable read (ED32 — the Raft *ReadIndex* method, the partner to the `lread` lease read): `read_index node key` confirms leadership with a **majority** before serving the read (no clock assumption), so a minority-stranded leader is `no_quorum` where a live-lease `lread` still serves locally (the two linearizable reads' opposite availability profiles), and a **deposed** leader is `not_leader` even after `heal` — refusing the stale read a plain `get` would serve; a pure read (no state field, no edit type, purely additive), Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-26 `delete` tombstone (ED33 — the fundamental KV remove the grammar lacked, and a canonical distributed hazard): `delete node key` is a **versioned write of a tombstone** (reusing the `put` replication path with the `TOMBSTONE` value), *not* a removal of the replica, so last-writer-wins orders the delete against concurrent/stale writes by version — which avoids the **resurrection problem** (a deleted key reappearing because a stale replica out-versions an absence): under `eventual` a partitioned minority still reads the deleted item, but after `heal` the tombstone's higher version wins the `anti_entropy`/`gossip` merge so the key converges to deleted (a *genuinely newer* `put` legitimately brings it back — a new write, not a resurrection); the tombstone is just a replica value (no state field, no edit type, purely additive), Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-27 `incr` atomic counter (ED34 — the first *read-modify-write* client op, and the canonical lost-update negative): `incr node key` reads the coordinator's local count (a non-numeric/absent value is `0`) and writes `count+1`, reusing the `put` path — sequentially correct, but under a partition the consistency model decides whether concurrent increments survive: **`eventual` silently loses a concurrent increment** (two acknowledged `incr`s, count short by one — LWW keeps one of two same-version writes), where `quorum` makes the minority `unavailable` (no silent loss) and `linearizable` rejects under any partition; *harder* than the blind-write CAP frontier (ED14) because LWW loses a read-modify-write where it merely makes a blind write stale (a loss-free eventual counter needs a CRDT — deferred); the counter is just a digit-valued replica (no state field, no edit type, purely additive), Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-28 `cincr`/`cget` CRDT G-counter (ED35 — the loss-free, always-available *resolution* to ED34's negative): a *state-based* grow-only counter where each node bumps **only its own** per-owner sub-count (`cincr` is purely node-local, so **always available** — a partitioned-alone node still counts, the AP property the LWW `incr` lacks) and the CRDT **join is the per-(key, owner) max** applied by `anti_entropy`/`gossip` (commutative/associative/idempotent); concurrent increments touch disjoint entries so there is **no lost update**, and the counter converges to the exact total — the three-increment partition that lost one under `incr` (ED34) now reads 3; one omitted-when-empty `gcounters` map + one `GCounterSet` edit, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-29 `cdecr` CRDT PN-counter (ED36 — the decrement that makes ED35's grow-only G-counter *decrementable*): a PN-counter pairs **two** G-counters, P (the `cincr` half) and N (the `cdecr` half), and `cget` reads **P − N**; `cdecr n key` is the exact twin of `cincr` over the N half (bumps only `n`'s own decrement sub-count), so it inherits every property — purely node-local (**always available**), single-writer-per-entry (concurrent decrements never conflict, no lost update), and merged by the same per-(key, owner) max join over *both* halves — while gaining the one the G-counter lacked: the value may go **negative** (the sub-counts stay monotone/non-negative, only their difference dips below zero; +2 majority − 1 minority converges to net 1 loss-free across a partition); one omitted-when-empty `ncounters` map + one `NCounterSet` edit, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-30 `sadd`/`srem`/`smembers` CRDT OR-Set (ED37 — the canonical *interesting* CRDT, a replicated set a naive implementation gets wrong): an element-level 2P-Set is **remove-wins** and can **never re-add**; the **observed-remove set** fixes both with a **unique dot** — `sadd n key elem` tags the element with a fresh `(owner=n, seq)` and stores it in `n`'s observed add-set, `srem` tombstones **only the dots `n` observed**, `smembers` is the elements with a non-tombstoned dot, and the join is **set union** of both halves (commutative/associative/idempotent); the fresh-dot identity buys **add-wins** (a concurrent add survives a concurrent remove — the unseen dot is never tombstoned) and **re-addability** (a removed element returns under a new dot), the two properties the 2P-Set lacks, with `sadd`/`srem` purely node-local (**always available**); two omitted-when-empty `orset_adds`/`orset_tombs` maps + the `ORSetAdd`/`ORSetTomb` edits, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-31 `mvput`/`mvget` CRDT MV-register (ED38 — the Dynamo/Riak register that *surfaces* a write conflict as **siblings** instead of silently dropping one): where the KV `put` and the counters resolve concurrent writes by last-writer-wins (one survives, one lost), the multi-value register keeps *both* and lets a later reader resolve them; it reuses the OR-Set's dot/union machinery — `mvput n key val` tags `val` with a fresh dot, **tombstones every dot it currently observes** (a write supersedes the values it saw), and adds its own, so a *sequential* overwrite collapses to one value while *concurrent* writes (neither observing the other) **both survive** as siblings (`mvget` reads `{a,b}` where a LWW `put` keeps one), and a later context-aware `mvput` (having seen both) **resolves** them (the Dynamo read-and-resolve); `mvput` is purely node-local (**always available**) and the join is **set union** of both halves; two omitted-when-empty `mvreg_vals`/`mvreg_tombs` maps + the `MVRegWrite`/`MVRegTomb` edits, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-32 `lwwput`/`lwwget` CRDT LWW-register (ED39 — the *policy-opposite* of the MV-register: where the MV-register surfaces a conflict as siblings, the LWW-register **deterministically picks one winner** by a **Lamport-timestamp total order**): `lwwput n key val` stamps `val` with `(ts, owner=n)` where `ts = lamport[n] + 1` (advancing `n`'s Lamport clock — a per-node logical counter that makes "happens-after" a comparable order without a shared real clock), and the join keeps the **max** copy by `(ts, owner, value)`, so a write that happened-after another (higher ts) wins regardless of node and truly concurrent (equal-ts) writes break the tie by node id (a single deterministic winner the whole cluster agrees on, where the MV-register keeps both as siblings — the concurrent loser is dropped); `lwwput` is purely node-local (**always available**); two omitted-when-empty `lwwreg`/`lamport` maps + the `LWWRegSet`/`LamportSet` edits, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-33 `mput`/`mget`/`mdel`/`mkeys` CRDT OR-Map (ED40 — the *capstone*, a CRDT **of** CRDTs): it composes the OR-Set (governing **field presence**, add-wins + observed-remove over field names) with the LWW-register (governing each field's **value**) — the in-CRDT-layer instance of the whole program's compositional thesis; `mput n map field val` adds a fresh presence dot for `field` *and* LWW-writes `val`, `mdel` observed-removes the field, `mget` reads a present field's value, `mkeys` enumerates the present fields (the map capability the flat KV/registers lack), and the join is the **OR-Set union** of the presence halves plus the **LWW max** of each field's value — so a concurrent `mput` survives a concurrent `mdel` (add-wins field presence) while a field's value resolves by last-writer-wins, the two halves converging independently; `mput`/`mdel` are purely node-local (**always available**), share the Lamport clock (now max-applied so the LWW + OR-Map merges compose order-independently); three omitted-when-empty `ormap_fields`/`ormap_tombs`/`ormap_vals` maps + the `ORMapField`/`ORMapTomb`/`ORMapVal` edits, Tier-A ≡ Tier-B bit-for-bit)** + **the DS0-increment-34 `rins`/`rdel`/`rget` CRDT RGA (ED41 — the first *ordered* CRDT, a sequence, the basis of collaborative text): where every prior CRDT is unordered, the RGA maintains a list in which any node inserts at any position and concurrent inserts converge to **one** deterministic order with no duplication; each element carries a unique id `(seq, owner)` and a `parent` id (the element it was inserted after, or `ROOT` for the head), and the visible order is a DFS with siblings ordered by id descending — so the **order is a pure function of the element set**, making the **set-union** join (elements + tombstones) converge every node to the same sequence; `rins` inserts after the i-th visible element, `rdel` tombstones it (delete preserves structure as an anchor), `rget` reads the visible values concatenated, all purely node-local (**always available**); two omitted-when-empty `rga_elems`/`rga_tombs` maps + the `RGAInsert`/`RGATomb` edits, Tier-A ≡ Tier-B bit-for-bit)**); the distributed world is complete through DS8, and the deterministic core now carries transactions with the **four standard SQL isolation levels** (`serializable`/`snapshot`/`read_committed`/`read_uncommitted`), two concurrency-control mechanisms (OCC/2PL), four consistency models (`eventual`/`causal`/`quorum`/`linearizable`), the **complete §3.4 unreliable-network fault grammar** (`partition`/`crash`/`drop`/`delay`/`reorder`/`clock_skew`), the `anti_entropy` read-repair + pairwise `gossip` convergence ops, the **Raft-subset `elect`/`propose`/`step_down`/`lease`/`lread`/`read_index`/`append`/`add_replica`/`remove_replica` consensus core** (election, leader-fenced writes, the voluntary-handoff lifecycle, the two linearizable reads — the lease-based local read and the quorum-confirmed ReadIndex, the replicated commit-index log with log-matching reconciliation, and quorum-tracking membership change), the **rolling-upgrade `deploy` and leader-committed `config_push` admin ops** (the two "will this … break the cluster?" change-safety questions), the **embedded SPEC-6 `host`** on every node (the compositional vision), plus a black-box Elle consistency checker with value-recoverable histories. **Audience:** the author,
collaborators, reviewers, and the autonomous research engine (SPEC-4), which reads this
file as part of its operating context.

---

## 0. Prime directive

> **SPEC-7's only job is to plot the distributed `H_ε(ρ)` curve once, cleanly, in a world
> where the full-state oracle is *intractable* — and to show whether a *tiered* oracle
> (cheap consistency checks plus rare bit-exact replay) buys more faithful horizon per
> oracle-dollar than spending the same budget on full-state truth alone (H17). It reports
> honestly if it does not.**

> **▶ First result (ED1 apparatus, DS6 — [`ed1_dist.png`](../../figures/ed1_dist.png)).** The
> distributed `H_ε(ρ)` curve is the same **floor→cliff** as every prior world (floor 0.2 at ρ=0 →
> ceiling 40 at ρ=1). And the H17 verdict, on a *controlled* error distribution (a synthetic
> tunable-noise proposer, the apparatus before the learned `M_θ`): **whether a cheap tier wins is
> not unconditional — it depends on where the model's errors fall.** For **gross** errors the cheap
> metamorphic tier buys faithful horizon at **$9.4/step vs bit-exact's $16**; for **subtle** errors
> the cheap tiers miss the drift (H≈0, **$848/step**) and full bit-exact truth is the only efficient
> verifier. So the tiered oracle is a *real lever, conditionally* — a sharper, more honest answer
> than "cheap always wins," and exactly the measurement the oracle-free distributed-systems field
> cannot make. The learned model (DS4) will supply the real error distribution this apparatus awaits.

> **▶ ED2 — the equal-*dollar*-budget form of H17 (DS7 — [`ed2.png`](../../figures/ed2.png)).** ED1
> measured cost-per-faithful-step at full consultation; ED2 asks the budget-form question the
> hypothesis is really about: *at an equal oracle-dollar budget, does a cheap or cheapest-refutation
> (`escalate`) tier policy buy more faithful horizon than spending the same dollars on bit-exact
> truth?* It sweeps ρ and plots, per tier policy, the **faithful-horizon-vs-oracle-dollar frontier**
> (the Pareto front), comparing policies at a matched budget by interpolating each one's horizon
> along its envelope — a true equal-dollar comparison, not an equal-ρ one. At the **sub-linear
> quarter budget** `B/4` (¼ of always-bit-exact's full-truth cost): for **gross** errors the
> metamorphic tier reaches **H=14.2 vs bit-exact's 4.2** (tiering wins per dollar — H17 holds, with
> a competitive ratio of **0.36** of the full-truth ceiling at ¼ the cost, the H18 readout); for
> **subtle** errors the cheap tiers are flat at the floor (H=1.5) and only bit-exact climbs (4.2),
> so `escalate` *loses* to single-tier bit-exact (H17's honest negative, ratio 0.11). The
> equal-budget frontier confirms ED1's mode-dependent verdict in the form the spec poses it.

> **▶ ED23 — leader election with terms: the consensus family (DS0 increment 16 — [`ed23.png`](../../figures/ed23.png)).**
> The third action family ships: `elect`/`propose`, the Raft-subset consensus core. It adds the one
> safety property a leaderless `quorum` write cannot give — a single, *fenced* writer — and the oracle
> verifies it bit-exact. **Panel A (no split-brain):** across a cluster-size sweep, only a strict-majority
> partition side can `elect` (minority blocked, rate **1.0**; majority elects, **1.0**), and an even
> `2 | 2` split is **leaderless rather than forked** (neither side elects, **1.0**) — two leaders are
> structurally impossible, not merely improbable. **Panel B (term-fencing / leader-completeness):** a
> leader deposed by a higher-term election on the majority side is rejected (`not_leader`) **even after
> the partition heals** (fenced, **1.0**), whereas a plain `put` by that same stale coordinator *still
> commits* (the control, **1.0**) — the stale write the fence exists to stop. Tier-A ≡ Tier-B over every
> transition: `elect`/`propose` are coordinator-level decisions, so the autonomous-actor system oracle
> reproduces the fencing byte-for-byte. The metadata is omitted from the canonical form until the first
> election, so the family is purely additive (no prior golden/hash/tokenization changes).

> **▶ ED24 — voluntary step-down: the graceful handoff (DS0 increment 17 — [`ed24.png`](../../figures/ed24.png)).**
> The consensus family's leadership lifecycle closes: `step_down` lets the *current* leader hand back
> power on its own, leaving the cluster **leaderless at the same term** — the voluntary counterpart to
> ED23's *involuntary*, higher-term deposition. **Panel A (the handoff lifecycle):** across a cluster-
> size sweep, after `step_down` the same node's `propose` is rejected (`not_leader`, rate **1.0**) — the
> term machinery admits **no leaderless commit window** — and a fresh `elect` of a successor lands at a
> strictly higher term (**1.0**) and commits (**1.0**); a clean handoff is exactly `step_down` then
> `elect <successor>`. **Panel B (authority + partition-independence):** only the current leader may step
> down (a non-leader is rejected, **1.0**; a second `step_down` is a no-op reject, **1.0** — idempotently
> safe), and the sharp case — a leader **stranded in a minority can still step down** (**1.0**) where its
> `propose` there is `no_quorum` (the control, **1.0**). Relinquishing power reads only the node's own
> leadership, never the medium, so it is always safe; *exercising* it needs a quorum. Tier-A ≡ Tier-B
> over every transition; the op touches no replica, so it is purely additive (no prior golden/hash change).

> **▶ ED25 — leader leases: local reads without a quorum (DS0 increment 18 — [`ed25.png`](../../figures/ed25.png)).**
> The Raft **leader lease** — a read optimization on the `elect`/`propose`/`step_down` core. `lease node dt`
> lets the *current* leader take a read lease through global clock `+ dt`; `lread node key` then serves a
> **local linearizable read with no quorum round-trip** while it holds. **Panel A (local reads without a
> quorum):** across a cluster-size sweep, a live lease serves `lread` (rate **1.0**), and the sharp case — a
> leader **partitioned into the minority can still `lread`** locally (**1.0**) where its `propose` there is
> `no_quorum` (the control, **1.0**) — the read-availability the lease buys; once the clock passes the
> deadline the same read is `lease_expired` (**1.0**). **Panel B (the lease/election safety tension):** a fresh
> `elect` is fenced `lease_held` while the incumbent's lease is live (a successor must **wait out the lease** —
> **1.0**) and unblocked past expiry (**1.0**) — so leadership cannot change hands under a live lease, which is
> what makes the local read safe; and a voluntary `step_down` **releases the lease immediately**, so a graceful
> handoff elects a successor with no wait (**1.0**) where a *crashed* leader forces the cluster to outlast it.
> Lease/lread touch no replica (coordinator-level metadata like `term`/`leader`), so Tier-A ≡ Tier-B bit-for-
> bit; the deadline is omitted from the canonical form until the first `lease`, so the op family is additive.

> **▶ ED26 — Raft log replication: commit-on-majority + log-matching reconciliation (DS0 increment 19 — [`ed26.png`](../../figures/ed26.png)).**
> The replicated **log** the spec named since increment 1 — what the one-shot `propose` (incr 16) elided.
> `append node key val` appends a `(term, index, key, value)` entry to the leader's log, replicates it to
> the reachable followers (who adopt the leader's prefix, overwriting any divergent uncommitted tail), and
> commits it — folding it into the KV state machine — **iff a majority holds it**. **Panel A (commit
> requires a majority):** across a cluster-size sweep, a majority-reachable `append` commits (the monotone
> `commit_index` grows, **1.0**), while a minority-stranded leader's `append` stays **uncommitted**
> (commit_index unchanged, **1.0**) yet is **retained on its log** (**1.0**) — not lost, just not durable;
> the commit index never moves backward (**1.0**). **Panel B (log-matching reconciliation):** the safety
> the one-shot `propose` lacked — while uncommitted, the stale entry is **never applied to the KV** (**1.0**);
> after a higher-term leader commits a conflicting entry at the same index and the partition heals, the
> deposed leader's uncommitted entry is **overwritten** (**1.0**), all live nodes hold an **identical log**
> (the log-matching property, **1.0**), and the rejoined node's KV converges (**1.0**). `append` reads the
> majority from the medium (coordinator-level, like `propose`), so Tier-A ≡ Tier-B bit-for-bit; the log +
> commit index are omitted from the canonical form until the first `append`, so the op is purely additive.

> **▶ ED27 — membership change: the quorum threshold tracks the voting set (DS0 increment 20 — [`ed27.png`](../../figures/ed27.png)).**
> The `add_replica`/`remove_replica` admin ops the §3.2 grammar named. They reconfigure the *consensus
> voting membership* (the nodes that count toward an election/commit quorum), a leader-committed change,
> so the **majority threshold follows the membership**. **Panel A (the threshold tracks the votes):**
> across a cluster-size sweep, a leader partitioned **alone** is a minority of the full cluster, so its
> `propose` is `no_quorum` (**1.0**); `remove_replica` the unreachable nodes until it is the sole member
> and the same lone leader commits (a majority of 1, **1.0**) — availability with no change in
> reachability, purely from shrinking the voting set; `add_replica` a node back raises the threshold and
> re-blocks it (**1.0**). **Panel B (restore availability after failure):** a 3-node cluster loses 2 to
> crashes; the lone survivor is stuck (`no_quorum` at majority-2-of-3, **1.0**); `remove_replica` the two
> dead nodes and it commits again (majority-1-of-1, **1.0**) — the standard operator lever. The change is
> fenced: the **active leader cannot be removed** (`is_leader`, **1.0** — step it down first), so a
> reconfiguration never strands the cluster leaderless mid-write. Membership is coordinator-level cluster
> metadata, so Tier-A ≡ Tier-B bit-for-bit; the voting set is omitted from the canonical form until the
> first change (empty = the "all nodes vote" sentinel), so the ops are purely additive.

> **▶ ED28 — the distributed FIFO queue: delivery semantics follow the consistency model (DS0 increment 21 — [`ed28.png`](../../figures/ed28.png)).**
> The §3.2 `enqueue`/`dequeue` client ops — a **second data type** beside the KV store. The headline:
> a queue's delivery guarantee is not a property of the queue but of the consistency model it runs
> under. **Panel A (delivery under partition):** one item is enqueued under full connectivity, the
> cluster partitions, and each side dequeues — `eventual` delivers it **twice** (at-least-once /
> duplicate, the head-removal never crosses the split), `quorum` **once** (exactly-once on the
> majority side, the minority `unavailable`), `linearizable` **zero** times (both sides lack all-
> replica reachability, so both are `unavailable`). Delivery count `2 → 1 → 0` as the model
> strengthens — the KV fork-vs-availability tradeoff (ED14) in delivery-semantics form. **Panel B
> (the connected happy path):** with full connectivity, `enqueue` of `a, b, c` then three `dequeue`s
> returns `a, b, c` in order (FIFO, **1.0**), each exactly once, then `empty` (**1.0**) — a correct
> FIFO queue when the network is whole, under every model. Queues are fully replicated and the
> reachable set / availability are read from the medium, so Tier-A ≡ Tier-B bit-for-bit (the
> duplicate delivery is reproduced on the autonomous actors too); queue replicas are omitted from the
> canonical form until the first `enqueue`, so the op family is purely additive.

> **▶ ED29 — the rolling upgrade: will this deploy break the cluster? (DS0 increment 22 — [`ed29.png`](../../figures/ed29.png)).**
> The `deploy` admin op answers the question SPEC-7 names in its introduction (§1: *"will this config
> push break the cluster?"*). `deploy node version` sets a node's running software version, and two
> nodes participate in the same consensus quorum only if their versions are within
> `max_version_skew` (the default `1` is the standard N-1 rolling-upgrade window). **Panel A (safe
> rolling upgrade):** across a cluster-size sweep, rolling every node `v0 → v1` one at a time, a
> `propose` commits after each bump (rate **1.0** over all steps) — the version spread never leaves
> the window, so a compatible majority always exists. **Panel B (the deploy that breaks the
> cluster):** an incompatible split with **no compatible majority** (2 at `v0`, 2 at `v2`, spread
> `2 > 1`) turns the next `propose` into `no_quorum` (**1.0**) — the deploy broke the cluster. The
> diagnostic isolates the cause: the *same* shape is safe at a smaller spread (`v0`/`v1`, **1.0**) or
> under a wider configured window (`skew 2`, **1.0**) — it is the spread *exceeding the window* that
> breaks consensus, not mixed versions per se. Compatibility gates *consensus* only (the KV/queue
> data plane is version-agnostic); versions are observable cluster metadata, so Tier-A ≡ Tier-B
> bit-for-bit, and the version map is omitted from the canonical form until the first `deploy`.

> **▶ ED30 — the embedded host: each cluster node runs a real SPEC-6 host (DS0 increment 23 — [`ed30.png`](../../figures/ed30.png)).**
> The compositional vision SPEC-7 names since increment 1 (§3.1/§4: a `HostDelta` on an embedded
> subsystem). A cluster node is no longer just a bag of KV replicas — it runs a real **SPEC-6 host** (a
> process table + per-process fd tables + an embedded v0 filesystem), and `host node <syscall>`
> delegates to the SPEC-6 `ReferenceHostOracle` on that node's own host. **Panel A (composition +
> isolation):** a `fork` runs on *that node's host only* (per-node isolation, rate **1.0**); a node
> serves a KV `put` *and* a host `fork` independently (the two subsystems coexist on one node, **1.0**);
> and `open` + `write` materializes the file in **that node's embedded v0 filesystem** — the composition
> runs all the way down to the v0 FS sub-oracle (**1.0**). **Panel B (the cross-layer crash linkage):** a
> `host` syscall on a **crashed** node is `unavailable` (**1.0**) — the same up/down gate the KV client
> ops obey, now reaching the host; `restart` resumes host ops (**1.0**), and the host **state survives
> the crash** (a process forked before the crash persists, and pids keep counting — **1.0**): a crash
> pauses the node, it does not wipe it. The host effect delegates to the SPEC-6 oracle (a node-local
> computation), so Tier-A ≡ Tier-B bit-for-bit; the embedded hosts join the observable `cluster_view`
> and the `hosts` map is omitted from the canonical form until the first `host` op (purely additive).

> **▶ ED31 — the config push: "will this config push break the cluster?" (DS0 increment 24 — [`ed31.png`](../../figures/ed31.png)).**
> The config-management admin op §3.2 names, the sibling of ED29's `deploy` and the answer to SPEC-7's
> *other* headline operational question. Unlike `deploy` (a node-local version *label* that gates
> consensus *compatibility*), a `config_push node key val` is a **leader-committed, majority-replicated**
> cluster setting — a Raft-style config entry — so it shares the leader-fence + majority-reachability
> rule of `propose`/`append`. **Panel A (leader-committed rollout + the leader fence):** a push at the
> elected leader with full connectivity **commits and reaches every voting member** (rate **1.0**); a
> push by a **non-leader** is `not_leader` (**1.0**) and one with **no leader** is rejected (**1.0**) —
> config changes go through consensus, not any node that asks. **Panel B (the partition):** a leader
> **stranded in the minority** gets `no_quorum` — the push **cannot commit and no node's config changes**
> (the all-or-nothing rule, **1.0**); a leader on the **majority** side **commits**, but the value
> reaches only the reachable majority, so the **partitioned minority retains its stale config** —
> *config divergence*, the broken-cluster outcome (**1.0**), repaired by a **re-push after `heal`** that
> converges every node (**1.0**). The commit quorum is read from the partition/down medium (a
> coordinator-level decision), so Tier-A ≡ Tier-B bit-for-bit; the `config` map joins the observable
> `cluster_view` and is omitted from the canonical form until the first push (purely additive).

> **▶ ED32 — the quorum-confirmed linearizable read: Raft ReadIndex, the partner to the lease read (DS0 increment 25 — [`ed32.png`](../../figures/ed32.png)).**
> `read_index` is the *other* way Raft serves a linearizable read — the partner to the lease read
> `lread` (ED25/incr 18). Where `lread` skips the quorum round-trip by relying on a time **lease**,
> `read_index` keeps no clock assumption and instead **confirms leadership with a majority** before
> serving the read (the ReadIndex heartbeat round). **Panel A (the two reads, opposite availability):**
> across a cluster-size sweep, a `read_index` at the leader with full connectivity serves the read
> (**1.0**); a **non-leader** is `not_leader` (**1.0**); a leader **stranded in a minority** is
> `no_quorum` (**1.0**) — it cannot confirm it is still leader, so it refuses. The sharp contrast: that
> same minority leader holding a **live lease** *can* serve `lread` locally (**1.0**) where its
> `read_index` is `no_quorum` — the read-availability the lease buys and the quorum read declines (and
> the clock dependence the quorum read avoids in return). **Panel B (linearizable safety + freshness):**
> a `read_index` reflects the **latest committed value** after an `append` (**1.0**); a leader
> **deposed** by a higher-term election (partitioned away while the majority committed a newer value) is
> `not_leader` on `read_index` **even after `heal`** (**1.0**) — *refusing* its now-stale local replica,
> where a plain `get` from that node **serves the stale value** (the read `read_index` exists to
> prevent), and the new leader's `read_index` returns the fresh committed value (**1.0**). A pure read
> (touches no replica), majority read from the medium, so Tier-A ≡ Tier-B bit-for-bit; it adds no state
> field and no edit type (purely additive).

> **▶ ED33 — the tombstone delete: versioned removal and the resurrection problem (DS0 increment 26 — [`ed33.png`](../../figures/ed33.png)).**
> The fundamental KV **remove** the grammar lacked, and a canonical distributed hazard. A `delete node
> key` is a **versioned write of a tombstone** (it reuses the `put` replication path with the
> `TOMBSTONE` value), *not* a removal of the replica: the deleted key keeps a replica at a bumped
> version, so last-writer-wins orders the delete against concurrent and stale writes by version — which
> is exactly what avoids the **resurrection problem** (a deleted key reappearing because a stale
> replica's old value out-versions an absence). **Panel A (the versioned tombstone):** across a
> cluster-size sweep (linearizable), `put` then `delete` leaves every replica reading `deleted`
> (**1.0**); the tombstone **out-versions the put it deleted** (**1.0**); and a *genuinely newer* `put`
> (a higher version than the tombstone) **legitimately brings the key back** (**1.0**) — a new write,
> not a resurrection. **Panel B (resurrection under partition + repair):** a `delete` on the majority
> side leaves the partitioned **minority still reading the old value** (the deleted item is "still
> there" — the danger, **1.0**); after `heal`, the tombstone's higher version **wins the merge**, so
> `anti_entropy` (**1.0**) and pairwise `gossip` (**1.0**) converge the minority to `deleted` rather
> than resurrecting it — the bug a naive removal would cause. A `get` on a tombstoned replica reports
> `deleted`. The op reuses the `put` write path, so Tier-A ≡ Tier-B bit-for-bit; the tombstone is just
> a replica value (no state field, no edit type — purely additive).

> **▶ ED34 — the atomic counter: read-modify-write and the lost-update problem (DS0 increment 27 — [`ed34.png`](../../figures/ed34.png)).**
> The first *read-modify-write* client op (`put`/`cas`/`delete` are blind or compare writes), and the
> canonical case where eventual-consistency last-writer-wins **silently loses updates**. An `incr node
> key` reads the coordinator's local counter (a non-numeric/absent value is `0`) and writes `count + 1`
> at a bumped version, reusing the `put` replication path. **Panel A (sequential correctness):** across
> a cluster-size sweep, `incr` applied `k` times counts to exactly `k` (**1.0**), and the same sequence
> is correct under **all three** consistency models (with no concurrency, every model counts right).
> **Panel B (the read-modify-write CAP tradeoff):** two `incr`s on opposite sides of a partition —
> under **`eventual`** both are *acknowledged* yet the count ends up **short by one** (a *lost update*,
> the danger, **1.0**); under **`quorum`** the minority side is **`unavailable`** so only the accepted
> increment counts (no silent loss, **1.0**); under **`linearizable`** an `incr` under any partition is
> **`unavailable`** (CP, **1.0**). This is *harder* than the blind-write CAP frontier (ED14): LWW
> **loses** a read-modify-write update where it merely makes a blind write **stale** — the textbook
> "you can't build a correct counter on last-writer-wins," banked as a first-class negative (a
> loss-free eventual counter needs a CRDT, a deferred later increment). `incr` reuses the `put` path,
> so Tier-A ≡ Tier-B bit-for-bit; the counter is just a digit-valued replica (no state field, no edit
> type — purely additive).

> **▶ ED35 — the CRDT G-counter: the loss-free, always-available resolution to ED34 (DS0 increment 28 — [`ed35.png`](../../figures/ed35.png)).**
> The **positive that resolves ED34's negative**. A *state-based* CRDT counter: each node keeps a
> per-owner vector of monotone sub-counts, `cincr n key` bumps **only `n`'s own** sub-count (`cget`
> reads the sum), and the CRDT **join is the per-(key, owner) max** applied by `anti_entropy`/`gossip`
> — commutative, associative, idempotent. **Panel A (loss-free + always available):** across a
> cluster-size sweep `cincr` `k` times reads back `k` (**1.0**); the direct contrast with ED34 — under
> a partition **three** `cincr`s (two majority, one on the partitioned minority) are *all*
> acknowledged, including the minority one a LWW `quorum`/`linearizable` `incr` would reject
> (**always available, 1.0**), and after `heal`+`gossip` the counter reads exactly **3** (**no lost
> update, 1.0**) where ED34's LWW counter read 2. **Panel B (convergence):** the join converges
> **every** node to the full total — a `gossip` chain spreads it epidemically (**1.0**), `anti_entropy`
> reaches the same total (**1.0**), and the join is **idempotent** (a second `gossip` is a no-op,
> **1.0**). Because concurrent increments touch *disjoint* vector entries, there is no conflict and no
> lost update; because `cincr` is purely node-local it is available even under partition (the AP
> property `incr` lacked). `cincr` reuses no replication and the merge is coordinator-level, so
> Tier-A ≡ Tier-B bit-for-bit; one omitted-when-empty `gcounters` map + one `GCounterSet` edit (purely
> additive).

> **▶ ED36 — the CRDT PN-counter: a decrementable counter that still converges loss-free (DS0 increment 29 — [`ed36.png`](../../figures/ed36.png)).**
> The decrement that turns ED35's grow-only G-counter into a full **PN-counter**. A G-counter only
> goes up; a PN-counter pairs **two** G-counters — `P` (the `cincr` half) and `N` (the `cdecr` half) —
> and reads **`P − N`**. `cdecr n key` is the exact twin of `cincr` over the N half: it bumps **only
> `n`'s own** decrement sub-count, so it inherits every property that made the G-counter work
> (node-local, **always available**, single-writer-per-entry, merged by the same per-(key, owner) max
> join over *both* halves). **Panel A (decrement works, loss-free, may go negative):** across a
> cluster-size sweep `k` `cincr`s then `m` `cdecr`s read back **`k − m`** (**1.0**); a fresh `cdecr`
> reads **−1** — the value goes **below zero** where a grow-only G-counter cannot (**1.0**); a
> partitioned-minority `cdecr` is acknowledged (**always available, 1.0**); and the concurrency
> contrast — **two** `cincr`s (majority) and **one** `cdecr` (minority) all count, and after
> `heal`+`gossip` the counter reads exactly **+2 − 1 = 1** (**no lost update across both halves,
> 1.0**). **Panel B (convergence):** the join over both halves converges **every** node to the net — a
> `gossip` chain spreads it epidemically (**1.0**), `anti_entropy` reaches the same net (**1.0**), and
> the join is **idempotent** (**1.0**). The one thing the PN-counter adds is the property the
> G-counter lacked — the value may go **negative** (the sub-counts stay monotone and non-negative;
> only their *difference* dips below zero). `cdecr` reuses no replication and the merge is
> coordinator-level, so Tier-A ≡ Tier-B bit-for-bit; one omitted-when-empty `ncounters` map + one
> `NCounterSet` edit (purely additive over increment 28).

> **▶ ED37 — the CRDT OR-Set: add-wins, re-addable, convergent (DS0 increment 30 — [`ed37.png`](../../figures/ed37.png)).**
> The canonical *interesting* CRDT — a replicated **set**, the data type a naive implementation gets
> wrong. An element-level **2P-Set** (a grow-only add-set + remove-set) is **remove-wins** and can
> **never re-add** a removed element. The **observed-remove set** fixes both with a **unique dot**:
> `sadd n key elem` tags the element with a fresh `(owner=n, seq)` dot and stores it in `n`'s observed
> add-set; `srem n key elem` tombstones **only the dots `n` has observed**; `smembers` is the elements
> with a non-tombstoned dot. The join is **set union** of both halves (commutative, associative,
> idempotent). **Panel A (the defining wins):** across a cluster-size sweep `sadd`-ing `k` distinct
> elements reads back all `k` (**1.0**); a removed element is **re-addable** (`srem` then `sadd`
> returns it, **1.0**) where a 2P-Set cannot; **add wins** — an element present cluster-wide, re-added
> at `n0` (a fresh dot) while `n3` removes the dot it saw, **survives** after `heal`+`gossip` (**1.0**)
> where a 2P-Set would drop it; and `sadd` is **always available** — a partitioned-alone node still
> adds (**1.0**). **Panel B (convergence):** from a diverged state (one side adds, the other removes)
> the union join converges **every** node to the same set — a `gossip` chain epidemically (**1.0**),
> `anti_entropy` on each node (**1.0**), idempotently (a second `gossip` is a no-op, **1.0**). The dot
> mechanism is the whole trick: identity is per-*dot*, not per-*element*, so a fresh add is never
> confused with a removed one. `sadd`/`srem` reuse no replication and the merge is coordinator-level,
> so Tier-A ≡ Tier-B bit-for-bit; two omitted-when-empty `orset_adds`/`orset_tombs` maps + the
> `ORSetAdd`/`ORSetTomb` edits (purely additive).

> **▶ ED38 — the CRDT MV-register: concurrent writes surface as siblings, not silent loss (DS0 increment 31 — [`ed38.png`](../../figures/ed38.png)).**
> The Dynamo/Riak data type that **surfaces** a write conflict instead of silently dropping one. Where
> the KV `put` and the counters resolve concurrent writes by last-writer-wins (one survives, one is
> lost — ED14/ED34), the **multi-value register** keeps *both* as **siblings** and lets a later reader
> resolve them. It reuses the OR-Set's dot/union machinery: `mvput n key val` tags `val` with a fresh
> dot, **tombstones every dot it currently observes** (a write supersedes the values it saw), and adds
> its own; `mvget n key` reads the surviving (non-tombstoned) sibling values. **Panel A (conflict
> surfaced, not lost):** across a cluster-size sweep `mvput` then `mvget` reads back the value
> (**1.0**); a *sequential* overwrite **resolves** to one value (**1.0**); but two *concurrent*
> `mvput`s on opposite sides of a partition — neither observing the other — **both survive** as
> siblings after `heal`+`gossip` (**1.0**), where a LWW `put` keeps only one (the conflict is
> *visible*); and `mvput` is **always available** (a partitioned-alone node still writes, **1.0**).
> **Panel B (convergence and resolution):** the union join converges **every** node to the same
> sibling set — a `gossip` chain epidemically (**1.0**), `anti_entropy` on each node (**1.0**),
> idempotently (**1.0**) — and a later context-aware `mvput` (observing both siblings) **resolves**
> them, so after convergence every node reads the single new value (**1.0**, the Dynamo
> read-and-resolve). The "supersede what you observed" rule is the whole design: a sequential write
> sees the prior and collapses it; concurrent writes see neither and both survive. `mvput` reuses no
> replication and the merge is coordinator-level, so Tier-A ≡ Tier-B bit-for-bit; two
> omitted-when-empty `mvreg_vals`/`mvreg_tombs` maps + the `MVRegWrite`/`MVRegTomb` edits (purely
> additive over increment 30).

> **▶ ED39 — the CRDT LWW-register: deterministic single-value resolution by a Lamport-timestamp order (DS0 increment 32 — [`ed39.png`](../../figures/ed39.png)).**
> The *policy-opposite* of ED38's MV-register: where the MV-register **surfaces** a write conflict as
> siblings, the LWW-register **deterministically picks one winner**. The mechanism is a **Lamport
> clock** — a per-node logical counter that makes "happens-after" a comparable order without a real
> clock (which a partitioned cluster cannot share, HW-5). `lwwput n key val` stamps `val` with `(ts,
> owner=n)` where `ts = lamport[n] + 1` (advancing `n`'s clock), and the join keeps the **max** copy by
> `(ts, owner, value)`. **Panel A (happens-after wins, deterministically):** across a cluster-size
> sweep `lwwput` then `lwwget` reads back the value (**1.0**); a write that **happened-after** another
> (a higher Lamport ts) **wins regardless of node id** — even a *lower*-id node's later write beats a
> higher-id node's earlier one (**causal LWW, 1.0**), where "highest node wins" gets it backwards;
> truly *concurrent* writes (equal ts) resolve to **one** value by the node-id tie-break, the same on
> every node (**deterministic resolution, 1.0**); and `lwwput` is **always available** (**1.0**).
> **Panel B (convergence):** the max-by-timestamp join converges **every** node to the single winner —
> a `gossip` chain epidemically (**1.0**), `anti_entropy` on each node (**1.0**), idempotently
> (**1.0**) — and the concurrent *loser* is **dropped** (the register holds one value, not two, the
> deterministic-resolution-vs-conflict-surfacing tradeoff made explicit, **1.0**). `lwwput` reuses no
> replication and the merge is coordinator-level, so Tier-A ≡ Tier-B bit-for-bit; two
> omitted-when-empty `lwwreg`/`lamport` maps + the `LWWRegSet`/`LamportSet` edits (purely additive over
> increment 31).

> **▶ ED40 — the CRDT OR-Map: a CRDT *of* CRDTs, the compositional capstone of the family (DS0 increment 33 — [`ed40.png`](../../figures/ed40.png)).**
> The capstone, because it is a CRDT **composed of** two CRDTs built earlier: the **OR-Set** (ED37)
> governs *field presence* (which fields the map has, add-wins + observed-remove over field names) and
> the **LWW-register** (ED39) governs each field's *value*. It is the in-CRDT-layer instance of the
> whole program's thesis — a faithful composite is a composition of faithful parts. `mput n map field
> val` adds a fresh presence dot for `field` *and* LWW-writes `val`; `mdel` observed-removes the field;
> `mget` reads a present field's value; `mkeys` enumerates the present fields (the map capability the
> flat KV/registers lack). **Panel A (map ops + the two composed semantics):** across a cluster-size
> sweep `mput` then `mget`/`mkeys` reads the field and value back (**1.0**); a `mdel` removes a field
> (**1.0**); a concurrent **value** update resolves by **LWW** (one winner, **1.0**); a concurrent
> `mput` survives a concurrent `mdel` — **add-wins field presence** (**1.0**), where a naive map loses
> the update; and `mput` is **always available** (**1.0**). **Panel B (convergence):** the composed
> join converges **every** node to the same fields *and* the same per-field values — a `gossip` chain
> epidemically (**1.0**), `anti_entropy` on each node (**1.0**), idempotently (**1.0**); the two halves
> converge independently (presence by set-union, value by LWW). `mput`/`mdel` reuse no replication and
> the merge is coordinator-level, so Tier-A ≡ Tier-B bit-for-bit; three omitted-when-empty
> `ormap_fields`/`ormap_tombs`/`ormap_vals` maps + the `ORMapField`/`ORMapTomb`/`ORMapVal` edits
> (sharing the Lamport clock, purely additive over increment 32).

> **▶ ED41 — the CRDT RGA: the first *ordered* CRDT, the basis of collaborative text (DS0 increment 34 — [`ed41.png`](../../figures/ed41.png)).**
> Every CRDT so far is **unordered** (set, counter, register, map); the RGA (replicated growable array)
> is the first *ordered* one — a sequence, in which any node can insert at any position and concurrent
> inserts converge to **one** deterministic order with no duplication (the property collaborative text
> editors need). Each element carries a unique id `(seq, owner)` and a `parent` id (the element it was
> inserted *after*, or `ROOT` for the head); the visible order is a DFS where siblings are ordered by
> id **descending**. `rins n list i val` inserts after the i-th visible element, `rdel` tombstones it
> (delete preserves structure — a tombstone is still an anchor), `rget` reads the visible values
> concatenated. **Panel A (sequence ops + deterministic concurrent insert):** across a cluster-size
> sweep sequential `rins` builds `"abc"` (**1.0**); a middle insert and a delete both work (**1.0**);
> two nodes inserting *different* characters at the *same* position concurrently read back the **same**
> string on every node after `heal`+`gossip` (one deterministic interleaving, both present, no
> duplication — **1.0**), where a naive list would diverge; and `rins` is **always available**
> (**1.0**). **Panel B (convergence):** the union join converges **every** node to the same sequence —
> a `gossip` chain epidemically (**1.0**), `anti_entropy` on each node (**1.0**), idempotently
> (**1.0**). The key insight: the **order is a pure function of the element set**, so convergence is
> free once the sets converge (set-union). `rins`/`rdel` reuse no replication and the merge is
> coordinator-level, so Tier-A ≡ Tier-B bit-for-bit; two omitted-when-empty `rga_elems`/`rga_tombs`
> maps + the `RGAInsert`/`RGATomb` edits (purely additive over increment 33).

Everything below serves that one figure. SPEC-2's prime directive was the same sentence
with "filesystem"; SPEC-5's with "network"; SPEC-6's with "composed host." SPEC-7's
difference is not a bigger world for its own sake — it is the **first world where you
*cannot* afford to verify every step even if you wanted to**, because (i) there is no
consistent global state to read in one shot (no global clock — HW-5), and (ii) checking
that an observed history is even *consistent* is NP-complete for serializability and
snapshot isolation (CAV 2025, §2.2). A budgeted oracle is no longer a convenience for
saving compute (SPEC.md §1); it is a *necessity*, and the cheapest sufficient verifier
must be *chosen*, not assumed. That choice — the **tiered oracle** — is SPEC-7's payload.

This spec is large because it is exhaustive by request, but the program is staged
(DS0–DS8, §13) so the deterministic core ships and is fully tested **with no runtime
dependencies and no GPU** before any learned model — exactly as M0–M3 did in v0, NW0–NW3
for the network, and HC0–HC3 for the host.

---

## 1. Why the distributed world, and why now

### 1.1 What SPEC-2, SPEC-5, and SPEC-6 left above them

The prior worlds model, respectively, a filesystem tree, a network of hosts, and a single
running machine. None models the thing those machines *cooperate to be*: a **service with
state replicated across machines**, kept consistent (or deliberately not) by a protocol.
A bank balance, a lock, a queue offset, a Raft log, a row under snapshot isolation — these
live *across* hosts, and their correctness is a property of the *ensemble*, not of any one
node SPEC-6 can snapshot. Three properties of the distributed world are absent from every
prior world and are exactly the properties that make the science bite next:

| Property | v0 / net / host | Distributed world (SPEC-7) |
|---|---|---|
| **Global state** | one consistent snapshot exists and is cheap to read | **none** — there is no instant at which all replicas agree; a "global state" is a *coordinated* (expensive) or *inconsistent* (cheap) construction |
| **Oracle cost** | bit-exact truth is free (`ρ=1` is the cheap ceiling) | **bit-exact truth is intractable** — checking serializability/SI of a history is NP-complete (CAV 2025); a consistent snapshot needs a protocol round |
| **Failure as first-class** | failure is an exit code | **partition, crash, message loss, reorder, clock skew** are the *medium* — the interesting dynamics only exist under fault |

These are not incremental. The disappearance of a free full-state oracle (W7) inverts the
economics SPEC.md §2 rests on, and it is *good news for the thesis*: the regime where you
*must* spend a scarce, expensive verifier wisely is precisely the regime the whole program
claims to be the metrology lab for (SPEC.md §4). The single-FS world was too easy to show
H1's knee (SPEC-2 §13, the honest negative); the distributed world is the first where
**not consulting is the only affordable default**, so the value of cheap-but-faithful
prediction is maximal.

### 1.2 The oracle survives — but it *tiers*

The reason to go here next rather than to vision/biology/robotics is unchanged (SPEC.md
§2): the oracle stays **constructible, deterministic, and free of real-world cost** —
*because the entire distributed-systems engineering field has spent a decade building
exactly the oracle we need.* The **deterministic-simulation-testing (DST)** tradition runs
the *real* distributed code inside a single-threaded discrete-event simulator with all
nondeterminism (network, disk, clock, RNG, thread schedule) quarantined behind seeded
shims, so a whole cluster's execution is a pure function of `(seed, commit)` and replays
bit-for-bit (FoundationDB's framework, Will Wilson, Strange Loop 2014; TigerBeetle's VOPR;
Antithesis's deterministic hypervisor, 2024; madsim/turmoil; Shadow, which executes real
unmodified binaries inside a deterministic network DES). What the field built as a *testing*
tool is, to us, a **free, fault-rich, perfectly-reproducible ground-truth factory**.

But "the oracle survives" now means a *spectrum* of oracles at different prices, because
bit-exact global truth is intractable (§1.1). SPEC-7's central design move is to make the
oracle **tiered** (§5):

```
oracle tier          what it verifies                         cost      analogue
-------------------  ---------------------------------------  --------  ----------------------------
bit-exact replay     the full next state, bit-for-bit         high      v0 ReferenceOracle, run on a
  (DES / Tier-B)       (a coordinated/simulated global snap)             whole cluster under DST
symbolic / formal    the protocol's next-state relation       medium    Batfish for routing (SPEC-5),
  (TLA+ / isolation)   (Raft step legal? SI satisfied?)                  now TLA+/isolation semantics
consistency-cycle    is the observed *history* admissible      low       Jepsen/Elle: cycle detection
  (Elle-style)         under the declared model (SER/SI/lin)?            over a partial observation
metamorphic / prop   does an equivalence/invariant hold        very low  SQLancer PQS/NoREC/TLP
  (SQLancer-style)     (rewrite-equivalence, no reference)?              metamorphic DB oracles
```

The cheap tiers do not return the truth; they return a *refutation or a pass*. That is
enough to gate a prediction, and it is enough to bound drift — and it is far cheaper than
reconstructing a consistent global state. **Choosing the cheapest oracle that can still
catch the model's current error is the new core of the consultation policy** (§8.2, the
`π_w` axis SPEC-6 introduced as "which sub-oracle," now generalized to "which *tier*").

### 1.3 The foils that frame the thesis: CWM and WebDreamer

Two 2024–2025 results are the cleanest external reference points, and neither is yet in the
repo's related-work set:

- **CWM — Code World Model (Meta FAIR, 2025; 32B open weights).** Mid-trained on ~120M
  Python execution traces to predict per-line local-variable state — a "neural debugger" —
  and reports strong SWE-bench-Verified and loop-termination-prediction numbers. CWM is the
  large-scale instantiation of *exactly* verisim's "predict the next program state given an
  action" thesis, at the code/host layer. **It is the foil that proves the gap:** CWM
  predicts state *un-verified* — there is no oracle in its loop, no budgeted correction, no
  faithful-horizon metric. Verisim is CWM *plus the deterministic oracle*: the same
  prediction target, but with a tiered verifier in the loop and `H_ε(ρ)` as the measured
  result CWM cannot produce. Position relative to CWM, do not compete with it on scale.
- **WebDreamer — "Is Your LLM Secretly a World Model of the Internet?" (Gu et al., 2024;
  NAACL 2025).** Uses an LLM as an *inference-time* world model to simulate the outcomes of
  web actions before committing, matching tree search at ~4–5× fewer real environment
  interactions. This is the propose-then-verify loop validated on the real web — and its
  world model is an *un-grounded* LLM. **It is the baseline to beat:** verisim's distributed
  world model is oracle-grounded, so it should sustain a longer faithful horizon per real
  interaction than an un-grounded LLM simulator, on a checkable distributed state.

> The contrast is the whole program in one line: CWM and WebDreamer show that *predicting
> system state is useful and learnable*; verisim adds the one thing the domain uniquely
> permits and they both lack — **a deterministic oracle in the loop, spent on a budget.**

And note what the foils share: CWM is a 32B transformer, WebDreamer is a prompted LLM — *different
model classes, same missing piece*. That is the point. The tiered oracle (§5) grounds **whatever**
sits in the proposer slot — a CWM-style transducer, the service-graph GNN of §6, an RSSM, or an LLM
caller — because the loop verifies the *prediction*, not the *predictor*. So the distributed
`H_ε(ρ)` curve and the tiered-oracle result (H17) are, like every prior world's, claims about the
oracle-loop rather than any one architecture: the distributed-layer instance of **deterministic
verification as a model-agnostic primitive** (SPEC.md §6 commitment 4, H22).

### 1.4 Why it is worth building for the community, not just for us

A deterministic, oracle-grounded, partially-observable model of a *distributed system* is a
benchmark the field does not have: a place to measure long-horizon faithfulness of a world
model over replicated state under fault, with ground truth, where the ground truth is
*tiered* because full truth is intractable. The DST tools (FoundationDB, TigerBeetle,
Antithesis, Jepsen) are evaluated on *bug-finding*, never on *faithful horizon under a
verification budget*; the learned-DB-component line (learned cost models, learned indexes)
optimizes performance, never predicts full consistency dynamics; the agent benchmarks
(τ-bench, AppWorld) grade *task success* against a stateful backend, but have no model of,
and no metric for, the backend's predicted next state. Verisim's contribution is the
missing metrology and the cheap faithful simulator the agents could call (§7), packaged
where researchers already look (an Inspect benchmark + a `verifiers`-spec RL env, SPEC-2
§15). This is the non-competitive contribution: not a bigger agent, but a free, honest
measuring instrument for distributed-state faithfulness and the verified simulator beneath
it.

---

## 2. Lessons folded in (and the design choice each one forces)

The "browse far and wide" deliverable, distributed-world edition. Each lesson is stated as
design guidance with its source; skeptical notes flag where the literature is hype.
Citations are consolidated in [`docs/related-work.md`](./docs/related-work.md), kept current
by the engine, **name + venue + year per that file's no-fabricated-links policy** (arXiv
IDs only where independently verified — several plausibly-recent IDs surfaced in research
are deliberately *omitted* here until verified).

### 2.1 Deterministic simulation testing — the oracle *and* the data factory

- **FoundationDB simulation (Will Wilson, Strange Loop 2014; SE-Radio 685, 2025); TigerBeetle
  VOPR ("Simulation Testing For Liveness," 2023); Antithesis "deterministic hypervisor"
  (2024); madsim / turmoil; Shadow (USENIX-pedigree, executes real binaries in a network
  DES).** Run real code in a single-threaded DES; quarantine *all* nondeterminism behind
  seeded shims; inject faults deterministically (`BUGGIFY`); a run is a pure function of
  `(seed, commit)`. TigerBeetle's time-dilation (seconds of sim ≈ tens of minutes of
  wall-clock) is the quantitative argument for a *faster-than-sim* learned surrogate. →
  **Design choices:** the Tier-A oracle (§5.1) is a from-scratch deterministic DES of a
  pinned distributed semantics, and the same DES is the **training-data factory** — seeded
  fault injection produces unlimited, perfectly-labeled, fault-rich trajectories
  (`DD-D3`); `(seed, commit)` is the reproducibility key v0 already uses (SPEC-2 §12);
  Tier-B (§5.2) can *wrap* madsim/Shadow/Antithesis-class runtimes rather than rebuild them.
- *Skeptical note:* DST's defensible core is rock-solid (these systems ship and find real
  bugs), but it explores a *seeded random walk* through fault space — robustness against
  *sampled* faults, not exhaustiveness. A model trained on DST traces inherits the
  simulator's coverage blind spots; coverage is a curriculum axis (§3.4), not a given.

### 2.2 The full oracle is intractable — and that is the point

- **"On the Complexity of Checking Mixed Isolation Levels for SQL Transactions" (CAV 2025):**
  consistency checking is polynomial for "saturable" isolation levels but **NP-complete for
  Snapshot Isolation and Serializability.** **Elle (Kingsbury & Alvaro, VLDB 2020):**
  black-box transactional-consistency checking via *cycle detection* on observed histories,
  scaling to hundreds of thousands of operations where linearizability checkers (Knossos)
  are exponential. → **Design choices:** define faithfulness against a *declared consistency
  model*, not a (nonexistent, intractable) single global state (`DD-D2`); use Elle-style
  cycle detection as the **cheap consistency-tier oracle** (§5.3) that gates predictions
  without reconstructing truth; and make the NP-hardness the *motivation* for the budgeted
  oracle, not an obstacle to it — this is the strongest H1 motivation the program has had.
- **Formal semantics as symbolic oracles: VerIso (PVLDB 2025) and weak-isolation separation
  logic (ICFP 2025)** for transactions; **Apalache (OOPSLA 2019, symbolic TLA+); Stateright;
  the P language** for consensus/protocol next-state relations. → **Design choice:** the
  *symbolic/formal tier* (§5.1) verifies "is this protocol step legal / does this history
  satisfy SI?" directly from a spec, the distributed analogue of SPEC-5's Batfish
  control-plane oracle; Stateright's *shared-code* model (the checker and the implementation
  are the same code) is the cleanest fit for a from-scratch reference.

### 2.3 Metamorphic / property oracles — verify without a reference

- **SQLancer metamorphic oracles — PQS, NoREC, TLP (Rigger & Su, OSDI/ESEC-FSE 2020);
  graph-based transactional-bug oracle (Jiang et al., OSDI 2023).** Verify behavior via
  *equivalence/invariant* relations (a query and its rewrite must agree) with **no reference
  state at all**. → **Design choice:** the cheapest oracle tier (§5.3) is metamorphic —
  invariants the predicted state must satisfy (a transferred balance conserves total; a
  dequeue cannot return an un-enqueued item; a committed read-your-writes must hold). It
  costs almost nothing and catches a large class of model hallucinations before any
  expensive tier is spent.
- *Skeptical note:* metamorphic oracles are *incomplete* — they catch violations of the
  specific relation, not arbitrary wrongness. They are a fast first filter, never the whole
  gate; the bits-to-correct headline (§9) is still anchored on the bit-exact tier where it is
  affordable.

### 2.4 Learning-augmented algorithms — the loop's missing theory

- **Algorithms with predictions / learning-augmented algorithms (the 2018–2025 line:
  learned indexes with error-bounded fallback; learning-augmented caching such as LARU; MAT
  "ML at the tail").** A prediction guides the common case; a worst-case-safe deterministic
  fallback bounds the damage when the prediction is wrong; the figure of merit is a
  **competitive ratio** that degrades gracefully with prediction error (and recovers the
  classical worst-case bound when the predictor is useless). → **Design choice:** this is
  *exactly* propose-verify-correct (`DD-D4`): the model is the predictor, the oracle is the
  worst-case-safe fallback, `ρ` is how often the fallback is consulted. SPEC-7 adopts the
  competitive-ratio *framing* as the theoretical backbone the loop has lacked, and reports
  faithful horizon as a competitive ratio against the full-oracle ceiling (H18) — giving the
  central mechanism a guarantee, not just a curve.
- *Skeptical note:* the clean competitive-ratio theorems assume a fallback that restores
  *exact* correctness each time it fires; our cheap tiers do not (they refute, they do not
  reconstruct). So the guarantee is *empirical* (a measured ratio) at the cheap tiers and
  *provable* only at the bit-exact tier. Stating which is which honestly is mandatory.

### 2.5 In-house patterns: worldify (state/belief) and securifine (change-safety eval)

- **`worldify` (sibling repo) — a deterministic, zero-dependency temporal fact store.** It
  models state as immutable, time-scoped *facts* with **supersession** (a change creates a
  new fact and marks the old one `superseded_by`, never deletes — full history is
  queryable), **confidence scores** (epistemic certainty per fact), **causal chains**
  (`caused`/`enabled`/`prevented` edges with `trace_causes`/`trace_effects`), and
  **snapshot / branch / restore** with versioned JSON export. → **Design choice (`DD-D5`):**
  the distributed world's state representation *is* this pattern — an **event-sourced causal
  log**, because a distributed system's ground truth is naturally a log of events with a
  *happens-before* (causal) partial order, replica-local **belief** (confidence) over the
  unobserved part, and **consistent snapshots** as the coordinated reads. worldify's
  supersession = MVCC versions; its causal edges = the happens-before relation that *defines*
  causal consistency; its confidence = the belief a partial-observation oracle (§5.4) leaves
  the model holding; its snapshot/branch = consistent global snapshots and the
  counterfactual branching the oracle makes free (§9, H5). The repo reuses worldify's data
  model rather than reinventing it.
- **`securifine` (sibling repo) — measurement-first differential safety eval.** A
  baseline → post-change → **differential comparison** pipeline with **severity-weighted**
  scoring, deterministic pattern-matching verifiers (not LLM-judged), config layering
  (CLI > env > file > defaults), SHA-256 artifact versioning, an abstract `ModelInterface`
  (HTTP + offline-cache), and an explicit *limitations* doc. → **Design choices:** the
  distributed world's **change-safety** evaluation (§7, §12) *is* securifine's differential
  pattern — score the *delta* in consistency-faithfulness between "before config push" and
  "after," **severity-weighted by consistency-violation class** (`DD-D8`); the
  offline-cache `ModelInterface` is the pattern for the LLM-callable simulator (§7); and the
  limitations-doc discipline carries into `docs/distributed-semantics.md`.

### 2.6 World-model-as-tool & executable distributed environments

- **WebDreamer (Gu et al., 2024; §1.3); τ-bench / τ²-bench (Sierra, 2024–2025);
  AppWorld (ACL 2024); ToolEmu (ICLR 2024).** τ-bench drives tool-agent-*user* dialogues
  against a *stateful database backend*; AppWorld has agents act via code+API over nine
  stateful apps; ToolEmu has an LM *emulate* tool execution (the un-grounded version of what
  we ground). → **Design choices:** the downstream framing (§7, §15) is the agent calling a
  **cheap faithful simulator of the service it is about to change**, and τ-bench/AppWorld are
  the *external legibility harness* (§12) the way CAGE-4 was for SPEC-5 and OSWorld for
  SPEC-6; ToolEmu's "LM-emulated tools" is the baseline verisim beats by grounding the
  emulator in an oracle.
- **SWE-Gym (2024) / R2E-Gym (2025) and the execution-based-vs-execution-free reward
  finding.** The SWE-agent field converged on *execution-based* verifiers (run the tests,
  expensive, exact) and *execution-free* reward models (cheap, approximate) being
  **complementary**. → **Design choice:** this is independent empirical support for the
  tiered-oracle design — you want both the expensive-exact and the cheap-approximate
  verifier, and the science is *how to mix them on a budget* (H17).

### 2.7 Model architecture for distributed state

- **m4 / RouteNet bipartite GNNs (SPEC-5 §2.2), DreamerV3 RSSM (SPEC-5 §2.4), and
  state-space models (Mamba/SSD, Gu & Dao, 2023–2024).** Mamba carries a *fixed-size explicit
  recurrent state* updated in linear time — a natural fit for "carry and update a compressed
  system state," and a complement to the delta-prediction target (v0's E4 finding that delta
  dominates full-state). **CLRS / TransNAR (neural algorithmic reasoning)** show GNNs that
  execute classical algorithms generalize OOD better with *step-wise intermediate-state
  supervision*, and **Jin & Rinard (ICML 2024)** show program semantics emerge in a model
  trained only on next-token prediction of code. → **Design choices:** `M_θ` is a
  message-passing predictor over the **service graph** (services ↔ shared resources, the m4
  template, §6.1) with an RSSM belief over unobserved replicas and an *optional SSM
  recurrent carry* as an EH-style architecture lever; supervise on *intermediate*
  consistency-relevant facts (commit order, lock holder, log index), not only the final
  reachable state.
- *Skeptical note:* neural algorithmic reasoning generalizes *narrowly* (single algorithms,
  modest sizes); CLRS success does not promise scalable distributed-protocol modeling.
  Architecture is an ablation axis (ED4), not a faith.

### 2.8 The SLM thesis, restated for this layer

- **"Small Language Models are the Future of Agentic AI" (NVIDIA, Belcak et al., 2025).**
  Narrow, repeated, verifiable tasks favor a small specialist that is escalated *from*, not a
  generalist. → **Design choice:** `M_θ` is a small, distributed-system-specialized
  simulator — the cheap verifiable specialist an LLM agent calls (§7), not a competitor to
  the LLM. The oracle is an *infinite, perfect distillation teacher* (unlimited labeled
  trajectories via §2.1), the unfair advantage of this domain.

---

## 3. The world (environment)

### 3.1 State as an event-sourced, replicated, partially-observable log

The distributed state `s` is **not** a single tree (SPEC-2), graph (SPEC-5), or bundle
(SPEC-6). It is a set of **replicas**, each holding a local copy (or shard) of logical
objects, plus an **event log** with a happens-before partial order, plus in-flight
messages — i.e. the worldify temporal-causal-fact model (§2.5, `DD-D5`) instantiated for a
cluster.

```
DistributedState = {
  nodes:    map[node_id -> NodeState]       # each NodeState ⊇ a SPEC-6 host (it *runs* on one)
  replicas: map[(object_id, node_id) -> ReplicaState]   # per-node copy/shard of a logical object
  log:      [ Event ]                        # the causal event log (worldify facts + happens-before)
  inflight: map[msg_id -> Message]           # messages sent, not yet delivered
  protocol: ProtocolState                    # consensus/leader/term/view, lock table, txn table
  global:   { sim_clock, rng_cursor, partition_set, last_result }
}
ReplicaState = { object_id, node_id, value_version (MVCC), confidence, last_applied_index }
Event        = { id, node, op, happens_before:[event_id], superseded_by?, confidence }
Message       = { id, src, dst, payload_kind, send_time, deliver_after }
```

- **There is no `global state` field.** A consistent global snapshot is *derived* by a
  coordinated read (an oracle call, §5), never stored — this is W7 made structural.
- **The happens-before edges are the spine.** Causal consistency, MVCC visibility, and the
  divergence metric (§9) are all defined over them; this is worldify's causal-chain model
  (`trace_causes`) doing real work.
- **Canonicalization** (sorted, hashed, volatile-IDs normalized) is mandatory exactly as in
  every prior world (SPEC-3 `DD-1`): node/replica/msg IDs are canonicalized so divergence
  measures *competence*, not identifier churn.
- **The prior worlds embed verbatim.** A `NodeState` *is* a SPEC-6 `HostState`; the links
  between nodes *are* SPEC-5's network; SPEC-7 owns only the replication/protocol/log glue.
  SPEC-7 is the integrating spec, not a fourth parallel one (mirroring how SPEC-6 embedded
  SPEC-2).

### 3.2 Actions

Three families, in a constrained grammar paired with the oracle, as every prior world pairs
a grammar with a reference oracle:

1. **Client ops (the workload):** `put/get/cas(key, val)`, `begin/commit/abort` (a
   transaction over keys), `enqueue/dequeue(queue)`, `read/write` at a declared isolation
   level, an API call against a service.
2. **Protocol / admin ops:** `propose(value)` to a consensus group, `add/remove replica`,
   `leader step-down`, `config push`, `deploy version`, `scale up/down`.
3. **Fault / time ops (the medium):** `partition(set_a | set_b)`, `heal`, `crash(node)`,
   `restart(node)`, `drop/delay/reorder(msg)`, `clock_skew(node, δ)`, and `advance Δt`
   (deliver in-flight messages, fire timers: election timeout, lease expiry, retransmit).

The fault/time family is the source of all interesting dynamics and is the distributed
analogue of SPEC-5's `advance Δt` and SPEC-6's scheduler — `partition`/`crash`/`reorder`
under a seed *is* the `BUGGIFY` of DST (§2.1).

### 3.3 Determinism contract

The world is deterministic given `(initial cluster, seed)`: the seed fixes message
delivery order, fault injection, and per-node RNG/clock — the DST contract (§2.1). Tier-A is
a pure function. Tier-B (a wrapped madsim/Shadow/Antithesis-class runtime) pins
commit + image + kernel and seeds the scheduler; **asynchrony/partition is the
nondeterminism source that cannot be sealed cheaply (HW-5)** — it is made a *seeded,
controllable input* (the fault schedule), and `determinism_report` (SPEC-3 §2.2) declares,
per figure, the consistency model assumed and whether ordering was simulated or recorded.

### 3.4 Scale & curriculum

Difficulty is a small set of dials (mirroring SPEC-2 §2.4 / SPEC-5 §3.4 / SPEC-6 §3.4):
number of nodes, replication factor, transaction concurrency, declared consistency model
(linearizable → quorum → causal → eventual for replication, plus serializable → snapshot →
read-committed for txn isolation — *weaker is harder to predict* because more histories are legal),
workload contention
(hot keys), and — the new ones —
**fault intensity** (the `BUGGIFY` rate) and **partition entropy** (how often/asymmetrically
the network splits). The curriculum starts at **one node, no replication, no faults** (the
distributed analogue of v0's smallest world) and ratchets up to **a partitioned, faulting,
weakly-consistent multi-replica cluster under contention**. Fault intensity and partition
entropy are explicit axes because they are exactly what H20/H21 measure.

---

## 4. The state delta `Δ`

`M_θ` predicts a structured **log/replica delta**, not a full global state and not raw wire
bytes (§2.7, and v0's E4 result). The delta vocabulary composes the prior worlds' edit types
with the new replication/protocol/log types:

```
EventAppend(node, op, happens_before)        # a new causal-log event
ReplicaWrite(object, node, new_version)      # MVCC version bump on one replica
ReplicaConverge(object, nodes, version)      # replicas agree on a version (anti-entropy/commit)
                                             #   DS0 incr 12: single-node read-repair (`anti_entropy`)
                                             #   is realized as a per-object `ReplicaWrite`; the
                                             #   multi-node atomic-convergence form is a later increment
                                             #   DS0 incr 19: `append`'s committed-log fold applies
                                             #   committed entries to the KV as per-node `ReplicaWrite`s
MsgSend(id, src, dst, kind) / MsgDeliver(id) / MsgDrop(id)   # message lifecycle
ProtocolStep(kind, term, index, leader?)     # consensus/leader/lease/view change
                                             #   DS0 incr 16: SHIPPED as ProtocolStep(kind, term,
                                             #   leader) — `elect` installs the leader at a new term,
                                             #   `propose` is the leader-fenced majority write (ED23)
                                             #   DS0 incr 17: `step_down` clears leader (→ None) at the
                                             #   same term — voluntary relinquishment (ED24)
LeaseSet(until)                              #   DS0 incr 18: the leader-lease deadline (the §4 lease
                                             #   form): `lease` grants it, `lread` reads under it with
                                             #   no quorum, `elect`/`step_down` clear it (ED25)
LogSet(node, entries) / CommitIndexSet(index)  # DS0 incr 19: the Raft replicated log — `append`
                                             #   sets a node's log (adopting the leader's prefix,
                                             #   reconciling any divergent tail) + advances the
                                             #   monotone commit index when a majority holds it (ED26)
MemberSet(members)                           #   DS0 incr 20: the consensus voting set —
                                             #   `add_replica`/`remove_replica` reconfigure it, so
                                             #   the quorum threshold tracks the membership (ED27)
QueueSet(queue, node, items)                 #   DS0 incr 21: a FIFO-queue replica's ordered items —
                                             #   `enqueue` appends, `dequeue` pops the head; the
                                             #   delivery semantics follow the consistency model (ED28)
VersionSet(node, version)                    #   DS0 incr 22: a node's running software version —
                                             #   `deploy` sets it; two nodes share a quorum only if
                                             #   within `max_version_skew` (rolling upgrade, ED29)
ConfigSet(node, key, value)                  #   DS0 incr 24: a leader-committed, majority-replicated
                                             #   cluster config value — `config_push` reaches the
                                             #   majority, the minority keeps stale config (ED31)
GCounterSet(key, holder, owner, count)       #   DS0 incr 28: a CRDT G-counter sub-count — `cincr`
                                             #   bumps the owner's own; anti_entropy/gossip join by
                                             #   per-(key, owner) max, loss-free + convergent (ED35)
NCounterSet(key, holder, owner, count)       #   DS0 incr 29: the PN-counter decrement half — `cdecr`
                                             #   bumps the owner's own N sub-count; cget = P − N, may
                                             #   go negative; same max join over both halves (ED36)
ORSetAdd(key, holder, elem, owner, seq)      #   DS0 incr 30: a CRDT OR-Set add-dot — `sadd` tags
                                             #   `elem` with a unique (owner, seq); union join (ED37)
ORSetTomb(key, holder, owner, seq)           #   DS0 incr 30: a CRDT OR-Set tombstone — `srem`
                                             #   tombstones an observed dot; add-wins + re-addable
MVRegWrite(key, holder, value, owner, seq)   #   DS0 incr 31: a CRDT MV-register write-dot — `mvput`
                                             #   tags `value` with (owner, seq); siblings on conflict
MVRegTomb(key, holder, owner, seq)           #   DS0 incr 31: a CRDT MV-register tombstone — a write
                                             #   supersedes every dot it observed (ED38)
LWWRegSet(key, holder, value, ts, owner)     #   DS0 incr 32: a CRDT LWW-register entry — `lwwput`
                                             #   stamps `value` with a Lamport (ts, owner); max wins
LamportSet(holder, value)                    #   DS0 incr 32: a node's Lamport clock — advanced on
                                             #   write and merge (max-apply; backs LWW + OR-Map; ED39)
ORMapField(map, holder, field, owner, seq)   #   DS0 incr 33: an OR-Map field-presence dot — `mput`
                                             #   tags `field` present; add-wins (OR-Set half)
ORMapTomb(map, holder, owner, seq)           #   DS0 incr 33: an OR-Map presence tombstone — `mdel`
                                             #   removes an observed field dot (observed-remove)
ORMapVal(map, field, holder, value, ts, owner)  # DS0 incr 33: an OR-Map field value — LWW max per
                                             #   (map, field) (the LWW-register half; ED40)
RGAInsert(list, holder, seq, owner, value, pseq, powner)  # DS0 incr 34: an RGA element — id (seq,
                                             #   owner), value, parent (pseq, powner); union join
RGATomb(list, holder, seq, owner)            #   DS0 incr 34: an RGA tombstone — delete preserves
                                             #   structure; concurrent inserts converge in order (ED41)
TxnBegin(id) / TxnCommit(id, order) / TxnAbort(id)           # transaction lifecycle
LockAcquire(obj, holder) / LockRelease(obj, holder)
HostDelta(...)  NetDelta(...)                 # the SPEC-6 / SPEC-5 deltas, on embedded subsystems
                                             #   DS0 incr 23: SHIPPED as HostStep(node, edits) — a
                                             #   SPEC-6 host bundle delta on a node's embedded host
                                             #   (`host node <syscall>`, ED30); NetDelta is later
SetResult(status, value_token)               # client-visible result (as in v0)
```

The **M1-analogue invariant** is required and tested: `apply(state, oracle.delta) ==
oracle.next_state` for every transition, by construction, and `apply` is *compositional* —
embedded host/net deltas reuse SPEC-6/SPEC-5's `apply` verbatim, and the new types apply to
the log/replica/protocol layer. Delta↔serialization round-trips. This invariant keeps the
loop (§8) model-agnostic, as in every prior world.

---

## 5. The oracle `O` — the tiered oracle (the heart of SPEC-7)

The structural novelty (§1.2): the oracle is a *menu* at four price points, and the
consultation policy chooses *which tier* to spend, not just *when*.

> **Design decision (`DD-D1`): consult the cheapest tier that can refute the current
> prediction.** A prediction that violates a metamorphic invariant is rejected by the
> very-low-cost tier; one that violates the declared consistency model is caught by the
> low-cost cycle-detection tier; only predictions that pass both and still need
> ground-truth correction spend the high-cost bit-exact tier. *Rationale:* full-state truth
> is intractable (§2.2), so spending it indiscriminately is the one thing the distributed
> world forbids. *Alternative considered:* always bit-exact (the v0/net/host default) —
> rejected as infeasible here, which is exactly what makes this world the thesis's proving
> ground.

### 5.1 Tier-A — reference distributed oracle (the deterministic core)

A **from-scratch deterministic DES of a pinned distributed semantics**: a replicated
key-value store with MVCC and a declared isolation level, a simplified but real **consensus
group** (leader election, log replication, commit — a Raft-subset), a lock/transaction
table, a message layer with seeded delivery/partition/loss, and the embedded SPEC-6 hosts +
SPEC-5 network. It has **no runtime dependencies and needs no GPU**, like every prior core.
It is the executable truth, paired with a normative `docs/distributed-semantics.md`. Golden
trajectories pin the semantics and are denylisted from the engine (§14).

Beside the DES runs the **symbolic/formal tier**: a checker that, given the protocol spec
(a TLA+/Stateright/P-style next-state relation) and the declared isolation semantics
(VerIso/ICFP-style), answers *"is this transition legal / does this history satisfy the
model?"* directly — the distributed analogue of SPEC-5's Batfish control-plane oracle, and
deterministic and free. Whether the symbolic tier is a *non-redundant* signal over the DES
is hypothesis **H17**-adjacent and tested in ED6.

### 5.2 Tier-B — system oracle (reality check)

A **wrapped real DST runtime** — madsim/turmoil (Rust, tokio-compatible), Shadow (real
binaries in a deterministic network DES), or an Antithesis-class deterministic hypervisor —
running a real (small) distributed system (e.g., a real embedded KV store / a real
Raft library) under seeded scheduling, pinned commit + image, **no real-internet egress**
(§15). Tier-B attacks SPEC-3 wall **W1** ("the oracle is a model, not reality") for the
distributed domain; the Tier-A↔Tier-B gap is itself a reportable result and a curriculum
signal (where our pinned semantics diverges from a real system is where data should
concentrate). DST is what makes Tier-B *deterministic at all* — without it, a real cluster
is not replayable and cannot be a bit-exact oracle.

> **▶ Shipped (DS8).** Tier-B ships as an **in-repo, dependency-free actor runtime**
> ([`distoracle/system.py`](../../src/verisim/distoracle/system.py)) rather than a wrapped external
> binary: `SystemDistOracle` runs the replicated-KV protocol as autonomous **node actors** (each
> holding *only its own replicas + inbox*, no global state — the cluster is emergent, W7 made
> operational) under a **seeded scheduler** whose delivery order is *seed-shuffled* vs Tier-A's
> sorted order, so agreement certifies delivery-order-independence (LWW is a commutative join). This
> *is* the DST principle — the same single-threaded-deterministic-scheduler-over-real-message-passing
> model madsim/turmoil use — realized without the heavy external dependency; a `threaded` tier puts
> the actors on **real OS threads + queues** for the strongest reality claim (the host
> `SandboxOracle` analog). The differential (the observable-cluster channel) measures the Tier-A↔Tier-B
> gap as **ED7**: bit-exact 1.000 (residual 0) across the grammar and all workloads, an oracle-invariant
> `H_ε(ρ)` curve, and a teeth-bearing broken-actor control the harness *catches* — the distributed W1
> retirement (§13 DS8, [`experiments/ed7.py`](../../src/verisim/experiments/ed7.py)). Wrapping an
> *external* real-binary DST runtime (madsim/Shadow/Antithesis) over the same differential remains
> open future work.

### 5.3 The cheap tiers — consistency-cycle and metamorphic

- **Consistency-cycle (low cost):** Elle-style cycle detection (VLDB 2020) over a *partial*
  observation of the history answers "is what we have seen so far admissible under the
  declared model?" in near-linear time, *without reconstructing global state* — the
  workhorse cheap oracle.
- **Metamorphic / invariant (very low cost):** SQLancer-style relations and domain
  invariants (conservation, monotonic queue offsets, read-your-writes, no-lost-update) the
  predicted state must satisfy — a first filter that costs almost nothing and catches gross
  hallucination before any expensive tier is touched (§2.3).

### 5.4 Partial observation & bits-to-correct

The oracle inherits SPEC-5/SPEC-6's **probe (cheap, localized) vs full (expensive)** modes,
and adds the §5.1–5.3 *tier* choice on top. **Bits-to-correct** (SPEC-3 §7, SPEC-4 §5)
generalizes directly and **decomposes by tier and by object**: `bits-to-correct = Σ_object
MDL(correction)`, 0 iff the prediction equals truth on the tier consulted, smooth and
unfakeable otherwise. The per-tier breakdown tells the engine (and H17) *which tier* is
buying the most faithfulness per dollar — the lever a single scalar hides.

> **◐ shipped — the probe projection** (DS3 increment 4, [`dist/observe.py`](../../src/verisim/dist/observe.py),
> [`distmetrics/observe.py`](../../src/verisim/distmetrics/observe.py), [`ED12`](../../src/verisim/experiments/ed12.py)).
> `observe(state, vantage)` realizes the **probe (cheap, localized)** mode as a deterministic
> projection: an observer connected to a set of `vantage` nodes sees the replicas on *reachable*
> (up + co-partitioned) nodes, **never the in-flight replication medium**, and labels every other
> node `unreachable` *without a reason* — so a **crashed** node and a **partitioned-away** node are
> byte-identical from one vantage (the failure-detector limit behind FLP) and separable only from a
> second. `observable_divergence` is the probe-mode faithfulness; because a bit-faithful step is
> necessarily observably faithful, the **observable horizon dominates the bit horizon** structurally,
> and the gap is the unobservable medium (ED12: probe gap **+9.0 steps** for `subtle` in-flight
> errors, the partial-observation form of H19). This is the deterministic substrate the (deferred)
> RSSM belief (§6.2) must roll forward under partition — the belief predicts the *full* state from
> the *observable* one, a task undefined until "observable" is. See
> [`docs/distributed-semantics.md` §10](../distributed-semantics.md).

> **Design decision (`DD-D2`): faithfulness is defined against a *declared consistency
> model*, not a single global state.** `ε` is a *consistency-level* threshold (does the
> predicted history still admit serializability / SI / causal / linearizability?), not only
> a bit distance. *Rationale:* there is no consistent global state to be bit-faithful to
> (W7); the operationally meaningful question is whether the model predicts the
> *observable consistency behavior*, which is what a defender or SRE actually relies on.
> Bit-exact divergence remains the headline *where the bit-exact tier is affordable*;
> consistency-faithfulness is the metric that survives where it is not.

---

## 6. The model `M_θ`

### 6.1 Architecture
A **message-passing predictor over the service graph** (services/replicas ↔ shared
resources: locks, logs, queues — the m4 bipartite template, SPEC-5 §6.1), action- and
clock-conditioned, emitting a structured delta under grammar-constrained decoding (the
constrained-decode machinery from M4/NW4/HC4 carries over). Message-passing depth is tuned
to the cluster diameter (a leader change `k` hops away cannot be represented with too few
steps). Heterogeneous magnitudes (log indices vs byte counters vs flags) pass through a
**symlog transform** (DreamerV3, reused). An **SSM/Mamba recurrent carry** (§2.7) is an
optional architecture arm for the long causal log.

### 6.2 Latent belief (RSSM) over the unobserved cluster
A recurrent latent carries a **belief over the replicas and in-flight messages the model
cannot see** (§2.5, worldify confidence). Under full observation it degenerates to a Markov
predictor; under partition/partial-observation it is the only way to roll forward the
unseen subgraph, and its variance is the *calibrated-by-construction* uncertainty signal
that the consultation policy reads (the H2-negative fix continuing from SPEC-5 §6.2).

### 6.3 Drift mitigations (required ablation levers, inherited)
The SPEC-5/SPEC-6 levers carry over and are mandatory ED4 arms because the distributed
world's deep causality and fault dynamics make drift compound hard: **noise-injected
rollout training** (GNS — the cheapest, highest-leverage one; here the "noise" is naturally
*fault* injection, §2.1), **self-forcing / scheduled sampling**, and the **multi-step /
latent-overshoot objective**.

### 6.4 Size & specialization (SLM)
`M_θ` is small and distributed-system-specialized (§2.8). ED4's size axis re-asks v0's
capacity question (v0: *no*, size is not the floor) in the harder, faulting, weakly-
consistent world — where the answer may differ, and the honest measurement is the point.

---

## 7. SLM/LLM complementarity — the cluster simulator an agent calls

This answers the user's "complement/integrate SLM/LLM" intent at the layer where it matters
most: a computer-use or cyber-defense agent acts on a *running distributed system*
("push this config," "fail over this primary," "drain this node"), so the distributed world
is exactly the simulator such an agent needs. The world model is not a competitor to the
LLM; it is the cheap, faithful, verifiable cluster the LLM reasons *over* — the grounded
version of WebDreamer (§1.3).

- **World model as a "what-if" the agent calls.** An LLM agent proposes a plan (a sequence
  of admin ops); `M_θ` simulates the consequences across replicas fast; the tiered oracle
  verifies on a budget. This is propose-verify-correct lifted from the *op* level to the
  *plan* level — Dreamer's "plan in imagination," made honest by a real oracle, for the
  thing τ-bench/AppWorld agents actually do (§2.6). The baseline to beat is an *un-grounded*
  LLM simulator (WebDreamer, ToolEmu).
- **Change-safety as differential faithfulness (securifine pattern, §2.5).** "Will this
  config push break consistency/availability?" is scored as the *delta* in
  consistency-faithfulness between before and after, severity-weighted by violation class
  (`DD-D8`) — securifine's baseline→post→differential pipeline, with the oracle (not a
  pattern-matcher) as the verifier.
- **Distillation from an infinite, perfect teacher.** The DES emits unlimited correctly-
  labeled, fault-rich trajectories (§2.1) → distill into the SLM. Ground truth at scale is
  the domain's unfair advantage.
- **Speculative execution / routing.** `M_θ` drafts cluster dynamics; the oracle verifies
  only when the policy fires, at the cheapest sufficient tier (`ρ` × tier). The LLM is
  escalated to only for natural-language intent → plan translation, never for simulating
  dynamics it is bad at. The execution-based-vs-execution-free complementarity (SWE-Gym,
  §2.6) is independent evidence this mixture is the right shape.

The integration is a **protocol**, specified here and built in DS8: a `Model` that
implements both "predict next cluster state" (for the loop) and "simulate a plan" (for an
LLM caller), packaged like v0's `eval`/`rl` modules and exposed behind a securifine-style
abstract interface (offline-cache + live), so external agents can call the verified cluster
simulator. Open question §17.8 (defining `H_ε` for a *plan* unit) is inherited from SPEC-5
§17.8 / SPEC-6 §17.8.

> **◐ shipped** ([`distsim/`](../../src/verisim/distsim/),
> [`test_distsim.py`](../../tests/test_distsim.py)): the LLM-callable cluster simulator, the
> dependency-free distributed analogue of the host [`hostsim/`](../../src/verisim/hostsim/).
> `DistSimulator` both *predicts the next cluster state* (the loop interface) and *simulates a plan*:
> `imagine` rolls `M_θ` over a proposed admin-op plan with **no oracle** (Dreamer's cheap draft),
> and `verify` runs that imagination against the oracle step-by-step, returning a `DistPlanReport`.
> Beyond the host's bit-exact plan horizon, the report carries the two readouts the cluster world
> makes meaningful: (1) a **consistency-faithful plan horizon** distinct from the bit-exact one — how
> many leading plan steps the agent can trust the model's split-brain prediction, which (per
> ED5/H19) outlasts the bit-exact horizon when the error hides in the in-flight medium; and (2)
> **change-safety as differential consistency-faithfulness** (the securifine pattern, §2.5,
> `DD-D8`) — *"will this plan break consistency?"* scored as the change in the cluster's consistency
> health (fraction of objects converged) from before to after, with the **model-vs-oracle agreement**
> on the safe/unsafe verdict as the task-level faithfulness. A composing **task oracle** (`DistGoal`,
> the §7 "third oracle": "object `x` converged everywhere", "no split-brain", "node back up") gives
> goal-level agreement too. Dependency-free except the model — the loop's dependency-free baselines
> satisfy `DistModel`, so the protocol is exercised in CI with no torch. *(Deferred: the offline-cache
> live/abstract split and the plan-level RL reward, inherited from the host packaging.)*

---

## 8. The loop (tiered, partially-observed propose-verify-correct)

Same skeleton as SPEC.md §5.2, with the distributed extensions (tier choice, consistency-
correction, the fault-driven advance).

```
for each step t:
    Δ̂ ← M_θ(belief, a_t)                          # PROPOSE: predict the log/replica delta
    ŝ' ← apply(ŝ, Δ̂); belief ← update(belief, Δ̂)
    if π_c decides to consult (budget ρ):                  # WHEN to consult
        tier ← π_w(belief, Δ̂)                              # WHICH TIER/oracle (§8.2) — the new axis
        v ← O[tier](s, a_t)                                # metamorphic | cycle | symbolic | bit-exact
        d ← divergence(v, ŝ')                              # refutation, consistency, or full
        ŝ', belief ← C(ŝ', belief, v)                      # CORRECT / belief-update (§8.3)
        replay.add(s_t, a_t, v)                            # accumulate the experience stream (SPEC-6 §8.5)
        if healer: θ ← heal(θ, replay)                     # SELF-HEAL: gated TTT step (SPEC-6 §8.4)
    s ← O.true_next(s, a_t)                                # the cluster advances regardless (under the seed)
```

### 8.1 Consultation policy `π_c` (when)
The v0/net/host policies carry over (`fixed`, `drift_triggered`, `uncertainty_triggered`,
`learned`), now reading the RSSM belief variance over the unobserved cluster (§6.2) — the
calibrated-by-construction signal partial observability supplies.

### 8.2 Tier/oracle policy `π_w` (which) — the central new axis
Given a consultation, *which tier* to spend (§5): metamorphic (almost free, incomplete),
consistency-cycle (cheap, refutes consistency violations), symbolic (medium, protocol-step
legality), or bit-exact replay (expensive, full truth). The information-gain choice — spend
the cheapest tier that can refute the *current* most-uncertain, most-consequential predicted
edit — is the generalization of SPEC-5's `π_o` (what-to-observe) and SPEC-6's `π_w`
(which-sub-oracle) to *which-price-of-truth-to-buy*. It is the heart of H17 and ED2, and it
is the operational form of `DD-D1`.

### 8.3 Correction / belief operators `C`
`hard_reset` (snap the observed slice to truth), `residual`, `projection` (onto the nearest
*consistency-model-consistent* state — e.g. repair the history so it re-admits SI), and the
`belief-filter` (Bayesian/particle update of the unobserved replicas). Partial observation +
weak consistency make these genuinely different (no v0 identity collapse) — ED3.

### 8.4 Self-healing & the experience stream (optional, gated, inherited)
Each consultation is a free labeled example; take a gated gradient step (SPEC-3 §6.3
recipe — small-lr, replay, PEFT, trust-region revert under the keep-if-better referee).
The SPEC-6 §8.5 experience stream applies: a never-ending sandboxed cluster run from which
the model predicts, the tiered oracle verifies on a budget, and the model heals — with the
plasticity probe (SPEC-6 §9.4, HW-4) tracked. Off by default so the static baseline is
clean (`DD-5`).

---

## 9. Metrics

### 9.1 Consistency-faithfulness (the headline-new metric)
Over the horizon, does the model's predicted history admit the *declared consistency model*
when the true one does (and forbid it when the true one does)? Graded by Elle-style cycle
detection over the predicted vs true partial histories. This is the operationally meaningful
number (`DD-D2`) and the one that survives W7. Reported alongside bit-exact divergence
wherever the bit-exact tier is affordable.

### 9.2 Composed / consistency divergence `d(s, ŝ)`
Normalized symmetric difference over canonical replica/log/protocol tuples (the worldify
fact-set), **plus** a consistency-class distance (do predicted and true histories sit in the
same level of the consistency hierarchy?). `d = 0` iff replicas, log, and consistency class
all agree. `ε` and what counts as "ε-close" for a consistency class is open question §17.3.

### 9.3 Faithful horizon `H_ε(ρ)` and the competitive ratio
`H_ε(ρ)` unchanged in definition (max steps within `ε`), now over a partitioned, faulting
cluster under tiered consultation. The headline curve (ED1, DS6). **New:** report it as a
**competitive ratio** `H_ε(ρ) / H_ε(ρ=full-oracle-ceiling)` against prediction error — the
learning-augmented-algorithms figure of merit (`DD-D4`, H18), with the cheap-tier ratio
labeled *empirical* and the bit-exact-tier ratio labeled *provable* (§2.4 skeptical note).

### 9.4 Bits-to-correct (per-tier), probe/tier efficiency, calibration
- **Bits-to-correct**, per-tier and per-object (§5.4): the scale-free gate.
- **Faithful horizon per oracle-dollar**, where each tier has a different dollar cost — the
  distributed enrichment of `ρ`, and the quantity H17 is about.
- **Belief calibration:** does RSSM belief variance predict error? (continuing SPEC-2 §7.2 /
  SPEC-5 §9.4 — decides whether smart `π_c`/`π_w` *can* work).

---

## 10. Hypotheses

SPEC-7 operationalizes two **already-stated** hypotheses that needed an intractable-oracle,
weakly-consistent world to bite, and adds four **new** ones (H17–H20, plus the data-factory
H21), each falsifiable and each naming its honest negative (SPEC.md §9: "the favorable curve
might not exist").

### 10.1 Existing hypotheses this spec operationalizes (do not re-coin)
- **H8 (SPEC-3 §13) — the interesting interior lives in harder worlds.** The distributed
  world — combinatorial reachability over replicas under fault, with an *intractable* full
  oracle — is the strongest test yet of H1's favorable knee (≥80% of ceiling horizon at
  ≤20% consultation). **This, with H17, is SPEC-7's headline** (ED1, DS6). *Honest negative:*
  the interior is flat/linear here too → the knee is not about world hardness; report it.
- **H5 (SPEC.md §9) — counterfactual lift.** Oracle-grounding improves interventional
  fidelity ("what if this node had not been partitioned at step `t`?") on identical data,
  trained on **branch-replay counterfactuals** the deterministic DES makes free (re-run from
  `(seed, t)` with one fault flipped) — ED6. *Honest negative:* counterfactual data adds
  nothing over factual.

### 10.2 New hypotheses (H17–H21, non-colliding with H1–H16)
- **H17 — tiered oracles dominate single-tier.** A budgeted *mix* of cheap (metamorphic +
  consistency-cycle) and rare expensive (bit-exact/symbolic) oracle calls, scheduled by
  `π_w` (§8.2), achieves higher faithful horizon **per oracle-dollar** than spending the same
  dollar budget entirely on bit-exact full-state checks. This is the central claim and has no
  prior-world analogue (every prior world had a single cheap full oracle). *Honest negative:*
  the cheap tiers refute too rarely to matter, so all the value is in the expensive tier and
  tiering buys nothing — in which case the distributed world is no different in kind, only in
  cost.
- **H18 — the loop is a learning-augmented algorithm with a bounded ratio.** The
  oracle-gated loop achieves faithful horizon within a *bounded factor* of the full-oracle
  ceiling at sub-linear oracle cost, and the factor degrades gracefully with the model's
  prediction error (recovering the trivial bound when the model is useless) — i.e.
  propose-verify-correct has a competitive ratio (`DD-D4`, §2.4). *Honest negative:* the
  ratio is unbounded / grows with horizon → cheap-tier correction does not bound drift and
  only bit-exact reset does (which would itself sharpen *why*). **Result — SPLIT, both
  halves reported (ED5, DS8, [`experiments/ed5.py`](../../src/verisim/experiments/ed5.py),
  [`ed5.png`](../../figures/ed5.png)).** Fitting the competitive ratio `H_ε(ρ)/ceiling`
  across `ρ × prediction error` (the noise dial) at the bit-exact tier (where ρ maps
  linearly to oracle-dollars, so the quarter ρ *is* the `B/4` budget ED2 reads): the
  **graceful-degradation-with-error half is CONFIRMED** — the quarter-budget ratio is
  monotone in the model's competence (1.00 → 0.45 → 0.11 → 0.07 → 0.05 as per-step error
  rises 0.0 → 1.0), recovering the trivial bound (ratio 1.0) for a perfect model and
  collapsing toward the free-running floor for a useless one, exactly the learning-augmented
  signature. But the **bounded-ratio-at-*sub-linear*-cost half reproduces the program's
  recurring floor→cliff / no-knee negative** — at the quarter budget a competent-but-noisy
  model's ratio sits near the floor (~0.11), and the cliff to the ceiling only appears as
  ρ→1, because a discrete-error world's faithful horizon is a *prefix* property only
  near-full consultation protects (the E1/EN1/EH1/ED1 finding, now in competitive-ratio
  form). So the loop *is* learning-augmented in the error axis, but the budget axis buys no
  free lunch on this world ([`test_ed5`](../../tests/test_ed5.py), dependency-free).
- **H19 — consistency-faithful outlasts bit-faithful.** Under a weak consistency model, a
  model is *consistency-faithful* (predicts the observable consistency behavior, §9.1) for
  materially longer than it is *bit-faithful* (predicts the exact replica state), because
  many bit-states map to the same admissible history. *Honest negative:* the two horizons
  coincide → consistency adds no slack, and W7's "no global state" does not buy the model
  any forgiveness. **Result — CONFIRMED, mode-dependently (ED5, DS8).** On the same
  free-running rollout (ρ=0, which exposes the model not the loop) the **consistency-faithful
  horizon outlasts the bit-faithful one for the `subtle` (in-flight) error class — H=13.1 vs
  H=1.5, a gap of +11.6 steps with a disjoint bootstrap CI [3.1, 21.8]** — because the
  asynchronous-replication in-flight message is the gap: a corrupted in-flight payload is
  immediately *bit*-visible (the message fact differs) but **consistency-invisible** (the
  per-object converged/split view reads only replicas) until that message is delivered by
  `advance` and writes a replica. For the `gross` (durable-replica) error class the two
  horizons coincide (gap +0.8, CI includes 0 — the control), exactly the gross/subtle
  structure H17/ED3 turn on. So W7's "no global state" *does* buy the model forgiveness, but
  only where the error lives in the consistency-invisible medium — reported, not assumed.
- **H20 — weaker consistency is harder to predict.** Faithful horizon *decreases*
  monotonically as the declared consistency model weakens (linearizable → eventual), because
  weaker models admit exponentially more legal histories and the model must track which one
  actually occurred. The curve `H_ε(consistency-level)` is the first quantification of
  "consistency strength vs predictability." *Honest negative:* horizon is flat across
  consistency levels → predictability is set by fault intensity, not consistency strength.
  **Result — the mechanism CONFIRMED, dependency-free (ED4 consistency-level arm, DS7,
  [`experiments/ed4_consistency.py`](../../src/verisim/experiments/ed4_consistency.py),
  [`ed4_consistency.png`](../../figures/ed4_consistency.png)).** The `CONSISTENCY_MODELS` axis
  (§3.4) gains its first implemented strong end — **`linearizable`**: synchronous all-replica
  writes, CP write-rejection under partition, so no replica is ever stale and there is **no
  in-flight medium** ([`docs/distributed-semantics.md` §2.1](../distributed-semantics.md),
  goldens in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). Sweeping the declared
  model resolves H20 *through its connection to H19*: the consistency-vs-bit gap (H19) is
  **exclusively a weak-consistency phenomenon** — it needs the consistency-invisible in-flight
  medium to hide errors in. Measured, free-running, with an *exact* in-flight-only error class:
  the `subtle` gap is **+10.5 steps under `eventual` (in-flight rate 3.2/step) and exactly 0
  under `linearizable` (in-flight rate 0)**, while the `gross` durable-replica control is 0 at
  both levels. So strong consistency buys the model no forgiveness because there is no hidden
  state to forgive — the H20 mechanism made concrete. *Honest scope:* the synthetic proposer's
  error distribution is tied to the eventual world's structure, so the *absolute*-predictability
  form of H20 (a monotone `H_ε(level)` curve) is left to the learned `M_θ`; this arm reports the
  gap, which the synthetic proposer measures cleanly ([`test_ed4_consistency`](../../tests/test_ed4_consistency.py)).
- **H21 — fault-injected training beats fault-free (the DST/BUGGIFY lesson).** A model
  trained on DST-style seeded-fault trajectories is more faithful *under fault* than one
  trained on equal-volume fault-free trajectories, at equal clean accuracy. *Honest
  negative:* fault-free training transfers to faulting rollout for free → the fault
  distribution is already implied by the fault-free dynamics.

### 10.3 Outcome → implication: where each distributed result routes the program

Per the epistemic engine (SPEC.md §10.1), each hypothesis is pre-registered to a forward move on *both*
branches. The distributed world is the sharpest illustration of the project's defining move — **a
limitation, faced honestly, becomes the contribution.** Here the limitation is fundamental: bit-exact
full-state truth is *intractable* (serializability/SI checking is NP-complete; there is no consistent
global state without coordination — the wall W7). A lesser program would call that the end of the road.
Instead it is the *premise* of SPEC-7: precisely *because* the full oracle is too expensive, faithfulness
must be verified at **tiered** cost (metamorphic → consistency-cycle → symbolic → bit-exact), and choosing
the cheapest sufficient tier (`π_w`) becomes the central new science. The wall did not stop us; it
*defined the spec*.

- **H17 (tiered oracles dominate single-tier).** *Confirmed* → the central claim holds: budgeted cheap
  tiers buy more faithful horizon per dollar than all-bit-exact → tiering is the distributed-world method.
  *Refuted* → the cheap tiers refute too rarely to matter and all value is in the expensive tier → the
  distributed world is no different *in kind* from the host world, only in cost — a clean simplification
  that retires a whole axis of complexity. Either branch is a real answer to "is tiering worth it?"
- **H18 (the loop has a bounded competitive ratio).** *Confirmed* → propose-verify-correct is a
  learning-augmented algorithm with a provable ratio that degrades gracefully with model error → the loop
  gets *theory*, not just curves (DD-D4, §2.4). *Refuted* → the ratio is unbounded / grows with horizon →
  cheap-tier correction does not bound drift and only bit-exact reset does, which **sharpens *why*** and
  tells us exactly where the cheap tiers fail — a negative that advances the theory by ruling out the easy
  conjecture.
- **H19 (consistency-faithful outlasts bit-faithful).** *Confirmed* → many bit-states map to one
  admissible history, so predicting *observable consistency* buys real slack → the right faithfulness
  target under weak consistency. *Refuted* → the horizons coincide and W7's "no global state" buys no
  forgiveness → a precise statement of when consistency-level abstraction *doesn't* help.
- **H20 (weaker consistency is harder to predict).** *Confirmed* → `H_ε(consistency-level)` is the first
  quantification of consistency-strength vs. predictability — a genuinely new measurement. *Refuted* →
  horizon is set by fault intensity, not consistency strength → redirects modeling effort to fault
  handling, a useful reprioritization.
- **H21 (fault-injected training beats fault-free — the DST/BUGGIFY lesson).** *Confirmed* → seeded-fault
  trajectories (the FoundationDB/TigerBeetle tradition, §2.1) train fault-robustness factual data cannot →
  validates DST as a *data factory*, not just a test harness. *Refuted* → fault-free transfers for free →
  bounds the value of fault injection for *modeling* (as opposed to testing).

The throughline, stated for the hardest world so it is unmistakable: **we do not retreat from
intractability; we tier around it, measure what the tiers buy, and report the number whichever way it
falls.** A wall that is named, quantified, and engineered around is not a limit on the program — it is the
program's next theorem.

---

## 11. Walls (relative to SPEC-3 / SPEC-5 / SPEC-6)

SPEC-7 makes concrete SPEC-3's **W1** (oracle-is-a-model) via Tier-B DST runtimes (§5.2),
inherits SPEC-5's **W5** (asynchronous/temporally-extended effects, now under partition) and
SPEC-6's **W6** (composed multi-subsystem state) and **HW-4** (plasticity loss, via the
experience stream). It adds one genuinely new wall and one hard wall:

- **W7 — there is no consistent global state to be faithful to.** Every prior world had one
  consistent snapshot you could read cheaply and compare bit-for-bit. The distributed world
  has none — a global state is either *coordinated* (expensive) or *inconsistent* (cheap and
  wrong). W7 is what forces `DD-D2` (faithfulness against a consistency model) and the tiered
  oracle (`DD-D1`), and it is what the prime directive attacks.
- **HW-5 (new) — asynchrony & partition.** No global clock; FLP impossibility; the CAP
  tradeoff. Message ordering and partition timing are the nondeterminism source that
  record/replay only tames at the cost of *fixing a schedule* (the DST move). SPEC-7 does not
  pretend to solve it; it makes the fault schedule a *seeded, declared* input (§3.3) and the
  consistency model an explicit choice, and `determinism_report` states both per figure.

---

## 12. Experiments (ED-series)

Non-colliding with E1–E4, the reserved E5/E6 (SPEC-2 §9), EN1–EN9 (SPEC-5), and EH1–EH6
(SPEC-6). The distributed suite is its own namespace, **ED1–ED6**. Each mirrors a prior
experiment's role and names the hypotheses it tests (§10). Every figure regenerates from
config + seeds (the `figures/reproduce.sh` discipline) and **negative results are
first-class** (the repo norm).

- **ED1 — the distributed `H_ε(ρ)` curve** (role of E1/EN1/EH1; the prime directive, DS6).
  Sweep `ρ × ε × difficulty × consistency-level × fault-intensity`. Bootstrap-CI aggregation.
  Reported also as the competitive ratio (§9.3). *Does the knee appear — H8 — and is it
  bigger here because the full oracle is dear?*
- **ED2 — when × which-tier policies** (role of E2/EN2/EH2): cross `π_c` (when) with `π_w`
  (which tier, §8.2), at equal *dollar* budget. *Does the cheapest-sufficient-tier mixture
  beat all-bit-exact and all-cheap — H17 — and does belief-variance scheduling beat fixed?*
  **◐ shipped (the fixed-tier × `escalate` arm at equal dollar budget)**
  ([`experiments/ed2.py`](../../src/verisim/experiments/ed2.py),
  [`ed2.png`](../../figures/ed2.png), [`ed2.csv`](../../figures/ed2.csv)): the
  **faithful-horizon-vs-oracle-dollar frontier** per tier policy on the synthetic proposer
  (dependency-free, GPU-free), with policies compared at a matched budget by interpolating each
  one's horizon along its Pareto envelope (a true equal-*dollar* comparison) and the **H18
  competitive ratio** read off at the sub-linear quarter budget. **H17 in budget form, confirmed
  mode-dependently:** at `B/4`, the metamorphic tier beats bit-exact for **gross** errors
  (H=14.2 vs 4.2, ratio 0.36) and loses for **subtle** errors (H=1.5 vs 4.2, where `escalate`
  also loses to single-tier bit-exact — the honest negative) ([`test_ed2`](../../tests/test_ed2.py)).
  **◐ the `π_c` "smart-when" half also ships** ([`experiments/ed2_smart.py`](../../src/verisim/experiments/ed2_smart.py),
  [`ed2_smart.png`](../../figures/ed2_smart.png), [`ed2_smart.csv`](../../figures/ed2_smart.csv)): at
  a *fixed* interior budget `ρ`, compare the three §6.1 policies (`fixed`/`uncertainty`/`drift`) at
  equal `ρ` on the real flat `M_θ`, the signal being its constrained-decode entropy (wired into the
  loop's `StepContext` by the DS5 runner — the network/host runners already did this; the distributed
  runner gained the `_predict`/`DistUncertaintyModel` plumbing here). **H9 — the standing H2/H9
  negative carried into the distributed world, and *sharper* than a tie:** entropy-gated consultation
  does **not** beat `fixed` — it is strictly *worse*, lift **0.08–0.12×** at every budget, because
  faithful horizon is a *prefix* property (the first divergence step) and `fixed` consults at step 0
  to protect the prefix while the entropy signal spends its budget on late high-entropy steps and lets
  the model derail early. The flat decode-entropy signal is a decode-time artifact, not a calibrated
  belief; this localizes the smart-`π_c` lever to the (deferred) structured `M_θ`'s RSSM belief
  variance — exactly the EH2 lesson, where the host's factored arm's belief variance beat fixed ~2.2×
  where the flat arm's entropy could not ([`test_ed2_smart`](../../tests/test_ed2_smart.py), torch
  extra). The **learned-`M_θ` equal-dollar arm** also shipped (see DS7, [`ed2_learned`](../../figures/ed2_learned.png)).
  *Deferred: the smart-`π_w` (which-tier) scheduling and the structured-arm `π_c` the flat-arm null motivates.*
- **ED3 — correction / belief operators** (role of E3/EN3/EH3): `hard_reset` vs `residual`
  vs `projection` (consistency-model-consistent) vs `belief-filter`. *Do operators differ
  under partition + weak consistency (no v0 identity), and does correction teach over the
  stream — H7?*
  **◐ shipped — and the distributed world *does* break the v0 identity, mode-dependently**
  ([`experiments/ed3.py`](../../src/verisim/experiments/ed3.py), [`ed3.png`](../../figures/ed3.png),
  [`ed3.csv`](../../figures/ed3.csv); the `distloop/operator.py` correction operators the DS5 runner
  was missing). v0 proved an *identity*: a consult returns the full one-step truth, so
  `hard_reset`/`residual`/`projection` all snap to the same `s'` and are behaviorally identical on
  `H_ε` (they differ only in diagnostics). ED3 asks whether the distributed world breaks it — and it
  does, because the cluster state has a part a *partial* correction can decline to fix: the **in-flight
  replication messages**, the stale-read source under partition and exactly the `subtle` error class
  the cheap tiers also miss (§5). The new `ReplicasOnlyCorrection` snaps the durable replicas to truth
  but **trusts the model's predicted in-flight**. Result (synthetic proposer, dependency-free): for
  **gross** (corrupted replica write) errors all four operators recover the same horizon (**H=7.2**,
  the v0 identity holds); for **subtle** (corrupted in-flight) errors the three full-correction
  operators hold the identity (**H=6.2**) but `ReplicasOnlyCorrection` **collapses to H=1.8** (gap
  4.5) — it trusts the corrupted in-flight and the coupled state keeps drifting. *The v0 operator
  identity holds for full correction and breaks for partial correction exactly on the in-flight
  medium — the distributed world's hidden state a partial correction cannot see, tied to the same
  gross/subtle structure H17 turns on* ([`test_ed3`](../../tests/test_ed3.py)). The residual/projection
  diagnostics (bits-to-correct, repaired fraction) quantify how much truth each correction injects.
  *Deferred: a consistency-model `projection` that corrects to the nearest weak-consistency-legal
  state (needs the multi-consistency DS0 increment); online correction-teaches-the-stream (H7).*
- **ED4 — representation & drift ablation** (role of E4/EN4/EH4): service-graph GNN vs flat
  serializer (H11's distributed analogue); RSSM-belief vs Markov; SSM-carry on/off;
  fault-injection (noise) on/off — the **H21 arm**; self-forcing on/off; size;
  **consistency-level sweep** (H20) and **fault-intensity / partition-entropy** as the new
  axes. *Which lesson buys horizon, and how does `H_ε` fall with weaker consistency — H20 —
  and does fault-injected training transfer — H21?*
  **◐ the H21 fault-injection arm shipped (DS7, [`ed4_fault.png`](../../figures/ed4_fault.png)).
  ◐ the consistency-level arm (H20) ships** ([`experiments/ed4_consistency.py`](../../src/verisim/experiments/ed4_consistency.py),
  [`ed4_consistency.png`](../../figures/ed4_consistency.png), [`ed4_consistency.csv`](../../figures/ed4_consistency.csv)),
  dependency-free: it gives the `CONSISTENCY_MODELS` axis (§3.4) its first strong end —
  **`linearizable`** (synchronous all-replica writes, CP write-rejection under partition, no
  in-flight medium) — and sweeps the declared model. **H20 mechanism confirmed through H19:** the
  consistency-vs-bit gap is exclusively a *weak*-consistency phenomenon — it needs the
  consistency-invisible in-flight medium. The `subtle` (in-flight) gap is **+10.5 under `eventual`
  (in-flight rate 3.2/step) and 0 under `linearizable` (rate 0)**, the `gross` durable-replica
  control 0 at both levels — strong consistency buys no forgiveness because there is no hidden
  state to forgive ([`test_ed4_consistency`](../../tests/test_ed4_consistency.py)).
  **◐ the absolute-predictability learned arm (H20) ships**
  ([`experiments/ed4_consistency_learned.py`](../../src/verisim/experiments/ed4_consistency_learned.py),
  [`ed4_consistency_learned.png`](../../figures/ed4_consistency_learned.png),
  [`ed4_consistency_learned.csv`](../../figures/ed4_consistency_learned.csv), torch extra): the named
  deferral — the synthetic arm can only report the *gap* (the absolute horizon at equal noise is
  confounded by delta composition across levels: a `put` is one local write + N async messages under
  `eventual` but N synchronous writes under `linearizable`), so it trains **one flat `M_θ` per level**
  (same init seed, only the *world* differs) and measures the free-running (ρ=0) horizon — an honest
  absolute predictability. **H20 confirmed in direction:** the model free-runs **~2.4× further under
  `linearizable` (bit `H_ε`=1.4) than under `eventual` (0.6)** — strong consistency is more
  predictable because there is less hidden state to track. *Honest caveat:* the absolute horizons are
  small (a weak flat free-runner, consistent with ED1-learned's low ρ=0 floor), so the CIs overlap —
  the lift is directional, not disjoint; the clean separation awaits a stronger free-runner (the
  structured arm). **And the honest difference from the synthetic arm:** the H19 gap on the *real*
  model is **positive at both levels** (eventual +0.4 [0.17, 0.67] disjoint; linearizable +1.3, noisy),
  not the synthetic arm's clean *eventual-only* gap — because a real model's errors land on
  consistency-invisible **bookkeeping** (clocks, the causal log, partition structure) present at both
  levels, not only the in-flight medium the dialed synthetic error targets, so the consistency oracle
  forgives more of the real model's errors than the "weak-consistency-only" reading predicts
  ([`test_ed4_consistency_learned`](../../tests/test_ed4_consistency_learned.py)). *Deferred: the
  service-graph GNN/RSSM representation arm.*
- **ED5 — consistency-faithful vs bit-faithful & the competitive ratio** (role of E4
  objective axis): measure the gap between the consistency-faithful and bit-faithful
  horizons (H19) and fit the learning-augmented competitive ratio (H18) across `ρ` and
  prediction error. *Does consistency buy slack — H19 — and is the loop's ratio bounded —
  H18?* **◐ shipped** ([`experiments/ed5.py`](../../src/verisim/experiments/ed5.py),
  [`ed5.png`](../../figures/ed5.png), [`ed5.csv`](../../figures/ed5.csv); the
  consistency-faithfulness trajectory the DS5 runner now records alongside bit-exact
  divergence — the §9.1 headline-new metric's first loop consumer). Both findings on the
  dependency-free synthetic proposer. **H19 confirmed mode-dependently:** free-running
  (ρ=0), the **consistency-faithful horizon outlasts the bit-faithful one for `subtle`
  (in-flight) errors — H=13.1 vs 1.5, gap +11.6, disjoint CI [3.1, 21.8]** — the in-flight
  message is bit-visible but consistency-invisible until delivered by `advance`; for `gross`
  (durable-replica) errors the two coincide (the control). **H18 split:** the
  competitive-ratio fit across `ρ × prediction error` shows graceful degradation with error
  **confirmed** (quarter-budget ratio monotone 1.00 → 0.05 as error rises, recovering the
  trivial bound for a perfect model) while the bounded-ratio-at-sub-linear-cost half
  reproduces the **floor→cliff / no-knee negative** (ratio near the floor at `B/4`, the cliff
  only at ρ→1) — the learning-augmented property holds in the *error* axis, no free lunch in
  the *budget* axis ([`test_ed5`](../../tests/test_ed5.py)). *Deferred: the consistency-level
  H20 sweep and the provable-vs-empirical competitive-ratio split across tiers (needs the
  multi-consistency-model DS0 increment); the counterfactual ED6.*
- **ED6 — counterfactual & multi-tier grounding** (H5, H17-adjacent): train with branch-
  replay counterfactuals (re-run from `(seed, t)` with one fault flipped, §10.1); add the
  symbolic/formal tier (§5.1) on top of the DES; measure interventional fidelity and whether
  the symbolic tier is a non-redundant signal (the distributed analogue of SPEC-5's H12 /
  SPEC-6's EH6). **◐ the counterfactual / H5 arm ships** ([`experiments/ed6.py`](../../src/verisim/experiments/ed6.py),
  [`ed6.png`](../../figures/ed6.png), [`ed6.csv`](../../figures/ed6.csv)): three matched-count arms
  train the same flat DS4 `M_θ` — `trajectory` (base light-fault on-policy), `trajectory-more`
  (5× more on-policy data, the volume control), `+counterfactual` (base + free oracle **fault**-flip
  branches, the near-miss partitions/crashes §17 Q7) — then predict held-out fault interventions,
  scored bit-exact (full next cluster state) and by **medium recall** (predicts the partition/crash
  split-brain). **H5 — and the distributed world is where it finally pays, the honest inverse of
  EN6/EH6.** `+counterfactual` beats **both** the base **and** the matched-volume control on **both**
  metrics with disjoint CIs (intervention-exact **0.51 vs 0.25 vs 0.06**, medium-recall **0.56 vs
  0.22 vs 0.05**) — where the network (EN6) and host (EH6/H16) found counterfactual supervision adds
  nothing over volume. The mechanism is the distributed **medium** (partition/crash/in-flight): a
  hidden state the light-fault on-policy distribution underrepresents, so on-policy *volume* buys
  little (0.06→0.25) while off-policy oracle **fault branches** buy a lot (0.25→0.51) — the held-out-
  intervention analogue of H21 (fault-injection beats fault-free at equal volume). *Honest caveat:*
  the counterfactual branches are fault-heavier than the on-policy control, so the lift conflates
  counterfactual *branching* with the fault *coverage* it carries — but this is the identical
  methodology under which EN6/EH6 found null, so the distributed positive under the same design is
  the result; the branching-vs-coverage split is future work (tied to H21) ([`test_ed6`](../../tests/test_ed6.py),
  torch extra; the matched-*volume* arm needs the minibatched `train_batched` K2 loop, the first
  distributed experiment to). **◐ the two-oracle / H12 slice ships**
  ([`experiments/ed6_two_oracle.py`](../../src/verisim/experiments/ed6_two_oracle.py),
  [`ed6_two_oracle.png`](../../figures/ed6_two_oracle.png),
  [`ed6_two_oracle.csv`](../../figures/ed6_two_oracle.csv)): the distributed analogue of SPEC-5's
  H12 / SPEC-6's EH6 — the cheap **consistency oracle** (the §9.1 split-brain decision: is each
  object converged or split?) as a *second oracle* against the full **bit-exact** one, scored
  teacher-forced over the fault-heavy `adversarial` workload on the dependency-free synthetic
  proposer. **H12 confirmed, mode-dependently:** (1) **non-redundant rate 0** by construction — the
  consistency view is a pure function of the replica state, so a bit-exact-correct prediction is
  always consistency-correct (the cheap oracle catches *nothing* the full one misses: *redundant for
  verification*); (2) **consistency-sufficient rate tracks the in-flight medium** — of the steps
  where the model's *full* prediction is wrong, it is still consistency-faithful **1.00 for `subtle`
  (in-flight) errors vs 0.00 for `gross` (durable-replica) errors** (disjoint CIs), the per-step
  teacher-forced form of ED5's free-running H19 horizon gap; (3) at a **consult-fact ratio of 0.28**
  — the consistency answer is ~3.6× cheaper than the full state, the gap *widening under fault*
  because the medium inflates the full state but never enters the consistency view. So the
  consistency oracle is *redundant* but a **cheaper, decision-sufficient** consult for the question
  an SRE/defender actually asks — the tiered-oracle premise (§5) made concrete, dependency-free
  ([`test_ed6_two_oracle`](../../tests/test_ed6_two_oracle.py)). **◐ the learned-`M_θ` re-pointing of
  this slice ships** ([`experiments/ed6_two_oracle_learned.py`](../../src/verisim/experiments/ed6_two_oracle_learned.py),
  [`ed6_two_oracle_learned.png`](../../figures/ed6_two_oracle_learned.png),
  [`ed6_two_oracle_learned.csv`](../../figures/ed6_two_oracle_learned.csv)): what ED1-learned is to
  ED1, this is to the two-oracle slice — train the flat DS4 `M_θ` (exactly as ED2-learned does) and
  run the **same** teacher-forced H12 measurement on the *real* error distribution rather than the
  dialled one (no `gross`/`subtle` knob — one model, one mixed error distribution). **H12 confirmed on
  the real model, and it is the honest *mirror* of ED2-learned read through the other oracle:**
  non-redundant **0.0** (unchanged — structural), the consistency oracle **decision-sufficient on
  0.57 [0.53, 0.61]** of the model's bit-wrong steps at a **consult-fact ratio of 0.28 (~3.6× cheaper)**
  — *even though the full prediction is wrong 87% of the time* (the uniform-trained model on the
  fault-heavy `adversarial` eval). The 0.57 lands **between the synthetic `gross` (0.0) and `subtle`
  (1.0) poles**: ED2-learned showed the constrained decoder removes the `gross` *out-of-vocab* class
  so the cheap *refutation* tiers are useless, but the residual learned errors are a **mixture**
  (predominantly the consistency-invisible in-flight class, not purely it), so the cheap *decision*
  oracle is sufficient on a majority but not all of them. The same model, same decoder: the cheap
  oracle **loses as a verifier** (ED2-learned's tiers refute nothing) yet is **decision-sufficient on
  the majority of errors as a decision oracle** — the clearest single statement of why the tiered
  oracle's value depends on *which question you ask it*, on the model that actually exists
  ([`test_ed6_two_oracle_learned`](../../tests/test_ed6_two_oracle_learned.py), torch extra).
  *Deferred: the symbolic/legality tier as a third oracle.*
- **ED12 — partial observation: the probe-faithful horizon + the crash/partition
  indistinguishability** (the DS3-increment-4 experiment for §5.4 / `DD-D2`). The increment
  experiments (ED7–ED18) extend the ED1–ED6 namespace one milestone-increment at a time.
  `observe(state, vantage)` projects the cluster onto what a probe at a set of `vantage` nodes can
  see — replicas on reachable nodes only, **never the in-flight medium**, dark nodes labelled
  `unreachable` with no crash-vs-partition reason. Two findings, dependency-free: **(a)** the
  *observable*-faithful horizon outlasts the *bit*-faithful one for `subtle` (in-flight) errors
  (free-running probe gap **+9.0 steps**, disjoint CI [4.0, 16.1]) and coincides for `gross`
  (durable-replica) errors — the partial-observation form of H19/ED5, read through *physical
  observability* rather than the consistency-view abstraction, and structurally guaranteed to
  dominate (a bit-faithful step is observably faithful); **(b)** a single external vantage cannot
  tell a crash from a partition (**indistinguishable rate 1.0** — the FLP failure-detector limit)
  while a paired vantage that reaches the node's side always can (**0.0**). The probe is the §5.4
  cheap-localized oracle mode and the deterministic substrate the deferred RSSM belief (§6.2) must
  roll forward under partition ([`ed12.py`](../../src/verisim/experiments/ed12.py),
  [`ed12.png`](../../figures/ed12.png), [`test_ed12`](../../tests/test_ed12.py),
  [`test_dist_observe`](../../tests/test_dist_observe.py)). **◐ the learned arm ships** (what
  ED1-learned is to ED1): the flat DS4 `M_θ` (trained exactly as ED2-learned) re-points ED12 onto a
  *real* error distribution ([`ed12_learned.py`](../../src/verisim/experiments/ed12_learned.py),
  [`ed12_learned.png`](../../figures/ed12_learned.png), [`test_ed12_learned`](../../tests/test_ed12_learned.py),
  torch extra). Free-running, the structural `bit ≤ observable` dominance holds on every rollout but
  the absolute horizons are small (the flat free-runner's low floor — bit 0.50, observable 0.50,
  consistency 0.62, directional). The clean signal is the **teacher-forced per-step accuracy**, free
  of derailing: the model predicts each delta from the *true* state and its correct-rate rises across
  the projections — **bit 0.15 ≤ observable 0.20 ≤ consistency 0.37** — quantifying which of a real
  model's per-step errors each projection forgives (the probe forgives the unobservable in-flight
  medium; consistency additionally forgives node placement). The partial-observation analogue of
  ED6-two-oracle's teacher-forced decision-sufficiency, on the same flat `M_θ`.
- **ED13 — causal consistency: the effect-before-cause anomaly** (the DS0-increment-5 experiment for
  §3.4). The third `CONSISTENCY_MODELS` end, **`causal`**, between `eventual` (weakest) and
  `linearizable` (strongest): `eventual`'s async replication plus the guarantee that *no replica
  observes an effect before its cause*, implemented as a delivery-order refinement (each `Message`
  carries a `deps` version-vector slice; `advance` defers it until the destination has applied those
  dependencies). A partition-toggle routes the effect `y` to observer `n2` while its cause `x` is
  still blocked. Two findings, dependency-free: **(a)** the effect-before-cause anomaly rate is
  **1.0 under `eventual`** (greedy delivery shows `y` before `x`) and **0.0 under `causal`** (the `y`
  message is held for `x@1`); **(b)** `causal` is a *minimal* refinement — it holds the *dependent*
  message but never the *independent* one (ordering only causally-linked writes, concurrency
  preserved), and after `heal`+`advance` reaches the **identical** durable state `eventual` does
  (in-flight drains to 0). The additive `deps` field is omitted from the canonical form when empty,
  so all prior goldens/hashes are byte-for-byte unchanged. **The DS0-increment-6 causal Tier-B closes
  it:** ED13 now reports that the autonomous-actor system oracle reproduces causal delivery
  **bit-for-bit** under the seed-shuffled scheduler (a stronger W1 test — Tier-B's fixed-point delivers
  exactly the causally-ready closure independent of the shuffle), pinned over a driver battery with
  the broken-arrival control still caught ([`ed13.py`](../../src/verisim/experiments/ed13.py),
  [`ed13.png`](../../figures/ed13.png), [`test_ed13`](../../tests/test_ed13.py),
  [`test_dist_causal`](../../tests/test_dist_causal.py), [`test_dist_system`](../../tests/test_dist_system.py)).
- **ED14 — the quorum (Raft-subset) consensus model: availability frontier + split-brain prevention**
  (the DS0-increment-7 experiment for §3.4). `quorum` is the realistic CP middle — synchronous commit
  to a reachable **majority**, reject when no majority is reachable, async catch-up to the minority. On
  a 5-node cluster (majority = 3): **(a)** the **availability frontier** — sweeping the writing
  partition side's size `k`, a write commits at all `k` under `eventual`, iff `k≥3` under `quorum` (the
  step at the majority threshold), and at *no* `k<5` under `linearizable` (it needs every replica); so
  `quorum` stays available on the majority side exactly where `linearizable` goes dark; **(b)**
  **split-brain prevention** — when both sides write the same key, `eventual` forks (rate **1.0**),
  while `quorum` and `linearizable` never do (**0.0**) — but only `quorum` is *also* available, the
  reason real systems use majority quorums. The `quorum` value is purely additive (every prior
  golden/hash unchanged), and Tier-A ≡ Tier-B bit-for-bit ([`ed14.py`](../../src/verisim/experiments/ed14.py),
  [`ed14.png`](../../figures/ed14.png), [`test_ed14`](../../tests/test_ed14.py),
  [`test_dist_quorum`](../../tests/test_dist_quorum.py)).
- **ED15 — optimistic (OCC) vs pessimistic (2PL) concurrency control** (the DS0-increment-8 experiment
  for §3.2). The `concurrency_control` dial adds the `DD-D3` alternative: **`2pl`**, strict two-phase
  locking with **deterministic wound-wait** (the older txn preempts the younger; no blocking, no
  scheduler — deadlock-free and deterministic). Both reach the *same* serializable guarantee by
  opposite routes (OCC validates the read-set late, 2PL locks it early), so ED15 measures *when each
  pays for a conflict*: both forbid write skew (rate **0.0**), but their **wasted work** differs —
  under OCC an aborted txn validated at commit and wasted *all* **3.0** of its data ops, while under
  2PL it failed fast at the conflicting lock (**2.0** ops). The lock table is purely additive (every
  prior golden/hash unchanged), and **Tier-B reproduces 2PL bit-for-bit** (the txn bookkeeping is
  coordinator-local, so it delegates to the same `txn_step`) ([`ed15.py`](../../src/verisim/experiments/ed15.py),
  [`ed15.png`](../../figures/ed15.png), [`test_ed15`](../../tests/test_ed15.py),
  [`test_dist_2pl`](../../tests/test_dist_2pl.py)).
- **ED16 — read-committed isolation: the lost-update anomaly + the price of preventing it** (the
  DS0-increment-9 experiment for §3.2). The third `TXN_ISOLATION_LEVELS` end, **`read_committed`** —
  the real-world default of Postgres/Oracle/SQL-Server, the *weakest* level (weaker = harder to
  predict, §3.4) — between `snapshot` and read-uncommitted: it does **no** commit-time concurrency
  validation (`validation_set = ()`). Reads still see only committed data (the MVCC `tget` gives no
  dirty reads), but with no write-write check two same-key read-modify-write txns both commit and the
  later silently overwrites the earlier — the classic **lost-update** anomaly snapshot's
  first-committer-wins prevents. Two findings, dependency-free: **(a)** the lost-update anomaly rate
  is **1.0 under `read_committed`** (both RMW txns commit, the earlier write gone) and **0.0 under
  `snapshot` and `serializable`** (the second committer aborts on the same-key write-write conflict);
  **(b)** the price it sells correctness for — under read-modify-write contention `read_committed`
  **never aborts** (`0.00` vs `~0.53` for both validating levels), the apparent throughput it buys by
  admitting the lost updates of (a). The empty validation set is purely additive (every prior
  golden/hash unchanged), and **Tier-B reproduces it bit-for-bit** (txn bookkeeping is
  coordinator-local). **Elle recovers lost update black-box** as a `{ww, rw}` G2 cycle — the same-key
  overwrite (`ww`) plus the stale read (`rw`), distinct from write-skew's pure `{rw}` cycle (ED10) —
  pinned by a lost-update golden ([`ed16.py`](../../src/verisim/experiments/ed16.py),
  [`ed16.png`](../../figures/ed16.png), [`test_ed16`](../../tests/test_ed16.py),
  [`test_dist_read_committed`](../../tests/test_dist_read_committed.py),
  lost-update + Elle goldens in [`test_dist_goldens`](../../tests/test_dist_goldens.py)).
- **ED17 — read-uncommitted isolation: the dirty-read anomaly + its black-box recovery** (the
  DS0-increment-10 experiment for §3.2). The fourth `TXN_ISOLATION_LEVELS` end, **`read_uncommitted`**
  — the *weakest* level, completing the standard SQL hierarchy `read_uncommitted ⊂ read_committed ⊂
  snapshot ⊂ serializable`. It drops even `read_committed`'s last guarantee: an OCC `tget` may observe
  another *active* transaction's **uncommitted** buffered write (the most-recent by lexicographic txn
  id — a deterministic stand-in for "the latest uncommitted writer"), so if that writer later
  **aborts**, the reader saw a value that never committed — the classic **dirty-read** anomaly (Adya
  G1a). The commit path is identical to `read_committed` (`validation_set = ()`); only the read path
  changes, and **only under OCC** (2PL's exclusive lock blocks any reader from seeing an uncommitted
  write — locking gives serializability regardless of the declared level). Purely additive (a new
  `txn_isolation` value), so every prior golden/hash/tokenization is unchanged. Two findings,
  dependency-free: **(a)** the dirty-read anomaly rate is **1.0 under `read_uncommitted`** (the reader
  sees the doomed write) and **0.0 under the three stronger levels** (MVCC reads see only committed
  data), Tier-B agreeing on every scenario; **(b)** **Elle recovers the dirty read black-box** — the
  §5.3 value oracle reconstructs it from the client-visible history alone as a `dirty-read` recovery
  anomaly (a value no committed txn ever appended), matching the oracle on every scenario
  ([`ed17.py`](../../src/verisim/experiments/ed17.py), [`ed17.png`](../../figures/ed17.png),
  [`test_ed17`](../../tests/test_ed17.py),
  [`test_dist_read_uncommitted`](../../tests/test_dist_read_uncommitted.py), dirty-read + Elle goldens
  in [`test_dist_goldens`](../../tests/test_dist_goldens.py)).
- **ED18 — message loss: the broken-convergence anomaly + the lost write only a newer write heals**
  (the DS0-increment-11 experiment for §3.2 / §3.4). The **`drop`** fault — `drop src dst` loses every
  in-flight replication message from `src` to `dst` — wires up the unreliable-network `BUGGIFY`
  primitive (§2.1, §3.2) the delta vocabulary already anticipated (`MsgDrop`) but no action produced.
  Unlike `partition` (which *holds* a message, delivered once the link `heal`s), `drop` **destroys**
  it. Two findings, dependency-free, on a 3-node cluster: **(a)** drop **breaks convergence where
  partition recovers** — cut a write off from a peer, then `advance`+`heal`+`advance`: under
  `partition` the held message is delivered on heal (convergence rate **1.0**), under `drop` the
  destroyed message never is (rate **0.0**, the peer permanently stale). Same symptom — a stale
  replica — from two media: a recoverable delay and an unrecoverable loss, the eventual-consistency
  convergence guarantee's hidden premise (reliable-if-delayed delivery) made visible. **(b)** **only a
  newer write heals a dropped write** — after the drop, `heal`+`advance` alone never repairs the
  replica (rate **0.0**), but a subsequent write to the same key (a higher MVCC version) does (rate
  **1.0**), and the dropped value is **never observed** by the peer (a lost update at the network
  layer, recoverable only by being superseded). `drop` is purely additive (it adds no state field, so
  every prior golden/hash/tokenization is unchanged) and composes with every consistency model;
  Tier-B reproduces the drop and the broken/repaired convergence bit-for-bit
  ([`ed18.py`](../../src/verisim/experiments/ed18.py), [`ed18.png`](../../figures/ed18.png),
  [`test_ed18`](../../tests/test_ed18.py), [`test_dist_drop`](../../tests/test_dist_drop.py), a drop
  golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). *Deferred: model/tokenizer +
  data-driver coverage of `drop` (as with the transaction family, the DS4 flat arm covers only the
  base KV+fault world), and the message-level `delay`/`reorder` faults.*
- **ED19 — anti-entropy / read-repair: convergence restored after message loss** (the
  DS0-increment-12 experiment for §3.2 / §5.1). The **`anti_entropy`** protocol op — the first of the
  §3.2 protocol/admin family — is the **read-repair** mechanism real eventually-consistent stores
  (Dynamo, Cassandra) use to converge *despite* lost messages, and the §4 `ReplicaConverge` op the
  spec named but had not implemented. `anti_entropy node` pulls each object to the winning
  `(version, value)` among the node's **reachable** replicas (a per-object `ReplicaWrite`, reusing the
  existing edit — no new state field, so every prior golden/hash/tokenization is unchanged). It is the
  counterpart to ED18's `drop`: where message loss *breaks* the convergence guarantee, anti-entropy
  *restores* it without a fresh write. Two findings, dependency-free, on a 3-node cluster: **(a)**
  anti-entropy **repairs a dropped write where advance cannot** — after a write is dropped to a peer
  and the link heals, `advance` alone never recovers the stale replica (rate **0.0** — no in-flight
  message remains) while `anti_entropy` pulls the latest directly and recovers it (rate **1.0**), with
  no new write; **(b)** anti-entropy is **bounded by reachability** — the *same* op repairs nothing
  while the peer is partitioned away (rate **0.0**, it cannot cross the split) and repairs once the
  partition heals (rate **1.0**): it converges only the reachable set, gossip not magic, so full
  convergence still needs the network back. Tier-A and Tier-B agree bit-for-bit over the read-repair
  battery, and the tiered oracle was taught to accept `anti_entropy` (its multi-version read-repair
  jump is admissible — the `cycle`/`symbolic` tiers defer it to bit-exact) and `drop`
  ([`ed19.py`](../../src/verisim/experiments/ed19.py), [`ed19.png`](../../figures/ed19.png),
  [`test_ed19`](../../tests/test_ed19.py),
  [`test_dist_anti_entropy`](../../tests/test_dist_anti_entropy.py), an anti-entropy golden in
  [`test_dist_goldens`](../../tests/test_dist_goldens.py)). *Deferred: the multi-node atomic
  `ReplicaConverge` form, cluster-wide gossip rounds, and model/tokenizer/data-driver coverage of
  `anti_entropy`.*

**External harness (optional, for community legibility).** Where cheap, ED1/ED3/ED5 are
additionally reported against **τ-bench / AppWorld**-class stateful-backend agent tasks
(§2.6) and against a **Jepsen**-style consistency test suite — gyms the distributed-systems
and agent communities already trust — with the speedup-vs-fidelity Pareto (model alone →
model + budgeted tiered oracle → full oracle) the way DST tools report bugs-per-CPU-hour and
learned simulators report speedup vs ns-3. Defensive tasks only (§15).

---

## 13. Milestones (DS0–DS8)

SPEC-7 is the buildable expansion of the distributed/service concern implicit in SPEC.md
§11's in-scope list ("web applications, APIs, key-value stores") and the integration of
SPEC-5 and SPEC-6. The `DS` series is to the distributed world what `M0–M8` were to the
filesystem, `NW0–NW8` to the network, and `HC0–HC8` to the host: deterministic core first
(DS0–DS3, **no runtime deps, no GPU**), learned model after. It does not collide with
`M0–M8`, `S1–S6`, `AR0–AR5`, `NW0–NW8`, or `HC0–HC8`.

| Milestone | What | Gate |
|---|---|---|
| **DS0** | Distributed env: event-sourced/replicated `State` (worldify-style log + happens-before), action grammar (client/protocol/fault), canonical serialization + **Tier-A reference DES** (replicated KV + Raft-subset + txn/lock table, embedding SPEC-6 hosts / SPEC-5 net) + `docs/distributed-semantics.md` + golden trajectories | property tests + goldens — **◐ increment 1 shipped**: the **replicated KV under partition** core ([`dist/`](../../src/verisim/dist/), [`distoracle/`](../../src/verisim/distoracle/)) — `DistributedState` (replicas + causal log + in-flight messages + partition/crash/clock), the client (`put`/`get`/`cas`) + fault/time (`advance`/`partition`/`heal`/`crash`/`restart`) grammar, the Tier-A async-replication DES with eventual-consistency LWW, canonical serialization, [`docs/distributed-semantics.md`](../distributed-semantics.md), and golden trajectories pinning stale-read-under-partition + convergence ([`test_dist_core`](../../tests/test_dist_core.py), [`test_dist_goldens`](../../tests/test_dist_goldens.py); dependency-free, GPU-free). **◐ a second consistency model ships** for the H20 sweep: **`linearizable`** (synchronous all-replica writes, CP write-rejection under partition, so no replica is ever stale and there is no in-flight medium — [`docs/distributed-semantics.md` §2.1](../distributed-semantics.md), goldens pinning synchronous replication + partitioned-write rejection). **◐ increment 2 ships — the multi-key transaction core** ([`dist/txn.py`](../../src/verisim/dist/txn.py)): the `begin`/`tget`/`tput`/`commit`/`abort` grammar under **optimistic concurrency control** (OCC, first-committer-wins; `DD-D3`) — a coordinator buffers a transaction's read-set (pinning `(key, version)` per first read) and write-buffer, and `commit` validates the read-set (aborting `conflict` if any read key's version changed) then applies every buffered write atomically (an MVCC bump + replication through the same in-flight medium as `put`, so it inherits the consistency model: async under `eventual`, synchronous-or-`unavailable` under `linearizable`). OCC is chosen over 2PL because it is *deterministic and deadlock-free* (no lock table / acquisition order / victim selection), the discipline the deterministic core pins first. Shared by Tier-A and Tier-B; the transaction state is purely additive (an empty `txns` set is omitted from the canonical form, so DS0-incr-1 goldens/hashes are unchanged). Goldens pin atomic-commit-replicates and conflict-aborts ([`test_dist_goldens`](../../tests/test_dist_goldens.py)); semantics, OCC conflict, `apply==oracle`, Tier-B agreement, and serialization backward-compat are pinned in [`test_dist_txn`](../../tests/test_dist_txn.py); the OCC commit/abort frontier tracks the balls-in-bins occupancy law in **ED8** ([`experiments/ed8.py`](../../src/verisim/experiments/ed8.py), [`ed8.png`](../../figures/ed8.png), [`ed8.csv`](../../figures/ed8.csv)). The `transactional` data-driver preset emits contended OCC workloads ([`distdata/drivers.py`](../../src/verisim/distdata/drivers.py)). **◐ increment 3 ships — the two transaction isolation levels** (`txn_isolation ∈ {serializable, snapshot}`, `DD-D4`): `serializable` validates the **read-set** (OCC backward validation — forbids write skew), `snapshot` validates only **write-write** conflicts (admits write skew). The write version is pinned at first `tput` (`TxnState.write_versions`) exactly as the read version is pinned at first `tget`; the validation set is the only difference. **ED9** ([`experiments/ed9.py`](../../src/verisim/experiments/ed9.py), [`ed9.png`](../../figures/ed9.png), [`ed9.csv`](../../figures/ed9.csv)) exhibits the classic **write-skew** anomaly (rate **1.0 under `snapshot`, 0.0 under `serializable`**) and **the price of serializability** (under read-heavy contention `serializable` aborts strictly more — `0.70` vs `0.55`, disjoint CIs), both composing with Tier-B; a golden pins the anomaly outcomes ([`test_dist_goldens`](../../tests/test_dist_goldens.py), [`test_ed9`](../../tests/test_ed9.py)). **◐ increment 5 ships — the `causal` consistency model** (the *middle* of the §3.4 curriculum, between `eventual` and `linearizable`): `eventual`'s async, available-under-partition replication plus **one guarantee** — if write `B` causally depends on write `A`, no replica observes `B` before `A`. Implemented as a **delivery-order refinement**, not a new write path: each `Message` carries a `deps` slice of the writing node's version vector (the `(object, version)` it had observed for *other* objects), and `advance` defers a message until the destination has applied those dependencies (held, not lost; delivered once the cause arrives). `deps` is empty under `eventual` / `linearizable` and **omitted from the canonical form when empty**, so every prior golden/hash/tokenization is byte-for-byte unchanged ([`dist/state.py`](../../src/verisim/dist/state.py), [`dist/delta.py`](../../src/verisim/dist/delta.py), [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py)). **ED13** ([`experiments/ed13.py`](../../src/verisim/experiments/ed13.py), [`ed13.png`](../../figures/ed13.png), [`ed13.csv`](../../figures/ed13.csv)) pins the **effect-before-cause anomaly** (a partition-toggle routes the effect `y` to observer `n2` while its cause `x` is still blocked): rate **1.0 under `eventual`, 0.0 under `causal`**; that `causal` holds the *dependent* message but never the *independent* one (it orders only causally-linked writes, leaving concurrency free); and that **convergence is preserved** (after `heal`+`advance`, `eventual` ≡ `causal` durable state, in-flight drains to 0) — a delivery-order refinement, not a different outcome ([`test_dist_causal`](../../tests/test_dist_causal.py), [`test_ed13`](../../tests/test_ed13.py), causal golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 6 ships — the causal Tier-B** ([`distoracle/system.py`](../../src/verisim/distoracle/system.py)): the autonomous-actor **system oracle** now honors causal delivery, extending the W1 retirement (ED7) to the third consistency model — and it is a *stronger* test than `eventual`'s, because the seed-shuffled scheduler may try a message before its cause, so Tier-B's `_advance` runs delivery to a **fixed point** (repeatedly deliver any message whose `deps` are satisfied at the destination actor, read from the actor's *own* replicas, until a pass delivers nothing). The fixed point delivers exactly the causally-ready closure *independent of the shuffle*, reproducing Tier-A's sorted-order result (msg ids are topologically ordered). Both oracles attach deps via the **shared `causal_deps` helper** (so they cannot drift), and the differential's observable channel now includes `deps`. **ED13 now reports the Tier-B agreement in-line** (mirroring ED8/ED9): Tier-A and Tier-B agree **bit-for-bit** over the causal scenarios, and a 1080+-step driver battery + the broken-arrival negative control are pinned in [`test_dist_system`](../../tests/test_dist_system.py). **◐ increment 7 ships — the `quorum` (Raft-subset) consensus model** ([`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), [`distoracle/system.py`](../../src/verisim/distoracle/system.py)): the realistic CP middle real consensus protocols occupy — a write commits **synchronously to a reachable majority** and is **rejected** only when a majority is *not* reachable, the unreachable minority catching up asynchronously. So `quorum` is **strictly more available than `linearizable`** (which needs *every* replica): under a minority partition it rejects, but under a *majority* partition it keeps serving the majority side, where `linearizable` goes completely dark — yet it is still **divergence-free** (only one side can hold the majority, so the object never forks). **ED14** ([`experiments/ed14.py`](../../src/verisim/experiments/ed14.py), [`ed14.png`](../../figures/ed14.png), [`ed14.csv`](../../figures/ed14.csv)) plots the **availability frontier** (the quorum commit-rate steps from 0 to 1 exactly at the majority threshold `k≥3` of 5, where `eventual` is flat-available and `linearizable` flat-unavailable) and **split-brain prevention** (when both partition sides write the same key: `eventual` forks at rate **1.0**, `quorum` and `linearizable` at **0.0** — but only `quorum` is *also* available). The `quorum` value is purely additive (no new state), so all prior goldens/hashes are byte-for-byte unchanged, and **Tier-A and Tier-B agree bit-for-bit** over a driver battery + the availability/split-brain scenarios ([`test_dist_quorum`](../../tests/test_dist_quorum.py), [`test_ed14`](../../tests/test_ed14.py), quorum golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 8 ships — lock-based 2PL** (the `concurrency_control ∈ {occ, 2pl}` dial, the `DD-D3` alternative; [`dist/txn.py`](../../src/verisim/dist/txn.py)): the pessimistic counterpart to OCC — **strict two-phase locking with deterministic wound-wait**. `tget`/`tput` acquire shared/exclusive locks held to commit, and a conflict is resolved by **wound-wait** (the *older* txn — lexicographically smaller id — preempts the younger; a younger requester aborts rather than waiting), so it is **deterministic and deadlock-free without a scheduler** — the deterministic 2PL the core can pin (`DD-D3` deferred the *blocking* 2PL, whose victim selection injects nondeterminism). The lock table (`DistributedState.locks`) is purely additive — empty and **omitted from the canonical form** under the `occ` default, so all prior goldens/hashes are unchanged. **ED15** ([`experiments/ed15.py`](../../src/verisim/experiments/ed15.py), [`ed15.png`](../../figures/ed15.png), [`ed15.csv`](../../figures/ed15.csv)) measures the optimistic/pessimistic tradeoff: both forbid write skew (serializable, rate 0.0), but their **wasted work** differs — OCC validates at commit, so an aborted txn wasted *all* its operations (**3.0** data ops/abort), while 2PL fails fast at the conflicting lock (**2.0** ops). Transaction bookkeeping is coordinator-local, so **Tier-B reproduces 2PL bit-for-bit** by delegating to the same `txn_step` ([`test_dist_2pl`](../../tests/test_dist_2pl.py), [`test_ed15`](../../tests/test_ed15.py), 2PL golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 9 ships — the `read_committed` isolation level** (`txn_isolation ∈ {serializable, snapshot, read_committed}`, `DD-D4`; [`dist/txn.py`](../../src/verisim/dist/txn.py)): the *weakest* level — the real-world default of Postgres/Oracle/SQL-Server — does **no** commit-time validation (`validation_set = ()`). Reads still see only committed data (the MVCC `tget` gives no dirty reads), but with no write-write check two same-key read-modify-write txns both commit and the later overwrites the earlier — the classic **lost-update** anomaly snapshot's first-committer-wins prevents. The new level is purely additive (the default config still serializes `txn_isolation="serializable"`), so all prior goldens/hashes are unchanged. **ED16** ([`experiments/ed16.py`](../../src/verisim/experiments/ed16.py), [`ed16.png`](../../figures/ed16.png), [`ed16.csv`](../../figures/ed16.csv)) exhibits the **lost-update** anomaly (rate **1.0 under `read_committed`, 0.0 under `snapshot` and `serializable`**) and **the price of preventing it** (under read-modify-write contention `read_committed` *never aborts*, `0.00` vs `~0.53` — the throughput it sells correctness for), both composing with Tier-B bit-for-bit; **Elle recovers lost update black-box** as a `{ww, rw}` G2 cycle (distinct from write-skew's pure `{rw}`), and lost-update + Elle goldens pin it ([`test_ed16`](../../tests/test_ed16.py), [`test_dist_read_committed`](../../tests/test_dist_read_committed.py), [`test_dist_goldens`](../../tests/test_dist_goldens.py)). *Fixed in the build:* the transaction commit's replication only handled `eventual`/`linearizable` (the `quorum` model added in incr 7 fell through to eventual-style async); the commit now replicates under the *same* discipline as a plain `put` across all four models. **◐ increment 10 ships — the `read_uncommitted` isolation level** (`txn_isolation ∈ {serializable, snapshot, read_committed, read_uncommitted}`, completing the standard SQL hierarchy `read_uncommitted ⊂ read_committed ⊂ snapshot ⊂ serializable`, `DD-D4`; [`dist/txn.py`](../../src/verisim/dist/txn.py)): the *weakest* level drops even `read_committed`'s last guarantee — a `tget` may observe another active transaction's **uncommitted** buffered write (the most-recent by lexicographic txn id, a deterministic stand-in for "the latest uncommitted writer"), so if that writer later **aborts** the reader saw a value that never committed — the classic **dirty-read** anomaly (Adya G1a). The commit path is identical to `read_committed` (`validation_set = ()`); only the read path changes, and **only under OCC** — 2PL's exclusive lock blocks any reader from seeing an uncommitted write (locking gives serializability regardless of the declared level). Purely additive (a new `txn_isolation` value), so all prior goldens/hashes/tokenization are unchanged. **ED17** ([`experiments/ed17.py`](../../src/verisim/experiments/ed17.py), [`ed17.png`](../../figures/ed17.png), [`ed17.csv`](../../figures/ed17.csv)) exhibits the **dirty-read** anomaly (rate **1.0 under `read_uncommitted`, 0.0 under the three stronger levels**, Tier-B agreeing on every scenario) and shows **Elle recovers the dirty read black-box** — the §5.3 value oracle reconstructs it from the client-visible history alone as a `dirty-read` recovery anomaly (a value no committed txn appended), matching the oracle on every scenario; dirty-read + Elle goldens pin it ([`test_ed17`](../../tests/test_ed17.py), [`test_dist_read_uncommitted`](../../tests/test_dist_read_uncommitted.py), [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 11 ships — the `drop` fault** (`drop src dst`, the unreliable-network `BUGGIFY` primitive; [`dist/action.py`](../../src/verisim/dist/action.py), [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), [`distoracle/system.py`](../../src/verisim/distoracle/system.py)): message loss — `drop src dst` destroys every in-flight replication message from `src` to `dst`. The delta vocabulary already carried `MsgDrop` (with `apply` + serialization + the `<msg_drop>` token) but **no action produced it**; this wires it up — the §3.2 fault op named but deferred since increment 1. Unlike `partition` (which *holds* a message until the link `heal`s), `drop` **destroys** it, so the destination replica permanently misses the write — the eventual-consistency convergence guarantee broken by an unreliable network until a *newer* write overwrites it. `drop` adds **no new state field** (it only removes in-flight messages), so every prior golden/hash/tokenization is byte-for-byte unchanged, and it composes with every consistency model. **ED18** ([`experiments/ed18.py`](../../src/verisim/experiments/ed18.py), [`ed18.png`](../../figures/ed18.png), [`ed18.csv`](../../figures/ed18.csv)) exhibits **(a)** drop **breaks convergence where partition recovers** (post-`heal`+`advance` convergence rate **1.0 under `partition`, 0.0 under `drop`** — a recoverable delay vs an unrecoverable loss, same stale-replica symptom) and **(b)** **only a newer write heals a dropped write** (`heal`+`advance` repair rate **0.0**, a subsequent higher-version write **1.0**; the dropped value is **never observed** by the peer — a lost update at the network layer). Tier-A and Tier-B agree bit-for-bit over the drop + broken/repaired-convergence battery ([`test_dist_drop`](../../tests/test_dist_drop.py), [`test_ed18`](../../tests/test_ed18.py), a drop golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 12 ships — anti-entropy / read-repair** (`anti_entropy node`, the first **protocol** op + the §4 `ReplicaConverge`; [`dist/action.py`](../../src/verisim/dist/action.py), [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), [`distoracle/system.py`](../../src/verisim/distoracle/system.py)): the **read-repair** convergence mechanism real eventually-consistent stores (Dynamo, Cassandra) use to converge *despite* lost messages — a node pulls each object to the winning `(version, value)` among its **reachable** replicas (a per-object `ReplicaWrite`, reusing the existing edit, so **no new state field** and every prior golden/hash/tokenization is byte-for-byte unchanged). The counterpart to ED18's `drop`: where message loss *breaks* the convergence guarantee, anti-entropy *restores* it without a fresh write. **ED19** ([`experiments/ed19.py`](../../src/verisim/experiments/ed19.py), [`ed19.png`](../../figures/ed19.png), [`ed19.csv`](../../figures/ed19.csv)) shows **(a)** anti-entropy **repairs a dropped write where advance cannot** (post-heal convergence rate **1.0 under `anti_entropy` vs 0.0 advance-only** — read-repair needs no in-flight message) and **(b)** it is **bounded by reachability** (repair rate **0.0 while the peer is partitioned away, 1.0 after heal** — it converges only the reachable set). Tier-A ≡ Tier-B bit-for-bit, and the tiered oracle gained correct handling of `anti_entropy` (its multi-version read-repair jump is admissible; `cycle`/`symbolic` defer it to bit-exact) and `drop` ([`test_dist_anti_entropy`](../../tests/test_dist_anti_entropy.py), [`test_ed19`](../../tests/test_ed19.py), an anti-entropy golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 13 ships — the `delay` / `reorder` message-timing faults** (`delay src dst dt`, `reorder src dst`; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `timing_fault_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py)): the message-timing half of the §3.4 medium ("partition, crash, message loss, **reorder**, clock skew"), the last fault SPEC named but deferred since increment 1. `delay` defers every in-flight `src`→`dst` message by `dt` (a *recoverable* delay — the write still arrives, just later, the counterpart to `drop`'s unrecoverable loss); `reorder` reverses that channel's delivery schedule (the multiset of times preserved, the order flipped). Both edit only the existing `Message.deliver_after` via a new `MsgReschedule` edit, so they add **no new state field** and every prior golden/hash/tokenization is byte-for-byte unchanged, and — being pure *medium* changes — Tier-A and Tier-B compute byte-identical deltas through the **shared `timing_fault_edits` helper** (so they cannot drift, exactly as `causal_deps` is shared). **ED20** ([`experiments/ed20.py`](../../src/verisim/experiments/ed20.py), [`ed20.png`](../../figures/ed20.png), [`ed20.csv`](../../figures/ed20.csv)) shows **(a)** `delay` **is recoverable where `drop` is not** (post-`advance` convergence rate **1.0 under `delay` vs 0.0 under `drop`** — completing ED18's two-media contrast: the convergence guarantee assumes delivery is *reliable if delayed*, and `delay` exercises exactly that premise) and **(b)** `reorder` **flips the in-transit observation but never the converged value** (with two staggered writes a peer transiently shows the older write after `reorder` vs the newer without it — flip rate **1.0** — yet both converge to the newer write, invariance rate **1.0**: last-writer-wins is a commutative join, so delivery order changes what you catch in flight but not where the cluster lands — the §5.2 order-independence made a *controllable* input). Tier-A ≡ Tier-B bit-for-bit, and the symbolic tier treats `delay`/`reorder` as no-replica-change transitions ([`test_dist_timing`](../../tests/test_dist_timing.py), [`test_ed20`](../../tests/test_ed20.py), delay + reorder goldens in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 14 ships — the `clock_skew` fault** (`clock_skew node δ`; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `clock_skew_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), the `DistributedState.sender_clock` helper): the **last** of the §3.4 medium faults ("partition, crash, message loss, reorder, **clock skew**") — the fault grammar is now complete. `clock_skew node δ` offsets a node's local clock by a signed `δ`, which shifts the `deliver_after` it stamps on **every** message it sends (a positive offset = a clock running ahead, defers its sends; negative = behind, rushes them). It adds one omitted-when-empty `skew` map (no per-message state), so a synchronized cluster is byte-identical to the pre-increment-14 form, and — being a pure medium change — Tier-A and Tier-B compute byte-identical deltas. **ED21** ([`experiments/ed21.py`](../../src/verisim/experiments/ed21.py), [`ed21.png`](../../figures/ed21.png), [`ed21.csv`](../../figures/ed21.csv)) shows **(a)** skew is a **persistent per-node timing shift** (the `deliver_after` of a node's sends moves by exactly `δ`, every send — so a positively-skewed node's writes are deferred past a short `advance`, unlike the one-shot `delay`) and **(b)** convergence is **clock-independent** (sweeping the writer's skew leaves the converged state byte-identical, invariance rate **1.0**): because the protocol resolves conflicts by last-writer-wins on `(version, value)` — never on a wall-clock timestamp — skew shifts *when* a write is delivered but never *which* write wins. This is exactly the property deterministic-simulation testing (FoundationDB, madsim) injects clock skew to verify; a timestamp-LWW store would diverge, version-LWW cannot be fooled. Tier-A ≡ Tier-B bit-for-bit, and the symbolic tier treats `clock_skew` as a no-replica-change transition ([`test_dist_clock_skew`](../../tests/test_dist_clock_skew.py), [`test_ed21`](../../tests/test_ed21.py), two clock-skew goldens in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 15 ships — the `gossip` protocol op** (`gossip a b`; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `gossip_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py)): the **pairwise, bidirectional** anti-entropy the §4 `ReplicaConverge` named — the Merkle-tree sync real eventually-consistent stores (Dynamo, Cassandra) run between *pairs* of nodes, vs `anti_entropy`'s one-directional pull-to-one-node. `gossip a b` reconciles **both** `a` and `b` to the per-object winner of their two replicas (a `ReplicaWrite` for whichever is behind), needs a live link (both up + connected), and is bounded by reachability. It reuses `ReplicaWrite` (no new state field) and is a pure coordinator-level reconciliation, so Tier-A and Tier-B compute byte-identical deltas through the shared helper. **ED22** ([`experiments/ed22.py`](../../src/verisim/experiments/ed22.py), [`ed22.png`](../../figures/ed22.png), [`ed22.csv`](../../figures/ed22.csv)) shows **(a)** one pairwise gossip reconciles **both** endpoints (with `a` stale on `x` and `b` stale on `y`, a single `gossip a b` fills both holes, where a single `anti_entropy a` fills only `a`'s — the one-directional vs bidirectional distinction, the reason real systems run pairwise anti-entropy) and **(b)** a chain of pairwise gossips converges the **whole reachable component epidemically** (a write dropped to every peer spreads hop-by-hop to convergence rate **1.0**, where `anti_entropy` would need every node to pull), bounded by reachability (a node partitioned off the chain stays stale). Tier-A ≡ Tier-B bit-for-bit, and the `cycle`/`symbolic` tiers defer `gossip`'s multi-version jump to bit-exact (like `anti_entropy`) ([`test_dist_gossip`](../../tests/test_dist_gossip.py), [`test_ed22`](../../tests/test_ed22.py), a gossip golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 16 ships — the `elect` / `propose` consensus core** (the Raft-subset third action family; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `elect_edits`/`propose_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), the `ProtocolStep` edit + `term`/`leader` state fields): `elect node` makes a node leader **iff its partition side holds a strict majority of the *live* cluster**, bumping a monotone `term` — so two disjoint sides can never both elect (**no split-brain**, and an even split is leaderless rather than forked); `propose node key val` is a **leader-fenced** majority write that commits only for the *current* leader, so a leader deposed by a higher-term election cannot commit **even after the partition heals** — the Raft **leader-completeness** safety property a leaderless `quorum` write lacks. The `term`/`leader` fields are omitted from the canonical form until the first election, so the family is purely additive (no prior golden/hash/tokenization change), and it touches no replica, so Tier-A ≡ Tier-B bit-for-bit with metamorphic term-monotone/known-leader invariants. **ED23** ([`experiments/ed23.py`](../../src/verisim/experiments/ed23.py), [`ed23.png`](../../figures/ed23.png), [`ed23.csv`](../../figures/ed23.csv)) pins **(a)** no split-brain (minority blocked / majority elects / even-split leaderless, all **1.0**) and **(b)** term-fencing (a deposed leader's `propose` fenced after heal **1.0** where a plain `put` by the same stale coordinator still commits — the control **1.0**, and the new leader commits **1.0**) ([`test_dist_consensus`](../../tests/test_dist_consensus.py), [`test_ed23`](../../tests/test_ed23.py), a consensus golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 17 ships — the `step_down` op** (`step_down node`, the leadership lifecycle's graceful close; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `step_down_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py)): the *current* leader voluntarily relinquishes power, leaving the cluster **leaderless at the same term** (the voluntary counterpart to ED23's higher-term deposition) — reusing the `ProtocolStep` edit with `leader → None`, so **no new state field** and every prior golden/hash is unchanged, Tier-A ≡ Tier-B bit-for-bit. **ED24** ([`experiments/ed24.py`](../../src/verisim/experiments/ed24.py), [`ed24.png`](../../figures/ed24.png), [`ed24.csv`](../../figures/ed24.csv)) pins **(a)** the handoff lifecycle (after `step_down` the same node's `propose` is `not_leader` — **no leaderless commit window** — and a fresh `elect` of a successor lands at a strictly higher term and commits, all **1.0**) and **(b)** authority + partition-independence (only the current leader may relinquish — a non-leader / second `step_down` is a no-op reject, **1.0**; and a **minority-stranded leader can still step down** where its `propose` is `no_quorum` — relinquishing needs no quorum, exercising power does, both **1.0**) ([`test_dist_consensus`](../../tests/test_dist_consensus.py), [`test_ed24`](../../tests/test_ed24.py), a step-down golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 18 ships — the `lease`/`lread` leader-lease** (the Raft read optimization; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `lease_edits`/`lread_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), the `LeaseSet` edit + `lease_until` state field): `lease node dt` lets the current leader take a read lease through global clock `+ dt`; `lread node key` then serves a **local linearizable read with no quorum round-trip** while the lease holds — so a leader partitioned into the minority can still `lread` locally where its `propose` is `no_quorum`. The safety tension: a fresh `elect` is fenced `lease_held` until the incumbent's lease expires (leadership cannot change hands under a live lease — what makes the local read safe), while a voluntary `step_down` **releases** the lease for an immediate handoff (vs a crashed leader the cluster must outlast). The `lease_until` field is omitted from the canonical form until the first `lease`, so the family is purely additive (no prior golden/hash change), and it touches no replica, so Tier-A ≡ Tier-B bit-for-bit. **ED25** ([`experiments/ed25.py`](../../src/verisim/experiments/ed25.py), [`ed25.png`](../../figures/ed25.png), [`ed25.csv`](../../figures/ed25.csv)) pins **(a)** local reads without a quorum (valid-lease `lread` / minority-leader `lread` / its `propose` `no_quorum` control / expired-lease rejection, all **1.0**) and **(b)** the lease/election tension (`elect` fenced `lease_held` under a live lease and unblocked past expiry; `step_down` releases the lease for a no-wait handoff, all **1.0**) ([`test_dist_consensus`](../../tests/test_dist_consensus.py), [`test_ed25`](../../tests/test_ed25.py), a lease golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 19 ships — the `append` replicated log** (Raft log-matching / log replication; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `append_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), the `LogSet`/`CommitIndexSet` edits + per-node `logs` + `commit_index` state): `append node key val` appends a `(term, index, key, value)` entry to the leader's log and replicates it to the reachable followers, who **adopt the leader's prefix — overwriting any divergent uncommitted tail** (the log-matching reconciliation). It commits the entry (advancing the monotone `commit_index` and folding the committed prefix into the KV state machine, **backfilling a rejoined follower**) **iff a majority holds it**; a minority-stranded leader still appends locally (`uncommitted`) but does not commit, so its entry is never applied to the KV and is **overwritten by a higher-term leader's entry at the same index** — the log-matching safety the one-shot `propose` could not express. The `logs`/`commit_index` are omitted from the canonical form until the first `append`, so the op is purely additive (no prior golden/hash change), the metamorphic tier gains a commit-index-monotone invariant, and `append` defers to bit-exact in the cheap tiers (its committed-log fold can jump a rejoined replica several versions). **ED26** ([`experiments/ed26.py`](../../src/verisim/experiments/ed26.py), [`ed26.png`](../../figures/ed26.png), [`ed26.csv`](../../figures/ed26.csv)) pins **(a)** commit-requires-a-majority (a majority append commits / a minority append is uncommitted-but-retained / commit index monotone, all **1.0**) and **(b)** log-matching reconciliation (an uncommitted entry is never applied to the KV / a deposed leader's uncommitted tail is overwritten after heal / all live logs identical / the rejoined KV converges, all **1.0**) ([`test_dist_consensus`](../../tests/test_dist_consensus.py), [`test_ed26`](../../tests/test_ed26.py), a log golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 20 ships — the `add_replica`/`remove_replica` membership change** (the §3.2 admin ops; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `add_replica_edits`/`remove_replica_edits` + the `active_members` helper in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), the `MemberSet` edit + `members` state field): they reconfigure the consensus *voting set* (a leader-committed change), so the **majority threshold tracks the membership** — every quorum computation (`elect`/`propose`/`append`) routes through `active_members`, so `remove_replica` shrinks the cluster (a smaller majority suffices) and `add_replica` grows it. The active leader cannot be removed (the `is_leader` fence — step it down first), the last member is protected, and a change requires a current leader. All config nodes still physically store replicas; `members` is the voting overlay, omitted from the canonical form until the first change (the empty set is the "all nodes vote" sentinel), so the ops are purely additive (no prior golden/hash change), and they touch no replica — Tier-A ≡ Tier-B bit-for-bit, with a metamorphic members-subset invariant. **ED27** ([`experiments/ed27.py`](../../src/verisim/experiments/ed27.py), [`ed27.png`](../../figures/ed27.png), [`ed27.csv`](../../figures/ed27.csv)) pins **(a)** the quorum threshold tracks the membership (a lone leader blocked at full membership / committing as the sole member / re-blocked after `add_replica`, all **1.0**) and **(b)** restore-availability (a lone survivor of a 3-node cluster stuck, then committing after removing the 2 dead; the active leader fenced from removal, all **1.0**) ([`test_dist_consensus`](../../tests/test_dist_consensus.py), [`test_ed27`](../../tests/test_ed27.py), a membership golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 21 ships — the `enqueue`/`dequeue` distributed FIFO queue** (the §3.2 client ops, a *second data type* beside the KV store; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `enqueue_edits`/`dequeue_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), the `QueueSet` edit + `queues` state field): `enqueue node queue val` appends to a replicated FIFO queue and `dequeue node queue` pops the coordinator's head, with **delivery semantics that follow the `consistency_model`** — under `eventual` a dequeue's head-removal reaches only the reachable side, so a partitioned peer re-delivers the same item (at-least-once / duplicate), where `quorum`/`linearizable` gate availability for exactly-once. The `queues` map is omitted from the canonical form until the first `enqueue` (purely additive, no prior golden/hash change), queue ops touch no replica, and queues are now part of the observable `cluster_view`, so Tier-A ≡ Tier-B bit-for-bit. **ED28** ([`experiments/ed28.py`](../../src/verisim/experiments/ed28.py), [`ed28.png`](../../figures/ed28.png), [`ed28.csv`](../../figures/ed28.csv)) pins **(a)** the delivery count stepping `2 → 1 → 0` across `eventual`/`quorum`/`linearizable` under a partition (duplicate / exactly-once-on-majority / CP-unavailable) and **(b)** FIFO + exactly-once on the connected path (all **1.0**) ([`test_dist_queue`](../../tests/test_dist_queue.py), [`test_ed28`](../../tests/test_ed28.py), a queue golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 22 ships — the `deploy` rolling-upgrade op** (the §3.2 admin op answering SPEC-7's headline *"will this deploy break the cluster?"*; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `deploy_edits` + the `_compatible` helper in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), the `VersionSet` edit + `versions` state field + the `max_version_skew` config dial): `deploy node version` sets a node's running software version, and every consensus quorum (`elect`/`propose`/`append`) routes through `_compatible` so two nodes interoperate only if their versions are within `max_version_skew` (the N-1 window, default 1). A rolling upgrade that stays inside the window keeps quorum; a deploy that creates an **incompatible version split with no compatible majority** loses quorum — the deploy broke the cluster. Compatibility gates *consensus* only (the best-effort KV/queue data plane is version-agnostic); the `versions` map is omitted from the canonical form until the first `deploy` and `max_version_skew` is omitted from the config hash at its default, so the op is purely additive (no prior golden/hash change), it touches no replica, and versions join the observable `cluster_view` — Tier-A ≡ Tier-B bit-for-bit, with a metamorphic known-node/non-negative-version invariant. **ED29** ([`experiments/ed29.py`](../../src/verisim/experiments/ed29.py), [`ed29.png`](../../figures/ed29.png), [`ed29.csv`](../../figures/ed29.csv)) pins **(a)** the safe rolling upgrade (a propose commits at every step of a `v0 → v1` roll, **1.0**) and **(b)** the break + diagnostic (an incompatible split with no compatible majority is `no_quorum`, while the same shape at a smaller spread or under a wider window commits, all **1.0**) ([`test_dist_deploy`](../../tests/test_dist_deploy.py), [`test_ed29`](../../tests/test_ed29.py), a deploy golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 23 ships — the `host` embedded-host op** (the compositional vision §3.1/§4 names since increment 1; [`dist/action.py`](../../src/verisim/dist/action.py), the shared `host_op_edits` in [`distoracle/reference.py`](../../src/verisim/distoracle/reference.py), the `HostStep` edit + per-node `hosts` state field): each cluster node runs a real **SPEC-6 host** (process table + per-process fd tables + an embedded v0 filesystem), and `host node <syscall>` (`fork`/`exit`/`setuid`/`open`/`write`/`close`) delegates to the SPEC-6 `ReferenceHostOracle` on that node's own host, wrapping its bundle delta in a `HostStep` edit (the §4 `HostDelta`). Per-node isolated (a `fork` on one node never touches another's host); host ops respect the node's up/down status (a crashed node's host is `unavailable` — the cross-layer crash linkage) and the host state survives a crash. The `hosts` map is omitted from the canonical form until the first `host` op, so the op is purely additive (no prior golden/hash change), it touches no replica, and hosts join the observable `cluster_view` — Tier-A ≡ Tier-B bit-for-bit. **ED30** ([`experiments/ed30.py`](../../src/verisim/experiments/ed30.py), [`ed30.png`](../../figures/ed30.png), [`ed30.csv`](../../figures/ed30.csv)) pins **(a)** composition + isolation (a fork is node-local, KV + host coexist on one node, open+write reaches the embedded FS — all **1.0**) and **(b)** the crash linkage (a host op on a crashed node is unavailable, restart restores it, the host state survives — all **1.0**) ([`test_dist_host`](../../tests/test_dist_host.py), [`test_ed30`](../../tests/test_ed30.py), a host golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). *Deferred to later DS0 increments: the embedded SPEC-5 net inside each node (the host is now embedded; the SPEC-5 network is the remaining subsystem), the broader SPEC-6 syscall surface (sockets/IPC/scheduler — only the HC0 subset is wired here), and the model/tokenizer/data-driver coverage of the fault/protocol/consensus/queue/deploy/host ops (the DS4 flat arm covers only the base KV+fault world).* |
| **DS1** | Log/replica `Delta` types, compositional `apply` (reusing SPEC-2/5/6 `apply` for embedded subsystems), delta↔serialization; the `apply == oracle` invariant | invariant tests — **◐ shipped for the DS0-increment-1 slice**: the `DistDelta` edit vocabulary (`ReplicaWrite`/`MsgSend`/`MsgDeliver`/`EventAppend`/`PartitionSet`/`NodeDown`/…), `apply` as a pure function, delta↔serialization round-trips, and the `apply(state, oracle.delta) == oracle.next_state` invariant tested on every transition ([`dist/delta.py`](../../src/verisim/dist/delta.py)). **◐ extended for DS0 increment 2** with the transaction edits `TxnSet`/`TxnDel` (upsert/remove an active transaction's buffered state), keeping `apply == oracle` exact across the `begin`/`tget`/`tput`/`commit`/`abort` family. The embedded host/net delta composition arrives with their embedding (later DS0 increment). |
| **DS2** | Drivers (workload + seeded fault injection = `BUGGIFY`; topology/replication/consistency generators), trajectory JSONL, manifests/splits, the **fault-intensity / partition-entropy dials** | data tests — **◐ shipped for the DS0-increment-1 world** ([`distdata/`](../../src/verisim/distdata/)): the seeded `DistDriver` (`uniform`/`contention`/`adversarial`) interleaving client ops + `advance` + faults, with the **explicit `fault_prob` (fault-intensity) and `partition_bias` (partition-entropy) dials** the H20/H21 sweeps need; trajectory JSONL + regenerable dataset manifests with disjoint trajectory-level splits; tested for valid-action/`apply==oracle`, determinism, dial monotonicity, and preset distinctness ([`test_dist_data`](../../tests/test_dist_data.py)). Extends with the consensus/transaction ops as DS0 grows. |
| **DS3** | Consistency-faithfulness (Elle-style cycle detection), consistency/composed divergence `d`, `H_ε`, per-tier bits-to-correct, run-record schema; the **tiered-oracle interface** (`O[tier]`) with metamorphic + cycle + symbolic + bit-exact tiers | metric + oracle-tier tests — **◐ the metric core AND the tiered oracle ship**. *Metrics* ([`distmetrics/`](../../src/verisim/distmetrics/)): the **live-cluster divergence** `d(s, ŝ)` (a normalized fact-set difference over replicas + in-flight + partition/crash/clock, feeding the generic `faithful_horizon` so distributed `H_ε(ρ)` is defined exactly as in every prior world), the **headline-new consistency-faithfulness** (§9.1 — the fraction of objects whose converged/split *consistency view* the model predicts right, which catches a model that mispredicts a partition split as converged), and **bits-to-correct / delta-exact** over the `DistDelta` ([`test_dist_metrics`](../../tests/test_dist_metrics.py)). *The tiered oracle — SPEC-7's payload (§5, DD-D1)* ([`distoracle/tiers.py`](../../src/verisim/distoracle/tiers.py)): the four-tier menu (**metamorphic** ¢1 → **cycle** ¢2 → **symbolic** ¢4 → **bit-exact** ¢16) with `cheapest_refutation` spending the cheapest tier that can refute a prediction, the cumulative oracle-dollar cost recorded — every error class caught at its right tier, and a subtle invariant-respecting error caught only by bit-exact (the non-redundancy H17 measures, [`test_dist_tiers`](../../tests/test_dist_tiers.py)). **◐ increment 2 ships — the Elle-style cross-object cycle detection** ([`distoracle/elle.py`](../../src/verisim/distoracle/elle.py)): the stronger-consistency, over-a-history sibling of the per-step `cycle` tier (which is the eventual-consistency form). The distributed analogue of Jepsen's **Elle** (Kingsbury & Alvaro, VLDB 2020) — given only the **observable transaction history** (what each committed transaction read and wrote, with the store's MVCC versions; no oracle, no cluster state), it reconstructs Adya's **Direct Serialization Graph** (`ww`/`wr`/`rw` edges) and reports a violation iff the DSG has a cycle, classified by Adya's G-hierarchy (`G0` dirty write / `G1c` circular information flow / `G2` anti-dependency cycle). **ED10** ([`experiments/ed10.py`](../../src/verisim/experiments/ed10.py), [`ed10.png`](../../figures/ed10.png), [`ed10.csv`](../../figures/ed10.csv)) shows it **recovers the ED9 write-skew anomaly black-box** — a `G2` cycle `A →rw B →rw A` at rate **1.0 under `snapshot`, 0.0 under `serializable`**, matching ED9's oracle-side commit-count on every scenario — and **certifies the `serializable` level** (0.0 of contended histories non-serializable vs 0.60 [0.30, 0.90] under `snapshot`): a free, reference-free verifier that agrees with the expensive oracle on the serializability question it is built to answer ([`test_elle`](../../tests/test_elle.py), [`test_ed10`](../../tests/test_ed10.py), and a black-box DSG golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 3 ships — the version oracle (list-append / value-recoverable histories)**: ED10 was black-box about *reads and writes* but still let the store hand Elle the integer MVCC version each transaction read and installed (`collect_history` peeked at `state.replicas[(k, node)].version`). That is the one cooperation Jepsen's Elle removes — over a **list-append** register (every write appends a globally-unique value, every read returns the whole list) the per-key version order is **recoverable from the read values themselves** (Kingsbury & Alvaro 2020, the "version oracle"): a read returning `[x, y, z]` is direct testimony that the append of `x` preceded `y` preceded `z`, with no question put to the store. [`recover_versions`](../../src/verisim/distoracle/elle.py) merges every observed read-list for a key (each a *prefix* of the one growing append log) into a single total order; [`check_serializable_appends`](../../src/verisim/distoracle/elle.py) assigns each appended value its recovered version and reuses the *unchanged* DSG/cycle machinery. **ED11** ([`experiments/ed11.py`](../../src/verisim/experiments/ed11.py), [`ed11.png`](../../figures/ed11.png), [`ed11.csv`](../../figures/ed11.csv)) shows two things: (a) **the version oracle is sound** — recovering versions from values alone reproduces the store's *exact* version history (`recovery_sound` true on every scenario), so the G2 write-skew rate is ED10's (**1.0 under `snapshot`, 0.0 under `serializable`**) recovered with *zero* store cooperation; and (b) **the split-brain fork only value-recovery can represent** — when a partition lets two sides extend one key divergently (a later read sees `[a, b]`, another `[a, c]`, neither a prefix of the other) the version oracle reports an **`incompatible-order`** anomaly (rate **1.0**, clean control **0.0**), the black-box signature of split-brain — the §9.1 consistency anomaly caught reference-free from the client history, which ED10's integer-version mode is *structurally unable* to express. Two further recovery anomalies surface before any cycle search: **`dirty-read`** (Adya G1a — a read of an uncommitted value) and **`duplicate-write`** ([`test_elle`](../../tests/test_elle.py), [`test_ed11`](../../tests/test_ed11.py), value-recovered goldens in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). **◐ increment 4 ships — partial observation, the probe projection (§5.4, DD-D2)**: every prior metric compared the *full* cluster state, but W7 says no observer ever holds one. [`dist/observe.py`](../../src/verisim/dist/observe.py) makes the epistemic limit deterministic — `observe(state, vantage)` projects a `DistributedState` onto the `Observation` an observer connected to a set of `vantage` nodes can obtain: replicas on *reachable* (up + co-partitioned) nodes only, **never the in-flight replication medium**, and every other node labelled `unreachable` *with no reason attached* — so a crashed node and a partitioned-away node project identically. [`observable_divergence`](../../src/verisim/distmetrics/observe.py) is the §5.4 **probe (cheap, localized)** oracle mode: identical to the bit-exact `divergence` when the vantage reaches the whole cluster, in-flight-forgiving under partition; because a bit-faithful step is necessarily observably faithful, the **observable horizon dominates the bit horizon** structurally. **ED12** ([`experiments/ed12.py`](../../src/verisim/experiments/ed12.py), [`ed12.png`](../../figures/ed12.png), [`ed12.csv`](../../figures/ed12.csv)) measures two findings dependency-free: (a) **the probe-faithful horizon outlasts the bit-faithful one for `subtle` (in-flight) errors** (free-running probe gap **+9.0 steps**, disjoint CI [4.0, 16.1]) and coincides for `gross` (durable-replica) errors — the partial-observation form of H19/ED5, read through *physical observability* rather than the consistency-view abstraction; and (b) **crash and partition are indistinguishable from one vantage** (a single external probe sees them as byte-identical, indistinguishable rate **1.0** — the failure-detector limit behind FLP) **but separable from two** (a paired vantage reaching the node's side exposes the live isolated replica, rate **0.0**) — one probe cannot localize a fault, a quorum can ([`test_dist_observe`](../../tests/test_dist_observe.py), [`test_ed12`](../../tests/test_ed12.py), and a probe-projection golden in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). This is the deterministic substrate the (deferred) RSSM belief (§6.2) must roll forward under partition. *Deferred: the run-record schema for the DS5 loop; the RSSM belief that predicts the full state from the observable one.* |
| **DS4** | `M_θ`: service-graph message-passing + RSSM belief (+ optional SSM carry), constrained delta decode, supervised training (SLM-sized) | model tests (torch extra) — **◐ increment 1 shipped — the dependency-free serialization foundation** ([`distmodel/`](../../src/verisim/distmodel/)): the closed token [`DistVocab`](../../src/verisim/distmodel/vocab.py) (specials + structure markers + the 10 delta ops + 8 commands + 5 result statuses + the node/object/value leaf pools + a single bounded **integer pool** `<int:0..max_int>` that closes the one unbounded family — the monotone bookkeeping counters `version`/`msg_id`/`deliver_after`/`clock` — the host's `max_pid`/`max_fd` trick), and the bidirectional [`tokenizer`](../../src/verisim/distmodel/tokenizer.py) mapping `<bos> state action <gen>` → `Δ <eos>` with an **exact inverse `parse_target`**. The design move that makes the distributed delta tokenizable: the causal-log `EventAppend` (whose `happens_before` is the one genuinely variable-length field) is encoded as a bare `<event_append>` marker and **reconstructed deterministically from `(state, action)`** on parse — `id`/`node`/`op`/`clock`/`happens_before` are all pure functions of the step context, exactly as the network tokenizer omits the always-1 `ClockAdvance` amount, so the unbounded list never enters the token grammar. The serialization module files stay torch-free if imported directly, so they remain in the dependency-free core; the **round-trip `parse(encode(Δ)) == Δ`** is tested exhaustively over every preset × 6 seeds × 40 steps (full edit-vocabulary coverage), on a 5-node/3-object cluster, on a multi-group partition, and the decoded delta is shown to still satisfy the M1 invariant `apply(state, Δ) == oracle.next_state` ([`test_dist_model`](../../tests/test_dist_model.py)). **◐ increment 2 shipped — the learned (flat) arm** ([`grammar.py`](../../src/verisim/distmodel/grammar.py), [`decode.py`](../../src/verisim/distmodel/decode.py), [`world_model.py`](../../src/verisim/distmodel/world_model.py), [`dataset.py`](../../src/verisim/distmodel/dataset.py)): the LL(1) constrained-decode `DistDeltaGrammar` (the distributed analogue of v0's `DeltaGrammar`, carrying two structured nonterminals the flat net/host grammars do not need — the **nested partition run** `<pgroup> NODE+ … <pgroups_end>` and the **status-typed result** where `advanced` is followed by an int and every other status by a value), the `NeuralDistWorldModel` over v0's `GPT` (a drop-in `DistModel` for the DS5 loop, with a `predict_delta_with_uncertainty` decode-entropy signal for `π_c`), and the supervised dataset builders feeding the generic `verisim.train` trainers. A **structural-bug fix found by free-running decode**: an untrained model could emit `<event_append>` after a non-client action (whose `args[0]` is not a coordinator node), so the decoder now masks `<event_append>` out of the top-level op set for fault/time ops — the one op whose reconstruction reads `action.args[0]`, kept to the client-op context the oracle's language actually produces (§5.1). Tested ([`test_dist_model_decode`](../../tests/test_dist_model_decode.py), torch extra): constrained decode is **grammar-valid from an untrained model** across the partition/advance/put shapes, a tiny cluster **overfits to <0.05 loss** and free-runs the training deltas back (each still satisfying the M1 invariant), the model **satisfies the `DistModel` loop protocol**, and decode is config-driven on a 5-node/3-object cluster. *Deferred: the service-graph message-passing + RSSM-belief arm (the structured `M_θ`, SPEC-7 §6.1-6.2) — under full observability it degenerates to this flat Markov predictor (§6.2), so it lands with the partial-observation work; supervised dataset JSONL/manifest persistence.* |
| **DS5** | Tiered propose-verify-correct loop with `π_c` × `π_w` (when × which-tier), consistency/belief operators, experience-stream scaffolding, baselines | loop invariants — **✅ the tiered loop ships** ([`distloop/`](../../src/verisim/distloop/)): the model-agnostic runner over any `DistModel` (with `DistNullModel`/`DistOracleBackedModel` baselines), the **`π_w` which-tier axis** ([`tier_policy.py`](../../src/verisim/distloop/tier_policy.py): `FixedTierPolicy` + the cheapest-refutation `EscalatingTierPolicy`), and the **oracle-dollar accounting** — each consult spends its tier's cost, a refutation adds the bit-exact correction cost, and a prediction the tier cannot refute is *trusted*; the run-record carries the divergence trajectory (→ `H_ε`) **and** the cumulative oracle-dollars (→ H17). Loop invariants tested ([`test_dist_loop`](../../tests/test_dist_loop.py)): ρ=1 reproduces truth (`H_ε=T`), the perfect model never drifts at ρ=0 spending $0, the null model drifts at step 0, the budget is spent exactly, and the oracle-dollar reflects the tier policy (escalation pays the cheap tiers before bit-exact when errors are caught late — the genuine H17 nuance). The §8.3 **correction operators `C`** now ship too ([`operator.py`](../../src/verisim/distloop/operator.py): `HardReset` (default) + `Residual`/`Projection` diagnostics + the partial `ReplicasOnlyCorrection`), wired into the runner via the `operator` parameter and the `π_c` uncertainty signal via `_predict`/`DistUncertaintyModel`. *Deferred: the experience stream; online correction-teaches-the-stream (H7).* |
| **DS6** | **ED1 distributed `H_ε(ρ)` curve** + the **tiered-oracle measurement (H17)** + competitive-ratio fit (H18) + bootstrap-CI aggregation + figure | **the prime directive** — **◐ the apparatus + the first distributed curve ship** ([`experiments/ed1.py`](../../src/verisim/experiments/ed1.py), [`ed1_dist.png`](../../figures/ed1_dist.png), [`ed1_dist.csv`](../../figures/ed1_dist.csv)): the distributed **`H_ε(ρ)` curve** (floor 0.2 at ρ=0 → ceiling 40 at ρ=1, bootstrap-CI over seeds — the standard prime-directive shape, comparable to v0/EN1/EH1) **and the H17 tiered-oracle measurement** — oracle-dollar *per faithful step* for each fixed tier × proposer error class. **H17 verdict (apparatus, on a controlled error distribution):** *whether a cheap tier buys more faithful horizon per oracle-dollar depends on where the model's errors fall.* For **gross** (out-of-vocab) errors the metamorphic tier is cheaper per faithful step ($9.4) than always-bit-exact ($16); for **subtle** (in-flight) errors the cheap tiers miss the drift entirely (H≈0, $848/step) and bit-exact is the only efficient choice. Run on a synthetic tunable-noise proposer ([`DistNoisyModel`](../../src/verisim/distloop/model.py)) before the learned `M_θ` (DS4) supplies a *real* error distribution; the loop + tiered-oracle + oracle-dollar machinery is exercised and the H17 tradeoff is exact ([`test_ed1`](../../tests/test_ed1.py)). **◐ the learned-model curve ships** ([`experiments/ed1_learned.py`](../../src/verisim/experiments/ed1_learned.py), [`ed1_learned.png`](../../figures/ed1_learned.png), [`ed1_learned.csv`](../../figures/ed1_learned.csv)): the flat DS4 `M_θ` trained on seeded rollouts and run through the *same* tiered loop, so the curve and H17 are measured on a **real** error distribution. The curve is the same **floor→cliff** (floor 0.2 at ρ=0 → ceiling 32 at ρ=1, in-distribution eval — the EN1/EH1 step for this world). The **real-model H17 finding, and it is the honest inverse of the synthetic one**: the constrained decoder (DS4 incr 2) removes the *gross* (out-of-vocab) error class by construction, so the learned model's residual errors are *subtle* — the cheap **metamorphic** tier catches none of them (H=0.2, **$624/faithful-step**), **symbolic** few (H=0.8, $411), and only **bit_exact** is efficient (H=32, **$16**); the cheapest-refutation **escalate** policy reaches full horizon but pays **more** ($21.6) because a real model's errors need the bit-exact correction anyway. *So a cheap tier helps exactly when a model makes catchable-cheaply errors — and a grammar-constrained learned model, by design, does not; the tiered oracle's value is model-dependent, reported not assumed* ([`test_ed1_learned`](../../tests/test_ed1_learned.py), torch extra). *Deferred: the competitive-ratio fit (H18).* |
| **DS7** | Smart `π_w` + consistency operators + drift mitigations + the **consistency-level (H20) and fault-injection (H21) sweeps**; ED2/ED3/ED4 (equal-dollar-budget, CIs) | comparison figures — **◐ the equal-dollar-budget ED2 (H17/H18) ships** ([`experiments/ed2.py`](../../src/verisim/experiments/ed2.py), [`ed2.png`](../../figures/ed2.png), [`ed2.csv`](../../figures/ed2.csv)): the **faithful-horizon-vs-oracle-dollar frontier** per tier policy (`metamorphic`/`symbolic`/`bit_exact` fixed + the cheapest-refutation `escalate`) on the synthetic proposer, dependency-free and GPU-free. Where ED1 reported cost *per faithful step at ρ=1*, ED2 sweeps ρ and compares policies **at a matched dollar budget** — interpolating each policy's horizon along its Pareto envelope (a true equal-*dollar*, not equal-ρ, comparison) — and reads the **H18 competitive ratio** off the same frontier at the sub-linear quarter budget `B/4`. **H17 in budget form, confirmed mode-dependently:** at `B/4` the metamorphic tier buys **H=14.2 vs bit-exact's 4.2** for **gross** (cheaply-catchable) errors (tiering wins, ratio 0.36 of the full-truth ceiling at ¼ the cost) but is flat at the floor (**H=1.5 vs 4.2**) for **subtle** (bit-exact-only) errors, where even `escalate` *loses* to single-tier bit-exact — H17's honest negative reported, not hidden ([`test_ed2`](../../tests/test_ed2.py)). **◐ the H21 / fault-injection arm of ED4 ships** ([`experiments/ed4_fault.py`](../../src/verisim/experiments/ed4_fault.py), [`ed4_fault.png`](../../figures/ed4_fault.png), [`ed4_fault.csv`](../../figures/ed4_fault.csv)): the **DST/BUGGIFY data-factory lesson**, made measurable by the DS2 driver's `fault_prob` dial — train two DS4 `M_θ` of **equal volume**, one fault-free (`fault_prob=0`) and one fault-injected, then sweep the eval workload's fault-intensity **free-running** (ρ=0 exposes the model, not the loop). **H21 confirmed, with the sharpest possible control:** at zero eval-fault the two coincide, but as faults intensify the **fault-injected** model holds ~3× more free-run faithful horizon (0.375 vs 0.125 steps) — *even though the fault-free model is the **better** clean predictor* (teacher-forced accuracy 0.60 vs 0.49). The fault-free model never saw a partition/crash/heal, so under fault it derails immediately; fault injection buys robustness factual data cannot — validating DST as a *data factory*, not just a test harness. A bonus in-figure instance of the program's proxy/truth divergence: higher per-token clean accuracy, lower compounding free-run horizon. The dataset builders gained the `fault_prob`/`partition_bias` dials the H20/H21 axes need ([`distmodel/dataset.py`](../../src/verisim/distmodel/dataset.py), [`test_ed4_fault`](../../tests/test_ed4_fault.py), torch extra). **◐ the learned-`M_θ` equal-dollar arm of ED2 ships** ([`experiments/ed2_learned.py`](../../src/verisim/experiments/ed2_learned.py), [`ed2_learned.png`](../../figures/ed2_learned.png), [`ed2_learned.csv`](../../figures/ed2_learned.csv)): ED2's synthetic frontier re-pointed at the **real** flat DS4 `M_θ` (trained exactly as ED1-learned), so the equal-dollar H17/H18 question is answered on a *real* error distribution — what ED1-learned is to ED1, this is to ED2. **The finding is the honest inverse of ED2's `gross` panel, in budget form, and the budget-form of ED1-learned's per-step H17:** the constrained decoder removes the gross (out-of-vocab) error class by construction, so a real model lives entirely in ED2's `subtle` regime — at the sub-linear quarter budget `B/4=$128` the cheap tiers stay flat at the floor (**metamorphic H=0.2, symbolic H=0.8**) while only **bit_exact buys horizon (H=2.0)**, and the cheapest-refutation **escalate** policy *loses* to single-tier bit-exact (H=1.6, and at every ρ it spends strictly more — `$691 vs $512` to reach the same H=32 ceiling — because it pays the cheap probes before the bit-exact correction a real model's subtle errors always need). The **H18 competitive ratio at `B/4` is just 0.06** of the full-truth ceiling: for a grammar-constrained learned model a sub-linear budget buys little horizon however the tiers are sliced — H17/H18's honest negative for the real model, *reported not assumed* ([`test_ed2_learned`](../../tests/test_ed2_learned.py), torch extra). *So the tiered oracle's value is model-dependent: a cheap tier helps exactly when a model makes catchable-cheaply errors, and a grammar-constrained learned model, by design, does not.* **◐ the `π_c` "smart-when" arm of ED2 ships** ([`experiments/ed2_smart.py`](../../src/verisim/experiments/ed2_smart.py), [`ed2_smart.png`](../../figures/ed2_smart.png), [`ed2_smart.csv`](../../figures/ed2_smart.csv)): the missing *when* axis of ED2 — at a fixed interior budget `ρ`, does spending the consults on the steps the flat `M_θ` is least sure about (its constrained-decode entropy) beat spreading them evenly? The DS5 runner gained the uncertainty plumbing it had been missing — a `_predict` helper + a `DistUncertaintyModel` protocol that feed the per-step decode entropy into the loop's `StepContext`, exactly as the network/host runners already do (the genuine gap this increment closed). **H9 — the standing H2/H9 negative carried into the distributed world, and sharper than a tie:** entropy-gated consultation does **not** beat `fixed` — it is strictly *worse* (lift **0.08–0.12×** at every budget), because faithful horizon is a *prefix* property and `fixed` consults at step 0 to protect the prefix while the entropy signal spends late and lets the model derail early. The flat decode-entropy signal is a decode-time artifact, not a calibrated belief; this localizes the smart-`π_c` lever to the (deferred) structured `M_θ`'s RSSM belief variance — the EH2 lesson (the host's factored belief-variance beat fixed ~2.2× where flat entropy could not), now the flat-arm baseline the distributed structured arm must beat ([`test_ed2_smart`](../../tests/test_ed2_smart.py), torch extra). **◐ ED3 — the correction operators ship, and the distributed world breaks v0's operator identity** ([`experiments/ed3.py`](../../src/verisim/experiments/ed3.py), [`ed3.png`](../../figures/ed3.png), [`ed3.csv`](../../figures/ed3.csv); [`distloop/operator.py`](../../src/verisim/distloop/operator.py)): the DS5 runner gained the §8.3 correction-operator axis it was missing (`HardReset` default + `Residual`/`Projection` diagnostics + the partial `ReplicasOnlyCorrection`). v0 proved an identity — a full-truth consult makes `hard_reset`/`residual`/`projection` behaviorally identical on `H_ε` — and ED3 shows the **distributed world breaks it, mode-dependently**, because the partial `ReplicasOnlyCorrection` snaps the durable replicas but **trusts the model's predicted in-flight** (the stale-read medium, the `subtle` error class §5). For **gross** (replica-write) errors all four operators recover the same horizon (H=7.2, identity holds); for **subtle** (in-flight) errors the three full-correction operators hold the identity (H=6.2) but `ReplicasOnlyCorrection` **collapses to H=1.8** (gap 4.5) — the in-flight medium is the distributed world's hidden state a partial correction cannot see, tied to the same gross/subtle structure H17 turns on ([`test_ed3`](../../tests/test_ed3.py), dependency-free). **◐ the consistency-level H20 arm of ED4 ships** ([`experiments/ed4_consistency.py`](../../src/verisim/experiments/ed4_consistency.py), [`ed4_consistency.png`](../../figures/ed4_consistency.png)): the `CONSISTENCY_MODELS` axis (§3.4) gains its first strong end — **`linearizable`** (synchronous all-replica writes, CP write-rejection under partition, no in-flight medium) — and the sweep resolves H20 *through* H19: the consistency-vs-bit gap is exclusively a *weak*-consistency phenomenon (it needs the consistency-invisible in-flight medium), measuring **+10.5 under `eventual` / in-flight rate 3.2 vs 0 under `linearizable` / in-flight rate 0**, gross control 0 at both ([`test_ed4_consistency`](../../tests/test_ed4_consistency.py), dependency-free). **◐ the absolute-predictability learned arm of ED4 ships** ([`experiments/ed4_consistency_learned.py`](../../src/verisim/experiments/ed4_consistency_learned.py), [`ed4_consistency_learned.png`](../../figures/ed4_consistency_learned.png), [`ed4_consistency_learned.csv`](../../figures/ed4_consistency_learned.csv), torch extra): the synthetic arm reports only the *gap* (the absolute horizon at equal noise is confounded by per-level delta composition), so this trains **one flat `M_θ` per consistency level** (same init seed, only the world differs) and measures the free-running (ρ=0) horizon. **H20 confirmed in direction:** the model free-runs **~2.4× further under `linearizable` (bit H=1.4) than `eventual` (0.6)** — strong consistency is more predictable (less hidden state) — though the absolute horizons are small (a weak flat free-runner, ED1-learned's low floor) so the CIs overlap, directional not disjoint. **And the honest difference from the synthetic arm:** the H19 gap on the *real* model is **positive at both levels** (not the synthetic's clean eventual-only gap), because a real model errs on consistency-invisible *bookkeeping* (clocks/log/partition) present at both levels, not only the in-flight medium the dialed error targets ([`test_ed4_consistency_learned`](../../tests/test_ed4_consistency_learned.py)). *Deferred: the smart-`π_w` (which-tier) scheduling, the consistency-model `projection` operator, and the GNN/RSSM representation arm of ED4 (which the flat-arm smart-`π_c` null motivates).* |
| **DS8** | Consistency-vs-bit horizon + competitive ratio (ED5/H18/H19), counterfactual replay (ED6/H5), **Tier-B system oracle** (madsim/Shadow/Antithesis-class), the **LLM-callable cluster-simulator protocol** (§7), the **verified-contribution protocol** (SPEC-6 §16), Inspect benchmark + `verifiers`-spec distributed RL env, technical report | packaging + report — **◐ ED5 (H18/H19) ships** ([`experiments/ed5.py`](../../src/verisim/experiments/ed5.py), [`ed5.png`](../../figures/ed5.png), [`ed5.csv`](../../figures/ed5.csv)): the §9.1 consistency-faithfulness metric gets its first loop consumer (the DS5 runner now records the consistency-divergence trajectory alongside bit-exact divergence), and ED5 reads both findings off the dependency-free synthetic proposer. **H19 confirmed mode-dependently** — free-running consistency-faithful horizon outlasts bit-faithful for `subtle` in-flight errors (H=13.1 vs 1.5, gap +11.6, disjoint CI; the in-flight message is bit-visible but consistency-invisible until delivery) and coincides for `gross` durable-replica errors (the control). **H18 split** — the competitive-ratio fit across `ρ × prediction error` confirms graceful degradation in the *error* axis (quarter-budget ratio monotone 1.00 → 0.05, trivial bound recovered for a perfect model) but reproduces the floor→cliff *no-knee* negative in the *budget* axis (ratio near the floor at `B/4`, cliff only at ρ→1) — learning-augmented in kind, no free lunch at sub-linear budget ([`test_ed5`](../../tests/test_ed5.py)). **◐ ED6 (H5 counterfactual lift) ships** ([`experiments/ed6.py`](../../src/verisim/experiments/ed6.py), [`ed6.png`](../../figures/ed6.png), [`ed6.csv`](../../figures/ed6.csv)): three matched-count arms train the same flat DS4 `M_θ` — `trajectory` (base light-fault on-policy), `trajectory-more` (5× more on-policy, the volume control), `+counterfactual` (base + free oracle **fault**-flip branches, the §10.1 "re-run from `(seed,t)` with one fault flipped") — then predict **held-out fault interventions**, scored bit-exact (full next cluster state) and by **medium recall** (predicts the partition/crash split-brain, §17 Q7). **H5 confirmed — and the distributed world is where it finally pays, the honest inverse of EN6/EH6:** `+counterfactual` beats **both** the base **and** the matched-volume control on **both** metrics, disjoint CIs (intervention-exact **0.51 vs 0.25 vs 0.06**, medium-recall **0.56 vs 0.22 vs 0.05**), where the network (EN6) and host (EH6/H16) found counterfactual supervision adds nothing over volume. The mechanism is the distributed **medium** (partition/crash/in-flight) — a hidden state the light-fault on-policy distribution underrepresents, so on-policy *volume* buys little (0.06→0.25) while off-policy oracle **fault branches** buy a lot (0.25→0.51): the held-out-intervention analogue of H21. *Honest caveat:* the branches are fault-heavier than the on-policy control, so the lift conflates counterfactual *branching* with the fault *coverage* it carries — but EN6/EH6 found null under the identical design, so the distributed positive is the result; the disentanglement is future work (tied to H21). The matched-*volume* arm is the first distributed experiment to need the minibatched `train_batched` K2 loop (a real perf fix vs the full-batch path the small-dataset learned arms use — [`test_ed6`](../../tests/test_ed6.py), torch extra). **◐ the ED6 two-oracle / H12 slice ships** ([`experiments/ed6_two_oracle.py`](../../src/verisim/experiments/ed6_two_oracle.py), [`ed6_two_oracle.png`](../../figures/ed6_two_oracle.png), [`ed6_two_oracle.csv`](../../figures/ed6_two_oracle.csv)): the distributed analogue of SPEC-5's H12 / SPEC-6's EH6 — the cheap **consistency oracle** (the §9.1 split-brain decision) as a *second oracle* against the full bit-exact one, teacher-forced over the fault-heavy `adversarial` workload on the dependency-free synthetic proposer. **H12 confirmed, mode-dependently:** **non-redundant rate 0** by construction (the consistency view is a pure function of replicas, so a bit-exact-correct prediction is always consistency-correct — *redundant for verification*); **consistency-sufficient rate 1.00 for `subtle` (in-flight) vs 0.00 for `gross` (durable-replica) errors** (disjoint CIs — the per-step teacher-forced form of ED5's H19 horizon gap, tracking the in-flight medium); at a **consult-fact ratio of 0.28** (~3.6× cheaper, the gap widening under fault as the medium inflates the full state but never the consistency view). The consistency oracle is *redundant* but a **cheaper, decision-sufficient** consult for the question an SRE/defender actually asks ([`test_ed6_two_oracle`](../../tests/test_ed6_two_oracle.py)). **◐ the §16 verified-contribution protocol ships** ([`distcontrib/`](../../src/verisim/distcontrib/), [`test_distcontrib.py`](../../tests/test_distcontrib.py)): the dependency-free distributed analogue of the host `contrib/` — `verify_transition`/`verify_trajectory` accept a contributed trace iff re-running the oracle reproduces it, with the distributed-specific **tiered acceptance** (`bit_exact` demands byte-for-byte, the cheap tiers admit any next-state legal under the declared model — the W7 path), trajectory chaining (no splicing), SHA-256 content-addressing, and hostile-input safety. *Fixed in the build:* the `from_canonical(to_canonical(s))` round-trip was non-exact because `partitions` was stored in construction order while serialization sorted it — fixed at the source (canonical partition order in `apply` + `DistributedState.__post_init__`), pinned by a round-trip test ([`test_dist_core`](../../tests/test_dist_core.py)). **◐ the §7 LLM-callable cluster simulator ships** ([`distsim/`](../../src/verisim/distsim/), [`test_distsim.py`](../../tests/test_distsim.py)): the dependency-free distributed analogue of the host `hostsim/` — `DistSimulator.imagine` (oracle-free plan rollout, the cheap draft) + `verify` (against the oracle → a `DistPlanReport`) with the two distributed-specific readouts: a **consistency-faithful plan horizon** distinct from the bit-exact one (ED5/H19 lifted to the plan level — the agent trusts the model's split-brain prediction longer than its byte prediction) and **change-safety as the differential in consistency health** (the securifine pattern: does the plan break consistency, and does the model agree with the oracle on that verdict?), plus a composing `DistGoal` task oracle. Exercised in CI with no torch (the dependency-free baselines satisfy `DistModel`). **◐ the learned-`M_θ` re-pointing of the ED6 two-oracle / H12 slice ships** ([`experiments/ed6_two_oracle_learned.py`](../../src/verisim/experiments/ed6_two_oracle_learned.py), [`ed6_two_oracle_learned.png`](../../figures/ed6_two_oracle_learned.png), [`ed6_two_oracle_learned.csv`](../../figures/ed6_two_oracle_learned.csv)): what ED1-learned is to ED1, this is to the two-oracle slice — train the flat DS4 `M_θ` (exactly as ED2-learned) and run the **same** teacher-forced H12 measurement on its *real* (un-dialled) error distribution. **H12 confirmed on the real model, the honest mirror of ED2-learned read through the other oracle:** non-redundant **0.0** (structural, unchanged), the consistency oracle **decision-sufficient on 0.57 [0.53, 0.61]** of the model's bit-wrong steps at a **consult-fact ratio 0.28 (~3.6× cheaper)** — even though the full prediction is wrong **87%** of the time on the fault-heavy eval. The 0.57 sits **between the synthetic `gross` (0.0) and `subtle` (1.0) poles** because a real error distribution is a *mixture* (predominantly the consistency-invisible in-flight class). The same model, same constrained decoder, **loses as a verifier** (ED2-learned's cheap tiers refute nothing) yet is **decision-sufficient on the majority of errors as a decision oracle** — the clearest single statement that the tiered oracle's value depends on *which question you ask it* ([`test_ed6_two_oracle_learned`](../../tests/test_ed6_two_oracle_learned.py), torch extra). **◐ the distributed world is packaged for reuse** (the DoD §4 deliverable): the **`verifiers`-spec distributed RL env** ([`distrl/`](../../src/verisim/distrl/), [`test_distrl.py`](../../tests/test_distrl.py)) — the distributed analogue of [`hostrl/`](../../src/verisim/hostrl/), a dependency-free reset/step env whose **reward is the tiered oracle's faithfulness verdict** (no learned reward model in the loop — the verifier-as-reward thesis), teacher-forced so the episode **return *is* the faithful horizon `H_ε`**, with the one distributed-specific knob `reward_mode ∈ {bit_exact, consistency}` so an agent can be graded on the §9.1 split-brain *decision* (the SRE/defender's question, which outlasts the bit-exact horizon where the error hides in the in-flight medium — ED5/H19) rather than on bytes; and the **Inspect benchmark** ([`disteval/`](../../src/verisim/disteval/), [`test_disteval.py`](../../tests/test_disteval.py)) — the distributed analogue of [`hosteval/`](../../src/verisim/hosteval/), a framework-agnostic faithfulness benchmark (rollout `score_dist_model` reporting **both** the bit-faithful horizon **and** the consistency-faithful one **and** the tiered **oracle-dollars**, plus single-step labels + divergence graders) with a lazily-imported `inspect_ai` task adapter — the metrology SPEC-7 §1.4 argues the field lacks (Jepsen grades a running system's history, never a simulator's predicted next cluster state). **◐ the DS8 technical report ships** ([`docs/report.md`](../report.md) — the distributed-world section): the honest hypothesis-by-hypothesis write-up the DoD §3 requires, reading the committed numbers off the CSVs for H8 (the floor→cliff carried into the fourth world), H17 (the model-dependent tiered oracle, synthetic + learned), H18 (learning-augmented in the error axis, floor→cliff in the budget axis), H19 (the consistency-vs-bit horizon gap that tracks the in-flight medium), H20 (the gap is a weak-consistency phenomenon), H21 (fault-injected training beats fault-free at equal volume), H5 (the counterfactual-replay positive, the honest inverse of EN6/EH6), and H12 (the redundant-but-decision-sufficient consistency oracle, synthetic + learned) — each with its honest negative and caveat, plus the SPEC-7 reproduce commands and a distributed threat-to-validity note. **◐ Tier-B — the system oracle — ships, and retires W1 for the distributed world** ([`distoracle/system.py`](../../src/verisim/distoracle/system.py), [`distoracle/differential.py`](../../src/verisim/distoracle/differential.py), [`experiments/ed7.py`](../../src/verisim/experiments/ed7.py), [`ed7.png`](../../figures/ed7.png), [`ed7.csv`](../../figures/ed7.csv)): the distributed analogue of the host `SandboxOracle` (SPEC-11). Where Tier-A is a *single-threaded analytic DES* that computes the next cluster state in closed form, `SystemDistOracle` **runs the replicated-KV protocol as a real distributed system** — autonomous **node actors** holding *only their own replicas and an inbox*, exchanging real replication messages with **no global-state access** (the cluster state is emergent, W7 made operational) — driven by a **seeded scheduler** (the madsim/turmoil DST model) whose delivery order is *seed-shuffled* (not Tier-A's sorted order), so agreement certifies the property the analytic DES quietly assumes: the eventual-consistency convergence is **delivery-order-independent** (LWW by `(version, value)` is a commutative join). Two disclosed isolation tiers (the SPEC-11 `process`/`namespaced` split): `simulated` (the default — actors single-stepped by the scheduler) and `threaded` (each actor in a **real OS thread** blocking on a real `queue.Queue`, one message in flight at a time — genuine kernel concurrency, deadlock-free; a tier that cannot run raises `SystemDistOracleUnavailable`, a first-class disclosed skip). The **differential** compares the *observable-cluster channel* (replicas + id-independent in-flight + medium + result; the causal log and id counters excluded as bookkeeping, exactly as the host excludes `last`). **ED7's four-tier finding, the distributed W1 retirement:** across the exhaustive grammar battery and all three workload drivers (including the fault-heavy `adversarial`), Tier-A and Tier-B agree **bit-for-bit 1.000 (residual 0)**; the prime-directive `H_ε(ρ)` curve is **oracle-invariant** (max gap 0 at every ρ — substituting a real distributed execution for the analytic model leaves the curve unchanged); and a **teeth-bearing negative control** (a deliberately-broken *arrival-order* actor whose convergence is order-**dependent**) is **caught** by the differential as the `delivery_order` boundary (the SY3 analog — the harness detects a faithfulness break, not just rubber-stamps the reimplementation). A disclosed reality attestation re-runs the battery on **real OS threads** and reports its 1.000 agreement ([`test_dist_system`](../../tests/test_dist_system.py), [`test_ed7`](../../tests/test_ed7.py); dependency-free, GPU-free). *Deferred: a wrapped external real-binary DST runtime (madsim/Shadow/Antithesis) — the actor runtime is the in-repo, dependency-free realization of the same DST principle.* |

DS0–DS3 + the DS5 loop are the deterministic core. `M_θ` (DS4) drops into the loop via the
same model-agnostic interface every prior world uses. Tier-B, torch, and the LLM client are
optional extras; the deterministic core has no runtime deps.

---

## 14. The autonomous research engine over the distributed world

SPEC-7 is built, as far as possible, with the human at the boundary — extending SPEC-4's
ratchet, not replacing it.

- **The gate is unfakeable, and now per-tier.** You cannot lower bits-to-correct (§5.4) or
  raise consistency-faithfulness (§9.1) without predicting the truth; the per-tier
  decomposition gives the engine a *richer* gradient — it can search for "improve the tier /
  consistency class that is leaking" — while staying ground-truth (SPEC-4 §9). And because
  the DES is the data factory (§2.1), the engine can *generate its own harder curriculum*
  (raise fault intensity, weaken consistency) under the same denylist.
- **Search space (knobs the proposer may turn):** consistency-level and fault-intensity
  curriculum, service-graph/SSM architecture, RSSM and SSM-carry toggles, drift-mitigation
  strength (fault-noise σ, schedule), `π_c`/`π_w` policies (including the tier mixture),
  TTT/stream learning rate and replay size, RLVR/GRPO settings.
- **Frozen distributed eval cells.** A held-out set of cluster configs + workloads + **fault
  seeds** the proposer never sees or mutates; anomalous jumps re-evaluate on a second
  held-out set (the SPEC-4 §5.3 tripwire). *Critical (W7-specific):* the eval must fix the
  **consistency model and the tiered-oracle cost schedule**, or a proposer could "win" by
  silently checking a weaker model or a cheaper tier — a new reward-hacking surface the
  denylist must close.
- **Denylist (the judge is not a knob, `DD-AR2`).** The proposer cannot edit the oracle (any
  tier), the metrics, the goldens, the gate, `docs/distributed-semantics.md`, the consistency
  checker, or the tier-cost schedule.
- **The four irreducibles stay with the human** (SPEC-4 §8), restated: the **objective**
  (faithfulness; bits-to-correct down, the H17 tier mixture and H18 ratio understood), the
  **safety/ethics boundary** (defensive-only, sandbox, **no real-internet egress**,
  editable-path denylist, §15), the **kill-switch + resource cap**, and **promotion to main**.

The distributed world *strengthens* the autoresearch story twice over: the per-tier signal
is denser than a single scalar, and the DES-as-data-factory lets the engine manufacture the
exact harder world (more faults, weaker consistency) that the science needs — the cleanest
realization yet of "the engine builds its own curriculum, the oracle keeps it honest."

---

## 15. Safety & ethics

This subsystem simulates a running distributed system, closer to operational capability than
any prior toy. The posture (SPEC.md §13 / SPEC-3 §14 / SPEC-4 §9 / SPEC-5 §15 / SPEC-6 §15)
holds and tightens:

- **Defensive framing only.** Downstream use is autonomous **defense**, **change-safety**
  ("will this config push / failover / deploy break the cluster?"), SRE, incident response,
  and capacity planning — predicting the consequences of *your own* changes in a sandbox
  before they touch production (the τ-bench/AppWorld framing, §2.6, §7). Not offense, not
  exploitation, not third-party targeting. No exploit-bearing workload is a goal or
  deliverable; any environment encoding real attack dynamics (e.g., a consensus-protocol
  exploit) is reviewed before release and may be held back (SPEC.md §13).
- **No real-internet egress.** Tier-A is a pure simulator. Tier-B runs only under DST
  runtimes (madsim/Shadow/Antithesis-class) in sandboxes with egress disabled and
  self-contained, seeded workloads. The model never touches a system it does not own.
- **Reproducibility as a safety property.** Everything replays from `(seed, commit)`; no
  telemetry, no runtime network call (the repo-wide posture).
- **Denylist + kill-switch + resource cap** govern the engine (§14); goldens pin semantics so
  the engine cannot quietly repurpose them or weaken the consistency model it is judged
  against.
- **Dual-use note.** A faithful distributed-system simulator is dual-use; the denylist, the
  defensive task framing, and the sandbox-only tiered oracle are the structural mitigations.
  Consistency-faithfulness is modeled to make the simulator *trustworthy about failures*
  (a defender's need — does the model correctly predict that a partition causes a stale read
  or a write rejection?), not to model exploitation.

---

## 16. Open, decentralized, verified contribution (continued)

SPEC-6 §16 specified the protocol; SPEC-7 strengthens it, because the distributed world is
where the user's "open, freely-available, decentralized" intent has the most natural form
and the cleanest verification story (research methodology and community infrastructure only;
no service, no commercial path — SPEC.md §11 holds).

> **◐ shipped** ([`distcontrib/`](../../src/verisim/distcontrib/),
> [`test_distcontrib.py`](../../tests/test_distcontrib.py)): the distributed verified-contribution
> protocol, the dependency-free distributed analogue of the host [`contrib/`](../../src/verisim/contrib/).
> `verify_transition` / `verify_trajectory` accept a contributed `(state, action, next_state[,
> delta, observation])` iff re-running the deterministic oracle reproduces it — and add the one
> thing the host/network protocols could not need, the **tiered acceptance** (`tier=`): `bit_exact`
> demands byte-for-byte reproduction, while `metamorphic`/`cycle`/`symbolic` accept *any* next-state
> the cheap tier admits as legal under the declared model — the W7 path made concrete (a contributor
> running an equally-valid but byte-different schedule is admitted by the consistency tier where
> bit-exact rejects it; a genuinely illegal one — e.g. a read that mutates a replica — is still
> caught). Trajectories must also *chain* (`next_state[i] == state[i+1]`) so transitions cannot be
> spliced; `content_address` gives the corpus its tamper-evident SHA-256 manifest hash; hostile input
> is *rejected, never raised*. **Bug fixed in the build:** the round-trip `from_canonical(to_canonical(s))`
> was *not* exact — `partitions` was stored in the oracle's construction order while serialization
> sorted it, so a re-executed genuine trajectory spuriously refuted at the first non-sorted partition.
> Fixed at the source (canonical sorted partition order in `apply` + `DistributedState.__post_init__`),
> pinned by a round-trip test on a non-sorted partition. *(Tier-B now ships as the in-repo actor-runtime
> DST oracle (§5.2, §13 DS8, ED7); deferred: a wrapped **external**-binary real-DST runtime, and the
> §16 (b)/(c) contributable-oracle / golden-trajectory ingestion.)*

- **The oracle makes contributed cluster traces trustless by construction.** As in SPEC-6
  (vs Prime Intellect's TOPLOC heuristic, §SPEC-6 2.9), any contributed
  `(state, action, next_state)` or `(state, action, delta)` is verified by re-running the
  deterministic DES on `(seed, commit)` and comparing bit-for-bit — *or*, where bit-exact is
  intractable (W7), by checking the contributed history against the cheap consistency tier.
  A contribution is accepted iff it reproduces (or admits the declared consistency model).
- **What can be contributed.** (a) Oracle-verified cluster trajectories (expand coverage of
  the fault/consistency space — the §2.1 curriculum lever, the part DST coverage blind spots
  most need); (b) new tiered oracles for a subsystem behind the existing `O[tier]` protocol
  (a new symbolic checker, a new metamorphic invariant, a new consistency model); (c) golden
  trajectories pinning additional semantics (denylist review, §14). Each carries its manifest
  + content-hash (SPEC-2 §4, §12).
- **Why this is the defensible community contribution.** The artifact others get is not a
  model to compete with but a **free, exact (or consistency-checked) verification layer** for
  distributed-state world-model data, plus a benchmark/RL-env anyone can extend and have
  certified by the tiered oracle — the open, decentralized contribution the deterministic
  oracle (and, where it is intractable, the cheap consistency tier) makes uniquely possible.

---

## 17. Open questions

The v0/§17 discipline: record them, resolve them in the open.

1. **Semantics boundary.** How much of MVCC / isolation levels / a real consensus protocol
   does Tier-A pin before it overfits to a toy or becomes too costly to keep correct? (The
   single most important design call — the distributed analogue of SPEC-5 §17.1 / SPEC-6
   §17.1.)
2. **Tier-B determinism under partition.** How far madsim/Shadow/Antithesis-class runtimes
   seal asynchrony/partition timing (HW-5), and where a regime is irreducibly *stochastic* so
   the divergence metric must account for it (SPEC-5 §17.2 / SPEC-6 §17.2, harder here).
3. **`ε` across consistency classes.** What "ε-close" means when the target is a consistency
   *class*, not a state — a per-class threshold, a distance in the consistency hierarchy, or
   weakest-admissible? Tied to H19/H20.
4. **Tier-cost calibration.** The *dollar* cost of each oracle tier (§5) sets the whole H17
   result; how to price metamorphic vs cycle vs symbolic vs bit-exact honestly and
   reproducibly so the per-dollar comparison is fair.
5. **Coverage of the fault space.** DST samples faults (§2.1 skeptical note); how to measure
   and report the coverage blind spots a model inherits, and whether the engine's curriculum
   search (§14) closes them.
6. **Competitive-ratio rigor.** Where the learning-augmented guarantee (H18) is *provable*
   (bit-exact tier, exact fallback) vs only *empirical* (cheap tiers that refute but do not
   reconstruct) — and whether a cheap-tier fallback can be made provably drift-bounding.
7. **Counterfactual sampling.** Which fault-flip distribution produces counterfactuals that
   transfer (H5) — random flips, or targeted "near-miss" partitions / split-brain scenarios
   (the operationally decisive ones)?
8. **Plan-level loop (inherited, SPEC-5 §17.8 / SPEC-6 §17.8).** How to define divergence and
   `H_ε` for the LLM-integration case (§7) where the unit is a *plan* (a sequence of admin
   ops), not a single op.

---

## 18. Definition of done

SPEC-7 is done when:

1. DS0–DS6 ship, tested, with the deterministic core dependency-free, GPU-free.
2. The **distributed `H_ε(ρ)` curve (ED1)** is plotted once, cleanly, regenerable from
   config + seeds — *whatever it shows* — **and** the **tiered-oracle measurement (H17,
   §9.4)** is reported: does spending a budget across cheap + rare-expensive tiers beat
   spending it all on bit-exact truth? A flat interior or a no-tier-benefit result is a
   reportable result (the honest negative), not a failure.
3. **✅ shipped.** The honest write-up (the `docs/report.md` discipline) states, for each
   hypothesis in §10 (the operationalized H5/H8 and the new H17–H21), what was found and what the
   honest negative looked like — including the consistency-vs-bit horizon gap (H19), the
   `H_ε(consistency-level)` result (H20), the fault-injection transfer result (H21), and the
   competitive-ratio fit (H18) — in the distributed-world section of
   [`docs/report.md`](../report.md), each number read off its committed CSV.
4. **✅ shipped.** The distributed world is packaged for reuse (Inspect benchmark
   [`disteval/`](../../src/verisim/disteval/) + `verifiers`-spec distributed RL env
   [`distrl/`](../../src/verisim/distrl/) + the LLM-callable cluster-simulator protocol of §7
   [`distsim/`](../../src/verisim/distsim/) + the verified-contribution protocol of §16
   [`distcontrib/`](../../src/verisim/distcontrib/)), so the community can measure long-horizon
   faithfulness of a *running distributed system* with ground truth — at *tiered* cost,
   because full truth is intractable — the contribution of §1.4. All four pieces are
   dependency-free at their core (the Inspect adapter alone needs the optional `[eval]` extra).

The science is one curve, again — but for the first time it is the curve of a world where
the full oracle is *unaffordable*, so the question underneath it is no longer "how little
oracle can we get away with" but "which *price of truth* do we buy, and when." That is the
question that decides whether oracle-grounded world models reach the systems that actually
run the internet, or stop at the machines that host them.

---

## 19. Provenance and reading order

- **Prereqs:** [SPEC.md](./SPEC.md) (science), [SPEC-2.md](./SPEC-2.md) (filesystem v0),
  [SPEC-5.md](./SPEC-5.md) (network world), [SPEC-6.md](./SPEC-6.md) (host world). SPEC-7
  composes the network and host worlds into a distributed system and assumes their builds.
  Companions: [SPEC-3.md](./SPEC-3.md) (the depth roadmap; W1/W3/W4 and the
  speculative-execution framing extended here) and [SPEC-4.md](./SPEC-4.md) (the engine; §14
  extends it).
- **Lessons grounding this spec** (name + venue + year, per
  [`docs/related-work.md`](./docs/related-work.md)'s no-fabricated-links policy; arXiv IDs
  only where independently verified — deliberately omitted where not): deterministic
  simulation testing — FoundationDB (Will Wilson, Strange Loop 2014; SE-Radio 685, 2025),
  TigerBeetle VOPR (2023), Antithesis deterministic hypervisor (2024), madsim / turmoil,
  Shadow (USENIX-pedigree); consistency checking — Elle (Kingsbury & Alvaro, VLDB 2020),
  mixed-isolation complexity (CAV 2025), VerIso (PVLDB 2025), weak-isolation separation logic
  (ICFP 2025); metamorphic DB oracles — SQLancer PQS/NoREC/TLP (Rigger & Su, OSDI/ESEC-FSE
  2020), graph-based transactional oracle (Jiang et al., OSDI 2023); formal protocol oracles —
  Apalache (OOPSLA 2019), Stateright, the P language; learning-augmented algorithms (learned
  indexes with error-bounded fallback; LARU; MAT); world-model-as-tool — WebDreamer
  (Gu et al., 2024; NAACL 2025), CWM / Code World Model (Meta FAIR, 2025); executable
  environments — τ-bench / τ²-bench (Sierra, 2024–2025), AppWorld (ACL 2024), ToolEmu
  (ICLR 2024), SWE-Gym (2024), R2E-Gym (2025); architecture — m4 / RouteNet bipartite GNNs,
  DreamerV3 RSSM, Mamba/SSD (Gu & Dao, 2023–2024), CLRS / TransNAR, emergent program
  semantics (Jin & Rinard, ICML 2024); the SLM thesis (NVIDIA / Belcak et al., 2025); and the
  in-house sibling repos `worldify` (temporal-causal fact store → the state/belief model,
  §2.5) and `securifine` (differential severity-weighted safety eval → the change-safety
  evaluation, §2.5). *(These should be added to `docs/related-work.md` when this spec moves
  from design to build.)*
- **Author:** Clay Good. **License:** MIT. The distributed oracle runs only inside the
  sandbox; no real-internet egress, no telemetry, defensive framing (§15).
- A living spec: as milestones (§13) land they are marked and their figures linked (mirroring
  SPEC-2 §13 / SPEC-5 §13 / SPEC-6 §13); as a hypothesis (§10) is tested its result is
  recorded in SPEC.md §9. The spec is the record of what we believed and what we learned.
