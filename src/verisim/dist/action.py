"""The distributed action grammar (SPEC-7 §3.2, DS0 increment 1).

Actions in a constrained grammar paired with the reference oracle, exactly as every prior world
pairs a grammar with its oracle. Two of SPEC-7's three families ship in increment 1 -- the client
workload and the fault/time medium (the consensus/admin family arrives with the Raft-subset):

    put <node> <key> <val>          # client: write <val> to <key> at <node> (async repl)
    get <node> <key>                # client: read <key>'s local replica at <node> (may be stale)
    cas <node> <key> <old> <new>    # client: set <key> to <new> iff <node>'s local value == <old>
    delete <node> <key>             # client: tombstone <key> (a versioned delete; resurrection-safe)
    incr <node> <key>               # client: atomic +1 (read-modify-write; LWW loses concurrent incrs)
    cincr <node> <key>              # client: CRDT G-counter +1 (loss-free, always available; sums on merge)
    cdecr <node> <key>              # client: CRDT PN-counter -1 (decrementable; may go negative; loss-free)
    cget <node> <key>               # client: read a CRDT counter (G-counter sum, minus PN decrements)
    sadd <node> <key> <elem>        # client: CRDT OR-Set add (add-wins, re-addable; unique-dot tagged)
    srem <node> <key> <elem>        # client: CRDT OR-Set remove (observed-remove; tombstones seen dots)
    smembers <node> <key>           # client: read a CRDT OR-Set (elements with a non-tombstoned dot)
    mvput <node> <key> <val>        # client: CRDT MV-register write (supersedes observed; siblings on conflict)
    mvget <node> <key>              # client: read a CRDT MV-register (surviving sibling values)
    lwwput <node> <key> <val>       # client: CRDT LWW-register write (Lamport-timestamped; one winner)
    lwwget <node> <key>             # client: read a CRDT LWW-register (the last-writer-wins value)
    advance <dt>                    # time: +<dt> clock; deliver in-flight msgs now due & reachable
    partition <nodes> | <nodes>     # fault: split the network into groups (| separates groups)
    heal                            # fault: remove all partitions (one fully-connected group)
    crash <node>                    # fault: <node> goes down (stops delivering/applying)
    restart <node>                  # fault: <node> comes back up
    drop <src> <dst>                # fault: lose every in-flight message from <src> to <dst>
    delay <src> <dst> <dt>          # fault: defer every in-flight <src>-><dst> message by <dt>
    reorder <src> <dst>             # fault: reverse the delivery schedule of <src>-><dst> messages
    clock_skew <node> <delta>       # fault: offset <node>'s clock by <delta> (signed; 0 clears it)
    anti_entropy <node>             # protocol: read-repair <node> to the latest reachable replica
    gossip <a> <b>                  # protocol: pairwise sync — a and b both adopt the per-obj winner
    elect <node>                    # consensus: <node> becomes leader iff its side holds a majority
    propose <node> <key> <val>      # consensus: leader-only write to a majority (term-fenced)
    step_down <node>                # consensus: the current leader relinquishes (leaderless, same term)
    lease <node> <dt>               # consensus: the leader takes a read lease until clock+<dt>
    lread <node> <key>              # consensus: a leader-lease local linearizable read (no quorum)
    read_index <node> <key>         # consensus: a quorum-confirmed linearizable read (Raft ReadIndex)
    append <node> <key> <val>       # consensus: append <key>=<val> to the replicated log (Raft log)
    add_replica <node>              # consensus: add <node> to the voting membership (quorum grows)
    remove_replica <node>           # consensus: remove <node> from the voting membership (quorum shrinks)
    enqueue <node> <queue> <val>    # client: append <val> to the FIFO <queue> (replicated)
    dequeue <node> <queue>          # client: pop the head of <queue> at <node> (delivery per model)
    deploy <node> <version>         # admin: set <node>'s software version (rolling upgrade)
    config_push <node> <key> <val>  # admin: leader-committed, majority-replicated cluster config
    host <node> <syscall...>        # host: run a SPEC-6 syscall on <node>'s embedded host

The fault/time family is the source of all interesting dynamics (stale reads under partition,
convergence after heal+advance) -- the distributed analogue of SPEC-5's ``advance Δt`` and SPEC-6's
scheduler, and the ``BUGGIFY`` of deterministic simulation testing (SPEC-7 §2.1). ``drop`` (DS0
increment 11) is the unreliable-network fault: where ``partition`` *holds* a message (delivered once
the link ``heal``s), ``drop`` **destroys** it, so the destination replica permanently misses that
write -- the lost-message anomaly that breaks the eventual-consistency convergence guarantee until a
*newer* write overwrites it (ED18). ``anti_entropy`` (DS0 increment 12) is the first **protocol**
op: the **read-repair / anti-entropy** mechanism real eventually-consistent stores (Dynamo,
Cassandra) use to converge *despite* lost messages -- a node pulls each object to the latest
``(version, value)`` among its reachable replicas, so it repairs a dropped write that ``advance``
never can, bounded only by what is currently reachable (ED19). ``gossip`` (DS0 increment 15) is its
**pairwise, bidirectional** sibling: ``gossip a b`` reconciles *both* ``a`` and ``b`` to the
per-object winner of their two replicas in one step (the Merkle-tree pairwise anti-entropy of
Dynamo/Cassandra, vs ``anti_entropy``'s one-directional pull-to-one-node), so a chain of pairwise
gossips spreads a write across the whole reachable component epidemically (ED22). ``delay`` and
``reorder`` (DS0
increment 13) are the message-timing faults: ``delay src dst dt`` defers every in-flight
``src``->``dst`` message by ``dt`` (a *recoverable* delay -- the counterpart to ``drop``'s
unrecoverable loss), and ``reorder src dst`` reverses the delivery schedule of that channel's
messages (the multiset of times preserved, the order flipped). Both only edit the existing
``Message.deliver_after`` field, so they add no state and compose with every consistency model;
they make the §3.4 "reorder/skew is the medium" axis a controllable input (ED20). ``clock_skew``
(DS0 increment 14) is the last of the §3.4 medium faults: ``clock_skew node delta`` offsets a node's
local clock by a signed ``delta``, which shifts the ``deliver_after`` it stamps on the messages it
sends (ahead = its sends are deferred, behind = rushed). Because the protocol resolves conflicts by
last-writer-wins on ``(version, value)`` -- never on a wall-clock timestamp -- skew shifts *when* a
write is delivered but never *which* write wins, so the converged state is clock-independent (ED21,
the property deterministic-simulation testing injects skew to verify). ``elect``/``propose`` (DS0
increment 16) are the **consensus** family — the third action family (SPEC-7 §3.2), a Raft-subset
leader-election core. ``elect node`` makes ``node`` the cluster leader **iff its partition side holds
a strict majority of the live nodes** (so two majorities — hence two leaders — can never coexist: no
split-brain), bumping the monotone ``term``. ``propose node key val`` is a **leader-fenced** write:
it commits a synchronous majority write (the consensus quorum, regardless of the KV
``consistency_model``) only if ``node`` is the *current* leader, so a leader deposed by a higher-term
election cannot commit even after the partition heals — the Raft leader-completeness safety property
plain ``quorum`` writes lack (ED23). ``step_down`` (DS0 increment 17) completes the leadership
lifecycle: ``step_down node`` lets the *current* leader **voluntarily relinquish** power, leaving the
cluster **leaderless at the same term** — the graceful counterpart to ``elect``'s involuntary,
higher-term deposition. Until a fresh ``elect`` installs a successor every ``propose`` is rejected
``not_leader``, so a clean handoff is ``step_down`` then ``elect <successor>`` and no leaderless
window ever commits. Relinquishing needs no quorum (it reads only the node's own leadership), so a
leader stranded in a minority can step down where its ``propose`` is ``no_quorum`` — giving up
authority is always safe, committing under it is not (ED24). ``lease``/``lread`` (DS0 increment 18)
are the **leader lease** — the Raft read optimization. ``lease node dt`` lets the *current* leader
take a read lease through global clock ``+ dt``; while that lease holds, ``lread node key`` serves a
**local linearizable read without a quorum round-trip** (safe because the lease guarantees the
leader's term is uncontested — a new ``elect`` is *blocked* until the lease expires). So a leader
partitioned into the minority can still ``lread`` (local, no quorum) where its ``propose`` is
``no_quorum`` — the lease's whole purpose. The safety tension it resolves: a new ``elect`` is
rejected ``lease_held`` until the incumbent's lease expires (a successor waits it out), so leadership
cannot change hands under a live lease — but ``step_down`` *releases* the lease immediately, so a
graceful handoff needs no wait where a crashed leader forces the cluster to wait the lease out (ED25).
``append`` (DS0 increment 19) is the **replicated-log** path the spec named since increment 1:
``append node key val`` appends a ``(term, index, key, value)`` entry to the leader's log and
replicates it to the reachable followers (who adopt the leader's prefix, **overwriting any divergent
uncommitted tail** — the log-matching reconciliation), committing it (and applying it to the KV)
**iff a majority holds it**. A minority-stranded leader still appends locally (``uncommitted``) but
does not commit, so its entry can be overwritten by a higher-term leader — the Raft log-matching
safety the one-shot ``propose`` could not express (ED26). ``add_replica``/``remove_replica`` (DS0
increment 20) are the **membership-change** admin ops the §3.2 grammar named: they reconfigure the
*consensus voting set* (the nodes that count toward an election/commit quorum), a leader-committed
change, so the **majority threshold tracks the membership** — ``remove_replica`` shrinks the cluster
(a smaller majority suffices, the standard way to restore availability after nodes fail) and
``add_replica`` grows it. All config nodes still physically store replicas; membership is the voting
overlay (the empty set is the "all nodes vote" default). The active leader cannot be removed
(``is_leader`` — step it down first), and the last member cannot be removed (ED27). ``enqueue``/
``dequeue`` (DS0 increment 21) are the §3.2 **FIFO-queue** client ops — a second data type beside the
KV store. ``enqueue node queue val`` appends ``val`` to a replicated queue; ``dequeue node queue``
pops the head of the coordinator's queue replica. The headline property is that **delivery semantics
follow the ``consistency_model``**: under ``eventual`` a dequeue's head-removal reaches only the
reachable side, so a partitioned peer can re-dequeue the same item (**at-least-once / duplicate
delivery**), where ``linearizable`` (needs every replica) and ``quorum`` (needs a majority) gate
availability to keep delivery **exactly-once** on the side that can serve it — the queue analogue of
the KV CAP tradeoff (ED28). ``deploy`` (DS0 increment 22) is the **rolling-upgrade** admin op that
answers SPEC-7's headline question — *"will this deploy break the cluster?"*: ``deploy node version``
sets a node's running software version, and two nodes can participate in the same consensus quorum
only if their versions are within ``DistConfig.max_version_skew`` (``1`` = the standard N-1
compatibility window). A rolling upgrade that stays inside the window keeps quorum throughout (safe),
but a deploy that creates an **incompatible version split with no compatible majority** loses quorum
— ``propose``/``elect`` become ``no_quorum`` mid-deploy — the deploy broke the cluster (ED29).
Compatibility gates *consensus* only; the best-effort KV/queue data plane is version-agnostic.
``host`` (DS0 increment 23) is the **embedded SPEC-6 host** op — the compositional vision §3.1/§4
name: ``host node <syscall...>`` runs a SPEC-6 syscall (``fork``/``exit``/``setuid``/``open``/
``write``/``close``) on ``node``'s own embedded host (a process table + per-process fd tables + an
embedded v0 filesystem), so a cluster node is not just a bag of KV replicas but a real host running
processes. Each node's host is **isolated** (a ``fork`` on one node does not appear on another), and
host ops respect the node's up/down status — a crashed node's host is ``unavailable`` (the cross-
layer crash linkage). The effect delegates to the SPEC-6 ``ReferenceHostOracle``, wrapping its bundle
delta in a ``HostStep`` edit (ED30). ``config_push`` (DS0 increment 24) is the **config-management**
admin op that answers SPEC-7's *other* headline question — *"will this config push break the
cluster?"*: ``config_push node key val`` pushes a cluster-config value, and unlike ``deploy`` (a
node-local version label that gates consensus *compatibility*) it is a **leader-committed,
majority-replicated** setting — a Raft-style config entry. It commits (writing the value to every
*reachable* voting member) only if ``node`` is the current leader and reaches a majority, so a push
issued by a non-leader is ``not_leader`` and one by a minority-stranded leader is ``no_quorum`` (the
push cannot commit). When it does commit under a partition it reaches only the majority side, so the
**partitioned minority retains its stale config** (config divergence) — repaired by re-pushing after
``heal`` (ED31). ``read_index`` (DS0 increment 25) is the **quorum-confirmed linearizable read** — the
Raft *ReadIndex* method, the partner to the lease read ``lread`` (incr 18). ``read_index node key``
serves ``key`` from the leader's replica **only after confirming leadership via a majority** of the
voting members (the heartbeat round), so a leader stranded in the minority is ``no_quorum`` (it cannot
confirm it is still leader, so it refuses the read — no stale read from a possibly-deposed leader), and
a deposed leader is ``not_leader`` even after ``heal``. The two linearizable reads have **opposite
availability profiles**: ``lread`` trades a quorum round-trip for a clock (a live lease serves a *local*
read, available even in the minority), while ``read_index`` trades the clock dependence for a quorum
round-trip (no lease/clock assumption, but unavailable without a majority) — the two standard ways Raft
serves a linearizable read, and the operational choice between them (ED32). ``delete`` (DS0 increment
26) is the **tombstone delete** — the fundamental KV remove, and a canonical distributed hazard.
``delete node key`` is a *versioned write of a tombstone* (it reuses the ``put`` replication path with
the ``TOMBSTONE`` value), **not** a removal of the replica: the deleted key keeps a replica at a
bumped version, so last-writer-wins orders the delete against concurrent writes by version. A ``get``
on a tombstoned replica reports ``("deleted", "")``. This is exactly what avoids the **resurrection
problem**: under ``eventual`` a delete on one side leaves a partitioned peer still reading the old
value (the deleted item is "still there" on the minority — the danger), but when the tombstone
propagates (``advance``/``anti_entropy``/``gossip``) its higher version **wins** over the stale value,
so the key converges to deleted rather than being resurrected — where a naive removal would let the
stale replica's value reappear. A genuinely newer ``put`` (a higher version than the tombstone) does
bring the key back, which is a new write, not a resurrection (ED33). ``incr`` (DS0 increment 27) is
the **atomic counter** — the first *read-modify-write* client op (``put``/``cas``/``delete`` are blind
or compare writes), and the canonical case where eventual-consistency last-writer-wins **silently
loses updates**. ``incr node key`` reads the coordinator's local counter value (a non-numeric/absent
value is ``0``) and writes ``value + 1`` at a bumped version, reusing the ``put`` replication path.
Sequentially it is correct (``incr`` ``k`` times → ``k``), but under a partition the consistency model
decides whether concurrent increments survive: **under ``eventual`` two increments on opposite sides
both read the same count and both write ``count + 1`` at the same version, so the merge keeps only one
— a *lost update* (two acknowledged ``incr``s, count short by one)**, where ``quorum`` makes the
minority side ``unavailable`` (only the accepted increment is counted — no silent loss) and
``linearizable`` rejects the write under *any* partition. The read-modify-write CAP tradeoff (ED34) —
harder than the blind-write one (ED14) because LWW loses an RMW update where it merely makes a blind
write stale. ``cincr``/``cget`` (DS0 increment 28) are the **CRDT G-counter** — the loss-free
resolution to ``incr``'s lost-update negative. A G-counter is a *state-based CRDT*: each node keeps a
per-owner vector of monotone sub-counts, and ``cincr n key`` bumps **only ``n``'s own sub-count**
(``cget n key`` reads the sum over owners). Because concurrent increments at different nodes touch
*disjoint* vector entries, there is **no conflict and no lost update**, and the CRDT **join is the
per-(key, owner) max** — a commutative, associative, idempotent merge that ``anti_entropy``/``gossip``
apply, so the counter converges to the exact total regardless of partition or message order. And
because ``cincr`` is purely node-local it is **always available** — a partitioned-alone node still
counts, where the LWW ``incr`` under ``quorum``/``linearizable`` is ``unavailable``. So ``cincr`` is
both more available *and* loss-free than ``incr``: the textbook reason CRDTs exist, recovered as the
positive that resolves ED34's negative (ED35). ``cdecr`` (DS0 increment 29) makes the counter
**decrementable** — the **CRDT PN-counter**: a grow-only G-counter cannot go down, so a PN-counter
pairs *two* G-counters, P (the ``cincr`` half) and N (the ``cdecr`` half), and ``cget`` reads
**P − N**. ``cdecr n key`` bumps **only ``n``'s own** N sub-count, so (like ``cincr``) it is purely
node-local and always available, concurrent decrements never conflict, and the per-(key, owner) max
join merges *both* halves — so the PN-counter still converges loss-free under partition, and its
value may now go **negative** (the property the grow-only G-counter lacks). ``cincr``/``cdecr``/``cget``
together are the full decrementable CRDT counter (ED36). ``sadd``/``srem``/``smembers`` (DS0 increment
30) are the **CRDT OR-Set** — the canonical *interesting* CRDT, a replicated set that a naive
implementation gets wrong. The trick is the **unique dot**: ``sadd n key elem`` tags the element with
``(owner=n, seq)`` (a per-(key, owner) monotone sequence) and adds it to ``n``'s observed add-set;
``srem n key elem`` moves the dots ``n`` has *observed* into ``n``'s tombstone-set (the *observed*-
remove); ``smembers n key`` is the elements with at least one **non-tombstoned** dot. The join is
**set union** of both halves (commutative/associative/idempotent), so a concurrent ``sadd`` (a fresh
dot the remover never saw) **survives** a concurrent ``srem`` (**add wins**), and a removed element is
**re-addable** (a new dot is not in the tombstone-set) — the two properties an element-level 2P-Set
(remove-wins, never re-addable) lacks (ED37). ``mvput``/``mvget`` (DS0 increment 31) are the **CRDT
MV-register** — the Dynamo/Riak register that *surfaces* concurrent conflicting writes as **siblings**
instead of silently dropping one (the LWW the KV ``put`` and the counters do). It reuses the OR-Set's
dot/union machinery: ``mvput n key val`` tags ``val`` with a fresh dot, **tombstones every dot it
currently observes** (a write supersedes the values it saw), and adds its own — so a *sequential*
overwrite collapses to one value, but *concurrent* writes (neither observing the other) **both
survive**, and a later context-aware ``mvput`` (observing both) **resolves** them. ``mvget n key``
reads the set of surviving sibling values. The conflict is made *visible and resolvable* rather than
silently lost — the textbook reason multi-value registers exist (ED38). ``lwwput``/``lwwget`` (DS0
increment 32) are the **CRDT LWW-register** — the *policy-opposite* of the MV-register: instead of
surfacing a conflict it **deterministically picks one winner** by a **Lamport-timestamp total order**.
``lwwput n key val`` stamps ``val`` with ``(ts, owner=n)`` where ``ts = lamport[n] + 1`` (advancing
``n``'s Lamport clock), and the join keeps the **max** copy by ``(ts, owner, value)`` — so a write
that *happened-after* another (higher ts) always wins, regardless of which node made it, and truly
*concurrent* writes (equal ts) break the tie by node id (a stable winner). ``lwwget n key`` reads the
single winning value. The Lamport clock — advanced on write and on merge — is what makes "happens-
after" a comparable order (ED39).
"""

from __future__ import annotations

from dataclasses import dataclass

# name -> fixed arity (number of whitespace tokens after the name), or None for variable arity
# (``partition`` carries a ``|``-separated group list; ``heal`` takes none).
_ARITY: dict[str, int | None] = {
    "put": 3,  # node key val
    "get": 2,  # node key
    "cas": 4,  # node key old new
    "delete": 2,  # node key  -- tombstone <key>, a versioned (resurrection-safe) delete (DS0 incr 26)
    "incr": 2,  # node key  -- atomic read-modify-write +1 on a counter value (DS0 incr 27)
    "cincr": 2,  # node key  -- CRDT G-counter +1 (node-local, loss-free, always available; incr 28)
    "cdecr": 2,  # node key  -- CRDT PN-counter -1 (decrementable; may go negative; DS0 incr 29)
    "cget": 2,  # node key  -- read a CRDT counter: G-counter sum minus PN decrements (incr 28/29)
    "sadd": 3,  # node key elem  -- CRDT OR-Set add (add-wins, re-addable; DS0 incr 30)
    "srem": 3,  # node key elem  -- CRDT OR-Set observed-remove (tombstones seen dots; DS0 incr 30)
    "smembers": 2,  # node key  -- read a CRDT OR-Set: non-tombstoned elements (DS0 incr 30)
    "mvput": 3,  # node key val  -- CRDT MV-register write (siblings on conflict; DS0 incr 31)
    "mvget": 2,  # node key  -- read a CRDT MV-register: surviving sibling values (DS0 incr 31)
    "lwwput": 3,  # node key val  -- CRDT LWW-register write (Lamport-timestamped; DS0 incr 32)
    "lwwget": 2,  # node key  -- read a CRDT LWW-register: the last-writer-wins value (DS0 incr 32)
    "advance": 1,  # dt
    "partition": None,  # <nodes> | <nodes> [| ...]
    "host": None,  # <node> <syscall...>  -- a SPEC-6 host syscall on <node>'s host (DS0 incr 23)
    "heal": 0,
    "crash": 1,  # node
    "restart": 1,  # node
    "drop": 2,  # src dst  -- lose every in-flight message from <src> to <dst> (DS0 incr 11)
    "delay": 3,  # src dst dt  -- defer every in-flight <src>-><dst> message by <dt> (DS0 incr 13)
    "reorder": 2,  # src dst  -- reverse the delivery schedule of <src>-><dst> messages (DS0 incr 13)
    "clock_skew": 2,  # node delta  -- set <node>'s clock offset (signed; DS0 incr 14)
    "anti_entropy": 1,  # node  -- read-repair <node> to the latest reachable replica (DS0 incr 12)
    "gossip": 2,  # a b  -- pairwise bidirectional anti-entropy: both adopt the winner (DS0 incr 15)
    "elect": 1,  # node  -- <node> becomes leader iff its side holds a majority (DS0 incr 16)
    "propose": 3,  # node key val  -- leader-only term-fenced majority write (DS0 incr 16)
    "step_down": 1,  # node  -- the current leader relinquishes; leaderless at the same term (incr 17)
    "lease": 2,  # node dt  -- the leader takes a read lease until clock+dt (DS0 incr 18)
    "lread": 2,  # node key  -- a leader-lease local linearizable read, no quorum (DS0 incr 18)
    "read_index": 2,  # node key  -- a quorum-confirmed linearizable read, Raft ReadIndex (DS0 incr 25)
    "append": 3,  # node key val  -- append to the Raft replicated log; commit on majority (incr 19)
    "add_replica": 1,  # node  -- add <node> to the voting membership (DS0 incr 20)
    "remove_replica": 1,  # node  -- remove <node> from the voting membership (DS0 incr 20)
    "enqueue": 3,  # node queue val  -- append <val> to the FIFO <queue> (DS0 incr 21)
    "dequeue": 2,  # node queue  -- pop the head of <queue> at <node> (DS0 incr 21)
    "deploy": 2,  # node version  -- set <node>'s software version, a rolling upgrade (DS0 incr 22)
    "config_push": 3,  # node key val  -- leader-committed, majority-replicated config (DS0 incr 24)
    # ``host`` (variable arity) is in ``_ARITY`` above with value None.
    # Transaction family (DS0 increment 2): a multi-key OCC transaction at a coordinator node.
    "begin": 2,  # node txn  -- open a transaction
    "tget": 3,  # node txn key  -- read <key> within the txn (pins the read version for validation)
    "tput": 4,  # node txn key val  -- buffer a write to <key> within the txn
    "commit": 2,  # node txn  -- validate the read-set + apply buffered writes atomically, or abort
    "abort": 2,  # node txn  -- discard the txn
}

# ``begin``/``commit``/``abort`` + the txn-scoped ``tget``/``tput`` are client ops (the workload), as
# are the ``enqueue``/``dequeue`` FIFO-queue ops (DS0 incr 21).
CLIENT_OPS = frozenset({
    "put", "get", "cas", "delete", "incr", "cincr", "cdecr", "cget",
    "sadd", "srem", "smembers", "mvput", "mvget", "lwwput", "lwwget",
    "begin", "tget", "tput", "commit", "abort", "enqueue", "dequeue",
})
TXN_OPS = frozenset({"begin", "tget", "tput", "commit", "abort"})
QUEUE_OPS = frozenset({"enqueue", "dequeue"})  # the FIFO-queue client ops (DS0 incr 21)
ADMIN_OPS = frozenset({"deploy", "config_push"})  # rolling-upgrade / config admin ops (incr 22, 24)
HOST_OPS = frozenset({"host"})  # the embedded SPEC-6 host syscall op (DS0 incr 23)
FAULT_OPS = frozenset(
    {"advance", "partition", "heal", "crash", "restart", "drop", "delay", "reorder", "clock_skew"}
)
# Protocol/admin ops (SPEC-7 §3.2, the third action family). ``anti_entropy`` (the read-repair
# convergence mechanism, DS0 increment 12) and ``gossip`` (pairwise anti-entropy, incr 15) are the
# convergence ops; ``elect``/``propose`` (the Raft-subset consensus core, incr 16) are the consensus
# ops. ``CONSENSUS_OPS`` is the subset that reads/writes the leader/term metadata.
CONSENSUS_OPS = frozenset({
    "elect", "propose", "step_down", "lease", "lread", "read_index", "append",
    "add_replica", "remove_replica",
})
PROTOCOL_OPS = frozenset({"anti_entropy", "gossip"}) | CONSENSUS_OPS


class DistParseError(ValueError):
    """Raised when a string is not a valid distributed action in the DS0 grammar."""


@dataclass(frozen=True)
class DistAction:
    """A parsed distributed action. ``args`` are the post-name tokens; ``groups`` holds the parsed
    ``partition`` node groups (empty for every other op)."""

    raw: str
    name: str
    args: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...] = ()


def parse_dist_action(raw: str) -> DistAction:
    """Parse an action string into a :class:`DistAction`, validating name + arity + structure."""
    parts = raw.split()
    if not parts:
        raise DistParseError("empty action")
    name = parts[0]
    if name not in _ARITY:
        raise DistParseError(f"unknown action {name!r}; choose from {sorted(_ARITY)}")
    rest = parts[1:]

    if name == "partition":
        groups = _parse_partition_groups(rest, raw)
        return DistAction(raw=raw, name=name, args=tuple(rest), groups=groups)

    if name == "host":
        # ``host <node> <syscall...>`` (DS0 incr 23): a SPEC-6 host syscall on <node>'s embedded host.
        # ``args[0]`` is the node; ``args[1:]`` is the host syscall, validated by the SPEC-6 parser.
        if len(rest) < 2:
            raise DistParseError(f"host needs a node and a syscall: {raw!r}")
        from verisim.host.action import HostParseError, parse_host_action

        try:
            parse_host_action(" ".join(rest[1:]))
        except HostParseError as exc:
            raise DistParseError(f"host: invalid syscall in {raw!r}: {exc}") from exc
        return DistAction(raw=raw, name=name, args=tuple(rest))

    arity = _ARITY[name]
    assert arity is not None
    if len(rest) != arity:
        raise DistParseError(f"{name} expects {arity} args, got {len(rest)}: {raw!r}")
    if name == "advance":
        _parse_positive_int(rest[0], raw)
    if name == "delay":
        _parse_positive_int(rest[2], raw)  # dt: a positive deferral, like advance's
    if name == "lease":
        _parse_positive_int(rest[1], raw)  # dt: a positive lease duration, like advance's
    if name == "deploy":
        _parse_nonneg_int(rest[1], raw)  # version: a non-negative software version (0 = base)
    if name == "clock_skew":
        _parse_signed_int(rest[1], raw)  # delta: a signed clock offset (may be negative or 0)
    return DistAction(raw=raw, name=name, args=tuple(rest))


def _parse_partition_groups(rest: list[str], raw: str) -> tuple[tuple[str, ...], ...]:
    """Parse ``n0 n1 | n2`` into ``(("n0","n1"), ("n2",))``; require >= 2 non-empty groups."""
    groups: list[tuple[str, ...]] = []
    current: list[str] = []
    for tok in rest:
        if tok == "|":
            if not current:
                raise DistParseError(f"empty partition group in {raw!r}")
            groups.append(tuple(current))
            current = []
        else:
            current.append(tok)
    if current:
        groups.append(tuple(current))
    if len(groups) < 2:
        raise DistParseError(f"partition needs >= 2 groups separated by '|': {raw!r}")
    seen: set[str] = set()
    for group in groups:
        for node in group:
            if node in seen:
                raise DistParseError(f"node {node!r} in more than one partition group: {raw!r}")
            seen.add(node)
    return tuple(groups)


def _parse_positive_int(tok: str, raw: str) -> int:
    try:
        value = int(tok)
    except ValueError as exc:
        raise DistParseError(f"expected an integer, got {tok!r} in {raw!r}") from exc
    if value <= 0:
        raise DistParseError(f"dt must be positive, got {value} in {raw!r}")
    return value


def _parse_nonneg_int(tok: str, raw: str) -> int:
    """Parse a non-negative integer (the ``deploy`` software version; ``0`` is the base)."""
    try:
        value = int(tok)
    except ValueError as exc:
        raise DistParseError(f"expected an integer, got {tok!r} in {raw!r}") from exc
    if value < 0:
        raise DistParseError(f"version must be >= 0, got {value} in {raw!r}")
    return value


def _parse_signed_int(tok: str, raw: str) -> int:
    """Parse a signed integer (the ``clock_skew`` offset may be negative or zero)."""
    try:
        return int(tok)
    except ValueError as exc:
        raise DistParseError(f"expected an integer, got {tok!r} in {raw!r}") from exc
