"""Tier-A reference distributed oracle (SPEC-7 §5.1, DS0 increment 1).

A from-scratch **deterministic discrete-event simulator** of a pinned distributed semantics: a
fully-replicated key-value store with MVCC versions and **asynchronous replication** under the
fault/time medium (partition, crash, clock). It is the executable truth, paired with the normative
``docs/distributed-semantics.md``, and -- like every prior core -- it has **no runtime dependencies,
needs no GPU, and is a pure function of `(state, action)`** (the determinism contract, SPEC-7 §3.3).

The semantics in one paragraph (the eventual-consistency core, §5.1):

  - **put / cas** write the coordinator's local replica immediately (a new MVCC version =
    local version + 1) and enqueue async replication messages to the object's other replicas; they
    deliver later, on ``advance``, only if the network is connected and the destination is up.
  - **get** returns the coordinator's *local* value -- which under partition may be **stale**.
  - **advance** moves the clock forward and delivers every in-flight message now due+reachable;
    a delivered write is adopted iff it wins **last-writer-wins by (version, value)**, so once every
    message is delivered the replicas **converge** to the same value (eventual consistency).
  - **partition / heal / crash / restart** are the fault medium that makes the above interesting.

Consensus (Raft-subset), transactions/locks, and the embedded SPEC-6 host inside each node are later
increments. The oracle returns the next state *and* the delta that produces it, with
``apply(state, delta) == next_state`` by construction.
"""

from __future__ import annotations

from collections.abc import Callable

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import (
    ClockSet,
    ClockSkewSet,
    CommitIndexSet,
    ConfigSet,
    DistDelta,
    DistEdit,
    EventAppend,
    GCounterSet,
    HostStep,
    LamportSet,
    LeaseSet,
    LogSet,
    LWWRegSet,
    MemberSet,
    MsgDeliver,
    MsgDrop,
    MsgReschedule,
    MsgSend,
    MVRegTomb,
    MVRegWrite,
    NCounterSet,
    NodeDown,
    NodeUp,
    ORMapField,
    ORMapTomb,
    ORMapVal,
    ORSetAdd,
    ORSetTomb,
    PartitionSet,
    ProtocolStep,
    QueueSet,
    ReplicaWrite,
    RGAInsert,
    RGATomb,
    SetResult,
    VersionSet,
    apply,
)
from verisim.dist.state import RGA_ROOT, TOMBSTONE, DistributedState, LogEntry, Message
from verisim.dist.txn import txn_step
from verisim.distoracle.base import DistStepResult
from verisim.host.action import parse_host_action
from verisim.host.state import HostState
from verisim.hostoracle.base import EXIT_OK
from verisim.hostoracle.reference import ReferenceHostOracle

# A single stateless host sub-oracle, shared across nodes (it owns only a stateless FS sub-oracle):
# the embedded SPEC-6 host (DS0 incr 23) is computed per-node from the node's own ``HostState``.
_HOST_ORACLE = ReferenceHostOracle()


def causal_deps(state: DistributedState, node: str, key: str) -> tuple[tuple[str, int], ...]:
    """The causal context a write at ``node`` to ``key`` carries (the ``causal`` model, §3.4).

    A write "happens after" everything the writing node has already observed, so the message it
    produces must not be delivered anywhere before those observations are. The context is the node's
    currently-applied ``(object, version)`` for every *other* object it holds at a non-boot version
    (``version > 0``); boot replicas (v0) are satisfied everywhere and omitted, so most messages
    still carry no deps. This is the version-vector slice that makes cross-object causal ordering
    hold. **Shared by Tier-A and Tier-B** so the two oracles compute identical deps (the
    differential would otherwise diverge); callers attach it only under ``consistency_model ==
    "causal"``.
    """
    deps = [
        (obj, r.version)
        for (obj, n), r in state.replicas.items()
        if n == node and obj != key and r.version > 0
    ]
    return tuple(sorted(deps))


def timing_fault_edits(
    state: DistributedState, action: DistAction
) -> tuple[DistDelta, str, str]:
    """``delay`` / ``reorder``: reschedule a channel's in-flight messages (DS0 incr 13, §3.4).

    A pure **medium** change (it only moves *when* messages become deliverable, never a replica), so
    -- like ``drop`` -- there is no actor work and **Tier-A and Tier-B compute byte-identical
    deltas**. Shared by both oracles (imported by :mod:`verisim.distoracle.system`) so they cannot
    drift, exactly as :func:`causal_deps` is shared.

    - ``delay src dst dt`` pushes every in-flight ``src``->``dst`` message ``dt`` clock units later
      (``deliver_after += dt``) -- a *recoverable* delay, the counterpart to ``drop``'s
      unrecoverable loss: the write still arrives, just later, so the cluster still converges.
    - ``reorder src dst`` reverses the channel's delivery schedule: among its messages sorted by
      ``(deliver_after, id)`` the multiset of delivery times is reassigned in reverse, so the
      message that was due first becomes due last. Last-writer-wins makes the *converged* state
      invariant under this (a commutative join), but it flips which write a peer sees *in transit*.
      A channel of fewer than two messages, or one whose times all coincide, is an observational
      no-op (the reversal changes nothing). Reports the count of messages actually moved.
    """
    src, dst = action.args[0], action.args[1]
    channel = sorted(
        (m for m in state.inflight.values() if m.src == src and m.dst == dst),
        key=lambda m: (m.deliver_after, m.id),
    )
    if action.name == "delay":
        dt = int(action.args[2])
        edits: DistDelta = [MsgReschedule(m.id, m.deliver_after + dt) for m in channel]
        value = str(len(edits))
        return [*edits, SetResult("delayed", value)], "delayed", value
    # reorder: reverse the schedule -- reassign the (sorted) delivery times in reverse order.
    new_times = [m.deliver_after for m in reversed(channel)]
    edits = [
        MsgReschedule(m.id, t)
        for m, t in zip(channel, new_times, strict=True)
        if t != m.deliver_after
    ]
    value = str(len(edits))
    return [*edits, SetResult("reordered", value)], "reordered", value


def clock_skew_edits(action: DistAction) -> tuple[DistDelta, str, str]:
    """``clock_skew node delta``: set a node's clock offset (DS0 increment 14, §3.4).

    A pure medium change (it sets a per-node offset that shifts only the ``deliver_after`` the node
    stamps on its future sends, via :meth:`DistributedState.sender_clock`), so -- like ``drop`` and
    ``delay``/``reorder`` -- Tier-A and Tier-B compute byte-identical deltas. Shared by both oracles
    so they cannot drift. A 0 offset clears the skew (no residue).
    """
    node, delta = action.args[0], int(action.args[1])
    return [ClockSkewSet(node, delta), SetResult("skewed", str(delta))], "skewed", str(delta)


def gossip_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``gossip a b``: pairwise bidirectional anti-entropy (DS0 increment 15, the §4 converge form).

    The **pairwise, push-pull** sibling of ``anti_entropy`` — the Merkle-tree anti-entropy real
    eventually-consistent stores (Dynamo, Cassandra) run in the background between *pairs* of nodes,
    vs ``anti_entropy``'s one-directional pull-to-one-node. For every object both ``a`` and ``b``
    replicate, **both** adopt the per-object winner of their two replicas by the same last-writer-
    wins rule (``(version, value)``), emitting a ``ReplicaWrite`` for whichever node is behind. So
    one pairwise gossip reconciles *both* endpoints fully (where ``anti_entropy`` repairs only the
    named node), and a chain of pairwise gossips spreads a write across the whole reachable
    component **epidemically** (ED22). It needs a live link (both up and connected) — a pairwise
    sync over a real channel — and is otherwise bounded by reachability like ``anti_entropy``. A
    pure coordinator-level reconciliation (it reads both replicas directly, no in-flight message),
    so Tier-A and Tier-B compute byte-identical deltas; shared by both oracles so they cannot drift.
    Reports the number of replicas that moved.
    """
    a, b = action.args[0], action.args[1]
    if not (state.is_up(a) and state.is_up(b) and state.connected(a, b)):
        return [SetResult("unavailable", "")], "unavailable", ""
    edits: DistDelta = []
    synced = 0
    for obj in config.objects:
        ra = state.replicas.get((obj, a))
        rb = state.replicas.get((obj, b))
        if ra is None or rb is None:
            continue  # an object one of the pair does not replicate
        win = max((ra.version, ra.value), (rb.version, rb.value))  # last-writer-wins by (ver, val)
        if (ra.version, ra.value) != win:
            edits.append(ReplicaWrite(obj, a, win[0], win[1]))
            synced += 1
        if (rb.version, rb.value) != win:
            edits.append(ReplicaWrite(obj, b, win[0], win[1]))
            synced += 1
    # The CRDT counter join (DS0 incr 28/29): both endpoints adopt the per-(key, owner) **max** of
    # their two copies — the commutative/idempotent state-based merge, over *both* halves (the
    # G-counter P and the PN-counter's decrement N). Disjoint from the LWW replica merge above and a
    # no-op when no CRDT counter is used (so the pre-incr-28 form is unchanged).
    edits.extend(_gcounter_merge_edits(state, [a, b], [a, b]))
    # The CRDT OR-Set join (DS0 incr 30): both endpoints adopt the union of their add/tomb sets.
    edits.extend(_orset_merge_edits(state, [a, b], [a, b]))
    # The CRDT MV-register join (DS0 incr 31): both endpoints union their write/tomb sets.
    edits.extend(_mvreg_merge_edits(state, [a, b], [a, b]))
    # The CRDT LWW-register join (incr 32): both endpoints adopt the (ts, owner, value)-max winner.
    edits.extend(_lwwreg_merge_edits(state, [a, b], [a, b]))
    # The CRDT OR-Map join (incr 33): the OR-Set presence union + the per-field LWW max.
    edits.extend(_ormap_merge_edits(state, [a, b], [a, b]))
    # The CRDT RGA join (incr 34): the set-union of the sequence elements + tombstones.
    edits.extend(_rga_merge_edits(state, [a, b], [a, b]))
    synced = sum(1 for e in edits
                 if isinstance(e, (ReplicaWrite, GCounterSet, NCounterSet, ORSetAdd, ORSetTomb,
                                   MVRegWrite, MVRegTomb, LWWRegSet, LamportSet,
                                   ORMapField, ORMapTomb, ORMapVal, RGAInsert, RGATomb)))  # moves
    value = str(synced)
    return [*edits, SetResult("gossiped", value)], "gossiped", value


def _vector_merge_edits(
    copies: dict[tuple[str, str, str], int], into: list[str], among: list[str],
    make: Callable[[str, str, str, int], DistEdit],
) -> list[DistEdit]:
    """Per-(key, owner) **max**-merge of one CRDT counter vector (DS0 incr 28/29, the CRDT join).

    For each node in ``into``, raise its copy of every ``(key, owner)`` sub-count to the max held by
    any node in ``among`` (the reachable set), emitting ``make(key, holder, owner, best)`` per copy
    that moves. The commutative, associative, idempotent state-based join: applying it converges the
    vector regardless of order and never loses an update (each owner's sub-count is monotone and
    single-writer). Shared by the G-counter (P) and PN-counter decrement (N) halves.
    """
    edits: list[DistEdit] = []
    for key, owner in sorted({(key, owner) for (key, _h, owner) in copies}):
        best = max(copies.get((key, h, owner), 0) for h in among)
        if best == 0:
            continue
        for holder in into:
            if copies.get((key, holder, owner), 0) < best:
                edits.append(make(key, holder, owner, best))
    return edits


def _gcounter_merge_edits(
    state: DistributedState, into: list[str], among: list[str]
) -> list[DistEdit]:
    """Max-merge **both** CRDT counter halves (DS0 incr 28/29): the G-counter P and PN-counter N.

    The join over the full PN-counter is just the join over each half independently — so this merges
    ``gcounters`` (emitting ``GCounterSet``) and ``ncounters`` (emitting ``NCounterSet``). A no-op
    for a half that is empty, so a cluster that only ``cincr``-s is byte-identical to the
    pre-incr-29 form. ``anti_entropy`` pulls one node; ``gossip`` a pair.
    """
    return [
        *_vector_merge_edits(state.gcounters, into, among, GCounterSet),
        *_vector_merge_edits(state.ncounters, into, among, NCounterSet),
    ]


def cincr_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``cincr node key``: CRDT G-counter increment (DS0 increment 28, the loss-free counter).

    A *state-based* CRDT increment: ``node`` bumps **only its own** sub-count for ``key`` — a purely
    node-local edit (``GCounterSet(key, node, node, +1)``), no replication and no in-flight message.
    So it is **always available** when the node is up — a partitioned-alone node counts (the AP
    property the LWW ``incr`` lacks under ``quorum``/``linearizable``) — and because each node only
    ever writes its own sub-count, two concurrent ``cincr``s never conflict and **never lose an
    update** (the resolution to ED34). The full count is reconciled by the per-owner max join in
    ``anti_entropy``/``gossip``. Node-local and deterministic, so Tier-A ≡ Tier-B bit-for-bit. A
    crashed node is ``unavailable``. Reports the node's own new sub-count.
    """
    node, key = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    new = state.gcounters.get((key, node, node), 0) + 1
    return [GCounterSet(key, node, node, new), SetResult("ok", str(new))], "ok", str(new)


def cdecr_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``cdecr node key``: CRDT PN-counter decrement (DS0 increment 29, the decrementable counter).

    The exact twin of :func:`cincr_edits` over the PN-counter's *decrement* half (``ncounters``):
    ``node`` bumps **only its own** N sub-count for ``key`` — a purely node-local edit
    (``NCounterSet(key, node, node, +1)``), no replication and no message. So it is **always
    available** (a partitioned-alone node still counts down — the AP property), concurrent
    ``cdecr``s never conflict, and the per-(key, owner) max join merges it in
    ``anti_entropy``/``gossip`` exactly like the P half. The counter's value (``cget`` = P − N) may
    now go **negative**, the property the grow-only G-counter lacks. Node-local and deterministic,
    so Tier-A ≡ Tier-B bit-for-bit. A crashed node is ``unavailable``. Reports the new decrement
    sub-count.
    """
    node, key = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    new = state.ncounters.get((key, node, node), 0) + 1
    return [NCounterSet(key, node, node, new), SetResult("ok", str(new))], "ok", str(new)


def cget_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``cget node key``: read a CRDT counter — ``node``'s G-counter sum **minus** decrements (29).

    The counter's value at ``node`` is ``sum(P) − sum(N)`` over owners (a PN-counter; for a
    grow-only counter that was never ``cdecr``-ed, N is empty and this is just the G-counter sum).
    Under a partition this is ``node``'s *local* view (it may not yet include the other side's incs
    or decrements, but never *loses* them: a later ``gossip``/``anti_entropy`` max-merge adds them).
    A pure read; a crashed node is ``unavailable``.
    """
    node, key = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    total = str(state.pncounter_value(key, node))
    return [SetResult("ok", total)], "ok", total


def _orset_merge_edits(
    state: DistributedState, into: list[str], among: list[str]
) -> list[DistEdit]:
    """**Set-union** join of the CRDT OR-Set copies (DS0 incr 30, the CRDT join over both halves).

    A no-op when no OR-Set is used (the pre-incr-30 form is unchanged). See
    :func:`_dotset_union_edits` for the shared union mechanism. ``anti_entropy`` pulls one node.
    """
    return _dotset_union_edits(state.orset_adds, state.orset_tombs, into, among,
                               ORSetAdd, ORSetTomb)


def _dotset_union_edits(
    vals: dict[tuple[str, str], frozenset[tuple[str, str, int]]],
    tombs: dict[tuple[str, str], frozenset[tuple[str, int]]],
    into: list[str], among: list[str],
    mk_val: Callable[[str, str, str, str, int], DistEdit],
    mk_tomb: Callable[[str, str, str, int], DistEdit],
) -> list[DistEdit]:
    """**Set-union** join of a dotted CRDT — the shared OR-Set / MV-register merge (incr 30/31).

    For each node in ``into``, raise its value-dot set and tombstone set to the **union** held by
    any node in ``among`` (the reachable set), emitting ``mk_val``/``mk_tomb`` per missing dot.
    Union is commutative, associative, and idempotent, so applying it in any order converges every
    node to the same set of surviving dots. ``anti_entropy`` pulls one node; ``gossip`` a pair.
    """
    edits: list[DistEdit] = []
    keys = {key for (key, _h) in vals} | {key for (key, _h) in tombs}
    for key in sorted(keys):
        union_vals: set[tuple[str, str, int]] = set()
        union_tombs: set[tuple[str, int]] = set()
        for h in among:
            union_vals |= vals.get((key, h), frozenset())
            union_tombs |= tombs.get((key, h), frozenset())
        for holder in into:
            have_v = vals.get((key, holder), frozenset())
            for value, owner, seq in sorted(union_vals - have_v):
                edits.append(mk_val(key, holder, value, owner, seq))
            have_t = tombs.get((key, holder), frozenset())
            for owner, seq in sorted(union_tombs - have_t):
                edits.append(mk_tomb(key, holder, owner, seq))
    return edits


def _mvreg_merge_edits(
    state: DistributedState, into: list[str], among: list[str]
) -> list[DistEdit]:
    """**Set-union** join of the CRDT MV-register copies (incr 31). See :func:`_dotset_union_edits`.

    A no-op when no register is used (the pre-incr-31 form is unchanged).
    """
    return _dotset_union_edits(state.mvreg_vals, state.mvreg_tombs, into, among,
                               MVRegWrite, MVRegTomb)


def _lwwreg_merge_edits(
    state: DistributedState, into: list[str], among: list[str]
) -> list[DistEdit]:
    """**Max-by-(ts, owner, value)** join of the CRDT LWW-register copies (DS0 incr 32).

    For each register ``key`` held by any node in ``among``, the winner is the **max** copy by
    ``(ts, owner, value)`` — the Lamport-timestamp total order (a higher timestamp, i.e. a write
    that happened-after, wins; ties break by node id then value). Each node in ``into`` that is
    behind adopts the winner (``LWWRegSet``). Each node also advances its Lamport clock to the
    max timestamp seen across the reachable set (``LamportSet``), so a later local write always
    out-stamps what it overwrote — the logical-clock invariant that keeps the join convergent. No-op
    when no LWW-register is used (the pre-incr-32 form is unchanged).
    """
    edits: list[DistEdit] = []
    keys = {key for (key, _h) in state.lwwreg}
    for key in sorted(keys):
        copies = [state.lwwreg[(key, h)] for h in among if (key, h) in state.lwwreg]
        if not copies:
            continue
        # stored as (value, ts, owner); compare by (ts, owner, value) for the LWW order.
        win = max(copies, key=lambda e: (e[1], e[2], e[0]))
        for holder in into:
            if state.lwwreg.get((key, holder)) != win:
                edits.append(LWWRegSet(key, holder, win[0], win[1], win[2]))
    # advance each into-node's Lamport clock to the max ts observed across the reachable set.
    max_ts = max(
        [state.lamport.get(h, 0) for h in among]
        + [ts for (k, h), (v, ts, o) in state.lwwreg.items() if h in among],
        default=0,
    )
    for holder in into:
        if state.lamport.get(holder, 0) < max_ts:
            edits.append(LamportSet(holder, max_ts))
    return edits


def lwwput_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``lwwput node key val``: CRDT LWW-register write (incr 32, deterministic resolution).

    ``node`` stamps ``val`` with ``(ts, owner=node)`` where ``ts = lamport[node] + 1`` (advancing
    its Lamport clock) and stores it as the current register value (``LWWRegSet`` + ``LamportSet``);
    the join keeps the **max** copy by ``(ts, owner, value)``, so a write that *happened-after*
    another (higher ts) wins regardless of node, and concurrent writes (equal ts) resolve by node id
    to **one** deterministic winner, where the MV-register would keep siblings (ED38). Node-local
    (no replication, no message), so **always available** even partitioned-alone, and deterministic,
    so Tier-A ≡ Tier-B bit-for-bit. A crashed node is ``unavailable``. Reports the written value.
    """
    node, key, val = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    ts = state.lamport.get(node, 0) + 1
    edits: DistDelta = [LWWRegSet(key, node, val, ts, node), LamportSet(node, ts)]
    return [*edits, SetResult("ok", val)], "ok", val


def lwwget_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``lwwget node key``: read a CRDT LWW-register (DS0 increment 32) — the winning value.

    ``node``'s local view: its current ``(value, ts, owner)`` winner for ``key`` (or ``""`` if never
    written). Under a partition this may lag the other side's writes, but never resolves *wrongly*:
    a later join adopts the global ``(ts, owner, value)`` max. A pure read; ``unavailable`` if down.
    """
    node, key = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    val = state.lwwreg_value(key, node)
    return [SetResult("ok", val)], "ok", val


def _ormap_merge_edits(
    state: DistributedState, into: list[str], among: list[str]
) -> list[DistEdit]:
    """Join of the CRDT OR-Map (DS0 incr 33) — the **composition** of two prior CRDT joins.

    The presence half is the **OR-Set union** of the field-presence dots (reusing
    :func:`_dotset_union_edits`); the value half is the **LWW max** by ``(ts, owner, value)`` of
    each ``(map, field)`` value (the LWW-register join); each node's shared Lamport clock rises to
    the max ts observed. A no-op when no OR-Map is used (the pre-incr-33 form is unchanged).
    """
    edits: list[DistEdit] = _dotset_union_edits(
        state.ormap_fields, state.ormap_tombs, into, among, ORMapField, ORMapTomb
    )
    for mapname, field in sorted({mf for (mf, _h) in state.ormap_vals}):
        copies = [state.ormap_vals[((mapname, field), h)]
                  for h in among if ((mapname, field), h) in state.ormap_vals]
        if not copies:
            continue
        win = max(copies, key=lambda e: (e[1], e[2], e[0]))  # (ts, owner, value) LWW order
        for holder in into:
            if state.ormap_vals.get(((mapname, field), holder)) != win:
                edits.append(ORMapVal(mapname, field, holder, win[0], win[1], win[2]))
    max_ts = max(
        [state.lamport.get(h, 0) for h in among]
        + [ts for ((m, f), h), (v, ts, o) in state.ormap_vals.items() if h in among],
        default=0,
    )
    for holder in into:
        if state.lamport.get(holder, 0) < max_ts:
            edits.append(LamportSet(holder, max_ts))
    return edits


def mput_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``mput node map field val``: CRDT OR-Map field write (DS0 increment 33, the composition).

    Two coordinated edits in the two halves: (1) a fresh **presence dot** for ``field`` (superseding
    ``node``'s own observed dots of that field, so a sequential write keeps the presence bounded),
    and (2) an **LWW value** stamped with the node's Lamport clock. Add-wins: the fresh presence dot
    survives a concurrent ``mdel`` (whose tombstone never saw it), so a concurrent update beats a
    concurrent remove. Purely node-local, so **always available** even partitioned-alone, and
    deterministic, so Tier-A ≡ Tier-B bit-for-bit. A crashed node is ``unavailable``. Reports the
    written value.
    """
    node, mapname, field, val = action.args[0], action.args[1], action.args[2], action.args[3]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    already = state.ormap_tombs.get((mapname, node), frozenset())
    observed = state.ormap_field_dots(mapname, field, node)
    edits: DistDelta = [ORMapTomb(mapname, node, o, s) for o, s in sorted(observed - already)]
    seq = state.ormap_next_seq(mapname, node)
    edits.append(ORMapField(mapname, node, field, node, seq))
    ts = state.lamport.get(node, 0) + 1
    edits.append(ORMapVal(mapname, field, node, val, ts, node))
    edits.append(LamportSet(node, ts))
    return [*edits, SetResult("ok", val)], "ok", val


def mdel_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``mdel node map field``: CRDT OR-Map field *observed*-remove (DS0 increment 33).

    Tombstones **only the presence dots of ``field`` that ``node`` has observed** (the OR-Set
    observed-remove) — so a concurrent ``mput``'s fresh, unseen dot survives (add-wins). The field's
    LWW value stays (a re-``mput`` overwrites it), but ``mget``/``mkeys`` no longer see the field.
    Purely node-local (always available); removing an absent field is a no-op. Reports the field.
    """
    node, mapname, field = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    already = state.ormap_tombs.get((mapname, node), frozenset())
    observed = state.ormap_field_dots(mapname, field, node)
    edits: DistDelta = [ORMapTomb(mapname, node, o, s) for o, s in sorted(observed - already)]
    return [*edits, SetResult("ok", field)], "ok", field


def mget_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``mget node map field``: read an OR-Map field (DS0 increment 33) — its LWW value or ``""``.

    Returns the field's last-writer-wins value if the field is **present** (it has a non-tombstoned
    presence dot at ``node``), else ``""`` (absent/removed). ``node``'s local view; a later join
    reconciles. A pure read; ``unavailable`` if the node is down.
    """
    node, mapname, field = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    return [SetResult("ok", state.ormap_get(mapname, field, node))], "ok", \
        state.ormap_get(mapname, field, node)


def mkeys_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``mkeys node map``: read an OR-Map's present field names (DS0 increment 33).

    The fields with at least one non-tombstoned presence dot at ``node``, sorted and rendered
    ``{a,b}`` (``{}`` empty) — the enumeration capability the flat KV/registers lack. A pure read;
    ``unavailable`` if down.
    """
    node, mapname = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    val = "{" + ",".join(state.ormap_keys(mapname, node)) + "}"
    return [SetResult("ok", val)], "ok", val


def _rga_merge_edits(
    state: DistributedState, into: list[str], among: list[str]
) -> list[DistEdit]:
    """**Set-union** join of the CRDT RGA sequences (DS0 incr 34).

    For each list, each node in ``into`` adopts the union of the elements and tombstones held by any
    node in ``among`` (emitting ``RGAInsert``/``RGATomb`` per missing one). Union is commutative,
    associative, idempotent, and the visible order is a pure function of the element set — so every
    node converges to the same set and so the same sequence. A no-op when no RGA list is used.
    """
    edits: list[DistEdit] = []
    lists = {ln for (ln, _h) in state.rga_elems} | {ln for (ln, _h) in state.rga_tombs}
    for ln in sorted(lists):
        union_elems: set[tuple[int, str, str, int, str]] = set()
        union_tombs: set[tuple[int, str]] = set()
        for h in among:
            union_elems |= state.rga_elems.get((ln, h), frozenset())
            union_tombs |= state.rga_tombs.get((ln, h), frozenset())
        for holder in into:
            have_e = state.rga_elems.get((ln, holder), frozenset())
            for seq, owner, value, pseq, powner in sorted(union_elems - have_e):
                edits.append(RGAInsert(ln, holder, seq, owner, value, pseq, powner))
            have_t = state.rga_tombs.get((ln, holder), frozenset())
            for seq, owner in sorted(union_tombs - have_t):
                edits.append(RGATomb(ln, holder, seq, owner))
    return edits


def rins_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``rins node list i val``: CRDT RGA insert (DS0 increment 34, the ordered sequence).

    Insert ``val`` so it lands immediately after the **i-th visible element** (``i=0`` = the head,
    anchored at ``RGA_ROOT``; ``i`` beyond the end appends after the last element). The new element
    gets a unique id ``(seq, node)`` and records its anchor's id as its ``parent`` — so the visible
    order (a DFS with siblings by id descending) is a pure function of the element set, so
    concurrent inserts at the same anchor converge to one deterministic order. Purely node-local, so
    **always available** even partitioned-alone. Deterministic, so Tier-A ≡ Tier-B. A crashed node
    is ``unavailable``. Reports the inserted value.
    """
    node, list_name, i_str, val = action.args[0], action.args[1], action.args[2], action.args[3]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    visible = state.rga_visible(list_name, node)
    i = max(0, min(int(i_str), len(visible)))  # clamp into [0, len]; i=0 = head
    pseq, powner = RGA_ROOT if i == 0 else visible[i - 1][0]
    seq = state.rga_next_seq(list_name, node)
    edit = RGAInsert(list_name, node, seq, node, val, pseq, powner)
    return [edit, SetResult("ok", val)], "ok", val


def rdel_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``rdel node list i``: CRDT RGA delete (incr 34) — tombstone the i-th visible element.

    Tombstones the element at visible position ``i`` (``i=1`` = the first element), so it leaves the
    visible sequence but keeps its position as an anchor for its children (delete preserves
    structure). An out-of-range ``i`` is a no-op (``not_found``). Node-local (always available).
    Reports the deleted value.
    """
    node, list_name, i_str = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    visible = state.rga_visible(list_name, node)
    i = int(i_str)
    if i < 1 or i > len(visible):
        return [SetResult("not_found", "")], "not_found", ""
    (seq, owner), value = visible[i - 1]
    return [RGATomb(list_name, node, seq, owner), SetResult("ok", value)], "ok", value


def rget_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``rget node list``: read a CRDT RGA sequence (DS0 increment 34) — the visible values joined.

    The concatenation of the visible (non-tombstoned) element values, in order. ``node``'s view;
    a later join reconciles by union, so concurrent inserts converge to one string. A pure read;
    ``unavailable`` if the node is down.
    """
    node, list_name = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    val = "".join(v for _eid, v in state.rga_visible(list_name, node))
    return [SetResult("ok", val)], "ok", val


def sadd_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``sadd node key elem``: CRDT OR-Set add (DS0 increment 30, add-wins, re-addable).

    ``node`` tags ``elem`` with a **unique dot** ``(owner=node, seq)`` — its next monotone sequence
    for ``key`` — and stores it in its own observed add-set (``ORSetAdd``). Purely node-local (no
    replication, no message), so it is **always available** (a partitioned-alone node still adds);
    a fresh dot **survives** a concurrent ``srem`` that never observed it (add wins),
    and re-adding a removed element makes a new dot that is not tombstoned (re-addable). The union
    join in ``anti_entropy``/``gossip`` spreads it. Deterministic, so Tier-A ≡ Tier-B bit-for-bit. A
    crashed node is ``unavailable``. Reports the added element.
    """
    node, key, elem = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    seq = state.orset_next_seq(key, node)
    return [ORSetAdd(key, node, elem, node, seq), SetResult("ok", elem)], "ok", elem


def srem_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``srem node key elem``: CRDT OR-Set *observed*-remove (DS0 increment 30).

    ``node`` tombstones **only the dots of ``elem`` it has observed** in its own add-set
    (``ORSetTomb`` per dot) — the *observed*-remove that makes add-wins work: a concurrent ``sadd``
    whose dot ``node`` never saw is *not* tombstoned, so it survives. The dot stays in the add-set
    (union semantics); membership simply no longer counts it. Purely node-local (always available);
    removing an absent element is a no-op. Reports the removed element. Tier-A ≡ Tier-B bit-for-bit.
    """
    node, key, elem = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    observed = {(owner, seq) for (e, owner, seq) in state.orset_adds.get((key, node), frozenset())
                if e == elem}
    already = state.orset_tombs.get((key, node), frozenset())
    edits: DistDelta = [ORSetTomb(key, node, owner, seq)
                        for owner, seq in sorted(observed - already)]
    return [*edits, SetResult("ok", elem)], "ok", elem


def smembers_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``smembers node key``: read a CRDT OR-Set (DS0 increment 30) — non-tombstoned elements.

    ``node``'s local view: the elements with at least one add-dot not in its tombstone-set, sorted
    and rendered ``{a,b,c}`` (``{}`` when empty). Under a partition this may lag the other side's
    adds/removes, but never loses them (a later join reconciles by union). A pure read; the node is
    ``unavailable`` when down.
    """
    node, key = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    val = "{" + ",".join(state.orset_members(key, node)) + "}"
    return [SetResult("ok", val)], "ok", val


def mvput_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``mvput node key val``: CRDT MV-register write (DS0 increment 31, conflict-surfacing).

    ``node`` tags ``val`` with a fresh dot ``(owner=node, seq)``, **tombstones every dot it now
    observes** (the values it can see — a write supersedes what it saw), and stores its write-dot.
    So a *sequential* overwrite (the writer observed the prior value) collapses to one, but two
    *concurrent* writes — neither observing the other — **both survive** the union join as siblings,
    and a later context-aware ``mvput`` (observing both) **resolves** them. Purely node-local (no
    replication, no message), so **always available** even partitioned-alone. Deterministic: Tier-A
    ≡ Tier-B bit-for-bit. A crashed node is ``unavailable``. Reports the written value.
    """
    node, key, val = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    already = state.mvreg_tombs.get((key, node), frozenset())
    edits: DistDelta = [MVRegTomb(key, node, owner, seq)
                        for owner, seq in sorted(state.mvreg_observed_dots(key, node) - already)]
    seq = state.mvreg_next_seq(key, node)
    edits.append(MVRegWrite(key, node, val, node, seq))
    return [*edits, SetResult("ok", val)], "ok", val


def mvget_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``mvget node key``: read a CRDT MV-register (incr 31) — the surviving sibling values.

    ``node``'s local view: the values whose write-dot is not tombstoned, sorted and rendered
    ``{a,b}`` (``{}`` empty, one value if resolved, several if concurrent writes left siblings).
    Under a partition this may lag the other side's writes, but never loses them (a later join
    reconciles by union). A pure read; the node is ``unavailable`` when down.
    """
    node, key = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    val = "{" + ",".join(state.mvreg_value(key, node)) + "}"
    return [SetResult("ok", val)], "ok", val


def active_members(state: DistributedState, config: DistConfig) -> frozenset[str]:
    """The consensus voting set (DS0 increment 20): ``state.members`` resolved against the sentinel.

    The empty ``state.members`` is the "every config node votes" default, so a cluster that never
    reconfigures resolves to the full node set and every quorum computation is byte-identical to the
    pre-increment-20 form. ``add_replica``/``remove_replica`` install a non-empty reconfigured set.
    """
    return state.members if state.members else frozenset(config.nodes)


def _compatible(state: DistributedState, config: DistConfig, a: str, b: str) -> bool:
    """Whether nodes ``a`` and ``b`` may share a consensus quorum (DS0 increment 22, `deploy`).

    Two nodes interoperate in consensus iff their running versions are within
    ``config.max_version_skew`` (``1`` = the standard N-1 rolling-upgrade window). When no node has
    deployed (every version is the base ``0``) every pair is compatible, so this is byte-identical
    to the pre-increment-22 form. Gates *consensus* only — the KV/queue data plane ignores versions.
    """
    va = state.versions.get(a, 0)
    vb = state.versions.get(b, 0)
    return abs(va - vb) <= config.max_version_skew


def elect_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``elect node``: leader election (DS0 increment 16, the Raft-subset consensus core, §3.2).

    ``node`` becomes the cluster leader **iff its partition side holds a strict majority of the live
    cluster nodes** — the Raft quorum a candidate needs to win an election. The voters are the *up*
    nodes in ``node``'s partition group (a node can only collect votes from peers it can reach), and
    a strict majority is ``> len(nodes) // 2``. Because two disjoint partition groups can never each
    hold a majority, **at most one leader can be elected** — split-brain at the leadership level
    is structurally impossible (ED23, Panel A). On success the monotone ``term`` bumps and the
    global ``leader`` is installed; the new, higher term **fences** the previous leader (its
    ``propose`` is now rejected — Panel B). A crashed candidate is ``unavailable``; a candidate
    without a live majority is ``no_quorum``. Election touches no replica (leadership is cluster
    metadata, not data), so — like ``gossip``/``anti_entropy`` — it is a coordinator-level decision
    Tier-A and Tier-B compute byte-identically; shared by both oracles so they cannot drift.
    """
    node = action.args[0]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    # Leader-lease safety (DS0 incr 18): a successor must wait out the incumbent's *unexpired*
    # lease — leadership cannot change hands while a lease is live, which is exactly what makes a
    # leader's lease read (`lread`) safe. A voluntary `step_down` releases the lease, so this only
    # blocks the crash/contested path; once the lease expires (or was never taken) elect proceeds.
    if state.lease_until > 0 and state.clock < state.lease_until:
        until = str(state.lease_until)
        return [SetResult("lease_held", until)], "lease_held", until
    members = active_members(state, config)
    if node not in members:  # a non-member (removed from the voting set) cannot be elected leader
        return [SetResult("not_member", "")], "not_member", ""
    # The quorum is a strict majority of the *voting membership* (DS0 incr 20), which equals the
    # full cluster until `remove_replica`/`add_replica` reconfigures it — so this is byte-identical
    # to the pre-increment-20 form when no membership change has happened.
    # voters: reachable (same partition group), up, voting members within the version-compatibility
    # window of the candidate (DS0 incr 22) — an incompatible (mid-upgrade) node cannot vote.
    voters = [n for n in state.group_of(node)
              if state.is_up(n) and n in members and _compatible(state, config, node, n)]
    if len(voters) <= len(members) // 2:  # a strict majority of the voting members is required
        return [SetResult("no_quorum", "")], "no_quorum", ""
    new_term = state.term + 1
    edits: DistDelta = [ProtocolStep("elect", new_term, node)]
    if state.lease_until != 0:  # a new term starts fresh: clear any (now-expired) incumbent lease
        edits.append(LeaseSet(0))
    edits.append(SetResult("elected", str(new_term)))
    return edits, "elected", str(new_term)


def propose_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``propose node key val``: a leader-fenced consensus write (DS0 increment 16, §3.2).

    The consensus counterpart of ``put``: it commits ``val`` to ``key`` only if ``node`` is the
    **current cluster leader** and can reach a **majority** of the replicas — the Raft commit
    rule, applied regardless of the KV ``consistency_model`` because consensus *is* majority-quorum.
    Two rejections encode the safety property plain ``quorum`` writes lack:

      - ``not_leader`` — ``node`` is not the leader. A leader deposed by a higher-term election
        (e.g. partitioned into the minority while the majority side elected a new leader) is fenced
        here **even after the partition heals**, because the global leader already moved on — the
        Raft leader-completeness property (ED23, Panel B). The result value carries the *current*
        leader (or ``""`` if none), so the rejection is diagnostic.
      - ``no_quorum`` — the leader cannot reach a majority of the replicas (a leader stranded in
        a minority cannot commit), so a write never proceeds on a side that cannot durably hold it.

    On commit it writes the reachable majority synchronously and queues async catch-up messages to
    the unreachable minority (delivered later by ``advance``), exactly as a ``quorum`` ``put`` does,
    so it commits once and never forks. A coordinator-level decision (the majority set is read
    from the medium, not an actor's local view), so Tier-A and Tier-B compute byte-identical deltas;
    shared by both oracles so they cannot drift.
    """
    node, key, val = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    if state.leader != node:
        cur = state.leader or ""
        return [SetResult("not_leader", cur)], "not_leader", cur
    replica = state.replicas.get((key, node))
    if replica is None:
        return [SetResult("no_replica", "")], "no_replica", ""
    # The consensus quorum is over the *voting members* that replicate the key (DS0 incr 20); when
    # membership is the full cluster (the default) this is byte-identical to the pre-incr-20 form.
    peers = [p for p in config.replicas_of(key) if p in active_members(state, config)]
    # reachable: co-partitioned, up, and within the version-compatibility window of the leader
    # (DS0 incr 22) — an incompatible node cannot acknowledge the consensus write.
    reachable = [
        p for p in peers
        if state.connected(node, p) and state.is_up(p) and _compatible(state, config, node, p)
    ]
    if len(reachable) < len(peers) // 2 + 1:  # consensus commits only on a majority of replicas
        return [SetResult("no_quorum", "")], "no_quorum", ""
    new_version = replica.version + 1
    edits: DistDelta = [ReplicaWrite(key, peer, new_version, val) for peer in reachable]
    msg_id = state.next_msg_id
    deliver_at = state.sender_clock(node) + 1
    for peer in peers:
        if peer in reachable:
            continue
        edits.append(MsgSend(msg_id, node, peer, key, new_version, val, deliver_at))
        msg_id += 1
    edits.append(SetResult("ok", val))
    return edits, "ok", val


def step_down_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``step_down node``: voluntary leadership relinquishment (DS0 increment 17, §3.2).

    The **graceful** counterpart of ED23's term-fencing: where ``elect`` *deposes* a leader by a
    higher term (an involuntary loss of power, fenced after heal — ED23 Panel B), ``step_down`` lets
    the *current* leader hand back power on its own, leaving the cluster **leaderless at the same
    term** (``leader → None``, ``term`` unchanged). The term machinery then closes the gap the same
    way it fences a deposed leader: until a fresh ``elect`` bumps the term and installs a successor,
    every ``propose`` is rejected (``not_leader``) — even one issued by the node that just stepped
    down. So a voluntary handoff is `step_down` then `elect <successor>` (a strictly higher term),
    and there is no window in which a leaderless cluster commits a consensus write (ED24, Panel A).

    Two properties distinguish it from the data-plane ops:

      - ``not_leader`` — only the *current* leader can step down; any other node (including one in a
        cluster that already has no leader) is rejected, the result carrying the current leader (or
        ``""``) for diagnosis. So ``step_down`` is idempotently safe: a second is a no-op reject.
      - **partition-independent** — relinquishing power needs no quorum, so a leader **stranded in a
        minority can still step down** (it reads only its own leadership, never the medium), where
        the same leader's ``propose`` there is ``no_quorum``. Giving up authority is always safe;
        committing under it is not (ED24, Panel B). A crashed leader is ``unavailable``.

    Like ``elect``, it touches no replica — leadership is cluster metadata — so Tier-A and Tier-B
    compute byte-identical leader/term deltas; shared by both oracles so they cannot drift.
    """
    node = action.args[0]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    if state.leader != node:
        cur = state.leader or ""
        return [SetResult("not_leader", cur)], "not_leader", cur
    # Relinquish: clear the leader, hold the term (a successor's `elect` bumps it strictly higher).
    # A voluntary handoff also *releases* the lease immediately (DS0 incr 18), so a successor need
    # not wait it out — the fast-handoff path, vs a crashed leader whose lease the cluster outlasts.
    edits: DistDelta = [ProtocolStep("step_down", state.term, None)]
    if state.lease_until != 0:
        edits.append(LeaseSet(0))
    edits.append(SetResult("stepped_down", str(state.term)))
    return edits, "stepped_down", str(state.term)


def lease_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``lease node dt``: the leader takes a read lease through clock+dt (DS0 increment 18, §4).

    The Raft **leader lease** — a read optimization. Only the *current* leader may take one; it sets
    the global-clock deadline ``state.clock + dt`` through which the leader may serve local reads
    without a quorum (`lread`) and through which a new election is fenced (`elect` → `lease_held`).
    A non-leader is ``not_leader``; a crashed node is ``unavailable``. Renewing simply re-stamps the
    deadline from *now*. Leadership/lease are coordinator-level cluster metadata in this model (like
    ``term``/``leader``), so Tier-A and Tier-B compute byte-identical deltas via this shared helper.
    """
    node = action.args[0]
    dt = int(action.args[1])
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    if state.leader != node:
        cur = state.leader or ""
        return [SetResult("not_leader", cur)], "not_leader", cur
    until = state.clock + dt
    return [LeaseSet(until), SetResult("leased", str(until))], "leased", str(until)


def lread_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``lread node key``: a leader-lease local linearizable read (DS0 increment 18, §4).

    The payoff of the lease: the leader serves ``key`` from its *own* replica with **no quorum
    round-trip**, linearizable w.r.t. consensus (`propose`) writes because (i) the leader is always
    in a `propose`'s commit majority, so its local replica holds every committed value, and (ii) a
    live lease guarantees no other leader was elected (a new `elect` is blocked until expiry). So
    a leader **partitioned into the minority can still `lread`** while its lease holds — where its
    `propose` is ``no_quorum`` — the read-availability the lease buys. Rejections: ``not_leader`` if
    ``node`` is not the leader (a deposed leader is fenced, so it cannot serve a stale read off a
    stale lease); ``lease_expired`` if no live lease (the leader must renew or fall back to a quorum
    read); ``unavailable``/``no_replica`` as usual. Touches no replica — a pure read.
    """
    node, key = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    if state.leader != node:
        cur = state.leader or ""
        return [SetResult("not_leader", cur)], "not_leader", cur
    if not (state.lease_until > 0 and state.clock < state.lease_until):
        return [SetResult("lease_expired", "")], "lease_expired", ""
    replica = state.replicas.get((key, node))
    if replica is None:
        return [SetResult("no_replica", "")], "no_replica", ""
    return [SetResult("ok", replica.value)], "ok", replica.value


def read_index_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``read_index node key``: a quorum-confirmed linearizable read (DS0 incr 25, Raft ReadIndex).

    The **partner to the lease read** (``lread``, incr 18) — the *other* way Raft serves a
    linearizable read. Where ``lread`` skips the quorum round-trip by relying on a time **lease**,
    ``read_index`` keeps no clock assumption and instead **confirms leadership with a majority**
    before serving the read (the Raft ReadIndex heartbeat round): the leader is always in a
    ``propose``/``append`` commit majority, so once a majority still acknowledges it as leader its
    own replica holds every committed value and the local read is linearizable. The two reads have
    **opposite availability profiles**:

      - ``not_leader`` — only the leader may serve it (a deposed leader is fenced even after a
        ``heal``: leader-completeness reaches the read path, so no stale read off a stale leader);
        the result carries the current leader.
      - ``no_quorum`` — a leader **stranded in a minority** cannot confirm it is still leader, so it
        **refuses** the read (where ``lread`` with a live lease would still serve it locally — the
        availability the lease buys and the quorum read declines, the safety it buys in return).
      - ``("ok", value)`` — leadership confirmed by a majority; the leader's committed replica.
      - ``unavailable`` (crashed) / ``no_replica`` as usual.

    Like ``lread``/``propose`` it touches no replica (a pure read) and reads the majority from the
    partition/down medium (a coordinator-level decision), so Tier-A and Tier-B compute the
    byte-identical verdict via this shared helper.
    """
    node, key = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    if state.leader != node:
        cur = state.leader or ""
        return [SetResult("not_leader", cur)], "not_leader", cur
    # Confirm leadership via a majority of the voting members (the ReadIndex heartbeat round); the
    # leader is always co-partitioned/up/compatible with itself, so a single-node cluster confirms
    # trivially and this is byte-identical to the propose/append majority rule.
    members = active_members(state, config)
    reachable = [
        p for p in members
        if state.connected(node, p) and state.is_up(p) and _compatible(state, config, node, p)
    ]
    if len(reachable) < len(members) // 2 + 1:  # cannot confirm leadership -> refuse the read
        return [SetResult("no_quorum", "")], "no_quorum", ""
    replica = state.replicas.get((key, node))
    if replica is None:
        return [SetResult("no_replica", "")], "no_replica", ""
    return [SetResult("ok", replica.value)], "ok", replica.value


def append_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``append node key val``: Raft replicated-log append + majority commit (DS0 incr 19, §5.1).

    The replicated-log path the spec named since increment 1 — what the one-shot ``propose`` (incr
    16) elided. The current leader appends a ``LogEntry(term, index, key, value)`` to its log and
    replicates it to the **reachable** followers, who **adopt the leader's full log** in one step —
    appending the new entry and, where their tail diverged, truncating the conflicting *uncommitted*
    entries (the **log-matching reconciliation**: a follower's log always becomes a prefix-
    consistent copy of the leader's). The entry **commits iff a majority holds it**, advancing
    the monotone ``commit_index``; the committed prefix is then folded into the KV state machine on
    every reachable node — a key's version is the count of its committed writes, value the last —
    which also **backfills a rejoined follower** that missed committed entries while partitioned.

    Two rejections + two outcomes:

      - ``unavailable`` (crashed) / ``not_leader`` (only the leader appends; result carries leader).
      - ``("appended", str(index))`` — the entry reached a majority and committed (applied to KV).
      - ``("uncommitted", str(index))`` — a minority-stranded leader appended the entry to its (and
        the reachable followers') log, but it did **not** commit, so it is **not** applied to the KV
        and may be overwritten by a higher-term leader's entry at the same index (the safety the
        one-shot ``propose`` could not express — ED26).

    Like ``propose``, the majority set is read from the partition/down medium (a coordinator-level
    decision), so Tier-A and Tier-B compute byte-identical deltas via this shared helper.
    """
    node, key, val = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    if state.leader != node:
        cur = state.leader or ""
        return [SetResult("not_leader", cur)], "not_leader", cur
    leader_log = state.logs.get(node, ())
    index = len(leader_log)
    new_log = (*leader_log, LogEntry(state.term, index, key, val))
    # Replicate to / commit on the *voting members* (DS0 incr 20); the full cluster by default, so
    # this is byte-identical to the pre-increment-20 form when no membership change has happened.
    members = active_members(state, config)
    # reachable: co-partitioned, up, and version-compatible with the leader (DS0 incr 22).
    reachable = [
        p for p in members
        if state.connected(node, p) and state.is_up(p) and _compatible(state, config, node, p)
    ]
    committed = len(reachable) >= len(members) // 2 + 1
    edits: DistDelta = []
    # every reachable node adopts the leader's log (reconciling any divergent uncommitted tail)
    for p in reachable:
        if state.logs.get(p, ()) != new_log:
            edits.append(LogSet(p, new_log))
    if not committed:
        edits.append(SetResult("uncommitted", str(index)))
        return edits, "uncommitted", str(index)
    new_commit = index + 1
    edits.append(CommitIndexSet(new_commit))
    # fold the committed prefix into the KV state machine: a key's version is the count of its
    # committed writes, its value the last. Applied to every reachable node, backfilling any that
    # missed committed entries while partitioned (so the KV never diverges from the committed log).
    folded: dict[str, tuple[int, str]] = {}
    for e in new_log[:new_commit]:
        prev = folded.get(e.key)
        folded[e.key] = ((prev[0] + 1) if prev else 1, e.value)
    for p in reachable:
        for k, (ver, value) in folded.items():
            rep = state.replicas.get((k, p))
            if rep is None or (rep.version, rep.value) != (ver, value):
                edits.append(ReplicaWrite(k, p, ver, value))
    edits.append(SetResult("appended", str(index)))
    return edits, "appended", str(index)


def add_replica_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``add_replica node``: grow the consensus voting membership (DS0 increment 20, §3.2).

    A leader-committed reconfiguration: ``node`` joins the voting set, so the **majority threshold
    grows** (a 3-member cluster needs 2 to commit; a 4-member cluster needs 3). All config nodes
    already store replicas, so this is purely a *voting-set* change. Rejections: ``unknown_node`` if
    ``node`` is not a config node; ``no_leader`` if there is no current leader to commit the change
    (membership goes through the leader, like a Raft config entry); ``already_member`` (a no-op) if
    ``node`` already votes. On success the new set is installed via ``MemberSet`` — restoring the
    full cluster stores the empty sentinel, so the canonical form stays clean. Touches no replica.
    """
    node = action.args[0]
    if node not in config.nodes:
        return [SetResult("unknown_node", "")], "unknown_node", ""
    if state.leader is None:
        return [SetResult("no_leader", "")], "no_leader", ""
    members = active_members(state, config)
    if node in members:
        return [SetResult("already_member", node)], "already_member", node
    new = members | {node}
    # restoring the full cluster collapses back to the empty "all vote" sentinel (clean canonical)
    installed = frozenset() if new == frozenset(config.nodes) else new
    return [MemberSet(installed), SetResult("added", node)], "added", node


def remove_replica_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``remove_replica node``: shrink the consensus voting membership (DS0 increment 20, §3.2).

    The availability lever: removing a (typically failed) node from the voting set **shrinks the
    majority threshold**, so the surviving members can keep committing — the standard way to restore
    progress after losing nodes (a 3-member cluster with 2 down is stuck at majority 2; remove the 2
    dead and the lone survivor is a majority of 1). Rejections: ``unknown_node``; ``no_leader``;
    ``not_member`` (a no-op) if ``node`` does not vote; ``is_leader`` — the *active leader* cannot
    be removed (step it down first), so a reconfiguration never strands the cluster leaderless
    mid-write; ``last_member`` — the final member cannot be removed. Touches no replica (a removed
    node keeps its now-non-voting replicas). A leader-committed change, identical Tier-A/Tier-B.
    """
    node = action.args[0]
    if node not in config.nodes:
        return [SetResult("unknown_node", "")], "unknown_node", ""
    if state.leader is None:
        return [SetResult("no_leader", "")], "no_leader", ""
    if node == state.leader:
        return [SetResult("is_leader", node)], "is_leader", node
    members = active_members(state, config)
    if node not in members:
        return [SetResult("not_member", node)], "not_member", node
    new = members - {node}
    if not new:
        return [SetResult("last_member", node)], "last_member", node
    return [MemberSet(new), SetResult("removed", node)], "removed", node


def _queue_available(state: DistributedState, config: DistConfig, reachable: list[str]) -> bool:
    """Whether a queue write may proceed under the configured consistency model (DS0 incr 21).

    Queues are fully replicated (every node), so — mirroring the KV write's availability — a
    ``linearizable`` op needs *every* replica reachable and a ``quorum`` op a strict majority, while
    ``eventual``/``causal`` proceed on whatever is reachable. This is the single knob that turns the
    at-least-once / exactly-once delivery tradeoff (ED28).
    """
    model = config.consistency_model
    if model == "linearizable":
        return len(reachable) == len(config.nodes)
    if model == "quorum":
        return len(reachable) >= len(config.nodes) // 2 + 1
    return True  # eventual / causal: available on the coordinator + whatever peers are reachable


def enqueue_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``enqueue node queue val``: append to a replicated FIFO queue (DS0 increment 21, §3.2).

    Appends ``val`` to **each reachable replica's own** queue list (the relative append op, so
    in-sync replicas stay in sync and diverged ones each grow by one). Availability follows the
    consistency model (``_queue_available``): ``linearizable`` needs every replica, ``quorum`` a
    majority, ``eventual``/``causal`` proceed on the reachable set; an unavailable op is rejected.
    Queues are fully replicated, so the reachable set is the coordinator's co-partitioned, up peers.
    A coordinator-level decision (the reachable set is read from the medium), so Tier-A and Tier-B
    compute byte-identical deltas via this shared helper.
    """
    node, queue, val = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    reachable = [p for p in config.nodes if state.connected(node, p) and state.is_up(p)]
    if not _queue_available(state, config, reachable):
        return [SetResult("unavailable", "")], "unavailable", ""
    edits: DistDelta = [
        QueueSet(queue, p, (*state.queues.get((queue, p), ()), val)) for p in reachable
    ]
    edits.append(SetResult("enqueued", val))
    return edits, "enqueued", val


def dequeue_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``dequeue node queue``: pop the head of a replicated FIFO queue (DS0 increment 21, §3.2).

    Returns the head of the *coordinator's* local queue replica and pops the head from **each
    reachable replica's own** list. The delivery semantics are exactly the consistency-model knob:

      - ``eventual`` / ``causal`` — proceed on the reachable set, so under a partition the head-
        removal does **not** reach the other side, and a peer there can ``dequeue`` the **same item
        again** (at-least-once / **duplicate delivery**) — the queue's availability cost.
      - ``linearizable`` (every replica) / ``quorum`` (a majority) — gated by ``_queue_available``,
        so a partitioned ``dequeue`` is ``unavailable`` rather than duplicating: **exactly-once** on
        the side that can serve it, the queue analogue of the KV CAP tradeoff (ED28).

    ``empty`` if the coordinator's replica has no items. A coordinator-level decision shared by both
    oracles, so Tier-A ≡ Tier-B compute byte-identical deltas.
    """
    node, queue = action.args[0], action.args[1]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    reachable = [p for p in config.nodes if state.connected(node, p) and state.is_up(p)]
    if not _queue_available(state, config, reachable):
        return [SetResult("unavailable", "")], "unavailable", ""
    items = state.queues.get((queue, node), ())
    if not items:
        return [SetResult("empty", "")], "empty", ""
    head = items[0]
    edits: DistDelta = []
    for p in reachable:  # pop each reachable replica's own head (the relative dequeue op)
        p_items = state.queues.get((queue, p), ())
        if p_items:
            edits.append(QueueSet(queue, p, p_items[1:]))
    edits.append(SetResult("dequeued", head))
    return edits, "dequeued", head


def deploy_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``deploy node version``: a rolling-upgrade admin op (DS0 increment 22, §3.2).

    Sets ``node``'s running software version (a node-local change — no consensus needed to restart a
    node with new code). The consequence is in the consensus quorum: two nodes interoperate only if
    their versions are within ``config.max_version_skew`` (`_compatible`), so a deploy that makes an
    **incompatible version split with no compatible majority** turns the next `elect`/`propose`/
    `append` into `no_quorum` — *the deploy broke the cluster* (ED29). A rolling upgrade that stays
    inside the window keeps a compatible majority and never breaks. ``unknown_node`` if ``node`` is
    not a config node. A pure metadata change touching no replica, so Tier-A ≡ Tier-B bit-for-bit.
    """
    node, version = action.args[0], int(action.args[1])
    if node not in config.nodes:
        return [SetResult("unknown_node", "")], "unknown_node", ""
    v = str(version)
    return [VersionSet(node, version), SetResult("deployed", v)], "deployed", v


def config_push_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``config_push node key val``: a leader-committed cluster config change (DS0 incr 24, §3.2).

    SPEC-7's *other* headline operational question — *"will this config push break the cluster?"*.
    Unlike ``deploy`` (a node-local version label that gates consensus *compatibility*), a config
    push is a **leader-committed, majority-replicated** setting — a Raft-style config entry — so it
    shares the leader-fence + majority-reachability rule of ``propose``/``append``:

      - ``not_leader`` — only the current leader may push config (result carries the current leader,
        or ``""``), so a deposed/non-leader push is fenced.
      - ``no_quorum`` — a leader stranded in a minority cannot reach a majority of the voting
        members, so the push **does not commit** and **no node's config changes** (a config rollout
        is all-or-nothing at commit; a minority side never installs a value it cannot durably hold).
      - ``("committed", val)`` — the value is written (``ConfigSet``) to every **reachable** voting
        member. Under a partition that leaves the leader on the majority side this reaches only the
        majority, so the **partitioned minority retains its stale config** — the config-divergence
        outcome (ED31, Panel B), repaired by re-pushing after ``heal``.

    Like ``propose``/``append`` the majority set is read from the partition/down medium (a
    coordinator-level decision, not an actor's local view), so Tier-A and Tier-B compute byte-
    identical config deltas via this shared helper; shared by both oracles so they cannot drift.
    """
    node, key, val = action.args[0], action.args[1], action.args[2]
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    if state.leader != node:
        cur = state.leader or ""
        return [SetResult("not_leader", cur)], "not_leader", cur
    # The commit quorum is over the *voting members* (DS0 incr 20); the full cluster by default, so
    # this is byte-identical to the pre-increment-20 form when no membership change has happened.
    members = active_members(state, config)
    # reachable: co-partitioned, up, and version-compatible with the leader (DS0 incr 22) — an
    # incompatible (mid-upgrade) node cannot acknowledge the config commit.
    reachable = [
        p for p in members
        if state.connected(node, p) and state.is_up(p) and _compatible(state, config, node, p)
    ]
    if len(reachable) < len(members) // 2 + 1:  # config commits only on a majority of voters
        return [SetResult("no_quorum", "")], "no_quorum", ""
    edits: DistDelta = [
        ConfigSet(p, key, val) for p in reachable if state.config.get((p, key)) != val
    ]
    edits.append(SetResult("committed", val))
    return edits, "committed", val


def host_op_edits(
    state: DistributedState, action: DistAction, config: DistConfig
) -> tuple[DistDelta, str, str]:
    """``host node <syscall...>``: a SPEC-6 syscall on a node's embedded host (incr 23, §3.1/§4).

    The compositional vision: each cluster node runs a real SPEC-6 host (process table + fd tables +
    an embedded v0 filesystem), so a node is not just a bag of KV replicas. The syscall is delegated
    to the SPEC-6 :class:`~verisim.hostoracle.reference.ReferenceHostOracle` on **this node's** host
    state (created lazily, ``HostState.initial()``), and its bundle delta is wrapped in a
    ``HostStep`` edit — the §4 ``HostDelta`` on an embedded subsystem. Per-node **isolated** (a
    ``fork`` on one node never touches another's host), and host ops respect the node's up/down
    status: a crashed node's host is ``unavailable`` (the cross-layer crash linkage). The host
    result maps to ``("ok", stdout)`` on success and ``("host_err", "")`` on a syscall failure
    (EPERM/EBADF/bad pid). A node-local computation, so Tier-A and Tier-B compute identical host
    deltas via this shared helper.
    """
    node = action.args[0]
    if node not in config.nodes:
        return [SetResult("unknown_node", "")], "unknown_node", ""
    if not state.is_up(node):
        return [SetResult("unavailable", "")], "unavailable", ""
    host_state = state.hosts.get(node, HostState.initial())
    host_action = parse_host_action(" ".join(action.args[1:]))
    result = _HOST_ORACLE.step(host_state, host_action)
    status = "ok" if result.exit_code == EXIT_OK else "host_err"
    value = result.stdout
    return [HostStep(node, result.delta), SetResult(status, value)], status, value


class ReferenceDistOracle:
    """The Tier-A deterministic DES (§5.1). Pure: ``step`` is a function of (state, action)."""

    def __init__(self, config: DistConfig = DEFAULT_DIST_CONFIG) -> None:
        self.config = config

    def step(self, state: DistributedState, action: DistAction) -> DistStepResult:
        delta, status, value = self._delta_for(state, action)
        return DistStepResult(apply(state, delta), delta, status, value)

    # --- per-op semantics ------------------------------------------------------------------------

    def _delta_for(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        name = action.name
        if name in ("put", "cas", "delete", "incr"):
            return self._write(state, action)
        if name == "get":
            return self._get(state, action)
        if name == "cincr":
            return cincr_edits(state, action, self.config)
        if name == "cdecr":
            return cdecr_edits(state, action, self.config)
        if name == "cget":
            return cget_edits(state, action, self.config)
        if name == "sadd":
            return sadd_edits(state, action, self.config)
        if name == "srem":
            return srem_edits(state, action, self.config)
        if name == "smembers":
            return smembers_edits(state, action, self.config)
        if name == "mvput":
            return mvput_edits(state, action, self.config)
        if name == "mvget":
            return mvget_edits(state, action, self.config)
        if name == "lwwput":
            return lwwput_edits(state, action, self.config)
        if name == "lwwget":
            return lwwget_edits(state, action, self.config)
        if name == "mput":
            return mput_edits(state, action, self.config)
        if name == "mget":
            return mget_edits(state, action, self.config)
        if name == "mdel":
            return mdel_edits(state, action, self.config)
        if name == "mkeys":
            return mkeys_edits(state, action, self.config)
        if name == "rins":
            return rins_edits(state, action, self.config)
        if name == "rdel":
            return rdel_edits(state, action, self.config)
        if name == "rget":
            return rget_edits(state, action, self.config)
        if name in ("begin", "tget", "tput", "commit", "abort"):
            # The transaction family (DS0 increment 2) — shared OCC logic; a committed write's
            # async replication flows through the same in-flight medium as ``put`` (delivered by
            # ``advance``), so it composes with the fault/time semantics unchanged.
            return txn_step(state, action, self.config)
        if name == "advance":
            return self._advance(state, action)
        if name == "partition":
            return self._partition(state, action)
        if name == "heal":
            return self._heal(state, action)
        if name in ("crash", "restart"):
            return self._node_status(state, action)
        if name == "drop":
            return self._drop(state, action)
        if name in ("delay", "reorder"):
            return timing_fault_edits(state, action)
        if name == "clock_skew":
            return clock_skew_edits(action)
        if name == "anti_entropy":
            return self._anti_entropy(state, action)
        if name == "gossip":
            return gossip_edits(state, action, self.config)
        if name == "elect":
            return elect_edits(state, action, self.config)
        if name == "propose":
            return propose_edits(state, action, self.config)
        if name == "step_down":
            return step_down_edits(state, action, self.config)
        if name == "lease":
            return lease_edits(state, action, self.config)
        if name == "lread":
            return lread_edits(state, action, self.config)
        if name == "read_index":
            return read_index_edits(state, action, self.config)
        if name == "append":
            return append_edits(state, action, self.config)
        if name == "add_replica":
            return add_replica_edits(state, action, self.config)
        if name == "remove_replica":
            return remove_replica_edits(state, action, self.config)
        if name == "enqueue":
            return enqueue_edits(state, action, self.config)
        if name == "dequeue":
            return dequeue_edits(state, action, self.config)
        if name == "deploy":
            return deploy_edits(state, action, self.config)
        if name == "config_push":
            return config_push_edits(state, action, self.config)
        if name == "host":
            return host_op_edits(state, action, self.config)
        raise ValueError(f"unhandled action {name!r}")  # pragma: no cover - grammar is closed

    def _event(self, state: DistributedState, node: str, raw: str) -> EventAppend:
        """A causal-log event for a client op (program-order happens-before on the same node)."""
        prior = tuple(e.id for e in state.log if e.node == node)
        return EventAppend(state.next_event_id, node, raw, state.clock, prior)

    def _write(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        """``put`` / ``cas``: write the coordinator's replica + replicate (async or synchronous).

        Four declared consistency models share this entry (SPEC-7 §3.4, §5.1; H20):

        - ``eventual`` / ``causal`` (the default family): write the coordinator's local replica now
          and enqueue **async** replication messages; peers converge later on ``advance``, so reads
          are stale under partition. ``causal`` additionally tags each message with the writer's
          observed versions (``deps``) so delivery respects happens-before (§3.4, ED13).
        - ``linearizable``: replicate **synchronously** to **every** replica in the same step (no
          in-flight messages, so no replica is ever stale) — but, being a CP system, a write that
          cannot reach all of an object's replicas is **rejected** (``unavailable``). Strong
          consistency trades availability under *any* partition for the absence of divergence (CAP).
        - ``quorum`` (the Raft-subset consensus model, DS0 increment 7): replicate **synchronously**
          to the reachable majority and reject only when a majority is *not* reachable. So a write
          commits iff the coordinator's side holds a strict majority of the object's replicas — the
          realistic consensus availability: available under a *minority* partition (the majority
          side keeps working) where ``linearizable`` is not, yet still CP (only one side can ever
          hold the majority, so the object never forks — no split-brain). The unreachable minority
          catches up **asynchronously** (one ``MsgSend`` each, delivered on ``heal``+``advance``):
          a minority replica is stale until it rejoins but never divergent.
        """
        node, key = action.args[0], action.args[1]
        ev = self._event(state, node, action.raw)
        replica = state.replicas.get((key, node))
        if not state.is_up(node) or replica is None:
            status = "unavailable" if not state.is_up(node) else "no_replica"
            return [ev, SetResult(status, "")], status, ""

        model = self.config.consistency_model
        peers = self.config.replicas_of(key)
        # The replicas the coordinator can synchronously reach (itself + co-partitioned, up peers).
        reachable = [p for p in peers if state.connected(node, p) and state.is_up(p)]

        # CP write-rejection: ``linearizable`` needs *all* replicas, ``quorum`` a strict majority.
        if model == "linearizable" and len(reachable) < len(peers):
            return [ev, SetResult("unavailable", "")], "unavailable", ""
        if model == "quorum" and len(reachable) < len(peers) // 2 + 1:
            return [ev, SetResult("unavailable", "")], "unavailable", ""

        if action.name == "cas":
            old, new = action.args[2], action.args[3]
            if replica.value != old:
                return [ev, SetResult("conflict", replica.value)], "conflict", replica.value
            value = new
        elif action.name == "delete":
            # A delete is a versioned write of a tombstone (DS0 incr 26) — same replication path as
            # put, so it inherits every consistency model and is version-ordered (resurrection-safe)
            value = TOMBSTONE
        elif action.name == "incr":
            # An atomic counter read-modify-write (DS0 incr 27): read the coordinator's local count
            # (a non-numeric/absent value is 0) and write count+1. Because it reuses the put path,
            # two concurrent incrs on partitioned replicas both read the same count and write the
            # same next version — so eventual LWW keeps only one (a lost update); quorum/lin
            # gate the minority out (no silent loss). The read-modify-write CAP tradeoff (ED34).
            cur = int(replica.value) if replica.value.isdigit() else 0
            value = str(cur + 1)
        else:
            value = action.args[2]

        new_version = replica.version + 1
        # The client-visible result: a delete reports ``("deleted", "")`` rather than leaking the
        # tombstone sentinel; the *replicated value* is still the tombstone (the wire/state form).
        rstatus, rvalue = ("deleted", "") if action.name == "delete" else ("ok", value)
        if model == "linearizable":
            # synchronous: every replica gets the new version now; no in-flight, no staleness.
            edits: list[DistEdit] = [ev]
            edits.extend(ReplicaWrite(key, peer, new_version, value) for peer in peers)
            edits.append(SetResult(rstatus, rvalue))
            return edits, rstatus, rvalue

        if model == "quorum":
            # synchronous to the reachable majority; async catch-up (MsgSend) to the unreachable
            # minority, so the object commits once but never forks (only a majority side can write).
            edits = [ev]
            edits.extend(ReplicaWrite(key, peer, new_version, value) for peer in reachable)
            msg_id = state.next_msg_id
            for peer in peers:
                if peer in reachable:
                    continue
                edits.append(
                    MsgSend(msg_id, node, peer, key, new_version, value,
                            state.sender_clock(node) + 1)
                )
                msg_id += 1
            edits.append(SetResult(rstatus, rvalue))
            return edits, rstatus, rvalue

        # eventual / causal: local write now + async replication to every other replica.
        edits = [ev, ReplicaWrite(key, node, new_version, value)]
        deps = causal_deps(state, node, key) if model == "causal" else ()
        msg_id = state.next_msg_id
        for peer in peers:
            if peer == node:
                continue
            edits.append(
                MsgSend(msg_id, node, peer, key, new_version, value,
                        state.sender_clock(node) + 1, deps)
            )
            msg_id += 1
        edits.append(SetResult(rstatus, rvalue))
        return edits, rstatus, rvalue

    def _get(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        node, key = action.args[0], action.args[1]
        ev = self._event(state, node, action.raw)
        replica = state.replicas.get((key, node))
        if not state.is_up(node) or replica is None:
            status = "unavailable" if not state.is_up(node) else "no_replica"
            return [ev, SetResult(status, "")], status, ""
        if replica.value == TOMBSTONE:  # a deleted key reads as deleted (DS0 incr 26); see the
            return [ev, SetResult("deleted", "")], "deleted", ""  # resurrection note in `delete`
        return [ev, SetResult("ok", replica.value)], "ok", replica.value

    def _advance(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        """Move the clock forward; deliver due+reachable in-flight messages (sequential LWW)."""
        dt = int(action.args[0])
        new_clock = state.clock + dt
        edits: list[DistEdit] = [ClockSet(new_clock)]
        # Deliver sequentially so each sees earlier deliveries' effects (LWW correctness).
        working = state.copy()
        delivered = 0
        for msg_id in sorted(state.inflight):
            msg = state.inflight[msg_id]
            deliverable = (
                msg.deliver_after <= new_clock
                and working.connected(msg.src, msg.dst)
                and working.is_up(msg.dst)
                and self._causal_ready(working, msg)
            )
            if not deliverable:
                continue
            step_edits: list[DistEdit] = [MsgDeliver(msg_id)]
            cur = working.replicas.get((msg.object_id, msg.dst))
            incoming = (msg.version, msg.value)
            if cur is None or incoming > (cur.version, cur.value):  # last-writer-wins by (ver, val)
                step_edits.append(ReplicaWrite(msg.object_id, msg.dst, msg.version, msg.value))
            edits.extend(step_edits)
            working = apply(working, step_edits)  # so the next message's LWW sees this effect
            delivered += 1
        edits.append(SetResult("advanced", str(delivered)))
        return edits, "advanced", str(delivered)

    def _causal_ready(self, state: DistributedState, msg: Message) -> bool:
        """Whether ``msg``'s causal dependencies are satisfied at its destination (``causal`` only).

        Under ``causal`` consistency a message is held until the destination has applied at least
        the versions the writing node had observed (``msg.deps``) -- so a replica never adopts an
        effect before its cause. Checked against ``state`` (the in-progress ``advance`` working
        copy), so a dependency delivered earlier *in the same advance* unblocks a later
        causally-dependent message (causally-ordered messages can both land in one step). Empty
        ``deps`` (eventual/linearizable, or a write that observed nothing) is always ready.
        """
        for obj, ver in msg.deps:
            cur = state.replicas.get((obj, msg.dst))
            if cur is None or cur.version < ver:
                return False
        return True

    def _partition(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        """Split the network into the named groups; unmentioned nodes form one isolated group."""
        mentioned = {n for group in action.groups for n in group}
        groups = list(action.groups)
        unmentioned = tuple(n for n in self.config.nodes if n not in mentioned)
        if unmentioned:
            groups.append(unmentioned)
        return [PartitionSet(tuple(groups)), SetResult("ok", "")], "ok", ""

    def _heal(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        return [PartitionSet((self.config.nodes,)), SetResult("ok", "")], "ok", ""

    def _node_status(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        node = action.args[0]
        edit: DistEdit = NodeDown(node) if action.name == "crash" else NodeUp(node)
        return [edit, SetResult("ok", "")], "ok", ""

    def _drop(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        """``drop src dst``: lose every in-flight message from ``src`` to ``dst`` (DS0 incr 11).

        The unreliable-network fault, the distributed analogue of SPEC-5's packet loss and the
        ``BUGGIFY`` of message drop (SPEC-7 §2.1, §3.2). Unlike ``partition`` (which *holds* a
        message until the link heals), ``drop`` **destroys** it: the destination replica permanently
        misses that write, so the cluster does **not** reconverge on ``heal``+``advance`` — only a
        *newer* write to the same key can overwrite the stale replica (ED18). The drop is
        unconditional (it does not need the link to be currently connected — a message can be lost
        whether or not it would have been delivered) and reports the number lost, mirroring
        ``advance``'s delivered count. Dropping a channel with no in-flight messages is a no-op.
        """
        src, dst = action.args[0], action.args[1]
        dropped = sorted(
            mid for mid, m in state.inflight.items() if m.src == src and m.dst == dst
        )
        edits: list[DistEdit] = [MsgDrop(mid) for mid in dropped]
        value = str(len(dropped))
        edits.append(SetResult("dropped", value))
        return edits, "dropped", value

    def _anti_entropy(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        """``anti_entropy node``: read-repair ``node`` to its latest reachable replica (DS0 inc 12).

        The **anti-entropy / read-repair** mechanism real eventually-consistent stores (Dynamo,
        Cassandra) use to converge *despite* lost messages — the SPEC-7 §4 ``ReplicaConverge`` op,
        the first **protocol** op (§3.2). For every object ``node`` holds, it adopts the winning
        ``(version, value)`` among its **reachable** replicas (itself + co-partitioned, up peers) by
        the same last-writer-wins rule ``advance`` uses, emitting a ``ReplicaWrite`` per object that
        moves. Unlike ``advance`` (which needs an in-flight message), anti-entropy reads the peers'
        *current* replicas directly, so it repairs a **dropped** write ``advance`` never can
        (ED18 → ED19) — but it is **bounded by reachability**: under partition it reconciles only
        within ``node``'s group, so full convergence still needs ``heal``. A crashed node is
        ``unavailable``; reachability never includes a partitioned-away or down peer, so this
        composes with the fault medium unchanged. Reports the number of replicas repaired.
        """
        node = action.args[0]
        if not state.is_up(node):
            return [SetResult("unavailable", "")], "unavailable", ""
        edits: list[DistEdit] = []
        repaired = 0
        for obj in self.config.objects:
            local = state.replicas.get((obj, node))
            if local is None:
                continue  # node does not replicate this object
            best_version, best_value = local.version, local.value
            for peer in self.config.replicas_of(obj):
                if peer == node or not (state.connected(node, peer) and state.is_up(peer)):
                    continue
                r = state.replicas.get((obj, peer))
                if r is not None and (r.version, r.value) > (best_version, best_value):
                    best_version, best_value = r.version, r.value
            if (best_version, best_value) != (local.version, local.value):
                edits.append(ReplicaWrite(obj, node, best_version, best_value))
                repaired += 1
        # CRDT G-counter / PN-counter join (DS0 incr 28/29): pull each (key, owner) sub-count to the
        # reachable max (a no-op when no CRDT counter is used; the pre-incr-28 form is unchanged).
        reachable = [node, *(p for p in self.config.nodes
                             if p != node and state.connected(node, p) and state.is_up(p))]
        gc = _gcounter_merge_edits(state, [node], reachable)
        edits.extend(gc)
        repaired += len(gc)
        # CRDT OR-Set / MV-register joins (incr 30/31): pull each dotted set to the reachable union.
        os_edits = _orset_merge_edits(state, [node], reachable)
        edits.extend(os_edits)
        repaired += len(os_edits)
        mv_edits = _mvreg_merge_edits(state, [node], reachable)
        edits.extend(mv_edits)
        repaired += len(mv_edits)
        lww_edits = _lwwreg_merge_edits(state, [node], reachable)
        edits.extend(lww_edits)
        repaired += len(lww_edits)
        om_edits = _ormap_merge_edits(state, [node], reachable)
        edits.extend(om_edits)
        repaired += len(om_edits)
        rga_edits = _rga_merge_edits(state, [node], reachable)
        edits.extend(rga_edits)
        repaired += len(rga_edits)
        value = str(repaired)
        edits.append(SetResult("repaired", value))
        return edits, "repaired", value
