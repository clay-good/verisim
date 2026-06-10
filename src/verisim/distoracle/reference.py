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

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import (
    ClockSet,
    ClockSkewSet,
    DistDelta,
    DistEdit,
    EventAppend,
    LeaseSet,
    MsgDeliver,
    MsgDrop,
    MsgReschedule,
    MsgSend,
    NodeDown,
    NodeUp,
    PartitionSet,
    ProtocolStep,
    ReplicaWrite,
    SetResult,
    apply,
)
from verisim.dist.state import DistributedState, Message
from verisim.dist.txn import txn_step
from verisim.distoracle.base import DistStepResult


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
    value = str(synced)
    return [*edits, SetResult("gossiped", value)], "gossiped", value


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
    voters = [n for n in state.group_of(node) if state.is_up(n)]
    if len(voters) <= len(config.nodes) // 2:  # a strict majority of the whole cluster is required
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
    peers = config.replicas_of(key)
    reachable = [p for p in peers if state.connected(node, p) and state.is_up(p)]
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
        if name in ("put", "cas"):
            return self._write(state, action)
        if name == "get":
            return self._get(state, action)
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
        else:
            value = action.args[2]

        new_version = replica.version + 1
        if model == "linearizable":
            # synchronous: every replica gets the new version now; no in-flight, no staleness.
            edits: list[DistEdit] = [ev]
            edits.extend(ReplicaWrite(key, peer, new_version, value) for peer in peers)
            edits.append(SetResult("ok", value))
            return edits, "ok", value

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
            edits.append(SetResult("ok", value))
            return edits, "ok", value

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
        edits.append(SetResult("ok", value))
        return edits, "ok", value

    def _get(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        node, key = action.args[0], action.args[1]
        ev = self._event(state, node, action.raw)
        replica = state.replicas.get((key, node))
        if not state.is_up(node) or replica is None:
            status = "unavailable" if not state.is_up(node) else "no_replica"
            return [ev, SetResult(status, "")], status, ""
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
        value = str(repaired)
        edits.append(SetResult("repaired", value))
        return edits, "repaired", value
