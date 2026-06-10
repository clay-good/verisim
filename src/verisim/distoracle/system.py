"""``SystemDistOracle``: Tier-B, the distributed *system oracle* (SPEC-7 §5.2).

This is the distributed analogue of the host ``SandboxOracle`` (SPEC-11): where Tier-A
(:class:`~verisim.distoracle.reference.ReferenceDistOracle`) is a *single-threaded analytic
discrete-event simulator* that computes the next cluster state in closed form, Tier-B **runs the
replicated-KV protocol as a real distributed system** -- a set of autonomous **node actors** that
each hold *only their own replicas and an inbox*, exchange real replication **messages**, and have
**no access to any global state**. The cluster state is *emergent*, reconstructed by polling the
actors, exactly as a real cluster's state is never stored in one place (W7).

Tier-B exists to attack SPEC-3 wall **W1** ("the oracle is a *model*, not reality") for the
distributed domain. If an independent, message-passing, actor-based execution reproduces Tier-A's
observable cluster bit-for-bit, then Tier-A's analytic shortcut is *faithful to a genuine
distributed execution*, not merely self-consistent -- and the Tier-A↔Tier-B gap, where it is
non-zero, is a first-class reportable result and a curriculum signal (SPEC-7 §5.2), never a silent
pass.

**Determinism via a seeded scheduler -- the DST thesis (SPEC-7 §2.1).** A real cluster's
replication-message delivery order is *nondeterministic*; that nondeterminism is exactly what makes
a cluster un-replayable and therefore unusable as a bit-exact oracle. Tier-B does what madsim /
turmoil / FoundationDB's deterministic simulator do: it keeps the *real* concurrent message-passing
structure but drives it with a **seeded scheduler** that picks a delivery order as a pure function
of ``(state, action)``. So Tier-B is replayable and deterministic -- and, crucially, the order it
picks is *not* Tier-A's fixed sorted-by-msg-id order but a seed-**shuffled** one. Agreement
therefore certifies the stronger property the analytic DES quietly assumes: the eventual-consistency
convergence is **delivery-order-independent** (LWW by ``(version, value)`` is a commutative join),
so Tier-A's order choice cannot bias the truth.

**Two isolation tiers, disclosed never assumed (the SPEC-11 ``process``/``namespaced`` split).**

  - ``simulated`` (the always-on default): the actors are plain objects single-stepped by the
    deterministic scheduler -- the madsim model, robust and fast, what the committed figures run on.
  - ``threaded`` (a probed, disclosed enhancement): each actor runs in a *real OS thread* blocking
    on a real :class:`queue.Queue` inbox, and the scheduler dispatches **one message at a time** and
    awaits its acknowledgement before the next -- genuine kernel concurrency primitives, made
    deadlock-free and deterministic by the strictly-sequential one-in-flight protocol. This is the
    strongest "reality" claim (real threads, real queues), the distributed echo of the host running
    a real ``/bin/sh`` over a real kernel.

The constructor raises :class:`SystemDistOracleUnavailable` only when a requested tier genuinely
cannot run (no thread support) -- a disclosed, first-class skip, never a silent pass.
"""

from __future__ import annotations

import hashlib
import queue
import threading
from dataclasses import dataclass

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import (
    ClockSet,
    DistDelta,
    DistEdit,
    EventAppend,
    MsgDeliver,
    MsgDrop,
    MsgSend,
    NodeDown,
    NodeUp,
    PartitionSet,
    ReplicaWrite,
    SetResult,
    apply,
)
from verisim.dist.state import DistributedState, Message
from verisim.dist.txn import txn_step
from verisim.distoracle.base import DistStepResult
from verisim.distoracle.reference import (
    add_replica_edits,
    append_edits,
    causal_deps,
    clock_skew_edits,
    dequeue_edits,
    elect_edits,
    enqueue_edits,
    gossip_edits,
    lease_edits,
    lread_edits,
    propose_edits,
    remove_replica_edits,
    step_down_edits,
    timing_fault_edits,
)

TIER_SIMULATED = "simulated"
TIER_THREADED = "threaded"

# How long the threaded tier waits for one actor to acknowledge one message before declaring the
# real-thread runtime unusable (a disclosed unavailability, never a silent hang).
THREAD_ACK_TIMEOUT_S = 5.0


class SystemDistOracleUnavailable(RuntimeError):
    """Raised when a requested Tier-B runtime cannot run (e.g. no OS-thread support).

    A first-class, *disclosed* skip (the SPEC-11 §2.5 discipline carried to the distributed world):
    callers catch it and record the skip in their figure rather than counting it as agreement.
    """


@dataclass(frozen=True)
class DistDeterminismReport:
    """What Tier-B seals, and how (the SPEC-11 determinism-report analogue, SPEC-7 §3.3).

    The seed is a pure function of ``(state, action)``, so a replay is bit-identical; the delivery
    order is *shuffled* (not Tier-A's sorted order), so agreement certifies order-independence.
    """

    tier: str
    delivery_order_seeded: bool  # the scheduler picks delivery order from a seed (replayable)
    delivery_order_shuffled: bool  # ...and it is NOT Tier-A's sorted order (the real DST test)
    no_global_state: bool  # actors hold only their own replicas; the cluster is emergent
    real_os_threads: bool  # the threaded tier uses real kernel threads + queues
    notes: str = ""


# --- the autonomous node actor: only its own replicas + an inbox, no global view -----------------

class _NodeActor:
    """One cluster node's autonomous KV state machine: its replicas of each object, nothing else.

    It reacts to exactly three messages, each touching *only its own* state -- never a global view:

      - ``("write", obj, version, value)`` -- a local client write at the coordinator: adopt
        unconditionally (the coordinator authored this version).
      - ``("deliver", obj, version, value)`` -- a replication message from a peer: adopt iff it
        **wins last-writer-wins by ``(version, value)``** over this node's current replica. This is
        the one order-sensitive decision in the protocol, and the reason a *shuffled* delivery order
        is a real test: LWW is a commutative join, so a correct actor converges regardless of order.
      - ``("read", obj)`` -- return this node's local replica value (which may be stale under
        partition -- the node cannot see writes it has not been sent).
    """

    def __init__(self, node_id: str, replicas: dict[str, tuple[int, str]]) -> None:
        self.node_id = node_id
        self._replicas = dict(replicas)  # object_id -> (version, value)

    def handle(self, msg: tuple[str, str, int, str]) -> tuple[int, str] | None:
        kind = msg[0]
        if kind == "read":
            return self._replicas.get(msg[1])
        obj, version, value = msg[1], msg[2], msg[3]
        if kind == "write":
            self._replicas[obj] = (version, value)
            return None
        # deliver: last-writer-wins by (version, value) over the *local* current replica
        cur = self._replicas.get(obj)
        if cur is None or (version, value) > cur:
            self._replicas[obj] = (version, value)
        return None

    def snapshot(self) -> dict[str, tuple[int, str]]:
        return dict(self._replicas)


class _BrokenArrivalActor(_NodeActor):
    """A deliberately-broken actor that adopts a delivered write **by arrival order**, ignoring the
    LWW version comparison (last-writer-by-arrival instead of last-writer-by-version).

    This is the SY3-style *teeth-bearing negative control* (SPEC-11 §4): its convergence is
    order-**dependent**, so under Tier-B's shuffled delivery order it disagrees with Tier-A's sorted
    order -- proving the differential harness can actually *detect* a faithfulness break, not just
    rubber-stamp an identical reimplementation.
    """

    def handle(self, msg: tuple[str, str, int, str]) -> tuple[int, str] | None:
        if msg[0] == "deliver":
            self._replicas[msg[1]] = (msg[2], msg[3])  # adopt by arrival, no version check
            return None
        return super().handle(msg)


class SystemDistOracle:
    """Tier-B: the replicated-KV protocol run as autonomous actors under a seeded scheduler.

    A drop-in :class:`~verisim.distoracle.base.DistOracle`: every consumer that accepts a
    ``DistOracle`` (the tiered oracle's ``bit_exact`` slot, the DS5 loop, the differential harness)
    accepts this unchanged. The only new surface is the actor runtime itself.
    """

    version = "system-dist-1"

    def __init__(
        self,
        config: DistConfig = DEFAULT_DIST_CONFIG,
        *,
        tier: str = TIER_SIMULATED,
        broken_arrival: bool = False,
    ) -> None:
        if tier not in (TIER_SIMULATED, TIER_THREADED):
            raise ValueError(
                f"unknown tier {tier!r}; choose {TIER_SIMULATED!r} or {TIER_THREADED!r}"
            )
        if tier == TIER_THREADED and not _threads_available():
            raise SystemDistOracleUnavailable(
                "the threaded Tier-B runtime requires OS-thread support (disclosed skip, §5.2)"
            )
        self.config = config
        self.tier = tier
        self._broken = broken_arrival

    # -- the DistOracle protocol ------------------------------------------------------------------

    def step(self, state: DistributedState, action: DistAction) -> DistStepResult:
        delta, status, value = self._run(state, action)
        return DistStepResult(apply(state, delta), delta, status, value)

    def determinism_report(self) -> DistDeterminismReport:
        return DistDeterminismReport(
            tier=self.tier,
            delivery_order_seeded=True,
            delivery_order_shuffled=True,
            no_global_state=True,
            real_os_threads=self.tier == TIER_THREADED,
            notes=(
                f"SystemDistOracle[tier={self.tier}]: autonomous node actors, emergent cluster "
                "state, delivery order seeded from (state, action) and shuffled vs Tier-A's sorted "
                "order so agreement certifies order-independence (LWW is a commutative join)."
            ),
        )

    # -- the runtime: build actors, drive the action, emit the edit list --------------------------

    def _run(self, state: DistributedState, action: DistAction) -> tuple[DistDelta, str, str]:
        """Drive one action through the actor runtime; return its edit list + client result.

        Faults and time are *medium* changes (no actor work): handled directly. Client ops and
        replication deliveries are routed to actors -- in ``simulated`` by direct ``handle``, in
        ``threaded`` by a real one-message-at-a-time inbox round-trip (see :meth:`_dispatch`).
        """
        name = action.name
        if name in ("begin", "tget", "tput", "commit", "abort"):
            # The transaction family (DS0 increment 2) is coordinator-local, deterministic
            # bookkeeping — not a distributed-concurrency concern — so Tier-B computes it with the
            # same shared OCC logic Tier-A uses (there is no delivery-order question to validate
            # independently). The genuinely-distributed part — a committed write's async replication
            # — flows through the in-flight medium and is delivered later by this oracle's own
            # autonomous-actor ``_advance``, exactly where Tier-B's independence does its work.
            return txn_step(state, action, self.config)
        ev = self._event(state, action)
        if name in ("put", "cas"):
            return self._write(state, action, ev)
        if name == "get":
            return self._get(state, action, ev)
        if name == "advance":
            return self._advance(state, action)
        if name == "partition":
            return self._partition(state, action)
        if name == "heal":
            return [PartitionSet((self.config.nodes,)), SetResult("ok", "")], "ok", ""
        if name in ("crash", "restart"):
            edit: DistEdit = NodeDown(action.args[0]) if name == "crash" else NodeUp(action.args[0])
            return [edit, SetResult("ok", "")], "ok", ""
        if name == "drop":
            return self._drop(state, action)
        if name in ("delay", "reorder"):
            # Message-timing faults (DS0 incr 13) are *medium* changes — no actor work — so Tier-B
            # computes the byte-identical reschedule via the shared helper and reproduces the
            # afterward (delayed-but-recoverable convergence / reordered transit) on its own
            # autonomous-actor delivery, exactly as it does for ``drop`` (ED20, §5.2).
            return timing_fault_edits(state, action)
        if name == "clock_skew":
            return clock_skew_edits(action)
        if name == "anti_entropy":
            return self._anti_entropy(state, action)
        if name == "gossip":
            # Pairwise anti-entropy (DS0 incr 15) is a coordinator-level reconciliation (it reads
            # both replicas directly, no in-flight message), so — like ``anti_entropy`` — Tier-B
            # computes the byte-identical reconciliation via the shared helper.
            return gossip_edits(state, action, self.config)
        if name == "elect":
            # Leader election (DS0 incr 16) is a coordinator-level consensus decision — the quorum
            # is read from the medium (partition/down), not an actor's local view (exactly like
            # ``_write``'s majority reachability) — so Tier-A and Tier-B compute byte-identical
            # leader/term deltas via the shared helper. The genuinely-distributed property Tier-B
            # then reproduces is that the elected leader's writes replicate over the autonomous
            # actors and the deposed leader is fenced (ED23).
            return elect_edits(state, action, self.config)
        if name == "propose":
            # A leader-fenced majority write (DS0 incr 16): like the quorum ``put``, the majority
            # set is the coordinator's decision from the medium, so the write edits are computed
            # byte-identically via the shared helper; the async catch-up to the minority lands
            # later by this oracle's own autonomous-actor ``_advance`` (where Tier-B's independence
            # does its work).
            return propose_edits(state, action, self.config)
        if name == "step_down":
            # Voluntary relinquishment (DS0 incr 17) reads only the cluster's leadership metadata,
            # never the medium or an actor's replicas, so Tier-A and Tier-B clear the leader
            # byte-identically via the shared helper — the partition-independence ED24 Panel B turns
            # on (a minority-stranded leader steps down where its ``propose`` is ``no_quorum``).
            return step_down_edits(state, action, self.config)
        if name in ("lease", "lread"):
            # The leader lease (DS0 incr 18) is coordinator-level cluster metadata (like leader/
            # term), read/written from the global lease deadline, not an actor's local view — so
            # Tier-A and Tier-B compute byte-identical lease/read deltas via the shared helpers.
            return (lease_edits if name == "lease" else lread_edits)(state, action, self.config)
        if name == "append":
            # The replicated-log append (DS0 incr 19): like ``propose``, the majority set is the
            # coordinator's decision from the medium and the log/commit/KV deltas are computed
            # byte-identically via the shared helper, so Tier-A ≡ Tier-B on the observable channel.
            return append_edits(state, action, self.config)
        if name in ("add_replica", "remove_replica"):
            # Membership change (DS0 incr 20) reconfigures the voting set — coordinator-level
            # cluster metadata (like leader/term), read/written from the global membership, not an
            # actor's local view — so Tier-A and Tier-B compute identical deltas via the helpers.
            helper = add_replica_edits if name == "add_replica" else remove_replica_edits
            return helper(state, action, self.config)
        if name in ("enqueue", "dequeue"):
            # The FIFO queue (DS0 incr 21): the reachable set + availability are read from the
            # medium (a coordinator-level decision, like the KV write), so Tier-A and Tier-B compute
            # byte-identical queue deltas via the shared helpers — the duplicate-delivery-under-
            # partition behavior is reproduced on the autonomous actors too.
            helper = enqueue_edits if name == "enqueue" else dequeue_edits
            return helper(state, action, self.config)
        raise ValueError(f"unhandled action {name!r}")  # pragma: no cover - grammar is closed

    def _event(self, state: DistributedState, action: DistAction) -> EventAppend:
        """The causal-log event -- reconstructed by the same program-order rule as Tier-A.

        The log + monotonic id counters are *bookkeeping* of our representation, not observable
        cluster behavior; the differential compares the observable-cluster channel (replicas +
        in-flight + medium), so this reconstruction is identical by construction and never the
        interesting signal (the host excludes ``last`` for the same orthogonality reason)."""
        node = action.args[0] if action.name in ("put", "get", "cas") else ""
        prior = tuple(e.id for e in state.log if e.node == node)
        return EventAppend(state.next_event_id, node, action.raw, state.clock, prior)

    def _actor_for(self, state: DistributedState, node: str) -> _NodeActor:
        """An actor holding *only* ``node``'s replicas (the no-global-state guarantee)."""
        local = {
            obj: (r.version, r.value)
            for (obj, n), r in state.replicas.items()
            if n == node
        }
        cls = _BrokenArrivalActor if self._broken else _NodeActor
        return cls(node, local)

    def _write(
        self, state: DistributedState, action: DistAction, ev: EventAppend
    ) -> tuple[DistDelta, str, str]:
        node, key = action.args[0], action.args[1]
        replica = state.replicas.get((key, node))
        if not state.is_up(node) or replica is None:
            status = "unavailable" if not state.is_up(node) else "no_replica"
            return [ev, SetResult(status, "")], status, ""

        model = self.config.consistency_model
        peers = self.config.replicas_of(key)
        reachable = [p for p in peers if state.connected(node, p) and state.is_up(p)]
        # CP write-rejection (same gates as Tier-A): linearizable needs all replicas, quorum a
        # strict majority. The reachability is read from the medium (partition/down) — the
        # coordinator's decision, not an actor's local view, exactly as Tier-A computes it.
        if model == "linearizable" and len(reachable) < len(peers):
            return [ev, SetResult("unavailable", "")], "unavailable", ""
        if model == "quorum" and len(reachable) < len(peers) // 2 + 1:
            return [ev, SetResult("unavailable", "")], "unavailable", ""

        if action.name == "cas":
            old, new = action.args[2], action.args[3]
            actor = self._actor_for(state, node)
            cur = self._dispatch(actor, ("read", key, 0, ""))
            cur_val = cur[1] if cur is not None else self.config.default_value
            if cur_val != old:
                return [ev, SetResult("conflict", cur_val)], "conflict", cur_val
            value = new
        else:
            value = action.args[2]

        new_version = replica.version + 1
        coordinator = self._actor_for(state, node)
        self._dispatch(coordinator, ("write", key, new_version, value))  # local write at the actor

        if model == "linearizable":
            edits: list[DistEdit] = [ev]
            edits.extend(ReplicaWrite(key, peer, new_version, value) for peer in peers)
            edits.append(SetResult("ok", value))
            return edits, "ok", value

        if model == "quorum":
            # synchronous to the reachable majority; async catch-up to the unreachable minority.
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

        edits = [ev, ReplicaWrite(key, node, new_version, value)]
        # The causal context the replication carries — the *shared* helper, so Tier-A and Tier-B
        # attach byte-identical deps; empty under eventual (which does not order delivery).
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
        self, state: DistributedState, action: DistAction, ev: EventAppend
    ) -> tuple[DistDelta, str, str]:
        node, key = action.args[0], action.args[1]
        replica = state.replicas.get((key, node))
        if not state.is_up(node) or replica is None:
            status = "unavailable" if not state.is_up(node) else "no_replica"
            return [ev, SetResult(status, "")], status, ""
        actor = self._actor_for(state, node)
        cur = self._dispatch(actor, ("read", key, 0, ""))
        val = cur[1] if cur is not None else self.config.default_value
        return [ev, SetResult("ok", val)], "ok", val

    def _advance(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        """Deliver due+reachable messages to peer actors in a **seed-shuffled** order.

        Each delivery is handled by the destination's autonomous actor against *its own* current
        replica, so a later delivery sees an earlier one's effect -- exactly as a real node does.
        Because the actors apply LWW by ``(version, value)`` (a commutative join), the converged
        replicas are independent of the shuffle, which is the property agreement certifies.

        **Causal delivery (the §3.4 ``causal`` model).** A message carries ``deps`` (a
        version-vector slice); an actor must not adopt it before those dependencies are applied. The
        shuffled order may try a message before its cause, so delivery runs to a **fixed point**:
        repeatedly scan the not-yet-delivered messages, delivering any whose deps are now satisfied
        at its destination, until a full pass delivers nothing. A message whose deps never arrive
        stays in flight (held, not lost). The fixed point delivers exactly the causally-ready
        closure -- *independent of the shuffle* -- so it reproduces Tier-A's sorted-order result
        (msg ids are topologically ordered: a causally-later write has a higher id). Under
        ``eventual`` no message has deps, so the first pass delivers all and the loop is one pass.
        """
        dt = int(action.args[0])
        new_clock = state.clock + dt
        edits: list[DistEdit] = [ClockSet(new_clock)]

        deliverable_ids = [
            mid
            for mid, msg in state.inflight.items()
            if msg.deliver_after <= new_clock
            and state.connected(msg.src, msg.dst)
            and state.is_up(msg.dst)
        ]
        order = self._delivery_order(state, action, deliverable_ids)
        causal = self.config.consistency_model == "causal"

        # One actor per destination node, carrying its current replicas across the batch so each
        # delivery's LWW (and deps check) sees the running result (the actor is the only mutator).
        actors: dict[str, _NodeActor] = {}
        delivered_ids: set[int] = set()
        delivered = 0
        progress = True
        while progress:
            progress = False
            for mid in order:
                if mid in delivered_ids:
                    continue
                msg = state.inflight[mid]
                actor = actors.setdefault(msg.dst, self._actor_for(state, msg.dst))
                if causal and not _deps_satisfied(actor, msg):
                    continue  # held: a causal dependency is not yet applied at the destination
                before = actor.snapshot().get(msg.object_id)
                self._dispatch(actor, ("deliver", msg.object_id, msg.version, msg.value))
                after = actor.snapshot().get(msg.object_id)
                edits.append(MsgDeliver(mid))
                if after != before and after is not None:
                    edits.append(ReplicaWrite(msg.object_id, msg.dst, after[0], after[1]))
                delivered_ids.add(mid)
                delivered += 1
                progress = True
        edits.append(SetResult("advanced", str(delivered)))
        return edits, "advanced", str(delivered)

    def _delivery_order(
        self, state: DistributedState, action: DistAction, ids: list[int]
    ) -> list[int]:
        """A seed-shuffled delivery order over ``ids`` -- the DST scheduler (SPEC-7 §2.1).

        The seed is a pure function of ``(state, action)`` (so a replay is bit-identical), and the
        order is a deterministic permutation that is *not* the sorted-by-id order Tier-A uses, so
        agreement is a real test of order-independence rather than a tautology.
        """
        int_ids = sorted(ids)
        if len(int_ids) <= 1:
            return int_ids
        seed = _seed(state, action)
        # A deterministic Fisher-Yates over a seeded LCG -- dependency-free and replayable.
        rng = _LCG(seed)
        out = list(int_ids)
        for i in range(len(out) - 1, 0, -1):
            j = rng.below(i + 1)
            out[i], out[j] = out[j], out[i]
        # Guard the degenerate case where the shuffle happened to reproduce the sorted order: a
        # single rotation still delivers a non-sorted order whenever there is real choice (>=2),
        # keeping the order-independence test honest.
        if out == int_ids:
            out = out[1:] + out[:1]
        return out

    def _partition(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        mentioned = {n for group in action.groups for n in group}
        groups = list(action.groups)
        unmentioned = tuple(n for n in self.config.nodes if n not in mentioned)
        if unmentioned:
            groups.append(unmentioned)
        return [PartitionSet(tuple(groups)), SetResult("ok", "")], "ok", ""

    def _drop(
        self, state: DistributedState, action: DistAction
    ) -> tuple[DistDelta, str, str]:
        """``drop src dst``: lose every in-flight message from ``src`` to ``dst`` (DS0 incr 11).

        Message loss is a *medium* change (the message simply never arrives), so — exactly like
        ``partition``/``crash`` — there is no actor work to do independently: the lost messages are
        removed from the in-flight set. Tier-A and Tier-B therefore compute byte-identical
        drop deltas (the in-flight medium is global, not an actor's local view), and the genuinely
        distributed signal Tier-B independently reproduces is what happens *afterward* — that the
        dropped write never reconverges on ``heal``+``advance`` (its autonomous-actor delivery has
        nothing to deliver), only an overwriting write does (ED18, the broken-convergence anomaly).
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
        """``anti_entropy node``: read-repair ``node`` to the latest reachable replica (DS0 inc 12).

        The read-repair the node performs is a *pull from its reachable peers* — a coordinator-level
        reconciliation whose reachable set is read from the medium (partition/down), exactly as
        ``_write``'s quorum/linearizable reachability is the coordinator's decision rather than an
        actor's local view. The winner per object is last-writer-wins by ``(version, value)``, the
        same commutative join the delivery actors apply, so Tier-A and Tier-B compute byte-identical
        repair deltas — and the genuinely-distributed property Tier-B then reproduces is that the
        repaired cluster converges only over what was reachable (ED19, anti-entropy bounded by the
        partition), independent of the delivery shuffle.
        """
        node = action.args[0]
        if not state.is_up(node):
            return [SetResult("unavailable", "")], "unavailable", ""
        edits: list[DistEdit] = []
        repaired = 0
        for obj in self.config.objects:
            local = state.replicas.get((obj, node))
            if local is None:
                continue
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

    # -- dispatch: the simulated/threaded split ---------------------------------------------------

    def _dispatch(
        self, actor: _NodeActor, msg: tuple[str, str, int, str]
    ) -> tuple[int, str] | None:
        """Hand one message to one actor and return its result.

        ``simulated`` calls ``handle`` directly (the madsim single-thread executor); ``threaded``
        sends the message to a real per-call OS thread blocking on a :class:`queue.Queue` and waits
        for its acknowledgement -- one in flight at a time, so it is deterministic and cannot
        deadlock.
        """
        if self.tier == TIER_SIMULATED:
            return actor.handle(msg)
        return self._dispatch_threaded(actor, msg)

    def _dispatch_threaded(
        self, actor: _NodeActor, msg: tuple[str, str, int, str]
    ) -> tuple[int, str] | None:
        inbox: queue.Queue[tuple[str, str, int, str]] = queue.Queue()
        outbox: queue.Queue[tuple[int, str] | None] = queue.Queue()

        def worker() -> None:
            received = inbox.get()
            outbox.put(actor.handle(received))

        thread = threading.Thread(target=worker, name=f"tierb-{actor.node_id}", daemon=True)
        thread.start()
        inbox.put(msg)
        try:
            result = outbox.get(timeout=THREAD_ACK_TIMEOUT_S)
        except queue.Empty as exc:  # pragma: no cover - only on a pathologically stalled host
            raise SystemDistOracleUnavailable(
                "the threaded Tier-B actor did not acknowledge within the timeout (disclosed)"
            ) from exc
        thread.join(timeout=THREAD_ACK_TIMEOUT_S)
        return result


def _deps_satisfied(actor: _NodeActor, msg: Message) -> bool:
    """Whether ``msg``'s causal dependencies are applied at its destination actor (``causal`` only).

    Checked against the actor's *own* current replicas (the no-global-state guarantee) -- the Tier-B
    analogue of Tier-A's ``_causal_ready``, but read from the autonomous actor rather than the
    global ``DistributedState``. A dep ``(obj, ver)`` is met iff the actor holds ``obj`` at
    ``version >= ver``.
    """
    snap = actor.snapshot()
    for obj, ver in msg.deps:
        cur = snap.get(obj)
        if cur is None or cur[0] < ver:
            return False
    return True


# --- seeding + a tiny dependency-free RNG (no global random state) -------------------------------

def _seed(state: DistributedState, action: DistAction) -> int:
    """A deterministic 64-bit seed from ``(state, action)`` -- the replay key (no clock/RNG)."""
    from verisim.dist.serialize import to_json

    blob = to_json(state) + "\x00" + action.raw
    return int.from_bytes(hashlib.sha256(blob.encode("utf-8")).digest()[:8], "big")


class _LCG:
    """A minimal seeded linear-congruential generator (no ``random`` global state, replayable)."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def below(self, n: int) -> int:
        # Numerical Recipes LCG constants; high bits are well-distributed, so take them below.
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self._state >> 33) % n


def _threads_available() -> bool:
    """Probe whether the host can start an OS thread (disclosed, never assumed)."""
    try:
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join(timeout=THREAD_ACK_TIMEOUT_S)
        return True
    except RuntimeError:  # pragma: no cover - only on a thread-less build
        return False
