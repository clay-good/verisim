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
    DistDelta,
    DistEdit,
    EventAppend,
    MsgDeliver,
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
                edits.append(MsgSend(msg_id, node, peer, key, new_version, value, state.clock + 1))
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
                MsgSend(msg_id, node, peer, key, new_version, value, state.clock + 1, deps)
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
