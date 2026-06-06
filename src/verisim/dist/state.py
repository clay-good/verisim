"""The distributed world's state (SPEC-7 §3.1, DS0 increment 1).

The distributed state is **not** a tree (SPEC-2), graph (SPEC-5), or single bundle (SPEC-6): it is a
set of **replicas** of logical objects spread across nodes, an append-only **causal event log**, the
**in-flight messages** that have been sent but not delivered, plus the fault/time medium (which
nodes are **down**, how the network is **partitioned**, and the simulation **clock**). This is the
worldify temporal-causal-fact model (SPEC-7 §2.5) instantiated for a cluster, and W7 made
structural: **there is no `global` state field** -- a consistent global snapshot is a *derived,
coordinated* read (an oracle call), never stored, so under partition different replicas legitimately
disagree.

DS0 increment 1 is the **replicated KV under partition** core: per-(object, node) MVCC replicas, an
asynchronous replication message layer, and partition/crash/clock. Consensus (Raft-subset),
transactions, and the embedded SPEC-6 host inside each node are later increments
(``docs/distributed-semantics.md``). Canonicalization (sorted maps, normalized ids) is mandatory so
the divergence metric (§9) measures protocol competence, not identifier churn. No runtime deps, no
GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verisim.dist.config import DistConfig


@dataclass(frozen=True)
class ReplicaState:
    """One node's copy of a logical object: an MVCC ``(version, value)`` pair.

    ``version`` is a monotone per-object logical timestamp (a Lamport-style counter the coordinator
    bumps on each write); a higher version wins on convergence (last-writer-wins by version, then by
    a deterministic tiebreak). ``version == 0`` is the boot replica with the config default value.
    """

    object_id: str
    node_id: str
    version: int
    value: str


@dataclass(frozen=True)
class Event:
    """One causal-log event: a client/protocol op that happened at ``node`` at logical ``clock``.

    ``happens_before`` is the set of prior event ids this event causally depends on (program order
    on the same node for now; cross-node causal edges arrive with the message-carried vector context
    in a later increment, SPEC-7 §3.1). The log is the spine the consistency metrics read.
    """

    id: int
    node: str
    op: str  # the action raw string that produced this event
    clock: int
    happens_before: tuple[int, ...] = ()


@dataclass(frozen=True)
class Message:
    """An in-flight replication message: a write the coordinator is propagating to a peer replica.

    Delivered to ``dst`` by ``advance`` once ``deliver_after <= clock`` and ``src``/``dst``
    are in the same partition group and ``dst`` is up. Until then it waits -- the source of stale
    reads under partition (SPEC-7 §3.1).
    """

    id: int
    src: str
    dst: str
    object_id: str
    version: int
    value: str
    deliver_after: int


def _all_connected(nodes: tuple[str, ...]) -> tuple[frozenset[str], ...]:
    """The healed network: a single partition group containing every node."""
    return (frozenset(nodes),)


@dataclass
class DistributedState:
    """The cluster: replicas + causal log + in-flight messages + the fault/time medium.

    ``replicas`` is keyed by ``(object_id, node_id)``. ``partitions`` is a tuple of disjoint node
    groups that together cover every node; two nodes can exchange messages iff they share a group
    (one all-nodes group = healed). ``down`` is the crashed-node set. Mutation is by convention done
    through the oracle, which builds a fresh state via ``apply``.
    """

    replicas: dict[tuple[str, str], ReplicaState]
    partitions: tuple[frozenset[str], ...]
    log: tuple[Event, ...] = ()
    inflight: dict[int, Message] = field(default_factory=dict)
    down: frozenset[str] = frozenset()
    clock: int = 0
    next_event_id: int = 0
    next_msg_id: int = 0
    last_result: tuple[str, str] | None = None  # (status, value_token) of the last client op

    def __post_init__(self) -> None:
        # ``partitions`` is conceptually a *set* of disjoint groups, but stored as a tuple; keep it
        # in canonical (sorted) order so equality and the to_canonical/from_canonical round-trip
        # are exact regardless of the order the oracle built the groups in (the §16 verified-
        # contribution protocol re-executes through from_canonical and compares bit-for-bit).
        ordered = tuple(sorted(self.partitions, key=sorted))
        if ordered != self.partitions:
            self.partitions = ordered

    @staticmethod
    def initial(config: DistConfig) -> DistributedState:
        """The boot cluster: every replica at version 0 holding the config default, no faults."""
        replicas: dict[tuple[str, str], ReplicaState] = {}
        for obj in config.objects:
            for node in config.replicas_of(obj):
                replicas[(obj, node)] = ReplicaState(obj, node, 0, config.default_value)
        return DistributedState(
            replicas=replicas,
            partitions=_all_connected(config.nodes),
        )

    def copy(self) -> DistributedState:
        """A fresh-container copy; ``ReplicaState``/``Event``/``Message`` are immutable."""
        return DistributedState(
            replicas=dict(self.replicas),
            partitions=self.partitions,
            log=self.log,
            inflight=dict(self.inflight),
            down=self.down,
            clock=self.clock,
            next_event_id=self.next_event_id,
            next_msg_id=self.next_msg_id,
            last_result=self.last_result,
        )

    def connected(self, a: str, b: str) -> bool:
        """``True`` iff nodes ``a`` and ``b`` share a partition group (can exchange messages)."""
        if a == b:
            return True
        return any(a in group and b in group for group in self.partitions)

    def is_up(self, node: str) -> bool:
        """``True`` iff ``node`` is not crashed."""
        return node not in self.down
