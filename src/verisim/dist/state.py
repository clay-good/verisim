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
class TxnState:
    """An in-flight client transaction at its coordinator (SPEC-7 §3.2, DS0 increment 2).

    A multi-key transaction under **optimistic concurrency control** (OCC, first-committer-wins):
    the coordinator buffers the txn's reads and writes locally and validates at ``commit``.
    ``reads`` pins the ``(key, version)`` each key was first read at (the snapshot it validates);
    ``writes`` is the ordered buffer of ``(key, value)`` the txn applies atomically on commit (a later
    write to the same key supersedes an earlier one). A transaction is *active* exactly while it is
    present in ``DistributedState.txns``; ``commit``/``abort`` remove it. OCC is deterministic and
    deadlock-free (no lock table), which is why it is the discipline the deterministic core pins
    first — 2PL/lock-based isolation is a later refinement (``docs/distributed-semantics.md`` §9).
    """

    txn_id: str
    node: str
    reads: tuple[tuple[str, int], ...] = ()
    writes: tuple[tuple[str, str], ...] = ()
    # The version each written key held when the txn first wrote it — pinned at `tput`. Snapshot
    # isolation validates *these* (write-write conflicts, first-committer-wins) where serializable
    # validates `reads`; the difference is exactly what admits or forbids write skew (DS0 incr 3).
    write_versions: tuple[tuple[str, int], ...] = ()

    def read_version(self, key: str) -> int | None:
        """The version this txn pinned for ``key`` on first read, or ``None`` if never read."""
        for k, v in self.reads:
            if k == key:
                return v
        return None

    def write_version(self, key: str) -> int | None:
        """The version this txn pinned for ``key`` on first write, or ``None`` if never written."""
        for k, v in self.write_versions:
            if k == key:
                return v
        return None

    def buffered_write(self, key: str) -> str | None:
        """The value this txn has buffered for ``key`` (read-your-writes), or ``None``."""
        result: str | None = None
        for k, val in self.writes:
            if k == key:
                result = val  # last buffered write to the key wins
        return result


@dataclass(frozen=True)
class Message:
    """An in-flight replication message: a write the coordinator is propagating to a peer replica.

    Delivered to ``dst`` by ``advance`` once ``deliver_after <= clock`` and ``src``/``dst``
    are in the same partition group and ``dst`` is up. Until then it waits -- the source of stale
    reads under partition (SPEC-7 §3.1).

    ``deps`` is the **causal context** the message carries (DS0 increment 5, the ``causal``
    consistency model): a sorted tuple of ``(object_id, version)`` that the *source* node had already
    observed (applied to its own replicas) when it produced this write, for objects other than the
    one being written. Under ``causal`` consistency ``advance`` will not deliver the message until the
    destination has applied at least those versions -- so no replica ever sees an effect before its
    cause (cross-object causal ordering). It is **empty under ``eventual`` / ``linearizable``** (those
    models do not order delivery), so the field is omitted from the canonical form when empty and the
    pre-DS0-incr-5 goldens/hashes are unchanged.
    """

    id: int
    src: str
    dst: str
    object_id: str
    version: int
    value: str
    deliver_after: int
    deps: tuple[tuple[str, int], ...] = ()


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
    txns: dict[str, TxnState] = field(default_factory=dict)  # active transactions, keyed by txn_id
    # The 2PL lock table (DS0 increment 8), keyed by object: the sorted ``(txn_id, mode)`` holders,
    # ``mode ∈ {"S", "X"}``. Held from acquisition (``tget``/``tput``) to ``commit``/``abort``. Empty
    # (and omitted from the canonical form) under the ``occ`` default, so prior hashes are unchanged.
    locks: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    # Per-node clock offset (DS0 increment 14, the ``clock_skew`` fault): a node's local clock is the
    # global ``clock`` plus its offset, which shifts the ``deliver_after`` it stamps on the messages
    # it sends (a positive offset = a fast clock, defers its sends; negative = a slow clock, rushes
    # them). Empty (and omitted from the canonical form) by default, so prior hashes are unchanged.
    skew: dict[str, int] = field(default_factory=dict)
    # The consensus leader + term (DS0 increment 16, the Raft-subset ``elect``/``propose`` core). A
    # leader is elected (``elect``) by a partition group holding a strict majority of *live* cluster
    # nodes — so two majorities cannot coexist and there is never a second leader (no split-brain).
    # ``term`` is the monotone election epoch that *fences* a deposed leader: a new election bumps the
    # global term and sets the global ``leader``, so an old leader's ``propose`` is rejected (it is no
    # longer ``leader``) even after the partition heals — the Raft leader-completeness safety property
    # plain ``quorum`` writes lack. ``step_down`` (DS0 increment 17) clears ``leader`` (``→ None``) at
    # the *same* ``term`` — voluntary relinquishment, the graceful counterpart to deposition — so the
    # cluster is leaderless until a fresh ``elect``. Both at default (``term == 0``, ``leader is None``)
    # until the first ``elect``, and omitted from the canonical form there, so every pre-increment-16
    # hash is unchanged.
    term: int = 0
    leader: str | None = None

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
            txns=dict(self.txns),
            locks=dict(self.locks),
            skew=dict(self.skew),
            term=self.term,
            leader=self.leader,
        )

    def group_of(self, node: str) -> frozenset[str]:
        """The partition group ``node`` belongs to (a singleton if it is mentioned by no group)."""
        for group in self.partitions:
            if node in group:
                return group
        return frozenset({node})

    def sender_clock(self, node: str) -> int:
        """``node``'s local clock = the global clock plus its skew offset (DS0 increment 14).

        The one place a node's (possibly skewed) clock is observable: the ``deliver_after`` it stamps
        on the replication messages it sends. A node with no offset reads the global clock, so the
        un-skewed path is byte-identical to the pre-increment-14 form.
        """
        return self.clock + self.skew.get(node, 0)

    def connected(self, a: str, b: str) -> bool:
        """``True`` iff nodes ``a`` and ``b`` share a partition group (can exchange messages)."""
        if a == b:
            return True
        return any(a in group and b in group for group in self.partitions)

    def is_up(self, node: str) -> bool:
        """``True`` iff ``node`` is not crashed."""
        return node not in self.down
