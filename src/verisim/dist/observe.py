"""Partial observation -- the probe projection of the cluster (SPEC-7 §5.4, §3.1, DS3 incr 4).

Every prior world handed the oracle (and the model) the *whole* state. The distributed world is
the first where that is a fiction: **there is no consistent global snapshot** (W7, ``state.py``),
and no real observer ever has one. A client, an SRE, or a node sees only the part of the cluster
it can *reach* -- and crucially it cannot see the **in-flight replication medium** at all (the
messages are in the network, not in any node's memory). This module makes that epistemic limit a
first-class, deterministic object: :func:`observe` projects a
:class:`~verisim.dist.state.DistributedState` onto the :class:`Observation` an observer connected
to a set of ``vantage`` nodes can actually obtain.

Three properties, all true of real distributed systems and all exercised by the tests:

  - **The in-flight medium is invisible.** An :class:`Observation` carries no messages -- the
    replication payloads in transit are exactly the hidden state SPEC-7's H19/ED5 turns on (a
    message is bit-visible to the *oracle* but invisible to any *observer* until ``advance``
    delivers it and writes a replica). A model is never judged wrong about what no probe can see.
  - **Crash and partition are indistinguishable.** A node that is ``down`` and a node that is
    partitioned away from the vantage both project to the *same* ``unreachable`` fact -- the
    observer cannot localize the fault to a crash versus a network split (the failure-detector
    limit behind FLP). :func:`observe` of a crashed-node world and a partitioned-away world are
    byte-identical from a single vantage; it takes a *second* vantage to tell them apart.
  - **A probe is cheap and local.** ``observe(state, vantage)`` is the §5.4 "probe (cheap,
    localized) vs full (expensive)" oracle mode made concrete: a smaller fact-set, read from one
    place, that catches exactly the errors visible from there. It is the deterministic substrate
    the (deferred) RSSM belief must roll forward under partition (§6.2) -- the belief's job is to
    predict the *full* state from the *observable* one, and you cannot define that task without
    first defining what is observable.

Pure and dependency-free, like every prior core. No runtime deps, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.dist.state import DistributedState, ReplicaState


@dataclass(frozen=True)
class Observation:
    """What an observer at ``vantage`` can see of the cluster -- never the whole state.

    ``vantage`` is the set of nodes the observer can query directly (a client's connections, an
    SRE's reachable hosts). ``reachable`` is the set of nodes the observer can actually talk to: a
    node is reachable iff it is **up** and shares a partition group with some vantage node.
    ``replicas`` holds only the replicas on reachable nodes -- the rest of the cluster is dark.
    ``unreachable`` is every other node, *without a reason*: the observer cannot tell a crashed
    node from one partitioned away (the indistinguishability property). ``clock`` is observable via
    any reachable node; if the whole vantage is dark (every vantage node down, or isolated with
    nothing reachable) ``clock`` is ``None`` -- the observer has lost the cluster.

    There is deliberately **no in-flight field**: the replication medium is unobservable. The
    Observation is canonical by construction (frozensets), so two states with the same observable
    projection produce equal Observations -- which is what makes the crash/partition
    indistinguishability a testable byte-equality.
    """

    vantage: frozenset[str]
    reachable: frozenset[str]
    unreachable: frozenset[str]
    replicas: frozenset[tuple[str, str, int, str]]  # (object_id, node_id, version, value)
    clock: int | None

    def replica_value(self, object_id: str, node_id: str) -> str | None:
        """The observed value of ``(object_id, node_id)``, or ``None`` if that replica is dark."""
        for obj, node, _version, value in self.replicas:
            if obj == object_id and node == node_id:
                return value
        return None

    def observed_objects(self) -> frozenset[str]:
        """The objects the observer sees a replica of (one fully on dark nodes is gone)."""
        return frozenset(obj for obj, _node, _version, _value in self.replicas)


def _reachable_from(state: DistributedState, vantage: frozenset[str]) -> frozenset[str]:
    """The up-nodes the observer can talk to: up vantage nodes plus their co-partitioned up peers.

    A vantage node that is *down* cannot host the observer's query, but if any vantage node is up
    the observer reaches every up node in that node's partition group. Reachability is the closure
    over the (already transitive) partition groups, restricted to up nodes.
    """
    live_vantage = {n for n in vantage if state.is_up(n)}
    reachable: set[str] = set()
    for group in state.partitions:
        if any(v in group for v in live_vantage):
            reachable |= {n for n in group if state.is_up(n)}
    return frozenset(reachable)


def observe(state: DistributedState, vantage: frozenset[str] | tuple[str, ...]) -> Observation:
    """Project ``state`` onto the :class:`Observation` an observer at ``vantage`` can obtain.

    Deterministic and pure. The observer sees replicas only on reachable (up + co-partitioned)
    nodes, never the in-flight medium, and labels every other node ``unreachable`` with no
    crash/partition distinction. ``vantage`` may be one node ``("n0",)`` (a client) or several (a
    quorum probe).
    """
    vantage = frozenset(vantage)
    reachable = _reachable_from(state, vantage)
    all_nodes = {node for (_obj, node) in state.replicas} | set(vantage)
    unreachable = frozenset(all_nodes - reachable)
    replicas = frozenset(
        (r.object_id, r.node_id, r.version, r.value)
        for (_obj, node), r in state.replicas.items()
        if node in reachable
    )
    clock = state.clock if reachable else None
    return Observation(
        vantage=vantage,
        reachable=reachable,
        unreachable=unreachable,
        replicas=replicas,
        clock=clock,
    )


def observed_replica_states(obs: Observation) -> tuple[ReplicaState, ...]:
    """The observed replicas as :class:`ReplicaState` records, canonically sorted (a test aid)."""
    return tuple(
        ReplicaState(obj, node, version, value)
        for obj, node, version, value in sorted(obs.replicas)
    )


__all__ = [
    "Observation",
    "observe",
    "observed_replica_states",
]
