"""Distributed divergence + consistency-faithfulness (SPEC-7 §9.1-9.2; DS3, increment 1).

``divergence(a, b)`` is the distributed analogue of v0's filesystem set-difference and the network's
graph difference: a normalized symmetric difference over the **live cluster facts** (per-replica
state, in-flight messages, the partition/crash/clock medium), ``0`` iff identical and ``∈ [0, 1]``.
It feeds the generic :func:`verisim.metrics.horizon.faithful_horizon`, so ``H_ε(ρ)`` is defined for
the distributed world as for every prior one. (The causal *log* is the unbounded audit trail,
not part of the predictable live state, so it is excluded from divergence — a model is judged on the
cluster it produces, not the history it narrates.)

``consistency_faithfulness(a, b)`` is the **headline-new metric** (SPEC-7 §9.1, the distributed
analogue of the network's reachability-faithfulness): the fraction of objects whose **consistency
view** — the set of `(version, value)` pairs present across that object's replicas, i.e. *is it
converged, and to what* — the predicted state gets right. This is the operationally meaningful
(does the model predict the cluster's *consistency state*, the thing an SRE/defender actually relies
on), and it is distinct from bit-exact replica match: a model can place the converged value on the
wrong node and still be consistency-faithful, or get every node's bytes right yet mispredict whether
the object is split under partition. Pure and dependency-free, like every prior metric core.
"""

from __future__ import annotations

from verisim.dist.state import DistributedState

Fact = tuple[object, ...]


def dist_facts(state: DistributedState) -> set[Fact]:
    """The set of distinguishable facts defining the **live** cluster state (the log excluded)."""
    facts: set[Fact] = set()
    for (obj, node), r in state.replicas.items():
        facts.add(("replica", obj, node, r.version, r.value))
    for m in state.inflight.values():
        facts.add(("msg", m.id, m.src, m.dst, m.object_id, m.version, m.value, m.deliver_after))
    for group in state.partitions:
        facts.add(("part", frozenset(group)))
    for node in state.down:
        facts.add(("down", node))
    facts.add(("\x00clock", state.clock))
    return facts


def divergence(a: DistributedState, b: DistributedState) -> float:
    """Normalized symmetric difference over the live cluster facts. ``0.0`` iff identical."""
    fa = dist_facts(a)
    fb = dist_facts(b)
    denom = len(fa) + len(fb)
    return len(fa ^ fb) / denom if denom else 0.0


def object_consistency_view(state: DistributedState, obj: str) -> frozenset[tuple[int, str]]:
    """The consistency view of ``obj``: the set of ``(version, value)`` across its replicas.

    A singleton set means the object is **converged** (all replicas agree); a larger set means it is
    **split** (e.g. a stale replica under partition) — the consistency abstraction the model is
    judged on, independent of which node holds which value.
    """
    return frozenset(
        (r.version, r.value) for (o, _node), r in state.replicas.items() if o == obj
    )


def _objects(state: DistributedState) -> set[str]:
    return {obj for (obj, _node) in state.replicas}


def consistency_faithfulness(a: DistributedState, b: DistributedState) -> float:
    """Fraction of objects whose consistency view agrees between ``a`` (truth) and ``b`` (pred).

    ``1.0`` iff the two states induce the same per-object converged/split structure and values.
    Over the union of objects (an object in one state but not the other counts as a mismatch).
    """
    objects = _objects(a) | _objects(b)
    if not objects:
        return 1.0
    agree = sum(
        object_consistency_view(a, obj) == object_consistency_view(b, obj) for obj in objects
    )
    return agree / len(objects)


__all__ = [
    "consistency_faithfulness",
    "dist_facts",
    "divergence",
    "object_consistency_view",
]
