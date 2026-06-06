"""Observable divergence -- faithfulness as seen through a probe (SPEC-7 §5.4, §9.2; DS3 incr 4).

:func:`divergence` (``distmetrics.divergence``) compares two *full* cluster states -- the bit-exact
oracle's view. But the §5.4 oracle also has a **probe (cheap, localized)** mode, and no real
observer ever has the full state (W7). :func:`observable_divergence` is the probe-mode divergence:
it compares what an observer at ``vantage`` can *see* of truth versus prediction
(:func:`~verisim.dist.observe.observe`), so a model is judged only on the part of the cluster a
probe at that vantage exposes.

The point this metric makes precise is the partial-observation analogue of ED5/H19: because the
:class:`~verisim.dist.observe.Observation` carries **no in-flight medium**, a model that
mispredicts only messages-in-transit is *observably faithful* (zero observable divergence) even
while it is bit-divergent -- until ``advance`` delivers the message and the error surfaces in a
replica the probe can read. So the **observable-faithful horizon >= the bit-faithful horizon**, and
the gap is exactly the hidden replication medium. Pure and dependency-free.
"""

from __future__ import annotations

from verisim.dist.observe import Observation, observe
from verisim.dist.state import DistributedState

ObsFact = tuple[object, ...]


def observation_facts(obs: Observation) -> set[ObsFact]:
    """The distinguishable facts of an Observation: observed replicas + reachability + clock.

    Deliberately *excludes* in-flight messages (an Observation has none) and gives the
    crash/partition-indistinguishable ``unreachable`` set as a single ``("unreachable", node)`` fact
    per dark node -- no reason attached, so a crashed-node truth and a partitioned-away truth share
    the same observable facts.
    """
    facts: set[ObsFact] = set()
    for obj, node, version, value in obs.replicas:
        facts.add(("oreplica", obj, node, version, value))
    for node in obs.reachable:
        facts.add(("reachable", node))
    for node in obs.unreachable:
        facts.add(("unreachable", node))
    facts.add(("\x00oclock", obs.clock))
    return facts


def observable_divergence(
    a: DistributedState, b: DistributedState, vantage: frozenset[str] | tuple[str, ...]
) -> float:
    """Normalized symmetric difference over the facts a probe at ``vantage`` reads. ``0`` iff equal.

    The probe-mode (§5.4) analogue of
    :func:`verisim.distmetrics.divergence.divergence`: identical to it when ``vantage`` reaches the
    whole cluster, but smaller (and forgiving of the in-flight medium) under partition. Feeds the
    generic ``faithful_horizon`` to define an *observable*-faithful horizon.
    """
    fa = observation_facts(observe(a, vantage))
    fb = observation_facts(observe(b, vantage))
    denom = len(fa) + len(fb)
    return len(fa ^ fb) / denom if denom else 0.0


def observably_indistinguishable(
    a: DistributedState, b: DistributedState, vantage: frozenset[str] | tuple[str, ...]
) -> bool:
    """``True`` iff a probe at ``vantage`` cannot tell ``a`` from ``b`` (equal Observations).

    The operational form of the crash/partition indistinguishability: a crashed-node state and a
    partitioned-away state are ``observably_indistinguishable`` from a single vantage, yet a second
    vantage that reaches the node in question separates them.
    """
    return observe(a, vantage) == observe(b, vantage)


__all__ = [
    "observable_divergence",
    "observably_indistinguishable",
    "observation_facts",
]
