"""Partial observation: the probe projection + its structural properties (SPEC-7 §5.4, DS3 incr 4).

Pins the three facts the partial-observation layer rests on, all dependency-free:

  - the in-flight replication medium is **unobservable** from any vantage;
  - a node that is **crashed** and a node that is **partitioned away** are byte-identical from a
    single vantage (the failure-detector indistinguishability), and separable from a paired one;
  - a *bit*-faithful state is necessarily *observably* faithful (observable divergence dominates),
    so the observable horizon can only exceed the bit horizon, never trail it.
"""

import random

from verisim.dist import DistConfig, DistributedState, observe, parse_dist_action
from verisim.dist.delta import MsgSend, apply
from verisim.dist.observe import Observation
from verisim.distdata import DistDriver
from verisim.distmetrics.observe import (
    observable_divergence,
    observably_indistinguishable,
    observation_facts,
)
from verisim.distoracle import ReferenceDistOracle

CONFIG = DistConfig(name="obs", nodes=("n0", "n1", "n2"), objects=("x", "y"))
ORACLE = ReferenceDistOracle(CONFIG)
ALL = frozenset(CONFIG.nodes)


def _run(cmds: list[str]) -> DistributedState:
    state = DistributedState.initial(CONFIG)
    for cmd in cmds:
        state = ORACLE.step(state, parse_dist_action(cmd)).state
    return state


# --- the in-flight medium is unobservable
# ---------------------------------------------------------


def test_inflight_is_never_observable():
    """A ``put`` enqueues replication messages; no probe, even the whole-cluster one, can see
    them."""
    state = _run(["put n0 x a"])
    assert state.inflight, "the async put should leave replication messages in flight"
    obs = observe(state, ALL)
    facts = observation_facts(obs)
    # No fact in the observation mentions a message id, payload, or the in-flight medium at all.
    assert all(f[0] != "msg" for f in facts)
    # And the Observation type carries no message field by construction.
    assert not hasattr(obs, "inflight")


def test_corrupting_only_inflight_leaves_observation_unchanged():
    """Mutating an in-flight message's payload changes the bit state but not any probe's view."""
    state = _run(["put n0 x a"])
    # corrupt one in-flight message payload (the ``subtle`` error class)
    msg_id = next(iter(state.inflight))
    m = state.inflight[msg_id]
    corrupted = apply(
        state, [MsgSend(m.id, m.src, m.dst, m.object_id, m.version, "Q", m.deliver_after)]
    )
    # Bit-divergent...
    from verisim.distmetrics.divergence import divergence

    assert divergence(state, corrupted) > 0.0
    # ...but observably identical at every vantage (the medium is invisible).
    assert observable_divergence(state, corrupted, ALL) == 0.0
    for node in CONFIG.nodes:
        assert observably_indistinguishable(state, corrupted, (node,))


# --- crash / partition indistinguishability
# -------------------------------------------------------


def test_crash_and_partition_indistinguishable_from_one_vantage():
    """From an external probe, a crashed node and a partitioned-away node project identically."""
    base = _run(["put n0 x a", "advance 5"])  # healed, all up, replicas converged
    crashed = ORACLE.step(base, parse_dist_action("crash n2")).state
    partitioned = ORACLE.step(base, parse_dist_action("partition n0 n1 | n2")).state
    # the single external vantage cannot tell which fault occurred
    assert observably_indistinguishable(crashed, partitioned, ("n0",))
    assert observe(crashed, ("n0",)) == observe(partitioned, ("n0",))


def test_paired_vantage_separates_crash_from_partition():
    """A probe that reaches the node's side sees a live isolated replica vs nothing —
    distinguishable."""
    base = _run(["put n0 x a", "advance 5"])
    crashed = ORACLE.step(base, parse_dist_action("crash n2")).state
    partitioned = ORACLE.step(base, parse_dist_action("partition n0 n1 | n2")).state
    # the paired vantage {n0, n2}: in the partition case n2 is up and its replica is visible; in
    # the
    # crash case n2 is down and contributes nothing.
    assert not observably_indistinguishable(crashed, partitioned, ("n0", "n2"))
    obs_part = observe(partitioned, ("n0", "n2"))
    obs_crash = observe(crashed, ("n0", "n2"))
    assert "n2" in obs_part.reachable
    assert "n2" in obs_crash.unreachable


def test_unreachable_carries_no_crash_or_partition_reason():
    """The ``unreachable`` set is reason-free — exactly what makes the two faults
    indistinguishable."""
    crashed = ORACLE.step(_run(["put n0 x a", "advance 5"]), parse_dist_action("crash n2")).state
    obs = observe(crashed, ("n0",))
    # n2 is simply 'unreachable'; nothing in the observation says 'down' vs 'partitioned'
    assert obs.unreachable == frozenset({"n2"})
    facts = observation_facts(obs)
    assert ("unreachable", "n2") in facts
    assert all("down" not in str(f) and "part" not in str(f) for f in facts)


# --- reachability + the bit-dominates property
# ----------------------------------------------------


def test_observe_full_vantage_sees_every_up_replica():
    """With no partition and all nodes up, the whole-cluster vantage observes every replica
    exactly."""
    state = _run(["put n0 x a", "advance 5"])
    obs = observe(state, ALL)
    assert obs.reachable == ALL
    assert not obs.unreachable
    assert obs.clock == state.clock
    for (obj, node), r in state.replicas.items():
        assert obs.replica_value(obj, node) == r.value


def test_down_vantage_observer_loses_the_cluster():
    """If every vantage node is down, the observer reaches nothing and the clock reads None."""
    state = ORACLE.step(_run(["put n0 x a"]), parse_dist_action("crash n0")).state
    obs = observe(state, ("n0",))  # the observer sits only on the crashed node
    assert obs.reachable == frozenset()
    assert obs.clock is None
    assert not obs.replicas


def test_bit_faithful_implies_observably_faithful_over_a_driver_rollout():
    """The structural dominance: whenever two states are bit-equal they are observably equal too.

    So the observable horizon dominates the bit horizon (ED12 Panel A's rigorous half). We exercise
    it by checking, over a fault-injecting driver rollout, that bit divergence 0 ⇒ observable
    divergence 0 at every vantage.
    """
    from verisim.distmetrics.divergence import divergence

    drv = DistDriver("contention", CONFIG, random.Random(7))
    truth = DistributedState.initial(CONFIG)
    # a second rollout that occasionally drops an edit, so some steps are bit-equal and some are
    # not
    pred = DistributedState.initial(CONFIG)
    rng = random.Random(11)
    for _ in range(40):
        action = drv.sample(truth)
        delta = ORACLE.step(truth, action).delta
        truth = apply(truth, delta)
        pdelta = list(delta)
        if pdelta and rng.random() < 0.3:
            pdelta.pop(rng.randrange(len(pdelta)))
        pred = apply(pred, pdelta)
        if divergence(truth, pred) == 0.0:
            for v in (("n0",), ("n1",), ("n0", "n2"), ALL):
                assert observable_divergence(truth, pred, v) == 0.0


def test_observation_is_canonical_and_hashable():
    """Two states with the same observable projection give equal, hashable Observations."""
    a = _run(["put n0 x a", "advance 5"])
    b = _run(["put n0 x a", "advance 5"])
    assert observe(a, ALL) == observe(b, ALL)
    assert isinstance(observe(a, ALL), Observation)
    assert hash(observe(a, ALL)) == hash(observe(b, ALL))  # frozen dataclass, frozenset fields
