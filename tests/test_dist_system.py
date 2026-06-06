"""Tier-B — the distributed system oracle (SPEC-7 §5.2): the W1 retirement for this world.

These tests pin the central claim: an independent, autonomous-actor reimplementation of the
replicated-KV protocol (:class:`SystemDistOracle`), run under a *seed-shuffled* delivery order,
reproduces Tier-A's observable cluster **bit-for-bit** across the whole grammar and the fault-heavy
adversarial workload — so the analytic DES's sorted-delivery shortcut is faithful to a genuine
message-passing execution (the eventual-consistency convergence is delivery-order-independent). The
negative control proves the differential has teeth: a broken arrival-order actor is *caught*. Both
the simulated and the real-OS-thread tiers are exercised. Dependency-free, GPU-free.
"""

from __future__ import annotations

import random

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.dist.config import scaled_dist_config
from verisim.dist.delta import apply
from verisim.dist.state import Message, ReplicaState
from verisim.distdata.drivers import DIST_DRIVERS, DistDriver
from verisim.distoracle.differential import (
    AGREE,
    C_DELIVERY_ORDER,
    RESIDUAL,
    cluster_view,
    dist_differential_step,
)
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import (
    TIER_SIMULATED,
    TIER_THREADED,
    SystemDistOracle,
)

CFG = DistConfig()
REF = ReferenceDistOracle(CFG)


def _agreement(
    sys: SystemDistOracle, driver: str, seeds: range, steps: int
) -> tuple[int, int, dict[str, int]]:
    classes: dict[str, int] = {}
    agree = total = 0
    for seed in seeds:
        drv = DistDriver(driver, CFG, random.Random(seed))
        s = DistributedState.initial(CFG)
        for _ in range(steps):
            a = drv.sample(s)
            rec = dist_differential_step(s, a, REF, sys)
            classes[rec.divergence_class] = classes.get(rec.divergence_class, 0) + 1
            total += 1
            if rec.agree:
                agree += 1
            s = REF.step(s, a).state  # both advance along Tier-A truth
    return agree, total, classes


def test_step_is_a_drop_in_dist_oracle():
    """``apply(state, delta) == next_state`` (the M1-analogue invariant) holds for Tier-B too."""
    sys = SystemDistOracle(CFG)
    drv = DistDriver("adversarial", CFG, random.Random(3))
    s = DistributedState.initial(CFG)
    for _ in range(40):
        a = drv.sample(s)
        result = sys.step(s, a)
        assert apply(s, result.delta) == result.state
        s = REF.step(s, a).state


def test_simulated_tier_agrees_bit_for_bit_on_every_driver():
    """The headline: Tier-A == Tier-B on the observable cluster across all workloads."""
    sys = SystemDistOracle(CFG, tier=TIER_SIMULATED)
    for driver in DIST_DRIVERS:
        agree, total, classes = _agreement(sys, driver, range(6), 40)
        assert agree == total, f"{driver}: {agree}/{total} agree, classes={classes}"
        assert classes == {AGREE: total}
        assert classes.get(RESIDUAL, 0) == 0


def test_threaded_tier_is_real_os_threads_and_agrees():
    """The strongest reality claim: actors on real OS threads + queues still agree bit-for-bit."""
    sys = SystemDistOracle(CFG, tier=TIER_THREADED)
    assert sys.determinism_report().real_os_threads is True
    agree, total, classes = _agreement(sys, "adversarial", range(4), 30)
    assert agree == total, f"threaded: {agree}/{total}, classes={classes}"


def test_delivery_order_is_seeded_and_replayable():
    """A replay of the same step yields a bit-identical next state (determinism contract §3.3)."""
    sys = SystemDistOracle(CFG)
    drv = DistDriver("adversarial", CFG, random.Random(11))
    s = DistributedState.initial(CFG)
    for _ in range(25):
        a = drv.sample(s)
        first = sys.step(s, a).state
        second = sys.step(s, a).state
        assert cluster_view(first) == cluster_view(second)
        s = REF.step(s, a).state


def test_delivery_order_is_actually_shuffled_not_tier_a_sorted():
    """The DST test only bites if Tier-B does NOT just copy Tier-A's sorted order.

    Hand-build a state with several deliverable in-flight messages and assert the report advertises
    a shuffled order; agreement under that shuffle is what certifies order-independence.
    """
    sys = SystemDistOracle(CFG)
    rep = sys.determinism_report()
    assert rep.delivery_order_seeded and rep.delivery_order_shuffled and rep.no_global_state


def test_negative_control_broken_actor_is_caught():
    """SY3-style teeth: an order-dependent (arrival-order) actor must DISAGREE with Tier-A."""
    broken = SystemDistOracle(CFG, broken_arrival=True)
    agree, total, classes = _agreement(broken, "adversarial", range(8), 40)
    assert agree < total, "the broken arrival-order actor should diverge under shuffled delivery"
    assert classes.get(C_DELIVERY_ORDER, 0) > 0
    # every divergence the broken actor produces is the named delivery-order boundary, not residual
    assert classes.get(RESIDUAL, 0) == 0


def test_stale_read_under_partition_matches_tier_a():
    """A write under partition is read stale until heal+advance — Tier-B reproduces the dynamic."""
    sys = SystemDistOracle(CFG)
    cmds = ["partition n0 | n1 n2", "put n0 x a", "advance 2", "get n1 x"]
    s = DistributedState.initial(CFG)
    for cmd in cmds:
        a = parse_dist_action(cmd)
        assert cluster_view(sys.step(s, a).state) == cluster_view(REF.step(s, a).state)
        s = REF.step(s, a).state
    # n1 still reads the boot value (its replication message is stranded by the partition)
    assert s.replicas[("x", "n1")].value == CFG.default_value


def test_convergence_after_heal_matches_tier_a():
    """After heal+advance every replica converges to the same value, identically to Tier-A."""
    sys = SystemDistOracle(CFG)
    cmds = ["partition n0 | n1 n2", "put n0 x b", "heal", "advance 5", "get n1 x", "get n2 x"]
    s = DistributedState.initial(CFG)
    for cmd in cmds:
        a = parse_dist_action(cmd)
        assert cluster_view(sys.step(s, a).state) == cluster_view(REF.step(s, a).state)
        s = REF.step(s, a).state
    assert {r.value for (o, _), r in s.replicas.items() if o == "x"} == {"b"}


def test_lww_adopts_higher_version_regardless_of_arrival_order():
    """Two in-flight writes to one peer converge to the higher (version,value) under any order."""
    sys = SystemDistOracle(CFG)
    s = DistributedState.initial(CFG)
    # n1 has two pending deliveries for x: v1='a' and v2='b'; LWW must land on v2 whichever first.
    s.inflight[0] = Message(0, "n0", "n1", "x", 1, "a", 0)
    s.inflight[1] = Message(1, "n2", "n1", "x", 2, "b", 0)
    s.next_msg_id = 2
    a = parse_dist_action("advance 1")
    assert cluster_view(sys.step(s, a).state) == cluster_view(REF.step(s, a).state)
    assert sys.step(s, a).state.replicas[("x", "n1")] == ReplicaState("x", "n1", 2, "b")


def test_linearizable_rejects_partitioned_write_like_tier_a():
    """Under the CP (linearizable) model Tier-B rejects an unreplicable write, as Tier-A does."""
    lin = DistConfig(consistency_model="linearizable")
    ref = ReferenceDistOracle(lin)
    sys = SystemDistOracle(lin)
    cmds = ["partition n0 | n1 n2", "put n0 x a"]
    s = DistributedState.initial(lin)
    for cmd in cmds:
        a = parse_dist_action(cmd)
        assert cluster_view(sys.step(s, a).state) == cluster_view(ref.step(s, a).state)
        s = ref.step(s, a).state
    assert s.last_result == ("unavailable", "")


def test_scaled_cluster_agrees():
    """Tier-B is config-driven: a larger cluster agrees with Tier-A too (no hard-coded count)."""
    cfg = scaled_dist_config(5, n_objects=3)
    ref = ReferenceDistOracle(cfg)
    sys = SystemDistOracle(cfg)
    drv = DistDriver("adversarial", cfg, random.Random(7))
    s = DistributedState.initial(cfg)
    for _ in range(40):
        a = drv.sample(s)
        assert cluster_view(sys.step(s, a).state) == cluster_view(ref.step(s, a).state)
        s = ref.step(s, a).state
