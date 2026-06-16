"""Tests for SPEC-22 CU32 (H125): the verification-latency barrier -- the THROUGHPUT cost of safety.

Torch-free throughout: every policy keys on the exact oracle, the CU21 covering surface, and the
action's static reversibility, never the model (the worst-case omitter is the model). CU32 prices
verification in WALL-CLOCK, not oracle calls: a verifier has latency ``L``, the latency-hiding trick
(pipeline / commit-speculatively) re-breaches the irreversible slice CU27 isolated, and routing by
reversibility is the only safe-and-fast policy. Safety costs ``L`` x the irreversible danger rate.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise

from verisim.acd.latency_barrier import (
    CU32Config,
    CU32Result,
    LatScenario,
    _lat_scenarios,
    adversarial_policy,
    cu32_verdict,
    run_cu32,
    run_policy,
    throughput,
)


def _small() -> CU32Config:
    return CU32Config(max_episodes=20, latency=8)


@lru_cache(maxsize=1)
def _result() -> CU32Result:
    return run_cu32(_small())


def test_both_pools_present_and_carry_attacks() -> None:
    rev, irr, horizon = _lat_scenarios(_small())
    assert rev and irr, "expected both a reversible and an irreversible deployment pool"
    assert horizon > 0
    assert all(sc.reversible for sc in rev)
    assert all(not sc.reversible for sc in irr)
    assert any(sc.danger.attacks(sc.start) for sc in rev + irr)


def test_pipeline_safe_on_reversible_fails_on_irreversible() -> None:
    """The latency-hiding trap: pipelining is safe on a reversible danger (the late verdict still
    rolls back) but FAILS on an irreversible one (the send left before the verdict returned).
    """
    r = _result()
    pa_rev = r.cell("pipeline_all", "reversible")
    pa_irr = r.cell("pipeline_all", "irreversible")
    assert pa_rev.random_breach <= 1e-9
    assert pa_rev.adversarial_breach <= 1e-9  # reversible: deferred rollback still saves it
    assert pa_irr.adversarial_breach > 0.5  # irreversible: the speculative send is consummated
    # pipelining never stalls -> full throughput at any latency (the speed that tempts the trap)
    assert pa_rev.mean_stalls <= 1e-9 and pa_irr.mean_stalls <= 1e-9
    assert abs(pa_rev.throughput - 1.0) <= 1e-9 and abs(pa_irr.throughput - 1.0) <= 1e-9


def test_routing_is_safe_everywhere_and_never_stalls_the_reversible_class() -> None:
    r = _result()
    rt_rev = r.cell("routed", "reversible")
    rt_irr = r.cell("routed", "irreversible")
    for c in (rt_rev, rt_irr):
        assert c.random_breach <= 1e-9
        assert c.adversarial_breach <= 1e-9
    # the reversible half is pipelined -> zero stalls -> free; the irreversible half stalls
    assert rt_rev.mean_stalls <= 1e-9
    assert abs(rt_rev.throughput - 1.0) <= 1e-9
    assert rt_irr.mean_stalls > 0.0
    assert rt_irr.throughput < 1.0


def test_barrier_all_is_safe_but_stalls_every_consult() -> None:
    r = _result()
    for cls in ("reversible", "irreversible"):
        c = r.cell("barrier_all", cls)
        assert c.adversarial_breach <= 1e-9  # safe everywhere
        # it stalls on every consult -- including the reversible ones it need not (the waste)
        assert c.mean_stalls > 0.0
        assert abs(c.mean_stalls - c.mean_consults) <= 1e-9
        assert c.throughput < 1.0


def test_unverified_is_unsafe_and_full_throughput() -> None:
    r = _result()
    for cls in ("reversible", "irreversible"):
        c = r.cell("unverified", cls)
        assert c.adversarial_breach > 0.5  # the blind model misses every danger
        assert c.mean_stalls <= 1e-9 and abs(c.throughput - 1.0) <= 1e-9


def test_latency_curve_routed_beats_barrier_and_both_decay() -> None:
    """Safety costs throughput: both safe policies decay with L; routing beats stalling-all."""
    r = _result()
    curve = r.latency_curve
    assert curve[0].latency == 0
    # at L=0 there is no barrier cost -- everything is full throughput
    assert abs(curve[0].routed_throughput - 1.0) <= 1e-9
    assert abs(curve[0].barrier_all_throughput - 1.0) <= 1e-9
    # routed strictly beats barrier for every L>0 (it never stalls the reversible consults)
    for p in curve:
        if p.latency > 0:
            assert p.routed_throughput > p.barrier_all_throughput + 1e-9
    # both decay monotonically in L; pipeline is flat at full throughput
    routed = [p.routed_throughput for p in curve]
    barrier = [p.barrier_all_throughput for p in curve]
    assert all(b <= a + 1e-9 for a, b in pairwise(routed)) and routed[-1] < routed[0]
    assert all(b <= a + 1e-9 for a, b in pairwise(barrier)) and barrier[-1] < barrier[0]
    assert all(abs(p.pipeline_all_throughput - 1.0) <= 1e-9 for p in curve)


def test_mix_law_safety_free_when_reversible_breach_grows_with_irreversibility() -> None:
    """The mix law: routed throughput rises to 1 as f->0 (free safety); pipeline's residual breach
    grows linearly with the irreversible fraction f.
    """
    r = _result()
    law = r.fraction_law
    assert law[0].irreversible_fraction == 0.0 and law[-1].irreversible_fraction == 1.0
    # f=0 (all reversible): routed pipelines everything -> full throughput, safety is free
    assert abs(law[0].routed_throughput - 1.0) <= 1e-9
    # routed throughput is non-increasing in f (it rises as the world gets more reversible)
    routed = [p.routed_throughput for p in law]
    assert all(b <= a + 1e-9 for a, b in pairwise(routed)) and routed[-1] < routed[0]
    # pipeline residual breach is zero at f=0 and increases with f
    breach = [p.pipeline_all_breach for p in law]
    assert breach[0] <= 1e-9
    assert all(b >= a - 1e-9 for a, b in pairwise(breach)) and breach[-1] > breach[0]


def test_throughput_formula() -> None:
    # actions / (horizon + L*stalls): no stalls -> 1.0; stalls and L>0 -> < 1.0; monotone in L
    assert throughput(48, 0.0, 16) == 1.0
    assert throughput(48, 4.0, 0) == 1.0  # zero latency -> stalls are free
    t8 = throughput(48, 4.0, 8)
    t16 = throughput(48, 4.0, 16)
    assert 0.0 < t16 < t8 < 1.0


def test_verdict_headlines() -> None:
    v = cu32_verdict(_result())
    assert v["pipeline_reversible_safe"] is True
    assert v["pipeline_irreversible_fails"] is True
    assert v["pipeline_full_throughput"] is True
    assert v["routed_safe_everywhere"] is True
    assert v["routed_reversible_never_stalls"] is True
    assert v["barrier_all_safe_everywhere"] is True
    assert v["routed_beats_barrier_for_latency"] is True
    assert v["routed_throughput_decays_with_latency"] is True
    assert v["routed_free_when_fully_reversible"] is True
    assert v["pipeline_breach_grows_with_irreversibility"] is True
    saving = v["throughput_saving_vs_barrier"]
    assert isinstance(saving, float) and saving > 1.0  # routing buys back the reversible slice


def test_pipeline_breaches_same_scenario_routing_saves_unit() -> None:
    """A decisive unit: on one irreversible deployment the speculative pipeline is breached by the
    attacker, while routing (which stalls the irreversible consult) is not -- at a throughput cost.
    """
    _rev, irr, _h = _lat_scenarios(_small())
    sc = next(s for s in irr if s.danger.attacks(s.start))
    assert isinstance(sc, LatScenario) and not sc.reversible
    # pipelining hides latency (zero stalls) but the adversary's send lands
    _b, _c, stalls_pipe, _r = run_policy(sc, "pipeline_all", 8)
    assert stalls_pipe == 0
    assert adversarial_policy(sc, "pipeline_all") is True
    # routing stalls the irreversible consult -> the oracle blocks the send before it commits
    _b, _c2, stalls_routed, _r2 = run_policy(sc, "routed", 8)
    assert stalls_routed > 0
    assert adversarial_policy(sc, "routed") is False
