"""Structural invariants for the speculative-rollout primitive (SPEC-13 §3, §7).

The speculative loop adds almost no new primitive over the shipped propose-verify-correct loop: the
accepted-prefix length is exactly ``faithful_horizon`` of the draft window. These tests pin the
*invariants* (faithful-by-construction, budget never exceeded, accepted prefix == horizon,
the i.i.d. law's monotonicity), not magnitudes (the macOS-first principle).
"""

from __future__ import annotations

from verisim.experiments.sr_common import StallDrafter, net_world
from verisim.loop.policy import SpeculativeConsult
from verisim.loop.speculative import (
    accepted_prefix_law,
    fixed_interval_rollout,
    free_run_divergences,
    speculative_rollout,
)
from verisim.metrics.horizon import faithful_horizon


def _setup(n: int = 40, alpha: float = 0.8, seed: int = 0):
    world = net_world(n_hosts=5, n_ports=2)
    s0, actions = world.make_actions(seed, n)
    drafter = StallDrafter(world.oracle_step, alpha, seed=seed)
    return world, s0, actions, drafter


def test_accepted_prefix_law_bounds_and_monotonicity() -> None:
    assert accepted_prefix_law(1.0, 8) == 8.0  # always accept -> whole window
    assert accepted_prefix_law(0.0, 8) == 0.0  # never accept a free-run step
    # Monotone increasing in alpha and in k.
    assert accepted_prefix_law(0.9, 8) > accepted_prefix_law(0.5, 8)
    assert accepted_prefix_law(0.8, 16) > accepted_prefix_law(0.8, 4)
    # Bounded above by k.
    for a in (0.1, 0.5, 0.9, 0.99):
        assert accepted_prefix_law(a, 10) <= 10.0


def test_speculative_is_faithful_by_construction() -> None:
    # With an unbounded correction budget every produced step is within epsilon (the exact-by-
    # construction property): faithful_steps == total_steps and no divergence exceeds epsilon.
    world, s0, actions, drafter = _setup()
    eps = 0.1
    rec = speculative_rollout(s0, actions, drafter, world.oracle_step, world.diverge,
        k=8, epsilon=eps)
    assert rec.faithful_steps == rec.total_steps == len(actions)
    assert all(d <= eps for d in rec.divergences)
    assert len(rec.divergences) == len(actions)


def test_accepted_prefix_equals_faithful_horizon_of_window() -> None:
    # The first window's accepted prefix is exactly faithful_horizon of the free-run draft window.
    world, s0, actions, drafter = _setup()
    eps = 0.05
    k = 8
    divs = free_run_divergences(s0, actions[:k], drafter, world.oracle_step, world.diverge)
    expected = faithful_horizon(divs, eps)
    rec = speculative_rollout(s0, actions, drafter, world.oracle_step, world.diverge,
        k=k, epsilon=eps)
    assert rec.accepted_prefixes[0] == expected


def test_budget_cap_never_exceeded_and_drift_accumulates() -> None:
    world, s0, actions, drafter = _setup(n=80, alpha=0.6)
    eps = 0.03
    capped = speculative_rollout(
        s0, actions, drafter, world.oracle_step, world.diverge,
        k=8, epsilon=eps, max_corrections=2,
    )
    assert capped.corrections <= 2
    # Under a tight budget the rollout free-runs the tail, so it is no longer fully faithful.
    assert capped.faithful_steps < capped.total_steps


def test_verify_steps_stop_at_first_divergence() -> None:
    # Each window verifies only the accepted prefix + 1 (it stops at the break), never the whole k.
    world, s0, actions, drafter = _setup(alpha=0.5)
    eps = 0.02
    rec = speculative_rollout(s0, actions, drafter, world.oracle_step, world.diverge,
        k=10, epsilon=eps)
    for a, w in zip(rec.accepted_prefixes, rec.window_lengths, strict=True):
        # verify cost of a window is min(a+1, w); summed it equals rec.verify_steps.
        assert a <= w
    assert rec.verify_steps == sum(min(a + 1, w) for a, w in
                                   zip(rec.accepted_prefixes, rec.window_lengths, strict=True))


def test_fixed_interval_rollout_consults_on_clock() -> None:
    world, s0, actions, drafter = _setup(n=60)
    eps = 0.05
    interval = 5
    rec = fixed_interval_rollout(
        s0, actions, drafter, world.oracle_step, world.diverge, interval=interval, epsilon=eps
    )
    assert rec.oracle_calls == len(actions) // interval
    assert rec.corrections == rec.oracle_calls  # every consult is a full snap
    assert 0 <= rec.faithful_steps <= rec.total_steps


def test_perfect_drafter_needs_no_corrections() -> None:
    world, s0, actions, _ = _setup()
    perfect = StallDrafter(world.oracle_step, alpha=1.0, seed=1)
    rec = speculative_rollout(s0, actions, perfect, world.oracle_step, world.diverge,
        k=8, epsilon=0.0)
    assert rec.corrections == 0
    assert rec.faithful_steps == rec.total_steps


def test_determinism() -> None:
    world, s0, actions, drafter = _setup()
    a = speculative_rollout(s0, actions, drafter, world.oracle_step, world.diverge,
        k=8, epsilon=0.05)
    b = speculative_rollout(s0, actions, drafter, world.oracle_step, world.diverge,
        k=8, epsilon=0.05)
    assert a == b


def test_cost_two_tier_ordering() -> None:
    # cheap verify (cheap_ratio -> 0) is cheaper than charging full for verify (cheap_ratio = 1).
    world, s0, actions, drafter = _setup(alpha=0.6)
    rec = speculative_rollout(s0, actions, drafter, world.oracle_step, world.diverge,
        k=8, epsilon=0.03)
    assert rec.cost(0.0) <= rec.cost(0.5) <= rec.cost(1.0)
    assert rec.cost(0.0) == rec.corrections


def test_speculative_consult_policy() -> None:
    from verisim.loop.policy import StepContext

    pol = SpeculativeConsult(k=4)
    assert [pol.should_consult(StepContext(step=t)) for t in range(8)] == [
        False, False, False, True, False, False, False, True
    ]
    assert pol.draft_length(0.0) == 4  # no calibration -> fixed k
    cal = SpeculativeConsult(k=4, calibrate=lambda s: 2 + round(8 * s))
    assert cal.draft_length(0.0) == 2
    assert cal.draft_length(1.0) == 10
