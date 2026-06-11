"""UA7 -- the predictive defender: faithfulness IS load-bearing for planning (SPEC-20 §7, H79).

The whole H74 → H78 → drift-profile chain converged on one mechanism: a *reactive* defender (observe
the true compromise state, isolate exposed hosts) does not need a faithful model, no matter how much
it drifts, because it reacts instead of plans. The drift profile sealed it -- the flat model is
faithful on control-relevant state (host-up 0%, services ~1-3%, links ~4-8%) and drifts mostly on
control-irrelevant flows (~25%), so reactive containment is robust at every horizon tested.

UA7 closes the boundary from the other side. A **model-predictive** defender does not react -- it
*plans*: each step it rolls the model forward `k` steps for each candidate isolation and picks the
one it predicts best contains the spread. Now the model's multi-step prediction *is* the thing
acted on, so faithfulness matters directly, and the model's drift **compounds** over the `k`-step
lookahead (the SPEC-10 compounding the program studies). The contrast is the result: the *same*
predictive defender, planning with the exact oracle vs the grounded (ρ-corrected) vs the free
(uncorrected) model, all evaluated against the *true* dynamics. H79: planning with a faithful model
contains materially better than planning with a drifted one -- the positive that draws the boundary
**reactive control needs no faithfulness; predictive control does**, the program's core thesis
(faithful models are *for planning*, SPEC-12) made a measurement.

No training (the planner is a fixed lookahead policy parameterized by its model), so it is fast and
clean. The adversary is made deterministic here (spread to the most-dangerous beachhead) so the
planner has a well-defined thing to predict; reality and planner share that adversary rule, which
isolates model faithfulness as the only variable. CPU-only, seeded.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from verisim.net.action import NetAction
from verisim.net.state import NetworkState

from .containment import (
    ContainmentConfig,
    DefenderAction,
    _beachheads,
    containment_fraction,
    legal_actions,
    seed_topology,
)

# A step function evolves the network state under one NetAction -- the planner's model of dynamics
# (the exact oracle, or a learned M_θ wrapped to predict-and-apply).
StepFn = Callable[[NetworkState, NetAction], NetworkState]

_ADVANCE = NetAction(raw="advance", name="advance", args=())


def deterministic_spread(net: NetworkState, compromised: frozenset[str]) -> frozenset[str]:
    """Spread to the single most-dangerous beachhead (most services; tie-break by id).

    Deterministic so the predictive planner has a well-defined adversary to predict; reality uses
    the *same* rule, so the only thing that varies across planners is the model they roll forward.
    """
    cands = _beachheads(net, compromised)
    if not cands:
        return compromised
    victim = max(cands, key=lambda h: (len(net.hosts[h].services), h))
    return compromised | {victim}


def plan_isolation(
    step: StepFn, net: NetworkState, compromised: frozenset[str], config: ContainmentConfig, k: int,
) -> DefenderAction:
    """Pick the isolate (or noop) whose ``k``-step model rollout best contains the spread.

    For each candidate the planner applies the action, then rolls ``k`` steps forward under its own
    ``step`` model and the deterministic adversary, and scores the predicted final containment. The
    model's drift compounds over the lookahead -- exactly why a faithful model plans better here.
    """
    candidates = [a for a in legal_actions(net, config) if a.kind in ("isolate", "noop")]
    best = DefenderAction("noop")
    best_score = -1.0
    for action in candidates:
        s = step(net, action.to_net_action())
        comp = deterministic_spread(s, compromised)
        for _ in range(k - 1):
            s = step(s, _ADVANCE)
            comp = deterministic_spread(s, comp)
        score = containment_fraction(s, comp)
        if score > best_score:
            best, best_score = action, score
    return best


def run_predictive_episode(
    true_step: StepFn, planner_step: StepFn, config: ContainmentConfig, seed: int, k: int,
) -> float:
    """One episode: reality evolves via ``true_step``; the defender plans with ``planner_step``.

    Returns the true final containment. The defender spends its ``cut_budget`` planning isolations
    with its model; once spent it can only noop. Reality (``true_step`` + deterministic adversary)
    governs the actual compromise, so a planner with a drifted model isolates the wrong hosts.
    """
    rng = random.Random(seed)
    net, compromised = seed_topology(config, rng)
    isolations = 0
    budget = config.cut_budget if config.cut_budget is not None else config.episode_steps
    for _ in range(config.episode_steps):
        if isolations < budget:
            action = plan_isolation(planner_step, net, compromised, config, k)
        else:
            action = DefenderAction("noop")
        if action.kind == "isolate":
            isolations += 1
        net = true_step(net, action.to_net_action())  # REALITY evolves
        compromised = deterministic_spread(net, compromised)  # REALITY adversary
    return containment_fraction(net, compromised)


def plan_open_loop(
    planner_step: StepFn, config: ContainmentConfig, seed: int,
) -> list[DefenderAction]:
    """Plan the whole episode's isolation sequence from the model's predicted trajectory.

    Open-loop: the defender rolls its model forward over the full horizon (no ground-truth feedback
    during execution -- the "planning in imagination" regime the oracle exists for) and greedily
    isolates the most-dangerous predicted-exposed host while budget remains. A faithful model's
    trajectory tracks reality, so the plan targets the right hosts; a drifted model's plan diverges.
    """
    rng = random.Random(seed)
    net, compromised = seed_topology(config, rng)
    isolations = 0
    budget = config.cut_budget if config.cut_budget is not None else config.episode_steps
    plan: list[DefenderAction] = []
    for _ in range(config.episode_steps):
        action = DefenderAction("noop")
        if isolations < budget:
            exposed = _beachheads(net, compromised)
            if exposed:
                host = max(exposed, key=lambda h: (len(net.hosts[h].services), h))
                action = DefenderAction("isolate", host=host)
        if action.kind == "isolate":
            isolations += 1
        plan.append(action)
        net = planner_step(net, action.to_net_action())  # PREDICTED evolution (model)
        compromised = deterministic_spread(net, compromised)
    return plan


def run_open_loop_episode(
    true_step: StepFn, planner_step: StepFn, config: ContainmentConfig, seed: int,
) -> float:
    """Plan once with ``planner_step`` (open-loop), then execute the fixed plan against reality."""
    plan = plan_open_loop(planner_step, config, seed)
    net, compromised = seed_topology(config, random.Random(seed))  # same seed -> same start
    for action in plan:
        net = true_step(net, action.to_net_action())
        compromised = deterministic_spread(net, compromised)
    return containment_fraction(net, compromised)


def model_step_fn(model: object) -> StepFn:
    """Wrap a ``NetModel`` (predict_delta) as a step function (predict the delta, apply it)."""
    from verisim.netdelta.apply import apply

    def step(net: NetworkState, action: NetAction) -> NetworkState:
        return apply(net, model.predict_delta(net, action))  # type: ignore[attr-defined]

    return step


def oracle_step_fn(oracle: object) -> StepFn:
    """Wrap an oracle as a step function -- the faithful planner's perfect lookahead model."""

    def step(net: NetworkState, action: NetAction) -> NetworkState:
        result: NetworkState = oracle.step(net, action).state  # type: ignore[attr-defined]
        return result

    return step


def run_reactive_episode(
    true_step: StepFn, config: ContainmentConfig, seed: int,
) -> float:
    """Baseline: a reactive defender that isolates the most-dangerous *exposed* host (no model).

    The drift-robust policy UA2 found needs no faithfulness -- here as the reference line the
    predictive planners are measured against (reactive ≈ flat across model faithfulness; predictive
    should separate).
    """
    rng = random.Random(seed)
    net, compromised = seed_topology(config, rng)
    isolations = 0
    budget = config.cut_budget if config.cut_budget is not None else config.episode_steps
    for _ in range(config.episode_steps):
        action: DefenderAction = DefenderAction("noop")
        if isolations < budget:
            exposed = _beachheads(net, compromised)
            if exposed:
                host = max(exposed, key=lambda h: (len(net.hosts[h].services), h))
                action = DefenderAction("isolate", host=host)
        if action.kind == "isolate":
            isolations += 1
        net = true_step(net, action.to_net_action())
        compromised = deterministic_spread(net, compromised)
    return containment_fraction(net, compromised)
