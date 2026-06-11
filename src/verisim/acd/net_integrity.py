"""UA10 -- network flow-integrity: the cross-world confirmation (SPEC-20 §7, H82).

The SPEC-20 boundary law was drawn on the **host** world: world-model faithfulness is load-bearing
for control *exactly when* the task keys on the **content** the model drifts on (host file-writes,
UA8/H80) and not on the **structure** it learns faithfully (UA2-UA7). UA9 then showed that content
faithfulness can be *bought* cheaply at a budget ρ (the useful knee). Both results live entirely on
the host world -- so the symmetric question is open: does the law (and the knee) reproduce on the
**network** world, whose content dimension is **flows**?

The drift profile says it should: free-running the network flagship beside the oracle, the model is
faithful on the discrete *structure* (host up/down 0.000, firewall 0.003, services 0.011, links
0.044) and drifts almost entirely on the *content* -- established **flows 0.252**. So the content-
keyed network task is **flow-integrity defense**: an adversarial workload opens connections (flows)
over an episode; the defender predicts which flows will be live and pre-positions to protect the
budget flows it predicts -- a decision riding entirely on the model's prediction of *which flows
exist*, the content the network model drifts ~25% on. Unlike host file-writes (which only
accumulate), flows come *and go* (``advance`` re-validates and drops unreachable flows), so the
final flow set is a genuinely harder content target.

    reward = |protected ∩ truly-live-flows| / min(budget, |truly-live-flows|)

A **faithful** predictor (oracle rollout) protects the right flows -> reward 1.0; a **free**
predictor (raw network `M_θ` rollout) mis-predicts the live flows -> protects the wrong ones, worse
as the horizon (and the compounded flow drift) grows. The **ρ-grounded** predictor re-anchors to the
oracle's truth every ``round(1/ρ)`` steps -- the UA9 useful knee, here on the network. H82: the
content-keyed positive (faithful > free, widening with horizon) and the useful knee (catch monotone
in ρ, recovered sub-linearly) both reproduce on the network world -- the law is cross-world, not
host-specific. No training; CPU-only, seeded.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from verisim.net.action import NetAction
from verisim.net.config import DEFAULT_NET_CONFIG, NetConfig
from verisim.net.state import Flow, NetworkState
from verisim.netdata.drivers import NetDriver
from verisim.netoracle.base import NetOracle
from verisim.netoracle.reference import ReferenceNetworkOracle

# A network step function evolves the state under one action -- the predictor's model of dynamics
# (the exact oracle, or a learned network `M_θ` wrapped to predict-and-apply).
NetStepFn = Callable[[NetworkState, NetAction], NetworkState]


def established_flows(state: NetworkState) -> set[Flow]:
    """The set of live ``(src, dst, port)`` flows in one state."""
    return set(state.flows)


def make_net_workload(
    seed: int, n_steps: int, *, driver: str = "weighted", oracle: NetOracle | None = None,
    config: NetConfig = DEFAULT_NET_CONFIG,
) -> tuple[NetworkState, tuple[NetAction, ...]]:
    """A seeded flow-establishing workload from a *connected* network (the adversarial workload).

    Flows only establish where there is reachability (links + listening services + a firewall
    allow), so a workload from the empty network almost never opens a persistent flow. We therefore
    start from the same seeded *connected* topology the SPEC-20 containment line uses
    (:func:`acd.containment.seed_topology`, the regime where the drift profile measured flow drift
    ~0.252), then run the in-distribution ``weighted`` driver -- so ``connect`` actions land on real
    reachability and the flow content set is non-trivial. ``ContainmentConfig()`` defaults to the
    flagship's world (``n_hosts=5, n_ports=3 == DEFAULT_NET_CONFIG``). The driver's rng is offset
    from the topology seed so the two are independent. Returns the connected *start* state + the
    actions (the predictor replays from there).
    """
    from verisim.acd.containment import ContainmentConfig, seed_topology

    oracle = oracle or ReferenceNetworkOracle()
    start, _ = seed_topology(ContainmentConfig(), random.Random(seed))
    drv = NetDriver(name=driver, config=config, rng=random.Random(seed + 7000))
    state = start
    actions: list[NetAction] = []
    for _ in range(n_steps):
        action = drv.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return start, tuple(actions)


def rollout_flows(
    step: NetStepFn, start: NetworkState, actions: Sequence[NetAction],
) -> set[Flow]:
    """Roll the workload under ``step``; return the **cumulative** set of flows ever established.

    The defensive content set is every connection the adversary opened during the episode (detect /
    pre-block them all), not merely the flows live at the end -- so it *accumulates*, exactly as the
    host file-integrity set does (written files never un-write, so the host final state already *is*
    the cumulative set). This makes a flow that appears then drops still count, and makes the
    mid-rollout grounding meaningful: every step the model mis-predicts pollutes the cumulative set.
    """
    state = start
    seen = established_flows(state)
    for action in actions:
        state = step(state, action)
        seen |= established_flows(state)
    return seen


def predictive_flow_defense_reward(
    predictor: NetStepFn, true_step: NetStepFn, start: NetworkState,
    actions: Sequence[NetAction], budget: int,
) -> float:
    """Protect the ``budget`` predicted-live flows; score against the true live flows.

    ``reward = |protected ∩ truly-live| / min(budget, |truly-live|)`` -- the fraction of (budget-
    limited) true flows the defender caught. A faithful predictor catches them all (1.0); a drifted
    one protects the wrong flows.
    """
    predicted = sorted(rollout_flows(predictor, start, actions))  # deterministic budget order
    true_live = rollout_flows(true_step, start, actions)
    if not true_live:
        return 1.0
    protected = set(predicted[:budget])
    caught = len(protected & true_live)
    return caught / min(budget, len(true_live))


def grounded_flow_rollout(
    model: object, oracle: NetOracle, start: NetworkState, actions: Sequence[NetAction], rho: float,
) -> tuple[set[Flow], set[Flow], int]:
    """The ρ-grounded predictor: free-run `M_θ`, re-anchor to truth every ``round(1/ρ)`` step.

    The UA9 useful-knee mechanism, on the network world. A parallel true trajectory advances under
    the oracle; at each consultation step the predicted state is snapped to that truth (the defender
    paid an oracle call), and between consultations the model free-runs. ``ρ=1`` recovers the
    faithful predictor (predicted ≡ true, ``|actions|`` calls); ``ρ=0`` recovers the free predictor.
    Returns ``(predicted_flows, true_flows, oracle_calls)``.
    """
    from verisim.netdelta.apply import apply

    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    true = start
    predicted = start
    true_seen = established_flows(true)
    pred_seen = established_flows(predicted)
    calls = 0
    for i, action in enumerate(actions, start=1):
        true = oracle.step(true, action).state
        true_seen |= established_flows(true)
        if rho >= 1.0 or (interval and i % interval == 0):
            predicted = true  # CONSULT -- re-anchor to the truth the oracle call returned
            calls += 1
        else:
            delta = model.predict_delta(predicted, action)  # type: ignore[attr-defined]
            predicted = apply(predicted, delta)
        pred_seen |= established_flows(predicted)
    return pred_seen, true_seen, calls


def grounded_flow_defense_reward(
    model: object, oracle: NetOracle, start: NetworkState, actions: Sequence[NetAction],
    budget: int, rho: float,
) -> tuple[float, int]:
    """Protect the ``budget`` flows the ρ-grounded predictor expects live; score vs the truth.

    Same scoring as :func:`predictive_flow_defense_reward` but the predictor is the ρ-grounded
    rollout, so the catch rate is a function of the budget. Returns ``(reward, oracle_calls)``.
    """
    predicted_set, true_live, calls = grounded_flow_rollout(model, oracle, start, actions, rho)
    if not true_live:
        return 1.0, calls
    protected = set(sorted(predicted_set)[:budget])
    caught = len(protected & true_live)
    return caught / min(budget, len(true_live)), calls


def oracle_step(oracle: NetOracle) -> NetStepFn:
    """The faithful predictor: the exact oracle as a step function."""

    def step(state: NetworkState, action: NetAction) -> NetworkState:
        return oracle.step(state, action).state

    return step


def model_step(model: object) -> NetStepFn:
    """The free predictor: a network ``M_θ`` (predict_delta) wrapped as a step function."""
    from verisim.netdelta.apply import apply

    def step(state: NetworkState, action: NetAction) -> NetworkState:
        return apply(state, model.predict_delta(state, action))  # type: ignore[attr-defined]

    return step
