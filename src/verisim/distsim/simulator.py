"""The cluster simulator an agent calls (SPEC-7 §7) — the SLM/LLM complementarity layer.

A computer-use or cyber-defense agent acts on a **running distributed system** ("push this config,"
"fail over this primary," "drain this node"), so the distributed world is exactly the simulator it
needs: not a competitor to the LLM, but the cheap, faithful, verifiable cluster the LLM reasons
*over* — the grounded WebDreamer (§1.3). This module packages the loop's proposer (`M_θ`, a
:class:`~verisim.distloop.model.DistModel`) + the reference oracle into the **protocol** §7
specifies — one object that both *predicts the next cluster state* (the loop interface) and
*simulates a plan* (the agent interface, new), the distributed analogue of :mod:`verisim.hostsim`:

  - :meth:`DistSimulator.imagine` — roll ``M_θ`` forward over a plan (a sequence of admin/client ops
    the LLM proposes) with **no oracle**: fast "plan in imagination" (Dreamer), the cheap draft.
  - :meth:`DistSimulator.verify` — the same plan, proposing-the-model **and** consulting the oracle
    step by step, returning a :class:`DistPlanReport`. Beyond the host's bit-exact plan horizon, the
    distributed report carries the two things the cluster world makes meaningful:

      * a **consistency-faithful plan horizon** distinct from the bit-exact one — how many leading
        plan steps the agent can trust the model's *consistency* (split-brain) prediction, which
        (per ED5/H19) outlasts the bit-exact horizon when the error hides in the consistency-
        invisible in-flight medium; and
      * **change-safety as differential consistency-faithfulness** (the securifine pattern, §2.5,
        `DD-D8`): *"will this plan break consistency/availability?"* scored as the change in the
        cluster's consistency health (fraction of objects converged) from before to after — and,
        crucially, whether the **model agrees with the oracle** on that safe/unsafe verdict. The
        oracle, not a pattern-matcher, is the verifier.

The split is the spec's standing one: the model drafts cluster dynamics fast (speculative
execution), the oracle verifies on the agent's budget, and the LLM is escalated to only for
natural-language-intent → plan translation. Dependency-free except the model (the loop's
``DistModel`` — the dependency-free baselines satisfy it), like the host ``hostsim`` package.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.dist.action import DistAction, parse_dist_action
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import apply
from verisim.dist.state import DistributedState
from verisim.distloop.model import DistModel
from verisim.distmetrics import (
    consistency_faithfulness,
    dist_facts,
    divergence,
    object_consistency_view,
)
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.metrics.horizon import faithful_horizon

from .goal import DistGoal

Plan = list[str]  # a sequence of admin/client op strings the agent proposes (e.g. "put n0 x v1")


def _parse(plan: Plan) -> list[DistAction]:
    return [parse_dist_action(step) for step in plan]


def consistency_health(state: DistributedState, config: DistConfig) -> float:
    """The cluster's consistency health: the fraction of objects that are **converged** (not split).

    ``1.0`` iff every object's replicas agree (no split-brain). The securifine *safety property* a
    plan's change-safety is the differential of (§7, `DD-D8`).
    """
    objects = config.objects
    if not objects:
        return 1.0
    converged = sum(len(object_consistency_view(state, o)) == 1 for o in objects)
    return converged / len(objects)


@dataclass(frozen=True)
class PlanRollout:
    """The model's imagined trajectory for a plan (no oracle) — the cheap draft (§7)."""

    states: list[DistributedState]  # [s_0, ŝ_1, ..., ŝ_T], predicted states after each plan step
    actions: list[DistAction]

    @property
    def final(self) -> DistributedState:
        return self.states[-1]


@dataclass(frozen=True)
class DistPlanReport:
    """The verified outcome of a plan — predicted vs true, faithfulness, change-safety, task."""

    n_steps: int
    epsilon: float
    predicted_final: DistributedState
    true_final: DistributedState
    divergences: list[float]  # bit-exact predicted-vs-true divergence after each plan step
    plan_faithful_horizon: int  # leading plan steps the agent can trust the model bit-for-bit
    consistency_divergences: list[float]  # 1 - consistency-faithfulness after each step
    consistency_plan_horizon: int  # leading steps the model's consistency prediction is trusted
    final_divergence: float
    oracle_calls: int
    oracle_cost: int  # the full-state fact count verified across the plan (the §5 consult cost)
    trusted: bool  # the model stayed bit-faithful for the *whole* plan
    # change-safety (the securifine differential, §7 / DD-D8): does the plan break consistency,
    # and does the model agree with the oracle on that verdict?
    health_before: float
    model_health_after: float
    true_health_after: float
    safe_predicted: bool  # the model says the plan does not reduce consistency health
    safe_true: bool  # the oracle says the plan does not reduce consistency health
    safety_agreement: bool  # the model agrees with the oracle on the change-safety verdict
    # the task oracle (§7, the "third oracle")
    goal_predicted: bool | None
    goal_true: bool | None
    goal_agreement: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "n_steps": self.n_steps, "epsilon": self.epsilon,
            "plan_faithful_horizon": self.plan_faithful_horizon,
            "consistency_plan_horizon": self.consistency_plan_horizon,
            "final_divergence": self.final_divergence,
            "oracle_calls": self.oracle_calls, "oracle_cost": self.oracle_cost,
            "trusted": self.trusted,
            "health_before": self.health_before,
            "model_health_after": self.model_health_after,
            "true_health_after": self.true_health_after,
            "safe_predicted": self.safe_predicted, "safe_true": self.safe_true,
            "safety_agreement": self.safety_agreement,
            "goal_predicted": self.goal_predicted, "goal_true": self.goal_true,
            "goal_agreement": self.goal_agreement,
        }


class DistSimulator:
    """The §7 protocol: predict-next-state (the loop) **and** simulate-a-plan (the agent)."""

    def __init__(
        self,
        model: DistModel,
        oracle: DistOracle | None = None,
        *,
        config: DistConfig = DEFAULT_DIST_CONFIG,
    ) -> None:
        self.model = model
        self.config = config
        self.oracle = oracle or ReferenceDistOracle(config)

    # -- the loop interface (predict next state) --------------------------------

    def predict_next(self, state: DistributedState, action: DistAction) -> DistributedState:
        """Apply the model's predicted delta — the loop's one-step prediction (§5, §8)."""
        return apply(state, self.model.predict_delta(state, action))

    # -- the agent interface (simulate a plan) ----------------------------------

    def imagine(self, state: DistributedState, plan: Plan) -> PlanRollout:
        """Roll ``M_θ`` forward over ``plan`` with no oracle — the fast "plan in imagination"."""
        actions = _parse(plan)
        states = [state]
        current = state
        for action in actions:
            current = self.predict_next(current, action)
            states.append(current)
        return PlanRollout(states=states, actions=actions)

    def verify(
        self, state: DistributedState, plan: Plan, *, epsilon: float = 0.0,
        goal: DistGoal | None = None,
    ) -> DistPlanReport:
        """Simulate ``plan`` and check it against the oracle step by step (§7, the honest version).

        Rolls the model's free-running imagination (what the agent would believe) against the
        oracle's true trajectory, recording both the bit-exact and the consistency divergence per
        step. The bit-exact and consistency plan horizons are the first step each divergence exceeds
        ``epsilon`` (the §17.8 ``H_ε`` for a plan, two flavors). Change-safety is the consistency-
        health differential (model vs oracle); ``goal`` adds the task-level faithfulness (§7).
        """
        actions = _parse(plan)
        predicted = state
        true = state
        divergences: list[float] = []
        consistency_divergences: list[float] = []
        oracle_cost = 0
        for action in actions:
            predicted = self.predict_next(predicted, action)  # model imagination (free-running)
            true = self.oracle.step(true, action).state  # the truth
            divergences.append(divergence(true, predicted))
            consistency_divergences.append(1.0 - consistency_faithfulness(true, predicted))
            oracle_cost += len(dist_facts(true))  # the cost of verifying this step in full (§5)
        n = len(actions)

        health_before = consistency_health(state, self.config)
        model_health_after = consistency_health(predicted, self.config)
        true_health_after = consistency_health(true, self.config)
        safe_predicted = model_health_after >= health_before
        safe_true = true_health_after >= health_before

        goal_pred = goal.holds(predicted) if goal is not None else None
        goal_true = goal.holds(true) if goal is not None else None
        plan_h = faithful_horizon(divergences, epsilon)
        return DistPlanReport(
            n_steps=n, epsilon=epsilon,
            predicted_final=predicted, true_final=true,
            divergences=divergences, plan_faithful_horizon=plan_h,
            consistency_divergences=consistency_divergences,
            consistency_plan_horizon=faithful_horizon(consistency_divergences, epsilon),
            final_divergence=divergences[-1] if divergences else 0.0,
            oracle_calls=n, oracle_cost=oracle_cost,
            trusted=plan_h == n,
            health_before=health_before,
            model_health_after=model_health_after,
            true_health_after=true_health_after,
            safe_predicted=safe_predicted, safe_true=safe_true,
            safety_agreement=safe_predicted == safe_true,
            goal_predicted=goal_pred, goal_true=goal_true,
            goal_agreement=(goal_pred == goal_true) if goal is not None else None,
        )
