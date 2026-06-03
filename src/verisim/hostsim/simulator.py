"""The whole-machine simulator an agent calls (SPEC-6 §7) -- the SLM/LLM complementarity layer.

A computer-use or cyber-defense agent acts on a **whole host**, so the host world is exactly the
simulator it needs: not a competitor to the LLM, but the cheap, faithful, verifiable machine the LLM
reasons *over*. This module packages the loop's proposer (`M_θ`, a
:class:`~verisim.hostloop.model.HostModel`) + the reference oracle into the **protocol** §7
specifies -- a single object that both *predicts the next host state* (the loop interface, reused)
and *simulates a plan* (the agent interface, new):

  - :meth:`HostSimulator.imagine` -- roll ``M_θ`` forward over a plan (a sequence of syscalls the
    LLM proposes -- "open this, write that, kill that process") with **no oracle**: fast "plan in
    imagination" (Dreamer), the draft an agent explores cheaply.
  - :meth:`HostSimulator.verify` -- the same plan, but propose-the-model **and** consult the oracle,
    step by step, returning a :class:`PlanReport`: the predicted vs true final state, the per-step
    composed divergence, the **plan-level faithful horizon** (how many leading plan steps the agent
    can trust the model -- the §17.8 ``H_ε``-for-a-plan), the oracle cost paid, and -- composing the
    **task oracle** (§7, :mod:`.goal`) -- whether the plan achieves a goal and whether the model
    *agrees with the oracle* on that. This is propose-verify-correct lifted from the syscall level
    to the *plan* level, made honest by a real oracle.

The split is the spec's standing one: the model drafts host dynamics fast (speculative execution),
the oracle verifies on the agent's budget (`ρ` is the verification rate), and the LLM is escalated
to only for natural-language-intent → syscall-plan translation, never for simulating dynamics it is
bad at. Dependency-free except the model (the loop's ``HostModel``), like v0's ``eval`` package.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.host.action import HostAction, parse_host_action
from verisim.host.delta import apply
from verisim.host.state import HostState
from verisim.hostloop.model import HostModel
from verisim.hostloop.observe import full_bits
from verisim.hostmetrics.divergence import divergence, divergence_by_subsystem
from verisim.hostmetrics.record import HostRunRecord
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.metrics.horizon import faithful_horizon

from .goal import Goal

Plan = list[str]  # a sequence of syscall strings the agent proposes (e.g. "open 1 /log")


def _parse(plan: Plan) -> list[HostAction]:
    return [parse_host_action(step) for step in plan]


@dataclass(frozen=True)
class PlanRollout:
    """The model's imagined trajectory for a plan (no oracle) -- the cheap draft (§7)."""

    states: list[HostState]  # [s_0, ŝ_1, ..., ŝ_T], the predicted states after each plan step
    actions: list[HostAction]

    @property
    def final(self) -> HostState:
        return self.states[-1]


@dataclass(frozen=True)
class PlanReport:
    """The verified outcome of a plan -- predicted vs true, faithfulness, cost, task success."""

    n_steps: int
    epsilon: float
    predicted_final: HostState
    true_final: HostState
    divergences: list[float]  # composed predicted-vs-true divergence after each plan step
    plan_faithful_horizon: int  # leading plan steps the agent can trust the model (§17.8 H_ε/plan)
    final_divergence: float
    final_subsystem_divergence: dict[str, float]
    oracle_calls: int
    oracle_bits: int
    trusted: bool  # the model stayed faithful for the *whole* plan (faithful horizon == n_steps)
    goal_predicted: bool | None  # the task oracle on the model's predicted final state
    goal_true: bool | None  # the task oracle on the oracle's true final state
    goal_agreement: bool | None  # predicted == true on the goal -- task-level faithfulness

    def to_dict(self) -> dict[str, object]:
        return {
            "n_steps": self.n_steps, "epsilon": self.epsilon,
            "plan_faithful_horizon": self.plan_faithful_horizon,
            "final_divergence": self.final_divergence,
            "final_subsystem_divergence": self.final_subsystem_divergence,
            "oracle_calls": self.oracle_calls, "oracle_bits": self.oracle_bits,
            "trusted": self.trusted, "goal_predicted": self.goal_predicted,
            "goal_true": self.goal_true, "goal_agreement": self.goal_agreement,
        }


class HostSimulator:
    """The §7 protocol: predict-next-state (the loop) **and** simulate-a-plan (the agent)."""

    def __init__(self, model: HostModel, oracle: HostOracle | None = None) -> None:
        self.model = model
        self.oracle = oracle or ReferenceHostOracle()

    # -- the loop interface (predict next state) --------------------------------

    def predict_next(self, state: HostState, action: HostAction) -> HostState:
        """Apply the model's predicted bundle delta -- the loop's one-step prediction (§5, §8)."""
        return apply(state, self.model.predict_delta(state, action))

    # -- the agent interface (simulate a plan) ----------------------------------

    def imagine(self, state: HostState, plan: Plan) -> PlanRollout:
        """Roll ``M_θ`` forward over ``plan`` with no oracle -- the fast "plan in imagination"."""
        actions = _parse(plan)
        states = [state]
        current = state
        for action in actions:
            current = self.predict_next(current, action)
            states.append(current)
        return PlanRollout(states=states, actions=actions)

    def verify(
        self, state: HostState, plan: Plan, *, epsilon: float = 0.0, goal: Goal | None = None
    ) -> PlanReport:
        """Simulate ``plan`` and check it against the oracle step by step (§7, the honest version).

        Rolls the model's free-running imagination (the trajectory the agent would believe) against
        the oracle's true trajectory, recording the composed predicted-vs-true divergence per step.
        The plan-level faithful horizon is the first step the divergence exceeds ``epsilon`` (the
        §17.8 ``H_ε`` for a plan). If a ``goal`` is given, the task oracle is evaluated on both the
        predicted and true final state, and their agreement is the task-level faithfulness (§7).
        """
        actions = _parse(plan)
        predicted = state
        true = state
        divergences: list[float] = []
        oracle_bits = 0
        for action in actions:
            predicted = self.predict_next(predicted, action)  # model imagination (free-running)
            true = self.oracle.step(true, action).state  # the truth
            divergences.append(divergence(true, predicted))
            oracle_bits += full_bits(true)  # the cost of verifying this step in full
        n = len(actions)
        plan_h = faithful_horizon(divergences, epsilon)
        goal_pred = goal.holds(predicted) if goal is not None else None
        goal_true = goal.holds(true) if goal is not None else None
        return PlanReport(
            n_steps=n,
            epsilon=epsilon,
            predicted_final=predicted,
            true_final=true,
            divergences=divergences,
            plan_faithful_horizon=plan_h,
            final_divergence=divergences[-1] if divergences else 0.0,
            final_subsystem_divergence=divergence_by_subsystem(true, predicted),
            oracle_calls=n,
            oracle_bits=oracle_bits,
            trusted=plan_h == n,
            goal_predicted=goal_pred,
            goal_true=goal_true,
            goal_agreement=(goal_pred == goal_true) if goal is not None else None,
        )

    def run_record(self, state: HostState, plan: Plan, *, epsilon: float = 0.0) -> HostRunRecord:
        """The plan's verification as a :class:`HostRunRecord` (composed + per-subsystem tracks).

        Lets a plan be aggregated/plotted with the same figures-from-records machinery as the loop
        experiments (SPEC-2 §7.3) -- the plan is just a hand-authored action sequence.
        """
        actions = _parse(plan)
        predicted = state
        true = state
        divergences: list[float] = []
        sub: dict[str, list[float]] = {}
        for action in actions:
            predicted = self.predict_next(predicted, action)
            true = self.oracle.step(true, action).state
            divergences.append(divergence(true, predicted))
            for s, d in divergence_by_subsystem(true, predicted).items():
                sub.setdefault(s, []).append(d)
        return HostRunRecord(
            config={"experiment": "hostsim-plan", "n_steps": len(actions)},
            seed=0, epsilon=epsilon, divergences=divergences,
            subsystem_divergences=sub, consultation_schedule=[True] * len(actions),
        )
