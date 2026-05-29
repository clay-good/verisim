"""The propose-verify-correct rollout runner (SPEC-2 §6.3; SPEC.md §5.2).

A single runner takes ``(model, oracle, π_c, C, ρ, ε, actions)`` and produces a
:class:`RunRecord`: the per-step divergence trajectory, the consultation schedule
actually used, and (via the record) ``H_ε``. Everything in the experiment sweeps
(SPEC-2 §9) is this runner under different settings.

Two rollouts run in lockstep:

  - the **ground-truth rollout** -- the oracle applied autoregressively from
    ``s_0`` with no model -- gives the true state at every step;
  - the **coupled rollout** -- the §5.2 loop: the model proposes a delta, and at
    steps the policy selects (subject to the budget) the oracle is consulted and
    the correction operator is applied.

Divergence at step ``t`` is ``d`` between the coupled state and the ground-truth
state *after* action ``t`` (SPEC.md §5.1), recorded every step -- that trajectory
defines ``H_ε``. The one-step truth used for *correction* is ``O(coupled_state,
a_t)`` per §5.2 (the oracle's transition from the current, possibly drifted,
coupled state), which is distinct from the ground-truth-rollout state once drift
has occurred.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from verisim.delta.apply import apply
from verisim.env.action import Action
from verisim.env.state import State
from verisim.metrics.divergence import divergence
from verisim.metrics.record import RunRecord
from verisim.oracle.base import Oracle

from .model import Model
from .operator import CorrectionOperator, HardReset
from .policy import ConsultationPolicy


def ground_truth_rollout(oracle: Oracle, s0: State, actions: Sequence[Action]) -> list[State]:
    """The true state after each prefix of ``actions``; ``[s_0, s_1, ..., s_T]``."""
    states = [s0]
    state = s0
    for action in actions:
        state = oracle.step(state, action).state
        states.append(state)
    return states


def budget_for_rho(rho: float, n_steps: int) -> int:
    """Max permitted oracle consultations for budget ``ρ`` over ``n_steps`` steps."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    return math.floor(rho * n_steps)


def run_rollout(
    model: Model,
    oracle: Oracle,
    s0: State,
    actions: Sequence[Action],
    policy: ConsultationPolicy,
    *,
    epsilon: float,
    operator: CorrectionOperator | None = None,
    budget: int | None = None,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> RunRecord:
    """Run one propose-verify-correct rollout and return its :class:`RunRecord`.

    ``budget`` caps total consultations (``None`` = unlimited); the cap is honored
    even when the policy proposes more, so the consultation count never exceeds it.
    """
    correct = (operator or HardReset()).correct
    truth_states = ground_truth_rollout(oracle, s0, actions)

    state = s0
    divergences: list[float] = []
    schedule: list[bool] = []
    calls = 0

    for t, action in enumerate(actions):
        predicted = apply(state, model.predict_delta(state, action))  # PROPOSE
        consult = policy.should_consult(t) and (budget is None or calls < budget)
        if consult:
            truth = oracle.step(state, action).state  # VERIFY (from current state)
            calls += 1
            state = correct(predicted, truth)  # CORRECT
        else:
            state = predicted
        schedule.append(consult)
        divergences.append(divergence(truth_states[t + 1], state))

    record_config = dict(config or {})
    record_config.setdefault("budget", budget)
    return RunRecord(
        config=record_config,
        seed=seed,
        epsilon=epsilon,
        divergences=divergences,
        consultation_schedule=schedule,
    )
