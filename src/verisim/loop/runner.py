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

from .model import Model, UncertaintyModel
from .operator import CorrectionOperator, HardReset
from .policy import ConsultationPolicy, StepContext


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


def _predict(model: Model, state: State, action: Action) -> tuple[Any, float]:
    """Predict the delta and the model's uncertainty (``0`` if it exposes none)."""
    if isinstance(model, UncertaintyModel):
        return model.predict_delta_with_uncertainty(state, action)
    return model.predict_delta(state, action), 0.0


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

    ``budget`` caps total consultations (``None`` = unlimited). Two budget controls
    are layered on the policy's proposals so every policy spends *exactly* ``budget``
    (true equal-``ρ`` comparison, SPEC-2 §16): the consultation count never exceeds
    the cap, and a spend-down backstop forces a consult once the remaining budget
    would otherwise be stranded (remaining budget ≥ remaining steps). For the
    ``fixed`` policy, which already spreads its calls across the rollout, the
    backstop never fires; it only matters for the triggered policies that may
    under-spend, ensuring they are compared at the same call count.
    """
    correct = (operator or HardReset()).correct
    truth_states = ground_truth_rollout(oracle, s0, actions)
    n_steps = len(actions)

    state = s0
    divergences: list[float] = []
    schedule: list[bool] = []
    calls = 0
    cumulative_signal = 0.0

    for t, action in enumerate(actions):
        delta, signal = _predict(model, state, action)  # PROPOSE (+ uncertainty)
        predicted = apply(state, delta)
        cumulative_signal += signal

        has_budget = budget is None or calls < budget
        must_spend = budget is not None and (budget - calls) >= (n_steps - t)
        ctx = StepContext(step=t, signal=signal, cumulative_signal=cumulative_signal)
        consult = has_budget and (must_spend or policy.should_consult(ctx))

        if consult:
            truth = oracle.step(state, action).state  # VERIFY (from current state)
            calls += 1
            state = correct(predicted, truth)  # CORRECT
            cumulative_signal = 0.0
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
