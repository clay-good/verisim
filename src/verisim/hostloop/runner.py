"""The composed propose-verify-correct rollout runner (SPEC-6 §8, HC5).

The host analogue of v0's :mod:`verisim.loop.runner` and the network :mod:`verisim.netloop.runner`,
with the §8 composed/per-subsystem extensions. Two rollouts run in lockstep, exactly as in the other
worlds:

  - the **ground-truth rollout** -- the oracle applied autoregressively from ``s_0`` -- gives the
    true bundle state at every step;
  - the **coupled rollout** -- the §8 loop: the model proposes a bundle delta, and at the steps the
    consultation policy ``π_c`` selects (subject to budget ``ρ``) the oracle is consulted and the
    coupled state is corrected.

The host-specific fork is the consultation *mode* (SPEC-6 §5.3, §8.2):

  - **full** (``subsystem_policy is None``): consult the complete one-step truth and apply a
    full-correction operator (``HardReset`` by default) -- identical in shape to v0/NW5, and the
    mode the EH1 composed ``H_ε(ρ)`` headline uses so it is directly comparable.
  - **per-subsystem** (``subsystem_policy`` given): the subsystem policy ``π_w`` (§8.2) picks one
    subsystem, the cheap probe reveals only that subsystem's truth, and the subsystem filter snaps
    just that subsystem -- a per-subsystem consult corrects strictly less than a full consult, so
    faithful horizon is no greater. This is the machinery EH3 / the composition law (H13) measure.

Every step records the **composed** divergence *and* the **per-subsystem** divergences (SPEC-6 §9.1)
between the coupled state and the ground-truth state after action ``t`` -- those trajectories define
the composed ``H_ε`` and the component ``H_ε^i`` the composition-law diagnostic consumes. The result
is a :class:`~verisim.hostmetrics.record.HostRunRecord` (HC3's schema, now populated). Budget
handling (hard cap + spend-down backstop, true equal-``ρ``) is the v0 logic verbatim, reused from
:func:`verisim.loop.runner.budget_for_rho`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verisim.host.action import HostAction
from verisim.host.delta import apply
from verisim.host.state import HostState
from verisim.hostmetrics.divergence import SUBSYSTEMS, divergence, divergence_by_subsystem
from verisim.hostmetrics.record import HostRunRecord
from verisim.loop.policy import ConsultationPolicy, StepContext
from verisim.loop.runner import budget_for_rho

from .model import HostModel, HostUncertaintyModel
from .observe import PartialHostOracle, full_bits
from .operator import FullCorrection, HardReset, SubsystemFilter
from .subsystem import SubsystemPolicy

__all__ = ["budget_for_rho", "ground_truth_rollout", "run_host_rollout"]


def ground_truth_rollout(
    oracle: PartialHostOracle, s0: HostState, actions: Sequence[HostAction]
) -> list[HostState]:
    """The true bundle state after each prefix of ``actions``; ``[s_0, s_1, ..., s_T]``."""
    states = [s0]
    state = s0
    for action in actions:
        state = oracle.full(state, action).state
        states.append(state)
    return states


def _predict(
    model: HostModel, state: HostState, action: HostAction
) -> tuple[Any, float]:
    """Predict the bundle delta and the model's uncertainty (``0`` if it exposes none)."""
    if isinstance(model, HostUncertaintyModel):
        return model.predict_delta_with_uncertainty(state, action)
    return model.predict_delta(state, action), 0.0


def run_host_rollout(
    model: HostModel,
    oracle: PartialHostOracle,
    s0: HostState,
    actions: Sequence[HostAction],
    policy: ConsultationPolicy,
    *,
    epsilon: float,
    operator: FullCorrection | None = None,
    subsystem_policy: SubsystemPolicy | None = None,
    subsystem_op: SubsystemFilter | None = None,
    budget: int | None = None,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> HostRunRecord:
    """Run one composed propose-verify-correct rollout; return its host run-record.

    ``budget`` caps total consultations (``None`` = unlimited); the spend-down backstop forces a
    consult once the remaining budget would otherwise be stranded, so every policy spends *exactly*
    ``budget`` (true equal-``ρ`` comparison, SPEC-6 §9 / SPEC-2 §16). ``subsystem_policy`` selects
    per-subsystem mode; otherwise it is full-consultation mode. ``oracle_bits`` (the cost
    denominator of per-subsystem efficiency, §9.4) is recorded on the run-record config.
    """
    correct = (operator or HardReset()).correct
    sop = subsystem_op or SubsystemFilter()
    truth_states = ground_truth_rollout(oracle, s0, actions)
    n_steps = len(actions)

    state = s0
    divergences: list[float] = []
    subsystem_divergences: dict[str, list[float]] = {sub: [] for sub in SUBSYSTEMS}
    schedule: list[bool] = []
    calls = 0
    oracle_bits = 0
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
            calls += 1
            cumulative_signal = 0.0
            if subsystem_policy is None:  # full consultation (the EH1 headline mode)
                truth = oracle.full(state, action).state  # VERIFY
                state = correct(predicted, truth)  # CORRECT
                oracle_bits += full_bits(truth)
            else:  # cheap probe of one subsystem (§5.3, §8.2)
                subsystem = subsystem_policy.select(state)  # WHICH to verify (§8.2)
                obs = oracle.probe(state, action, subsystem)  # VERIFY (partial)
                state = sop.correct(predicted, obs)  # subsystem-filter the observed subsystem
                oracle_bits += obs.bits
        else:
            state = predicted
        schedule.append(consult)

        truth_next = truth_states[t + 1]
        divergences.append(divergence(truth_next, state))
        for sub, d in divergence_by_subsystem(truth_next, state).items():
            subsystem_divergences[sub].append(d)

    record_config = dict(config or {})
    record_config.setdefault("budget", budget)
    record_config["oracle_bits"] = oracle_bits
    return HostRunRecord(
        config=record_config,
        seed=seed,
        epsilon=epsilon,
        divergences=divergences,
        subsystem_divergences=subsystem_divergences,
        consultation_schedule=schedule,
    )
