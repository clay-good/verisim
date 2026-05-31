"""The partial-observation propose-verify-correct rollout runner (SPEC-5 §8).

The network analogue of v0's :mod:`verisim.loop.runner`, with the §8 partial-observation
extensions. Two rollouts run in lockstep, exactly as in v0:

  - the **ground-truth rollout** -- the oracle applied autoregressively from ``s_0`` with no
    model -- gives the true state at every step;
  - the **coupled rollout** -- the §8 loop: the model proposes a graph delta, and at the
    steps the consultation policy ``π_c`` selects (subject to budget ``ρ``) the oracle is
    consulted and the coupled state is corrected.

The one network-specific fork is the consultation *mode* (SPEC-5 §5.3):

  - **full** (``probe_policy is None``): consult the complete one-step truth and apply a
    full-correction operator (``HardReset`` by default) -- identical in shape to v0, and the
    mode the EN1 ``H_ε(ρ)`` headline uses so it is directly comparable to v0's E1.
  - **probe** (``probe_policy`` given): the probe policy ``π_o`` (§8.2) picks one host, the
    cheap probe reveals only that host's subgraph, and the belief filter snaps just that
    subgraph -- a probe corrects strictly less than a full consult, so faithful horizon is
    no greater. This is the machinery EN2/EN3 (NW7) measure.

Divergence at step ``t`` is the graph divergence (SPEC-5 §9.1) between the coupled state and
the ground-truth state *after* action ``t``, recorded every step -- that trajectory defines
``H_ε``. Budget handling (hard cap + spend-down backstop, true equal-``ρ`` comparison) is the
v0 logic verbatim, reused from :func:`verisim.loop.runner.budget_for_rho`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verisim.loop.policy import ConsultationPolicy, StepContext
from verisim.loop.runner import budget_for_rho
from verisim.metrics.record import RunRecord
from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netdelta.apply import apply
from verisim.netmetrics.divergence import divergence

from .model import NetModel, NetUncertaintyModel
from .observe import PartialNetOracle, full_bits
from .operator import BeliefFilter, FullCorrection, HardReset
from .probe import ProbePolicy

__all__ = ["budget_for_rho", "ground_truth_rollout", "run_net_rollout"]


def ground_truth_rollout(
    oracle: PartialNetOracle, s0: NetworkState, actions: Sequence[NetAction]
) -> list[NetworkState]:
    """The true state after each prefix of ``actions``; ``[s_0, s_1, ..., s_T]``."""
    states = [s0]
    state = s0
    for action in actions:
        state = oracle.full(state, action).state
        states.append(state)
    return states


def _predict(model: NetModel, state: NetworkState, action: NetAction) -> tuple[Any, float]:
    """Predict the delta and the model's uncertainty (``0`` if it exposes none)."""
    if isinstance(model, NetUncertaintyModel):
        return model.predict_delta_with_uncertainty(state, action)
    return model.predict_delta(state, action), 0.0


def run_net_rollout(
    model: NetModel,
    oracle: PartialNetOracle,
    s0: NetworkState,
    actions: Sequence[NetAction],
    policy: ConsultationPolicy,
    *,
    epsilon: float,
    operator: FullCorrection | None = None,
    probe_policy: ProbePolicy | None = None,
    belief_op: BeliefFilter | None = None,
    budget: int | None = None,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> RunRecord:
    """Run one partial-observation propose-verify-correct rollout; return its record.

    ``budget`` caps total consultations (``None`` = unlimited); the spend-down backstop
    forces a consult once the remaining budget would otherwise be stranded, so every policy
    spends *exactly* ``budget`` (true equal-``ρ`` comparison, SPEC-5 §12). ``probe_policy``
    selects probe mode; otherwise it is full-consultation mode. ``oracle_bits`` (the cost
    denominator of probe efficiency, §9.4) is recorded on the run-record config.
    """
    correct = (operator or HardReset()).correct
    bop = belief_op or BeliefFilter()
    truth_states = ground_truth_rollout(oracle, s0, actions)
    n_steps = len(actions)

    state = s0
    divergences: list[float] = []
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
            if probe_policy is None:  # full consultation (the EN1 headline mode)
                truth = oracle.full(state, action).state  # VERIFY
                state = correct(predicted, truth)  # CORRECT
                oracle_bits += full_bits(truth)
            else:  # cheap probe of one host (§5.3, §8.2)
                host = probe_policy.select(state)  # WHAT to observe (§8.2)
                obs = oracle.probe(state, action, host)  # VERIFY (partial)
                state = bop.correct(predicted, obs)  # belief-filter the observed subgraph
                oracle_bits += obs.bits
        else:
            state = predicted
        schedule.append(consult)
        divergences.append(divergence(truth_states[t + 1], state))

    record_config = dict(config or {})
    record_config.setdefault("budget", budget)
    record_config["oracle_bits"] = oracle_bits
    return RunRecord(
        config=record_config,
        seed=seed,
        epsilon=epsilon,
        divergences=divergences,
        consultation_schedule=schedule,
    )
