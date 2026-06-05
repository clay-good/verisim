"""The tiered propose-verify-correct rollout runner (SPEC-7 §8; DS5).

The distributed analogue of v0's :mod:`verisim.loop.runner`, the network's
:mod:`verisim.netloop.runner`, and the host's :mod:`verisim.hostloop.runner` -- with the one
structural fork that makes the distributed world the thesis's proving ground: **a consultation does
not buy full truth for free.** Two rollouts run in lockstep:

  - the **ground-truth rollout** -- the Tier-A oracle applied autoregressively from ``s_0``
    -- gives the true state at every step;
  - the **coupled rollout** -- the §8 loop: the model proposes a ``DistDelta``, and at the steps the
    consultation policy ``π_c`` selects (subject to budget ``ρ``) the **tiered** oracle is consulted
    at the tier ``π_w`` chooses, and the coupled state is corrected only if that tier *refutes* the
    prediction.

The new axis is the **oracle-dollar** (§5, §9.4): each consult spends the cost of the tier(s) it
runs; a refutation additionally pays the bit-exact cost to *recompute the truth needed to correct*
(unless the refuting tier was already bit-exact). A prediction the chosen tier cannot refute is
*trusted* -- no correction, only the cheap tier's cost. The record carries the per-step divergence
trajectory (which defines ``H_ε``) **and** the cumulative oracle-dollars, so the ED1 curve (DS6) can
plot faithful horizon against *oracle-dollar* spent, not just consultation count -- the quantity H17
asks whether cheap tiers minimize.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import apply
from verisim.dist.state import DistributedState
from verisim.distmetrics.divergence import divergence
from verisim.distoracle.base import DistOracle
from verisim.distoracle.tiers import TIER_COSTS, TieredOracle
from verisim.loop.policy import ConsultationPolicy, StepContext
from verisim.loop.runner import budget_for_rho
from verisim.metrics.record import RunRecord

from .model import DistModel
from .tier_policy import FixedTierPolicy, TierPolicy

__all__ = ["budget_for_rho", "ground_truth_rollout", "run_dist_rollout"]


def ground_truth_rollout(
    oracle: DistOracle, s0: DistributedState, actions: Sequence[DistAction]
) -> list[DistributedState]:
    """The true state after each prefix of ``actions``; ``[s_0, s_1, ..., s_T]``."""
    states = [s0]
    state = s0
    for action in actions:
        state = oracle.step(state, action).state
        states.append(state)
    return states


def run_dist_rollout(
    model: DistModel,
    oracle: DistOracle,
    s0: DistributedState,
    actions: Sequence[DistAction],
    policy: ConsultationPolicy,
    *,
    epsilon: float,
    config: DistConfig = DEFAULT_DIST_CONFIG,
    tier_policy: TierPolicy | None = None,
    budget: int | None = None,
    seed: int = 0,
    record_config: dict[str, Any] | None = None,
) -> RunRecord:
    """Run one tiered propose-verify-correct rollout; return its run-record.

    ``budget`` caps total consultations (``None`` = unlimited); the spend-down backstop forces a
    consult once the remaining budget would otherwise be stranded, so every policy spends *exactly*
    ``budget`` (true equal-``ρ`` comparison, SPEC-2 §16). ``tier_policy`` (``π_w``) chooses which
    tier each consult spends -- default full-truth (``bit_exact``), comparable to every prior world.
    The cumulative ``oracle_dollars`` is recorded on the run-record config (the §9.4 cost
    H17 measures against faithful horizon).
    """
    tier_policy = tier_policy or FixedTierPolicy("bit_exact")
    tiered = TieredOracle(config)
    truth_states = ground_truth_rollout(oracle, s0, actions)
    n_steps = len(actions)

    state = s0
    divergences: list[float] = []
    schedule: list[bool] = []
    calls = 0
    oracle_dollars = 0

    for t, action in enumerate(actions):
        predicted = apply(state, model.predict_delta(state, action))  # PROPOSE

        has_budget = budget is None or calls < budget
        must_spend = budget is not None and (budget - calls) >= (n_steps - t)
        consult = has_budget and (must_spend or policy.should_consult(StepContext(step=t)))

        if consult:
            calls += 1
            if tier_policy.escalate:
                verdict = tiered.cheapest_refutation(state, action, predicted)  # VERIFY (cheapest)
            else:
                tier = tier_policy.tier(t)
                verdict = tiered.check(tier, state, action, predicted)  # VERIFY (chosen tier)
            oracle_dollars += verdict.cost
            if verdict.refuted:
                if verdict.tier != "bit_exact":
                    oracle_dollars += TIER_COSTS["bit_exact"]  # recompute truth to CORRECT
                state = oracle.step(state, action).state  # CORRECT: snap to truth
            else:
                state = predicted  # trust the model (the tier could not refute it)
        else:
            state = predicted  # trust the model

        schedule.append(consult)
        divergences.append(divergence(state, truth_states[t + 1]))

    cfg = dict(record_config or {})
    cfg.setdefault("world", "distributed")
    cfg["oracle_dollars"] = oracle_dollars
    cfg["tier_policy"] = "escalate" if tier_policy.escalate else tier_policy.tier(0)
    return RunRecord(
        config=cfg, seed=seed, epsilon=epsilon, divergences=divergences,
        consultation_schedule=schedule,
    )
