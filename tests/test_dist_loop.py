"""DS5 — the tiered propose-verify-correct loop (SPEC-7 §8).

Loop-invariant tests for the dependency-free, GPU-free runner (mirroring v0/net/host §8 tests),
plus the distributed-world-specific oracle-dollar accounting (the §9.4 / H17 cost):

  - **ρ=1 full consultation reproduces truth** (``H_ε = T``) even for the null model;
  - the **perfect (oracle-backed) model never drifts** at ρ=0 and spends nothing;
  - the **null model drifts immediately** at ρ=0 (the task is nontrivial);
  - the **budget is spent exactly** (true equal-ρ comparison, the spend-down backstop);
  - the **oracle-dollar** is tracked, and reflects the tier policy: escalation pays the cheap
    tiers before bit-exact when the model's errors are only caught late — the genuine H17 nuance.
"""

from __future__ import annotations

import random

from verisim.dist import DEFAULT_DIST_CONFIG, DistributedState
from verisim.distdata import DistDriver
from verisim.distloop import (
    DistNullModel,
    DistOracleBackedModel,
    EscalatingTierPolicy,
    FixedTierPolicy,
    budget_for_rho,
    run_dist_rollout,
)
from verisim.distoracle import ReferenceDistOracle
from verisim.loop.policy import fixed_interval_for_rho

CFG = DEFAULT_DIST_CONFIG
ORACLE = ReferenceDistOracle(CFG)
S0 = DistributedState.initial(CFG)


def _actions(n: int = 24, seed: int = 4) -> list:
    driver = DistDriver("adversarial", CFG, random.Random(seed))
    state = S0
    out = []
    for _ in range(n):
        action = driver.sample(state)
        out.append(action)
        state = ORACLE.step(state, action).state
    return out


def test_full_consultation_reproduces_truth():
    actions = _actions()
    record = run_dist_rollout(
        DistNullModel(), ORACLE, S0, actions, fixed_interval_for_rho(1.0),
        epsilon=0.0, budget=budget_for_rho(1.0, len(actions)),
    )
    assert record.faithful_horizon == len(actions)  # H_ε = T: every step snapped to truth
    assert record.oracle_calls == len(actions)


def test_perfect_model_never_drifts_at_zero_budget():
    actions = _actions()
    record = run_dist_rollout(
        DistOracleBackedModel(ORACLE), ORACLE, S0, actions, fixed_interval_for_rho(0.0),
        epsilon=0.0, budget=budget_for_rho(0.0, len(actions)),
    )
    assert record.faithful_horizon == len(actions)  # ceiling: the model alone is exact
    assert record.oracle_calls == 0
    assert record.config["oracle_dollars"] == 0  # spent nothing


def test_null_model_drifts_immediately_at_zero_budget():
    actions = _actions()
    record = run_dist_rollout(
        DistNullModel(), ORACLE, S0, actions, fixed_interval_for_rho(0.0),
        epsilon=0.0, budget=budget_for_rho(0.0, len(actions)),
    )
    assert record.faithful_horizon == 0  # the floor: drifts on the first unaided step


def test_budget_is_spent_exactly():
    actions = _actions(n=20)
    for rho in (0.0, 0.25, 0.5, 1.0):
        record = run_dist_rollout(
            DistNullModel(), ORACLE, S0, actions, fixed_interval_for_rho(rho),
            epsilon=0.0, budget=budget_for_rho(rho, len(actions)),
        )
        assert record.oracle_calls == budget_for_rho(rho, len(actions))


def test_oracle_dollar_tracks_tier_policy():
    actions = _actions()
    n = len(actions)
    bit_exact = run_dist_rollout(
        DistNullModel(), ORACLE, S0, actions, fixed_interval_for_rho(1.0),
        epsilon=0.0, budget=budget_for_rho(1.0, n), tier_policy=FixedTierPolicy("bit_exact"),
    )
    escalate = run_dist_rollout(
        DistNullModel(), ORACLE, S0, actions, fixed_interval_for_rho(1.0),
        epsilon=0.0, budget=budget_for_rho(1.0, n), tier_policy=EscalatingTierPolicy(),
    )
    # both reach the same horizon (both correct every wrong step), but spend differently
    assert bit_exact.faithful_horizon == escalate.faithful_horizon == n
    # always-bit_exact costs 16 per consult
    assert bit_exact.config["oracle_dollars"] == 16 * n
    # the null model's errors are write-legality violations (symbolic-level), so escalation pays the
    # cheap tiers *before* bit-exact for the correction -> strictly more here (the H17 nuance:
    # cheap tiers pay off only when errors are gross / caught early)
    assert escalate.config["oracle_dollars"] > bit_exact.config["oracle_dollars"]


def test_intermediate_budget_is_between_floor_and_ceiling():
    actions = _actions()
    n = len(actions)
    floor = run_dist_rollout(DistNullModel(), ORACLE, S0, actions, fixed_interval_for_rho(0.0),
                             epsilon=0.0, budget=budget_for_rho(0.0, n)).faithful_horizon
    mid = run_dist_rollout(DistNullModel(), ORACLE, S0, actions, fixed_interval_for_rho(0.5),
                           epsilon=0.0, budget=budget_for_rho(0.5, n)).faithful_horizon
    ceil = run_dist_rollout(DistNullModel(), ORACLE, S0, actions, fixed_interval_for_rho(1.0),
                            epsilon=0.0, budget=budget_for_rho(1.0, n)).faithful_horizon
    assert floor <= mid <= ceil
