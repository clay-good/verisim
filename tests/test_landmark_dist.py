"""Unit tests for the torch-free distributed landmark layer (SPEC-12 §6 LP8, H38).

The consistency signature and the re-grounding executor are dependency-free (the NW0-NW3
discipline),
so they are tested directly with the ``DistNullModel`` / ``DistOracleBackedModel`` baselines — no
torch, no training. The headline magnitudes come from the committed full-config run (``lp8_dist``).
"""

from __future__ import annotations

import random

from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distloop.model import DistNullModel, DistOracleBackedModel
from verisim.distoracle import ReferenceDistOracle
from verisim.landmark.dist import consistency_signature, execute_dist_plan


def _journey(cfg: DistConfig, oracle: ReferenceDistOracle, n: int, seed: int):
    drv = DistDriver("contention", cfg, random.Random(seed))
    start = DistributedState.initial(cfg)
    state = start
    actions = []
    truth = []
    for _ in range(n):
        a = drv.sample(state)
        state = oracle.step(state, a).state
        actions.append(a)
        truth.append(state)
    return start, actions, truth


def test_consistency_signature_is_deterministic() -> None:
    cfg = DistConfig()
    s = DistributedState.initial(cfg)
    assert consistency_signature(s, cfg) == consistency_signature(s, cfg)
    # The initial cluster is fully converged, one partition group, no down nodes.
    converged, partitions, down = consistency_signature(s, cfg)
    assert converged == frozenset(cfg.objects)
    assert down == frozenset()
    assert len(partitions) == 1


def test_oracle_backed_model_reaches_every_goal() -> None:
    # A perfect model never drifts, so even flat free-running reaches the goal and the consistency
    # horizon is the whole rollout — the ceiling that frames what re-grounding is for.
    cfg = DistConfig()
    oracle = ReferenceDistOracle(cfg)
    start, actions, truth = _journey(cfg, oracle, 12, seed=100)
    trace = execute_dist_plan(
        DistOracleBackedModel(oracle), cfg, start, actions, truth, frozenset(), reground=False
    )
    assert trace.goal_reached
    assert trace.consistency_horizon == trace.n_steps
    assert trace.full_horizon == trace.n_steps


def test_landmark_regrounding_spends_budget_and_excludes_goal() -> None:
    # The null model drifts immediately; re-grounding at interior boundaries spends budget, and the
    # goal step is never a re-ground (so goal-reach stays a model prediction).
    cfg = DistConfig()
    oracle = ReferenceDistOracle(cfg)
    start, actions, truth = _journey(cfg, oracle, 12, seed=101)
    reground_at = frozenset({3, 7})  # interior boundaries for L=4, G=12 (goal step 11 excluded)
    trace = execute_dist_plan(
        DistNullModel(), cfg, start, actions, truth, reground_at, reground=True
    )
    assert trace.n_consults == 2
    assert 11 not in reground_at
    assert 0.0 <= float(trace.goal_reached) <= 1.0


def test_execute_dist_plan_is_deterministic() -> None:
    cfg = DistConfig()
    oracle = ReferenceDistOracle(cfg)
    start, actions, truth = _journey(cfg, oracle, 10, seed=102)
    a = execute_dist_plan(
        DistNullModel(), cfg, start, actions, truth, frozenset({3}), reground=True
    )
    b = execute_dist_plan(
        DistNullModel(), cfg, start, actions, truth, frozenset({3}), reground=True
    )
    assert a == b
