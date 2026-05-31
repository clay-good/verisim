"""Tests for the K3 difficulty dial and the K4 knee experiment (SPEC-2.1 §7-8)."""

import random

from verisim.data.drivers import Driver, path_depth
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.experiments.k4 import K4Config, run_k4, run_k4_policies
from verisim.oracle.reference import ReferenceOracle


def test_path_depth():
    assert path_depth("/") == 0
    assert path_depth("/a") == 1
    assert path_depth("/a/b/c") == 3


def test_max_depth_caps_structural_creates():
    oracle = ReferenceOracle()
    driver = Driver("structural", DEFAULT_CONFIG, random.Random(0), max_depth=2)
    state = State.empty()
    for _ in range(40):
        action = driver.sample(state)
        # every created structural path stays within the depth cap.
        assert path_depth(action.args[0]) <= 2
        state = oracle.step(state, action).state


def test_structural_uncapped_goes_deep():
    oracle = ReferenceOracle()
    driver = Driver("structural", DEFAULT_CONFIG, random.Random(0))  # no cap
    state = State.empty()
    max_seen = 0
    for _ in range(60):
        action = driver.sample(state)
        max_seen = max(max_seen, path_depth(action.args[0]))
        state = oracle.step(state, action).state
    assert max_seen >= 3  # nests deeper than the capped run


def test_run_k4_tiny_emits_curve_records():
    config = K4Config(
        train_seeds=(0, 1, 2, 3),
        val_seeds=(50,),
        train_steps_per_traj=6,
        max_depth=2,
        n_layer=1,
        n_embd=32,
        train_steps=20,
        batch_size=8,
        eval_interval=10,
        eval_seeds=(300, 301),
        eval_steps=8,
        rhos=(0.0, 0.5, 1.0),
        epsilons=(0.0, 0.05),
    )
    records = run_k4(config)
    # one record per (eval_seed, rho, epsilon).
    assert len(records) == 2 * 3 * 2
    rhos = {float(r.config["rho"]) for r in records}
    assert rhos == {0.0, 0.5, 1.0}
    # ρ=1 (consult every step) is fully faithful: H_ε == T.
    for r in records:
        if float(r.config["rho"]) == 1.0:
            assert r.faithful_horizon == config.eval_steps
        # consultation budget is never exceeded.
        assert r.oracle_calls <= config.eval_steps


def test_run_k4_policies_tiny():
    config = K4Config(
        train_seeds=(0, 1, 2, 3),
        val_seeds=(50,),
        train_steps_per_traj=6,
        max_depth=2,
        n_layer=1,
        n_embd=32,
        train_steps=20,
        batch_size=8,
        eval_interval=10,
        eval_seeds=(300, 301),
        eval_steps=8,
        epsilons=(0.0, 0.05),
    )
    records = run_k4_policies(config, rho=0.25, policies=("fixed", "uncertainty"))
    policies = {r.config["policy"] for r in records}
    assert policies == {"fixed", "uncertainty"}
    # equal-budget by construction: both policies spend the same number of oracle calls.
    budget = int(0.25 * config.eval_steps)
    for r in records:
        assert r.oracle_calls == budget
