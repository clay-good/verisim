"""Distributed oracle-as-reward RL environment tests (SPEC-7 §12 / DS8). Torch-free.

Mirrors the host RL-env invariants with the dependency-free baselines: a perfect (oracle-backed)
policy scores reward 1 every step (return == H_ε == T) under both reward modes, a null policy drifts
(return < T), the episode mechanics hold, and ``load_environment`` constructs the env. The one
distributed-specific check: the ``consistency`` reward mode is a well-formed alternate horizon (the
§9.1 split-brain decision), and the env rejects an unknown reward mode.
"""

from __future__ import annotations

import pytest

from verisim.distloop import DistNullModel, DistOracleBackedModel
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distrl import DistWorldModelEnv, Observation, load_environment


def _episode_return(env: DistWorldModelEnv, policy: object) -> float:
    obs: Observation | None = env.reset()
    total = 0.0
    while obs is not None:  # the env returns observation=None exactly at episode end
        t = env.step(policy.predict_delta(obs.state, obs.action))  # type: ignore[attr-defined]
        total += t.reward
        obs = t.observation
    return total


def test_perfect_policy_gets_full_return():
    env = DistWorldModelEnv(driver="adversarial", seed=1, n_steps=20, epsilon=0.0)
    ret = _episode_return(env, DistOracleBackedModel(ReferenceDistOracle(env.config)))
    assert ret == env.n_steps  # reward 1 every step -> return == H_ε == T


def test_perfect_policy_full_return_under_consistency_mode():
    env = DistWorldModelEnv(
        driver="adversarial", seed=1, n_steps=20, reward_mode="consistency"
    )
    ret = _episode_return(env, DistOracleBackedModel(ReferenceDistOracle(env.config)))
    assert ret == env.n_steps  # a perfect model is consistency-faithful too


def test_null_policy_drifts():
    env = DistWorldModelEnv(driver="adversarial", seed=2, n_steps=20, epsilon=0.0)
    ret = _episode_return(env, DistNullModel())
    assert ret < env.n_steps  # the empty delta drifts the moment the oracle changes anything


def test_terminate_on_divergence_makes_return_the_faithful_horizon():
    env = DistWorldModelEnv(driver="adversarial", seed=3, n_steps=24, terminate_on_divergence=True)
    obs: Observation | None = env.reset()
    steps = 0
    while obs is not None:
        t = env.step(DistNullModel().predict_delta(obs.state, obs.action))
        steps += 1
        obs = t.observation
    assert 1 <= steps <= env.n_steps  # ends at the first unfaithful step (or T)


def test_info_carries_both_faithfulness_signals():
    env = DistWorldModelEnv(driver="uniform", seed=4, n_steps=8, terminate_on_divergence=False)
    obs = env.reset()
    t = env.step(DistNullModel().predict_delta(obs.state, obs.action))
    assert "divergence" in t.info
    assert "consistency_faithfulness" in t.info
    assert 0.0 <= t.info["consistency_faithfulness"] <= 1.0


def test_unknown_reward_mode_rejected():
    with pytest.raises(ValueError):
        DistWorldModelEnv(reward_mode="nonsense")


def test_step_after_done_raises_and_load_environment_constructs():
    env = load_environment(driver="uniform", seed=0, n_steps=2, terminate_on_divergence=False)
    assert isinstance(env, DistWorldModelEnv)
    env.reset()
    env.step([])
    env.step([])  # episode now done (n_steps reached)
    with pytest.raises(RuntimeError):
        env.step([])
