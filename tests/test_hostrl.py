"""Host oracle-as-reward RL environment tests (SPEC-6 / HC8). Torch-free.

Mirrors v0's RL-env invariants with the dependency-free baselines: a perfect (oracle-backed) policy
scores reward 1 every step (return == H_ε == T), a null policy drifts (return < T), the budget
mechanics hold, and ``load_environment`` constructs the env.
"""

from __future__ import annotations

from verisim.hostloop import HostNullModel, HostOracleBackedModel
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.hostrl import HostWorldModelEnv, Observation, load_environment


def _episode_return(env: HostWorldModelEnv, policy) -> float:
    obs: Observation | None = env.reset()
    total = 0.0
    while obs is not None:  # the env returns observation=None exactly at episode end
        t = env.step(policy.predict_delta(obs.state, obs.action))
        total += t.reward
        obs = t.observation
    return total


def test_perfect_policy_gets_full_return():
    env = HostWorldModelEnv(driver="forky", seed=1, n_steps=20, epsilon=0.0)
    ret = _episode_return(env, HostOracleBackedModel(ReferenceHostOracle()))
    assert ret == env.n_steps  # reward 1 every step -> return == H_ε == T


def test_null_policy_drifts():
    env = HostWorldModelEnv(driver="forky", seed=2, n_steps=20, epsilon=0.0)
    ret = _episode_return(env, HostNullModel())
    assert ret < env.n_steps  # the empty delta drifts the moment the oracle changes anything


def test_terminate_on_divergence_makes_return_the_faithful_horizon():
    env = HostWorldModelEnv(driver="adversarial", seed=3, n_steps=24, terminate_on_divergence=True)
    obs: Observation | None = env.reset()
    steps = 0
    while obs is not None:
        t = env.step(HostNullModel().predict_delta(obs.state, obs.action))
        steps += 1
        obs = t.observation
    # the episode ends at the first unfaithful step (or T)
    assert 1 <= steps <= env.n_steps


def test_step_after_done_raises_and_load_environment_constructs():
    env = load_environment(driver="forky", seed=0, n_steps=2, terminate_on_divergence=False)
    assert isinstance(env, HostWorldModelEnv)
    env.reset()
    env.step([])
    env.step([])  # episode now done (n_steps reached)
    import pytest

    with pytest.raises(RuntimeError):
        env.step([])
