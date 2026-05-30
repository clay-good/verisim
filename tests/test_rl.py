"""Oracle-as-reward RL environment tests (SPEC-2 §15, milestone M8).

Pins the verifiable-reward contract: a perfect prediction earns reward every step
(return == n_steps), a wrong prediction earns 0 and (by default) ends the episode
so the return equals the faithful horizon. Dependency-free -- no RL framework.
"""

from __future__ import annotations

from verisim.loop import OracleBackedModel
from verisim.rl import Observation, WorldModelEnv, load_environment


def _perfect_delta(env: WorldModelEnv, obs):
    return OracleBackedModel(env.oracle).predict_delta(obs.state, obs.action)


def test_perfect_policy_earns_full_return():
    env = WorldModelEnv(driver="adversarial", seed=200, n_steps=12)
    obs: Observation | None = env.reset()
    total = 0.0
    steps = 0
    done = False
    while not done:
        transition = env.step(_perfect_delta(env, obs))
        total += transition.reward
        steps += 1
        done = transition.done
        obs = transition.observation
    assert steps == env.n_steps == 12
    assert total == 12.0  # return == faithful horizon == T for a perfect model
    assert obs is None


def test_wrong_prediction_terminates_at_the_horizon():
    env = WorldModelEnv(driver="weighted", seed=100, n_steps=12)
    env.reset()
    # The empty delta misses the oracle's result/structure edits -> immediate drift.
    transition = env.step([])
    assert transition.reward == 0.0
    assert transition.done is True
    assert transition.info["divergence"] > 0.0


def test_non_terminating_mode_runs_full_episode():
    env = WorldModelEnv(driver="weighted", seed=100, n_steps=10, terminate_on_divergence=False)
    env.reset()
    steps = 0
    done = False
    while not done:
        transition = env.step([])  # always wrong, but episode does not end early
        steps += 1
        done = transition.done
    assert steps == 10


def test_step_after_done_raises():
    env = WorldModelEnv(driver="weighted", seed=100, n_steps=5)
    env.reset()
    env.step([])  # terminates on first divergence
    try:
        env.step([])
    except RuntimeError:
        pass
    else:
        raise AssertionError("step() after a finished episode should raise")


def test_reset_returns_first_observation():
    env = WorldModelEnv(driver="weighted", seed=100, n_steps=5)
    obs = env.reset()
    assert obs.step == 0
    assert obs.action.raw in obs.prompt


def test_load_environment_entrypoint():
    env = load_environment(driver="weighted", seed=1, n_steps=3)
    assert isinstance(env, WorldModelEnv)
    assert env.n_steps == 3
