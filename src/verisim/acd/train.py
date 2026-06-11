"""UA1 -- REINFORCE training + evaluation for the containment defender (SPEC-20 §5).

The training engine that turns the UA0 env + linear policy into a learned defender, and the
evaluation that scores any policy against any backend. The whole UA2/H74 experiment is two calls
into this module with the *backend* swapped (train in `E_grounded`/`E_free`, eval in `E_oracle`) --
the learner code path is identical, which is exactly what makes the comparison isolate the value of
oracle-grounding rather than anything about the agent.

REINFORCE with a moving-average baseline (the smallest policy-gradient method that does the job,
SPEC-2 §5.3): roll a batch of episodes, compute each step's return, subtract the batch-mean
baseline, and nudge the weights along the score function. Torch-free and seeded: it trains against
the oracle backend in CI without the RL stack, and against the model backends locally.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .containment import ContainmentEnv
from .policy import N_ACTION_FEATURES, LinearPolicy, action_features


@dataclass(frozen=True)
class TrainConfig:
    """The REINFORCE schedule (deliberately small -- the result is the env, not the optimizer)."""

    episodes: int = 300
    batch: int = 10  # episodes per gradient step (the baseline is the batch mean)
    lr: float = 0.2
    gamma: float = 0.98
    seed: int = 0

    @staticmethod
    def smoke() -> TrainConfig:
        return TrainConfig(episodes=12, batch=4, lr=0.2)


@dataclass(frozen=True)
class Step:
    """One env step's REINFORCE record: the candidate features, the choice, and the reward."""

    feats_list: list[list[float]]
    chosen: int
    reward: float


def rollout(env: ContainmentEnv, policy: LinearPolicy, rng: random.Random, seed: int) -> list[Step]:
    """Run one episode under ``policy``; return the per-step REINFORCE records."""
    env.reset(seed)
    steps: list[Step] = []
    done = False
    while not done:
        actions = env.legal_actions()
        feats_list = [action_features(env.net, env.compromised, a) for a in actions]
        idx = policy.sample(feats_list, rng)
        out = env.step(actions[idx])
        steps.append(Step(feats_list, idx, out.reward))
        done = out.done
    return steps


def _returns(rewards: list[float], gamma: float) -> list[float]:
    """Discounted return-to-go at each step."""
    out = [0.0] * len(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


def reinforce(
    make_env: Callable[[], ContainmentEnv], config: TrainConfig | None = None,
    *, policy: LinearPolicy | None = None,
) -> LinearPolicy:
    """Train a containment defender by REINFORCE against the env ``make_env`` builds (one backend).

    ``make_env`` is a thunk so each episode gets a fresh env over the *same* backend (the backend
    carries the model/oracle; the episode seed varies the topology). Returns the trained policy.
    """
    config = config or TrainConfig()
    policy = policy or LinearPolicy()
    rng = random.Random(config.seed)
    env = make_env()

    episode = 0
    while episode < config.episodes:
        batch_returns: list[list[float]] = []
        batch_steps: list[list[Step]] = []
        for _ in range(min(config.batch, config.episodes - episode)):
            steps = rollout(env, policy, rng, seed=config.seed + 1000 + episode)
            rets = _returns([s.reward for s in steps], config.gamma)
            batch_steps.append(steps)
            batch_returns.append(rets)
            episode += 1
        # baseline = mean return-to-go across the batch's first step (a simple, stable baseline)
        flat_returns = [r for rets in batch_returns for r in rets]
        baseline = fmean(flat_returns) if flat_returns else 0.0
        grad = [0.0] * N_ACTION_FEATURES
        n_terms = 0
        for steps, rets in zip(batch_steps, batch_returns, strict=True):
            for step, ret in zip(steps, rets, strict=True):
                advantage = ret - baseline
                g = policy.logprob_grad(step.feats_list, step.chosen)
                for k in range(N_ACTION_FEATURES):
                    grad[k] += advantage * g[k]
                n_terms += 1
        if n_terms:
            policy.weights = [
                w + config.lr * grad[k] / n_terms for k, w in enumerate(policy.weights)
            ]
    return policy


def evaluate(
    make_env: Callable[[], ContainmentEnv], policy: LinearPolicy, *, seeds: tuple[int, ...],
) -> float:
    """Mean final containment of ``policy`` (greedy) over ``seeds`` in ``make_env``'s env."""
    env = make_env()
    finals: list[float] = []
    for seed in seeds:
        env.reset(seed)
        done = False
        info: dict[str, Any] = {"containment": 1.0}
        while not done:
            actions = env.legal_actions()
            feats_list = [action_features(env.net, env.compromised, a) for a in actions]
            idx = policy.greedy(feats_list)
            out = env.step(actions[idx])
            info = out.info
            done = out.done
        finals.append(float(info["containment"]))
    return fmean(finals)
