"""Oracle-as-reward RL environment for training a faithful world model (SPEC-2 §15).

Wraps the v0 env + reference oracle as a ``verifiers``-spec RL environment (the
Prime Intellect Environments Hub shape, SPEC.md §14): a reset/step/reward loop
whose reward is *verifiable* -- the oracle adjudicates each predicted transition,
so no learned reward model is in the loop (the author's verifier-as-reward thesis,
SPEC.md §8). This is the realization of "train a world model against a verifiable
oracle reward" (SPEC-2 §5.3 Stage 2 / §6.3).

The episode rolls a fixed seeded action sequence. At each step the policy (the
world model under training) is shown the true current state and the next action
and must emit a predicted delta; the oracle supplies the true next state; the
reward is whether the prediction stayed within tolerance ``ε`` of truth. The
environment is *teacher-forced* -- it advances along ground truth regardless of the
prediction -- so the per-step signal is clean. With ``terminate_on_divergence``
(the default) the episode ends at the first unfaithful step, making the **return
equal to the faithful horizon** ``H_ε`` -- the headline metric *is* the RL return.

No framework dependency: this is a plain reset/step env conforming to the verifiers
protocol, plus the :func:`load_environment` entrypoint the hub discovers it by.
Dependency-free so it is testable and usable without the RL stack installed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from verisim.delta.apply import apply
from verisim.delta.edits import Delta
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.serialize import to_canonical_str
from verisim.env.state import State
from verisim.metrics.divergence import divergence
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle


@dataclass(frozen=True)
class Observation:
    """What the policy sees before predicting a step's delta."""

    step: int
    state: State
    action: Action
    prompt: str  # serialized (state, action) -- the text form for an LLM policy


@dataclass(frozen=True)
class Transition:
    """The outcome of one :meth:`WorldModelEnv.step`."""

    observation: Observation | None  # next observation, or None at episode end
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


def _prompt(state: State, action: Action) -> str:
    return f"STATE: {to_canonical_str(state)}\nACTION: {action.raw}"


class WorldModelEnv:
    """A reset/step env whose reward is the oracle's faithfulness verdict.

    Reward per step is ``1.0`` while the predicted next state is within ``epsilon``
    divergence of the oracle's truth, else ``0.0``. With ``terminate_on_divergence``
    the episode ends on the first ``0.0`` reward, so the episode return equals the
    faithful horizon ``H_ε``.
    """

    def __init__(
        self,
        *,
        driver: str = "weighted",
        seed: int = 0,
        n_steps: int = 24,
        epsilon: float = 0.0,
        env: EnvConfig = DEFAULT_CONFIG,
        oracle: Oracle | None = None,
        terminate_on_divergence: bool = True,
    ) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        self.oracle = oracle or ReferenceOracle()
        self.env = env
        self.epsilon = epsilon
        self.terminate_on_divergence = terminate_on_divergence
        self._actions = self._roll_actions(driver, seed, n_steps)
        self._state = State.empty()
        self._step = 0
        self._done = False

    def _roll_actions(self, driver_name: str, seed: int, n_steps: int) -> list[Action]:
        from verisim.data.drivers import Driver

        driver = Driver(name=driver_name, config=self.env, rng=random.Random(seed))
        state = State.empty()
        actions: list[Action] = []
        for _ in range(n_steps):
            action = driver.sample(state)
            actions.append(action)
            state = self.oracle.step(state, action).state
        return actions

    @property
    def n_steps(self) -> int:
        return len(self._actions)

    def reset(self) -> Observation:
        """Restart the episode; return the first observation."""
        self._state = State.empty()
        self._step = 0
        self._done = False
        return self._observe()

    def _observe(self) -> Observation:
        action = self._actions[self._step]
        return Observation(
            step=self._step,
            state=self._state,
            action=action,
            prompt=_prompt(self._state, action),
        )

    def step(self, predicted_delta: Delta) -> Transition:
        """Apply the policy's predicted delta, score it against the oracle, advance.

        ``predicted_delta`` is the world model's prediction for the current
        observation's ``(state, action)``. Returns the reward (faithful = within
        ``ε``), the next observation (or ``None`` at episode end), and the true
        divergence in ``info``.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        action = self._actions[self._step]
        truth = self.oracle.step(self._state, action).state
        predicted = apply(self._state, predicted_delta)
        d = divergence(predicted, truth)
        faithful = d <= self.epsilon
        reward = 1.0 if faithful else 0.0

        self._state = truth  # teacher-forced: advance along ground truth
        self._step += 1
        self._done = self._step >= self.n_steps or (
            self.terminate_on_divergence and not faithful
        )
        info = {"divergence": d, "faithful": faithful, "true_next_state": to_canonical_str(truth)}
        observation = None if self._done else self._observe()
        return Transition(observation=observation, reward=reward, done=self._done, info=info)


def load_environment(**kwargs: Any) -> WorldModelEnv:
    """Entrypoint the verifiers / Prime Intellect hub discovers the env by (§15).

    Accepts the :class:`WorldModelEnv` keyword arguments (``driver``, ``seed``,
    ``n_steps``, ``epsilon``, ``terminate_on_divergence``).
    """
    return WorldModelEnv(**kwargs)
