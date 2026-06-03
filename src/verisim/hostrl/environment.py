"""Oracle-as-reward RL environment for the host world (SPEC-6 §12 / HC8; the v0 §15 shape).

The host analogue of :mod:`verisim.rl.environment`: the host env + reference oracle wrapped as a
``verifiers``-spec RL environment (the Prime Intellect Environments Hub shape) whose reward is
*verifiable* -- the oracle adjudicates each predicted bundle transition, so no learned reward model
sits in the loop (the verifier-as-reward thesis). It is the realization of "train a host world model
against a verifiable oracle reward" and the substrate a future denial-aware / task-grounded reward
(EH8/EH9, §7) plugs into.

The episode rolls a fixed seeded syscall sequence. At each step the policy (the world model under
training) is shown the true current bundle state and the next syscall and must emit a predicted
**bundle delta**; the oracle supplies the true next state; the reward is whether the prediction
stayed within composed-divergence tolerance ``ε`` of truth (§9.1). The env is *teacher-forced* -- it
advances along ground truth regardless of the prediction -- so the per-step signal is clean. With
``terminate_on_divergence`` (the default) the episode ends at the first unfaithful step, so the
**return equals the composed faithful horizon** ``H_ε`` -- the headline metric *is* the RL return.

No framework dependency: a plain reset/step env conforming to the verifiers protocol, plus the
:func:`load_environment` entrypoint the hub discovers it by. Dependency-free, like v0's.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.delta import HostDelta, apply
from verisim.host.state import HostState, to_canonical_host
from verisim.hostdata.drivers import HostDriver
from verisim.hostmetrics.divergence import divergence
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle


@dataclass(frozen=True)
class Observation:
    """What the policy sees before predicting a step's bundle delta."""

    step: int
    state: HostState
    action: HostAction
    prompt: str  # serialized (state, action) -- the text form for an LLM policy


@dataclass(frozen=True)
class Transition:
    """The outcome of one :meth:`HostWorldModelEnv.step`."""

    observation: Observation | None  # next observation, or None at episode end
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


def _prompt(state: HostState, action: HostAction) -> str:
    body = json.dumps(to_canonical_host(state), separators=(',', ':'))
    return f"STATE: {body}\nACTION: {action.raw}"


class HostWorldModelEnv:
    """A reset/step env whose reward is the host oracle's composed-faithfulness verdict.

    Reward per step is ``1.0`` while the predicted next bundle state is within ``epsilon`` composed
    divergence of the oracle's truth, else ``0.0``. With ``terminate_on_divergence`` the episode
    ends on the first ``0.0`` reward, so the return equals the composed faithful horizon ``H_ε``.
    """

    def __init__(
        self,
        *,
        driver: str = "forky",
        seed: int = 0,
        n_steps: int = 24,
        epsilon: float = 0.0,
        config: HostConfig = DEFAULT_HOST_CONFIG,
        oracle: HostOracle | None = None,
        terminate_on_divergence: bool = True,
    ) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        self.oracle = oracle or ReferenceHostOracle()
        self.config = config
        self.epsilon = epsilon
        self.terminate_on_divergence = terminate_on_divergence
        self._actions = self._roll_actions(driver, seed, n_steps)
        self._state = HostState.initial()
        self._step = 0
        self._done = False

    def _roll_actions(self, driver_name: str, seed: int, n_steps: int) -> list[HostAction]:
        driver = HostDriver(name=driver_name, config=self.config, rng=random.Random(seed))
        state = HostState.initial()
        actions: list[HostAction] = []
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
        self._state = HostState.initial()
        self._step = 0
        self._done = False
        return self._observe()

    def _observe(self) -> Observation:
        action = self._actions[self._step]
        return Observation(
            step=self._step, state=self._state, action=action,
            prompt=_prompt(self._state, action),
        )

    def step(self, predicted_delta: HostDelta) -> Transition:
        """Apply the policy's predicted bundle delta, score it against the oracle, advance.

        Returns the reward (faithful = composed divergence within ``ε``), the next observation (or
        ``None`` at episode end), and the true divergence in ``info``. Teacher-forced: the state
        advances along ground truth regardless of the prediction.
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
        info = {"divergence": d, "faithful": faithful}
        observation = None if self._done else self._observe()
        return Transition(observation=observation, reward=reward, done=self._done, info=info)


def load_environment(**kwargs: Any) -> HostWorldModelEnv:
    """Entrypoint the verifiers / Prime Intellect hub discovers the host env by (§12 / HC8).

    Accepts the :class:`HostWorldModelEnv` keyword arguments (``driver``, ``seed``, ``n_steps``,
    ``epsilon``, ``terminate_on_divergence``).
    """
    return HostWorldModelEnv(**kwargs)
