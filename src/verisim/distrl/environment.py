"""Oracle-as-reward RL environment for the distributed world (SPEC-7 §12 / DS8; the v0 §15 shape).

The distributed analogue of :mod:`verisim.hostrl.environment`: the distributed env + tiered oracle
wrapped as a ``verifiers``-spec RL environment (the Prime Intellect Environments Hub shape) whose
reward is *verifiable* -- the oracle adjudicates each predicted cluster transition, so no learned
reward model sits in the loop (the verifier-as-reward thesis). It is the realization of "train a
distributed-system world model against a verifiable oracle reward" and the substrate a future
change-safety / SRE task reward (§7, the `distsim` plan oracle) plugs into.

The episode rolls a fixed seeded admin/client/fault sequence. At each step the policy (the world
model under training) is shown the true current cluster state and the next action and must emit a
predicted **cluster delta**; the oracle supplies the true next state; the reward is whether the
prediction stayed faithful. The env is *teacher-forced* -- it advances along ground truth regardless
of the prediction -- so the per-step signal is clean. With ``terminate_on_divergence`` (the default)
the episode ends at the first unfaithful step, so the **return equals the faithful horizon** ``H_ε``
-- the headline metric *is* the RL return.

The one distributed-specific knob (the SPEC-7 headline, ED5/H19/H12): ``reward_mode`` chooses
*which* faithfulness the reward grades.

  - ``"bit_exact"`` (default) -- reward ``1`` while the full predicted cluster state is within
    ``epsilon`` bit-divergence of truth; the return is the bit-faithful horizon, comparable to every
    prior world.
  - ``"consistency"`` -- reward ``1`` while the model's **consistency view** (each object converged
    or split, and to what value, §9.1) matches truth; the return is the *consistency*-faithful
    horizon. This is the decision-relevant reward an SRE/defender cares about (is there a
    split-brain?), and it outlasts the bit-exact one exactly where the error hides in the
    consistency-invisible in-flight medium (ED5/H19) -- so an agent trained to this reward is graded
    on the question that matters, not on bytes it cannot be expected to nail in flight.

No framework dependency: a plain reset/step env conforming to the verifiers protocol, plus the
:func:`load_environment` entrypoint the hub discovers it by. Dependency-free, GPU-free, like v0's.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import DistDelta, apply
from verisim.dist.serialize import to_canonical
from verisim.dist.state import DistributedState
from verisim.distdata.drivers import DistDriver
from verisim.distmetrics.divergence import consistency_faithfulness, divergence
from verisim.distoracle.base import DistOracle
from verisim.distoracle.reference import ReferenceDistOracle

REWARD_MODES: tuple[str, ...] = ("bit_exact", "consistency")


@dataclass(frozen=True)
class Observation:
    """What the policy sees before predicting a step's cluster delta."""

    step: int
    state: DistributedState
    action: DistAction
    prompt: str  # serialized (state, action) -- the text form for an LLM policy


@dataclass(frozen=True)
class Transition:
    """The outcome of one :meth:`DistWorldModelEnv.step`."""

    observation: Observation | None  # next observation, or None at episode end
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


def _prompt(state: DistributedState, action: DistAction) -> str:
    body = json.dumps(to_canonical(state), separators=(",", ":"))
    return f"STATE: {body}\nACTION: {action.raw}"


class DistWorldModelEnv:
    """A reset/step env whose reward is the distributed oracle's faithfulness verdict.

    Reward per step is ``1.0`` while the predicted next cluster state is faithful (bit-divergence
    within ``epsilon``, or -- under ``reward_mode="consistency"`` -- the §9.1 consistency view
    matches), else ``0.0``. With ``terminate_on_divergence`` the episode ends on the first ``0.0``
    reward, so the return equals the faithful horizon ``H_ε`` in the chosen mode.
    """

    def __init__(
        self,
        *,
        driver: str = "adversarial",
        seed: int = 0,
        n_steps: int = 24,
        epsilon: float = 0.0,
        reward_mode: str = "bit_exact",
        config: DistConfig = DEFAULT_DIST_CONFIG,
        oracle: DistOracle | None = None,
        terminate_on_divergence: bool = True,
    ) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        self.oracle = oracle or ReferenceDistOracle(config)
        self.config = config
        self.epsilon = epsilon
        self.reward_mode = reward_mode
        self.terminate_on_divergence = terminate_on_divergence
        self._actions = self._roll_actions(driver, seed, n_steps)
        self._state = DistributedState.initial(config)
        self._step = 0
        self._done = False

    def _roll_actions(self, driver_name: str, seed: int, n_steps: int) -> list[DistAction]:
        driver = DistDriver(driver_name, self.config, random.Random(seed))
        state = DistributedState.initial(self.config)
        actions: list[DistAction] = []
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
        self._state = DistributedState.initial(self.config)
        self._step = 0
        self._done = False
        return self._observe()

    def _observe(self) -> Observation:
        action = self._actions[self._step]
        return Observation(
            step=self._step, state=self._state, action=action,
            prompt=_prompt(self._state, action),
        )

    def _faithful(self, predicted: DistributedState, truth: DistributedState) -> tuple[bool, float]:
        """The (faithful?, divergence) verdict under the configured reward mode."""
        d = divergence(predicted, truth)
        if self.reward_mode == "consistency":
            return consistency_faithfulness(truth, predicted) == 1.0, d
        return d <= self.epsilon, d

    def step(self, predicted_delta: DistDelta) -> Transition:
        """Apply the policy's predicted cluster delta, score it against the oracle, advance.

        Returns the reward (faithful in the configured mode), the next observation (or ``None`` at
        episode end), and the true bit-divergence + consistency-faithfulness in ``info``.
        Teacher-forced: the state advances along ground truth regardless of the prediction.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        action = self._actions[self._step]
        truth = self.oracle.step(self._state, action).state
        predicted = apply(self._state, predicted_delta)
        faithful, d = self._faithful(predicted, truth)
        reward = 1.0 if faithful else 0.0

        self._state = truth  # teacher-forced: advance along ground truth
        self._step += 1
        self._done = self._step >= self.n_steps or (
            self.terminate_on_divergence and not faithful
        )
        info = {
            "divergence": d,
            "consistency_faithfulness": consistency_faithfulness(truth, predicted),
            "faithful": faithful,
            "reward_mode": self.reward_mode,
        }
        observation = None if self._done else self._observe()
        return Transition(observation=observation, reward=reward, done=self._done, info=info)


def load_environment(**kwargs: Any) -> DistWorldModelEnv:
    """Entrypoint the verifiers / Prime Intellect hub discovers the distributed env by (§12 / DS8).

    Accepts the :class:`DistWorldModelEnv` keyword arguments (``driver``, ``seed``, ``n_steps``,
    ``epsilon``, ``reward_mode``, ``terminate_on_divergence``).
    """
    return DistWorldModelEnv(**kwargs)
