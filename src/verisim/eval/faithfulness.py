"""The faithfulness benchmark, framework-agnostic (SPEC-2 §15, milestone M8).

Packages the v0 env + divergence metric + oracle ground-truth labels as a small,
dependency-free benchmark surface that external evaluators (Inspect, ad-hoc
scripts) wrap. Two granularities, both built on the existing oracle/loop/metrics:

  - **Rollout faithfulness** (:func:`score_model`) -- run a model through the
    propose-verify-correct loop on a seeded rollout and report its faithful horizon
    `H_ε`, final divergence, and oracle calls. This is the headline benchmark a
    world model is graded on; it is model-agnostic (any ``verisim.loop.Model``).
  - **Single-step labels** (:func:`step_labels`, :func:`grade_prediction`) -- the
    per-step ground-truth pairs `(serialize(s, a) -> s')` and a divergence-based
    grader, the natural shape for a question-answer eval framework (Inspect).

Nothing here imports torch: the benchmark scores *any* model implementing the loop
``Model`` protocol, learned or symbolic.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass

from verisim.data.drivers import Driver
from verisim.delta.apply import apply
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.serialize import from_canonical, to_canonical_str
from verisim.env.state import State
from verisim.loop.model import Model
from verisim.loop.policy import ConsultationPolicy, fixed_interval_for_rho
from verisim.loop.runner import budget_for_rho, run_rollout
from verisim.metrics.divergence import divergence
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle


@dataclass(frozen=True)
class FaithfulnessSample:
    """One benchmark item: a seeded rollout the world model must stay faithful on.

    The rollout is fully determined by ``(driver, seed, n_steps)`` over the fixed
    env config, so the benchmark is regenerable from this manifest alone (SPEC-2
    §12). ``difficulty`` is a free-text label carried into results for grouping.
    """

    driver: str
    seed: int
    n_steps: int
    difficulty: str = "default"

    def actions(self, oracle: Oracle, env: EnvConfig = DEFAULT_CONFIG) -> list[Action]:
        """The action sequence this sample rolls the oracle forward on."""
        driver = Driver(name=self.driver, config=env, rng=random.Random(self.seed))
        state = State.empty()
        actions: list[Action] = []
        for _ in range(self.n_steps):
            action = driver.sample(state)
            actions.append(action)
            state = oracle.step(state, action).state
        return actions


@dataclass(frozen=True)
class FaithfulnessScore:
    """A model's result on one :class:`FaithfulnessSample`."""

    sample: FaithfulnessSample
    epsilon: float
    rho: float
    faithful_horizon: int
    n_steps: int
    final_divergence: float
    oracle_calls: int

    @property
    def normalized_horizon(self) -> float:
        """Faithful horizon as a fraction of rollout length (``H_ε / T``)."""
        return self.faithful_horizon / self.n_steps if self.n_steps else 0.0


# A small fixed suite spanning the v0 difficulty range (the committed benchmark).
DEFAULT_SUITE: tuple[FaithfulnessSample, ...] = (
    FaithfulnessSample("weighted", 100, 24, "low"),
    FaithfulnessSample("weighted", 101, 24, "low"),
    FaithfulnessSample("adversarial", 200, 24, "high"),
    FaithfulnessSample("adversarial", 201, 24, "high"),
)


def score_model(
    model: Model,
    sample: FaithfulnessSample,
    *,
    oracle: Oracle | None = None,
    env: EnvConfig = DEFAULT_CONFIG,
    epsilon: float = 0.0,
    rho: float = 0.0,
    policy: ConsultationPolicy | None = None,
) -> FaithfulnessScore:
    """Grade ``model`` on ``sample`` via the propose-verify-correct loop.

    ``rho`` sets the oracle-consultation budget (``0`` = unaided, the pure
    faithfulness floor). A custom ``policy`` overrides the default ``fixed`` policy
    at that budget.
    """
    oracle = oracle or ReferenceOracle()
    actions = sample.actions(oracle, env)
    n = len(actions)
    record = run_rollout(
        model,
        oracle,
        State.empty(),
        actions,
        policy or fixed_interval_for_rho(rho),
        epsilon=epsilon,
        budget=budget_for_rho(rho, n),
    )
    return FaithfulnessScore(
        sample=sample,
        epsilon=epsilon,
        rho=rho,
        faithful_horizon=record.faithful_horizon,
        n_steps=n,
        final_divergence=record.divergences[-1] if record.divergences else 0.0,
        oracle_calls=record.oracle_calls,
    )


def score_suite(
    model: Model, suite: Sequence[FaithfulnessSample] = DEFAULT_SUITE, **kwargs: object
) -> list[FaithfulnessScore]:
    """Grade ``model`` on every sample in ``suite`` (see :func:`score_model`)."""
    return [score_model(model, s, **kwargs) for s in suite]  # type: ignore[arg-type]


@dataclass(frozen=True)
class StepLabel:
    """A single-step ground-truth label: ``serialize(s, a) -> serialize(s')``.

    ``state`` and ``action`` are the human/serialized prompt; ``next_state`` is the
    canonical true next state -- the answer an eval framework grades against.
    """

    state: str
    action: str
    next_state: str


def step_labels(
    sample: FaithfulnessSample, *, oracle: Oracle | None = None, env: EnvConfig = DEFAULT_CONFIG
) -> list[StepLabel]:
    """The per-step ``(state, action) -> next_state`` labels along ``sample``."""
    oracle = oracle or ReferenceOracle()
    driver = Driver(name=sample.driver, config=env, rng=random.Random(sample.seed))
    state = State.empty()
    labels: list[StepLabel] = []
    for _ in range(sample.n_steps):
        action = driver.sample(state)
        result = oracle.step(state, action)
        labels.append(
            StepLabel(
                state=to_canonical_str(state),
                action=action.raw,
                next_state=to_canonical_str(result.state),
            )
        )
        state = result.state
    return labels


def grade_prediction(predicted_next_state: str, true_next_state: str) -> float:
    """Score a predicted next-state string in ``[0, 1]`` (``1`` = exact).

    The score is ``1 - d`` under the §7.1 divergence: an exactly-correct prediction
    scores ``1.0``, and unparseable output scores ``0.0``. Both inputs are canonical
    JSON state strings (see :func:`verisim.env.serialize.to_canonical_str`).
    """
    truth = from_canonical(json.loads(true_next_state))
    try:
        predicted = from_canonical(json.loads(predicted_next_state))
    except (ValueError, KeyError, TypeError):
        return 0.0
    return 1.0 - divergence(predicted, truth)


def applied_divergence(state: str, predicted_delta_json: str, true_next_state: str) -> float:
    """Score a predicted *delta* (JSON edits) applied to ``state`` vs. truth.

    Convenience grader for evaluators whose model emits a delta rather than a full
    next state. Returns ``1 - d`` (``0.0`` if the delta JSON is unparseable). Kept
    minimal: only the edit fields v0 needs are read.
    """
    from verisim.delta.serialize import delta_from_list

    truth = from_canonical(json.loads(true_next_state))
    base = from_canonical(json.loads(state))
    try:
        delta = delta_from_list(json.loads(predicted_delta_json))
    except (ValueError, KeyError, TypeError):
        return 0.0
    return 1.0 - divergence(apply(base, delta), truth)
