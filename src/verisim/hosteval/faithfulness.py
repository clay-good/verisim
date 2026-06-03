"""The composed-host faithfulness benchmark, framework-agnostic (SPEC-6 §7/§12 §15, HC8).

The host analogue of :mod:`verisim.eval.faithfulness`: packages the host env + composed-divergence
metric + oracle ground-truth labels as a small, dependency-free benchmark surface external
evaluators (Inspect, ad-hoc scripts) wrap. What is new here is *what* is being graded -- a model of
a **whole running machine** (process table + fd tables + the embedded filesystem), not one
subsystem -- which is the missing metrology SPEC-6 §1.4 argues the computer-use field lacks: OSWorld
and TheAgentCompany grade the *agent*, never a simulator of the host's predicted next state.

Two granularities, both built on the existing oracle/loop/metrics (the HC5 composed loop):

  - **Rollout faithfulness** (:func:`score_host_model`) -- run a host model through the composed
    propose-verify-correct loop on a seeded rollout and report its composed faithful horizon
    ``H_ε``, final divergence, and oracle calls. Model-agnostic (any ``HostModel``).
  - **Single-step labels** (:func:`host_step_labels`, :func:`grade_host_prediction`) -- the per-step
    ``(serialize(s, a) -> s')`` pairs and a composed-divergence grader, the natural shape for a
    question-answer eval framework (Inspect).

Nothing here imports torch: the benchmark scores *any* model implementing the
:class:`~verisim.hostloop.model.HostModel` protocol, learned or symbolic.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.delta import apply, delta_from_list
from verisim.host.state import HostState, from_canonical_host, to_canonical_host
from verisim.hostdata.drivers import HostDriver
from verisim.hostloop import PartialHostOracle, budget_for_rho, run_host_rollout
from verisim.hostloop.model import HostModel
from verisim.hostmetrics.divergence import divergence
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import ConsultationPolicy, fixed_interval_for_rho


@dataclass(frozen=True)
class HostFaithfulnessSample:
    """One benchmark item: a seeded host rollout the world model must stay faithful on.

    Fully determined by ``(driver, seed, n_steps)`` over the fixed host config, so the benchmark is
    regenerable from this manifest alone (SPEC-6 §3 / SPEC-2 §12). ``difficulty`` is a free-text
    label carried into results for grouping.
    """

    driver: str
    seed: int
    n_steps: int
    difficulty: str = "default"

    def actions(
        self, oracle: HostOracle, config: HostConfig = DEFAULT_HOST_CONFIG
    ) -> list[HostAction]:
        """The syscall sequence this sample rolls the oracle forward on, from the boot state."""
        driver = HostDriver(name=self.driver, config=config, rng=random.Random(self.seed))
        state = HostState.initial()
        actions: list[HostAction] = []
        for _ in range(self.n_steps):
            action = driver.sample(state)
            actions.append(action)
            state = oracle.step(state, action).state
        return actions


@dataclass(frozen=True)
class HostFaithfulnessScore:
    """A model's result on one :class:`HostFaithfulnessSample` (composed-host metrics)."""

    sample: HostFaithfulnessSample
    epsilon: float
    rho: float
    faithful_horizon: int
    n_steps: int
    final_divergence: float
    oracle_calls: int

    @property
    def normalized_horizon(self) -> float:
        """Composed faithful horizon as a fraction of rollout length (``H_ε / T``)."""
        return self.faithful_horizon / self.n_steps if self.n_steps else 0.0


# A small fixed suite spanning the host difficulty range (the committed benchmark).
DEFAULT_HOST_SUITE: tuple[HostFaithfulnessSample, ...] = (
    HostFaithfulnessSample("uniform", 100, 24, "low"),
    HostFaithfulnessSample("forky", 101, 24, "mid"),
    HostFaithfulnessSample("adversarial", 200, 24, "high"),
    HostFaithfulnessSample("adversarial", 201, 24, "high"),
)


def score_host_model(
    model: HostModel,
    sample: HostFaithfulnessSample,
    *,
    oracle: HostOracle | None = None,
    config: HostConfig = DEFAULT_HOST_CONFIG,
    epsilon: float = 0.0,
    rho: float = 0.0,
    policy: ConsultationPolicy | None = None,
) -> HostFaithfulnessScore:
    """Grade ``model`` on ``sample`` via the composed propose-verify-correct loop (full mode).

    ``rho`` sets the oracle-consultation budget (``0`` = unaided, the pure composed-faithfulness
    floor). A custom ``policy`` overrides the default ``fixed`` policy at that budget.
    """
    oracle = oracle or ReferenceHostOracle()
    actions = sample.actions(oracle, config)
    n = len(actions)
    record = run_host_rollout(
        model,
        PartialHostOracle(oracle),
        HostState.initial(),
        actions,
        policy or fixed_interval_for_rho(rho),
        epsilon=epsilon,
        budget=budget_for_rho(rho, n),
        seed=sample.seed,
    )
    return HostFaithfulnessScore(
        sample=sample,
        epsilon=epsilon,
        rho=rho,
        faithful_horizon=record.faithful_horizon,
        n_steps=n,
        final_divergence=record.divergences[-1] if record.divergences else 0.0,
        oracle_calls=record.oracle_calls,
    )


def score_host_suite(
    model: HostModel,
    suite: Sequence[HostFaithfulnessSample] = DEFAULT_HOST_SUITE,
    **kwargs: object,
) -> list[HostFaithfulnessScore]:
    """Grade ``model`` on every sample in ``suite`` (see :func:`score_host_model`)."""
    return [score_host_model(model, s, **kwargs) for s in suite]  # type: ignore[arg-type]


@dataclass(frozen=True)
class HostStepLabel:
    """A single-step ground-truth label: ``serialize(s, a) -> serialize(s')`` (composed host)."""

    state: str
    action: str
    next_state: str


def host_step_labels(
    sample: HostFaithfulnessSample,
    *,
    oracle: HostOracle | None = None,
    config: HostConfig = DEFAULT_HOST_CONFIG,
) -> list[HostStepLabel]:
    """The per-step ``(state, action) -> next_state`` labels along ``sample`` (canonical JSON)."""
    oracle = oracle or ReferenceHostOracle()
    driver = HostDriver(name=sample.driver, config=config, rng=random.Random(sample.seed))
    state = HostState.initial()
    labels: list[HostStepLabel] = []
    for _ in range(sample.n_steps):
        action = driver.sample(state)
        result = oracle.step(state, action)
        labels.append(
            HostStepLabel(
                state=json.dumps(to_canonical_host(state), separators=(",", ":")),
                action=action.raw,
                next_state=json.dumps(to_canonical_host(result.state), separators=(",", ":")),
            )
        )
        state = result.state
    return labels


def grade_host_prediction(predicted_next_state: str, true_next_state: str) -> float:
    """Score a predicted next-state string in ``[0, 1]`` (``1`` = exact) by composed divergence.

    The score is ``1 - d`` under the §9.1 composed divergence: an exactly-correct prediction scores
    ``1.0``, and unparseable output scores ``0.0``. Both inputs are canonical host JSON strings
    (:func:`verisim.host.state.to_canonical_host`).
    """
    truth = from_canonical_host(json.loads(true_next_state))
    try:
        predicted = from_canonical_host(json.loads(predicted_next_state))
    except (ValueError, KeyError, TypeError):
        return 0.0
    return 1.0 - divergence(predicted, truth)


def applied_host_divergence(state: str, predicted_delta_json: str, true_next_state: str) -> float:
    """Score a predicted *bundle delta* (JSON edits) applied to ``state`` vs. truth.

    Convenience grader for evaluators whose model emits a delta rather than a full next state.
    Returns ``1 - d`` (``0.0`` if the delta JSON is unparseable).
    """
    truth = from_canonical_host(json.loads(true_next_state))
    base = from_canonical_host(json.loads(state))
    try:
        delta = delta_from_list(json.loads(predicted_delta_json))
    except (ValueError, KeyError, TypeError):
        return 0.0
    return 1.0 - divergence(apply(base, delta), truth)
