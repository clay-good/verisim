"""The distributed-cluster faithfulness benchmark, framework-agnostic (SPEC-7 §7/§12, DS8).

The distributed analogue of :mod:`verisim.hosteval.faithfulness`: packages the distributed env +
divergence/consistency metrics + tiered-oracle ground-truth labels as a small, dependency-free
benchmark surface external evaluators (Inspect, ad-hoc scripts) wrap. What is new here is *what* is
graded -- a model of a **running replicated service under faults** (replicas + in-flight messages +
partition/crash/clock), the first world where the bit-exact global oracle is *intractable*, so the
benchmark reports **two** horizons: the bit-faithful one and the §9.1 **consistency-faithful** one
(the split-brain decision), which outlasts it where the error hides in the consistency-invisible
in-flight medium (ED5/H19). This is the metrology SPEC-7 §1.4 argues the distributed-systems field
lacks: Jepsen grades a *running* system's history, never a simulator's predicted next cluster state.

Two granularities, both built on the existing oracle/loop/metrics (the DS5 tiered loop):

  - **Rollout faithfulness** (:func:`score_dist_model`) -- run a distributed model through the
    tiered propose-verify-correct loop on a seeded rollout and report its bit-faithful horizon
    ``H_ε``, its consistency-faithful horizon, final divergence, and **oracle-dollars** (the §9.4
    tiered cost, the distributed-specific accounting -- a consult's price depends on the tier it
    spends). Model-agnostic (any ``DistModel``).
  - **Single-step labels** (:func:`dist_step_labels`, :func:`grade_dist_prediction`) -- the per-step
    ``(serialize(s, a) -> s')`` pairs and a divergence grader, the natural shape for a
    question-answer eval framework (Inspect).

Nothing here imports torch: the benchmark scores *any* model implementing the
:class:`~verisim.distloop.model.DistModel` protocol, learned or symbolic.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import apply, delta_from_list
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.dist.state import DistributedState
from verisim.distdata.drivers import DistDriver
from verisim.distloop import FixedTierPolicy, budget_for_rho, run_dist_rollout
from verisim.distloop.model import DistModel
from verisim.distloop.tier_policy import TierPolicy
from verisim.distmetrics.divergence import divergence
from verisim.distoracle.base import DistOracle
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.loop.policy import ConsultationPolicy, fixed_interval_for_rho
from verisim.metrics.horizon import faithful_horizon


@dataclass(frozen=True)
class DistFaithfulnessSample:
    """One benchmark item: a seeded cluster rollout the world model must stay faithful on.

    Fully determined by ``(driver, seed, n_steps)`` over the fixed distributed config, so the
    benchmark is regenerable from this manifest alone (SPEC-7 §3 / SPEC-2 §12). ``difficulty`` is a
    free-text label carried into results for grouping.
    """

    driver: str
    seed: int
    n_steps: int
    difficulty: str = "default"

    def actions(
        self, oracle: DistOracle, config: DistConfig = DEFAULT_DIST_CONFIG
    ) -> list[DistAction]:
        """The action sequence this sample rolls the oracle forward on, from the boot cluster."""
        driver = DistDriver(self.driver, config, random.Random(self.seed))
        state = DistributedState.initial(config)
        actions: list[DistAction] = []
        for _ in range(self.n_steps):
            action = driver.sample(state)
            actions.append(action)
            state = oracle.step(state, action).state
        return actions


@dataclass(frozen=True)
class DistFaithfulnessScore:
    """A model's result on one :class:`DistFaithfulnessSample` (distributed-cluster metrics)."""

    sample: DistFaithfulnessSample
    epsilon: float
    rho: float
    faithful_horizon: int
    consistency_horizon: int
    n_steps: int
    final_divergence: float
    oracle_calls: int
    oracle_dollars: int

    @property
    def normalized_horizon(self) -> float:
        """Bit-faithful horizon as a fraction of rollout length (``H_ε / T``)."""
        return self.faithful_horizon / self.n_steps if self.n_steps else 0.0

    @property
    def normalized_consistency_horizon(self) -> float:
        """Consistency-faithful horizon as a fraction of rollout length (§9.1 decision metric)."""
        return self.consistency_horizon / self.n_steps if self.n_steps else 0.0


# A small fixed suite spanning the distributed difficulty range (the committed benchmark).
DEFAULT_DIST_SUITE: tuple[DistFaithfulnessSample, ...] = (
    DistFaithfulnessSample("uniform", 100, 24, "low"),
    DistFaithfulnessSample("contention", 101, 24, "mid"),
    DistFaithfulnessSample("adversarial", 200, 24, "high"),
    DistFaithfulnessSample("adversarial", 201, 24, "high"),
)


def score_dist_model(
    model: DistModel,
    sample: DistFaithfulnessSample,
    *,
    oracle: DistOracle | None = None,
    config: DistConfig = DEFAULT_DIST_CONFIG,
    epsilon: float = 0.0,
    rho: float = 0.0,
    tier_policy: TierPolicy | None = None,
    policy: ConsultationPolicy | None = None,
) -> DistFaithfulnessScore:
    """Grade ``model`` on ``sample`` via the tiered propose-verify-correct loop.

    ``rho`` sets the oracle-consultation budget (``0`` = unaided, the pure faithfulness floor); a
    custom ``policy`` overrides the default ``fixed`` policy at that budget. ``tier_policy``
    (``π_w``) chooses which oracle tier each consult spends (default full bit-exact). The score
    carries both the bit-faithful and the §9.1 consistency-faithful horizon, plus oracle-dollars.
    """
    oracle = oracle or ReferenceDistOracle(config)
    actions = sample.actions(oracle, config)
    n = len(actions)
    record = run_dist_rollout(
        model,
        oracle,
        DistributedState.initial(config),
        actions,
        policy or fixed_interval_for_rho(rho),
        epsilon=epsilon,
        config=config,
        tier_policy=tier_policy or FixedTierPolicy("bit_exact"),
        budget=budget_for_rho(rho, n),
        seed=sample.seed,
    )
    consistency_divergences = record.config.get("consistency_divergences", [])
    return DistFaithfulnessScore(
        sample=sample,
        epsilon=epsilon,
        rho=rho,
        faithful_horizon=record.faithful_horizon,
        consistency_horizon=faithful_horizon(consistency_divergences, epsilon),
        n_steps=n,
        final_divergence=record.divergences[-1] if record.divergences else 0.0,
        oracle_calls=record.oracle_calls,
        oracle_dollars=record.config.get("oracle_dollars", 0),
    )


def score_dist_suite(
    model: DistModel,
    suite: Sequence[DistFaithfulnessSample] = DEFAULT_DIST_SUITE,
    **kwargs: object,
) -> list[DistFaithfulnessScore]:
    """Grade ``model`` on every sample in ``suite`` (see :func:`score_dist_model`)."""
    return [score_dist_model(model, s, **kwargs) for s in suite]  # type: ignore[arg-type]


@dataclass(frozen=True)
class DistStepLabel:
    """A single-step ground-truth label: ``serialize(s, a) -> serialize(s')`` (distributed)."""

    state: str
    action: str
    next_state: str


def dist_step_labels(
    sample: DistFaithfulnessSample,
    *,
    oracle: DistOracle | None = None,
    config: DistConfig = DEFAULT_DIST_CONFIG,
) -> list[DistStepLabel]:
    """The per-step ``(state, action) -> next_state`` labels along ``sample`` (canonical JSON)."""
    oracle = oracle or ReferenceDistOracle(config)
    driver = DistDriver(sample.driver, config, random.Random(sample.seed))
    state = DistributedState.initial(config)
    labels: list[DistStepLabel] = []
    for _ in range(sample.n_steps):
        action = driver.sample(state)
        result = oracle.step(state, action)
        labels.append(
            DistStepLabel(
                state=json.dumps(to_canonical(state), separators=(",", ":")),
                action=action.raw,
                next_state=json.dumps(to_canonical(result.state), separators=(",", ":")),
            )
        )
        state = result.state
    return labels


def grade_dist_prediction(predicted_next_state: str, true_next_state: str) -> float:
    """Score a predicted next-state string in ``[0, 1]`` (``1`` = exact) by cluster divergence.

    The score is ``1 - d`` under the §9.1 divergence: an exactly-correct prediction scores ``1.0``,
    and unparseable output scores ``0.0``. Both inputs are canonical distributed JSON strings
    (:func:`verisim.dist.serialize.to_canonical`).
    """
    truth = from_canonical(json.loads(true_next_state))
    try:
        predicted = from_canonical(json.loads(predicted_next_state))
    except (ValueError, KeyError, TypeError):
        return 0.0
    return 1.0 - divergence(predicted, truth)


def applied_dist_divergence(state: str, predicted_delta_json: str, true_next_state: str) -> float:
    """Score a predicted *cluster delta* (JSON edits) applied to ``state`` vs. truth.

    Convenience grader for evaluators whose model emits a delta rather than a full next state.
    Returns ``1 - d`` (``0.0`` if the delta JSON is unparseable).
    """
    truth = from_canonical(json.loads(true_next_state))
    base = from_canonical(json.loads(state))
    try:
        delta = delta_from_list(json.loads(predicted_delta_json))
    except (ValueError, KeyError, TypeError):
        return 0.0
    return 1.0 - divergence(apply(base, delta), truth)
