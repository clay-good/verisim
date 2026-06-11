"""UA8 -- file-integrity: faithfulness IS load-bearing here (SPEC-20 §7, H80; the positive side).

The cross-world law (SPEC-20 §7): the flat world-model learns discrete *structure* faithfully and
drifts on *content* (network flows / host file-writes). Every control task tested so far keyed on
structure, so they were drift-robust and faithfulness was not load-bearing. This module builds the
*content-keyed* task the law predicts is the exception: **predictive file-integrity defense.**

An adversarial host workload writes (corrupts) files over an episode. The defender has a budget and
must **protect the files it predicts will be corrupted** -- a predictive-defense decision depending
on the model's prediction of *which files get written*, which the host `M_θ` drifts on (~25%
on the written-file set; SPEC-20 §7 host diagnostic). The defender rolls a model forward over the
workload, takes the budget files it predicts corrupted, "protects" them, and is scored on how
many of the *true* corruptions it actually caught:

    reward = |protected ∩ truly-corrupted| / min(budget, |truly-corrupted|)

A **faithful** predictor (oracle rollout) predicts the corrupted set exactly -> protects the right
files -> reward 1.0. A **free** predictor (raw `M_θ` rollout) mis-predicts the written files ->
protects the wrong files -> reward < 1.0, and worse as the horizon (and so the compounded content
drift) grows. H80: the faithful predictor beats the free one, and the gap widens with horizon -- the
**positive** that closes the boundary from the other side (content-keyed control *does* need
faithfulness). No training; CPU-only, seeded.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.state import HostState
from verisim.hostdata import HostDriver
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

# A host step function evolves the state under one action -- the predictor's model of dynamics
# (the exact oracle, or a learned host M_θ wrapped to predict-and-apply).
HostStepFn = Callable[[HostState, HostAction], HostState]


def written_files(state: HostState) -> set[str]:
    """The set of file paths with non-empty content -- the 'corrupted' (written) files."""
    return {
        path for path, node in state.fs.fs.items()
        if getattr(node, "content", "") != ""
    }


def make_workload(
    seed: int, n_steps: int, *, driver: str = "forky", oracle: HostOracle | None = None,
    host: HostConfig = DEFAULT_HOST_CONFIG,
) -> tuple[HostState, tuple[HostAction, ...]]:
    """A seeded host action sequence from the initial state (the adversarial workload)."""
    oracle = oracle or ReferenceHostOracle()
    drv = HostDriver(name=driver, config=host, rng=random.Random(seed))
    state = HostState.initial()
    actions: list[HostAction] = []
    for _ in range(n_steps):
        action = drv.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return HostState.initial(), tuple(actions)


def rollout_writes(
    step: HostStepFn, start: HostState, actions: Sequence[HostAction],
) -> set[str]:
    """Roll the workload under ``step``; return the set of files written (predicted/true set)."""
    state = start
    for action in actions:
        state = step(state, action)
    return written_files(state)


def predictive_defense_reward(
    predictor: HostStepFn, true_step: HostStepFn, start: HostState,
    actions: Sequence[HostAction], budget: int,
) -> float:
    """Protect the ``budget`` predicted-corrupted files; score against the true corruptions.

    ``reward = |protected ∩ truly-corrupted| / min(budget, |truly-corrupted|)`` -- the fraction
    of (budget-limited) true corruptions the defender caught. A faithful predictor catches them
    all (1.0); a drifted one protects the wrong files.
    """
    predicted = sorted(rollout_writes(predictor, start, actions))  # deterministic budget order
    true_corrupted = rollout_writes(true_step, start, actions)
    if not true_corrupted:
        return 1.0
    protected = set(predicted[:budget])
    caught = len(protected & true_corrupted)
    return caught / min(budget, len(true_corrupted))


def oracle_step(oracle: HostOracle) -> HostStepFn:
    """The faithful predictor: the exact oracle as a step function."""

    def step(state: HostState, action: HostAction) -> HostState:
        return oracle.step(state, action).state

    return step


def model_step(model: object) -> HostStepFn:
    """The free predictor: a host ``M_θ`` (predict_delta) wrapped as a step function."""
    from verisim.host.delta import apply

    def step(state: HostState, action: HostAction) -> HostState:
        return apply(state, model.predict_delta(state, action))  # type: ignore[attr-defined]

    return step
