"""SPEC-21 CP1-CP3 -- the ordered structure->content computer-use task suite.

The measurement substrate of the scale law (SPEC-21 §3): a battery of computer-use
predictive-defense tasks ordered along the structure->content spectrum -- the dimension whose
model-accuracy the capacity axis moves. Each task is a SPEC-20-style predictive defense (UA8): a
defender rolls a model forward over an adversarial workload, predicts the **cumulative keyed set**
the workload touches (every process spawned / fd opened / file written / content emitted), protects
the budget it predicts, and is scored on how much of the *true* keyed set it caught:

    reward = |protected ∩ true_keyed_set| / min(budget, |true_keyed_set|)

A **faithful** predictor (oracle rollout) predicts the keyed set exactly -> reward 1.0; a **free**
predictor (`M_θ` rollout) drifts in proportion to the model's error *on that keyed dimension*, so
the
faithful-vs-free **gap** is the load-bearing signal: the oracle is load-bearing for the task iff the
free predictor cannot match the faithful one. The suite spans the spectrum (SPEC-21 §3 table):

  - **process-control** (order 0, *structure*): keyed on the process tree -- the model is ~0% drift
    here, so the gap should be ~0 (faithfulness not load-bearing), at every scale.
  - **fd-control** (order 1, *near-structure*, CP2): keyed on the open-fd table -- a structural
    lever that drifts moderately.
  - **file-integrity** (order 2, *content*, UA8/CP1): keyed on *which* files written -- the model
    drifts ~25-36% here, so the gap is material (the SPEC-20 positive).
  - **content-value** (order 3, *deep content*, CP3): keyed on the actual *(path, content)* pairs --
    not just which file but what was written, the highest-entropy dimension, the candidate for the
    irreducible residue (H88): content depends on input the model cannot learn, so its accuracy
    should plateau below threshold at *every* scale.

The science is sweeping each task across capacity (the scale law), not the count -- the suite is
deliberately small and ordered. Reuses the shipped UA8 workload/predictor machinery
([`acd.host_integrity`](../acd/host_integrity.py)); adds only the keyed-set abstraction, the new
rungs, and the structure->content ordering. CPU-only, seeded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.acd.host_integrity import (
    HostStepFn,
    make_workload,
    model_step,
    oracle_step,
)
from verisim.acd.host_integrity import (
    written_files as written_files,  # re-exported: the file-integrity keyed set (UA8)
)
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

if TYPE_CHECKING:
    from verisim.hostmodel import NeuralHostWorldModel

# A keyed-set extractor maps a host state to the set of objects a task defends (its keyed dimension)
KeyFn = Callable[[HostState], set[Any]]


# --- the keyed-set extractors (the structure->content spectrum) ----------------------------------


def alive_procs(state: HostState) -> set[object]:
    """Structure: the process tree -- ``(pid, state)`` of every live process (model ~0% drift)."""
    return {(p.pid, p.state) for p in state.procs.values()}


def open_fds(state: HostState) -> set[object]:
    """Near-structure: the open-fd table -- the ``(pid, fd)`` keys (a structural lever, drifts)."""
    return set(state.fds)


def file_contents(state: HostState) -> set[object]:
    """Deep content: the *(path, content)* pairs -- not just which file but *what was written*.

    The highest-entropy dimension: a free predictor can name the right file yet mis-predict its
    content, so this keyed set is strictly harder than :func:`written_files`, the candidate for
    the irreducible residue (H88) -- content keyed on effectively-unlearnable input.
    """
    return {
        (path, getattr(node, "content", ""))
        for path, node in state.fs.fs.items()
        if getattr(node, "content", "") != ""
    }


# --- the generic predictive-defense (the UA8 pattern, over any keyed dimension) ------------------


def rollout_keyed(
    step: HostStepFn, start: HostState, actions: Sequence[HostAction], key_fn: KeyFn,
) -> set[object]:
    """Roll the workload under ``step``; return the **cumulative** keyed set ever touched.

    Cumulative (the union over the rollout) so an object that appears then vanishes (a process that
    exits, an fd that closes) still counts -- the defensive "detect everything the workload did"
    framing, and the form that makes mid-rollout prediction error accumulate (parallel to UA8's
    accumulating file set and UA10's cumulative flow set).
    """
    state = start
    seen = set(key_fn(state))
    for action in actions:
        state = step(state, action)
        seen |= key_fn(state)
    return seen


def keyed_defense_reward(
    predictor: HostStepFn, true_step: HostStepFn, start: HostState,
    actions: Sequence[HostAction], budget: int, key_fn: KeyFn,
) -> float:
    """Protect the ``budget`` predicted-keyed objects; score against the true keyed set.

    The generic SPEC-20 predictive-defense reward over any keyed dimension. A faithful predictor
    catches the whole (budget-limited) true set (1.0); a drifted one protects the wrong objects.
    ``sorted(..., key=repr)`` gives a deterministic, type-heterogeneous budget order.
    """
    predicted = sorted(rollout_keyed(predictor, start, actions, key_fn), key=repr)
    true_set = rollout_keyed(true_step, start, actions, key_fn)
    if not true_set:
        return 1.0
    protected = set(predicted[:budget])
    return len(protected & true_set) / min(budget, len(true_set))


# --- the task abstraction + the ordered suite ----------------------------------------------------


@dataclass(frozen=True)
class CueTask:
    """One computer-use predictive-defense task, positioned on the structure->content spectrum."""

    name: str
    keyed_dimension: str  # the drift dimension it keys on: "procs" | "fds" | "fs"
    order: int  # 0 = structure ... 3 = deep content (the spectrum position)
    key_fn: KeyFn
    budget: int = 2


@dataclass(frozen=True)
class TaskGapConfig:
    """How a task's faithful-vs-free gap is measured (a fixed workload regime across scales)."""

    horizon: int = 16
    driver: str = "forky"
    workload_seeds: tuple[int, ...] = tuple(range(700, 724))


@dataclass(frozen=True)
class TaskGap:
    """One task's load-bearing measurement at one scale: faithful, free, and the gap."""

    task: str
    keyed_dimension: str
    order: int
    faithful: float
    free: float
    gap: float
    n: int


def keyed_drift(
    task: CueTask, model: NeuralHostWorldModel, config: TaskGapConfig | None = None, *,
    oracle: HostOracle | None = None,
) -> float:
    """The cheap per-task drift: the fraction of the true keyed set the free model misses (CP3/H89).

    Free-run the model beside the oracle and measure, per workload, ``|true \\ free| / |true|`` on
    the cumulative keyed set -- a single free-run measurement (no faithful predictor, no budget)
    is the *cheap forecast* of the task's gap (H89): a task is load-bearing iff its keyed dimension
    drifts. For the content-value task this is the per-dimension content-accuracy diagnostic of CP3
    (keys on ``(path, content)``, so it measures *content* drift, distinct from which-file drift).
    """
    config = config or TaskGapConfig()
    oracle = oracle or ReferenceHostOracle()
    free = model_step(model)
    true_step = oracle_step(oracle)
    misses: list[float] = []
    for seed in config.workload_seeds:
        start, actions = make_workload(seed, config.horizon, driver=config.driver, oracle=oracle)
        free_set = rollout_keyed(free, start, actions, task.key_fn)
        true_set = rollout_keyed(true_step, start, actions, task.key_fn)
        if true_set:
            misses.append(len(true_set - free_set) / len(true_set))
    return fmean(misses) if misses else 0.0


def task_gap(
    task: CueTask, model: NeuralHostWorldModel, config: TaskGapConfig | None = None, *,
    oracle: HostOracle | None = None,
) -> TaskGap:
    """The faithful-vs-free predictive-defense gap for ``task`` (the load-bearing signal)."""
    config = config or TaskGapConfig()
    oracle = oracle or ReferenceHostOracle()
    faithful = oracle_step(oracle)
    free = model_step(model)
    true_step = oracle_step(oracle)
    workloads = [
        make_workload(s, config.horizon, driver=config.driver, oracle=oracle)
        for s in config.workload_seeds
    ]
    f_rewards = [
        keyed_defense_reward(faithful, true_step, start, actions, task.budget, task.key_fn)
        for start, actions in workloads
    ]
    free_rewards = [
        keyed_defense_reward(free, true_step, start, actions, task.budget, task.key_fn)
        for start, actions in workloads
    ]
    f_mean, free_mean = fmean(f_rewards), fmean(free_rewards)
    return TaskGap(
        task=task.name, keyed_dimension=task.keyed_dimension, order=task.order,
        faithful=f_mean, free=free_mean, gap=f_mean - free_mean, n=len(workloads),
    )


# --- the ρ-grounded knee, generalized over keyed dimensions (UA9, for load-bearing tasks) --------


def grounded_keyed_rollout(
    model: object, oracle: HostOracle, start: HostState,
    actions: Sequence[HostAction], rho: float, key_fn: KeyFn,
) -> tuple[set[object], set[object], int]:
    """The ρ-grounded predictor over any keyed dimension (UA9): re-anchor every round(1/ρ) steps.

    A parallel true trajectory advances under the oracle; at each consultation step the predicted
    state is snapped to that truth (the defender paid an oracle call); between consults the model
    free-runs. ``ρ=1`` recovers the faithful predictor, ``ρ=0`` the free one. Returns the cumulative
    ``(predicted_keyed, true_keyed, oracle_calls)``.
    """
    from verisim.host.delta import apply

    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    true = start
    predicted = start
    true_seen = set(key_fn(true))
    pred_seen = set(key_fn(predicted))
    calls = 0
    for i, action in enumerate(actions, start=1):
        true = oracle.step(true, action).state
        true_seen |= key_fn(true)
        if rho >= 1.0 or (interval and i % interval == 0):
            predicted = true  # CONSULT -- re-anchor to truth
            calls += 1
        else:
            delta = model.predict_delta(predicted, action)  # type: ignore[attr-defined]
            predicted = apply(predicted, delta)
        pred_seen |= key_fn(predicted)
    return pred_seen, true_seen, calls


def grounded_keyed_reward(
    model: object, oracle: HostOracle, start: HostState,
    actions: Sequence[HostAction], budget: int, rho: float, key_fn: KeyFn,
) -> tuple[float, int]:
    """Protect the ``budget`` keyed objects the ρ-grounded predictor expects; score vs truth."""
    predicted_set, true_set, calls = grounded_keyed_rollout(
        model, oracle, start, actions, rho, key_fn
    )
    if not true_set:
        return 1.0, calls
    protected = set(sorted(predicted_set, key=repr)[:budget])
    return len(protected & true_set) / min(budget, len(true_set)), calls


def task_knee_rho(
    task: CueTask, model: NeuralHostWorldModel, rhos: Sequence[float],
    config: TaskGapConfig | None = None, *, oracle: HostOracle | None = None,
    knee_frac: float = 0.9,
) -> tuple[float, dict[float, float]]:
    """The smallest ρ recovering ``knee_frac`` of the faithful catch on ``task`` (the useful knee).

    Returns ``(knee_rho, catch_by_rho)``. The knee is the cheap-purchase signal for a load-bearing
    task: ``ρ=1`` (full oracle) is the faithful ceiling; the knee is where grounding recovers it.
    """
    config = config or TaskGapConfig()
    oracle = oracle or ReferenceHostOracle()
    workloads = [
        make_workload(s, config.horizon, driver=config.driver, oracle=oracle)
        for s in config.workload_seeds
    ]
    catch: dict[float, float] = {}
    for rho in rhos:
        rewards = [
            grounded_keyed_reward(model, oracle, start, actions, task.budget, rho, task.key_fn)[0]
            for start, actions in workloads
        ]
        catch[rho] = fmean(rewards)
    ceiling = catch[max(rhos)]
    threshold = knee_frac * ceiling
    knee = next((r for r in sorted(rhos) if catch[r] >= threshold), max(rhos))
    return knee, catch


#: The ordered structure->content suite (SPEC-21 §3). The order is the spectrum position the scale
#: law sweeps capacity through; the gap at each scale is the load-bearing signal.
TASK_SUITE: tuple[CueTask, ...] = (
    CueTask("process-control", "procs", 0, alive_procs),
    CueTask("fd-control", "fds", 1, open_fds),
    CueTask("file-integrity", "fs", 2, written_files),
    CueTask("content-value", "fs", 3, file_contents),
)
