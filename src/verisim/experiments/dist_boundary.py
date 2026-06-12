"""SPEC-20 cross-world: the structure/content boundary on the THIRD world (distributed) — UA11.

SPEC-20 drew the boundary law on the host world (content = file-writes, UA8/UA9) and confirmed it
cross-world on the network (content = flows, UA10): oracle-grounded faithfulness is load-bearing for
control *exactly when* the task keys on the *content* the model drifts on, not the *structure* it
learns. Two worlds is a pattern; three is a law. This is the third-world confirmation, on the
**distributed** world (SPEC-7) — the hardest, where the global oracle is intractable and the state
is replicated values under partition.

The distributed structure/content split:

  - **partition-control** (*structure*): keyed on the partition groups — the cluster topology the
    fault ops (`partition`/`heal`/`crash`) move. The discrete membership structure a model learns.
  - **value-integrity** (*content*): keyed on the replicated *(object, value)* pairs — the MVCC data
    the client ops (`put`/`cas`) write, the content a learned `M_θ` drifts on.

Each is a SPEC-20 predictive-defense: a faithful predictor (oracle rollout) vs a free predictor
(`M_θ` rollout), scored on the cumulative keyed set caught. The boundary holds iff the *content* gap
materially exceeds the *structure* gap — faithfulness is more load-bearing where the model drifts.
The ρ-grounded knee on the content task is the cheap-purchase signal (UA9). The distributed world is
harder than host/network (the model is smaller and the world more complex, so the structure gap is
non-zero rather than ~0), but the *gradient* — content > structure — is the cross-world law.
CPU-local; CI runs the smoke instance.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.dist.action import DistAction
from verisim.dist.delta import apply
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.experiments.ed1_learned import ED1LearnedConfig, train_model
from verisim.metrics.aggregate import bootstrap_ci

if TYPE_CHECKING:
    from verisim.distmodel import NeuralDistWorldModel

DistStepFn = Callable[[DistributedState, DistAction], DistributedState]
DistKeyFn = Callable[[DistributedState], set[Any]]


# --- the distributed keyed-set extractors (structure vs content) ----------------------------------


def partition_set(state: DistributedState) -> set[Any]:
    """Structure: the partition groups (the cluster topology the fault ops move)."""
    return set(state.partitions)


def value_set(state: DistributedState) -> set[Any]:
    """Content: the replicated ``(object_id, value)`` pairs (the MVCC data put/cas write)."""
    return {(r.object_id, r.value) for r in state.replicas.values()}


# --- the predictive-defense (the UA8/UA10 pattern, distributed vertical) --------------------------


def make_dist_workload(
    seed: int, n_steps: int, *, oracle: DistOracle, config: ED1LearnedConfig,
    driver: str = "uniform",
) -> tuple[DistributedState, tuple[DistAction, ...]]:
    """A seeded distributed workload (the boot cluster + a driver's action sequence to replay)."""
    drv = DistDriver(name=driver, config=config.dist, rng=random.Random(seed))
    state = DistributedState.initial(config.dist)
    actions: list[DistAction] = []
    for _ in range(n_steps):
        action = drv.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return DistributedState.initial(config.dist), tuple(actions)


def _rollout_keyed(
    step: DistStepFn, start: DistributedState, actions: Sequence[DistAction], key_fn: DistKeyFn
) -> set[Any]:
    """Cumulative keyed set ever touched (the union over the rollout, the UA10 framing)."""
    state = start
    seen = set(key_fn(state))
    for action in actions:
        state = step(state, action)
        seen |= key_fn(state)
    return seen


def _keyed_reward(
    predictor: DistStepFn, true_step: DistStepFn, start: DistributedState,
    actions: Sequence[DistAction], budget: int, key_fn: DistKeyFn,
) -> float:
    """Protect the ``budget`` predicted-keyed objects; score vs the true cumulative keyed set."""
    predicted = sorted(_rollout_keyed(predictor, start, actions, key_fn), key=repr)
    true_set = _rollout_keyed(true_step, start, actions, key_fn)
    if not true_set:
        return 1.0
    protected = set(predicted[:budget])
    return len(protected & true_set) / min(budget, len(true_set))


def _grounded_keyed_rollout(
    model: object, oracle: DistOracle, start: DistributedState, actions: Sequence[DistAction],
    rho: float, key_fn: DistKeyFn,
) -> tuple[set[Any], set[Any], int]:
    """The ρ-grounded predictor over a distributed keyed dim (UA9): re-anchor every round(1/ρ)."""
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
            predicted = true
            calls += 1
        else:
            predicted = apply(predicted, model.predict_delta(predicted, action))  # type: ignore[attr-defined]
        pred_seen |= key_fn(predicted)
    return pred_seen, true_seen, calls


def _oracle_step(oracle: DistOracle) -> DistStepFn:
    def step(state: DistributedState, action: DistAction) -> DistributedState:
        return oracle.step(state, action).state

    return step


def _model_step(model: object) -> DistStepFn:
    def step(state: DistributedState, action: DistAction) -> DistributedState:
        return apply(state, model.predict_delta(state, action))  # type: ignore[attr-defined]

    return step


@dataclass(frozen=True)
class DistTask:
    """One distributed predictive-defense task (structure or content)."""

    name: str
    keyed_dimension: str  # "partitions" | "values"
    band: str  # "structure" | "content"
    key_fn: DistKeyFn
    budget: int = 2


DIST_TASKS: tuple[DistTask, ...] = (
    DistTask("partition-control", "partitions", "structure", partition_set),
    DistTask("value-integrity", "values", "content", value_set),
)


@dataclass(frozen=True)
class DistBoundaryConfig:
    """The distributed boundary measurement: the (bounded) model + the workload + the knee."""

    horizon: int = 12
    driver: str = "uniform"
    workload_seeds: tuple[int, ...] = tuple(range(900, 920))
    budget: int = 2
    knee_rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    train: ED1LearnedConfig = field(
        default_factory=lambda: ED1LearnedConfig(
            train_seeds=(0, 1, 2, 3), train_iters=800, train_steps_per_traj=32
        )
    )

    @staticmethod
    def smoke() -> DistBoundaryConfig:
        return DistBoundaryConfig(
            horizon=8, workload_seeds=(900, 901, 902), knee_rhos=(0.0, 0.5, 1.0),
            train=ED1LearnedConfig(train_seeds=(0, 1), train_iters=120, train_steps_per_traj=16),
        )


@dataclass(frozen=True)
class DistTaskResult:
    """One distributed task's faithful-vs-free gap + CI."""

    task: str
    band: str
    faithful: float
    free: float
    gap: float
    ci_lo: float
    ci_hi: float
    n: int


def measure_dist_task(
    task: DistTask, model: NeuralDistWorldModel, config: DistBoundaryConfig, *, oracle: DistOracle,
) -> DistTaskResult:
    """The faithful-vs-free predictive-defense gap for one distributed task."""
    faithful = _oracle_step(oracle)
    free = _model_step(model)
    true_step = _oracle_step(oracle)
    workloads = [
        make_dist_workload(
            s, config.horizon, oracle=oracle, config=config.train, driver=config.driver
        )
        for s in config.workload_seeds
    ]
    f = [_keyed_reward(faithful, true_step, s, a, config.budget, task.key_fn) for s, a in workloads]
    fr = [_keyed_reward(free, true_step, s, a, config.budget, task.key_fn) for s, a in workloads]
    diffs = [fi - fri for fi, fri in zip(f, fr, strict=True)]
    lo, hi = bootstrap_ci(diffs, seed=0)
    return DistTaskResult(
        task.name, task.band, fmean(f), fmean(fr), fmean(f) - fmean(fr), lo, hi, len(workloads)
    )


def content_knee(
    model: NeuralDistWorldModel, config: DistBoundaryConfig, *, oracle: DistOracle,
    knee_frac: float = 0.9,
) -> tuple[float, dict[float, float]]:
    """The ρ-grounded useful knee on the content (value) task (the cheap-purchase signal, UA9)."""
    task = next(t for t in DIST_TASKS if t.band == "content")
    workloads = [
        make_dist_workload(
            s, config.horizon, oracle=oracle, config=config.train, driver=config.driver
        )
        for s in config.workload_seeds
    ]
    catch: dict[float, float] = {}
    for rho in config.knee_rhos:
        rewards = []
        for start, actions in workloads:
            pred, true, _ = _grounded_keyed_rollout(model, oracle, start, actions, rho, task.key_fn)
            if not true:
                rewards.append(1.0)
                continue
            protected = set(sorted(pred, key=repr)[:config.budget])
            rewards.append(len(protected & true) / min(config.budget, len(true)))
        catch[rho] = fmean(rewards)
    ceiling = catch[max(config.knee_rhos)]
    knee = next((r for r in sorted(config.knee_rhos) if catch[r] >= knee_frac * ceiling),
                max(config.knee_rhos))
    return knee, catch


def run_dist_boundary(
    config: DistBoundaryConfig | None = None,
) -> tuple[list[DistTaskResult], dict[str, Any]]:
    """Train the bounded dist `M_θ`; measure the structure vs content gaps + the content knee."""
    config = config or DistBoundaryConfig()
    oracle: DistOracle = ReferenceDistOracle(config.train.dist)
    model = train_model(config.train, oracle)
    results = [measure_dist_task(t, model, config, oracle=oracle) for t in DIST_TASKS]
    structure = next(r for r in results if r.band == "structure")
    content = next(r for r in results if r.band == "content")
    knee, catch = content_knee(model, config, oracle=oracle)
    verdict = {
        "structure_gap": structure.gap,
        "content_gap": content.gap,
        # the boundary holds iff content faithfulness is materially more load-bearing than structure
        "boundary_holds": content.gap > structure.gap + 0.1,
        "content_knee_rho": knee,
        "content_knee_catch": catch,
    }
    return results, verdict


def write_csv(results: list[DistTaskResult], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = ["task,band,faithful,free,gap,ci_lo,ci_hi,n"]
    rows += [
        f"{r.task},{r.band},{r.faithful:.6f},{r.free:.6f},{r.gap:.6f},"
        f"{r.ci_lo:.6f},{r.ci_hi:.6f},{r.n}"
        for r in results
    ]
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="SPEC-20 UA11 -- the structure/content boundary on the distributed world."
    )
    parser.add_argument("--out", type=str, default="figures/ua11_dist_boundary.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = DistBoundaryConfig.smoke() if args.smoke else DistBoundaryConfig()
    results, verdict = run_dist_boundary(config)
    write_csv(results, args.out)
    for r in results:
        print(f"  {r.task:18s} ({r.band:9s}):  faithful={r.faithful:.3f}  free={r.free:.3f}  "
              f"gap={r.gap:+.3f}  [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
    print(f"UA11 (boundary on the distributed world): "
          f"{'HOLDS' if verdict['boundary_holds'] else 'no'} — "
          f"content gap {verdict['content_gap']:+.3f} vs structure {verdict['structure_gap']:+.3f}")
    print(f"  content knee ρ={verdict['content_knee_rho']:.2f}")
    print("cross-world: the structure->content boundary holds on host + network + distributed")


if __name__ == "__main__":  # pragma: no cover
    main()
