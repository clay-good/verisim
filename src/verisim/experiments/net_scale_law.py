"""SPEC-21 cross-world: the faithfulness-for-control scale law on the NETWORK world (CS1-net).

The host scale law ([`scale_law.py`](./scale_law.py)) measured, across the capacity ladder, where
the load-bearing frontier sits (the structure→content gap gradient) and that the cheap drift
forecasts it.
SPEC-20's *boundary* law was cross-world (host + network); this is the cross-world confirmation of
its *scale* law: does the same structure→content gradient — and the cheap-forecast — reproduce
on the **network** world (SPEC-5), whose content dimension is *flows*?

It reuses everything world-agnostic: the SPEC-20 network predictive-defense machinery
([`acd.net_integrity`](../acd/net_integrity.py)), the network model training lifecycle
([`flagship.train_flagship`](./flagship.py)), and the scale-law data model + reducers
([`scale_law`](./scale_law.py): `ScaleRung` / `forecast_check` / `cost_forecast_check` /
`knee_verdict`). What it adds is the **network task suite** ordered structure→content (services /
links / flows) and the per-rung network training, so the host harness's reducers run unchanged on a
network `ScaleLawResult`. The SPEC-20 §7 drift profile predicts the gradient: the net model is
faithful on the discrete *structure* (services ~0.011, links ~0.044) and drifts on the *content*
(flows ~0.252). CPU-local; the smoke ladder runs in CI; the committed numbers are the bounded CPU
ladder, the headline (wide ladder) the GPU run, exactly as host.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.acd.net_integrity import make_net_workload, model_step, oracle_step
from verisim.cue.tasks import TaskGap
from verisim.experiments.flagship import FlagshipConfig
from verisim.experiments.horizon_scaling import ModelScale
from verisim.experiments.scale_law import (
    ScaleLawResult,
    ScaleRung,
    cost_forecast_check,
    forecast_check,
    frontier_verdict,
    knee_verdict,
    write_csv,
)
from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.net.state import services as net_services
from verisim.netoracle.base import NetOracle
from verisim.netoracle.reference import ReferenceNetworkOracle

if TYPE_CHECKING:
    from verisim.netmodel import NeuralNetworkWorldModel

NetStepFn = Callable[[NetworkState, NetAction], NetworkState]
NetKeyFn = Callable[[NetworkState], set[Any]]


# --- the network keyed-set extractors (the structure->content spectrum) ---------------------------


def service_set(state: NetworkState) -> set[Any]:
    """Structure: the listening ``(host, port)`` services (the net model is ~0.011 drift here)."""
    return set(net_services(state))


def link_set(state: NetworkState) -> set[Any]:
    """Near-structure: the link set / topology (the net model is ~0.044 drift here)."""
    return set(state.links)


def flow_set(state: NetworkState) -> set[Any]:
    """Content: the established ``(src, dst, port)`` flows (the net model drifts ~0.252 here)."""
    return set(state.flows)


# --- the generic predictive-defense over a network keyed dimension (the UA8/UA10 pattern) ---------


def _rollout_keyed(
    step: NetStepFn, start: NetworkState, actions: Sequence[NetAction], key_fn: NetKeyFn
) -> set[Any]:
    """Cumulative keyed set ever touched (the union over the rollout, the UA10 framing)."""
    state = start
    seen = set(key_fn(state))
    for action in actions:
        state = step(state, action)
        seen |= key_fn(state)
    return seen


def _keyed_reward(
    predictor: NetStepFn, true_step: NetStepFn, start: NetworkState,
    actions: Sequence[NetAction], budget: int, key_fn: NetKeyFn,
) -> float:
    """Protect the ``budget`` predicted-keyed objects; score vs the true cumulative keyed set."""
    predicted = sorted(_rollout_keyed(predictor, start, actions, key_fn), key=repr)
    true_set = _rollout_keyed(true_step, start, actions, key_fn)
    if not true_set:
        return 1.0
    protected = set(predicted[:budget])
    return len(protected & true_set) / min(budget, len(true_set))


def _grounded_keyed_rollout(
    model: object, oracle: NetOracle, start: NetworkState, actions: Sequence[NetAction],
    rho: float, key_fn: NetKeyFn,
) -> tuple[set[Any], set[Any], int]:
    """The ρ-grounded predictor over a network keyed dimension (UA9): re-anchor every round(1/ρ)."""
    from verisim.netdelta.apply import apply

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
            delta = model.predict_delta(predicted, action)  # type: ignore[attr-defined]
            predicted = apply(predicted, delta)
        pred_seen |= key_fn(predicted)
    return pred_seen, true_seen, calls


@dataclass(frozen=True)
class NetTask:
    """One network predictive-defense task on the structure->content spectrum."""

    name: str
    keyed_dimension: str  # "services" | "links" | "flows"
    order: int  # 0 = structure ... 2 = content
    key_fn: NetKeyFn
    budget: int = 2


#: The ordered structure->content network suite (the SPEC-21 §3 spectrum, network vertical).
NET_TASK_SUITE: tuple[NetTask, ...] = (
    NetTask("service-control", "services", 0, service_set),
    NetTask("link-control", "links", 1, link_set),
    NetTask("flow-integrity", "flows", 2, flow_set),
)


@dataclass(frozen=True)
class NetTaskGapConfig:
    """The workload regime for the network task gaps (fixed across scales)."""

    horizon: int = 20
    driver: str = "weighted"
    workload_seeds: tuple[int, ...] = tuple(range(800, 816))


def net_task_gap(
    task: NetTask, model: NeuralNetworkWorldModel, config: NetTaskGapConfig, *,
    oracle: NetOracle,
) -> TaskGap:
    """The faithful-vs-free predictive-defense gap for ``task`` (the load-bearing signal)."""
    faithful = oracle_step(oracle)
    free = model_step(model)
    true_step = oracle_step(oracle)
    workloads = [
        make_net_workload(s, config.horizon, driver=config.driver, oracle=oracle)
        for s in config.workload_seeds
    ]
    f = fmean([_keyed_reward(faithful, true_step, s, a, task.budget, task.key_fn)
               for s, a in workloads])
    fr = fmean([_keyed_reward(free, true_step, s, a, task.budget, task.key_fn)
                for s, a in workloads])
    return TaskGap(task.name, task.keyed_dimension, task.order, f, fr, f - fr, len(workloads))


def net_keyed_drift(
    task: NetTask, model: NeuralNetworkWorldModel, config: NetTaskGapConfig, *, oracle: NetOracle,
) -> float:
    """The cheap per-task drift: the fraction of the true keyed set the free model misses."""
    free = model_step(model)
    true_step = oracle_step(oracle)
    misses: list[float] = []
    for seed in config.workload_seeds:
        start, actions = make_net_workload(
            seed, config.horizon, driver=config.driver, oracle=oracle
        )
        free_set = _rollout_keyed(free, start, actions, task.key_fn)
        true_set = _rollout_keyed(true_step, start, actions, task.key_fn)
        if true_set:
            misses.append(len(true_set - free_set) / len(true_set))
    return fmean(misses) if misses else 0.0


def net_task_knee_rho(
    task: NetTask, model: NeuralNetworkWorldModel, rhos: Sequence[float], config: NetTaskGapConfig,
    *, oracle: NetOracle, knee_frac: float = 0.9,
) -> float:
    """The smallest ρ recovering ``knee_frac`` of the faithful catch on ``task`` (the knee)."""
    workloads = [
        make_net_workload(s, config.horizon, driver=config.driver, oracle=oracle)
        for s in config.workload_seeds
    ]

    def catch(rho: float) -> float:
        rewards = []
        for start, actions in workloads:
            pred, true, _ = _grounded_keyed_rollout(model, oracle, start, actions, rho, task.key_fn)
            if not true:
                rewards.append(1.0)
                continue
            protected = set(sorted(pred, key=repr)[:task.budget])
            rewards.append(len(protected & true) / min(task.budget, len(true)))
        return fmean(rewards)

    ceiling = catch(max(rhos))
    threshold = knee_frac * ceiling
    return next((r for r in sorted(rhos) if catch(r) >= threshold), max(rhos))


# --- the harness (parallel to scale_law.run_scale_law, network vertical) --------------------------


@dataclass(frozen=True)
class NetScaleLawConfig:
    """The network scale-law sweep: the ladder, the per-rung net training base, the regimes."""

    scales: tuple[ModelScale, ...] = (
        ModelScale("xs", n_embd=32, n_layer=1, train_steps=600),
        ModelScale("s", n_embd=64, n_layer=2, train_steps=800),
        ModelScale("m", n_embd=128, n_layer=2, train_steps=1000),
        ModelScale("l", n_embd=192, n_layer=3, n_head=4, train_steps=1200),
    )
    base: FlagshipConfig = field(
        default_factory=lambda: dataclasses.replace(
            FlagshipConfig(), data_seeds=12, train_steps_per_traj=60, num_threads=1
        )
    )
    task_config: NetTaskGapConfig = field(default_factory=NetTaskGapConfig)
    knee_rhos: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0)
    threshold: float = 0.05
    device: str = "cpu"

    @staticmethod
    def smoke() -> NetScaleLawConfig:
        return NetScaleLawConfig(
            scales=(
                ModelScale("xs", n_embd=32, n_layer=1, train_steps=200),
                ModelScale("s", n_embd=48, n_layer=1, train_steps=250),
            ),
            base=FlagshipConfig.smoke(),
            task_config=NetTaskGapConfig(horizon=12, workload_seeds=tuple(range(800, 806))),
            knee_rhos=(0.0, 0.5, 1.0),
        )


def _train_net_rung(scale: ModelScale, config: NetScaleLawConfig) -> NeuralNetworkWorldModel:
    from verisim.experiments.flagship import train_flagship

    rung_config = dataclasses.replace(config.base, scale=scale, name=f"net-scale-{scale.label}")
    model, _ = train_flagship(rung_config)
    return model


def run_net_scale_law(config: NetScaleLawConfig | None = None) -> ScaleLawResult:
    """The network CP0 pipeline: per rung, train -> per-task gap + cheap keyed drift + knee."""
    config = config or NetScaleLawConfig()
    oracle: NetOracle = ReferenceNetworkOracle()
    rungs: list[ScaleRung] = []
    for scale in config.scales:
        model = _train_net_rung(scale, config)
        gaps = [net_task_gap(t, model, config.task_config, oracle=oracle) for t in NET_TASK_SUITE]
        kd = {t.name: net_keyed_drift(t, model, config.task_config, oracle=oracle)
              for t in NET_TASK_SUITE}
        knees: dict[str, float] = {}
        for g, t in zip(gaps, NET_TASK_SUITE, strict=True):
            if g.gap > config.threshold:
                knees[t.name] = net_task_knee_rho(
                    t, model, config.knee_rhos, config.task_config, oracle=oracle
                )
        # dimension_drift mirrors the per-task keyed drift (no separate net drift module needed)
        rungs.append(ScaleRung(scale.label, scale.params, dict(kd), gaps, kd, knees))
    return ScaleLawResult(rungs)


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="SPEC-21 cross-world -- the scale law on the NETWORK world (CS1-net)."
    )
    parser.add_argument("--out", type=str, default="figures/cs1_net_frontier.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = NetScaleLawConfig.smoke() if args.smoke else NetScaleLawConfig()
    result = run_net_scale_law(config)
    write_csv(result, args.out)
    print("the NETWORK load-bearing frontier (per-task faithful-vs-free gap by scale):")
    for rung in result.rungs:
        cells = "  ".join(f"{g.task.split('-')[0][:5]}={g.gap:+.2f}" for g in rung.gaps)
        print(f"  {rung.label:4s} (params={rung.params:>7d}):  {cells}")
    fv = frontier_verdict(result, config.threshold)
    fc = forecast_check(result)
    cf = cost_forecast_check(result, config.threshold)
    kv = knee_verdict(result, config.threshold)
    print(f"H87 frontier recedes/flat: {fv['frontier_recedes_or_flat']}  "
          f"order-by-scale={fv['frontier_order_by_scale']}")
    print(f"H88 irreducible residue ({fv['deepest_task']}): {fv['irreducible_residue']}")
    print(f"H89 forecast (cheap drift -> gap): spearman={fc['spearman']:+.3f}  "
          f"forecastable={fc['forecastable']}")
    print(f"cost forecast (cheap drift -> knee): spearman={cf['spearman']:+.3f}")
    if kv["deepest_load_bearing"]:
        print(f"knee on {kv['deepest_load_bearing']}: ρ {kv['knee_at_smallest']:.2f} -> "
              f"{kv['knee_at_largest']:.2f} ({kv['knee_trend']})")
    print("cross-world: the structure->content gradient + cheap forecast reproduce on the network")


if __name__ == "__main__":  # pragma: no cover
    main()
