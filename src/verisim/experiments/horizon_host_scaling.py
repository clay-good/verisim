"""HS2 -- the faithful-horizon scaling law, re-run on the *host* world (SPEC-10 §5, universality).

HS1 (:mod:`verisim.experiments.horizon_scaling`) measured, in the **network** world, whether free-
running faithful horizon `H_ε(ρ=0)` grows with model capacity, and found the floor+cliff was in
substantial part an under-resourcing artifact. HS1's own honest caveat (SPEC-10 §4.1) named the next
lever explicitly: *world difficulty* -- a harder world should re-lower the floor and re-open the
question. HS2 runs that test by sweeping the **identical capacity axis** on the **host** world
(SPEC-6: the composed process/fd/filesystem/exit bundle, strictly richer per-step dynamics than the
network world), holding the world, oracle, grammar, drivers, and loop fixed.

The measurement is HS1 verbatim, only the world swapped:

  - **one-step acceptance `p`** -- the teacher-forced fraction of steps whose predicted bundle delta
    is *exactly* the oracle's (host :func:`~verisim.hostmetrics.bits.delta_exact`);
  - **free-running horizon `H_free` = `H_ε(ρ=0)`** -- steps the composed host model self-rolls
    (:func:`~verisim.hostloop.run_host_rollout` at ρ=0) before any subsystem diverges;
  - **independence prediction `H_indep = p/(1-p)`** and **horizon efficiency `η = H_free/H_indep`**
    -- the scale-free compounding penalty, reused unchanged from HS1.

The question HS2 answers: does HS1's verdict -- *capacity lifts the free-running horizon, and η
stays above the i.i.d. line (no compounding wall at this world's single-machine scale)* --
**survive a harder world**, or does the richer host dynamics re-lower the floor and expose the
compounding wall the network world did not reach? Either branch is bankable (SPEC.md §10.1): a
confirmed lift makes the HS1 result a cross-world property of the oracle loop; a re-lowered floor
localizes the wall to world difficulty, exactly the §4.1 prediction.

The committed sweep is local (CPU, the SPEC-9 envelope discipline); CI runs only the smoke instance.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.state import HostState
from verisim.hostdata import HostDriver
from verisim.hostloop import PartialHostOracle, budget_for_rho, run_host_rollout
from verisim.hostmetrics.bits import delta_exact
from verisim.hostmodel import HostVocab, NeuralHostWorldModel, build_host_dataset
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon

# Reuse HS1's capacity-axis types, the independence math, the per-cell stat, the CSV writer, and
# the metric grid verbatim -- HS2 is the *same* measurement on a different world, so the harness
# shape is shared and only the world plumbing below differs (the SPEC-10 §5 universality habit).
from .horizon_scaling import (
    CSV_HEADER,
    METRICS,
    ModelScale,
    ScaleStat,
    independence_horizon,
    write_csv,
)

__all__ = [
    "CSV_HEADER",
    "DEFAULT_SCALES",
    "METRICS",
    "HostHorizonScalingConfig",
    "run_host_horizon_scaling",
    "write_csv",
]

# The same wide, CPU-feasible default capacity axis HS1 uses (~100x params, the SPEC-9 envelope),
# so the host curve is read on the *identical* x-axis as the network one (the universality compare).
DEFAULT_SCALES: tuple[ModelScale, ...] = (
    ModelScale("xs", n_embd=32, n_layer=1, train_steps=2000),
    ModelScale("s", n_embd=64, n_layer=2, train_steps=2500),
    ModelScale("m", n_embd=128, n_layer=2, train_steps=3000),
    ModelScale("l", n_embd=192, n_layer=3, n_head=4, train_steps=3500),
)


@dataclass(frozen=True)
class HostHorizonScalingConfig:
    """HS2 config: the host analogue of :class:`HorizonScalingConfig`, no EN1 base needed.

    The shared coverage set is built once from ``train_driver``/``train_seeds`` (the oracle's labels
    are free, SPEC-9), so the only thing that varies down the capacity axis is the model.
    """

    name: str = "hs2-host"
    scales: tuple[ModelScale, ...] = DEFAULT_SCALES
    seeds: tuple[int, ...] = (0, 1, 2)  # one freshly-trained model per seed; CIs are over seeds
    # the shared coverage set (the host EH1 driver/lengths)
    train_driver: str = "forky"
    train_seeds: tuple[int, ...] = tuple(range(16))
    train_steps_per_traj: int = 40
    block_size: int = 256  # decode window (host bundle deltas are longer than the network's)
    batch_size: int = 64  # minibatch SGD (train_batched) -- constant per-step cost (SPEC-2.1 §6)
    lr: float = 3e-3
    num_threads: int = 1  # 1 = bit-deterministic (the repo default); 0 = use all cores (the sweep)
    eval_driver: str = "forky"  # in-distribution (id) free-running -- the trained regime
    eval_driver_hard: str = "adversarial"  # harder/out-of-distribution (ood) regime
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    eval_steps: int = 48  # horizon cap; headroom above the largest model's free-running horizon
    one_step_seeds: tuple[int, ...] = (200, 201)  # held-out (state, action) for the p measurement
    one_step_steps: int = 40
    epsilon: float = 0.0  # exact-match acceptance / horizon (the strictest, cleanest reading)
    verbose: bool = False  # print per-cell progress (the long local sweep; off in CI/tests)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> HostHorizonScalingConfig:
        b = HostHorizonScalingConfig()
        scales = (
            tuple(
                ModelScale(
                    label=s["label"], n_embd=s["n_embd"], n_layer=s["n_layer"],
                    n_head=s.get("n_head", 2), train_steps=s.get("train_steps", 2500),
                )
                for s in d["scales"]
            )
            if "scales" in d
            else b.scales
        )
        return HostHorizonScalingConfig(
            name=d.get("name", b.name),
            scales=scales,
            seeds=tuple(d.get("seeds", b.seeds)),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            block_size=d.get("block_size", b.block_size),
            batch_size=d.get("batch_size", b.batch_size),
            lr=d.get("lr", b.lr),
            num_threads=d.get("num_threads", b.num_threads),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_driver_hard=d.get("eval_driver_hard", b.eval_driver_hard),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            epsilon=d.get("epsilon", b.epsilon),
            verbose=d.get("verbose", b.verbose),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> HostHorizonScalingConfig:
        return HostHorizonScalingConfig.from_dict(json.loads(Path(path).read_text()))


def _eval_actions(
    oracle: HostOracle, config: HostConfig, driver: str, seed: int, n_steps: int
) -> list[HostAction]:
    """A seeded action sequence by rolling a workload driver against the oracle (eval rollout)."""
    driver_obj = HostDriver(name=driver, config=config, rng=random.Random(seed))
    state = HostState.initial()
    actions: list[HostAction] = []
    for _ in range(n_steps):
        action = driver_obj.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def _eval_triples(
    oracle: HostOracle, host: HostConfig, seeds: tuple[int, ...], n_steps: int, driver: str
) -> list[tuple[HostState, HostAction, Any]]:
    """Seeded held-out ``(state, action, true_delta)`` triples for the one-step `p` measurement."""
    triples: list[tuple[HostState, HostAction, Any]] = []
    for seed in seeds:
        drv = HostDriver(name=driver, config=host, rng=random.Random(seed))
        state = HostState.initial()
        for _ in range(n_steps):
            action = drv.sample(state)
            result = oracle.step(state, action)
            triples.append((state, action, result.delta))
            state = result.state
    return triples


def _train_scaled(
    config: HostHorizonScalingConfig, scale: ModelScale, seed: int, vocab: HostVocab,
    examples: list[Any],
) -> NeuralHostWorldModel:
    """Build + minibatch-train one flat host ``M_θ`` at ``scale``/``seed`` (the SPEC-2.1 K2 loop).

    Identical to HS1's trainer but over the host vocab/dataset: ``train_batched`` keeps the per-step
    cost constant so the free large coverage set is affordable (§2). ``num_threads`` governs
    determinism (1 = bit-deterministic).
    """
    import torch

    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_batched

    if config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
    torch.manual_seed(seed)
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=config.block_size,
            n_layer=scale.n_layer, n_head=scale.n_head, n_embd=scale.n_embd,
        )
    )
    train_batched(
        model, examples, vocab.pad, steps=scale.train_steps, lr=config.lr,
        batch_size=config.batch_size, seed=seed,
    )
    return NeuralHostWorldModel(model, vocab)


def _delta_exact_rate(
    pairs: list[tuple[Any, Any]],
) -> float:
    """Fraction of ``(predicted_delta, true_delta)`` host bundle pairs that match exactly."""
    total = 0
    exact = 0
    for predicted, true in pairs:
        total += 1
        exact += delta_exact(predicted, true)
    return exact / total if total else 1.0


def _evaluate(
    world_model: NeuralHostWorldModel, config: HostHorizonScalingConfig, oracle: HostOracle,
    host: HostConfig, driver: str,
) -> dict[str, float]:
    """One trained model's `p`, free-running `H_ε(ρ=0)`, the independence baseline, and `η`."""
    triples = _eval_triples(oracle, host, config.one_step_seeds, config.one_step_steps, driver)
    p = _delta_exact_rate(
        [(world_model.predict_delta(s, a), true) for s, a, true in triples]
    )

    partial = PartialHostOracle(oracle)
    horizons: list[float] = []
    for eseed in config.eval_seeds:
        actions = _eval_actions(oracle, host, driver, eseed, config.eval_steps)
        record = run_host_rollout(
            world_model, partial, HostState.initial(), actions,
            fixed_interval_for_rho(0.0), epsilon=config.epsilon,
            budget=budget_for_rho(0.0, len(actions)), seed=eseed,
        )
        horizons.append(float(faithful_horizon(list(record.divergences), config.epsilon)))

    h_free = fmean(horizons)
    h_indep = independence_horizon(p, cap=float(config.eval_steps))
    return {
        "one_step_acc": p,
        "h_free": h_free,
        "h_indep": h_indep,
        "horizon_efficiency": (h_free / h_indep) if h_indep > 0 else 0.0,
    }


def _cell(
    config: HostHorizonScalingConfig, scale: ModelScale, seed: int, oracle: HostOracle,
    host: HostConfig, vocab: HostVocab, examples: list[Any],
) -> dict[str, float]:
    """Train one model at ``scale``/``seed``; evaluate it in-distribution (id) and harder (ood)."""
    world_model = _train_scaled(config, scale, seed, vocab, examples)
    out: dict[str, float] = {}
    for regime, driver in (("id", config.eval_driver), ("ood", config.eval_driver_hard)):
        for base, value in _evaluate(world_model, config, oracle, host, driver).items():
            out[f"{base}_{regime}"] = value
    return out


def run_host_horizon_scaling(
    config: HostHorizonScalingConfig | None = None, *, oracle: HostOracle | None = None
) -> list[ScaleStat]:
    """Sweep the capacity axis on the host world; reduce each metric over seeds to mean + CI."""
    config = config or HostHorizonScalingConfig()
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host)
    # Build the coverage set ONCE -- the oracle's labels are free (SPEC-9), shared across all cells.
    examples = build_host_dataset(
        oracle, vocab, host, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )

    stats: list[ScaleStat] = []
    for scale in config.scales:
        per_seed = []
        for seed in config.seeds:
            cell = _cell(config, scale, seed, oracle, host, vocab, examples)
            per_seed.append(cell)
            if config.verbose:  # pragma: no cover - progress for the long local sweep
                print(
                    f"  [{scale.label} N={scale.params} seed={seed}] "
                    f"p={cell['one_step_acc_id']:.3f} H_free={cell['h_free_id']:.2f} "
                    f"(ood H_free={cell['h_free_ood']:.2f})",
                    flush=True,
                )
        for metric in METRICS:
            values = [c[metric] for c in per_seed]
            lo, hi = bootstrap_ci(values, seed=0)
            stats.append(
                ScaleStat(scale.label, scale.params, metric, fmean(values), lo, hi, len(values))
            )
    return stats


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run HS2 (faithful-horizon scaling, host world).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/horizon_host_scaling.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_host_scaling.png")
    args = parser.parse_args()
    config = (
        HostHorizonScalingConfig.from_json_file(args.config)
        if args.config
        else HostHorizonScalingConfig()
    )
    stats = run_host_horizon_scaling(config)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    by_scale: dict[str, dict[str, float]] = {}
    for s in stats:
        by_scale.setdefault(s.scale, {"params": float(s.params)})[s.metric] = s.mean
    for label in sorted(by_scale, key=lambda k: by_scale[k]["params"]):
        d = by_scale[label]
        print(f"  {label:3s} N={int(d['params']):>8d}  "
              f"[id]  p={d['one_step_acc_id']:.3f} H_free={d['h_free_id']:.2f} "
              f"H_indep={d['h_indep_id']:.2f} eta={d['horizon_efficiency_id']:.2f}   "
              f"[ood] p={d['one_step_acc_ood']:.3f} H_free={d['h_free_ood']:.2f} "
              f"eta={d['horizon_efficiency_ood']:.2f}")
    try:
        from figures.plot_horizon_scaling import plot_horizon_scaling

        plot_horizon_scaling(
            stats, args.plot,
            suptitle="HS2 — the faithful-horizon scaling law on the HOST world (SPEC-10, H26)",
            left_title="Does capacity lift the free-running horizon in a harder world?",
        )
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
