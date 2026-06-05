"""ED4-fault -- H21: does fault-injected training beat fault-free under fault? (SPEC-7 §10.2, DS7).

The **H21 / DST-BUGGIFY arm** of ED4 (SPEC-7 §12): the FoundationDB/TigerBeetle tradition holds that
a model exposed to **seeded faults** in training (partitions, crashes, the recoveries between them)
is more faithful *under fault* than one trained on equal-volume fault-free data -- because the fault
dynamics (a write stranded across a partition, a stale read until heal+advance) are not implied by
the fault-free transitions. SPEC-7 turns the deterministic-simulation tester into a **data factory**
(§2.1), so this is directly measurable: the DS2 driver's ``fault_prob`` dial gives a fault-free
(``fault_prob=0``) and a fault-injected training set of **equal volume**, and the DS4 `M_θ` trains
on each. Both are then evaluated **under fault** through the DS5 loop.

The honest negative (SPEC-7 §10.3): fault-free training transfers to faulting rollout for free ->
the fault distribution is already implied by the fault-free dynamics, and fault injection adds
nothing for *modeling* (as opposed to testing). Reported whichever way it falls (the repo norm).

The fairness control H21 demands -- *at equal clean accuracy* -- is reported explicitly: the
teacher-forced accuracy of each model on a held-out **fault-free** set. If the two are comparable,
the horizon-under-fault gap is about fault-robustness, not overall model quality.

Two panels from two trained models -- **the fault-injection sweep** (DS7):

  - **left -- free-run `H_ε` vs eval fault-intensity**: the free-running (ρ=0, no oracle)
    faithful horizon as the eval workload's ``fault_prob`` rises, one curve per training regime.
    Free-run exposes the model itself (the loop is not correcting). The H21 signal is the **gap
    opening as faults intensify**: the fault-injected model degrades more slowly because it has
    seen partitions/crashes/recoveries; the fault-free model has not.
  - **right -- the control**: each model's clean (fault-free) teacher-forced accuracy, so the left
    gap is read against matched clean quality (H21 asks for the comparison *at equal clean
    accuracy* -- and the result is sharpest when the fault-free model is *better* on clean data yet
    *worse* under fault).

CI runs a tiny smoke instance; the committed figure comes from ``configs/ed4_fault.json``. Torch.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distloop import FixedTierPolicy, budget_for_rho, run_dist_rollout
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class ED4FaultConfig:
    name: str = "ed4-fault"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    # training -- both regimes share everything but the fault dial (equal volume, H21 control)
    train_driver: str = "uniform"
    train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    train_steps_per_traj: int = 40
    train_fault_prob: float = 0.45  # the fault-injected regime's intensity
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 512
    train_iters: int = 700
    lr: float = 3e-3
    model_seed: int = 0
    max_int: int = 256
    # evaluation -- the fault-intensity sweep, measured free-running (ρ=0 exposes the model)
    eval_fault_probs: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45)
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105, 106, 107)
    n_steps: int = 40
    rho: float = 0.0
    epsilon: float = 0.0
    clean_eval_seeds: tuple[int, ...] = (200, 201)  # for the clean-accuracy control

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED4FaultConfig:
        b = ED4FaultConfig()
        return ED4FaultConfig(
            name=d.get("name", b.name),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            train_fault_prob=d.get("train_fault_prob", b.train_fault_prob),
            n_layer=d.get("n_layer", b.n_layer),
            n_head=d.get("n_head", b.n_head),
            n_embd=d.get("n_embd", b.n_embd),
            block_size=d.get("block_size", b.block_size),
            train_iters=d.get("train_iters", b.train_iters),
            lr=d.get("lr", b.lr),
            model_seed=d.get("model_seed", b.model_seed),
            max_int=d.get("max_int", b.max_int),
            eval_fault_probs=tuple(d.get("eval_fault_probs", b.eval_fault_probs)),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            rho=d.get("rho", b.rho),
            epsilon=d.get("epsilon", b.epsilon),
            clean_eval_seeds=tuple(d.get("clean_eval_seeds", b.clean_eval_seeds)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED4FaultConfig:
        return ED4FaultConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED4FaultResult:
    """H21 deliverables: per-regime free-run-`H_ε`-vs-fault curves + the clean-accuracy control."""

    curves: dict[str, list[dict[str, float]]] = field(default_factory=dict)  # regime -> curve pts
    clean_accuracy: dict[str, float] = field(default_factory=dict)  # regime -> teacher-forced acc


REGIMES = ("fault_free", "fault_injected")


def _eval_actions(
    oracle: DistOracle, config: DistConfig, fault_prob: float, seed: int, n: int
) -> list[DistAction]:
    drv = DistDriver("uniform", config, random.Random(seed), fault_prob=fault_prob)
    state = DistributedState.initial(config)
    actions: list[DistAction] = []
    for _ in range(n):
        action = drv.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def train_models(
    config: ED4FaultConfig, oracle: DistOracle
) -> tuple[Any, Any, Any, dict[str, float]]:
    """Train the fault-free + fault-injected `M_θ` (equal volume); return both, the vocab, acc."""
    import torch

    from verisim.distmodel import (
        DistVocab,
        NeuralDistWorldModel,
        build_dist_dataset,
    )
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import teacher_forced_accuracy, train_supervised

    vocab = DistVocab(config.dist, max_int=config.max_int)

    # A held-out, fault-free set for the clean-accuracy control (same for both models).
    clean_eval = build_dist_dataset(
        oracle, vocab, config.dist, driver=config.train_driver,
        seeds=config.clean_eval_seeds, n_steps=config.train_steps_per_traj, fault_prob=0.0,
    )

    fault_probs = {"fault_free": 0.0, "fault_injected": config.train_fault_prob}
    models: dict[str, NeuralDistWorldModel] = {}
    clean_acc: dict[str, float] = {}
    for regime, fp in fault_probs.items():
        torch.manual_seed(config.model_seed)
        torch.set_num_threads(1)  # process-reproducibility (SPEC-2 §12)
        examples = build_dist_dataset(
            oracle, vocab, config.dist, driver=config.train_driver,
            seeds=config.train_seeds, n_steps=config.train_steps_per_traj, fault_prob=fp,
        )
        needed = max(len(p) + len(t) for p, t in [*examples, *clean_eval]) + 8
        model = GPT(
            GPTConfig(
                vocab_size=len(vocab), block_size=max(config.block_size, needed),
                n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
            )
        )
        train_supervised(
            model, examples, vocab.pad, steps=config.train_iters, lr=config.lr,
            seed=config.model_seed,
        )
        models[regime] = NeuralDistWorldModel(model, vocab)
        clean_acc[regime] = teacher_forced_accuracy(model, clean_eval, vocab.pad)
    return models["fault_free"], models["fault_injected"], vocab, clean_acc


def run_ed4_fault(
    config: ED4FaultConfig | None = None, *, oracle: DistOracle | None = None
) -> ED4FaultResult:
    """Train both regimes and sweep eval fault-intensity free-running: the H21 curves + control."""
    config = config or ED4FaultConfig()
    oracle = oracle or ReferenceDistOracle(config.dist)
    fault_free, fault_injected, _vocab, clean_acc = train_models(config, oracle)
    models = {"fault_free": fault_free, "fault_injected": fault_injected}
    result = ED4FaultResult(clean_accuracy=clean_acc)

    for regime, model in models.items():
        curve: list[dict[str, float]] = []
        for fp in config.eval_fault_probs:
            hs = []
            for s in config.eval_seeds:
                actions = _eval_actions(oracle, config.dist, fp, s, config.n_steps)
                record = run_dist_rollout(
                    model, oracle, DistributedState.initial(config.dist), actions,
                    fixed_interval_for_rho(config.rho), epsilon=config.epsilon, config=config.dist,
                    tier_policy=FixedTierPolicy("bit_exact"),
                    budget=budget_for_rho(config.rho, len(actions)), seed=s,
                )
                hs.append(record.faithful_horizon)
            lo, hi = bootstrap_ci([float(h) for h in hs], seed=0)
            curve.append({"fault_prob": fp, "h_eps": fmean(hs), "ci_lo": lo, "ci_hi": hi})
        result.curves[regime] = curve
    return result


CSV_HEADER = "regime,fault_prob,h_eps,ci_lo,ci_hi,clean_accuracy"


def write_csv(result: ED4FaultResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime in REGIMES:
        acc = result.clean_accuracy.get(regime, float("nan"))
        for p in result.curves.get(regime, []):
            rows.append(f"{regime},{p['fault_prob']},{p['h_eps']:.4f},"
                        f"{p['ci_lo']:.4f},{p['ci_hi']:.4f},{acc:.4f}")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run ED4-fault (H21: fault-inject vs fault-free)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed4_fault.csv")
    parser.add_argument("--plot", type=str, default="figures/ed4_fault.png")
    args = parser.parse_args()
    config = ED4FaultConfig.from_json_file(args.config) if args.config else ED4FaultConfig()
    result = run_ed4_fault(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    for regime in REGIMES:
        pts = [(p["fault_prob"], round(p["h_eps"], 1)) for p in result.curves[regime]]
        print(f"  {regime:14s} clean_acc={result.clean_accuracy[regime]:.3f}  H_eps(fault)={pts}")
    try:
        from figures.plot_ed4_fault import plot_ed4_fault

        plot_ed4_fault(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
