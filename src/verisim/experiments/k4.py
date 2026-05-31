"""Experiment K3+K4 — earn the knee (SPEC-2.1 §7-8).

K2 produced a *competent* model on the non-trivial ``structural`` world (acceptance ~0.875,
inside the K3 "competent-but-compounding" band). K3/K4 cash that in:

  - **K3 (difficulty sweet spot).** The ``max_depth`` dial (SPEC-2 §2.4, on the driver) bounds
    the per-step copy difficulty so acceptance stays in [0.7, 0.95]; the rollout length ``T``
    is set long enough that the unaided (ρ=0) faithful horizon is a small fraction of the
    ceiling — i.e. there is *room* for cheap consultation to buy horizon back.
  - **K4 (the knee).** Run the propose-verify-correct loop (the M5 runner, fixed-interval
    policy + hard_reset) across the consultation budget ``ρ`` on held-out ``structural``
    trajectories, and plot ``H_ε(ρ)`` — **the prime directive of SPEC-2.1**. The records are
    E1-compatible, so ``figures/plot_e1.py`` renders the curve directly.

A knee (interior horizon materially above the ρ=0 floor at small ρ) is the headline; a flat
curve is the honest negative (SPEC.md §9). Either way it is records-only and regenerable from
``configs/k4.json`` + seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import State
from verisim.loop.policy import fixed_interval_for_rho
from verisim.loop.runner import budget_for_rho, run_rollout
from verisim.metrics.record import RunRecord, write_records
from verisim.model.transformer import GPT, GPTConfig
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle
from verisim.train.dataset import build_dataset
from verisim.train.supervised import train_batched

from .e1 import eval_actions
from .e2 import build_policy, unaided_signals


@dataclass(frozen=True)
class K4Config:
    name: str = "k4"
    difficulty: str = "structural"  # the K2-competent world
    train_driver: str = "structural"
    train_seeds: tuple[int, ...] = tuple(range(160))
    val_seeds: tuple[int, ...] = (160, 161, 162, 163)
    train_steps_per_traj: int = 16
    max_depth: int | None = 4  # K3 difficulty dial (sweet spot)
    # model + training budget (the K2 budget that reached acceptance ~0.875)
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    block_size: int = 512
    train_steps: int = 6000
    lr: float = 3e-3
    batch_size: int = 64
    eval_interval: int = 500
    model_seed: int = 0
    # the curve
    eval_seeds: tuple[int, ...] = tuple(range(300, 312))
    eval_steps: int = 48  # T: long enough that the ρ=0 floor << ceiling (room for a knee)
    rhos: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    epsilons: tuple[float, ...] = (0.0, 0.05, 0.1)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> K4Config:
        b = K4Config()
        g = d.get
        md = g("max_depth", b.max_depth)
        return K4Config(
            name=g("name", b.name),
            difficulty=g("difficulty", b.difficulty),
            train_driver=g("train_driver", b.train_driver),
            train_seeds=tuple(g("train_seeds", b.train_seeds)),
            val_seeds=tuple(g("val_seeds", b.val_seeds)),
            train_steps_per_traj=g("train_steps_per_traj", b.train_steps_per_traj),
            max_depth=md,
            n_layer=g("n_layer", b.n_layer),
            n_head=g("n_head", b.n_head),
            n_embd=g("n_embd", b.n_embd),
            block_size=g("block_size", b.block_size),
            train_steps=g("train_steps", b.train_steps),
            lr=g("lr", b.lr),
            batch_size=g("batch_size", b.batch_size),
            eval_interval=g("eval_interval", b.eval_interval),
            model_seed=g("model_seed", b.model_seed),
            eval_seeds=tuple(g("eval_seeds", b.eval_seeds)),
            eval_steps=g("eval_steps", b.eval_steps),
            rhos=tuple(g("rhos", b.rhos)),
            epsilons=tuple(g("epsilons", b.epsilons)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> K4Config:
        return K4Config.from_dict(json.loads(Path(path).read_text()))


def train_competent(config: K4Config, vocab: Vocab, oracle: Oracle, env: EnvConfig) -> GPT:
    """Train the K2-competent structural model (depth-capped) with the batched trainer."""
    import torch

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)
    train_examples = build_dataset(
        oracle, vocab, env, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj, max_depth=config.max_depth,
    )
    val_examples = build_dataset(
        oracle, vocab, env, driver=config.difficulty, seeds=config.val_seeds,
        n_steps=config.train_steps_per_traj, max_depth=config.max_depth,
    )
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=config.block_size,
            n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
        )
    )
    train_batched(
        model, train_examples, vocab.pad, steps=config.train_steps, lr=config.lr,
        batch_size=config.batch_size, seed=config.model_seed,
        val_examples=val_examples, eval_interval=config.eval_interval,
    )
    return model


def run_k4(config: K4Config | None = None, *, oracle: Oracle | None = None) -> list[RunRecord]:
    """Train the competent model and sweep the loop over rho x eps; return E1 records."""
    config = config or K4Config()
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    model = NeuralWorldModel(train_competent(config, vocab, oracle, env), vocab)

    records: list[RunRecord] = []
    for seed in config.eval_seeds:
        actions = eval_actions(
            oracle, env, config.difficulty, seed, config.eval_steps, config.max_depth
        )
        for rho in config.rhos:
            # ε does not affect loop dynamics (only H_ε), so roll once per (seed, ρ) and
            # spin out per-ε records (the E1 discipline).
            rollout = run_rollout(
                model, oracle, State.empty(), actions, fixed_interval_for_rho(rho),
                epsilon=config.epsilons[0], budget=budget_for_rho(rho, len(actions)), seed=seed,
            )
            for epsilon in config.epsilons:
                records.append(
                    RunRecord(
                        config={
                            "experiment": config.name,
                            "model": "neural",
                            "difficulty": config.difficulty,
                            "driver": config.difficulty,
                            "rho": rho,
                            "n_steps": len(actions),
                        },
                        seed=seed,
                        epsilon=epsilon,
                        divergences=list(rollout.divergences),
                        consultation_schedule=list(rollout.consultation_schedule),
                    )
                )
    return records


def run_k4_policies(
    config: K4Config | None = None,
    *,
    oracle: Oracle | None = None,
    rho: float = 0.2,
    policies: tuple[str, ...] = ("fixed", "uncertainty", "drift"),
) -> list[RunRecord]:
    """E2 with the *competent* model: does error-targeting consultation beat fixed-interval?

    Fixed-interval consultation cannot lift `H_ε` on a discrete-error world (the K4 finding —
    one wrong edit spikes divergence past ε, and even resets cannot push out the first error).
    An *uncertainty-triggered* policy can, **if** the model's uncertainty predicts which steps
    it will err on — exactly H2. At equal budget the runner's spend-down makes every policy
    spend the same `floor(ρ·T)` calls, so any horizon gap is about *where* the budget is spent.
    """
    config = config or K4Config()
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    model = NeuralWorldModel(train_competent(config, vocab, oracle, env), vocab)

    records: list[RunRecord] = []
    for seed in config.eval_seeds:
        actions = eval_actions(
            oracle, env, config.difficulty, seed, config.eval_steps, config.max_depth
        )
        signals = unaided_signals(model, State.empty(), actions)
        budget = budget_for_rho(rho, len(actions))
        for policy_name in policies:
            policy = build_policy(policy_name, rho, signals)
            rollout = run_rollout(
                model, oracle, State.empty(), actions, policy,
                epsilon=config.epsilons[0], budget=budget, seed=seed,
            )
            for epsilon in config.epsilons:
                records.append(
                    RunRecord(
                        config={
                            "experiment": f"{config.name}-policies",
                            "policy": policy_name,
                            "difficulty": config.difficulty,
                            "rho": rho,
                            "n_steps": len(actions),
                        },
                        seed=seed,
                        epsilon=epsilon,
                        divergences=list(rollout.divergences),
                        consultation_schedule=list(rollout.consultation_schedule),
                    )
                )
    return records


def _summary(records: list[RunRecord], config: K4Config) -> str:
    """A compact H_ε(ρ) table (mean over seeds) per ε — the knee at a glance."""
    from statistics import fmean

    lines: list[str] = []
    ceiling = float(config.eval_steps)
    for epsilon in config.epsilons:
        by_rho: dict[float, list[int]] = {rho: [] for rho in config.rhos}
        for r in records:
            if r.epsilon == epsilon:
                by_rho[float(r.config["rho"])].append(r.faithful_horizon)
        cells = " ".join(
            f"ρ{rho:g}={fmean(hs):.1f}" for rho, hs in by_rho.items() if hs
        )
        floor = fmean(by_rho[0.0]) if by_rho.get(0.0) else 0.0
        knee = next(
            (rho for rho in config.rhos if rho <= 0.2 and by_rho[rho]
             and fmean(by_rho[rho]) >= 0.5 * ceiling),
            None,
        )
        verdict = "KNEE@rho<=0.2 (>=50% ceiling)" if knee is not None else "no knee <=0.2"
        lines.append(
            f"eps={epsilon:g}: {cells}  | floor={floor:.1f} ceiling={ceiling:g} -> {verdict}"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment K3+K4 (earn the knee).")
    parser.add_argument("--config", type=str, default=None, help="path to a K4 config")
    parser.add_argument("--out", type=str, default="runs/k4/records.jsonl")
    parser.add_argument("--mode", choices=["curve", "policies"], default="curve",
                        help="curve = H_eps(rho) sweep (E1); policies = fixed vs smart (E2)")
    parser.add_argument("--rho", type=float, default=0.2, help="budget for --mode policies")
    args = parser.parse_args()
    config = K4Config.from_json_file(args.config) if args.config else K4Config()
    if args.mode == "policies":
        records = run_k4_policies(config, rho=args.rho)
        path = write_records(records, args.out)
        print(f"wrote {len(records)} records to {path}")
        from statistics import fmean
        for policy in ("fixed", "uncertainty", "drift"):
            hs = [r.faithful_horizon for r in records if r.epsilon == 0.05
                  and r.config.get("policy") == policy]
            if hs:
                print(f"  policy={policy:12s} H_0.05={fmean(hs):.2f} (n={len(hs)})")
        return
    records = run_k4(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")
    print(_summary(records, config))


if __name__ == "__main__":  # pragma: no cover
    main()
