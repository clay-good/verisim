"""Experiment EN1 -- the network ``H_ε(ρ)`` curve (SPEC-5 §12, milestone NW6).

The network analogue of v0's E1 (:mod:`verisim.experiments.e1`) and **SPEC-5's prime
directive** (§0): plot the faithful-horizon-vs-consultation-budget curve once, cleanly, in a
world hard enough that the interior is informative -- and report honestly what it shows
(H8, §10.1). The machinery is v0's verbatim; only the world is harder.

It trains one flat network ``M_θ`` (NW4) on seeded oracle rollouts, then sweeps consultation
budget ``ρ`` x tolerance ``ε`` x difficulty x seed, running the NW5 partial-observation loop
(:func:`verisim.netloop.run_net_rollout`) in **full-consultation** mode (so ``ρ`` means the
same thing it did in v0 and the curve is directly comparable). One :class:`RunRecord` is
emitted per rollout; the aggregated curve + figure come from the records only
(``verisim.metrics.aggregate``), so everything regenerates from config + seeds.

Note (SPEC-5 §17.5 / §0): whether the swept difficulties make ``ρ=0`` drift informative
without being pathological is exactly the empirical tuning this experiment exists to do; the
committed config is a small, fast instance of the machinery, not a tuned publication run.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.record import RunRecord, write_records
from verisim.net.action import NetAction
from verisim.net.config import DEFAULT_NET_CONFIG, NetConfig
from verisim.net.state import NetworkState
from verisim.netdata import NetDriver
from verisim.netloop import PartialNetOracle, budget_for_rho, run_net_rollout
from verisim.netmodel import NetVocab, NeuralNetworkWorldModel, build_net_dataset
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle


@dataclass(frozen=True)
class EN1Config:
    name: str = "en1-small"
    # training
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 256
    train_iters: int = 600
    lr: float = 3e-3
    model_seed: int = 0
    # evaluation (difficulty name -> driver)
    difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "weighted", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    # sweep
    rhos: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    epsilons: tuple[float, ...] = (0.0, 0.05, 0.1)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EN1Config:
        base = EN1Config()
        return EN1Config(
            name=d.get("name", base.name),
            train_driver=d.get("train_driver", base.train_driver),
            train_seeds=tuple(d.get("train_seeds", base.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", base.train_steps_per_traj),
            n_layer=d.get("n_layer", base.n_layer),
            n_head=d.get("n_head", base.n_head),
            n_embd=d.get("n_embd", base.n_embd),
            block_size=d.get("block_size", base.block_size),
            train_iters=d.get("train_iters", base.train_iters),
            lr=d.get("lr", base.lr),
            model_seed=d.get("model_seed", base.model_seed),
            difficulties=dict(d.get("difficulties", base.difficulties)),
            eval_seeds=tuple(d.get("eval_seeds", base.eval_seeds)),
            eval_steps=d.get("eval_steps", base.eval_steps),
            rhos=tuple(d.get("rhos", base.rhos)),
            epsilons=tuple(d.get("epsilons", base.epsilons)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EN1Config:
        return EN1Config.from_dict(json.loads(Path(path).read_text()))


def eval_actions(
    oracle: NetOracle, config: NetConfig, driver: str, seed: int, n_steps: int
) -> list[NetAction]:
    """A seeded action sequence by rolling a driver against the oracle (the eval rollout)."""
    driver_obj = NetDriver(name=driver, config=config, rng=random.Random(seed))
    state = NetworkState.initial(config.hosts)
    actions: list[NetAction] = []
    for _ in range(n_steps):
        action = driver_obj.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def train_model(config: EN1Config, vocab: NetVocab, oracle: NetOracle, net: NetConfig) -> Any:
    """Train the flat network ``M_θ`` -- process-reproducibly, as v0's E1 does."""
    import torch

    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_supervised

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)  # process-reproducibility (SPEC-2 §12); see E1's note

    examples = build_net_dataset(
        oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=config.block_size,
            n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
        )
    )
    train_supervised(
        model, examples, vocab.pad, steps=config.train_iters, lr=config.lr, seed=config.model_seed
    )
    return model


def run_en1(config: EN1Config | None = None, *, oracle: NetOracle | None = None) -> list[RunRecord]:
    """Train the model and run the full sweep; return one record per (cell, ε)."""
    config = config or EN1Config()
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)
    model = train_model(config, vocab, oracle, net)
    world_model = NeuralNetworkWorldModel(model, vocab)
    partial = PartialNetOracle(oracle)

    records: list[RunRecord] = []
    for difficulty, driver in config.difficulties.items():
        for seed in config.eval_seeds:
            actions = eval_actions(oracle, net, driver, seed, config.eval_steps)
            for rho in config.rhos:
                # ε does not affect loop dynamics (only H_ε), so run the rollout once per
                # (difficulty, seed, ρ) and spin out per-ε records (v0's E1 convention).
                rollout = run_net_rollout(
                    world_model, partial, NetworkState.initial(net.hosts), actions,
                    fixed_interval_for_rho(rho), epsilon=config.epsilons[0],
                    budget=budget_for_rho(rho, len(actions)), seed=seed,
                )
                for epsilon in config.epsilons:
                    records.append(
                        RunRecord(
                            config={
                                "experiment": config.name,
                                "model": "neural",
                                "difficulty": difficulty,
                                "driver": driver,
                                "rho": rho,
                                "n_steps": len(actions),
                                "oracle_bits": rollout.config["oracle_bits"],
                            },
                            seed=seed,
                            epsilon=epsilon,
                            divergences=list(rollout.divergences),
                            consultation_schedule=list(rollout.consultation_schedule),
                        )
                    )
    return records


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment EN1 (network H_eps(rho) curve).")
    parser.add_argument("--config", type=str, default=None, help="path to an EN1 config JSON")
    parser.add_argument("--out", type=str, default="runs/en1/records.jsonl")
    args = parser.parse_args()
    config = EN1Config.from_json_file(args.config) if args.config else EN1Config()
    records = run_en1(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
