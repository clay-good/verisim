"""Experiment EH1 -- the composed-host ``H_ε(ρ)`` curve + the composition law (SPEC-6 §0, HC6).

**SPEC-6's prime directive** (§0): plot the faithful-horizon-vs-consultation-budget curve once,
cleanly, in the composed host world -- and, the headline-new question, measure the **composition
law (H13, §9.2)**: is whole-machine faithfulness predictable from the faithfulness of its parts
(multiplicative ↔ weakest-link ↔ coupled)? The curve machinery is v0's E1 / network EN1 verbatim
(the loop, the records, the bootstrap-CI aggregation); only the world is the coupled bundle, and the
composition law is the new object only this world can ask.

It trains one flat host ``M_θ`` (HC4) on seeded oracle rollouts, then sweeps consultation budget
``ρ`` x tolerance ``ε`` x difficulty x seed, running the HC5 composed loop
(:func:`verisim.hostloop.run_host_rollout`) in **full-consultation** mode (so ``ρ`` means what it
did in v0/EN1 and the curve is directly comparable). One :class:`HostRunRecord` per rollout carries
the composed *and* per-subsystem divergence trajectories; the composed curve and the
composition-law verdict are both read off those records, so everything regenerates from config +
seeds. The composition law is read at ``ρ=0`` (the pure-model regime, where the model's own
per-subsystem acceptance shows -- corrections at ``ρ>0`` would mask it).

Note (SPEC-6 §0): whether the swept difficulties make ``ρ=0`` drift informative without being
pathological is exactly the empirical tuning this experiment exists to do; the committed config is a
small, fast instance of the machinery, not a tuned publication run.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.delta import apply
from verisim.host.state import HostState
from verisim.hostdata import HostDriver
from verisim.hostloop import PartialHostOracle, budget_for_rho, run_host_rollout
from verisim.hostloop.model import HostModel
from verisim.hostmetrics.composition import CompositionLaw, composition_law
from verisim.hostmetrics.divergence import step_faithful_by_subsystem
from verisim.hostmetrics.record import HostRunRecord
from verisim.hostmodel import HostVocab, NeuralHostWorldModel, build_host_dataset
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import fixed_interval_for_rho


@dataclass(frozen=True)
class EH1Config:
    name: str = "eh1-small"
    # training
    train_driver: str = "forky"
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
        default_factory=lambda: {"low": "forky", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    # sweep
    rhos: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    epsilons: tuple[float, ...] = (0.0, 0.05, 0.1)
    # The tolerance the composition law (H13) is read at. It is a *per-step* (teacher-forced)
    # acceptance, not a free-run cumulative one (SPEC-6 §9.2: the speculative-decoding view), so
    # compounding subsystems are not permanently penalized by one early drift.
    composition_epsilon: float = 0.05

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH1Config:
        base = EH1Config()
        return EH1Config(
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
            composition_epsilon=d.get("composition_epsilon", base.composition_epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH1Config:
        return EH1Config.from_dict(json.loads(Path(path).read_text()))


def eval_actions(
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


def train_model(config: EH1Config, vocab: HostVocab, oracle: HostOracle, host: HostConfig) -> Any:
    """Train the flat host ``M_θ`` (HC4) -- process-reproducibly, as v0's E1 / EN1 do."""
    import torch

    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_supervised

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)  # process-reproducibility (SPEC-2 §12); see E1's note

    examples = build_host_dataset(
        oracle, vocab, host, driver=config.train_driver, seeds=config.train_seeds,
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


@dataclass
class EH1Result:
    """The EH1 deliverables: the composed ``H_ε(ρ)`` curve records + the composition-law verdict."""

    records: list[HostRunRecord]  # one per (difficulty, seed, ρ, ε) -- the curve source
    composition: dict[str, CompositionLaw]  # per difficulty -- the H13 verdict (teacher-forced)


def teacher_forced_faithful(
    model: HostModel, oracle: HostOracle, actions: list[HostAction], epsilon: float
) -> list[dict[str, bool]]:
    """Per-step, per-subsystem faithfulness of the model's **one-step** prediction (SPEC-6 §9.2).

    The composition-law acceptance is a *per-step* quantity (the speculative-decoding view, §9.2),
    so it is measured teacher-forced: at each step the model predicts the bundle delta from the
    **true** current state, and each subsystem is faithful iff its one-step divergence is ``≤ ε``.
    This is the honest ``a_i`` -- not contaminated by the free-run drift that would permanently sink
    a compounding subsystem after a single early error (making acceptance bimodal, not a rate). The
    state advances on ground truth (teacher forcing), so every step is a clean probe.
    """
    state = HostState.initial()
    steps: list[dict[str, bool]] = []
    for action in actions:
        predicted = apply(state, model.predict_delta(state, action))
        truth = oracle.step(state, action).state
        steps.append(step_faithful_by_subsystem(truth, predicted, epsilon))
        state = truth
    return steps


def run_eh1(config: EH1Config | None = None, *, oracle: HostOracle | None = None) -> EH1Result:
    """Train the model, run the full sweep, and measure the composition law (H13).

    The records carry the composed *and* per-subsystem divergence trajectories, so the composed
    ``H_ε(ρ)`` curve is derivable from them alone (the SPEC-2 §7.3 figures-from-records discipline).
    The composition law is a per-step (teacher-forced) measure (§9.2), computed over the same eval
    rollouts and pooled per difficulty.
    """
    config = config or EH1Config()
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host)
    model = train_model(config, vocab, oracle, host)
    world_model = NeuralHostWorldModel(model, vocab)
    partial = PartialHostOracle(oracle)

    records: list[HostRunRecord] = []
    pooled_faithful: dict[str, list[dict[str, bool]]] = {}
    for difficulty, driver in config.difficulties.items():
        for seed in config.eval_seeds:
            actions = eval_actions(oracle, host, driver, seed, config.eval_steps)
            pooled_faithful.setdefault(difficulty, []).extend(
                teacher_forced_faithful(world_model, oracle, actions, config.composition_epsilon)
            )
            for rho in config.rhos:
                # ε does not affect loop dynamics (only H_ε), so run the rollout once per
                # (difficulty, seed, ρ) and spin out per-ε records (the E1/EN1 convention).
                rollout = run_host_rollout(
                    world_model, partial, HostState.initial(), actions,
                    fixed_interval_for_rho(rho), epsilon=config.epsilons[0],
                    budget=budget_for_rho(rho, len(actions)), seed=seed,
                )
                for epsilon in config.epsilons:
                    records.append(
                        HostRunRecord(
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
                            subsystem_divergences={
                                sub: list(traj)
                                for sub, traj in rollout.subsystem_divergences.items()
                            },
                            consultation_schedule=list(rollout.consultation_schedule),
                        )
                    )
    composition = {
        difficulty: composition_law(steps) for difficulty, steps in sorted(pooled_faithful.items())
    }
    return EH1Result(records=records, composition=composition)


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    from verisim.hostmetrics.record import write_host_records

    parser = argparse.ArgumentParser(description="Run EH1 (composed-host H_eps(rho) + H13).")
    parser.add_argument("--config", type=str, default=None, help="path to an EH1 config JSON")
    parser.add_argument("--out", type=str, default="runs/eh1/host_records.jsonl")
    parser.add_argument("--comp-out", type=str, default="runs/eh1/composition.json")
    args = parser.parse_args()
    config = EH1Config.from_json_file(args.config) if args.config else EH1Config()
    result = run_eh1(config)
    path = write_host_records(result.records, args.out)
    comp_path = Path(args.comp_out)
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comp_path.write_text(
        json.dumps({d: law.to_dict() for d, law in result.composition.items()}, indent=2)
    )
    print(f"wrote {len(result.records)} host records to {path}")
    print(f"wrote composition law to {comp_path}")
    for difficulty, law in result.composition.items():
        print(f"  composition law [{difficulty}]: {law.verdict} "
              f"(composed={law.composed_acceptance:.3f}, "
              f"mult={law.multiplicative_prediction:.3f}, "
              f"weakest={law.weakest_link_prediction:.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()
