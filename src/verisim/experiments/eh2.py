"""Experiment EH2 -- consultation-policy comparison (SPEC-6 §8.1, HC7; H9's host analogue).

Fixes the budget ``ρ`` at the EH1 interior and compares the §8.1 consultation policies ``π_c`` --
``fixed`` vs. ``uncertainty_triggered`` vs. ``drift_triggered`` -- at *equal* ``ρ`` (the runner's
spend-down backstop makes every arm spend exactly ``floor(ρ·T)`` consultations, so the comparison
isolates *where* a policy spends its budget). This is **H9**: does spending the budget on the steps
the model is least sure about earn more faithful horizon than spreading it evenly?

The host twist over v0/EN2 is the **uncertainty signal itself**. v0 and EN2 read the flat ``M_θ``'s
mean decode entropy and found the triggered policies do not beat ``fixed`` (the standing
H2-negative, SPEC-2 §7.2). SPEC-6 §8.1 conjectures the factored arm's **RSSM belief variance** -- a
calibrated-by-construction signal (§6.2), not a decode-time artifact -- is the richer signal that
*could* fix it. EH2 runs both arms head to head: ``flat`` (decode entropy) and ``factored`` (belief
variance), each across the three policies, so the figure shows directly whether the better signal
buys H9 where the flat one cannot. Full-consultation mode, so ``ρ`` means what it does in EH1/EH3.

``build_policy`` is env-agnostic (it operates on the signal list + budget), so it is reused verbatim
from v0's E2 -- only the signal source differs by arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.experiments.e2 import build_policy
from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.delta import apply
from verisim.host.state import HostState
from verisim.hostloop import PartialHostOracle, budget_for_rho, run_host_rollout
from verisim.hostloop.model import HostUncertaintyModel
from verisim.hostmodel import HostVocab
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.metrics.record import RunRecord, write_records

from .eh1 import EH1Config, eval_actions
from .eh1 import train_model as train_flat


@dataclass(frozen=True)
class EH2Config:
    name: str = "eh2-small"
    base: EH1Config = field(default_factory=EH1Config)
    rho: float = 0.3  # the EH1 interior the policies are compared at
    policies: tuple[str, ...] = ("fixed", "uncertainty", "drift")
    arms: tuple[str, ...] = ("flat", "factored")
    max_pid: int = 64
    graph_iters: int = 800
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_batch: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH2Config:
        b = EH2Config()
        return EH2Config(
            name=d.get("name", b.name),
            base=EH1Config.from_dict(d.get("base", {})),
            rho=d.get("rho", b.rho),
            policies=tuple(d.get("policies", b.policies)),
            arms=tuple(d.get("arms", b.arms)),
            max_pid=d.get("max_pid", b.max_pid),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_batch=d.get("graph_batch", b.graph_batch),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH2Config:
        return EH2Config.from_dict(json.loads(Path(path).read_text()))


def unaided_signals(
    model: HostUncertaintyModel, actions: list[HostAction]
) -> list[float]:
    """The model's per-step uncertainty along the unaided (ρ=0) rollout (entropy / belief-var)."""
    state = HostState.initial()
    signals: list[float] = []
    for action in actions:
        delta, signal = model.predict_delta_with_uncertainty(state, action)
        signals.append(signal)
        state = apply(state, delta)
    return signals


def run_eh2(
    config: EH2Config | None = None, *, oracle: HostOracle | None = None
) -> list[RunRecord]:
    """Train the arm(s) and compare the consultation policies at the fixed interior ``ρ``."""
    from verisim.hostmodel import NeuralHostWorldModel
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, train_host_graph_model

    config = config or EH2Config()
    base = config.base
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)
    partial = PartialHostOracle(oracle)

    models: dict[str, HostUncertaintyModel] = {}
    if "flat" in config.arms:
        models["flat"] = NeuralHostWorldModel(train_flat(base, vocab, oracle, host), vocab)
    if "factored" in config.arms:
        graph_examples = build_host_graph_dataset(
            oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
            n_steps=base.train_steps_per_traj,
        )
        factored = build_host_graph_model(
            vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
            mp_rounds=config.graph_mp_rounds, seed=base.model_seed,
        )
        train_host_graph_model(
            factored, graph_examples, steps=config.graph_iters, lr=base.lr,
            batch_size=config.graph_batch, seed=base.model_seed,
        )
        models["factored"] = factored

    records: list[RunRecord] = []
    for arm in config.arms:
        model = models[arm]
        for difficulty, driver in base.difficulties.items():
            for seed in base.eval_seeds:
                actions = eval_actions(oracle, host, driver, seed, base.eval_steps)
                signals = unaided_signals(model, actions)
                budget = budget_for_rho(config.rho, len(actions))
                for policy_name in config.policies:
                    policy = build_policy(policy_name, config.rho, signals)
                    rollout = run_host_rollout(
                        model, partial, HostState.initial(), actions, policy,
                        epsilon=base.epsilons[0], budget=budget, seed=seed,
                    )
                    for epsilon in base.epsilons:
                        records.append(
                            RunRecord(
                                config={
                                    "experiment": config.name,
                                    "arm": arm,
                                    "policy": policy_name,
                                    "label": f"{arm}/{policy_name}",
                                    "difficulty": difficulty,
                                    "rho": config.rho,
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

    parser = argparse.ArgumentParser(description="Run EH2 (host consultation-policy comparison).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/eh2/records.jsonl")
    args = parser.parse_args()
    config = EH2Config.from_json_file(args.config) if args.config else EH2Config()
    records = run_eh2(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
