"""Experiment EH5 -- the smart which-subsystem policy ``π_w`` (SPEC-6 §8.2, HC7; H10's analogue).

EH2 showed *when* to consult is a real lever once the signal is calibrated (the factored arm's
belief variance). EH5 asks the companion question on the other consultation axis: *which subsystem*
to verify (``π_w``, §8.2). It runs the factored arm in **per-subsystem** consultation mode at a
fixed budget and compares the which-subsystem policies at *equal* ``ρ``:

  - ``fixed_proc`` / ``fixed_fd`` -- always verify one subsystem (the EH3 ablation arms: ``proc`` is
    the H13-weakest, ``fd`` the cheapest);
  - ``round_robin`` -- cycle the subsystems (uniform coverage, the dumb baseline);
  - ``uncertainty`` -- the **information-gain** policy: spend the consult on the subsystem whose
    predicted delta the model is *least certain* about, read from the factored arm's per-subsystem
    decode entropy (§5.4). The §8.2 active oracle-selection the smart-``π_w`` work is about.

Reported per policy: faithful horizon ``H_ε``, oracle-bits, and **horizon-per-oracle-bit** (§9.4) --
so the question is whether *targeting the uncertain subsystem* earns more faithful horizon (per
consult, or per bit) than fixed/round-robin. Whatever it shows is a datum: a smoke-scale instance of
the apparatus, not a tuned publication run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.state import HostState
from verisim.hostloop import (
    FixedSubsystem,
    PartialHostOracle,
    RoundRobinSubsystem,
    SubsystemFilter,
    UncertaintySubsystem,
    budget_for_rho,
    run_host_rollout,
)
from verisim.hostloop.subsystem import SubsystemPolicy
from verisim.hostmodel import HostVocab
from verisim.hostmodel.graph_model import GraphHostWorldModel
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.record import RunRecord, write_records

from .eh1 import EH1Config, eval_actions

_POLICIES = ("fixed_proc", "fixed_fd", "round_robin", "uncertainty")


@dataclass(frozen=True)
class EH5Config:
    name: str = "eh5-small"
    base: EH1Config = field(default_factory=EH1Config)
    rho: float = 0.3
    policies: tuple[str, ...] = _POLICIES
    max_pid: int = 64
    graph_iters: int = 800
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_batch: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH5Config:
        b = EH5Config()
        return EH5Config(
            name=d.get("name", b.name),
            base=EH1Config.from_dict(d.get("base", {})),
            rho=d.get("rho", b.rho),
            policies=tuple(d.get("policies", b.policies)),
            max_pid=d.get("max_pid", b.max_pid),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_batch=d.get("graph_batch", b.graph_batch),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH5Config:
        return EH5Config.from_dict(json.loads(Path(path).read_text()))


def _make_policy(name: str) -> SubsystemPolicy:
    if name == "fixed_proc":
        return FixedSubsystem("proc")
    if name == "fixed_fd":
        return FixedSubsystem("fd")
    if name == "round_robin":
        return RoundRobinSubsystem()
    if name == "uncertainty":
        return UncertaintySubsystem()
    raise ValueError(f"unknown π_w policy {name!r}")


def _train_factored(config: EH5Config, vocab: HostVocab, oracle: HostOracle) -> GraphHostWorldModel:
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, train_host_graph_model

    base = config.base
    host = DEFAULT_HOST_CONFIG
    examples = build_host_graph_dataset(
        oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
        n_steps=base.train_steps_per_traj,
    )
    model = build_host_graph_model(
        vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
        mp_rounds=config.graph_mp_rounds, seed=base.model_seed,
    )
    train_host_graph_model(
        model, examples, steps=config.graph_iters, lr=base.lr,
        batch_size=config.graph_batch, seed=base.model_seed,
    )
    return model


def run_eh5(
    config: EH5Config | None = None, *, oracle: HostOracle | None = None
) -> list[RunRecord]:
    """Train the factored arm; compare the which-subsystem policies at the fixed interior ``ρ``."""
    config = config or EH5Config()
    base = config.base
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)
    model = _train_factored(config, vocab, oracle)
    partial = PartialHostOracle(oracle)

    records: list[RunRecord] = []
    for difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            actions = eval_actions(oracle, host, driver, seed, base.eval_steps)
            budget = budget_for_rho(config.rho, len(actions))
            for policy_name in config.policies:
                rollout = run_host_rollout(
                    model, partial, HostState.initial(), actions,
                    fixed_interval_for_rho(config.rho),
                    epsilon=base.epsilons[0], budget=budget,
                    subsystem_policy=_make_policy(policy_name), subsystem_op=SubsystemFilter(),
                    seed=seed,
                )
                for epsilon in base.epsilons:
                    records.append(
                        RunRecord(
                            config={
                                "experiment": config.name,
                                "policy": policy_name,
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


def efficiency_by_policy(records: list[RunRecord]) -> dict[str, dict[str, float]]:
    """Mean faithful horizon, oracle-bits, and horizon-per-oracle-bit by π_w policy (§9.4)."""
    eps = min(r.epsilon for r in records) if records else 0.0
    by_p: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        if r.epsilon == eps:
            by_p.setdefault(str(r.config["policy"]), []).append(
                (float(r.faithful_horizon), float(r.config["oracle_bits"]))
            )
    out: dict[str, dict[str, float]] = {}
    for p, pairs in sorted(by_p.items()):
        mean_h = fmean(h for h, _ in pairs)
        mean_bits = fmean(b for _, b in pairs)
        out[p] = {
            "mean_h": mean_h,
            "mean_bits": mean_bits,
            "h_per_bit": (mean_h / mean_bits) if mean_bits else 0.0,
        }
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run EH5 (host which-subsystem π_w comparison).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/eh5/records.jsonl")
    args = parser.parse_args()
    config = EH5Config.from_json_file(args.config) if args.config else EH5Config()
    records = run_eh5(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")
    for p, stats in efficiency_by_policy(records).items():
        print(f"  {p:14s} H_ε={stats['mean_h']:.2f}  bits={stats['mean_bits']:.1f}  "
              f"H_ε/bit={stats['h_per_bit']:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
