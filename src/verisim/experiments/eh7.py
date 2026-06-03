"""Experiment EH7 — model-invariance of the composed-host H_ε(ρ) curve (SPEC-6, H22; SPEC.md §0).

The program's *deepest* claim is not about a model but a method: **deterministic verification is a
model-agnostic primitive** — the qualitative shape of the faithful-horizon-vs-consultation curve is
a property of the oracle-loop, not of the proposer's architecture (SPEC.md §0, the network EN7/H22).
EN7 established this in the network world; EH7 asks whether it survives the *hardest* world — the
**composed, coupled** host bundle (H13) — by dropping four materially different proposers into the
**same** HC5 loop and asking whether `H_ε(ρ)` is the same *in kind* across them:

  - **null** (:class:`~verisim.hostloop.model.HostNullModel`) — the empty bundle delta; drift floor.
  - **flat** (the HC4 transformer over the serialized bundle delta).
  - **factored** (the HC4-incr-2 GNN+RSSM over the process-interaction graph).
  - **oracle-backed** (:class:`~verisim.hostloop.model.HostOracleBackedModel`) — the oracle's own
    delta; the ceiling.

This turns the loop's "model-agnostic by construction" plumbing into evidence in the composed world.
It composes with EH4 (which asks *which* proposer is most faithful per-step) rather than duplicating
it: EH7 asks whether the *loop's* `H_ε(ρ)` behaviour is invariant to that choice. **H22 supported in
the host world** iff the curve is the same in kind across the imperfect learned proposers (flat,
factored) — both floor-then-cliff, not a knee for one and none for the other — with the proposer
setting the floor *height* (oracle > factored > flat > null) and the loop setting the *shape*.
*Refuted* iff the shape depends strongly on the proposer (a narrower but still-reportable result).
The honest strength: flat and factored have *materially different* per-step competence (EH4), so a
*shared* shape is the stronger evidence for H22. Regenerates from config + seeds; CPU, fast.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.state import HostState
from verisim.hostloop import (
    HostNullModel,
    HostOracleBackedModel,
    PartialHostOracle,
    budget_for_rho,
    run_host_rollout,
)
from verisim.hostloop.model import HostModel
from verisim.hostmetrics.record import HostRunRecord
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci

from .eh1 import EH1Config, eval_actions

PROPOSERS = ("null", "flat", "factored", "oracle")


@dataclass(frozen=True)
class EH7Config:
    """Small, fast model-invariance instance. Scale up (seeds/iters) for a publication run."""

    base: EH1Config = field(default_factory=EH1Config)
    max_pid: int = 64
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 800
    graph_batch: int = 32
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    epsilon: float = 0.05

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH7Config:
        b = EH7Config()
        return EH7Config(
            base=EH1Config.from_dict(d.get("base", {})),
            max_pid=d.get("max_pid", b.max_pid),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_batch=d.get("graph_batch", b.graph_batch),
            rhos=tuple(d.get("rhos", b.rhos)),
            epsilon=d.get("epsilon", b.epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH7Config:
        import json

        return EH7Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CurvePoint:
    """One (proposer, ρ) cell: mean faithful horizon + bootstrap CI over difficulty x seed."""

    proposer: str
    rho: float
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.proposer},{self.rho},{self.mean:.4f},"
            f"{self.ci_lo:.4f},{self.ci_hi:.4f},{self.n}"
        )


CSV_HEADER = "proposer,rho,mean_horizon,ci_lo,ci_hi,n"


def _train_proposers(config: EH7Config, oracle: ReferenceHostOracle) -> dict[str, HostModel]:
    """Train the flat + factored arms and assemble all four proposers behind the same protocol."""
    import torch

    from verisim.hostmodel import HostVocab, NeuralHostWorldModel
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, train_host_graph_model

    torch.set_num_threads(1)  # process-reproducibility (the EN1/EN7 discipline)
    base = config.base
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)

    from .eh1 import train_model as train_flat

    flat = NeuralHostWorldModel(train_flat(base, vocab, oracle, host), vocab)

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

    return {
        "null": HostNullModel(),
        "flat": flat,
        "factored": factored,
        "oracle": HostOracleBackedModel(oracle),
    }


def run_eh7(config: EH7Config | None = None) -> list[CurvePoint]:
    """Train the proposers and sweep H_ε(ρ) for each in the same loop; return one point per cell."""
    config = config or EH7Config()
    base = config.base
    oracle = ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    proposers = _train_proposers(config, oracle)
    partial = PartialHostOracle(oracle)

    points: list[CurvePoint] = []
    for name in PROPOSERS:
        model = proposers[name]
        per_rho: dict[float, list[float]] = {rho: [] for rho in config.rhos}
        for _difficulty, driver in base.difficulties.items():
            for seed in base.eval_seeds:
                actions = eval_actions(oracle, host, driver, seed, base.eval_steps)
                for rho in config.rhos:
                    rollout: HostRunRecord = run_host_rollout(
                        model, partial, HostState.initial(), actions,
                        fixed_interval_for_rho(rho), epsilon=config.epsilon,
                        budget=budget_for_rho(rho, len(actions)), seed=seed,
                    )
                    per_rho[rho].append(float(rollout.faithful_horizon))
        for rho in config.rhos:
            vals = per_rho[rho]
            lo, hi = bootstrap_ci(vals, seed=0)
            points.append(CurvePoint(name, rho, fmean(vals), lo, hi, len(vals)))
    return points


def _print_summary(points: list[CurvePoint], config: EH7Config) -> None:
    print(f"EH7 model-invariance: composed H_ε(ρ) by proposer (ε={config.epsilon}):")
    rhos = sorted({p.rho for p in points})
    print("  proposer  " + "".join(f"ρ={r:<6}" for r in rhos))
    for name in PROPOSERS:
        row = {p.rho: p.mean for p in points if p.proposer == name}
        print(f"  {name:<9} " + "".join(f"{row[r]:<8.1f}" for r in rhos))


def _plot(points: list[CurvePoint], path: Path, config: EH7Config) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = {"null": "#c66", "flat": "#9bd", "factored": "#16a", "oracle": "#393"}
    for name in PROPOSERS:
        cells = sorted((p for p in points if p.proposer == name), key=lambda p: p.rho)
        xs = [p.rho for p in cells]
        ys = [p.mean for p in cells]
        lo = [p.ci_lo for p in cells]
        hi = [p.ci_hi for p in cells]
        (line,) = ax.plot(xs, ys, marker="o", label=name, color=colors.get(name))
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("consultation budget ρ")
    ax.set_ylabel(f"composed faithful horizon H_ε (ε={config.epsilon})")
    ax.set_title("EH7 / H22: composed-host H_ε(ρ) is the same shape across proposers (95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH7 model-invariance of composed H_ε(ρ) (H22).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh7_invariance.csv")
    args = parser.parse_args()
    cfg = EH7Config.from_json_file(args.config) if args.config else EH7Config()
    points = run_eh7(cfg)
    _print_summary(points, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    print(f"wrote {out}")
    _plot(points, out.with_suffix(".png"), cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
