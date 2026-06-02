"""Experiment EN7 — model-invariance of the H_ε(ρ) curve (SPEC-5 §12, H22; SPEC.md §6 commitment 4).

The project's most general claim: **the qualitative shape of the faithful-horizon-vs-consultation
curve is a property of the oracle-loop, not of the proposer's architecture.** EN1 plotted `H_ε(ρ)`
for one proposer (the flat NW4 transformer); EN7 drops *materially different* proposers into the
**same** NW5 loop and asks whether the curve is the same *in kind* across them:

  - **null** (:class:`~verisim.netloop.model.NetNullModel`) -- the empty delta; the drift floor.
  - **flat** (the NW4 transformer) -- serializes the (state, action) and decodes the delta.
  - **graph** (the NW8 GNN + RSSM arm) -- message-passing over the host graph + belief.
  - **oracle-backed** (:class:`~verisim.netloop.model.NetOracleBackedModel`) -- the oracle's own
    delta; the ceiling.

This turns SPEC.md §6's "model-agnostic by construction" plumbing into evidence. It composes with
EN4 (which asks *which* proposer is most faithful per-step) rather than duplicating it: EN7 asks
whether the *loop's* `H_ε(ρ)` behavior is invariant to that choice. *H22 supported* iff the curve
shape is the same in kind across the imperfect learned proposers (flat, graph) -- e.g. both
floor-then-cliff, not a knee for one and none for the other. *Refuted* iff the shape depends
strongly on the proposer (a narrower but still-reportable result: the benefit is model-specific).
The two baselines bound the plot (null = floor, oracle = ceiling). Honest caveat: this is *not*
matched per-step competence (graph > flat, per EN4) -- so a *shared* shape across differing
competence is the stronger evidence for H22. Regenerates from config + seeds; CPU, deterministic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import NetConfig, scaled_net_config
from verisim.net.state import NetworkState
from verisim.netloop import (
    NetNullModel,
    NetOracleBackedModel,
    PartialNetOracle,
    budget_for_rho,
    run_net_rollout,
)
from verisim.netloop.model import NetModel
from verisim.netoracle import ReferenceNetworkOracle

from .en1 import eval_actions

PROPOSERS = ("null", "flat", "graph", "oracle")


@dataclass(frozen=True)
class EN7Config:
    """Small, fast model-invariance instance. Scale up (seeds/iters/world) for a publication run."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    # flat NW4 arm
    flat_n_layer: int = 2
    flat_n_head: int = 2
    flat_n_embd: int = 64
    flat_block_size: int = 256
    flat_iters: int = 600
    # graph NW8 arm
    graph_d_model: int = 48
    graph_mp_rounds: int = 3
    graph_iters: int = 1500
    lr: float = 3e-3
    model_seed: int = 0
    # evaluation / sweep
    difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "weighted", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    epsilon: float = 0.05


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
            f"{self.proposer},{self.rho},{self.mean:.4f},{self.ci_lo:.4f},{self.ci_hi:.4f},{self.n}"
        )


CSV_HEADER = "proposer,rho,mean_horizon,ci_lo,ci_hi,n"


def _faithful_horizon(divergences: list[float], epsilon: float) -> int:
    """Steps the rollout stays within ε — the first-exceedance horizon (mirrors EN1/EN4)."""
    for i, d in enumerate(divergences):
        if d > epsilon:
            return i
    return len(divergences)


def _train_proposers(
    config: EN7Config, oracle: ReferenceNetworkOracle, net: NetConfig
) -> dict[str, NetModel]:
    """Train the flat + graph arms and assemble all four proposers behind the same protocol."""
    import torch

    from verisim.model.transformer import GPT, GPTConfig
    from verisim.netmodel import NetVocab, NeuralNetworkWorldModel, build_net_dataset
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model
    from verisim.train.supervised import train_supervised

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)  # process-reproducibility (SPEC-2 §12 / the EN1 discipline)
    vocab = NetVocab(net)

    # flat NW4 transformer
    flat_examples = build_net_dataset(
        oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    flat = GPT(GPTConfig(
        vocab_size=len(vocab), block_size=config.flat_block_size,
        n_layer=config.flat_n_layer, n_head=config.flat_n_head, n_embd=config.flat_n_embd,
    ))
    train_supervised(flat, flat_examples, vocab.pad, steps=config.flat_iters, lr=config.lr,
                     seed=config.model_seed)
    flat_wm = NeuralNetworkWorldModel(flat, vocab)

    # graph NW8 arm
    graph_examples = build_graph_dataset(
        oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    graph_wm = build_graph_model(
        vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds,
        seed=config.model_seed,
    )
    train_graph_model(graph_wm, graph_examples, steps=config.graph_iters, seed=config.model_seed)

    return {
        "null": NetNullModel(),
        "flat": flat_wm,
        "graph": graph_wm,
        "oracle": NetOracleBackedModel(oracle),
    }


def run_en7(config: EN7Config | None = None) -> list[CurvePoint]:
    """Train the proposers and sweep H_ε(ρ) for each in the same loop; return one point per cell."""
    config = config or EN7Config()
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    proposers = _train_proposers(config, oracle, net)
    partial = PartialNetOracle(oracle)

    points: list[CurvePoint] = []
    for name in PROPOSERS:
        model = proposers[name]
        per_rho: dict[float, list[float]] = {rho: [] for rho in config.rhos}
        for _difficulty, driver in config.difficulties.items():
            for seed in config.eval_seeds:
                actions = eval_actions(oracle, net, driver, seed, config.eval_steps)
                for rho in config.rhos:
                    rollout = run_net_rollout(
                        model, partial, NetworkState.initial(net.hosts), actions,
                        fixed_interval_for_rho(rho), epsilon=config.epsilon,
                        budget=budget_for_rho(rho, len(actions)), seed=seed,
                    )
                    per_rho[rho].append(
                        _faithful_horizon(list(rollout.divergences), config.epsilon)
                    )
        for rho in config.rhos:
            vals = per_rho[rho]
            lo, hi = bootstrap_ci(vals, seed=0)
            points.append(CurvePoint(name, rho, fmean(vals), lo, hi, len(vals)))
    return points


def _print_summary(points: list[CurvePoint], config: EN7Config) -> None:
    print(f"EN7 model-invariance: H_ε(ρ) by proposer (ε={config.epsilon}, T={config.eval_steps}):")
    rhos = sorted({p.rho for p in points})
    print("  proposer  " + "".join(f"ρ={r:<6}" for r in rhos))
    for name in PROPOSERS:
        row = {p.rho: p.mean for p in points if p.proposer == name}
        print(f"  {name:<9} " + "".join(f"{row[r]:<8.1f}" for r in rhos))


def _plot(points: list[CurvePoint], path: Path, config: EN7Config) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = {"null": "#c66", "flat": "#9bd", "graph": "#16a", "oracle": "#393"}
    for name in PROPOSERS:
        cells = sorted((p for p in points if p.proposer == name), key=lambda p: p.rho)
        xs = [p.rho for p in cells]
        ys = [p.mean for p in cells]
        lo = [p.ci_lo for p in cells]
        hi = [p.ci_hi for p in cells]
        (line,) = ax.plot(xs, ys, marker="o", label=name, color=colors.get(name))
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("consultation budget ρ")
    ax.set_ylabel(f"faithful horizon H_ε (ε={config.epsilon}, ceiling T={config.eval_steps})")
    ax.set_title("EN7 / H22: H_ε(ρ) is the same shape in kind across proposers (95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN7 model-invariance of H_ε(ρ) (H22).")
    parser.add_argument("--n-hosts", type=int, default=5)
    parser.add_argument("--graph-iters", type=int, default=1500)
    parser.add_argument("--flat-iters", type=int, default=600)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--out", type=str, default="figures/en7_invariance.csv")
    args = parser.parse_args()
    cfg = EN7Config(
        n_hosts=args.n_hosts, graph_iters=args.graph_iters, flat_iters=args.flat_iters,
        eval_seeds=tuple(args.eval_seeds),
    )
    points = run_en7(cfg)
    _print_summary(points, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    print(f"wrote {out}")
    _plot(points, out.with_suffix(".png"), cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
