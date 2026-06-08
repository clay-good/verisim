"""Experiment LP1: does the ``embed()`` latent encode planning geometry? (H31, SPEC-12 §5-§6).

The gate that decides SPEC-12 §4's branch. L3P scatters landmarks in a learned latent and trusts
*latent distance ≈ reachability* - the single assumption the whole architecture rests on. Verisim
can **measure** it rather than assume it, because it has the oracle continuous-control worlds lack.

On held-out network states LP1 measures, between an anchor state and every state within a few oracle
steps of it (the action-graph ball), three distances and their rank/linear correlation:

  - **latent distance** - Euclidean distance between the two states' graph-arm ``embed()`` vectors;
  - **oracle transition-distance** - the exact BFS geodesic over the action graph (the fewest oracle
    steps between the states), :func:`verisim.landmark.geometry.bfs_geodesics`;
  - **control-plane reachability distance** - the count of differing reachability entries,
    :func:`verisim.netoracle.reachability_bits_to_correct`.

H31's pre-registered branch (SPEC-12 §4): if latent distance tracks oracle/reachability distance
(target Spearman ρ ≥ 0.6), landmarks live in the latent (the L3P recipe transfers); if it does not,
SPEC-12 builds the graph directly in reachability space - a clean result either way about *what* the
EN8 representation encodes (planning geometry, or only one-step-prediction geometry). The
``geodesic↔reachability`` correlation is reported as a descriptive sanity check that the two free
ground truths agree. CPU, deterministic, seeded (SPEC-2 §12).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from verisim.landmark.geometry import (
    bfs_geodesics,
    enumerate_actions,
    pearson,
    spearman,
)
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import NetConfig, scaled_net_config
from verisim.net.state import NetworkState
from verisim.netoracle import ReferenceNetworkOracle, reachability_bits_to_correct

METRICS = (
    "spearman_latent_geodesic",
    "pearson_latent_geodesic",
    "spearman_latent_reach",
    "pearson_latent_reach",
    "spearman_geodesic_reach",
)


@dataclass(frozen=True)
class LP1Config:
    """A small, fast latent-planning-geometry (H31) measurement instance."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 48
    graph_mp_rounds: int = 3
    graph_iters: int = 1500
    model_seed: int = 0
    anchor_driver: str = "weighted"
    anchor_seeds: tuple[int, ...] = (100, 101, 102, 103)
    anchor_stride: int = 6
    anchors_per_seed: int = 4
    bfs_max_depth: int = 3
    bfs_max_nodes: int = 220

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LP1Config:
        b = LP1Config()
        return LP1Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            model_seed=d.get("model_seed", b.model_seed),
            anchor_driver=d.get("anchor_driver", b.anchor_driver),
            anchor_seeds=tuple(d.get("anchor_seeds", b.anchor_seeds)),
            anchor_stride=d.get("anchor_stride", b.anchor_stride),
            anchors_per_seed=d.get("anchors_per_seed", b.anchors_per_seed),
            bfs_max_depth=d.get("bfs_max_depth", b.bfs_max_depth),
            bfs_max_nodes=d.get("bfs_max_nodes", b.bfs_max_nodes),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> LP1Config:
        return LP1Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class MetricStat:
    """One correlation reduced over anchors: mean + bootstrap CI (the EN10 reduction)."""

    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return f"{self.metric},{self.mean:.6f},{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"


CSV_HEADER = "metric,mean,ci_lo,ci_hi,n"


def _sample_anchors(
    config: LP1Config, net_config: NetConfig, oracle: ReferenceNetworkOracle
) -> list[NetworkState]:
    """Held-out anchor states: every ``anchor_stride`` steps of seeded driver rollouts."""
    from verisim.netdata import NetDriver

    anchors: list[NetworkState] = []
    for seed in config.anchor_seeds:
        drv = NetDriver(name=config.anchor_driver, config=net_config, rng=random.Random(seed))
        state = NetworkState.initial(net_config.hosts)
        for step in range(1, config.anchor_stride * config.anchors_per_seed + 1):
            state = oracle.step(state, drv.sample(state)).state
            if step % config.anchor_stride == 0:
                anchors.append(state)
    return anchors


def run_lp1(config: LP1Config | None = None) -> list[MetricStat]:
    """Train the graph arm, then correlate latent distance with oracle/reach distance (H31)."""
    import torch

    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph import build_graph
    from verisim.netmodel.graph_model import build_graph_model, graphs_to_tensors
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    config = config or LP1Config()
    torch.set_num_threads(1)  # process-reproducibility (the EN1 discipline)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)

    model = build_graph_model(
        vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds,
        seed=config.model_seed,
    )
    examples = build_graph_dataset(
        oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    train_graph_model(model, examples, steps=config.graph_iters, seed=config.model_seed)
    model.net.eval()

    def embed(states: list[NetworkState]) -> torch.Tensor:
        graphs = [build_graph(s, None, net) for s in states]
        node, gfeat, a_link, a_flow = graphs_to_tensors(graphs, model.net.device)
        with torch.no_grad():
            return model.net.embed(node, gfeat, a_link, a_flow)  # [B, d]

    actions = enumerate_actions(net)
    anchors = _sample_anchors(config, net, oracle)

    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    for anchor in anchors:
        reps = bfs_geodesics(
            oracle, anchor, actions,
            max_depth=config.bfs_max_depth, max_nodes=config.bfs_max_nodes,
        )
        if len(reps) < 4:  # need a few non-trivial pairs for a meaningful correlation
            continue
        states = [s for s, _ in reps]
        emb = embed(states)
        anchor_emb = emb[0]
        latent = [
            float(torch.linalg.vector_norm(emb[i] - anchor_emb).item()) for i in range(len(reps))
        ]
        geodesic = [float(d) for _, d in reps]
        reach = [float(reachability_bits_to_correct(anchor, s)) for s in states]
        # Drop the self-pair (index 0: distance 0 to itself, trivially perfect on every axis).
        latent, geodesic, reach = latent[1:], geodesic[1:], reach[1:]

        per_metric["spearman_latent_geodesic"].append(spearman(latent, geodesic))
        per_metric["pearson_latent_geodesic"].append(pearson(latent, geodesic))
        per_metric["spearman_latent_reach"].append(spearman(latent, reach))
        per_metric["pearson_latent_reach"].append(pearson(latent, reach))
        per_metric["spearman_geodesic_reach"].append(spearman(geodesic, reach))

    stats: list[MetricStat] = []
    for metric in METRICS:
        vals = per_metric[metric]
        lo, hi = bootstrap_ci(vals, seed=0) if vals else (float("nan"), float("nan"))
        mean = fmean(vals) if vals else float("nan")
        stats.append(MetricStat(metric, mean, lo, hi, len(vals)))
    return stats


def _print_summary(stats: list[MetricStat]) -> None:
    print("LP1 / H31 - does embed() latent distance track planning geometry?")
    print(f"  {'metric':<26} {'mean':>9} {'95% CI':>20}")
    for s in stats:
        print(f"  {s.metric:<26} {s.mean:>9.3f} {f'[{s.ci_lo:.3f}, {s.ci_hi:.3f}]':>20}")
    by = {s.metric: s for s in stats}
    rho = by["spearman_latent_geodesic"].mean
    verdict = "latent encodes planning geometry - H31 supported" if rho >= 0.6 else (
        "latent does NOT reach the ρ≥0.6 bar - build in reachability space (§4 fallback)"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[MetricStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {s.metric: s for s in stats}
    order = list(METRICS)
    labels = [
        "Spearman\nlatent×geodesic", "Pearson\nlatent×geodesic",
        "Spearman\nlatent×reach", "Pearson\nlatent×reach",
        "Spearman\ngeodesic×reach",
    ]
    means = [by[m].mean for m in order]
    lo = [by[m].mean - by[m].ci_lo for m in order]
    hi = [by[m].ci_hi - by[m].mean for m in order]
    colors = ["#16a", "#9bd", "#393", "#9c9", "#888"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(range(len(order)), means, yerr=[lo, hi], color=colors, capsize=4)
    ax.axhline(0.6, ls="-", color="#c33", lw=1, label="H31 bar (ρ=0.6)")
    ax.axhline(0.0, color="#000", lw=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(min(-0.2, min(means) - 0.1), 1.0)
    ax.set_ylabel("correlation (↑ latent tracks dynamics)")
    ax.set_title("LP1 / H31: does the embed() latent encode planning geometry?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="LP1 latent planning-geometry gate (H31).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/lp1_latent_geometry.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = LP1Config.from_json_file(args.config) if args.config else LP1Config()
    stats = run_lp1(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
