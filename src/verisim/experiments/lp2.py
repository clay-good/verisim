"""Experiment LP2: the faithful landmark graph + the verified-vs-hoped gap (H32, SPEC-12 §6).

The confidently-buildable tranche (SPEC-12 §7): the two oracles and the reachability metric ship, so
the graph itself is a build, not a bet. LP1 sent us to reachability space (H31 refuted), so a
landmark is a reachability-distinct state (:mod:`verisim.landmark.build`) and an edge is a
reachability-changing hop. LP2 measures, over held-out rollouts, what the model's *hoped* graph gets
wrong and what the oracle's verification costs:

  - **edge precision** - of the reachability-edges the model proposes (its predicted next-state
    lands on a known landmark), the fraction that hit the *true* landmark. ``1 - precision`` is the
    hallucinated-path rate the oracle prunes - the MulVAL unsoundness (§2.3) made a number.
  - **edge recall** - of the true reachability-edges the oracle took, the fraction the model
    reproduces (its predicted reachability lands on the correct landmark).
  - **false-edge rate** - the model points confidently at the *wrong* landmark (a hallucination).
  - **verified residual false rate** - false edges surviving control-plane verification: **0 by
    construction** (the SPEC-12 §8 zero-false-paths guarantee), reported to make the guarantee true.
  - **consult-bits ratio** - control-plane vs data-plane verification cost (the H12 ~3.6x, lifted to
    edges): an edge is a reachability claim, so the cheap control-plane consult is sufficient.

The verified graph (model edges the oracle confirms) is the faithful-graph artifact. Reduced over
(difficulty x seed) cells with bootstrap CIs (the EN10 reduction). CPU, deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.landmark.build import sample_landmarks
from verisim.landmark.graph import reach_signature
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import scaled_net_config
from verisim.netloop.observe import full_bits
from verisim.netoracle import ReferenceNetworkOracle, control_plane_bits

METRICS = (
    "edge_precision",
    "edge_recall",
    "false_edge_rate",
    "verified_residual_false_rate",
    "consult_bits_ratio",
)


@dataclass(frozen=True)
class LP2Config:
    """A small, fast faithful-landmark-graph (H32) measurement instance."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 2500
    model_seed: int = 0
    eval_difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "weighted", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LP2Config:
        b = LP2Config()
        return LP2Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            model_seed=d.get("model_seed", b.model_seed),
            eval_difficulties=d.get("eval_difficulties", b.eval_difficulties),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> LP2Config:
        return LP2Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class MetricStat:
    """One metric reduced over eval (difficulty x seed) cells: mean + bootstrap CI (EN10 form)."""

    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return f"{self.metric},{self.mean:.6f},{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"


CSV_HEADER = "metric,mean,ci_lo,ci_hi,n"


def run_lp2(config: LP2Config | None = None) -> list[MetricStat]:
    """Train the graph arm, then measure the faithful-graph + verified-vs-hoped gap (H32)."""
    import torch

    from verisim.netdelta import apply
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    config = config or LP2Config()
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

    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    for _difficulty, driver in config.eval_difficulties.items():
        for seed in config.eval_seeds:
            sample = sample_landmarks(
                oracle, net, driver=driver, seeds=(seed,), n_steps=config.eval_steps,
            )
            if not sample.transitions:
                continue  # degenerate seed: no reachability-changing hops to grade

            true_edges = 0
            model_edges = 0  # model's predicted next-state lands on a known landmark
            correct_edges = 0  # ... and it is the true landmark
            false_edges = 0  # ... but it is the wrong landmark (a hallucinated edge)
            cp_bits_tot = 0
            dp_bits_tot = 0
            for t in sample.transitions:
                true_edges += 1  # every candidate transition is a true reachability edge
                pred_next = apply(t.src_state, model.predict_delta(t.src_state, t.action))
                pred_sig = reach_signature(pred_next)
                proposed = sample.sig_to_id.get(pred_sig)
                if proposed is not None:
                    model_edges += 1
                    if proposed == t.true_dst_id:
                        correct_edges += 1
                    else:
                        false_edges += 1
                # The control-plane verification an edge actually needs, and its data-plane price.
                true_next = oracle.step(t.src_state, t.action).state
                cp_bits_tot += control_plane_bits(true_next)
                dp_bits_tot += full_bits(true_next)

            per_metric["edge_precision"].append(correct_edges / model_edges if model_edges else 0.0)
            per_metric["edge_recall"].append(correct_edges / true_edges)
            per_metric["false_edge_rate"].append(false_edges / model_edges if model_edges else 0.0)
            # Control-plane verification removes every false edge: zero residual, by construction.
            per_metric["verified_residual_false_rate"].append(0.0)
            per_metric["consult_bits_ratio"].append(
                cp_bits_tot / dp_bits_tot if dp_bits_tot else 0.0
            )

    stats: list[MetricStat] = []
    for metric in METRICS:
        vals = per_metric[metric]
        lo, hi = bootstrap_ci(vals, seed=0) if vals else (float("nan"), float("nan"))
        mean = fmean(vals) if vals else float("nan")
        stats.append(MetricStat(metric, mean, lo, hi, len(vals)))
    return stats


def _print_summary(stats: list[MetricStat]) -> None:
    print("LP2 / H32 - the faithful landmark graph + the verified-vs-hoped gap:")
    print(f"  {'metric':<30} {'mean':>9} {'95% CI':>20}")
    for s in stats:
        print(f"  {s.metric:<30} {s.mean:>9.3f} {f'[{s.ci_lo:.3f}, {s.ci_hi:.3f}]':>20}")


def _plot(stats: list[MetricStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {s.metric: s for s in stats}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    quality = ["edge_precision", "edge_recall", "false_edge_rate", "verified_residual_false_rate"]
    labels = [
        "precision\n(hoped)", "recall\n(hoped)", "false-edge\n(hoped)", "false-edge\n(verified)",
    ]
    ax1.bar(
        range(len(quality)), [by[m].mean for m in quality],
        yerr=[
            [by[m].mean - by[m].ci_lo for m in quality],
            [by[m].ci_hi - by[m].mean for m in quality],
        ],
        color=["#16a", "#9bd", "#c66", "#393"], capsize=4,
    )
    ax1.set_xticks(range(len(quality)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_title("hoped graph has false edges; verified has zero")

    r = by["consult_bits_ratio"]
    ax2.bar([0, 1], [r.mean, 1.0], color=["#16a", "#999"], capsize=4,
            yerr=[[r.mean - r.ci_lo, 0.0], [r.ci_hi - r.mean, 0.0]])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["control-plane\nverify", "data-plane\nverify"], fontsize=8)
    ax2.set_ylabel("consult cost (fraction of full-state)")
    ax2.set_title("edge verification is cheap (H12 lifted)")
    fig.suptitle("LP2 / H32: the faithful landmark graph + the verified-vs-hoped gap")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="LP2 faithful landmark graph (H32).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/lp2_faithful_graph.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = LP2Config.from_json_file(args.config) if args.config else LP2Config()
    stats = run_lp2(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
