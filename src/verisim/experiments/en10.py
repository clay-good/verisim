"""Experiment EN10: two-oracle grounding -- is the control-plane oracle non-redundant? (H12).

Completes EN6's deferred two-oracle axis (SPEC-5 §12). The data-plane oracle returns the exact next
state (the full delta); the Batfish-style **control-plane oracle** (`netoracle.control_plane`)
returns only the **reachability** truth. H12 asks whether it is *non-redundant* -- does consulting
it catch reachability errors a full-state data-plane consult misses?

On held-out trajectory transitions of the trained graph arm, EN10 measures, per step:

  - **data-plane bits-to-correct** -- the MDL of correcting the predicted *delta* to the truth.
  - **control-plane bits-to-correct** -- the count of reachability entries the predicted state gets
    wrong.
  - **non-redundant rate** -- fraction of steps with control-plane error *but no* data-plane error.
    H12's claim is this is large; the pre-registered honest negative is ~0: reachability is a
    deterministic function of the full state -- get the state right, get reachability right.
  - **control-plane-sufficient rate** -- fraction with data-plane error *but no* reachability error
    (the model is right on the coarse, decision-relevant query while wrong on the exact delta).
  - **consult-bits ratio** -- control-plane consult cost (``|R|``) / data-plane cost: how much
    cheaper (or pricier) the coarse oracle is to consult in this world.

The expected verdict is H12's honest negative -- the control-plane oracle is *redundant for
verification* (it cannot catch what the full-state oracle does not) -- but the *cheaper +
decision-relevant* framing (low control-plane bits-to-correct, positive sufficiency) is the real
value, and the tiered-oracle premise SPEC-7 builds on. CPU, deterministic.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import scaled_net_config
from verisim.net.state import NetworkState
from verisim.netdelta import apply
from verisim.netloop.observe import full_bits
from verisim.netmetrics.bits import bits_to_correct
from verisim.netoracle import (
    ReferenceNetworkOracle,
    control_plane_bits,
    reachability_bits_to_correct,
)

METRICS = (
    "data_plane_btc",
    "control_plane_btc",
    "nonredundant_rate",
    "cp_sufficient_rate",
    "consult_bits_ratio",
)


@dataclass(frozen=True)
class EN10Config:
    """Small, fast two-oracle (H12) measurement instance."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 48
    graph_mp_rounds: int = 3
    graph_iters: int = 1500
    model_seed: int = 0
    eval_difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "weighted", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24


@dataclass(frozen=True)
class MetricStat:
    """One metric reduced over eval (difficulty x seed) cells: mean + bootstrap CI."""

    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return f"{self.metric},{self.mean:.6f},{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"


CSV_HEADER = "metric,mean,ci_lo,ci_hi,n"


def run_en10(config: EN10Config | None = None) -> list[MetricStat]:
    """Train the graph arm, then measure the two-oracle (H12) quantities on held-out transitions."""
    import torch

    from verisim.netdata import NetDriver
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    config = config or EN10Config()
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

    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    for _difficulty, driver in config.eval_difficulties.items():
        for seed in config.eval_seeds:
            drv = NetDriver(name=driver, config=net, rng=random.Random(seed))
            state = NetworkState.initial(net.hosts)
            dp_btcs: list[float] = []
            cp_btcs: list[float] = []
            nonredundant = 0
            cp_sufficient = 0
            bits_ratios: list[float] = []
            total = 0
            for _ in range(config.eval_steps):
                action = drv.sample(state)
                result = oracle.step(state, action)
                pred_delta = model.predict_delta(state, action)
                pred_next = apply(state, pred_delta)
                dp = bits_to_correct(pred_delta, result.delta)
                cp = reachability_bits_to_correct(pred_next, result.state)
                dp_btcs.append(dp)
                cp_btcs.append(float(cp))
                nonredundant += int(cp > 0 and dp == 0.0)
                cp_sufficient += int(cp == 0 and dp > 0.0)
                fb = full_bits(result.state)
                bits_ratios.append(control_plane_bits(result.state) / fb if fb else 0.0)
                total += 1
                state = result.state
            per_metric["data_plane_btc"].append(fmean(dp_btcs))
            per_metric["control_plane_btc"].append(fmean(cp_btcs))
            per_metric["nonredundant_rate"].append(nonredundant / total)
            per_metric["cp_sufficient_rate"].append(cp_sufficient / total)
            per_metric["consult_bits_ratio"].append(fmean(bits_ratios))

    stats: list[MetricStat] = []
    for metric, vals in per_metric.items():
        lo, hi = bootstrap_ci(vals, seed=0)
        stats.append(MetricStat(metric, fmean(vals), lo, hi, len(vals)))
    return stats


def _print_summary(stats: list[MetricStat]) -> None:
    print("EN10 two-oracle (H12) — control-plane vs data-plane on held-out transitions:")
    print(f"  {'metric':<22} {'mean':>9} {'95% CI':>20}")
    for s in stats:
        print(f"  {s.metric:<22} {s.mean:>9.3f} {f'[{s.ci_lo:.3f}, {s.ci_hi:.3f}]':>20}")


def _plot(stats: list[MetricStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {s.metric: s for s in stats}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    btc = ["data_plane_btc", "control_plane_btc"]
    ax1.bar(
        range(len(btc)), [by[m].mean for m in btc],
        yerr=[[by[m].mean - by[m].ci_lo for m in btc], [by[m].ci_hi - by[m].mean for m in btc]],
        color=["#9bd", "#16a"], capsize=4,
    )
    ax1.set_xticks(range(len(btc)))
    ax1.set_xticklabels(["data-plane\n(full delta)", "control-plane\n(reachability)"], fontsize=8)
    ax1.set_title("bits-to-correct per step (↓ cheaper)")

    rates = ["nonredundant_rate", "cp_sufficient_rate"]
    ax2.bar(
        range(len(rates)), [by[m].mean for m in rates],
        yerr=[[by[m].mean - by[m].ci_lo for m in rates], [by[m].ci_hi - by[m].mean for m in rates]],
        color=["#c66", "#393"], capsize=4,
    )
    ax2.set_xticks(range(len(rates)))
    ax2.set_xticklabels(["non-redundant\n(H12)", "cp-sufficient\n(coarse easier)"], fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_title("redundancy of the control-plane oracle")
    fig.suptitle("EN10 / H12: is the control-plane oracle a non-redundant signal?")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN10 two-oracle grounding measurement (H12).")
    parser.add_argument("--n-hosts", type=int, default=5)
    parser.add_argument("--graph-iters", type=int, default=1500)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--out", type=str, default="figures/en10_two_oracle.csv")
    args = parser.parse_args()
    cfg = EN10Config(
        n_hosts=args.n_hosts, graph_iters=args.graph_iters, eval_seeds=tuple(args.eval_seeds)
    )
    stats = run_en10(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
