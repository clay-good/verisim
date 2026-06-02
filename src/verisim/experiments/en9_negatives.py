"""EN9 negative-count diagnostic: does scaling ``k_negatives`` recover the H5 lift? (SPEC-9 S2).

The LS2 surface found the oracle-over-VICReg interventional lift (H25-S/H5) *reverses* at large
worlds + high capacity (100-200 hosts/`d128`: VICReg overtakes the oracle). SPEC-9 §4 S2
pre-registered the most likely cause as a **fixable artifact**: ``k_negatives`` is fixed at 8 while
the counterfactual branch space grows with hosts, so the oracle's InfoNCE negatives become an
ever-sparser sample. This experiment tests that lever directly -- at a world where the lift had
reversed (default 100 hosts, `d128`), sweep ``k_negatives`` and re-measure the lift with CIs.

The honest prior, from reading the OG2 factory ([`netdata/negatives.py`](../netdata/negatives.py)):
:func:`~verisim.netdata.negatives.enumerate_actions` enumerates one action per command name on
``hosts[0]``/``hosts[1]``/``ports[0]`` -- so the *counterfactual* branch set is **fixed** (~11
actions), not growing with hosts; raising ``k`` therefore adds *one-edit* negatives, not branches.
So we expect the lift to **not** recover with ``k`` alone -- refuting S2's stated mechanism and
relocating the cause to *signal dilution* (an intervention touching 2 of N hosts barely moves a
mean-pooled embedding), pre-registering the real next lever: broaden the counterfactual branches
across the graph. Either way the datum is bankable. CPU, deterministic.

**Result (100 hosts/`d128`, 3 seeds): this prior was WRONG; S2 confirmed.** Scaling ``k_negatives``
8->16->32 flips ``lift_top1`` -0.075 -> +0.017 -> +0.032 [0.024, 0.044] (disjoint-positive), with
``lift_mrr`` tracking it. The reversal is a negative-count artifact after all: more *one-edit*
negatives recover branch retrieval by sharpening the contrastive geometry, even without more
counterfactual branches. Recovery is modest (vs the +0.10-0.35 small-world lift), so the rule is
**scale negatives with the world** (SPEC-9 §4 S2).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import NetConfig, scaled_net_config
from verisim.netmodel import NetVocab
from verisim.netmodel.graph_model import GraphRSSMWorldModel, build_graph_model
from verisim.netmodel.grounded_train import build_contrastive_dataset, train_contrastive
from verisim.netoracle import ReferenceNetworkOracle

METRICS = ("lift_top1", "lift_mrr", "oracle_top1", "vicreg_top1")
GAP_METRICS = ("lift_top1", "lift_mrr")


@dataclass(frozen=True)
class KNegStat:
    """One (k_negatives, metric) cell reduced over seeds: mean + bootstrap CI."""

    k_negatives: int
    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    @property
    def disjoint_positive(self) -> bool:
        return self.ci_lo > 0.0

    def csv_row(self) -> str:
        return (
            f"{self.k_negatives},{self.metric},"
            f"{self.mean:.6f},{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"
        )


CSV_HEADER = "k_negatives,metric,mean,ci_lo,ci_hi,n"


@dataclass(frozen=True)
class EN9NegConfig:
    """The S2-recovery diagnostic grid: a fixed reversed-regime world, swept ``k_negatives``."""

    world_size: int = 100  # a world where the LS2 lift had reversed (100/d128: -0.086)
    n_ports: int = 5
    d_model: int = 128
    mp_rounds: int = 3
    k_negatives: tuple[int, ...] = (8, 16, 32)
    seeds: tuple[int, ...] = (0, 1, 2)
    train_seeds: tuple[int, ...] = (0, 1, 2)
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    train_steps_per_traj: int = 40
    eval_steps: int = 24
    contrastive_iters: int = 400
    temperature: float = 0.1
    device: str = "cpu"


def _build(
    vocab: NetVocab, net: NetConfig, d_model: int, mp_rounds: int, seed: int, device: str
) -> GraphRSSMWorldModel:
    return build_graph_model(
        vocab, net, d_model=d_model, mp_rounds=mp_rounds, seed=seed, device=device
    )


def run_en9_negatives(config: EN9NegConfig | None = None) -> list[KNegStat]:
    """Run the k_negatives sweep; return one :class:`KNegStat` per (k, metric) cell."""
    import torch

    config = config or EN9NegConfig()
    if config.device == "cpu":
        torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net: NetConfig = scaled_net_config(config.world_size, config.n_ports)
    vocab = NetVocab(net)
    stats: list[KNegStat] = []

    for k in config.k_negatives:
        examples, _ = build_contrastive_dataset(
            oracle, vocab, net, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, k_negatives=k,
        )
        _, branches = build_contrastive_dataset(
            oracle, vocab, net, seeds=config.eval_seeds,
            n_steps=config.eval_steps, k_negatives=k,
        )
        per_seed: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in config.seeds:
            results = {}
            for mode in ("vicreg", "oracle"):
                model = _build(vocab, net, config.d_model, config.mp_rounds, seed, config.device)
                results[mode] = train_contrastive(
                    model, examples, branches, mode=mode,
                    steps=config.contrastive_iters, temperature=config.temperature, seed=seed,
                )
            o_r, v_r = results["oracle"], results["vicreg"]
            per_seed["lift_top1"].append(o_r.intervention_top1 - v_r.intervention_top1)
            per_seed["lift_mrr"].append(o_r.intervention_mrr - v_r.intervention_mrr)
            per_seed["oracle_top1"].append(o_r.intervention_top1)
            per_seed["vicreg_top1"].append(v_r.intervention_top1)
        for metric, values in per_seed.items():
            lo, hi = bootstrap_ci(values, seed=0)
            stats.append(KNegStat(k, metric, fmean(values), lo, hi, len(values)))
    return stats


def _print_summary(stats: list[KNegStat]) -> None:
    print(f"  {'k_neg':>6} {'metric':>12} {'mean':>9} {'95% CI':>20} {'>0?':>5}")
    for s in stats:
        if s.metric not in GAP_METRICS:
            continue
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        flag = "yes" if s.disjoint_positive else "no"
        print(f"  {s.k_negatives:>6} {s.metric:>12} {s.mean:>9.3f} {ci:>20} {flag:>5}")


def _plot(stats: list[KNegStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for metric in GAP_METRICS:
        cells = sorted((s for s in stats if s.metric == metric), key=lambda s: s.k_negatives)
        xs = [s.k_negatives for s in cells]
        ys = [s.mean for s in cells]
        lo = [s.ci_lo for s in cells]
        hi = [s.ci_hi for s in cells]
        (line,) = ax.plot(xs, ys, marker="o", label=metric)
        ax.fill_between(xs, lo, hi, alpha=0.18, color=line.get_color())
    ax.axhline(0.0, ls="--", lw=1, color="#555")
    ax.set_xlabel("k_negatives (contrastive negatives per anchor)")
    ax.set_ylabel("oracle - vicreg interventional lift")
    ax.set_title("EN9 S2-recovery: does scaling k_negatives recover the H5 lift? (95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN9 k_negatives S2-recovery diagnostic.")
    parser.add_argument("--world-size", type=int, default=100)
    parser.add_argument("--n-ports", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--k-negatives", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--contrastive-iters", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", type=str, default="figures/en9_negatives.csv")
    args = parser.parse_args()

    cfg = EN9NegConfig(
        world_size=args.world_size,
        n_ports=args.n_ports,
        d_model=args.d_model,
        k_negatives=tuple(args.k_negatives),
        seeds=tuple(args.seeds),
        contrastive_iters=args.contrastive_iters,
        device=args.device,
    )
    print(f"EN9 S2-recovery: lift vs k_negatives ({cfg.world_size}h/d{cfg.d_model}, CIs):")
    stats = run_en9_negatives(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
