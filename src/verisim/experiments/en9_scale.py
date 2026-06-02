"""EN9 scale-up: the interventional lift (H25-S / H5) across scale (SPEC-8 §7.1, OG5/OG6).

Runs the EN9 oracle-contrastive ablation (:mod:`verisim.experiments.en9`) across a **world-size x
model-size x seed** grid and reduces each cell to the gap the scale-up is pre-registered on:

  - **interventional lift (H25-S / H5)** -- ``top1(oracle) - top1(vicreg)`` (and the MRR analogue)
    on held-out counterfactual branches. Only the oracle's *counterfactual* negatives carry
    information about which intervention leads where; VICReg keeps the representation full-rank but
    interventionally blind. Pre-registered to *widen* with world size, because more hosts create
    more distinct branches so retrieval chance ``1/m`` falls (SPEC-8 §7.1).

Branches are mined from held-out (eval-seed) states, so the fidelity probe is genuinely
out-of-sample -- a tightening over the smoke EN9. Single seed reproduces the OG4 datum; many seeds
give the CIs OG4 lacked. CPU is the deterministic gate; ``--device {mps,cuda}`` runs the same path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from verisim.net.config import NetConfig, scaled_net_config
from verisim.netmodel import NetVocab
from verisim.netmodel.graph_model import GraphRSSMWorldModel, build_graph_model
from verisim.netmodel.grounded_train import (
    ContrastiveResult,
    build_contrastive_dataset,
    train_contrastive,
)
from verisim.netoracle import ReferenceNetworkOracle

from .scale_common import (
    GapStat,
    ModelSize,
    disjoint_from_zero,
    plot_scaling_curves,
    summarize,
    write_csv,
)


def _build(
    vocab: NetVocab, net: NetConfig, ms: ModelSize, seed: int, device: str
) -> GraphRSSMWorldModel:
    """Construct a graph arm at model size ``ms`` on ``device`` (typed wrapper over the knobs)."""
    return build_graph_model(
        vocab, net, d_model=ms.d_model, mp_rounds=ms.mp_rounds,
        n_layer=ms.n_layer, n_head=ms.n_head, seed=seed, device=device,
    )

GAP_METRICS = ("lift_top1", "lift_mrr")
CONTEXT_METRICS = ("oracle_top1", "vicreg_top1", "oracle_std", "vicreg_std", "none_std")


@dataclass(frozen=True)
class EN9ScaleConfig:
    """The OG5/OG6 sweep grid for EN9. Defaults are a fast *local proof*; CLI scales each axis."""

    world_sizes: tuple[int, ...] = (5, 10)
    n_ports: int = 3
    model_sizes: tuple[ModelSize, ...] = (ModelSize("d48-mp3", 48, 3),)
    seeds: tuple[int, ...] = (0, 1, 2)  # model seeds → the CI population
    train_seeds: tuple[int, ...] = (0, 1, 2)
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    train_steps_per_traj: int = 40
    eval_steps: int = 24
    k_negatives: int = 8
    contrastive_iters: int = 400
    temperature: float = 0.1
    device: str = "cpu"


def run_en9_scale(config: EN9ScaleConfig | None = None) -> list[GapStat]:
    """Run the EN9 sweep; return one :class:`GapStat` per (world, model, metric) cell."""
    import torch

    config = config or EN9ScaleConfig()
    if config.device == "cpu":
        torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    stats: list[GapStat] = []

    for n_hosts in config.world_sizes:
        net = scaled_net_config(n_hosts, config.n_ports)
        vocab = NetVocab(net)
        examples, _ = build_contrastive_dataset(
            oracle, vocab, net, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, k_negatives=config.k_negatives,
        )
        # Held-out branches for the interventional-fidelity probe (out-of-sample, §7.1).
        _, branches = build_contrastive_dataset(
            oracle, vocab, net, seeds=config.eval_seeds,
            n_steps=config.eval_steps, k_negatives=config.k_negatives,
        )
        for ms in config.model_sizes:
            per_seed: dict[str, list[float]] = {m: [] for m in (*GAP_METRICS, *CONTEXT_METRICS)}
            for seed in config.seeds:
                results: dict[str, ContrastiveResult] = {}
                for mode in ("none", "vicreg", "oracle"):
                    model = _build(vocab, net, ms, seed, config.device)
                    results[mode] = train_contrastive(
                        model, examples, branches, mode=mode,
                        steps=config.contrastive_iters, temperature=config.temperature, seed=seed,
                    )
                o_r, v_r, n_r = results["oracle"], results["vicreg"], results["none"]
                per_seed["lift_top1"].append(o_r.intervention_top1 - v_r.intervention_top1)
                per_seed["lift_mrr"].append(o_r.intervention_mrr - v_r.intervention_mrr)
                per_seed["oracle_top1"].append(o_r.intervention_top1)
                per_seed["vicreg_top1"].append(v_r.intervention_top1)
                per_seed["oracle_std"].append(o_r.emb_std)
                per_seed["vicreg_std"].append(v_r.emb_std)
                per_seed["none_std"].append(n_r.emb_std)

            for metric, values in per_seed.items():
                stats.append(summarize(n_hosts, ms.label, metric, values))
    return stats


def _print_summary(stats: list[GapStat]) -> None:
    print(f"  {'world':>5} {'model':<10} {'metric':<12} {'mean':>9} {'95% CI':>20} {'>0?':>5}")
    for s in stats:
        if s.metric not in GAP_METRICS:
            continue
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        flag = "yes" if disjoint_from_zero(s) else "no"
        print(
            f"  {s.world_size:>5} {s.model_label:<10} {s.metric:<12}"
            f" {s.mean:>9.3f} {ci:>20} {flag:>5}"
        )


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN9 scale-up (H25-S / H5 interventional lift).")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d-models", type=int, nargs="+", default=[48],
                        help="model-size axis (one ModelSize per d_model, S1/S2)")
    parser.add_argument("--mp-rounds", type=int, default=3)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--contrastive-iters", type=int, default=400)
    parser.add_argument("--k-negatives", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", type=str, default="figures/en9_scale.csv")
    args = parser.parse_args()

    cfg = EN9ScaleConfig(
        world_sizes=tuple(args.world_sizes),
        seeds=tuple(args.seeds),
        model_sizes=tuple(
            ModelSize(f"d{d}-mp{args.mp_rounds}", d, args.mp_rounds, args.n_layer, args.n_head)
            for d in args.d_models
        ),
        contrastive_iters=args.contrastive_iters,
        k_negatives=args.k_negatives,
        device=args.device,
    )
    stats = run_en9_scale(cfg)
    print("EN9 scale-up: interventional lift oracle-vs-vicreg (H25-S / H5), bootstrap CIs:")
    _print_summary(stats)
    out = Path(args.out)
    write_csv(stats, out)
    print(f"wrote {out}")
    plot_scaling_curves(
        stats, list(GAP_METRICS), out.with_suffix(".png"),
        title="EN9 scale-up: oracle interventional lift over VICReg vs world size (95% CI bands)",
        gap_metrics=set(GAP_METRICS),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
