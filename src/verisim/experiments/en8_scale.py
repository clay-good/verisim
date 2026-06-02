"""EN8 scale-up: collapse tax (H23-S) + residual objective (H24-S) (SPEC-8 §7.1, OG5/OG6).

Runs the EN8 ablation (:mod:`verisim.experiments.en8`) across a **world-size x model-size x seed**
grid and reduces each cell to the two gaps the scale-up is pre-registered on (SPEC-8 §7.1):

  - **collapse gap (H23-S)** -- ``eff_rank(oracle, off) - eff_rank(learned, off)`` and the
    ``emb_std`` analogue: the oracle-anchored target's health advantage over the learned target,
    *with the EMA+VICReg machinery ablated*. Undismissable iff its bootstrap CI stays clear of 0
    and is stable/growing with scale.
  - **residual-objective gap (H24-S)** -- ``residual_acc(residual) - residual_acc(likelihood)`` on
    the bits ``R``. Per SPEC-8 §7.2 this is a *capacity-allocation* effect: it opens only when ``R``
    is hard and capacity binds, so the headline axis is **world size at fixed (small) capacity**.

Single seed reproduces the OG3 smoke datum; many seeds give the CIs OG3 lacked. Reuses
:func:`verisim.metrics.aggregate.bootstrap_ci` (the EN1 CI machinery) via
:mod:`verisim.experiments.scale_common`. CPU is the deterministic gate; ``--device {mps,cuda}`` runs
the identical code path for a local speedup or the rented GPU (§7.3).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from verisim.net.config import NetConfig, scaled_net_config
from verisim.netmodel import NetVocab
from verisim.netmodel.graph_model import GraphRSSMWorldModel, build_graph_model
from verisim.netmodel.grounded_train import (
    build_grounded_dataset,
    residual_token_accuracy,
    train_grounded_decoder,
    train_jepa,
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


# Headline gap metrics (get a y=0 reference in the figure) and context metrics (absolute levels).
GAP_METRICS = ("collapse_gap_rank", "collapse_gap_std", "residual_gap")
CONTEXT_METRICS = (
    "oracle_off_rank", "learned_off_rank", "residual_acc_residual", "residual_acc_likelihood",
)


@dataclass(frozen=True)
class EN8ScaleConfig:
    """The OG5/OG6 sweep grid. Defaults are a fast *local proof*; CLI scales each axis (§7.3)."""

    world_sizes: tuple[int, ...] = (5, 10)
    n_ports: int = 3
    model_sizes: tuple[ModelSize, ...] = (ModelSize("d48-mp3", 48, 3),)
    seeds: tuple[int, ...] = (0, 1, 2)  # model seeds → the CI population
    train_seeds: tuple[int, ...] = (0, 1, 2)
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    train_steps_per_traj: int = 40
    eval_steps: int = 24
    observed_fraction: float = 0.5
    decoder_iters: int = 600
    jepa_iters: int = 400
    device: str = "cpu"


def run_en8_scale(config: EN8ScaleConfig | None = None) -> list[GapStat]:
    """Run the EN8 sweep; return one :class:`GapStat` per (world, model, metric) cell."""
    import torch

    config = config or EN8ScaleConfig()
    if config.device == "cpu":
        torch.set_num_threads(1)  # the EN1 reproducibility discipline (bit-exact on CPU)
    oracle = ReferenceNetworkOracle()
    stats: list[GapStat] = []

    for n_hosts in config.world_sizes:
        net = scaled_net_config(n_hosts, config.n_ports)
        vocab = NetVocab(net)
        train = build_grounded_dataset(
            oracle, vocab, net, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, observed_fraction=config.observed_fraction,
        )
        held = build_grounded_dataset(
            oracle, vocab, net, seeds=config.eval_seeds,
            n_steps=config.eval_steps, observed_fraction=config.observed_fraction,
        )
        for ms in config.model_sizes:
            per_seed: dict[str, list[float]] = {m: [] for m in (*GAP_METRICS, *CONTEXT_METRICS)}
            for seed in config.seeds:
                # --- H23-S collapse axis: both targets with the machinery ablated -------
                m_oracle = _build(vocab, net, ms, seed, config.device)
                r_oracle = train_jepa(
                    m_oracle, train, target="oracle", collapse_machinery=False,
                    steps=config.jepa_iters, seed=seed,
                )
                m_learned = _build(vocab, net, ms, seed, config.device)
                r_learned = train_jepa(
                    m_learned, train, target="learned", collapse_machinery=False,
                    steps=config.jepa_iters, seed=seed,
                )
                # --- H24-S objective axis: residual vs raw-likelihood, accuracy on R -----
                m_res = _build(vocab, net, ms, seed, config.device)
                train_grounded_decoder(
                    m_res, train, objective="residual", steps=config.decoder_iters, seed=seed
                )
                _, acc_res = residual_token_accuracy(m_res, held)
                m_lik = _build(vocab, net, ms, seed, config.device)
                train_grounded_decoder(
                    m_lik, train, objective="likelihood", steps=config.decoder_iters, seed=seed
                )
                _, acc_lik = residual_token_accuracy(m_lik, held)

                per_seed["collapse_gap_rank"].append(r_oracle.eff_rank - r_learned.eff_rank)
                per_seed["collapse_gap_std"].append(r_oracle.emb_std - r_learned.emb_std)
                per_seed["oracle_off_rank"].append(r_oracle.eff_rank)
                per_seed["learned_off_rank"].append(r_learned.eff_rank)
                per_seed["residual_gap"].append(acc_res - acc_lik)
                per_seed["residual_acc_residual"].append(acc_res)
                per_seed["residual_acc_likelihood"].append(acc_lik)

            for metric, values in per_seed.items():
                stats.append(summarize(n_hosts, ms.label, metric, values))
    return stats


def _print_summary(stats: list[GapStat]) -> None:
    print(f"  {'world':>5} {'model':<10} {'metric':<22} {'mean':>9} {'95% CI':>20} {'!=0?':>5}")
    for s in stats:
        if s.metric not in GAP_METRICS:
            continue
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        flag = "yes" if disjoint_from_zero(s) else "no"
        print(
            f"  {s.world_size:>5} {s.model_label:<10} {s.metric:<22}"
            f" {s.mean:>9.3f} {ci:>20} {flag:>5}"
        )


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN8 scale-up (H23-S collapse, H24-S residual).")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d-models", type=int, nargs="+", default=[48],
                        help="model-size axis (one ModelSize per d_model, S1/S2)")
    parser.add_argument("--mp-rounds", type=int, default=3)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--decoder-iters", type=int, default=600)
    parser.add_argument("--jepa-iters", type=int, default=400)
    parser.add_argument("--observed-fraction", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", type=str, default="figures/en8_scale.csv")
    args = parser.parse_args()

    cfg = EN8ScaleConfig(
        world_sizes=tuple(args.world_sizes),
        seeds=tuple(args.seeds),
        model_sizes=tuple(
            ModelSize(f"d{d}-mp{args.mp_rounds}", d, args.mp_rounds, args.n_layer, args.n_head)
            for d in args.d_models
        ),
        decoder_iters=args.decoder_iters,
        jepa_iters=args.jepa_iters,
        observed_fraction=args.observed_fraction,
        device=args.device,
    )
    stats = run_en8_scale(cfg)
    print("EN8 scale-up — collapse gap (H23-S) + residual gap (H24-S), bootstrap CIs over seeds:")
    _print_summary(stats)
    out = Path(args.out)
    write_csv(stats, out)
    print(f"wrote {out}")
    plot_scaling_curves(
        stats, list(GAP_METRICS), out.with_suffix(".png"),
        title="EN8 scale-up: the oracle's advantage vs world size (95% CI bands)",
        gap_metrics=set(GAP_METRICS),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
