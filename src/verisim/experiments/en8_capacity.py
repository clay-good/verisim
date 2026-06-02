"""EN8 capacity-binding frontier: when does residual supervision beat raw-likelihood? (H24/S3).

H24-S landed an honest near-tie at full capacity (SPEC-8 §7.1): masking the oracle-decidable bits
``D`` and training only the residual ``R`` (*verify, don't learn*, SPEC-8 §4.2) bought nothing over
the raw-likelihood baseline. SPEC-8 §7.2 / SPEC-9 §4 (S3) pre-registered *why*: H24 is a
**capacity-allocation** claim -- masking ``D`` can only pay when (a) model capacity is *binding* and
(b) the freed capacity has somewhere useful to go. This experiment maps that frontier directly,
sweeping the two governing axes at a fixed, deliberately-hard world:

  - **capacity** (``d_model``): under-provisioned to adequate. The residual objective's edge should
    appear where capacity binds and vanish where the model fits both ``D`` and ``R``.
  - **observed fraction**: the share of hosts observed, which sets the ``D``/``R`` split. A *higher*
    fraction means more decidable ``D`` to offload (more capacity freed by masking it) but a
    *smaller* residual ``R``; *lower* means a bigger, harder ``R`` but less ``D`` to skip. Which way
    the frontier tilts is exactly what this measures -- the offload benefit and the residual
    hardness pull in opposite directions, so the sign of the gap across this axis is empirical.

Headline: the **residual gap** ``residual_acc(residual) - residual_acc(likelihood)`` on bits ``R``,
with a bootstrap CI over seeds (reusing the EN1 machinery). A cell whose CI is
disjoint-positive locates the H24 frontier; if no cell is disjoint-positive anywhere in the local
envelope, that is the strong, bankable form of the H24 negative (SPEC-9 S3). CPU, deterministic.
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
from verisim.netmodel.grounded_train import (
    GroundedExample,
    build_grounded_dataset,
    residual_token_accuracy,
    train_grounded_decoder,
)
from verisim.netoracle import ReferenceNetworkOracle

GAP_METRIC = "residual_gap"
METRICS = (GAP_METRIC, "residual_acc_residual", "residual_acc_likelihood")


@dataclass(frozen=True)
class CapacityStat:
    """One (d_model, observed_fraction, metric) cell reduced over seeds: mean + bootstrap CI."""

    d_model: int
    observed_fraction: float
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
            f"{self.d_model},{self.observed_fraction},{self.metric},"
            f"{self.mean:.6f},{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"
        )


CSV_HEADER = "d_model,observed_fraction,metric,mean,ci_lo,ci_hi,n"


@dataclass(frozen=True)
class EN8CapacityConfig:
    """The H24/S3 frontier grid. Defaults are a fast *local proof*; CLI scales each axis."""

    world_size: int = 40  # fixed, deliberately hard (many candidate host/port identities for R)
    n_ports: int = 5
    d_models: tuple[int, ...] = (16, 32, 64)  # capacity axis: under-provisioned -> adequate
    observed_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)  # the D/R split axis
    mp_rounds: int = 3
    seeds: tuple[int, ...] = (0, 1, 2, 3)  # the CI population
    train_seeds: tuple[int, ...] = (0, 1, 2)
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    train_steps_per_traj: int = 40
    eval_steps: int = 24
    decoder_iters: int = 600  # matched compute across both objectives
    device: str = "cpu"


def _residual_fraction(examples: list[GroundedExample]) -> float:
    """Share of target tokens that are residual (``R``) — context for reading the frontier."""
    res = sum(sum(ex.residual_mask) for ex in examples)
    tot = sum(len(ex.residual_mask) for ex in examples)
    return res / tot if tot else 0.0


def run_en8_capacity(config: EN8CapacityConfig | None = None) -> list[CapacityStat]:
    """Run the H24/S3 frontier sweep; one :class:`CapacityStat` per (d_model, obs, metric) cell."""
    import torch

    config = config or EN8CapacityConfig()
    if config.device == "cpu":
        torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net: NetConfig = scaled_net_config(config.world_size, config.n_ports)
    vocab = NetVocab(net)
    stats: list[CapacityStat] = []

    for obs in config.observed_fractions:
        train = build_grounded_dataset(
            oracle, vocab, net, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, observed_fraction=obs,
        )
        held = build_grounded_dataset(
            oracle, vocab, net, seeds=config.eval_seeds,
            n_steps=config.eval_steps, observed_fraction=obs,
        )
        rfrac = _residual_fraction(train)
        for d in config.d_models:
            per_seed: dict[str, list[float]] = {m: [] for m in METRICS}
            for seed in config.seeds:
                m_res = _build(vocab, net, d, config.mp_rounds, seed, config.device)
                train_grounded_decoder(
                    m_res, train, objective="residual", steps=config.decoder_iters, seed=seed
                )
                _, acc_res = residual_token_accuracy(m_res, held)
                m_lik = _build(vocab, net, d, config.mp_rounds, seed, config.device)
                train_grounded_decoder(
                    m_lik, train, objective="likelihood", steps=config.decoder_iters, seed=seed
                )
                _, acc_lik = residual_token_accuracy(m_lik, held)
                per_seed["residual_gap"].append(acc_res - acc_lik)
                per_seed["residual_acc_residual"].append(acc_res)
                per_seed["residual_acc_likelihood"].append(acc_lik)
            for metric, values in per_seed.items():
                lo, hi = bootstrap_ci(values, seed=0)
                stats.append(CapacityStat(d, obs, metric, fmean(values), lo, hi, len(values)))
        # residual-fraction context is per observed-fraction (constant across d/seed); log it
        print(f"  observed_fraction={obs:.2f}: residual token fraction R = {rfrac:.3f}")
    return stats


def _build(
    vocab: NetVocab, net: NetConfig, d_model: int, mp_rounds: int, seed: int, device: str
) -> GraphRSSMWorldModel:
    return build_graph_model(
        vocab, net, d_model=d_model, mp_rounds=mp_rounds, seed=seed, device=device
    )


def _print_summary(stats: list[CapacityStat]) -> None:
    print(f"  {'d_model':>7} {'obs_frac':>8} {'residual_gap':>13} {'95% CI':>20} {'>0?':>5}")
    for s in stats:
        if s.metric != GAP_METRIC:
            continue
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        flag = "yes" if s.disjoint_positive else "no"
        print(f"  {s.d_model:>7} {s.observed_fraction:>8.2f} {s.mean:>13.3f} {ci:>20} {flag:>5}")


def _plot(stats: list[CapacityStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaps = [s for s in stats if s.metric == GAP_METRIC]
    fracs = sorted({s.observed_fraction for s in gaps})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for obs in fracs:
        cells = sorted((s for s in gaps if s.observed_fraction == obs), key=lambda s: s.d_model)
        xs = [s.d_model for s in cells]
        ys = [s.mean for s in cells]
        lo = [s.ci_lo for s in cells]
        hi = [s.ci_hi for s in cells]
        (line,) = ax.plot(xs, ys, marker="o", label=f"observed={obs:.2f}")
        ax.fill_between(xs, lo, hi, alpha=0.18, color=line.get_color())
    ax.axhline(0.0, ls="--", lw=1, color="#555")
    ax.set_xlabel("model capacity (d_model)")
    ax.set_ylabel("residual - likelihood R-accuracy")
    ax.set_title("H24/S3 capacity-binding frontier (residual objective gap, 95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN8 capacity-binding frontier (H24/S3).")
    parser.add_argument("--world-size", type=int, default=40)
    parser.add_argument("--n-ports", type=int, default=5)
    parser.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--observed-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--decoder-iters", type=int, default=600)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", type=str, default="figures/en8_capacity.csv")
    args = parser.parse_args()

    cfg = EN8CapacityConfig(
        world_size=args.world_size,
        n_ports=args.n_ports,
        d_models=tuple(args.d_models),
        observed_fractions=tuple(args.observed_fractions),
        seeds=tuple(args.seeds),
        decoder_iters=args.decoder_iters,
        device=args.device,
    )
    print("EN8 capacity-binding frontier (H24/S3): residual-vs-likelihood R-acc, bootstrap CIs:")
    stats = run_en8_capacity(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
