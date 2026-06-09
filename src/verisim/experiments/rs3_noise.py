"""Experiment RS3: noise injection — the ``noise_prob`` × magnitude grid (SPEC-16 §5, H57).

The second cheap, shipped-lever sweep. NA6 ran oracle-relabeled noise injection at a *single* point
(``noise_prob=0.3``, magnitude 1) against teacher forcing; RS3 makes the lever a 2-D response
surface — sweeping the noise branch of
[`train_graph_model`](../../src/verisim/netmodel/graph_train.py) over ``noise_prob`` *and* the
[`corrupt_state`](../../src/verisim/netmodel/graph_train.py) **magnitude** (the number of stacked
off-trajectory mutations, a knob this experiment adds). Magnitude is the GNS/Stachenfeld lesson made
measurable: too little noise fails to cover where rollout drift lands, too much corrupts the input
past anything the deploy distribution visits and only hurts the one-step map.

The free total oracle makes the lever *exact*: every corrupted state ``s̃`` is relabeled with
``O(s̃, a)``, the correct delta there — the one thing classic GNS noise injection can only
approximate. RS3 reads the ``H_free`` response surface and the ``p``-cost contour over the grid: the
pre-registered H57 question is whether any (``noise_prob``, magnitude) cell lifts the free-running
horizon over the no-noise baseline, and whether it pays a one-step ``p`` cost (the signed tradeoff)
or buys the horizon for free.

Torch-gated, ``skipif``-guarded in tests; CPU-only, deterministic, seeded
(``torch.set_num_threads(1)``); multi-seed with bootstrap CIs; the committed figure is generated on
the primary host (the SPEC-10 scale discipline).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.experiments.horizon_scaling import independence_horizon
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon

if TYPE_CHECKING:
    from verisim.net.state import NetworkState


@dataclass(frozen=True)
class RS3Config:
    """A small, fast noise_prob × magnitude grid on the structured graph arm (runs on the M4)."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    train_steps: int = 1000
    lr: float = 3e-3
    batch_size: int = 32
    noise_probs: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45)  # the noise-rate axis (0 = no noise)
    magnitudes: tuple[int, ...] = (1, 2, 3)  # stacked off-trajectory mutations per corruption
    model_seeds: tuple[int, ...] = (0, 1, 2)
    eval_driver: str = "weighted"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    eval_steps: int = 32
    one_step_seeds: tuple[int, ...] = (200, 201)
    one_step_steps: int = 32
    epsilons: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)
    headline_epsilon: float = 0.3  # the graded tolerance the response surface is read at

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RS3Config:
        b = RS3Config()
        return RS3Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            train_steps=d.get("train_steps", b.train_steps),
            lr=d.get("lr", b.lr),
            batch_size=d.get("batch_size", b.batch_size),
            noise_probs=tuple(d.get("noise_probs", b.noise_probs)),
            magnitudes=tuple(d.get("magnitudes", b.magnitudes)),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            epsilons=tuple(d.get("epsilons", b.epsilons)),
            headline_epsilon=d.get("headline_epsilon", b.headline_epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> RS3Config:
        return RS3Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CellStat:
    """One (noise_prob, magnitude) grid cell: one-step `p`, `eta0`, `H_free` at the headline ε."""

    noise_prob: float
    magnitude: int
    p_one_step: float
    p_lo: float
    p_hi: float
    eta0: float
    h_free: float  # at the headline ε
    h_lo: float
    h_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.noise_prob:.3f},{self.magnitude},{self.h_free:.4f},{self.h_lo:.4f},"
            f"{self.h_hi:.4f},{self.p_one_step:.4f},{self.p_lo:.4f},{self.p_hi:.4f},"
            f"{self.eta0:.4f},{self.n}"
        )


CSV_HEADER = "noise_prob,magnitude,h_free,h_lo,h_hi,p_one_step,p_lo,p_hi,eta0,n"


def run_rs3(config: RS3Config | None = None) -> list[CellStat]:
    """Sweep the (noise_prob, magnitude) grid on the structured arm; H_free + p-cost (H57)."""
    import random

    import torch

    from verisim.net.config import scaled_net_config
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netdelta import apply
    from verisim.netmetrics.divergence import divergence as net_divergence
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model
    from verisim.netoracle import ReferenceNetworkOracle

    config = config or RS3Config()
    torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)
    hosts = net.hosts
    headline_eps = (
        config.headline_epsilon
        if config.headline_epsilon in config.epsilons
        else config.epsilons[-1]
    )

    def train(noise_prob: float, magnitude: int, seed: int) -> Any:
        wm = build_graph_model(
            vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds, seed=seed
        )
        ex = build_graph_dataset(
            oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, noise_prob=noise_prob, noise_seed=seed,
            noise_magnitude=magnitude,
        )
        train_graph_model(wm, ex, steps=config.train_steps, lr=config.lr,
                          batch_size=config.batch_size, seed=seed)
        return wm

    def one_step_p(wm: Any) -> float:
        correct = total = 0
        for seed in config.one_step_seeds:
            drv = NetDriver(name=config.eval_driver, config=net, rng=random.Random(seed))
            state: NetworkState = NetworkState.initial(hosts)
            for _ in range(config.one_step_steps):
                action = drv.sample(state)
                true_next = oracle.step(state, action).state
                correct += int(apply(state, wm.predict_delta(state, action)) == true_next)
                total += 1
                state = true_next
        return correct / total if total else 0.0

    def free_run_h(wm: Any, eps: float) -> float:
        hs = []
        for eseed in config.eval_seeds:
            drv = NetDriver(name=config.eval_driver, config=net, rng=random.Random(eseed))
            actions = []
            st: NetworkState = NetworkState.initial(hosts)
            for _ in range(config.eval_steps):
                a = drv.sample(st)
                actions.append(a)
                st = oracle.step(st, a).state
            s_hat: NetworkState = NetworkState.initial(hosts)
            s_true: NetworkState = NetworkState.initial(hosts)
            divs: list[float] = []
            for a in actions:
                s_hat = apply(s_hat, wm.predict_delta(s_hat, a))
                s_true = oracle.step(s_true, a).state
                divs.append(net_divergence(s_true, s_hat))
            hs.append(faithful_horizon(divs, eps))
        return fmean(hs)

    stats: list[CellStat] = []
    for noise_prob in config.noise_probs:
        # magnitude is irrelevant when noise_prob == 0 (no corruption is applied); collapse to a
        # single no-noise baseline cell so the baseline column is not redundantly retrained.
        mags = config.magnitudes if noise_prob > 0.0 else (config.magnitudes[0],)
        for magnitude in mags:
            ps: list[float] = []
            hs0: list[float] = []
            hsH: list[float] = []
            for model_seed in config.model_seeds:
                wm = train(noise_prob, magnitude, model_seed)
                wm.net.eval()
                ps.append(one_step_p(wm))
                hs0.append(free_run_h(wm, 0.0))
                hsH.append(free_run_h(wm, headline_eps))
            p_mean = fmean(ps)
            p_lo, p_hi = bootstrap_ci(ps, seed=0)
            h_mean = fmean(hsH)
            h_lo, h_hi = bootstrap_ci(hsH, seed=0)
            h_indep = independence_horizon(p_mean, cap=float(config.eval_steps))
            eta0 = (fmean(hs0) / h_indep) if h_indep > 0 else 0.0
            stats.append(
                CellStat(noise_prob, magnitude, p_mean, p_lo, p_hi, eta0, h_mean, h_lo, h_hi,
                         len(config.model_seeds))
            )
    return stats


def _verdict(stats: list[CellStat]) -> str:
    base = next((s for s in stats if s.noise_prob == 0.0), stats[0])
    best = max(stats, key=lambda s: s.h_free)
    h_halfwidth = max((s.h_hi - s.h_lo) / 2 for s in stats)
    p_halfwidth = max((s.p_hi - s.p_lo) / 2 for s in stats)
    lift = best.h_free - base.h_free
    dp = best.p_one_step - base.p_one_step
    cell = f"(noise_prob={best.noise_prob:g}, magnitude={best.magnitude})"
    if lift <= h_halfwidth:
        return (
            f"H57 NULL on the noise lever: no (noise_prob, magnitude) cell lifts H_free over the "
            f"no-noise baseline beyond seed noise — best {cell} +{lift:.2f}, within "
            f"±{h_halfwidth:.2f}. Oracle-relabeled noise augmentation, at any rate or magnitude, "
            f"does not buy free-running horizon on the structured arm (corroborating NA6's null)."
        )
    cost = (
        f"at a one-step cost ({dp:+.3f}, beyond ±{p_halfwidth:.3f}) — the signed bias-stability "
        f"tradeoff"
        if dp < -p_halfwidth
        else f"with one-step p held ({dp:+.3f}, within ±{p_halfwidth:.3f}) — a free-lunch lift, "
        f"not a bias-stability tradeoff"
    )
    return (
        f"H57: noise injection lifts H_free best at {cell} (+{lift:.2f} over no-noise, beyond "
        f"±{h_halfwidth:.2f}) {cost}."
    )


def _print_summary(stats: list[CellStat], config: RS3Config) -> None:
    print("RS3 / H57 — noise injection: the noise_prob × magnitude grid (structured arm):")
    print(f"  headline ε={config.headline_epsilon:g}; cell = (noise_prob, magnitude) -> H_free [p]")
    for s in stats:
        print(f"  noise_prob={s.noise_prob:>4.2f} mag={s.magnitude}  "
              f"H_free={s.h_free:>6.2f} [{s.h_lo:.2f},{s.h_hi:.2f}]  p={s.p_one_step:.3f} "
              f"η0={s.eta0:.2f}")
    print("  " + _verdict(stats))


def _plot(stats: list[CellStat], path: Path, config: RS3Config) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nps = sorted({s.noise_prob for s in stats})
    mags = sorted({s.magnitude for s in stats})
    base = next((s for s in stats if s.noise_prob == 0.0), stats[0])

    def cell(noise_prob: float, magnitude: int, attr: str) -> float:
        # noise_prob == 0 has a single magnitude cell; broadcast it across the magnitude row.
        for s in stats:
            if s.noise_prob == noise_prob and (s.magnitude == magnitude or noise_prob == 0.0):
                return float(getattr(s, attr))
        return float("nan")

    h_grid = [[cell(np_, m, "h_free") for np_ in nps] for m in mags]
    p_grid = [[cell(np_, m, "p_one_step") for np_ in nps] for m in mags]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    im1 = ax1.imshow(h_grid, origin="lower", aspect="auto", cmap="viridis")
    ax1.set_xticks(range(len(nps)), [f"{x:g}" for x in nps])
    ax1.set_yticks(range(len(mags)), [str(m) for m in mags])
    ax1.set_xlabel("noise_prob")
    ax1.set_ylabel("corruption magnitude")
    ax1.set_title(f"H_free response surface (ε={config.headline_epsilon:g})")
    for i, m in enumerate(mags):
        for j, np_ in enumerate(nps):
            ax1.text(j, i, f"{cell(np_, m, 'h_free'):.1f}", ha="center", va="center",
                     color="w", fontsize=8)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(p_grid, origin="lower", aspect="auto", cmap="magma")
    ax2.set_xticks(range(len(nps)), [f"{x:g}" for x in nps])
    ax2.set_yticks(range(len(mags)), [str(m) for m in mags])
    ax2.set_xlabel("noise_prob")
    ax2.set_ylabel("corruption magnitude")
    ax2.set_title(f"one-step p cost (no-noise baseline p={base.p_one_step:.2f})")
    for i, m in enumerate(mags):
        for j, np_ in enumerate(nps):
            ax2.text(j, i, f"{cell(np_, m, 'p_one_step'):.2f}", ha="center", va="center",
                     color="w", fontsize=8)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.suptitle("RS3 / H57: oracle-relabeled noise injection — the noise_prob × magnitude grid")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RS3 noise_prob × magnitude grid (H57).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/rs3_noise_surface.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = RS3Config.from_json_file(args.config) if args.config else RS3Config()
    stats = run_rs3(cfg)
    _print_summary(stats, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"), cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
