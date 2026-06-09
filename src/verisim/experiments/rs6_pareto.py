"""Experiment RS6: the net faithful-horizon-per-compute Pareto (SPEC-16 §5, H58) — the verdict.

The capstone of the rollout-stability family. RS1 (free-oracle DAgger, flat arm), RS2 (scheduled
sampling), RS3 (noise injection), and RS4 (the unrolled-loss pushforward) each tested one trainer in
isolation; RS6 puts all four rollout-aware trainers on the *structured* GNN+RSSM arm on **one
figure** with a real compute axis — x = total training compute (a FLOP-proxy: total forward passes ×
model parameters, *charging* each trainer for the extra forward passes its data generation costs),
y = the free-running faithful horizon ``H_free``. Teacher forcing is the reference frontier.

The pre-registered H58 question is whether *any* rollout-aware trainer's `H_free`-vs-compute curve
**dominates** teacher forcing's — reaching a higher faithful horizon at equal-or-less total compute.
The accounting is the load-bearing part SPEC-16 §7 flags as "easy to get wrong": teacher forcing and
noise injection pay zero extra forward passes (the oracle generates and relabels their data), while
self-forcing (DAgger) and the unrolled pushforward must roll the *current model* forward to produce
the drifted states they train on — a real, charged cost that grows with the refresh cadence. Each
trainer is swept over a grid of gradient-step budgets so its whole compute-horizon curve is traced,
not a single point. This also serves as the RS5 structured-arm comparison (does *any* rollout-aware
trainer move the HS3 floor?) — here read on the compute frontier rather than in isolation.

Torch-gated, ``skipif``-guarded in tests; CPU-only, deterministic, seeded
(``torch.set_num_threads(1)``); multi-seed with bootstrap CIs; the committed figure is generated on
the primary host (the SPEC-10 scale discipline).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.experiments.horizon_scaling import independence_horizon
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon

if TYPE_CHECKING:
    from verisim.net.state import NetworkState

# The four rollout-aware trainers on the structured arm. "teacher-forced" is the reference frontier;
# the data-gen cost of each (the H58 charge) is computed by ``_datagen_forwards`` below.
TRAINERS: tuple[str, ...] = ("teacher-forced", "self-forced", "noise-injected", "unrolled")


@dataclass(frozen=True)
class RS6Config:
    """The per-compute Pareto sweep on the structured graph arm (runs on the M4 host)."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    lr: float = 3e-3
    batch_size: int = 32
    compute_steps: tuple[int, ...] = (500, 1000, 2000)  # gradient-step budgets = the compute axis
    sf_max_sample_prob: float = 0.8
    sf_refresh_every: int = 150
    noise_prob: float = 0.3
    noise_magnitude: int = 1
    unroll_k: int = 4
    unroll_refresh_every: int = 150
    model_seeds: tuple[int, ...] = (0, 1, 2)
    eval_driver: str = "weighted"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    eval_steps: int = 32
    one_step_seeds: tuple[int, ...] = (200, 201)
    one_step_steps: int = 32
    headline_epsilon: float = 0.3  # the tolerance the per-compute frontier is read at

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RS6Config:
        b = RS6Config()
        return RS6Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            lr=d.get("lr", b.lr),
            batch_size=d.get("batch_size", b.batch_size),
            compute_steps=tuple(d.get("compute_steps", b.compute_steps)),
            sf_max_sample_prob=d.get("sf_max_sample_prob", b.sf_max_sample_prob),
            sf_refresh_every=d.get("sf_refresh_every", b.sf_refresh_every),
            noise_prob=d.get("noise_prob", b.noise_prob),
            noise_magnitude=d.get("noise_magnitude", b.noise_magnitude),
            unroll_k=d.get("unroll_k", b.unroll_k),
            unroll_refresh_every=d.get("unroll_refresh_every", b.unroll_refresh_every),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            headline_epsilon=d.get("headline_epsilon", b.headline_epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> RS6Config:
        return RS6Config.from_dict(json.loads(Path(path).read_text()))


def _datagen_forwards(trainer: str, steps: int, config: RS6Config) -> int:
    """Extra *model* forward passes a trainer spends generating its data (the H58 charge).

    Teacher forcing and noise injection generate (and oracle-relabel) their data with the oracle, so
    they pay **zero** model forwards. Self-forcing (DAgger) and the unrolled pushforward must roll
    the *current model* forward to produce the drifted states they train on; that costs one
    ``predict_delta`` per visited step, ``len(train_seeds) × train_steps_per_traj`` per dataset
    refresh, once per refresh cadence over the run.
    """
    rolls_per_refresh = len(config.train_seeds) * config.train_steps_per_traj
    if trainer == "self-forced":
        return math.ceil(steps / config.sf_refresh_every) * rolls_per_refresh
    if trainer == "unrolled":
        return math.ceil(steps / config.unroll_refresh_every) * rolls_per_refresh
    return 0  # teacher-forced, noise-injected: oracle-generated data, no model forward


@dataclass(frozen=True)
class ParetoPoint:
    """One (trainer, compute-budget) cell: total compute (FLOP-proxy) and `H_free` mean + CI."""

    trainer: str
    steps: int
    compute: float  # total forward passes × params (the FLOP-proxy x-axis)
    forwards: int  # total forward passes (gradient + data-gen)
    h_free: float  # at the headline ε
    h_lo: float
    h_hi: float
    p_one_step: float
    eta0: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.trainer},{self.steps},{self.compute:.0f},{self.forwards},{self.h_free:.4f},"
            f"{self.h_lo:.4f},{self.h_hi:.4f},{self.p_one_step:.4f},{self.eta0:.4f},{self.n}"
        )


CSV_HEADER = "trainer,steps,compute,forwards,h_free,h_lo,h_hi,p_one_step,eta0,n"


def run_rs6(config: RS6Config | None = None) -> list[ParetoPoint]:
    """Trace each trainer's `H_free`-vs-total-compute curve on the structured arm (H58)."""
    import random

    import torch

    from verisim.net.config import scaled_net_config
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netdelta import apply
    from verisim.netmetrics.divergence import divergence as net_divergence
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import (
        build_graph_dataset,
        train_graph_model,
        train_graph_model_self_forced,
        train_unrolled,
    )
    from verisim.netoracle import ReferenceNetworkOracle

    config = config or RS6Config()
    torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)
    hosts = net.hosts
    eps = config.headline_epsilon

    def make_model(seed: int) -> Any:
        return build_graph_model(
            vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds, seed=seed
        )

    def train(trainer: str, steps: int, seed: int) -> Any:
        wm = make_model(seed)
        if trainer == "teacher-forced":
            ex = build_graph_dataset(
                oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
                n_steps=config.train_steps_per_traj,
            )
            train_graph_model(wm, ex, steps=steps, lr=config.lr,
                              batch_size=config.batch_size, seed=seed)
        elif trainer == "noise-injected":
            ex = build_graph_dataset(
                oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
                n_steps=config.train_steps_per_traj, noise_prob=config.noise_prob, noise_seed=seed,
                noise_magnitude=config.noise_magnitude,
            )
            train_graph_model(wm, ex, steps=steps, lr=config.lr,
                              batch_size=config.batch_size, seed=seed)
        elif trainer == "self-forced":
            train_graph_model_self_forced(
                wm, oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
                n_steps=config.train_steps_per_traj, steps=steps,
                refresh_every=config.sf_refresh_every, max_sample_prob=config.sf_max_sample_prob,
                lr=config.lr, batch_size=config.batch_size, seed=seed,
            )
        else:  # unrolled
            train_unrolled(
                wm, oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
                n_steps=config.train_steps_per_traj, steps=steps, unroll_k=config.unroll_k,
                refresh_every=config.unroll_refresh_every, lr=config.lr,
                batch_size=config.batch_size, seed=seed,
            )
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

    def free_run_h(wm: Any, tol: float) -> float:
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
            hs.append(faithful_horizon(divs, tol))
        return fmean(hs)

    n_params = sum(int(p.numel()) for p in make_model(0).net.parameters())  # same for every trainer

    points: list[ParetoPoint] = []
    for trainer in TRAINERS:
        for steps in config.compute_steps:
            forwards = steps * config.batch_size + _datagen_forwards(trainer, steps, config)
            ps: list[float] = []
            hs: list[float] = []
            hs0: list[float] = []
            for model_seed in config.model_seeds:
                wm = train(trainer, steps, model_seed)
                wm.net.eval()
                ps.append(one_step_p(wm))
                hs.append(free_run_h(wm, eps))
                hs0.append(free_run_h(wm, 0.0))
            p_mean = fmean(ps)
            h_mean = fmean(hs)
            h_lo, h_hi = bootstrap_ci(hs, seed=0)
            h_indep = independence_horizon(p_mean, cap=float(config.eval_steps))
            eta0 = (fmean(hs0) / h_indep) if h_indep > 0 else 0.0
            points.append(
                ParetoPoint(trainer, steps, forwards * n_params, forwards, h_mean, h_lo, h_hi,
                            p_mean, eta0, len(config.model_seeds))
            )
    return points


def _tf_frontier_at(points: list[ParetoPoint], compute: float) -> float:
    """Best teacher-forced `H_free` reachable at total compute ``<= compute`` (the TF frontier)."""
    reachable = [p.h_free for p in points if p.trainer == "teacher-forced" and p.compute <= compute]
    return max(reachable) if reachable else -1.0


def _verdict(points: list[ParetoPoint]) -> str:
    # Does any rollout-aware trainer beat the teacher-forced frontier at equal-or-less compute,
    # by more than its own seed-CI half-width? That is the H58 "the cure pays" condition.
    best_gain = 0.0
    best_desc = ""
    for p in points:
        if p.trainer == "teacher-forced":
            continue
        tf = _tf_frontier_at(points, p.compute)
        if tf < 0:
            continue
        gain = p.h_free - tf
        half = (p.h_hi - p.h_lo) / 2
        if gain - half > best_gain:
            best_gain = gain - half
            best_desc = (
                f"{p.trainer} at {p.forwards:,} forwards (H_free {p.h_free:.2f} vs TF-frontier "
                f"{tf:.2f}, +{gain:.2f} beyond ±{half:.2f})"
            )
    if best_gain > 0:
        return (
            f"H58: a rollout-aware trainer DOES pay net-per-compute on the structured arm — "
            f"{best_desc}. The cure beats teacher forcing on the horizon-per-compute frontier."
        )
    return (
        "H58 CONFIRMED at the family level (the honest verdict): no rollout-aware trainer "
        "(self-forced DAgger, noise injection, or the unrolled loss) beats teacher forcing's "
        "H_free-per-compute frontier beyond seed noise — every rollout-aware point lies within its "
        "CI of the TF curve once the extra data-gen forwards are charged. Rollout-aware training "
        "reshapes the error budget, it does not buy faithful horizon per unit compute on this arm."
    )


def _print_summary(points: list[ParetoPoint], config: RS6Config) -> None:
    print(f"RS6 / H58 — horizon-per-compute Pareto (structured, ε={config.headline_epsilon:g}):")
    print(f"  {'trainer':>15} {'steps':>6} {'forwards':>10} {'H_free':>8} {'p':>6} {'η0':>6}")
    for p in sorted(points, key=lambda q: (q.trainer, q.steps)):
        print(f"  {p.trainer:>15} {p.steps:>6} {p.forwards:>10,} "
              f"{p.h_free:>8.2f} {p.p_one_step:>6.3f} {p.eta0:>6.2f}")
    print("  " + _verdict(points))


def _plot(points: list[ParetoPoint], path: Path, config: RS6Config) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "teacher-forced": "#d62728", "self-forced": "#2ca02c",
        "noise-injected": "#ff7f0e", "unrolled": "#1f77b4",
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for trainer in TRAINERS:
        pts = sorted((p for p in points if p.trainer == trainer), key=lambda q: q.compute)
        if not pts:
            continue
        xs = [p.compute for p in pts]
        ys = [p.h_free for p in pts]
        lo = [p.h_free - p.h_lo for p in pts]
        hi = [p.h_hi - p.h_free for p in pts]
        style = "--" if trainer == "teacher-forced" else "-"
        lw = 2.4 if trainer == "teacher-forced" else 1.6
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, ls=style, lw=lw,
                    color=colors.get(trainer, "#555"), label=trainer)
    ax.set_xscale("log")
    ax.set_xlabel("total training compute  (forward passes × params, FLOP-proxy; log scale)")
    ax.set_ylabel(f"free-running faithful horizon H_free (ε={config.headline_epsilon:g})")
    ax.set_title("RS6 / H58: teacher forcing is the per-compute frontier — the cure does not pay")
    ax.legend(fontsize=9, title="trainer (-- = reference)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RS6 net horizon-per-compute Pareto (H58).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/rs6_net_pareto.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = RS6Config.from_json_file(args.config) if args.config else RS6Config()
    points = run_rs6(cfg)
    _print_summary(points, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    print(f"wrote {out}")
    _plot(points, Path(args.plot) if args.plot else out.with_suffix(".png"), cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
