"""Experiment RS2: scheduled sampling — the ``sample_prob`` tradeoff curve (SPEC-16 §5, H57).

The first of the two shipped-lever sweeps SPEC-16 §6 puts at the front of the build order. NA6
ran scheduled-sampling DAgger (self-forcing) at a *single* point (``max_sample_prob=0.8``) against
teacher forcing and noise injection; RS2 instead makes the lever itself the x-axis — sweeping
[`train_graph_model_self_forced`](../../src/verisim/netmodel/graph_train.py)'s ``max_sample_prob``
∈ {0, 0.25, 0.5, 0.75, 1.0} (0 = teacher forcing) on the structured GNN+RSSM arm and reading the
one-step exact rate ``p`` and the free-running faithful horizon ``H_free`` on the *same* axis.

The pre-registered question is H57 — **the signed bias-stability tradeoff.** Rollout-aware training
is *documented* to lower one-step ``p`` (it trains on harder, drifted inputs) while raising
``H_free`` (it sees the deploy distribution): the HS1.1 signature run in reverse, on purpose. So the
headline is the *shape* of the two curves as ``sample_prob`` rises: if ``p`` falls while ``H_free``
climbs, H57 is supported (a real, signed tradeoff); if they move together — both flat, or both up —
there is no tradeoff (rollout-aware training is a free lunch or a null, not a bias exchange). RS4
found the unrolled lever lifts horizon with ``p`` *held*, hinting the structured arm pays little
one-step cost; RS2 maps where, if anywhere, the cost shows up along the scheduled-sampling knob.

Torch-gated, ``skipif``-guarded in tests; CPU-only, deterministic, seeded
(``torch.set_num_threads(1)``); multi-seed with bootstrap CIs; the committed figure is generated on
the primary host (the SPEC-10 scale discipline).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.experiments.horizon_scaling import independence_horizon
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon

if TYPE_CHECKING:
    from verisim.net.state import NetworkState


@dataclass(frozen=True)
class RS2Config:
    """A small, fast scheduled-sampling sweep on the structured graph arm (runs on the M4 host)."""

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
    sample_probs: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)  # the scheduled-sampling x-axis
    refresh_every: int = 150
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_driver: str = "weighted"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    eval_steps: int = 32
    one_step_seeds: tuple[int, ...] = (200, 201)
    one_step_steps: int = 32
    epsilons: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)
    headline_epsilon: float = 0.3  # the graded tolerance the two-curve tradeoff is read at

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RS2Config:
        b = RS2Config()
        return RS2Config(
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
            sample_probs=tuple(d.get("sample_probs", b.sample_probs)),
            refresh_every=d.get("refresh_every", b.refresh_every),
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
    def from_json_file(path: str | Path) -> RS2Config:
        return RS2Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class SampleStat:
    """One ``sample_prob`` cell: one-step `p`, `eta0`, and `H_free` over the ε sweep (CIs/seeds)."""

    sample_prob: float
    p_one_step: float
    p_lo: float
    p_hi: float
    eta0: float
    h_free: dict[float, float] = field(default_factory=dict)
    h_lo: dict[float, float] = field(default_factory=dict)
    h_hi: dict[float, float] = field(default_factory=dict)
    n: int = 0

    def csv_rows(self) -> list[str]:
        return [
            f"{self.sample_prob:.3f},{eps:.3f},{self.h_free[eps]:.4f},{self.h_lo[eps]:.4f},"
            f"{self.h_hi[eps]:.4f},{self.p_one_step:.4f},{self.p_lo:.4f},{self.p_hi:.4f},"
            f"{self.eta0:.4f},{self.n}"
            for eps in sorted(self.h_free)
        ]


CSV_HEADER = "sample_prob,epsilon,h_free,h_lo,h_hi,p_one_step,p_lo,p_hi,eta0,n"


def run_rs2(config: RS2Config | None = None) -> list[SampleStat]:
    """Sweep ``max_sample_prob`` on the structured arm; read `p` and `H_free`(ε) per cell (H57)."""
    import random

    import torch

    from verisim.net.config import scaled_net_config
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netdelta import apply
    from verisim.netmetrics.divergence import divergence as net_divergence
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import train_graph_model_self_forced
    from verisim.netoracle import ReferenceNetworkOracle

    config = config or RS2Config()
    torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)
    hosts = net.hosts

    def train(sample_prob: float, seed: int) -> Any:
        wm = build_graph_model(
            vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds, seed=seed
        )
        train_graph_model_self_forced(
            wm, oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, steps=config.train_steps,
            refresh_every=config.refresh_every, max_sample_prob=sample_prob,
            lr=config.lr, batch_size=config.batch_size, seed=seed,
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

    def free_run_divs(wm: Any, eseed: int) -> list[float]:
        drv = NetDriver(name=config.eval_driver, config=net, rng=random.Random(eseed))
        actions = []
        st: NetworkState = NetworkState.initial(hosts)
        for _ in range(config.eval_steps):
            a = drv.sample(st)
            actions.append(a)
            st = oracle.step(st, a).state
        s_hat: NetworkState = NetworkState.initial(hosts)
        s_true: NetworkState = NetworkState.initial(hosts)
        out: list[float] = []
        for a in actions:
            s_hat = apply(s_hat, wm.predict_delta(s_hat, a))
            s_true = oracle.step(s_true, a).state
            out.append(net_divergence(s_true, s_hat))
        return out

    stats: list[SampleStat] = []
    for sp in config.sample_probs:
        ps: list[float] = []
        per_eps_hf: dict[float, list[float]] = {e: [] for e in config.epsilons}
        for model_seed in config.model_seeds:
            wm = train(sp, model_seed)
            wm.net.eval()
            ps.append(one_step_p(wm))
            rollouts = [free_run_divs(wm, e) for e in config.eval_seeds]
            for eps in config.epsilons:
                per_eps_hf[eps].append(fmean(faithful_horizon(d, eps) for d in rollouts))
        p_mean = fmean(ps)
        p_lo, p_hi = bootstrap_ci(ps, seed=0)
        hf = {e: fmean(per_eps_hf[e]) for e in config.epsilons}
        h_lo = {e: bootstrap_ci(per_eps_hf[e], seed=0)[0] for e in config.epsilons}
        h_hi = {e: bootstrap_ci(per_eps_hf[e], seed=0)[1] for e in config.epsilons}
        h_indep = independence_horizon(p_mean, cap=float(config.eval_steps))
        eta0 = (hf.get(0.0, 0.0) / h_indep) if h_indep > 0 else 0.0
        stats.append(
            SampleStat(sp, p_mean, p_lo, p_hi, eta0, hf, h_lo, h_hi, len(config.model_seeds))
        )
    return stats


def _verdict(stats: list[SampleStat], headline_eps: float) -> str:
    base = next((s for s in stats if s.sample_prob == 0.0), stats[0])
    top = max(stats, key=lambda s: s.sample_prob)
    eps = headline_eps if headline_eps in base.h_free else max(base.h_free)
    dp = top.p_one_step - base.p_one_step
    # Best horizon lift over the teacher-forced (sample_prob=0) baseline at the headline tolerance.
    best = max(stats, key=lambda s: s.h_free[eps])
    h_halfwidth = max((s.h_hi[eps] - s.h_lo[eps]) / 2 for s in stats)
    p_halfwidth = max((s.p_hi - s.p_lo) / 2 for s in stats)
    lift = best.h_free[eps] - base.h_free[eps]
    moved_h = lift > h_halfwidth
    dropped_p = dp < -p_halfwidth
    if moved_h and dropped_p:
        return (
            f"H57 SUPPORTED (the signed bias-stability tradeoff): as sample_prob 0→"
            f"{top.sample_prob:g} the one-step rate falls ({dp:+.3f}, beyond ±{p_halfwidth:.3f}) "
            f"while H_free(ε={eps:g}) rises (best +{lift:.2f} at sample_prob={best.sample_prob:g}, "
            f"beyond ±{h_halfwidth:.2f}) — the HS1.1 signature run in reverse, on purpose."
        )
    if moved_h and not dropped_p:
        return (
            f"H57 NOT a tradeoff here (a free lunch / no p-cost): scheduled sampling lifts "
            f"H_free(ε={eps:g}) (best +{lift:.2f} at sample_prob={best.sample_prob:g}, beyond "
            f"±{h_halfwidth:.2f}) while one-step p holds ({dp:+.3f}, within ±{p_halfwidth:.3f}) — "
            f"the structured arm pays no one-step cost for the horizon, so the gap was pure "
            f"exposure bias, not a bias-stability exchange (RS4's p-held result, along the knob)."
        )
    return (
        f"H57 NULL on the scheduled-sampling lever: across sample_prob 0→{top.sample_prob:g} "
        f"neither H_free(ε={eps:g}) (best +{lift:.2f}, within ±{h_halfwidth:.2f}) nor one-step "
        f"p ({dp:+.3f}, within ±{p_halfwidth:.3f}) moves beyond seed noise — no signed tradeoff, "
        f"corroborating NA6's single-point self-forced null on the structured arm."
    )


def _print_summary(stats: list[SampleStat], config: RS2Config) -> None:
    eps_grid = sorted(stats[0].h_free) if stats else []
    print("RS2 / H57 — scheduled sampling: the sample_prob tradeoff curve (structured arm):")
    head = f"  {'sample_prob':>11} {'p':>6} {'η0':>6}  " + " ".join(f"ε={e:<5g}" for e in eps_grid)
    print(head)
    for s in stats:
        cells = " ".join(f"{s.h_free[e]:>6.2f}" for e in eps_grid)
        print(f"  {s.sample_prob:>11.2f} {s.p_one_step:>6.3f} {s.eta0:>6.2f}  {cells}")
    print("  " + _verdict(stats, config.headline_epsilon))


def _plot(stats: list[SampleStat], path: Path, headline_eps: float) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_grid = sorted(stats[0].h_free) if stats else []
    eps = headline_eps if (stats and headline_eps in stats[0].h_free) else (max(eps_grid) or 0.0)
    xs = [s.sample_prob for s in stats]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel 1: the two-curve tradeoff — one-step p (purple) and H_free at the headline ε (blue).
    axp = ax1
    axp.plot(xs, [s.p_one_step for s in stats], "-o", color="#9467bd", label="one-step rate p")
    axp.fill_between(xs, [s.p_lo for s in stats], [s.p_hi for s in stats],
                     color="#9467bd", alpha=0.15)
    axp.set_xlabel("scheduled-sampling max_sample_prob (0 = teacher forcing)")
    axp.set_ylabel("one-step exact rate p", color="#9467bd")
    axp.tick_params(axis="y", labelcolor="#9467bd")
    axh = axp.twinx()
    axh.plot(xs, [s.h_free[eps] for s in stats], "-s", color="#1f77b4",
             label=f"H_free (ε={eps:g})")
    axh.fill_between(xs, [s.h_lo[eps] for s in stats], [s.h_hi[eps] for s in stats],
                     color="#1f77b4", alpha=0.15)
    axh.set_ylabel(f"free-running faithful horizon H_free (ε={eps:g})", color="#1f77b4")
    axh.tick_params(axis="y", labelcolor="#1f77b4")
    axp.set_title("the bias-stability tradeoff: p vs H_free along the knob")

    # Panel 2: H_free vs ε for each sample_prob setting (do the curves separate by knob?).
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(stats):
        col = cmap(0.1 + 0.8 * i / max(1, len(stats) - 1))
        ax2.plot(eps_grid, [s.h_free[e] for e in eps_grid], "-o", color=col,
                 label=f"sp={s.sample_prob:g} (p={s.p_one_step:.2f})")
    ax2.set_xlabel("faithfulness tolerance ε")
    ax2.set_ylabel("free-running faithful horizon H_free")
    ax2.set_title("H_free(ε) per scheduled-sampling setting")
    ax2.legend(fontsize=7, loc="upper left")
    fig.suptitle("RS2 / H57: scheduled sampling — the sample_prob tradeoff curve (structured arm)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RS2 scheduled-sampling tradeoff sweep (H57).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/rs2_sample_prob_tradeoff.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = RS2Config.from_json_file(args.config) if args.config else RS2Config()
    stats = run_rs2(cfg)
    _print_summary(stats, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for s in stats for r in s.csv_rows()]
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"), cfg.headline_epsilon)


if __name__ == "__main__":  # pragma: no cover
    main()
