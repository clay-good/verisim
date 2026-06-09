"""Experiment RS4: the multi-step unrolled loss — the pushforward made exact (SPEC-16 §5).

Tests H55/H57/H58 with the fourth and last rollout-stability trainer, the one genuinely new piece of
training machinery the spec calls for. RS1 ran free-oracle DAgger on the flat ``M_θ`` (a null at CPU
scale); NA6 ran the two *shipped* levers — scheduled sampling (self-forcing) and oracle-relabeled
noise injection — on the **structured** GNN+RSSM arm (the competent-one-step / zero-horizon HS3
subject) and banked the compounding negative: neither lever lifts ``H_free`` over teacher forcing.
RS4 closes the family with the lever NA6 did not test — Brandstetter's **pushforward**, made exact
by the free total oracle.

Where self-forcing advances on the model's own prediction with a *probability* per step, the
unrolled loss re-anchors to the true trajectory every ``unroll_k`` steps and supervises **every**
step of the model's own ``unroll_k``-deep drift against the oracle's exact delta at the visited
state. ``unroll_k`` is the controlled pushforward depth: ``unroll_k = 1`` is teacher forcing
byte-for-byte; larger ``unroll_k`` supervises the model deeper into its own compounding error —
exactly the off-distribution states a free-running rollout lands in, with perfectly correct targets
the oracle-free literature can only approximate (SPEC-16 §2.2).

RS4 trains the structured arm one teacher-forced baseline + one unrolled model per ``unroll_k``, and
reads the HS3 grid on each across an ε sweep (the graded-tolerance window NA6 showed is where there
is any horizon to move): the one-step exact rate ``p``, the free-running faithful horizon
``H_free(ε)``, the compounding efficiency ``η0 = H_free(0)/H_indep``, and — the H58 axis — a
FLOP-proxy **forward-cost multiplier** ``1 + (unroll_k - 1)/2`` (the average pushforward depth
charged against the single gradient forward), so the honest verdict is ``net = H_free / cost``.

The pre-registered question (H55/H57): does the deepest rollout-aware lever move the structured
``H_free`` where the others tied teacher forcing, and at what one-step / compute cost (H58)?
*Refuted (the banked compounding negative) if* unrolling ties TF within seed-CI at every ε. The
committed result is more interesting than either branch: the unrolled loss *does* lift raw H_free
(η crosses above 1 at exact tolerance, where SF/NZ and the flat-arm DAgger all tied TF), but the
lift does **not** pay net-per-compute (H58) — the cure *reshapes* the error budget, it does not
reduce it; the GPU-scale regime is the standing open bet (RS1's caveat).

Torch-gated: imports torch lazily and is ``skipif``-guarded in tests. CPU-only, deterministic,
seeded (``torch.set_num_threads(1)``); multi-seed with bootstrap CIs; the committed figure is
generated on the primary host (the SPEC-10 scale discipline).
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
class RS4Config:
    """A small, fast unrolled-loss instance on the structured graph arm (runs on the M4 host)."""

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
    unroll_ks: tuple[int, ...] = (1, 2, 4, 8)  # pushforward depths (k=1 == teacher forcing)
    refresh_every: int = 150  # dataset refresh cadence (re-rolls the current model's drift)
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_driver: str = "weighted"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    eval_steps: int = 32
    one_step_seeds: tuple[int, ...] = (200, 201)
    one_step_steps: int = 32
    epsilons: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RS4Config:
        b = RS4Config()
        return RS4Config(
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
            unroll_ks=tuple(d.get("unroll_ks", b.unroll_ks)),
            refresh_every=d.get("refresh_every", b.refresh_every),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            epsilons=tuple(d.get("epsilons", b.epsilons)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> RS4Config:
        return RS4Config.from_dict(json.loads(Path(path).read_text()))


def cost_mult_for_k(unroll_k: int) -> float:
    """FLOP-proxy forward-cost multiplier of an unroll depth (the H58 denominator).

    Producing a supervised example at offset ``j`` within a ``unroll_k``-window costs ``j`` extra
    model forward passes to generate the drift; averaged over the window that is
    ``(unroll_k - 1)/2`` model forwards per example, charged against the one gradient forward — so
    teacher forcing (``k = 1``) is ``1.0`` and the pushforward depth ``k`` costs ``1 + (k - 1)/2``.
    """
    return 1.0 + (unroll_k - 1) / 2.0


@dataclass(frozen=True)
class ArmStat:
    """One arm: one-step `p`, `eta0`, the H58 cost multiplier, and `H_free` over the ε sweep."""

    arm: str
    unroll_k: int  # 0 for the teacher-forced baseline; the pushforward depth for unrolled arms
    p_one_step: float
    p_lo: float
    p_hi: float
    eta0: float  # H_free(ε=0) / H_indep(p) — the HS3 compounding efficiency at exact tolerance
    cost_mult: float  # the H58 forward-cost multiplier (1.0 for teacher forcing)
    h_free: dict[float, float] = field(default_factory=dict)
    h_lo: dict[float, float] = field(default_factory=dict)
    h_hi: dict[float, float] = field(default_factory=dict)
    n: int = 0

    def csv_rows(self) -> list[str]:
        return [
            f"{self.arm},{self.unroll_k},{eps:.3f},{self.h_free[eps]:.4f},{self.h_lo[eps]:.4f},"
            f"{self.h_hi[eps]:.4f},{self.p_one_step:.4f},{self.eta0:.4f},"
            f"{self.cost_mult:.4f},{self.n}"
            for eps in sorted(self.h_free)
        ]


CSV_HEADER = "arm,unroll_k,epsilon,h_free,h_lo,h_hi,p_one_step,eta0,cost_mult,n"


def run_rs4(config: RS4Config | None = None) -> list[ArmStat]:
    """Train the structured arm teacher-forced + unrolled per ``unroll_k``; read the HS3 grid."""
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
        train_unrolled,
    )
    from verisim.netoracle import ReferenceNetworkOracle

    config = config or RS4Config()
    torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)
    hosts = net.hosts

    def make_model(seed: int) -> Any:
        return build_graph_model(
            vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds, seed=seed
        )

    def train_tf(seed: int) -> Any:
        wm = make_model(seed)
        ex = build_graph_dataset(
            oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj,
        )
        train_graph_model(wm, ex, steps=config.train_steps, lr=config.lr,
                          batch_size=config.batch_size, seed=seed)
        return wm

    def train_unroll(k: int, seed: int) -> Any:
        wm = make_model(seed)
        train_unrolled(
            wm, oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, steps=config.train_steps, unroll_k=k,
            refresh_every=config.refresh_every, lr=config.lr, batch_size=config.batch_size,
            seed=seed,
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

    # (arm label, unroll_k, trainer): k=0 marks the teacher-forced baseline.
    arms: list[tuple[str, int, Any]] = [("teacher-forced", 0, train_tf)]
    arms += [
        (f"unrolled-k{k}", k, (lambda s, kk=k: train_unroll(kk, s))) for k in config.unroll_ks
    ]

    stats: list[ArmStat] = []
    for arm_name, k, trainer in arms:
        ps: list[float] = []
        per_eps_hf: dict[float, list[float]] = {e: [] for e in config.epsilons}
        for model_seed in config.model_seeds:
            wm = trainer(model_seed)
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
        cost = 1.0 if k == 0 else cost_mult_for_k(k)
        stats.append(
            ArmStat(arm_name, k, p_mean, p_lo, p_hi, eta0, cost, hf, h_lo, h_hi,
                    len(config.model_seeds))
        )
    return stats


def _verdict(stats: list[ArmStat]) -> str:
    by = {s.arm: s for s in stats}
    tf = by.get("teacher-forced")
    if tf is None:
        return "inconclusive (no teacher-forced baseline)"
    eps_grid = sorted(tf.h_free)
    # Largest raw lift any unrolled arm gets over TF at any tolerance, vs TF's seed-CI half-width.
    best_lift = 0.0
    best_desc = ""
    best_net_gain = float("-inf")  # the H58 read: best net = H_free/cost lift over TF at any ε
    for s in stats:
        if s.unroll_k == 0:
            continue
        for eps in eps_grid:
            lift = s.h_free[eps] - tf.h_free[eps]
            if lift > best_lift:
                best_lift, best_desc = lift, f"{s.arm} at ε={eps:g} (+{lift:.2f})"
            net_gain = s.h_free[eps] / s.cost_mult - tf.h_free[eps] / tf.cost_mult
            best_net_gain = max(best_net_gain, net_gain)
    tf_halfwidth = max((tf.h_hi[eps] - tf.h_lo[eps]) / 2 for eps in eps_grid)
    wall = tf.h_free.get(0.0, 0.0)
    graded = tf.h_free.get(max(eps_grid), 0.0)
    if best_lift <= tf_halfwidth:
        return (
            f"H55/H57 NOT SUPPORTED on the unrolled lever (the banked compounding negative, "
            f"completing NA6): the multi-step unrolled loss — the pushforward made exact, "
            f"supervising every step of the model's own k-deep drift with the oracle's exact label "
            f"— does not lift H_free over teacher forcing at any depth or tolerance (best lift "
            f"{best_desc or '+0.00'}, within TF's seed-CI half-width ±{tf_halfwidth:.2f}). The "
            f"H_free wall is exact-tolerance-specific (ε=0 H_free {wall:.2f}, η0={tf.eta0:.2f}; "
            f"ε={max(eps_grid):g} H_free {graded:.1f}) and survives the deepest rollout-aware "
            f"lever: fundamental compounding, not exposure bias, on the structured arm — so "
            f"net-per-compute (H58) only worsens with depth. GPU-scale is the standing open bet."
        )
    net_clause = (
        f"and it pays net-per-compute (best net gain +{best_net_gain:.2f} after the H58 charge)"
        if best_net_gain > 0
        else (
            f"but does NOT pay net-per-compute (H58): the cost charge erases it (best net gain "
            f"{best_net_gain:+.2f}) — the cure reshapes the error budget, it does not reduce it"
        )
    )
    return (
        f"H55/H57 SUPPORTED on the unrolled lever: the pushforward made exact lifts H_free over "
        f"teacher forcing — best {best_desc}, beyond TF's seed-CI half-width (±{tf_halfwidth:.2f}) "
        f"— {net_clause}."
    )


def _print_summary(stats: list[ArmStat]) -> None:
    eps_grid = sorted(stats[0].h_free) if stats else []
    print("RS4 / H55,H57,H58 — the unrolled loss (pushforward made exact) on the structured arm:")
    head = (f"  {'arm':>15} {'p':>6} {'η0':>6} {'cost':>5}  "
            + " ".join(f"ε={e:<5g}" for e in eps_grid))
    print(head)
    for s in stats:
        cells = " ".join(f"{s.h_free[e]:>6.2f}" for e in eps_grid)
        print(f"  {s.arm:>15} {s.p_one_step:>6.3f} {s.eta0:>6.2f} {s.cost_mult:>5.2f}  {cells}")
    print("  " + _verdict(stats))


def _plot(stats: list[ArmStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_grid = sorted(stats[0].h_free) if stats else []
    tf = next((s for s in stats if s.unroll_k == 0), None)
    unrolled = [s for s in stats if s.unroll_k != 0]
    cmap = plt.get_cmap("viridis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel 1: H_free vs ε, teacher forcing vs each unroll depth — do the curves separate?
    if tf is not None:
        ys = [tf.h_free[e] for e in eps_grid]
        lo = [tf.h_free[e] - tf.h_lo[e] for e in eps_grid]
        hi = [tf.h_hi[e] - tf.h_free[e] for e in eps_grid]
        ax1.errorbar(eps_grid, ys, yerr=[lo, hi], marker="s", capsize=3, color="#d62728",
                     lw=2.0, label=f"teacher-forced (p={tf.p_one_step:.2f})")
    for i, s in enumerate(unrolled):
        col = cmap(0.15 + 0.7 * i / max(1, len(unrolled) - 1))
        ax1.plot(eps_grid, [s.h_free[e] for e in eps_grid], marker="o", color=col,
                 label=f"k={s.unroll_k} (p={s.p_one_step:.2f}, {s.cost_mult:.1f}×)")
    ax1.set_xlabel("faithfulness tolerance ε")
    ax1.set_ylabel("free-running faithful horizon H_free")
    ax1.set_title("the pushforward made exact vs teacher forcing")
    ax1.legend(fontsize=8, loc="upper left")

    # Panel 2: the H58 net-per-compute read — H_free/cost at the most graded ε, per arm.
    eps_top = max(eps_grid) if eps_grid else 0.0
    x = range(len(stats))
    ax2.bar(x, [s.h_free.get(eps_top, 0.0) / s.cost_mult for s in stats], 0.6,
            color=["#d62728" if s.unroll_k == 0 else cmap(0.5) for s in stats])
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([s.arm.replace("-", "-\n") for s in stats], fontsize=7)
    ax2.set_ylabel(f"net H_free / forward-cost  (ε={eps_top:g})")
    ax2.set_title("H58: deeper unroll pays compute — does it buy net horizon?")
    fig.suptitle(
        "RS4 / H55,H57,H58: the unrolled loss lifts raw H_free but not net-per-compute (CPU scale)"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RS4 multi-step unrolled loss (the pushforward).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/rs4_unroll_depth.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = RS4Config.from_json_file(args.config) if args.config else RS4Config()
    stats = run_rs4(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for s in stats for r in s.csv_rows()]
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
