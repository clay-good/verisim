"""Experiment RS7: cross-world fork — rollout-aware trainers on the HOST world (SPEC-16 §5, H59).

The final SPEC-16 experiment, and the one that closes the spec. RS1-RS6 ran the rollout-stability
trainer family on the **network** world (flat and structured arms) and settled it: free-oracle
DAgger does not cure the flat arm; scheduled sampling and noise injection buy nothing on the
structured arm; only the unrolled-loss pushforward lifts *raw* `H_free`, and even that does not beat
teacher forcing once compute is charged (RS6). RS7 asks the H59 question the program reserved for
last: **does that picture transfer across worlds?** It re-runs the four-arm comparison —
teacher-forced, self-forced (DAgger), noise-injected, and the unrolled pushforward — on the **host**
world's structured factored arm (the SPEC-6 GNN+RSSM host proposer, EH4/HS2 subject), reading the
same free-running faithful horizon `H_free` across an ε sweep, with multi-seed bootstrap CIs.

The host world is a genuinely different test, not a re-run: HS2 (SPEC-6) showed its floor is
re-lowered and its headroom re-opened relative to the network arm, and its oracle is the composed
`ReferenceHostOracle` over a different (process/fd/mount) state grammar. So a transferring verdict
means the RS findings are a property of the oracle-grounded *loop*, not of one world×proposer cell;
a diverging verdict localizes them. *Refuted (H59)* if the winning network trainer behaves
differently here — itself a result about which (world, proposer) pairs admit an exposure-bias cure,
mirroring the HS2/HS3 proposer-dependence verdicts.

Reuses host machinery wholesale: `build_host_graph_model`, the `train_host_*` trainers (incl. the
new `train_host_unrolled`), and the `run_host_rollout` / `faithful_horizon` eval (the EH4-drift
apparatus). Torch-gated, ``skipif``-guarded in tests; CPU-only, deterministic, seeded
(``torch.set_num_threads(1)``); the committed figure is generated on the primary host (SPEC-10).
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
    from verisim.host.action import HostAction
    from verisim.host.state import HostState


@dataclass(frozen=True)
class RS7Config:
    """A small, fast cross-world fork on the host factored arm (runs on the M4 host)."""

    max_pid: int = 64
    train_driver: str = "forky"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 1000  # total gradient-step budget, equal across arms
    lr: float = 3e-3
    batch_size: int = 32
    noise_prob: float = 0.3
    noise_magnitude: int = 1
    sf_rounds: int = 4
    sf_sample_prob: float = 0.8
    unroll_rounds: int = 4
    unroll_k: int = 4
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_driver: str = "forky"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    eval_steps: int = 32
    one_step_seeds: tuple[int, ...] = (200, 201)
    one_step_steps: int = 32
    epsilons: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RS7Config:
        b = RS7Config()
        return RS7Config(
            max_pid=d.get("max_pid", b.max_pid),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            lr=d.get("lr", b.lr),
            batch_size=d.get("batch_size", b.batch_size),
            noise_prob=d.get("noise_prob", b.noise_prob),
            noise_magnitude=d.get("noise_magnitude", b.noise_magnitude),
            sf_rounds=d.get("sf_rounds", b.sf_rounds),
            sf_sample_prob=d.get("sf_sample_prob", b.sf_sample_prob),
            unroll_rounds=d.get("unroll_rounds", b.unroll_rounds),
            unroll_k=d.get("unroll_k", b.unroll_k),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            epsilons=tuple(d.get("epsilons", b.epsilons)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> RS7Config:
        return RS7Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ArmStat:
    """One training arm on the host arm: one-step `p`, `eta0`, and `H_free` over the ε sweep."""

    arm: str
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
            f"{self.arm},{eps:.3f},{self.h_free[eps]:.4f},{self.h_lo[eps]:.4f},"
            f"{self.h_hi[eps]:.4f},{self.p_one_step:.4f},{self.eta0:.4f},{self.n}"
            for eps in sorted(self.h_free)
        ]


CSV_HEADER = "arm,epsilon,h_free,h_lo,h_hi,p_one_step,eta0,n"
ARMS: tuple[str, ...] = ("teacher-forced", "self-forced", "noise-injected", "unrolled")


def run_rs7(config: RS7Config | None = None) -> list[ArmStat]:
    """Train the host factored arm four ways; read `p`, `H_free`(ε), `η` per arm (H59 transfer)."""
    import random

    import torch

    from verisim.host.config import DEFAULT_HOST_CONFIG
    from verisim.host.state import HostState
    from verisim.hostdata.drivers import HostDriver
    from verisim.hostloop import PartialHostOracle, run_host_rollout
    from verisim.hostmodel import HostVocab
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import (
        build_host_graph_dataset,
        train_host_graph_model,
        train_host_graph_model_self_forced,
        train_host_unrolled,
    )
    from verisim.hostoracle.reference import ReferenceHostOracle
    from verisim.loop.policy import Never

    config = config or RS7Config()
    torch.set_num_threads(1)
    oracle = ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)

    def make_model(seed: int) -> Any:
        return build_host_graph_model(
            vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
            mp_rounds=config.graph_mp_rounds, seed=seed,
        )

    def train_tf(seed: int) -> Any:
        wm = make_model(seed)
        ex = build_host_graph_dataset(
            oracle, vocab, host, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj,
        )
        train_host_graph_model(wm, ex, steps=config.graph_iters, lr=config.lr,
                               batch_size=config.batch_size, seed=seed)
        return wm

    def train_nz(seed: int) -> Any:
        wm = make_model(seed)
        ex = build_host_graph_dataset(
            oracle, vocab, host, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, noise_prob=config.noise_prob, noise_seed=seed,
        )
        train_host_graph_model(wm, ex, steps=config.graph_iters, lr=config.lr,
                               batch_size=config.batch_size, seed=seed)
        return wm

    def train_sf(seed: int) -> Any:
        wm = make_model(seed)
        train_host_graph_model_self_forced(
            wm, oracle, vocab, host, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, rounds=config.sf_rounds,
            steps_per_round=config.graph_iters // config.sf_rounds,
            sample_prob=config.sf_sample_prob, lr=config.lr, batch_size=config.batch_size,
            seed=seed,
        )
        return wm

    def train_un(seed: int) -> Any:
        wm = make_model(seed)
        train_host_unrolled(
            wm, oracle, vocab, host, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, rounds=config.unroll_rounds,
            steps_per_round=config.graph_iters // config.unroll_rounds, unroll_k=config.unroll_k,
            lr=config.lr, batch_size=config.batch_size, seed=seed,
        )
        return wm

    trainers = {
        "teacher-forced": train_tf, "self-forced": train_sf,
        "noise-injected": train_nz, "unrolled": train_un,
    }

    def one_step_p(wm: Any) -> float:
        correct = total = 0
        for seed in config.one_step_seeds:
            drv = HostDriver(name=config.eval_driver, config=host, rng=random.Random(seed))
            state: HostState = HostState.initial()
            for _ in range(config.one_step_steps):
                action = drv.sample(state)
                truth = oracle.step(state, action)
                correct += int(wm.predict_delta(state, action) == truth.delta)
                total += 1
                state = truth.state
        return correct / total if total else 0.0

    def eval_actions(seed: int) -> list[HostAction]:
        drv = HostDriver(name=config.eval_driver, config=host, rng=random.Random(seed))
        state = HostState.initial()
        acts: list[HostAction] = []
        for _ in range(config.eval_steps):
            a = drv.sample(state)
            acts.append(a)
            state = oracle.step(state, a).state
        return acts

    def free_run_divs(wm: Any, eseed: int) -> list[float]:
        partial = PartialHostOracle(oracle)
        rollout = run_host_rollout(
            wm, partial, HostState.initial(), eval_actions(eseed), Never(),
            epsilon=config.epsilons[0], budget=0, seed=eseed,
        )
        return [float(d) for d in rollout.divergences]

    stats: list[ArmStat] = []
    for arm in ARMS:
        trainer = trainers[arm]
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
        stats.append(
            ArmStat(arm, p_mean, p_lo, p_hi, eta0, hf, h_lo, h_hi, len(config.model_seeds))
        )
    return stats


def _verdict(stats: list[ArmStat]) -> str:
    by = {s.arm: s for s in stats}
    tf = by.get("teacher-forced")
    if tf is None:
        return "inconclusive (no teacher-forced baseline)"
    eps_grid = sorted(tf.h_free)
    best_lift = 0.0
    best_desc = ""
    for s in stats:
        if s.arm == "teacher-forced":
            continue
        for eps in eps_grid:
            lift = s.h_free[eps] - tf.h_free[eps]
            if lift > best_lift:
                best_lift, best_desc = lift, f"{s.arm} at ε={eps:g} (+{lift:.2f})"
    tf_halfwidth = max((tf.h_hi[eps] - tf.h_lo[eps]) / 2 for eps in eps_grid)
    if best_lift <= tf_halfwidth:
        return (
            f"H59: the network verdict TRANSFERS to the host world — no rollout-aware trainer "
            f"(self-forced, noise, or unrolled) lifts H_free over teacher forcing beyond seed "
            f"noise (best {best_desc or '+0.00'}, within TF's CI half-width ±{tf_halfwidth:.2f}). "
            f"The rollout-stability picture is a property of the oracle-grounded loop, not one "
            f"world: on the host arm too, the levers reshape the error budget for no horizon."
        )
    return (
        f"H59: the network verdict does NOT fully transfer — on the host arm a rollout-aware "
        f"trainer lifts H_free beyond TF's CI ({best_desc}, ±{tf_halfwidth:.2f}). The cure is "
        f"(world, proposer)-dependent: the host arm admits a lift the network arm did not, "
        f"localizing the exposure-bias cure rather than generalizing it (the H59 fork verdict)."
    )


def _print_summary(stats: list[ArmStat]) -> None:
    eps_grid = sorted(stats[0].h_free) if stats else []
    print("RS7 / H59 — rollout-aware trainers on the HOST world (cross-world transfer):")
    head = f"  {'arm':>15} {'p':>6} {'η0':>6}  " + " ".join(f"ε={e:<5g}" for e in eps_grid)
    print(head)
    for s in stats:
        cells = " ".join(f"{s.h_free[e]:>6.2f}" for e in eps_grid)
        print(f"  {s.arm:>15} {s.p_one_step:>6.3f} {s.eta0:>6.2f}  {cells}")
    print("  " + _verdict(stats))


def _plot(stats: list[ArmStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_grid = sorted(stats[0].h_free) if stats else []
    colors = {
        "teacher-forced": "#d62728", "self-forced": "#2ca02c",
        "noise-injected": "#ff7f0e", "unrolled": "#1f77b4",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for s in stats:
        ys = [s.h_free[e] for e in eps_grid]
        lo = [s.h_free[e] - s.h_lo[e] for e in eps_grid]
        hi = [s.h_hi[e] - s.h_free[e] for e in eps_grid]
        style = "--" if s.arm == "teacher-forced" else "-"
        ax1.errorbar(eps_grid, ys, yerr=[lo, hi], marker="o", capsize=3, ls=style,
                     color=colors.get(s.arm, "#555"), label=f"{s.arm} (p={s.p_one_step:.2f})")
    ax1.set_xlabel("faithfulness tolerance ε")
    ax1.set_ylabel("free-running faithful horizon H_free")
    ax1.set_title("rollout-aware trainers on the HOST factored arm")
    ax1.legend(fontsize=8, loc="upper left")

    x = range(len(stats))
    ax2.bar(x, [s.h_free.get(0.0, 0.0) for s in stats], 0.6,
            color=[colors.get(s.arm, "#555") for s in stats])
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([s.arm.replace("-", "-\n") for s in stats], fontsize=8)
    ax2.set_ylabel("H_free at exact tolerance (ε=0)")
    ax2.set_title("the exact-tolerance floor, per trainer")
    fig.suptitle("RS7 / H59: the rollout-aware trainers fork onto the host world (cross-world)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RS7 cross-world fork to the host world (H59).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/rs7_host_transfer.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = RS7Config.from_json_file(args.config) if args.config else RS7Config()
    stats = run_rs7(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for s in stats for r in s.csv_rows()]
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
