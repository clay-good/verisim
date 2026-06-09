"""Experiment NA6: does decoder-side rollout-stability training lift the structured arm's H_free?

The redirected SPEC-14 headline — and a banked negative. NA0 refuted the original NA1 (hint
supervision is redundant: the processor already executes the reachability propagation) and NA5
confirmed at the rollout level that the HS3 `H_free = 0` wall lives in the **decoder/rollout**, not
the processor. The move the redirection demands is to attack the decoder side at *training* time —
the SPEC-16 RS-family lever, now on the structured graph arm, the arm that (unlike the flat RS1
trained) is the *competent one-step / zero-horizon* regime where there is genuinely a gap to cure
(EN4: the graph arm beats the flat arm ~6.6x on one-step delta-exact, yet HS3 pins horizon to 0).

NA6 trains the structured arm three ways at fixed capacity and reads the HS3 grid (`p`, `H_free`):

  - **teacher-forced (TF)** — the HS3 baseline (`train_graph_model`), trained on the oracle's true
    states, rolled out free-running;
  - **self-forced (SF)** — scheduled-sampling DAgger (`train_graph_model_self_forced`): roll the
    current model forward on its *own* predictions and oracle-relabel each drifted state, ramping
    the self-sample probability over training so late training matches the deployment drift;
  - **noise-injected (NZ)** — oracle-relabeled state-noise augmentation (`build_graph_dataset`'s
    `noise_prob`): broaden the input with one-mutation corruptions, each relabeled by the oracle.

The result is read across an **ε sweep**, exposing structure HS3's single ε = 0 number hides: the
`H_free = 0` wall is **exact-tolerance-specific** — at ε = 0 a competent-but-imperfect one-step
predictor (`p ≈ 0.55`) misses instantly so `H_free → 0` (and `η < 1`), but at graded tolerance the
same arm free-runs many steps. The headline question (H46, redirected): do SF or NZ lift `H_free`
over TF — at exact tolerance, or in the graded window where compounding could be cured? *Refuted if*
they tie TF within seed noise — the **bankable compounding negative** the spec pre-registered as "as
valuable as the positive": even on the competent structured arm, with the decoder-side fix NA0/NA5
pointed to, the exact-tolerance wall is fundamental compounding, not a curable train/deploy mismatch
(sharpening RS1's flat-arm null and the HS3 η < 1 verdict). CPU-only, deterministic, seeded;
multi-seed with bootstrap CIs. The scale caveat is RS1's: whether the cure pays at GPU scale is the
open question.
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
class NA6Config:
    """A small, fast decoder-side rollout-stability instance on the structured graph arm."""

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
    noise_prob: float = 0.3  # the NZ arm's oracle-relabeled state-noise rate (§6.3)
    sf_max_sample_prob: float = 0.8  # the SF arm's terminal self-sample probability
    sf_refresh_every: int = 150  # SF dataset refresh cadence
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_driver: str = "weighted"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    eval_steps: int = 32
    one_step_seeds: tuple[int, ...] = (200, 201)
    one_step_steps: int = 32
    epsilons: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> NA6Config:
        b = NA6Config()
        return NA6Config(
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
            noise_prob=d.get("noise_prob", b.noise_prob),
            sf_max_sample_prob=d.get("sf_max_sample_prob", b.sf_max_sample_prob),
            sf_refresh_every=d.get("sf_refresh_every", b.sf_refresh_every),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            epsilons=tuple(d.get("epsilons", b.epsilons)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> NA6Config:
        return NA6Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ArmStat:
    """One training arm: one-step `p`, `eta` at e=0, and `H_free` over the e sweep (CIs/seeds)."""

    arm: str
    p_one_step: float
    p_lo: float
    p_hi: float
    eta0: float  # H_free(ε=0) / H_indep(p) — the HS3 compounding-efficiency at exact tolerance
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


def run_na6(config: NA6Config | None = None) -> list[ArmStat]:
    """Train the structured arm TF / SF / NZ; read `p`, `H_free`(ε), `η` on each (the HS3 grid)."""
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
    )
    from verisim.netoracle import ReferenceNetworkOracle

    config = config or NA6Config()
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

    def train_nz(seed: int) -> Any:
        wm = make_model(seed)
        ex = build_graph_dataset(
            oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, noise_prob=config.noise_prob, noise_seed=seed,
        )
        train_graph_model(wm, ex, steps=config.train_steps, lr=config.lr,
                          batch_size=config.batch_size, seed=seed)
        return wm

    def train_sf(seed: int) -> Any:
        wm = make_model(seed)
        train_graph_model_self_forced(
            wm, oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, steps=config.train_steps,
            refresh_every=config.sf_refresh_every, max_sample_prob=config.sf_max_sample_prob,
            lr=config.lr, batch_size=config.batch_size, seed=seed,
        )
        return wm

    arms = (("teacher-forced", train_tf), ("self-forced", train_sf), ("noise-injected", train_nz))

    def one_step_p(wm: Any) -> float:
        correct = total = 0
        for seed in config.one_step_seeds:
            drv = NetDriver(name=config.eval_driver, config=net, rng=random.Random(seed))
            state = NetworkState.initial(hosts)
            for _ in range(config.one_step_steps):
                action = drv.sample(state)
                true_next = oracle.step(state, action).state
                correct += int(apply(state, wm.predict_delta(state, action)) == true_next)
                total += 1
                state = true_next
        return correct / total if total else 0.0

    def free_run_divs(wm: Any, eseed: int) -> list[float]:
        drv = NetDriver(name=config.eval_driver, config=net, rng=random.Random(eseed))
        s_true: NetworkState = NetworkState.initial(hosts)
        actions = []
        st = s_true
        for _ in range(config.eval_steps):
            a = drv.sample(st)
            actions.append(a)
            st = oracle.step(st, a).state
        s_hat: NetworkState = NetworkState.initial(hosts)
        s_true = NetworkState.initial(hosts)
        out: list[float] = []
        for a in actions:
            s_hat = apply(s_hat, wm.predict_delta(s_hat, a))
            s_true = oracle.step(s_true, a).state
            out.append(net_divergence(s_true, s_hat))
        return out

    stats: list[ArmStat] = []
    for arm_name, trainer in arms:
        ps: list[float] = []
        # per (seed, eval_seed) divergence trajectories, reduced per eps over the pooled rollouts
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
            ArmStat(arm_name, p_mean, p_lo, p_hi, eta0, hf, h_lo, h_hi, len(config.model_seeds))
        )
    return stats


def _verdict(stats: list[ArmStat], config: NA6Config) -> str:
    by = {s.arm: s for s in stats}
    tf = by.get("teacher-forced")
    if tf is None:
        return "inconclusive (no teacher-forced baseline)"
    eps_grid = sorted(tf.h_free)
    # Largest lift a decoder-side arm achieves over TF at any tolerance, vs TF's CI half-width.
    best_lift = 0.0
    best_desc = ""
    for s in stats:
        if s.arm == "teacher-forced":
            continue
        for eps in eps_grid:
            lift = s.h_free[eps] - tf.h_free[eps]
            if lift > best_lift:
                best_lift = lift
                best_desc = f"{s.arm} at ε={eps:g} (+{lift:.2f})"
    tf_halfwidth = max(
        (tf.h_hi[eps] - tf.h_lo[eps]) / 2 for eps in eps_grid
    )
    wall = tf.h_free.get(0.0, 0.0)
    graded = tf.h_free.get(max(eps_grid), 0.0)
    if best_lift <= tf_halfwidth:
        return (
            f"H46-redirected NOT SUPPORTED (the banked compounding negative): the decoder-side "
            f"fixes (self-forced DAgger, noise injection) do not lift H_free over TF at "
            f"any tolerance — best lift {best_desc or '+0.00'}, within TF's own seed-CI half-width "
            f"(±{tf_halfwidth:.2f}). The H_free=0 wall is exact-tolerance-specific (ε=0 H_free "
            f"{wall:.2f}, η0={tf.eta0:.2f}<1; ε={max(eps_grid):g} H_free {graded:.1f}): at exact "
            "tolerance a competent-but-imperfect one-step arm misses instantly, and decoder-side "
            "fix neither lifts p nor buys rollout horizon — fundamental compounding, not exposure "
            "bias, on the structured arm too (sharpening RS1). Scale caveat: GPU-scale is the bet."
        )
    return (
        f"H46-redirected SUPPORTED: a decoder-side fix lifts H_free over teacher-forced — best "
        f"{best_desc}, beyond TF's seed-CI half-width (±{tf_halfwidth:.2f}). The decoder-side "
        "rollout-stability training NA0/NA5 pointed to converts the structured arm's one-step "
        "advantage into horizon."
    )


def _print_summary(stats: list[ArmStat], config: NA6Config) -> None:
    eps_grid = sorted(stats[0].h_free) if stats else []
    print("NA6 / H46-redirected — does decoder-side rollout-stability training lift graph H_free?")
    head = f"  {'arm':>15} {'p':>6} {'η0':>6}  " + " ".join(f"ε={e:<5g}" for e in eps_grid)
    print(head)
    for s in stats:
        cells = " ".join(f"{s.h_free[e]:>6.2f}" for e in eps_grid)
        print(f"  {s.arm:>15} {s.p_one_step:>6.3f} {s.eta0:>6.2f}  {cells}")
    print("  " + _verdict(stats, config))


def _plot(stats: list[ArmStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_grid = sorted(stats[0].h_free) if stats else []
    colors = {"teacher-forced": "#1f77b4", "self-forced": "#2ca02c", "noise-injected": "#ff7f0e"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    for s in stats:
        ys = [s.h_free[e] for e in eps_grid]
        lo = [s.h_free[e] - s.h_lo[e] for e in eps_grid]
        hi = [s.h_hi[e] - s.h_free[e] for e in eps_grid]
        ax1.errorbar(eps_grid, ys, yerr=[lo, hi], marker="o", capsize=3,
                     color=colors.get(s.arm, "#555"), label=f"{s.arm} (p={s.p_one_step:.2f})")
    ax1.set_xlabel("faithfulness tolerance ε")
    ax1.set_ylabel("free-running faithful horizon H_free")
    ax1.set_title("decoder-side fixes tie teacher forcing at every ε (the negative)")
    ax1.legend(fontsize=8, loc="upper left")

    # Panel 2: the exact-tolerance wall — H_free(ε=0) and η0 per arm (all pinned at the floor).
    x = range(len(stats))
    width = 0.6
    ax2.bar(x, [s.h_free.get(0.0, 0.0) for s in stats], width,
            color=[colors.get(s.arm, "#555") for s in stats])
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([s.arm.replace("-", "-\n") for s in stats], fontsize=8)
    ax2.set_ylabel("H_free at exact tolerance (ε=0)")
    ax2.set_title("the H_free=0 wall is exact-tolerance-specific, and it stays")
    fig.suptitle("NA6 / H46-redirected: decoder-side training does not lift the structured H_free")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="NA6 decoder-side rollout-stability training.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/na6_decode_training.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = NA6Config.from_json_file(args.config) if args.config else NA6Config()
    stats = run_na6(cfg)
    _print_summary(stats, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for s in stats for r in s.csv_rows()]
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
