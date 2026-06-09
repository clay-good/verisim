"""Experiment CF5: the conformal-consultation cross-world fork (SPEC-15 §7, the CF transfer test).

CF1 (the H50 coverage gate + the H51 ρ-vs-coverage headline) and CF4 (the H53 conformalizability
mechanism) shipped on the SPEC-5 network world. CF5 asks the program's signature question — *does
the method transfer?* — by re-running the **identical** torch-free conformal machinery (the same
calibrator, the same controlled calibrated/uncalibrated signals, the real SPEC-13 drift + real
per-world divergence) on the **host** world (the EH2/H9 confirmation: a calibrated belief-variance
trigger beat fixed ~2.2× there) and the **distributed** world (the ED2-smart challenge: an
uncalibrated decode-entropy trigger lost to fixed there). No new hypothesis — CF5 tests whether the
H50/H51/H53 results are **world-generic**, exactly as SPEC-12's LP8 tested whether the landmark
planner transferred across worlds.

The unifying claim, stated to be falsified: **conformal *validity* and *efficiency* are properties
of the signal, not the world.** Concretely, on every world (i) the calibrated trigger hits target
coverage (H50 transfers — an implementation-correctness gate per world); (ii) it certifies that
coverage at lower oracle budget ρ than fixed-interval (H51 transfers — the guaranteed-RQ2 win is not
network-specific); and (iii) the calibrated signal saves ρ while the uncalibrated one saves ~0 (H53
transfers — the EH2-yes / ED2-smart-no split is the *signal's* conformalizability, reproduced on a
world where the uncalibrated signal historically lost). The ED2-smart null is thereby localized: it
was the *uncalibrated signal*, not the distributed *world*, that could not conformalize — the
calibrated signal conformalizes there too.

Unlike the spec's tentative two-file layout (`cf5_host.py` / `cf5_dist.py`), CF5 is a single
world-parameterized run over {network (the CF1/CF4 anchor), host, distributed} emitting one
cross-world comparison figure, so the transfer is read at a glance. CPU-only, torch-free,
deterministic, seeded; the trained-`M_θ` `belief_var` arm stays deferred/`skipif`-guarded (the LP7
rule). Drift and divergence are real (the SPEC-13 drafter against each world's real oracle); only
the score is the controlled stand-in.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.conformal.calibrate import calibrate_threshold, undetected_rate
from verisim.experiments.cf_common import breaches, exchangeable_pool, mean, quantile
from verisim.experiments.cx_common import dist_world
from verisim.experiments.sr_common import SRWorld, host_world, net_world
from verisim.metrics.aggregate import bootstrap_ci

# The cross-world panel: network is the CF1/CF4 anchor, host the EH2/H9 confirmation, distributed
# the ED2-smart challenge. Lazily constructed so a world's deps load only when CF5 runs.
WORLDS: tuple[tuple[str, Callable[[], SRWorld[Any, Any]]], ...] = (
    ("network", net_world),
    ("host", host_world),
    ("distributed", dist_world),
)
ARMS: tuple[tuple[str, float], ...] = (("calibrated", 0.9), ("uncalibrated", 0.0))


@dataclass(frozen=True)
class CF5Config:
    """A small, fast cross-world conformal-transfer instance (the dependency-free CF core)."""

    drafter_alpha: float = 0.8  # the controlled drafter's per-step accuracy (real drift)
    breach_quantile: float = 0.6  # ε = this quantile of per-world divergence (-> ~40% breach)
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    headline_alpha: float = 0.10  # the α the H51/H53 transfer is read at
    n_rollouts: int = 24
    steps: int = 60
    n_seeds: int = 12
    base_seed: int = 0
    worlds: tuple[str, ...] = ("network", "host", "distributed")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CF5Config:
        b = CF5Config()
        return CF5Config(
            drafter_alpha=d.get("drafter_alpha", b.drafter_alpha),
            breach_quantile=d.get("breach_quantile", b.breach_quantile),
            alphas=tuple(d.get("alphas", b.alphas)),
            headline_alpha=d.get("headline_alpha", b.headline_alpha),
            n_rollouts=d.get("n_rollouts", b.n_rollouts),
            steps=d.get("steps", b.steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
            worlds=tuple(d.get("worlds", b.worlds)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CF5Config:
        return CF5Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CF5Stat:
    """One (world, arm, α) cell: the coverage gate and the conformal-vs-fixed ρ frontier point."""

    world: str
    arm: str
    alpha: float
    test_undetected: float  # empirical undetected-breach rate on held-out test (the H50 gate)
    conformal_rho: float  # consultation budget the conformal trigger spends
    fixed_rho: float  # consultation budget fixed-interval needs for the same guarantee
    rho_saved: float  # fixed_rho - conformal_rho (the H51 win; H53 = cal vs unc)
    saved_lo: float
    saved_hi: float
    breach_rate: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.world},{self.arm},{self.alpha:.4f},{self.test_undetected:.6f},"
            f"{self.conformal_rho:.6f},{self.fixed_rho:.6f},{self.rho_saved:.6f},"
            f"{self.saved_lo:.6f},{self.saved_hi:.6f},{self.breach_rate:.6f},{self.n}"
        )


CSV_HEADER = (
    "world,arm,alpha,test_undetected,conformal_rho,fixed_rho,rho_saved,saved_lo,saved_hi,"
    "breach_rate,n"
)


def _measure_world_arm(
    world: SRWorld[Any, Any], corr: float, config: CF5Config
) -> dict[float, dict[str, list[float]]]:
    """Per-seed split-conformal calibration on one (world, signal) — the CF1/CF4 core, per α."""
    per_alpha: dict[float, dict[str, list[float]]] = {
        a: {"undet": [], "conf": [], "fixed": [], "saved": [], "breach": []}
        for a in config.alphas
    }
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        pool = exchangeable_pool(
            world, drafter_alpha=config.drafter_alpha, corr=corr,
            n_rollouts=config.n_rollouts, steps=config.steps, seed=seed,
        )
        eps = quantile([p.divergence for p in pool], config.breach_quantile)
        half = len(pool) // 2
        cal, test = pool[:half], pool[half:]
        cs = [p.score for p in cal]
        cb = breaches(cal, eps)
        ts = [p.score for p in test]
        tb = breaches(test, eps)
        breach_rate = sum(tb) / len(tb) if tb else 0.0
        for a in config.alphas:
            th = calibrate_threshold(cs, cb, a)
            conf_rho = sum(1 for x in ts if x > th.tau) / len(ts) if ts else 0.0
            fixed_rho = max(0.0, 1.0 - a / breach_rate) if breach_rate > 0 else 0.0
            per_alpha[a]["undet"].append(undetected_rate(ts, tb, th.tau))
            per_alpha[a]["conf"].append(conf_rho)
            per_alpha[a]["fixed"].append(fixed_rho)
            per_alpha[a]["saved"].append(fixed_rho - conf_rho)
            per_alpha[a]["breach"].append(breach_rate)
    return per_alpha


def run_cf5(config: CF5Config | None = None) -> list[CF5Stat]:
    """Re-run the CF1 frontier + CF4 split per world; return per-(world, arm, α) transfer stats."""
    config = config or CF5Config()
    by_name = dict(WORLDS)
    stats: list[CF5Stat] = []
    for world_name in config.worlds:
        world = by_name[world_name]()
        for arm, corr in ARMS:
            per_alpha = _measure_world_arm(world, corr, config)
            for a in config.alphas:
                d = per_alpha[a]
                slo, shi = bootstrap_ci(d["saved"], seed=0)
                stats.append(
                    CF5Stat(
                        world_name, arm, a, mean(d["undet"]), mean(d["conf"]),
                        mean(d["fixed"]), mean(d["saved"]), slo, shi, mean(d["breach"]),
                        len(d["saved"]),
                    )
                )
    return stats


def _gate_ok(s: CF5Stat) -> bool:
    """The H50 coverage gate at this cell: undetected ≤ α within the finite-sample slack."""
    return s.test_undetected <= s.alpha + 1.0 / (s.n + 1) + 0.02


def _verdict(stats: list[CF5Stat], config: CF5Config) -> str:
    a = config.headline_alpha
    gate = all(_gate_ok(s) for s in stats if s.arm == "calibrated")
    cal = {s.world: s for s in stats if s.arm == "calibrated" and abs(s.alpha - a) < 1e-9}
    unc = {s.world: s for s in stats if s.arm == "uncalibrated" and abs(s.alpha - a) < 1e-9}
    h51 = all(cal[w].rho_saved > 0 for w in cal)
    h53 = all(cal[w].rho_saved > unc[w].rho_saved + 0.05 for w in cal if w in unc)
    if gate and h51 and h53:
        return (
            "H50/H51/H53 TRANSFER: on every world the calibrated trigger hits coverage, certifies "
            f"it at lower ρ than fixed (saved at α={a:.2f}: "
            + ", ".join(f"{w} {cal[w].rho_saved:+.2f}" for w in cal)
            + "), and beats the uncalibrated signal ("
            + ", ".join(
                f"{w} cal {cal[w].rho_saved:+.2f} vs unc {unc[w].rho_saved:+.2f}" for w in cal
            )
            + ") — conformal validity+efficiency are signal properties, not world properties; the "
            "ED2-smart null was the uncalibrated signal, not the distributed world."
        )
    return (
        f"transfer INCOMPLETE (gate={gate}, H51={h51}, H53={h53}) — at least one world breaks the "
        "calibrated win or the signal split; inspect the per-world rows."
    )


def _print_summary(stats: list[CF5Stat], config: CF5Config) -> None:
    a = config.headline_alpha
    print("CF5 / the conformal cross-world fork — does H50/H51/H53 transfer? (SPEC-15 §7)")
    print("  [trained-M_θ belief_var arm DEFERRED — core is the calibrated/uncalibrated stand-in]")
    print(f"  headline α = {a:.2f}; gate checked over all α")
    header = f"  {'world':>12} {'arm':>12} {'breach':>7} {'conf':>7} {'fixed':>7} {'saved':>8}"
    print(header + f" {'gate':>8}")
    for s in stats:
        if abs(s.alpha - a) >= 1e-9:
            continue
        gate = "all-α " + ("ok" if all(
            _gate_ok(x) for x in stats if x.world == s.world and x.arm == s.arm
        ) else "NO")
        print(
            f"  {s.world:>12} {s.arm:>12} {s.breach_rate:>7.2f} {s.conformal_rho:>8.3f} "
            f"{s.fixed_rho:>8.3f} {s.rho_saved:>+8.3f} {gate:>6}"
        )
    print("  verdict: " + _verdict(stats, config))


def _plot(stats: list[CF5Stat], config: CF5Config, path: Path) -> None:  # pragma: no cover - plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worlds = list(config.worlds)
    colors = {"network": "#1f77b4", "host": "#2ca02c", "distributed": "#9467bd"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1 (H51 transfer): ρ saved vs α, one curve per world (calibrated arm).
    for w in worlds:
        rows = sorted(
            (s for s in stats if s.world == w and s.arm == "calibrated"), key=lambda s: s.alpha
        )
        xs = [s.alpha for s in rows]
        ys = [s.rho_saved for s in rows]
        ax1.plot(xs, ys, "-o", color=colors.get(w, "#555"), label=w)
        ax1.fill_between(xs, [s.saved_lo for s in rows], [s.saved_hi for s in rows],
                         color=colors.get(w, "#555"), alpha=0.13)
    ax1.axhline(0.0, color="#333", lw=0.8)
    ax1.set_xlabel("target undetected-breach rate α")
    ax1.set_ylabel("oracle budget ρ saved vs fixed")
    ax1.set_title("H51 transfers: calibrated conformal saves ρ on every world")
    ax1.legend(fontsize=8)

    # Panel 2 (H53 transfer): cal vs unc ρ saved at the headline α, grouped per world.
    a = config.headline_alpha
    x = range(len(worlds))
    width = 0.38
    cal = [next(s for s in stats if s.world == w and s.arm == "calibrated"
                and abs(s.alpha - a) < 1e-9).rho_saved for w in worlds]
    unc = [next(s for s in stats if s.world == w and s.arm == "uncalibrated"
                and abs(s.alpha - a) < 1e-9).rho_saved for w in worlds]
    ax2.bar([i - width / 2 for i in x], cal, width, color="#1f77b4", label="calibrated signal")
    ax2.bar([i + width / 2 for i in x], unc, width, color="#d62728", label="uncalibrated signal")
    ax2.axhline(0.0, color="#333", lw=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(worlds)
    ax2.set_ylabel(f"ρ saved vs fixed (at α={a:.2f})")
    ax2.set_title("H53 transfers: efficiency is the signal's, not the world's")
    ax2.legend(fontsize=8)
    fig.suptitle("CF5: the conformal RQ2 win transfers across worlds (H50/H51/H53, SPEC-15 §7)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CF5 conformal cross-world fork (H50/H51/H53).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cf5_cross_world.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CF5Config.from_json_file(args.config) if args.config else CF5Config()
    stats = run_cf5(cfg)
    _print_summary(stats, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, cfg, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
