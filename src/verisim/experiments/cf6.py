"""Experiment CF6: the trained-`M_θ` conformalizability check — is the REAL belief_var calibrated?

SPEC-15's committed CF core (CF1/CF4) ran on a **controlled** signal: a stand-in whose correlation
with the true divergence is a knob (`corr = 0.9` the calibrated `belief_var` archetype, `corr = 0`
the uncalibrated decode-entropy archetype). The deferred LP7 trained-arm bet is whether the *real*
model's uncertainty signal behaves like the calibrated stand-in. CF6 closes it: it trains the real
structured graph arm and uses its actual RSSM `belief_var` (`predict_delta_with_uncertainty`) as the
conformal score, head-to-head against the two controlled stand-ins on the identical calibrator.

The honest finding this is built around: on the **network** graph arm the real `belief_var` is *not*
calibrated — it is weakly **anti**-correlated with the step's divergence (high posterior variance
does not predict a breach). So CF6 does not validate the optimistic stand-in; it **instantiates
H53's mechanism on the real arm**. Conformal *validity* is signal-agnostic, so the real-`belief_var`
trigger still hits coverage (H50). But conformal *efficiency* is the signal's conformalizability,
and the real network `belief_var` sits at the **uncalibrated** end of that axis — it buys ≈ 0 oracle
budget over fixed, where the calibrated stand-in buys ~0.43 (CF1). The calibrated stand-in is the
*achievable best case*, not the real network signal; the host `belief_var` (the EH2/H9 win) is the
known positive contrast, so conformalizability is world/arm-dependent — a refinement of H53, not a
refutation. The actionable reading: a conformal trigger on the network arm needs a better-calibrated
signal than its own RSSM variance.

CPU-only, deterministic, seeded; trains a real graph arm (torch), so it is `skipif`-guarded. The two
stand-in arms are the CF1/CF4 controlled signals (`cf_common`), run as reference anchors.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.conformal.calibrate import calibrate_threshold, undetected_rate
from verisim.experiments.cf_common import ScoredStep, breaches, mean, quantile
from verisim.metrics.aggregate import bootstrap_ci

if TYPE_CHECKING:
    from verisim.net.state import NetworkState

# The controlled stand-in arms (CF1/CF4 archetypes), as reference anchors beside the real signal.
STANDIN_ARMS: tuple[tuple[str, float], ...] = (
    ("calibrated stand-in", 0.9),
    ("uncalibrated stand-in", 0.0),
)


@dataclass(frozen=True)
class CF6Config:
    """A small, fast trained-arm conformalizability instance (real `belief_var` is the headline)."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 800
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    drafter_alpha: float = 0.8  # the controlled stand-in drafter's per-step accuracy
    breach_quantile: float = 0.6  # ε = this quantile of divergence (~40% breach rate)
    headline_alpha: float = 0.10
    pool_rollouts: int = 24
    pool_steps: int = 60
    max_window: int = 12
    n_score_bins: int = 5  # for the score↔divergence conformalizability slope

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CF6Config:
        b = CF6Config()
        return CF6Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
            drafter_alpha=d.get("drafter_alpha", b.drafter_alpha),
            breach_quantile=d.get("breach_quantile", b.breach_quantile),
            headline_alpha=d.get("headline_alpha", b.headline_alpha),
            pool_rollouts=d.get("pool_rollouts", b.pool_rollouts),
            pool_steps=d.get("pool_steps", b.pool_steps),
            max_window=d.get("max_window", b.max_window),
            n_score_bins=d.get("n_score_bins", b.n_score_bins),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CF6Config:
        return CF6Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CF6Stat:
    """One arm: coverage (validity), ρ saved (efficiency), the conformalizability slope (H53)."""

    arm: str
    test_undetected: float  # empirical undetected-breach rate at τ_α (the H50 coverage gate)
    conformal_rho: float
    fixed_rho: float
    rho_saved: float  # fixed_rho - conformal_rho (the H51 efficiency)
    saved_lo: float
    saved_hi: float
    score_div_slope: float  # div(high-score bin) − div(low-score bin): conformalizability (H53)
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.arm},{self.test_undetected:.6f},{self.conformal_rho:.6f},{self.fixed_rho:.6f},"
            f"{self.rho_saved:.6f},{self.saved_lo:.6f},{self.saved_hi:.6f},"
            f"{self.score_div_slope:.6f},{self.n}"
        )


CSV_HEADER = (
    "arm,test_undetected,conformal_rho,fixed_rho,rho_saved,saved_lo,saved_hi,score_div_slope,n"
)


def _calibrate_pool(
    raw: list[ScoredStep], config: CF6Config
) -> tuple[float, float, float, float]:
    """Run the CF1 split-conformal calibration on one (score, divergence) pool at the headline α.

    Returns ``(test_undetected, conformal_rho, fixed_rho, score_div_slope)``.
    """
    eps = quantile([p.divergence for p in raw], config.breach_quantile)
    half = len(raw) // 2
    cal, test = raw[:half], raw[half:]
    cs = [p.score for p in cal]
    cb = breaches(cal, eps)
    ts = [p.score for p in test]
    tb = breaches(test, eps)
    breach_rate = sum(tb) / len(tb) if tb else 0.0
    th = calibrate_threshold(cs, cb, config.headline_alpha)
    conf_rho = sum(1 for x in ts if x > th.tau) / len(ts) if ts else 0.0
    fixed_rho = (
        max(0.0, 1.0 - config.headline_alpha / breach_rate) if breach_rate > 0 else 0.0
    )
    test_undet = undetected_rate(ts, tb, th.tau)
    # Conformalizability slope: mean divergence in the top vs bottom score bin.
    ordered = sorted(raw, key=lambda p: p.score)
    k = max(1, len(ordered) // config.n_score_bins)
    lo_div = fmean(p.divergence for p in ordered[:k])
    hi_div = fmean(p.divergence for p in ordered[-k:])
    return test_undet, conf_rho, fixed_rho, hi_div - lo_div


def run_cf6(config: CF6Config | None = None) -> list[CF6Stat]:
    """Train the real graph arm; compare its `belief_var` conformalizability to the stand-ins."""
    import random

    import torch

    from verisim.experiments.cf_common import exchangeable_pool
    from verisim.experiments.sr_common import net_world
    from verisim.net.config import scaled_net_config
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netdelta import apply
    from verisim.netmetrics.divergence import divergence as net_divergence
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model
    from verisim.netoracle import ReferenceNetworkOracle

    config = config or CF6Config()
    torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)
    hosts = net.hosts
    world = net_world(n_hosts=config.n_hosts, n_ports=config.n_ports, driver=config.train_driver)

    def real_belief_var_pool(wm: Any, seed: int) -> list[ScoredStep]:
        """Free-run the real arm from fresh anchors; record (belief_var, divergence) per window."""
        rng = random.Random(seed)
        raw: list[ScoredStep] = []
        for r in range(config.pool_rollouts):
            drv = NetDriver(name=config.train_driver, config=net, rng=random.Random(seed + r))
            st: NetworkState = NetworkState.initial(hosts)
            actions = []
            for _ in range(config.pool_steps):
                a = drv.sample(st)
                actions.append(a)
                st = oracle.step(st, a).state
            i = 0
            while i < len(actions):
                length = rng.randint(1, config.max_window)
                window = actions[i : i + length]
                anchor: NetworkState = NetworkState.initial(hosts)
                for a in actions[:i]:
                    anchor = oracle.step(anchor, a).state
                s_hat = anchor
                s_true = anchor
                bv_last = 0.0
                for a in window:
                    delta, bv = wm.predict_delta_with_uncertainty(s_hat, a)
                    s_hat = apply(s_hat, delta)
                    s_true = oracle.step(s_true, a).state
                    bv_last = bv
                raw.append(ScoredStep(bv_last, net_divergence(s_true, s_hat), depth=0))
                i += length
        rng.shuffle(raw)
        return raw

    # Accumulate per-arm metrics across model seeds.
    acc: dict[str, dict[str, list[float]]] = {}

    def _record(arm: str, undet: float, conf: float, fixed: float, slope: float) -> None:
        d = acc.setdefault(arm, {"undet": [], "conf": [], "fixed": [], "saved": [], "slope": []})
        d["undet"].append(undet)
        d["conf"].append(conf)
        d["fixed"].append(fixed)
        d["saved"].append(fixed - conf)
        d["slope"].append(slope)

    for model_seed in config.model_seeds:
        # The real trained graph arm.
        wm = build_graph_model(
            vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds,
            seed=model_seed,
        )
        examples = build_graph_dataset(
            oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj,
        )
        train_graph_model(wm, examples, steps=config.graph_iters, seed=model_seed)
        wm.net.eval()
        with torch.no_grad():
            real_pool = real_belief_var_pool(wm, 100 + model_seed)
        _record("real belief_var", *_calibrate_pool(real_pool, config))
        # The two controlled stand-ins on the same world (CF1/CF4 reference anchors).
        for arm, corr in STANDIN_ARMS:
            pool = exchangeable_pool(
                world, drafter_alpha=config.drafter_alpha, corr=corr,
                n_rollouts=config.pool_rollouts, steps=config.pool_steps, seed=100 + model_seed,
                max_window=config.max_window,
            )
            _record(arm, *_calibrate_pool(pool, config))

    stats: list[CF6Stat] = []
    for arm in ("real belief_var", *(a for a, _ in STANDIN_ARMS)):
        d = acc[arm]
        slo, shi = bootstrap_ci(d["saved"], seed=0)
        stats.append(
            CF6Stat(
                arm, mean(d["undet"]), mean(d["conf"]), mean(d["fixed"]), mean(d["saved"]),
                slo, shi, mean(d["slope"]), len(d["saved"]),
            )
        )
    return stats


def _verdict(stats: list[CF6Stat], config: CF6Config) -> str:
    by = {s.arm: s for s in stats}
    real = by.get("real belief_var")
    cal = by.get("calibrated stand-in")
    if real is None or cal is None:
        return "inconclusive"
    a = config.headline_alpha
    covers = real.test_undetected <= a + 0.06
    real_eff = real.rho_saved
    cal_eff = cal.rho_saved
    if covers and real_eff < 0.5 * cal_eff:
        return (
            f"H50 holds, H53 instantiated on the real arm: the real network belief_var trigger "
            f"hits coverage (undetected {real.test_undetected:.2f} <= a+slack) — validity is "
            f"signal-agnostic — but it is NOT conformalizable (slope {real.score_div_slope:+.3f} ~ "
            f"0), saving only {real_eff:+.2f} rho vs the calibrated stand-in's {cal_eff:+.2f}. "
            "The real RSSM variance sits at the uncalibrated end of the H53 axis; the calibrated "
            "stand-in is the achievable best case, host belief_var (EH2) the known positive — "
            "conformalizability is world/arm-dependent."
        )
    if covers and real_eff >= 0.5 * cal_eff:
        return (
            f"the real belief_var DOES conformalize: it saves {real_eff:+.2f} rho (slope "
            f"{real.score_div_slope:+.3f}), validating the calibrated stand-in on the real arm — "
            "the CF1/CF4 headline is not a stand-in artifact."
        )
    return (
        f"the real belief_var trigger misses coverage (undetected {real.test_undetected:.2f} > a): "
        "inspect the calibrator on the real signal."
    )


def _print_summary(stats: list[CF6Stat], config: CF6Config) -> None:
    print("CF6 / the trained-M_θ conformalizability check — does the real belief_var conformalize?")
    print(f"  headline α = {config.headline_alpha:.2f}")
    print(f"  {'arm':>22} {'undet':>7} {'conf ρ':>7} {'fixed ρ':>8} {'saved':>8} {'slope':>8}")
    for s in stats:
        print(
            f"  {s.arm:>22} {s.test_undetected:>7.3f} {s.conformal_rho:>7.3f} {s.fixed_rho:>8.3f} "
            f"{s.rho_saved:>+8.3f} {s.score_div_slope:>+8.3f}"
        )
    print("  verdict: " + _verdict(stats, config))


def _plot(stats: list[CF6Stat], config: CF6Config, path: Path) -> None:  # pragma: no cover - plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [s.arm for s in stats]
    colors = {
        "real belief_var": "#9467bd",
        "calibrated stand-in": "#1f77b4",
        "uncalibrated stand-in": "#d62728",
    }
    bar_colors = [colors.get(a, "#555") for a in arms]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    x = range(len(arms))
    saved = [s.rho_saved for s in stats]
    err = [[s.rho_saved - s.saved_lo for s in stats], [s.saved_hi - s.rho_saved for s in stats]]
    ax1.bar(x, saved, 0.6, yerr=err, color=bar_colors, capsize=4)
    ax1.axhline(0.0, color="#333", lw=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([a.replace(" ", "\n") for a in arms], fontsize=8)
    ax1.set_ylabel(f"oracle budget ρ saved vs fixed (α={config.headline_alpha:.2f})")
    ax1.set_title("efficiency: the real belief_var saves ~0, like the uncalibrated stand-in")

    slopes = [s.score_div_slope for s in stats]
    ax2.bar(x, slopes, 0.6, color=bar_colors)
    ax2.axhline(0.0, color="#333", lw=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([a.replace(" ", "\n") for a in arms], fontsize=8)
    ax2.set_ylabel("score↔divergence slope (conformalizability)")
    ax2.set_title("the real network belief_var is not calibrated (slope ≈ 0)")
    fig.suptitle("CF6 / H53 on the real arm: validity is signal-agnostic, efficiency is the signal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CF6 conformalizability of the real belief_var.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cf6_real_signal.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CF6Config.from_json_file(args.config) if args.config else CF6Config()
    stats = run_cf6(cfg)
    _print_summary(stats, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, cfg, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
