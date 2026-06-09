"""Experiment SR4: calibrated draft length & the EAGLE-2 acceptance link (H41, SPEC-13 §6).

EAGLE-2's idea: don't draft a fixed ``k`` -- draft *longer where the model is confident* and shorter
where it is not. SR4 tests whether that transfers to the verisim loop, against the exact oracle (the
check the LLM line can only approximate, SPEC-13 §2), and finds a sharp **split**:

  - **the link transfers** (H41 part a, SUPPORTED). With a calibrated signal the per-step acceptance
    -- did the drafted step stay within ``ε``, scored by the oracle -- rises monotonically with the
    confidence signal; with an uncalibrated signal (``signal_corr = 0``, the flat-arm decode-entropy
    case, ED2-smart) the calibration curve is flat. The confidence↔acceptance link the EAGLE line
    relies on is real here and *checkable against ground truth*.

  - **the policy does not** (H41 part b, REFUTED -- the headline reason). Calibrated draft
    length does **not** reduce oracle calls versus the trivial **draft-long-everywhere** policy
    (``k = k_max``). The cause is the cost inversion SPEC-13 §8 names: in LLM speculative decoding
    the
    *drafter's GPU* is the cost, so a long draft that gets rejected wastes compute and short drafts
    in
    low-confidence regions pay off. Here the **oracle is the cost**, the draft is free, and the
    verify
    **stops at the first divergence** -- so a long draft that rejects early costs no more than a
    short
    one. Calibrating ``k`` *down* in low-confidence regions only chops them into more windows (more
    oracle calls). The right policy is draft-long, accept-longest-prefix; calibration buys nothing.

This is the program's bankable-negative shape: a method (EAGLE-2 draft-length calibration) whose
*premise* (the confidence link) holds but whose *mechanism* (avoid wasted drafts) is absent because
the expensive resource is the oracle, not the GPU. SR4 reports both halves so the split is exact.

CPU-only, torch-free, deterministic, seeded (the controlled
:class:`~verisim.experiments.sr_common.VaryingDrafter`
stand-in; the trained-``M_θ`` ``belief_var``/decode-entropy arm is the deferred/``skipif``-guarded
one,
never counted -- SPEC-13 §9, the LP7 discipline).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.experiments.sr2 import SR2Config, granularity
from verisim.experiments.sr_common import SRWorld, VaryingDrafter, all_worlds, mean
from verisim.loop.speculative import speculative_rollout
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class SR4Config:
    """A small, fast calibrated-draft-length instance (the dependency-free core)."""

    alpha_easy: float = 0.95
    alpha_hard: float = 0.5
    period: int = 24  # easy/hard region length (>= k_max so a long draft stays within a region)
    k_min: int = 2
    k_max: int = 16
    g: float = 0.8  # ε = g·δ (tight enough that a single stall is detectable -- the link)
    n_steps: int = 96
    n_seeds: int = 24
    base_seed: int = 0
    n_conf_bins: int = 5

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SR4Config:
        b = SR4Config()
        return SR4Config(
            alpha_easy=d.get("alpha_easy", b.alpha_easy),
            alpha_hard=d.get("alpha_hard", b.alpha_hard),
            period=d.get("period", b.period),
            k_min=d.get("k_min", b.k_min),
            k_max=d.get("k_max", b.k_max),
            g=d.get("g", b.g),
            n_steps=d.get("n_steps", b.n_steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
            n_conf_bins=d.get("n_conf_bins", b.n_conf_bins),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> SR4Config:
        return SR4Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class SR4Stat:
    """One (world, signal-arm) cell: calibrated-k vs draft-long oracle calls, and the calibration
    slope.
    """

    world: str
    arm: str  # "calibrated" (signal_corr=1) | "uncalibrated" (signal_corr=0)
    calibrated_calls: float  # oracle calls under calibrated-k (lower is better)
    cal_lo: float
    cal_hi: float
    draftlong_calls: float  # oracle calls under draft-long-everywhere (k = k_max)
    dl_lo: float
    dl_hi: float
    calib_slope: float  # acceptance(high-conf) − acceptance(low-conf): the EAGLE-2 link
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.world},{self.arm},{self.calibrated_calls:.6f},{self.cal_lo:.6f},{self.cal_hi:.6f},"
            f"{self.draftlong_calls:.6f},{self.dl_lo:.6f},{self.dl_hi:.6f},{self.calib_slope:.6f},{self.n}"
        )


CSV_HEADER = (
    "world,arm,calibrated_calls,cal_lo,cal_hi,draftlong_calls,dl_lo,dl_hi,calib_slope,n"
)


@dataclass(frozen=True)
class CalibPoint:
    """One calibration-curve bin: mean confidence vs measured acceptance (for the figure)."""

    world: str
    arm: str
    conf_bin: float
    acceptance: float
    acc_lo: float
    acc_hi: float
    n: int


def _calibration(
    world: SRWorld[Any, Any], config: SR4Config, epsilon: float, signal_corr: float
) -> tuple[list[float], list[float]]:
    """Per-step (confidence, accepted?) pairs over all seeds -- the raw calibration data."""
    confs: list[float] = []
    accepts: list[float] = []
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        s0, actions = world.make_actions(4000 + seed, config.n_steps)
        drafter = VaryingDrafter(
            world.oracle_step, config.alpha_easy, config.alpha_hard, config.period,
            seed=seed, signal_corr=signal_corr,
        )
        truth = s0
        for t, action in enumerate(actions):
            drafted = drafter(truth, action, t, 0)
            nxt = world.oracle_step(truth, action)
            confs.append(drafter.confidence(t))
            accepts.append(1.0 if world.diverge(nxt, drafted) <= epsilon else 0.0)
            truth = nxt
    return confs, accepts


def _calls(
    world: SRWorld[Any, Any], config: SR4Config, epsilon: float, signal_corr: float
) -> tuple[list[float], list[float]]:
    """Oracle calls for calibrated-k and for draft-long-everywhere, per seed."""
    cal: list[float] = []
    dl: list[float] = []
    span = config.k_max - config.k_min
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        s0, actions = world.make_actions(4000 + seed, config.n_steps)
        drafter = VaryingDrafter(
            world.oracle_step, config.alpha_easy, config.alpha_hard, config.period,
            seed=seed, signal_corr=signal_corr,
        )

        def k_of(_s: Any, step: int, _d: Any = drafter) -> int:
            return int(config.k_min + round(span * _d.confidence(step)))
        rec_cal = speculative_rollout(
            s0, actions, drafter, world.oracle_step, world.diverge,
            k=config.k_max, epsilon=epsilon, k_of=k_of,
        )
        rec_dl = speculative_rollout(
            s0, actions, drafter, world.oracle_step, world.diverge,
            k=config.k_max, epsilon=epsilon,
        )
        cal.append(float(rec_cal.oracle_calls))
        dl.append(float(rec_dl.oracle_calls))
    return cal, dl


def calibration_curve(config: SR4Config | None = None) -> list[CalibPoint]:
    """The acceptance-vs-confidence curve per (world, arm) -- the figure's calibration panel."""
    config = config or SR4Config()
    gran_cfg = SR2Config(n_steps=config.n_steps)
    points: list[CalibPoint] = []
    for world in all_worlds():
        epsilon = config.g * granularity(world, gran_cfg)
        for arm, corr in (("calibrated", 1.0), ("uncalibrated", 0.0)):
            confs, accepts = _calibration(world, config, epsilon, corr)
            edges = [i / config.n_conf_bins for i in range(config.n_conf_bins + 1)]
            for b in range(config.n_conf_bins):
                lo_e, hi_e = edges[b], edges[b + 1]
                cell = [
                    (c, a)
                    for c, a in zip(confs, accepts, strict=True)
                    if (lo_e <= c < hi_e) or (b == config.n_conf_bins - 1 and c == hi_e)
                ]
                if not cell:
                    continue
                accs = [a for _, a in cell]
                lo, hi = bootstrap_ci(accs, seed=0)
                points.append(
                    CalibPoint(world.name, arm, mean([c for c, _ in cell]), mean(accs), lo, hi, len(accs))  # noqa: E501
                )
    return points


def run_sr4(config: SR4Config | None = None) -> list[SR4Stat]:
    """Per (world, arm): calibrated-k vs draft-long oracle calls and the calibration slope (H41)."""
    config = config or SR4Config()
    gran_cfg = SR2Config(n_steps=config.n_steps)
    stats: list[SR4Stat] = []
    for world in all_worlds():
        epsilon = config.g * granularity(world, gran_cfg)
        for arm, corr in (("calibrated", 1.0), ("uncalibrated", 0.0)):
            cal, dl = _calls(world, config, epsilon, corr)
            cl, ch = bootstrap_ci(cal, seed=0)
            dlo, dhi = bootstrap_ci(dl, seed=0)
            confs, accepts = _calibration(world, config, epsilon, corr)
            hi_acc = [a for c, a in zip(confs, accepts, strict=True) if c >= 0.6]
            lo_acc = [a for c, a in zip(confs, accepts, strict=True) if c < 0.4]
            slope = (mean(hi_acc) - mean(lo_acc)) if (hi_acc and lo_acc) else 0.0
            stats.append(
                SR4Stat(world.name, arm, mean(cal), cl, ch, mean(dl), dlo, dhi, slope, len(cal))
            )
    return stats


def _print_summary(stats: list[SR4Stat]) -> None:
    print("SR4 / H41 - calibrated draft length: the link transfers, the policy does not:")
    print("  [trained-M_θ belief_var/entropy arm DEFERRED -- committed core is the stand-in (§9)]")
    for s in stats:
        delta = s.calibrated_calls - s.draftlong_calls
        print(
            f"  {s.world:>11} {s.arm:>12}: calibrated-k calls={s.calibrated_calls:5.1f} "
            f"draft-long calls={s.draftlong_calls:5.1f} ({delta:+.1f})  slope={s.calib_slope:+.2f}"
        )
    cal = [s for s in stats if s.arm == "calibrated"]
    unc = [s for s in stats if s.arm == "uncalibrated"]
    link_real = mean([s.calib_slope for s in cal]) > mean([s.calib_slope for s in unc]) + 0.05
    policy_helps = mean([s.draftlong_calls - s.calibrated_calls for s in cal]) > 0.5
    cal_slope, unc_slope = mean([s.calib_slope for s in cal]), mean([s.calib_slope for s in unc])
    verdict = (
        f"the confidence↔acceptance link transfers (calibrated slope "
        f"{cal_slope:+.2f} / {unc_slope:+.2f} null), "
        f"but calibrated-k does NOT beat draft-long-everywhere (it uses "
        f"{mean([s.calibrated_calls - s.draftlong_calls for s in cal]):+.1f} more oracle calls) - "
        "H41 split: link real, EAGLE-2 policy does not transfer (the oracle-cost inversion, §8)"
        if link_real and not policy_helps
        else "H41 outcome differs from the pre-registered split on this drafter"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[SR4Stat], curve: list[CalibPoint], path: Path) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # Panel 1: calibrated-k vs draft-long oracle calls (calibrated arm) -- lower is better.
    worlds = sorted({s.world for s in stats})
    x = range(len(worlds))
    cal = [next(s.calibrated_calls for s in stats if s.world == w and s.arm == "calibrated")
           for w in worlds]
    dl = [next(s.draftlong_calls for s in stats if s.world == w and s.arm == "calibrated")
          for w in worlds]
    ax1.bar([i - 0.2 for i in x], dl, 0.4, color="#1f77b4", label="draft-long (k_max)")
    ax1.bar([i + 0.2 for i in x], cal, 0.4, color="#d62728", label="calibrated-k")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(worlds)
    ax1.set_ylabel("oracle calls (lower is better)")
    ax1.set_title("calibrated k costs more, not less (the §8 cost inversion)")
    ax1.legend(fontsize=8)
    # Panel 2: the calibration curve (acceptance vs confidence), one line per arm (network).
    for arm, color in (("calibrated", "#1f77b4"), ("uncalibrated", "#d62728")):
        pts = sorted(
            (p for p in curve if p.world == "network" and p.arm == arm),
            key=lambda p: p.conf_bin,
        )
        xs = [p.conf_bin for p in pts]
        ys = [p.acceptance for p in pts]
        ax2.plot(xs, ys, "-o", color=color, label=arm)
        ax2.fill_between(xs, [p.acc_lo for p in pts], [p.acc_hi for p in pts],
                         color=color, alpha=0.12)
    ax2.set_xlabel("confidence signal")
    ax2.set_ylabel("measured acceptance (oracle-scored)")
    ax2.set_title("the link is real: monotone (calibrated) vs flat (null)")
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend(fontsize=8)
    fig.suptitle("SR4 / H41: the EAGLE-2 link transfers; its draft-length policy does not")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="SR4 calibrated draft length (H41).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/sr4_calibration.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = SR4Config.from_json_file(args.config) if args.config else SR4Config()
    stats = run_sr4(cfg)
    curve = calibration_curve(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, curve, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
