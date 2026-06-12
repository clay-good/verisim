"""Experiment UA12 -- the operational detection characteristic (SPEC-20 §7, H92).

The operational completion of UA8. UA8 proved faithfulness load-bearing for the content-keyed
file-integrity task by the **catch rate** (recall) -- but scored a budget-limited recall only, blind
to false alarms. A real detector flags every file it predicts corrupted and a SOC gates on its
**precision** (false-alarm rate); a drifting world model mis-predicts *which* files are written, so
it both misses real corruptions and flags untouched files. UA12 scores the full confusion matrix
([`acd/host_detection.py`](../acd/host_detection.py)) -- precision, recall, F1 -- across horizon and
across the ρ-grounding budget, faithful vs free vs ρ-grounded.

Two panels of result:
  - **horizon sweep:** the faithful detector holds P = R = F1 = 1.000 at every horizon; the free
    detector's **precision collapses alongside its recall**, so the **F1 gap is wider than UA8's
    recall gap** -- the operational cost of drift was understated.
  - **the ρ-knee for F1:** grounding buys back *deployable* precision *and* recall (not just the
    catch rate) at the cheap UA9 knee -- the operating characteristic is restored sub-linearly.

The trained host `M_θ` is the free predictor (torch-gated); the metric core is torch-free. Seeded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from verisim.acd.host_detection import (
    Detection,
    detection_scores,
    grounded_detection,
    mean_detection,
)
from verisim.acd.host_integrity import make_workload, model_step, oracle_step
from verisim.hostoracle.reference import ReferenceHostOracle


@dataclass(frozen=True)
class UA12Config:
    """The horizon sweep + the ρ-knee grid for the operating-characteristic measurement."""

    horizons: tuple[int, ...] = (8, 14, 20, 26)
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    knee_horizon: int = 20  # the horizon at which the ρ-knee is measured
    driver: str = "forky"
    seeds: tuple[int, ...] = tuple(range(700, 724))

    @staticmethod
    def smoke() -> UA12Config:
        return UA12Config(
            horizons=(8, 16), rhos=(0.0, 0.5, 1.0), knee_horizon=16,
            seeds=tuple(range(700, 708)),
        )


@dataclass(frozen=True)
class HorizonRow:
    horizon: int
    faithful: dict[str, float]
    free: dict[str, float]


@dataclass(frozen=True)
class KneeRow:
    rho: float
    grounded: dict[str, float]
    mean_calls: float


@dataclass(frozen=True)
class UA12Result:
    horizon_sweep: list[HorizonRow]
    knee: list[KneeRow]
    knee_horizon: int


def run_ua12(model: object, config: UA12Config | None = None) -> UA12Result:
    """Sweep horizon (faithful vs free P/R/F1) and ρ (the grounded F1 knee)."""
    config = config or UA12Config()
    oracle = ReferenceHostOracle()
    true_step = oracle_step(oracle)
    faithful_step = oracle_step(oracle)
    free_step = model_step(model)

    horizon_sweep: list[HorizonRow] = []
    for h in config.horizons:
        f_scores: list[Detection] = []
        free_scores: list[Detection] = []
        for seed in config.seeds:
            start, actions = make_workload(seed, h, driver=config.driver, oracle=oracle)
            f_scores.append(detection_scores(faithful_step, true_step, start, actions))
            free_scores.append(detection_scores(free_step, true_step, start, actions))
        horizon_sweep.append(
            HorizonRow(h, mean_detection(f_scores), mean_detection(free_scores))
        )

    knee: list[KneeRow] = []
    for rho in config.rhos:
        scores: list[Detection] = []
        calls: list[int] = []
        for seed in config.seeds:
            start, actions = make_workload(
                seed, config.knee_horizon, driver=config.driver, oracle=oracle
            )
            det, c = grounded_detection(model, oracle, start, actions, rho)
            scores.append(det)
            calls.append(c)
        knee.append(KneeRow(rho, mean_detection(scores), fmean(calls)))

    return UA12Result(horizon_sweep, knee, config.knee_horizon)


def knee_rho(result: UA12Result, frac: float = 0.9) -> float:
    """The smallest ρ recovering ``frac`` of the faithful F1 ceiling -- the UA9 knee, for F1."""
    ceiling = max(k.grounded["f1"] for k in result.knee) or 1.0
    threshold = frac * ceiling
    for k in sorted(result.knee, key=lambda r: r.rho):
        if k.grounded["f1"] >= threshold:
            return k.rho
    return 1.0


CSV_HEADER = "kind,x,precision,recall,f1,n_flagged,mean_calls"


def write_csv(result: UA12Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for r in result.horizon_sweep:
        rows.append(
            f"faithful,{r.horizon},{r.faithful['precision']:.6f},{r.faithful['recall']:.6f},"
            f"{r.faithful['f1']:.6f},{r.faithful['n_flagged']:.4f},-1"
        )
        rows.append(
            f"free,{r.horizon},{r.free['precision']:.6f},{r.free['recall']:.6f},"
            f"{r.free['f1']:.6f},{r.free['n_flagged']:.4f},-1"
        )
    for k in result.knee:
        rows.append(
            f"grounded,{k.rho},{k.grounded['precision']:.6f},{k.grounded['recall']:.6f},"
            f"{k.grounded['f1']:.6f},{k.grounded['n_flagged']:.4f},{k.mean_calls:.4f}"
        )
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="UA12 -- the operational detection characteristic (SPEC-20 H92)."
    )
    parser.add_argument("--out", type=str, default="figures/ua12_host_detection.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from verisim.experiments.host_flagship import HostFlagshipConfig, train_host_flagship

    base = HostFlagshipConfig.smoke() if args.smoke else HostFlagshipConfig()
    model, _ = train_host_flagship(base)
    config = UA12Config.smoke() if args.smoke else UA12Config()
    result = run_ua12(model, config)

    print("UA12 / H92 -- the operational detection characteristic (precision = false-alarm cost):")
    print(f"  {'horizon':>7} {'faithful P/R/F1':>22} {'free P/R/F1':>22}  F1 gap (vs recall gap)")
    for r in result.horizon_sweep:
        fa = f"{r.faithful['precision']:.2f}/{r.faithful['recall']:.2f}/{r.faithful['f1']:.2f}"
        fr = f"{r.free['precision']:.2f}/{r.free['recall']:.2f}/{r.free['f1']:.2f}"
        f1gap = r.faithful["f1"] - r.free["f1"]
        rgap = r.faithful["recall"] - r.free["recall"]
        print(f"  {r.horizon:>7} {fa:>22} {fr:>22}  +{f1gap:.2f} (recall +{rgap:.2f})")
    print(f"\n  the ρ-knee for F1 (at horizon {result.knee_horizon}):")
    for k in result.knee:
        g = k.grounded
        print(f"    ρ={k.rho:.2f}: P/R/F1 = {g['precision']:.2f}/{g['recall']:.2f}/{g['f1']:.2f}  "
              f"({k.mean_calls:.1f} oracle calls)")
    print(f"  F1 knee (≥90% of the faithful ceiling): ρ = {knee_rho(result):.2f}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ua12 import plot_ua12

        plot_path = Path(args.plot) if args.plot else out.with_suffix(".png")
        plot_ua12(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
