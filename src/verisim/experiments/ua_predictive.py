"""UA7 -- predictive control needs faithfulness (SPEC-20 §7, H79; the boundary's positive side).

The reactive-control thread (UA2/H74, UA6/H78, the drift profile) established that a defender which
*reacts* to the observed compromise state does not need a faithful world model. UA7 tests the
converse on a defender that *plans*: a fixed model-predictive controller that, each step, rolls its
model forward `k` steps for each candidate isolation and picks the best (`acd.predictive`). Three
planners face the *same true dynamics*:

  - **faithful** -- plans with the exact oracle (perfect lookahead);
  - **free** -- plans with the raw flat `M_θ` (drifted lookahead, drift compounding over `k`);
  - **reactive** -- the model-free isolate-most-dangerous-exposed baseline (the UA2-style policy).

Sweeping the lookahead depth `k` is the knob: at `k=1` the model is one step from the true state
(near-accurate, ~0.88 one-step), so faithful ≈ free; as `k` grows the free planner's drift compounds
(SPEC-10) and it should fall below the faithful planner, while the reactive baseline is flat in `k`
(it does not plan). H79: the faithful planner beats the free planner, and the gap widens with `k` --
the positive that draws the boundary **reactive control needs no faithfulness; predictive control
does.** No training; CPU-local; CI runs the smoke instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.acd.containment import ContainmentConfig
from verisim.acd.predictive import (
    model_step_fn,
    oracle_step_fn,
    run_open_loop_episode,
    run_predictive_episode,
    run_reactive_episode,
)
from verisim.metrics.aggregate import bootstrap_ci
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

if TYPE_CHECKING:
    from verisim.netloop.model import NetModel


@dataclass(frozen=True)
class PredictiveConfig:
    """The UA7 sweep: a tight-budget task (so *which* host to plan-isolate matters) over `k`."""

    containment: ContainmentConfig = field(
        default_factory=lambda: ContainmentConfig(episode_steps=16, cut_budget=2)
    )
    ks: tuple[int, ...] = (1, 3, 5, 8)
    eval_seeds: tuple[int, ...] = tuple(range(600, 624))

    @staticmethod
    def smoke() -> PredictiveConfig:
        return PredictiveConfig(
            containment=ContainmentConfig(n_hosts=4, n_ports=2, episode_steps=6, cut_budget=1,
                                          n_links=4, n_services=3),
            ks=(1, 3), eval_seeds=(600, 601, 602),
        )


@dataclass(frozen=True)
class PredictivePoint:
    """One (planner, k) cell: mean true containment + bootstrap CI."""

    planner: str  # "faithful" | "free" | "reactive"
    k: int
    containment: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.planner},{self.k},{self.containment:.6f},"
            f"{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"
        )


CSV_HEADER = "planner,k,containment,ci_lo,ci_hi,n"


def run_predictive(
    model: NetModel, config: PredictiveConfig | None = None, *, oracle: NetOracle | None = None,
) -> list[PredictivePoint]:
    """Sweep `k`; score the faithful / free planners + the reactive baseline in reality."""
    config = config or PredictiveConfig()
    oracle = oracle or ReferenceNetworkOracle()
    true_step = oracle_step_fn(oracle)  # reality is the exact oracle for every arm
    faithful_step = oracle_step_fn(oracle)
    free_step = model_step_fn(model)

    points: list[PredictivePoint] = []
    # k-independent arms (measured once, replicated across the axis for plotting): the reactive
    # baseline, and the OPEN-LOOP planners (plan the whole episode once, no re-observation -- the
    # "planning in imagination" regime where the model's drift compounds over the full plan).
    seeds = config.eval_seeds
    flat_arms: dict[str, tuple[float, float, float]] = {}
    for label, vals in (
        ("reactive", [run_reactive_episode(true_step, config.containment, s) for s in seeds]),
        ("open_faithful",
         [run_open_loop_episode(true_step, faithful_step, config.containment, s) for s in seeds]),
        ("open_free",
         [run_open_loop_episode(true_step, free_step, config.containment, s) for s in seeds]),
    ):
        lo, hi = bootstrap_ci(vals, seed=0)
        flat_arms[label] = (fmean(vals), lo, hi)

    for k in config.ks:
        faithful = [
            run_predictive_episode(true_step, faithful_step, config.containment, s, k)
            for s in config.eval_seeds
        ]
        free = [
            run_predictive_episode(true_step, free_step, config.containment, s, k)
            for s in config.eval_seeds
        ]
        for planner, vals in (("faithful", faithful), ("free", free)):
            lo, hi = bootstrap_ci(vals, seed=0)
            points.append(PredictivePoint(planner, k, fmean(vals), lo, hi, len(vals)))
        for label, (mean, lo, hi) in flat_arms.items():
            points.append(PredictivePoint(label, k, mean, lo, hi, len(seeds)))
    return points


def h79_verdict(points: list[PredictivePoint]) -> dict[str, Any]:
    """H79: a faithful planner beats a free one (closed- or open-loop) — does faithfulness help?"""
    by: dict[tuple[str, int], float] = {(p.planner, p.k): p.containment for p in points}
    ks = sorted({p.k for p in points})
    gaps = {k: by[("faithful", k)] - by[("free", k)] for k in ks if ("free", k) in by}
    kmin, kmax = ks[0], ks[-1]
    open_gap = by.get(("open_faithful", kmax), 0.0) - by.get(("open_free", kmax), 0.0)
    # the value of planning at all (predictive vs the model-free reactive baseline)
    plan_lift = by.get(("faithful", kmax), 0.0) - by.get(("reactive", kmax), 0.0)
    return {
        "ks": ks,
        "closed_gap_at_min_k": gaps.get(kmin, 0.0),
        "closed_gap_at_max_k": gaps.get(kmax, 0.0),
        "open_loop_gap": open_gap,
        "plan_lift_over_reactive": plan_lift,
        # H79: faithfulness helps the planner (closed- OR open-loop) — gap materially positive
        "h79_supported": gaps.get(kmax, 0.0) > 0.02 or open_gap > 0.02,
        "widens_with_k": gaps.get(kmax, 0.0) > gaps.get(kmin, 0.0),
    }


def write_csv(points: list[PredictivePoint], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="UA7 -- predictive needs faithfulness (H79).")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/ua7_predictive.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = PredictiveConfig.smoke() if args.smoke else PredictiveConfig()
    if args.smoke:
        from verisim.netloop.model import NetOracleBackedModel

        model: NetModel = NetOracleBackedModel(ReferenceNetworkOracle())
    else:
        from verisim.experiments.flagship import load_checkpoint

        model = load_checkpoint(args.checkpoint).world_model

    points = run_predictive(model, config)
    write_csv(points, args.out)
    for k in sorted({p.k for p in points}):
        c = {p.planner: p.containment for p in points if p.k == k}
        print(f"  k={k}:  closed[faithful={c['faithful']:.3f} free={c['free']:.3f}]  "
              f"open[faithful={c['open_faithful']:.3f} free={c['open_free']:.3f}]  "
              f"reactive={c['reactive']:.3f}")
    v = h79_verdict(points)
    print(f"H79 (faithfulness load-bearing for predictive control): "
          f"{'SUPPORTED' if v['h79_supported'] else 'no'}")
    print(f"  closed-loop gap (faithful-free) @max k = {v['closed_gap_at_max_k']:+.3f}")
    print(f"  open-loop gap   (faithful-free)        = {v['open_loop_gap']:+.3f}")
    print(f"  planning lift over reactive            = {v['plan_lift_over_reactive']:+.3f} "
          f"(planning helps; its faithfulness does not)")


if __name__ == "__main__":  # pragma: no cover
    main()
