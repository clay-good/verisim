"""SPEC-21 CP0/CP4/CP5 -- the faithfulness-for-control scale law (one pipeline, one dial).

The harness that sweeps the SPEC-10 capacity ladder *through* the SPEC-20 measurements to ask not
"does the boundary hold" but "where does the boundary go" (SPEC-21 §4). One device-agnostic, seeded
pipeline per rung:

    train + freeze a host M_θ at scale S  ->  the drift profile at S  ->  the per-task
    faithful-vs-free gap (the load-bearing signal) + the cheap per-task keyed drift (the forecast)
    + the ρ-grounded knee for each load-bearing task.

The **CPU-proven / GPU-ready contract** (SPEC-21 §5) is the load-bearing discipline: the *only*
difference between the CI smoke run and the committed GPU run is the ladder in the config -- no code
path differs. The CPU core proves every component is correct, deterministic, and composes; the GPU
extends the same pipeline to the capacity range where the frontier's *motion* becomes measurable.

This module ships **CP0** (`run_scale_law`), **CP4** (`load_bearing_frontier` + `forecast_check` +
`frontier_verdict`, the CS1/CS2 reducers + H87/H88/H89 verdicts), and **CP5** (`estimate_cost` +
`dry_run` + the `--config`/`--dry-run` entry point, the GPU-readiness gate). The committed scale law
(CS1/CS2/CS3) is the GPU run of this same pipeline. CPU-local; CI runs the smoke ladder.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verisim.cue.tasks import (
    TASK_SUITE,
    TaskGap,
    TaskGapConfig,
    keyed_drift,
    task_gap,
    task_knee_rho,
)
from verisim.experiments.horizon_scaling import ModelScale
from verisim.experiments.host_drift import HostDriftConfig, dimension_drift
from verisim.experiments.host_flagship import HostFlagshipConfig
from verisim.metrics.calibration import pearson, spearman

if TYPE_CHECKING:
    from verisim.hostmodel import NeuralHostWorldModel

# The CPU ladder proves the apparatus over the shipped SPEC-10 envelope (minutes-hours); the GPU run
# extends it at the top, where the capacity range is wide enough to fit a frontier trajectory.
CPU_LADDER: tuple[ModelScale, ...] = (
    ModelScale("xs", n_embd=32, n_layer=1, train_steps=2000),
    ModelScale("s", n_embd=64, n_layer=2, train_steps=2500),
    ModelScale("m", n_embd=128, n_layer=2, train_steps=3000),
    ModelScale("l", n_embd=192, n_layer=3, n_head=4, train_steps=3500),
)
GPU_LADDER: tuple[ModelScale, ...] = (
    *CPU_LADDER,
    ModelScale("xl", n_embd=320, n_layer=4, n_head=4, train_steps=6000),
    ModelScale("xxl", n_embd=512, n_layer=6, n_head=8, train_steps=10000),
    ModelScale("xxxl", n_embd=768, n_layer=8, n_head=8, train_steps=16000),
)


@dataclass(frozen=True)
class ScaleLawConfig:
    """The scale-law sweep: the ladder, the per-rung training base, and the measurement regimes."""

    scales: tuple[ModelScale, ...] = CPU_LADDER
    base: HostFlagshipConfig = field(default_factory=HostFlagshipConfig)
    task_config: TaskGapConfig = field(default_factory=TaskGapConfig)
    drift_config: HostDriftConfig = field(default_factory=HostDriftConfig)
    knee_rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    threshold: float = 0.05  # gap above which a task is "load-bearing" (the frontier contour)
    device: str = "cpu"

    @staticmethod
    def smoke() -> ScaleLawConfig:
        return ScaleLawConfig(
            scales=(
                ModelScale("xs", n_embd=32, n_layer=1, train_steps=200),
                ModelScale("s", n_embd=48, n_layer=1, train_steps=250),
            ),
            base=HostFlagshipConfig.smoke(),
            task_config=TaskGapConfig(horizon=10, workload_seeds=tuple(range(700, 708))),
            drift_config=HostDriftConfig.smoke(),
            knee_rhos=(0.0, 0.5, 1.0),
        )

    @staticmethod
    def cpu() -> ScaleLawConfig:
        """A real-but-bounded 4-rung CPU ladder (xs..l) -- the committed CPU-proven result.

        Heavier than ``smoke`` (real training, four rungs of the SPEC-10 envelope) but bounded to
        run in minutes on a laptop; it is *not* the headline (the range is too narrow to fit
        the frontier's motion -- that is the GPU run's job, SPEC-21 §5); it proves the apparatus is
        correct and composes across the real ladder.
        """
        base = dataclasses.replace(
            HostFlagshipConfig(), data_seeds=8, train_steps_per_traj=40, num_threads=1,
        )
        return ScaleLawConfig(
            scales=(
                ModelScale("xs", n_embd=32, n_layer=1, train_steps=600),
                ModelScale("s", n_embd=64, n_layer=2, train_steps=800),
                ModelScale("m", n_embd=128, n_layer=2, train_steps=1000),
                ModelScale("l", n_embd=192, n_layer=3, n_head=4, train_steps=1200),
            ),
            base=base,
            task_config=TaskGapConfig(horizon=16, workload_seeds=tuple(range(700, 716))),
            drift_config=HostDriftConfig(
                n_episodes=8, accuracy_steps=12, drift_steps=10, drift_episodes=8,
            ),
            # a fine ρ grid so the useful-knee (the cost dim) resolves below the coarse 0.1 step
            knee_rhos=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0),
        )


@dataclass(frozen=True)
class ScaleRung:
    """One rung's full measurement: drift profile, per-task gaps + keyed drift, and knees."""

    label: str
    params: int
    dimension_drift: dict[str, float]  # proc_drift / fd_drift / fs_drift (the cheap profile)
    gaps: list[TaskGap]
    keyed_drift: dict[str, float]  # per-task cheap drift (the forecast predictor)
    knees: dict[str, float]  # per-load-bearing-task useful-knee ρ


@dataclass(frozen=True)
class ScaleLawResult:
    """The full sweep across the ladder -- the substrate of the load-bearing frontier."""

    rungs: list[ScaleRung]


def _train_rung(scale: ModelScale, config: ScaleLawConfig) -> NeuralHostWorldModel:
    """Train + freeze a host M_θ at one ladder rung (the HFL0 lifecycle, scale swapped in)."""
    from verisim.experiments.host_flagship import train_host_flagship

    rung_config = dataclasses.replace(
        config.base, scale=scale, name=f"scale-law-{scale.label}"
    )
    model, _ = train_host_flagship(rung_config)
    return model


def run_scale_law(config: ScaleLawConfig | None = None) -> ScaleLawResult:
    """The CP0 pipeline: per rung, train -> drift profile -> per-task gap + keyed drift + knee."""
    config = config or ScaleLawConfig()
    rungs: list[ScaleRung] = []
    for scale in config.scales:
        model = _train_rung(scale, config)
        dim = dimension_drift(model, config.drift_config)
        gaps = [task_gap(t, model, config.task_config) for t in TASK_SUITE]
        kd = {t.name: keyed_drift(t, model, config.task_config) for t in TASK_SUITE}
        knees: dict[str, float] = {}
        for g, t in zip(gaps, TASK_SUITE, strict=True):
            if g.gap > config.threshold:
                knee, _ = task_knee_rho(t, model, config.knee_rhos, config.task_config)
                knees[t.name] = knee
        rungs.append(ScaleRung(scale.label, scale.params, dim, gaps, kd, knees))
    return ScaleLawResult(rungs)


# --- CP4: the load-bearing frontier (CS1) + the forecast (CS2) ------------------------------------


def load_bearing_frontier(result: ScaleLawResult, threshold: float = 0.05) -> list[dict[str, Any]]:
    """At each scale, which tasks are load-bearing (gap > threshold) and the frontier order.

    The frontier order = the highest task-order still load-bearing at that scale (the contour). The
    claim (H87) is that it **recedes** (decreases) with capacity -- structural tasks lose their gap
    first, leaving the deepest content.
    """
    out: list[dict[str, Any]] = []
    for rung in result.rungs:
        load_bearing = [g for g in rung.gaps if g.gap > threshold]
        frontier_order = max((g.order for g in load_bearing), default=-1)
        out.append({
            "label": rung.label,
            "params": rung.params,
            "load_bearing": [g.task for g in load_bearing],
            "frontier_order": frontier_order,
            "gaps": {g.task: g.gap for g in rung.gaps},
        })
    return out


def forecast_check(result: ScaleLawResult) -> dict[str, Any]:
    """H89: does the cheap per-task keyed drift forecast the (expensive) per-task gap?

    Pool every ``(rung, task)`` cell and correlate the gap against the cheap keyed drift. A strong
    positive correlation means the load-bearing verdict is forecastable from the cheap free-run
    profile, without running each task's full faithful-vs-free ablation.
    """
    gaps: list[float] = []
    drifts: list[float] = []
    for rung in result.rungs:
        for g in rung.gaps:
            gaps.append(g.gap)
            drifts.append(rung.keyed_drift.get(g.task, 0.0))
    return {
        "n_cells": len(gaps),
        "pearson": pearson(drifts, gaps),
        "spearman": spearman(drifts, gaps),
        # H89 supported if the cheap drift orders the gaps well (forecast is reliable)
        "forecastable": spearman(drifts, gaps) > 0.6,
    }


def cost_forecast_check(result: ScaleLawResult, threshold: float = 0.05) -> dict[str, Any]:
    """H89 extended to the *cost* dimension: does the cheap drift forecast the knee, not the gap?

    `forecast_check` (H89) showed the cheap keyed drift forecasts *whether* a task is load-bearing
    (the gap). This asks whether it also forecasts *how expensive it is to buy back* — the ρ-knee
    — on the load-bearing cells (gap > threshold, knee computed). A strong correlation means a
    cheap free-run profile predicts both the load-bearing verdict *and* the budget a task
    needs, without ever running the ρ-sweep. Pool the load-bearing ``(rung, task)`` cells and
    correlate the cheap keyed drift against the knee.
    """
    drifts: list[float] = []
    knees: list[float] = []
    for rung in result.rungs:
        for g in rung.gaps:
            if g.gap > threshold and g.task in rung.knees:
                drifts.append(rung.keyed_drift.get(g.task, 0.0))
                knees.append(rung.knees[g.task])
    if len(drifts) < 2:
        return {"n_cells": len(drifts), "spearman": 0.0, "pearson": 0.0, "cost_forecastable": False}
    return {
        "n_cells": len(drifts),
        "pearson": pearson(drifts, knees),
        "spearman": spearman(drifts, knees),
        # the cost forecast is weaker than the gap forecast (the knee is on a coarse ρ grid)
        "cost_forecastable": spearman(drifts, knees) > 0.6,
    }


def frontier_verdict(result: ScaleLawResult, threshold: float = 0.05) -> dict[str, Any]:
    """H87 (the frontier recedes with scale) + H88 (an irreducible residue remains).

    H87: the frontier order is non-increasing across the ladder (structural-first recession). H88:
    the deepest-content task (max order) keeps a material gap at *every* rung. The CPU core only
    *checks the apparatus computes these*; the committed verdict is the GPU run's wider ladder.
    """
    frontier = load_bearing_frontier(result, threshold)
    orders = [f["frontier_order"] for f in frontier]
    deepest_order = max(g.order for g in result.rungs[0].gaps)
    deepest_task = next(g.task for g in result.rungs[0].gaps if g.order == deepest_order)
    deepest_gaps = [
        next(g.gap for g in rung.gaps if g.order == deepest_order) for rung in result.rungs
    ]
    recedes = all(orders[i + 1] <= orders[i] for i in range(len(orders) - 1))
    residue = all(gap > threshold for gap in deepest_gaps)
    return {
        "frontier_order_by_scale": {f["label"]: f["frontier_order"] for f in frontier},
        "frontier_recedes_or_flat": recedes,  # H87 (monotone non-increasing)
        "deepest_task": deepest_task,
        "deepest_gap_by_scale": {
            rung.label: gap for rung, gap in zip(result.rungs, deepest_gaps, strict=True)
        },
        "irreducible_residue": residue,  # H88 (deepest content load-bearing at every rung)
    }


def knee_trajectory(
    result: ScaleLawResult, threshold: float = 0.05
) -> dict[str, list[tuple[int, float]]]:
    """The *cost* dimension of the scale law: the useful-knee ρ per load-bearing task across scale.

    The frontier (`load_bearing_frontier`) says *where* faithfulness is load-bearing; the knee says
    *how expensive it is to buy back* — the smallest consultation budget ρ recovering the
    faithful catch (UA9/H81). Per task, the ρ-knee at each rung *where the task is load-bearing*
    (gap > threshold; the harness leaves the knee at -1 elsewhere). Returns
    ``{task: [(params, knee_ρ), ...]}`` sorted by capacity — the trajectory a figure reads to ask
    whether the residue stays cheap to buy as the model scales, or gets more expensive.
    """
    traj: dict[str, list[tuple[int, float]]] = {}
    for rung in result.rungs:
        for g in rung.gaps:
            if g.gap > threshold and g.task in rung.knees:
                traj.setdefault(g.task, []).append((rung.params, rung.knees[g.task]))
    for points in traj.values():
        points.sort()
    return traj


def knee_verdict(result: ScaleLawResult, threshold: float = 0.05) -> dict[str, Any]:
    """Does the cost of buying back faithfulness change with scale (the deepest-residue knee trend)?

    For the deepest task that is load-bearing at ≥2 rungs, compare its knee at the smallest and
    largest such capacity. A *rising* knee means the irreducible residue is doubly hard — both
    load-bearing (H88) *and* increasingly expensive to buy; a *flat/falling* knee means the residue
    stays cheaply buyable at every scale. CPU-scale observation on a coarse ρ grid; the GPU run
    resolves it.
    """
    traj = knee_trajectory(result, threshold)
    multi = {t: pts for t, pts in traj.items() if len(pts) >= 2}
    if not multi:
        return {"knee_by_task": traj, "deepest_load_bearing": None, "knee_trend": "n/a"}
    order_of = {g.task: g.order for rung in result.rungs for g in rung.gaps}
    deepest = max(multi, key=lambda t: order_of.get(t, -1))
    pts = multi[deepest]
    delta = pts[-1][1] - pts[0][1]
    trend = "rising" if delta > 0.01 else ("falling" if delta < -0.01 else "flat")
    return {
        "knee_by_task": {t: {p: k for p, k in pts} for t, pts in traj.items()},
        "deepest_load_bearing": deepest,
        "knee_at_smallest": pts[0][1],
        "knee_at_largest": pts[-1][1],
        "knee_delta": delta,
        # rising -> the residue is load-bearing AND increasingly expensive to buy (sharpens H88)
        "knee_trend": trend,
    }


# --- CP5: the GPU-readiness gate (config + dry-run + cost) ----------------------------------------


def estimate_cost(config: ScaleLawConfig) -> dict[str, Any]:
    """A pre-registered, order-of-magnitude GPU-hours estimate for the committed run (SPEC-21 §5).

    Cost proxy = Σ_rung (train_steps × params), turned into GPU-hours by a stated throughput
    assumption (a single mid-range card processing ~``PARAMS_STEPS_PER_GPU_HOUR`` param-steps/hour).
    The number is an envelope for the rent decision, not a benchmark -- the assumption is stated
    can be revised, not hidden.
    """
    params_steps_per_gpu_hour = 5.0e11  # stated assumption: ~one 24-48GB card, small transformers
    per_rung = {
        s.label: {"params": s.params, "train_steps": s.train_steps,
                  "param_steps": s.params * s.train_steps}
        for s in config.scales
    }
    total_param_steps = sum(v["param_steps"] for v in per_rung.values())
    return {
        "device": config.device,
        "n_rungs": len(config.scales),
        "per_rung": per_rung,
        "total_param_steps": total_param_steps,
        "assumed_param_steps_per_gpu_hour": params_steps_per_gpu_hour,
        "estimated_gpu_hours": total_param_steps / params_steps_per_gpu_hour,
    }


def dry_run(config: ScaleLawConfig) -> dict[str, Any]:
    """Validate the GPU config WITHOUT training (CP5): shapes parse, device valid, cost estimated.

    The gate that makes the rented GPU run a *proven* program: it confirms the ladder is well-formed
    (monotone in params), the device string is recognized, the per-rung scales construct, and
    the cost estimate computes -- everything the GPU run needs, checked for free.
    """
    issues: list[str] = []
    params = [s.params for s in config.scales]
    if not config.scales:
        issues.append("empty ladder")
    if params != sorted(params):
        issues.append("ladder not monotone in params (rungs should be ordered by capacity)")
    if config.device not in {"cpu", "cuda", "mps"}:
        issues.append(f"unrecognized device {config.device!r}")
    for s in config.scales:
        if min(s.n_embd, s.n_layer, s.train_steps) <= 0:
            issues.append(f"rung {s.label} has a non-positive dimension")
        if s.n_embd % s.n_head != 0:
            issues.append(f"rung {s.label}: n_embd {s.n_embd} not divisible by n_head {s.n_head}")
    cost = estimate_cost(config)
    return {
        "ok": not issues,
        "issues": issues,
        "ladder": [s.label for s in config.scales],
        "device": config.device,
        "cost": cost,
    }


def config_from_json(path: str | Path) -> ScaleLawConfig:
    """Load a scale-law config (GPU ladder + device + base) from JSON (the one dial, SPEC-21 §5)."""
    d = json.loads(Path(path).read_text())
    scales = tuple(
        ModelScale(
            label=s["label"], n_embd=s["n_embd"], n_layer=s["n_layer"],
            n_head=s.get("n_head", 2), train_steps=s.get("train_steps", 2500),
        )
        for s in d["scales"]
    )
    base = HostFlagshipConfig()
    if "base" in d:
        base = dataclasses.replace(base, **d["base"])
    return ScaleLawConfig(
        scales=scales, base=base, device=d.get("device", "cpu"),
        threshold=d.get("threshold", 0.05),
    )


# --- output + CLI --------------------------------------------------------------------------------

CSV_HEADER = "label,params,task,order,keyed_dimension,faithful,free,gap,keyed_drift,knee_rho"


def write_csv(result: ScaleLawResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for rung in result.rungs:
        for g in rung.gaps:
            knee = rung.knees.get(g.task, -1.0)
            rows.append(
                f"{rung.label},{rung.params},{g.task},{g.order},{g.keyed_dimension},"
                f"{g.faithful:.6f},{g.free:.6f},{g.gap:.6f},"
                f"{rung.keyed_drift.get(g.task, 0.0):.6f},{knee:.4f}"
            )
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="SPEC-21 -- the faithfulness-for-control scale law (CPU-proven / GPU-ready)."
    )
    parser.add_argument("--config", type=str, default=None, help="GPU ladder config JSON")
    parser.add_argument("--out", type=str, default="figures/cs1_loadbearing_frontier.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true",
                        help="the bounded 4-rung CPU ladder (the committed CPU-proven result)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the config + estimate cost WITHOUT training (CP5)")
    args = parser.parse_args()

    if args.config:
        config = config_from_json(args.config)
    elif args.smoke:
        config = ScaleLawConfig.smoke()
    elif args.cpu:
        config = ScaleLawConfig.cpu()
    else:
        config = ScaleLawConfig()
    if args.device:
        config = dataclasses.replace(config, device=args.device)

    if args.dry_run:  # CP5: prove the GPU run will work, without training
        report = dry_run(config)
        print(json.dumps(report, indent=2))
        verdict = 'OK -- a rented GPU runs a proven program' if report['ok'] else 'FAILED'
        print(f"\nGPU-readiness: {verdict}")
        print(f"estimated cost: {report['cost']['estimated_gpu_hours']:.2f} GPU-hours "
              f"({report['cost']['n_rungs']} rungs, device={report['device']})")
        return

    result = run_scale_law(config)
    write_csv(result, args.out)
    print("the load-bearing frontier (per-task faithful-vs-free gap by scale):")
    for rung in result.rungs:
        cells = "  ".join(f"{g.task.split('-')[0][:5]}={g.gap:+.2f}" for g in rung.gaps)
        print(f"  {rung.label:4s} (params={rung.params:>7d}):  {cells}")
    fv = frontier_verdict(result, config.threshold)
    fc = forecast_check(result)
    print(f"H87 frontier recedes/flat: {fv['frontier_recedes_or_flat']}  "
          f"order-by-scale={fv['frontier_order_by_scale']}")
    print(f"H88 irreducible residue ({fv['deepest_task']}): {fv['irreducible_residue']}  "
          f"gap-by-scale={ {k: round(v, 2) for k, v in fv['deepest_gap_by_scale'].items()} }")
    print(f"H89 forecast (cheap drift -> gap): spearman={fc['spearman']:+.3f}  "
          f"forecastable={fc['forecastable']}")
    kv = knee_verdict(result, config.threshold)
    if kv["deepest_load_bearing"]:
        print(f"knee (cost to buy back faithfulness) on {kv['deepest_load_bearing']}: "
              f"ρ {kv['knee_at_smallest']:.2f} -> {kv['knee_at_largest']:.2f} "
              f"({kv['knee_trend']}) across scale")
    cf = cost_forecast_check(result, config.threshold)
    print(f"cost forecast (cheap drift -> knee): spearman={cf['spearman']:+.3f}  "
          f"cost_forecastable={cf['cost_forecastable']}")


if __name__ == "__main__":  # pragma: no cover
    main()
