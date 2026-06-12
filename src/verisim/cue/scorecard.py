"""The verisim-cue scorecard: run a world-model through the benchmark (SPEC-21 §8.2, the eval API).

The packaging ([`pack`](./pack.py)) froze the battery + emitted Croissant/datasheet; the conformance
([`conformance`](./conformance.py)) asserted its contract. This module is the *thing adopters call*:
``score_model(model)`` runs any host world-model through the ordered computer-use task suite and
returns its per-task **scorecard** -- the model's catch rate, the faithful ceiling, and, crucially,
*whether the oracle was load-bearing for that model on that task* (the property that makes the bench
the only computer-use benchmark that measures not just whether a model succeeds but whether
faithfulness was load-bearing for its success, SPEC-21 §2). It is the SPEC-18 ``score_model``
parallel for the computer-use vertical, and it backs the **model-card** the spec's §8.2 names.

A model's scorecard is the model-facing dual of the load-bearing frontier: the frontier asks "across
models of increasing capacity, where does the boundary sit?"; the scorecard asks "for *this* model,
which tasks still need the oracle?" -- and a task is load-bearing for the model iff a faithful
predictor beats it by more than the manifest threshold.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from .pack import CueManifest
from .tasks import TASK_SUITE, TaskGapConfig, task_gap

if TYPE_CHECKING:
    from verisim.hostmodel import NeuralHostWorldModel


@dataclass(frozen=True)
class TaskScore:
    """One task's scorecard entry for a scored model."""

    task: str
    order: int
    catch_rate: float  # the model's (free-predictor) catch rate on the task
    faithful_ceiling: float  # the oracle's catch rate (== 1.0 by the ground-truth-labels contract)
    gap: float  # faithful_ceiling - catch_rate
    load_bearing: bool  # the oracle is load-bearing for this model on this task (gap > threshold)


def _config_for(manifest: CueManifest) -> TaskGapConfig:
    return TaskGapConfig(
        horizon=manifest.horizon, driver=manifest.driver, workload_seeds=manifest.seeds
    )


def score_model(
    model: NeuralHostWorldModel, manifest: CueManifest | None = None,
    config: TaskGapConfig | None = None,
) -> list[TaskScore]:
    """Run ``model`` through the verisim-cue suite; return its per-task scorecard.

    The model-facing entry point: for each ordered structure->content task, measure the model's
    free-predictor catch rate against the exact faithful ceiling, and flag whether the oracle was
    load-bearing for the model (gap above the manifest threshold).
    """
    manifest = manifest or CueManifest()
    config = config or _config_for(manifest)
    scores: list[TaskScore] = []
    for task in TASK_SUITE:
        g = task_gap(task, model, config)
        scores.append(
            TaskScore(g.task, g.order, g.free, g.faithful, g.gap, g.gap > manifest.threshold)
        )
    return scores


def scorecard_headline(
    scores: list[TaskScore], manifest: CueManifest | None = None
) -> dict[str, Any]:
    """The one-line read: mean catch, the load-bearing footprint, structure-clean?."""
    manifest = manifest or CueManifest()
    return {
        "version": manifest.version_tag(),
        "mean_catch_rate": fmean([s.catch_rate for s in scores]) if scores else 0.0,
        "n_load_bearing": sum(s.load_bearing for s in scores),
        "load_bearing_tasks": [s.task for s in scores if s.load_bearing],
        # a competent model should catch the *structural* tasks (oracle not load-bearing there)
        "structure_clean": all(not s.load_bearing for s in scores if s.order == 0),
    }


def reference_scores_from_csv(
    frontier_csv: str | Path, threshold: float = 0.05
) -> dict[str, list[TaskScore]]:
    """The reference-model scorecards (one per ladder rung) from a committed CS1 frontier CSV.

    Each rung's trained `M_θ` is a reference model; its per-task ``free`` catch rate + load-bearing
    flag is its scorecard. Returns ``{rung_label: [TaskScore, ...]}`` ordered by task. ``{}`` if the
    CSV is absent.
    """
    path = Path(frontier_csv)
    if not path.exists():
        return {}
    by_rung: dict[str, list[TaskScore]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            faithful, free, gap = float(row["faithful"]), float(row["free"]), float(row["gap"])
            score = TaskScore(
                row["task"], int(row["order"]), free, faithful, gap, gap > threshold
            )
            by_rung.setdefault(row["label"], []).append(score)
    for scores in by_rung.values():
        scores.sort(key=lambda s: s.order)
    return by_rung


def model_card(manifest: CueManifest, frontier_csv: str | Path | None = None) -> str:
    """A model-card for verisim-cue (SPEC-21 §8.2): the scorecard schema + the reference scorecards.

    The standard metadata triple's third member (Croissant + datasheet + model-card). It documents
    what ``score_model`` returns and shows the reference models' scorecards -- the per-rung trained
    `M_θ` from the committed scale-law run, each with its per-task catch rate + load-bearing flag --
    so a reader sees both the schema and a worked example without retraining.
    """
    ref = reference_scores_from_csv(frontier_csv, manifest.threshold) if frontier_csv else {}
    ref_section = ""
    if ref:
        rung_lines = []
        for label, scores in ref.items():
            cells = " · ".join(
                f"{s.task.split('-')[0]} {s.catch_rate:.2f}{'*' if s.load_bearing else ''}"
                for s in scores
            )
            n_lb = sum(s.load_bearing for s in scores)
            rung_lines.append(f"| `{label}` | {cells} | {n_lb} |")
        ref_section = (
            "\n## Reference scorecards (the scale-law rungs)\n\n"
            "Each rung's trained `M_θ` scored through the suite (catch rate per task; `*` = the "
            "oracle was load-bearing for that model on that task).\n\n"
            "| model (rung) | per-task catch rate | # load-bearing |\n"
            "|---|---|---|\n" + "\n".join(rung_lines) + "\n"
        )
    task_rows = "\n".join(
        f"| {t.order} | {t.name} | {t.keyed_dimension} | catch rate ∈ [0,1], load-bearing if "
        f"gap > {manifest.threshold} |"
        for t in sorted(manifest.tasks, key=lambda s: s.order)
    )
    return f"""# Model card — verisim-cue scorecard — {manifest.version_tag()}

`score_model(model)` runs any host world-model through the ordered computer-use task suite and
returns a **scorecard**: per task, the model's catch rate, the exact faithful ceiling (1.000 by the
ground-truth-labels contract), the gap, and whether the **oracle was load-bearing** for the model on
that task (gap > {manifest.threshold}). This is the property no oracle-free computer-use bench can
report — it measures not just whether a model succeeds but whether faithfulness was load-bearing for
its success.

## Scorecard schema (per task)
| order | task | keyed dimension | scored |
|---|---|---|---|
{task_rows}
{ref_section}
## Discriminative validity (CL1 / H91)
The scorecard is a *trustworthy* frozen eval because it **stably ranks** models by faithfulness:
scoring a controlled fidelity ladder by recall over the keyed set, the ranking is rank-stable
(Kendall τ = +1.000 between disjoint seed splits) and every adjacent fidelity tier resolves above
its paired seed noise — the SPEC-18 H65 discriminative-validity test for the computer-use vertical.
See [`cue/leaderboard.py`](../src/verisim/cue/leaderboard.py) and the committed
`figures/cl1_cue_leaderboard.csv`. The ranking is carried by the structure→content gradient: a model
is separated from its neighbors by *content* recall (structure tasks saturate for every tier).

## Intended use
Scoring host (shell/file/process) world-models for computer-use faithfulness, and locating *which*
dynamics a given model still needs the oracle for. **Not** for offensive automation (SPEC.md §13).

## Caveats
The faithful ceiling is exact (deterministic oracle); catch rates and load-bearing flags are
comparable only within a fixed manifest hash and at a stated scale. Computer use here is
shell/file/process, not GUI — the oracle-grounded slice.
"""
