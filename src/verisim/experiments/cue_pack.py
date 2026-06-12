"""Experiment cue-pack: package verisim-cue (SPEC-21 §8, deliverable #2 -- the adopted artifact).

The engineering hardening that turns the SPEC-21 CPU core into a *versioned, conformant, documented*
computer-use benchmark -- the artifact half of the spec (the scale law is the result half). Three
deliverables, all CPU-only and dependency-free:

  - **the metadata**: emit the **Croissant** descriptor, the **datasheet**, & the **task-card** for
    the frozen battery (the standardized, machine-readable provenance/composition/limits BetterBench
    found almost no benchmark ships, SPEC-18 §7) -- under ``cue/`` (a committed artifact directory),
    regenerable from the manifest hash.
  - **the load-bearing verdicts**: the thing that distinguishes verisim-cue -- per task, does the
    faithful predictor beat the free one (is the oracle load-bearing for control)? Read from a
    committed CS1 frontier CSV at the top CPU rung and folded into the task-card.
  - **the conformance suite**: assert the benchmark's contract -- ground-truth labels (the faithful
    predictor is exact) and a well-ordered structure->content spectrum.

CPU-only, deterministic, seeded. Adoption is *not* a hypothesis (SPEC-18 §9); the artifact ships
whether or not it is adopted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from verisim.cue.conformance import all_passed, run_conformance
from verisim.cue.pack import (
    CueManifest,
    croissant_metadata,
    datasheet,
    task_card,
)
from verisim.cue.scorecard import model_card


def load_verdicts(frontier_csv: str | Path) -> dict[str, dict[str, float]]:
    """Read per-task load-bearing verdicts (gap at the top rung) from a committed CS1 frontier CSV.

    Picks each task's row at the largest ``params`` (the top CPU rung) -- the most-trained verdict
    the committed run provides. Returns ``{}`` if the CSV is absent (the card then marks pending).
    """
    path = Path(frontier_csv)
    if not path.exists():
        return {}
    best: dict[str, tuple[float, float]] = {}  # task -> (params, gap)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            task, params, gap = row["task"], float(row["params"]), float(row["gap"])
            if task not in best or params > best[task][0]:
                best[task] = (params, gap)
    return {task: {"gap": gap, "scale": params} for task, (params, gap) in best.items()}


def emit(
    manifest: CueManifest, out_dir: str | Path, *,
    frontier_csv: str | Path = "figures/cs1_loadbearing_frontier.csv",
) -> dict[str, Path]:
    """Write the Croissant / datasheet / task-card files; return the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    verdicts = load_verdicts(frontier_csv)
    paths = {
        "croissant": out / "croissant.json",
        "datasheet": out / "datasheet.md",
        "task_card": out / "task-card.md",
        "model_card": out / "model-card.md",
    }
    paths["croissant"].write_text(
        json.dumps(croissant_metadata(manifest), indent=2, sort_keys=True) + "\n"
    )
    paths["datasheet"].write_text(datasheet(manifest))
    paths["task_card"].write_text(task_card(manifest, verdicts))
    paths["model_card"].write_text(model_card(manifest, frontier_csv=frontier_csv))
    return paths


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="cue-pack -- package the verisim-cue computer-use benchmark (SPEC-21 §8)."
    )
    parser.add_argument("--out", type=str, default="cue", help="the committed artifact directory")
    parser.add_argument("--frontier", type=str, default="figures/cs1_loadbearing_frontier.csv")
    args = parser.parse_args()

    manifest = CueManifest()
    paths = emit(manifest, args.out, frontier_csv=args.frontier)
    print(f"verisim-cue {manifest.version_tag()}")
    for name, path in paths.items():
        print(f"  emitted {name}: {path}")

    results = run_conformance(manifest)
    print("conformance:")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.check}: {r.detail}")
    print(f"conformance: {'ALL PASS' if all_passed(results) else 'FAILURES'}")


if __name__ == "__main__":  # pragma: no cover
    main()
