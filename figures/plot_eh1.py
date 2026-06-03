"""Plot the EH1 composed-host ``H_ε(ρ)`` curve + the composition law (SPEC-6 §0, §9.2, HC6).

**SPEC-6's headline figure** (§0). Figures are produced *only* from run-records: this reads the host
JSONL written by ``verisim.experiments.eh1`` (each record carrying the composed *and* per-subsystem
divergence trajectories), and renders two figures from them:

  - ``eh1_curve.png`` -- the composed faithful horizon ``H_ε`` vs consultation budget ``ρ`` (the
    prime-directive curve), reusing v0's aggregator + renderer verbatim (the record's composed view
    is identical to v0's).
  - ``eh1_composition.png`` -- the composition law (H13): per difficulty, the composed per-step
    acceptance against the multiplicative (``∏ a_i``) and weakest-link (``min a_i``) predictions,
    with the verdict. The honest object the prime directive is about (§9.2, §10.2).

Usage:
    python figures/plot_eh1.py --records runs/eh1/host_records.jsonl \\
        --out figures/eh1_curve.png --csv figures/eh1_curve.csv \\
        --comp-out figures/eh1_composition.png --comp-csv figures/eh1_composition.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Reuse v0's CSV writer + renderer verbatim -- the record schema's composed view is identical.
from plot_e1 import plot_curve, write_csv

from verisim.hostmetrics.composition import CompositionLaw
from verisim.hostmetrics.record import HostRunRecord, read_host_records
from verisim.metrics.aggregate import aggregate_curve
from verisim.metrics.record import RunRecord


def _read_composition(path: str | Path) -> dict[str, CompositionLaw]:
    """Load the composition-law summary (H13) the experiment wrote (per difficulty)."""
    data = json.loads(Path(path).read_text())
    return {
        difficulty: CompositionLaw(
            subsystem_acceptance=d["subsystem_acceptance"],
            composed_acceptance=d["composed_acceptance"],
            multiplicative_prediction=d["multiplicative_prediction"],
            weakest_link_prediction=d["weakest_link_prediction"],
            multiplicative_residual=d["multiplicative_residual"],
            weakest_link_residual=d["weakest_link_residual"],
            verdict=d["verdict"],
        )
        for difficulty, d in data.items()
    }


def _to_run_records(host_records: list[HostRunRecord]) -> list[RunRecord]:
    """Project each host record onto its **composed** view -- a v0 RunRecord for the curve."""
    return [
        RunRecord(
            config=dict(r.config),
            seed=r.seed,
            epsilon=r.epsilon,
            divergences=list(r.divergences),
            consultation_schedule=list(r.consultation_schedule),
        )
        for r in host_records
    ]


def write_composition_csv(law: dict[str, CompositionLaw], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    subsystems = sorted({s for result in law.values() for s in result.subsystem_acceptance})
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["difficulty", "composed", "multiplicative", "weakest_link",
             "r_mult", "r_weak", "verdict", *[f"a_{s}" for s in subsystems]]
        )
        for difficulty, result in sorted(law.items()):
            writer.writerow([
                difficulty, result.composed_acceptance, result.multiplicative_prediction,
                result.weakest_link_prediction, result.multiplicative_residual,
                result.weakest_link_residual, result.verdict,
                *[result.subsystem_acceptance.get(s, "") for s in subsystems],
            ])
    return out


def plot_composition(law: dict[str, CompositionLaw], out_path: str | Path) -> Path:
    """Per difficulty: composed acceptance vs the multiplicative / weakest-link laws (H13)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difficulties = sorted(law)
    x = range(len(difficulties))
    composed = [law[d].composed_acceptance for d in difficulties]
    mult = [law[d].multiplicative_prediction for d in difficulties]
    weak = [law[d].weakest_link_prediction for d in difficulties]

    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.26
    ax.bar([i - w for i in x], mult, w, label="multiplicative  ∏ aᵢ", color="#4c72b0")
    ax.bar(list(x), composed, w, label="composed  a (measured)", color="#dd8452")
    ax.bar([i + w for i in x], weak, w, label="weakest-link  min aᵢ", color="#55a868")
    for i, d in enumerate(difficulties):
        ax.annotate(
            law[d].verdict, (i, max(composed[i], weak[i]) + 0.02),
            ha="center", fontsize="small", fontweight="bold",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(difficulties)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("difficulty")
    ax.set_ylabel("per-step acceptance  a")
    ax.set_title("Verisim EH1 — composition law (H13): is composed faithfulness ∏ aᵢ or min aᵢ?")
    ax.legend(fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the EH1 composed-host H_eps(rho) + H13.")
    parser.add_argument("--records", type=str, default="runs/eh1/host_records.jsonl")
    parser.add_argument("--composition", type=str, default="runs/eh1/composition.json")
    parser.add_argument("--out", type=str, default="figures/eh1_curve.png")
    parser.add_argument("--csv", type=str, default="figures/eh1_curve.csv")
    parser.add_argument("--comp-out", type=str, default="figures/eh1_composition.png")
    parser.add_argument("--comp-csv", type=str, default="figures/eh1_composition.csv")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    host_records = read_host_records(args.records)
    points = aggregate_curve(_to_run_records(host_records), n_resamples=args.resamples, seed=0)
    csv_path = write_csv(points, args.csv)
    fig_path = plot_curve(points, args.out, title="Verisim EH1 — composed-host H_ε(ρ)")

    law = _read_composition(args.composition)
    comp_csv = write_composition_csv(law, args.comp_csv)
    comp_fig = plot_composition(law, args.comp_out)

    print(f"{len(host_records)} records -> {len(points)} curve points")
    print(f"wrote {csv_path} and {fig_path}")
    print(f"wrote {comp_csv} and {comp_fig}")
    for difficulty, result in sorted(law.items()):
        print(f"  composition law [{difficulty}]: {result.verdict}")


if __name__ == "__main__":
    main()
