"""Experiment EN9 — oracle hard-negative / counterfactual contrastive (SPEC-8 §7, milestone OG4).

The second SPEC-8 placement of the oracle in the **self-supervised bulk**, and the one that
consumes the OG2 hard-negative factory (:mod:`verisim.netdata.negatives`). A single pre-registered
ablation on the NW8 graph+RSSM arm: a contrastive predictor over the graph summary whose *only*
anti-collapse referent varies across three cells (SPEC-8 §4.3, H25/H5):

  - **none** — naked BYOL: regress the predicted anchor embedding onto the stop-grad online
    embedding of the true successor. No referent → the representation collapses.
  - **vicreg** — add VICReg variance/covariance, the field's *statistical* stand-in for "push the
    representations apart." Prevents collapse, but is blind to *which* state is which.
  - **oracle** — InfoNCE against the OG2 exact hard negatives (counterfactual successors
    ``O(s, a')`` + one-edit-wrong neighbors of the positive). An external, non-degenerate referent.

Two readouts: representation health (embedding std + effective rank, the collapse diagnostic) and
**interventional fidelity** (does the representation map each intervention ``a'`` to its true
successor ``O(s, a')``? — the H5 / EN6 branch-replay question). *Does the exact near-miss referent
match or beat the statistical regularizer at preventing collapse (H25), and do the counterfactual
negatives additionally lift interventional fidelity (H5)?*

A committed **smoke** instance of the apparatus (the EN1/EN4/EN8 honesty caveat), not a tuned
publication run — whatever it shows is a datum, and the honest negative is first-class (SPEC-8 §10).
Regenerates from config + seeds with ``maxΔ = 0``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netmodel import NetVocab
from verisim.netmodel.graph_model import build_graph_model
from verisim.netmodel.grounded_train import build_contrastive_dataset, train_contrastive
from verisim.netoracle import ReferenceNetworkOracle


@dataclass(frozen=True)
class EN9Config:
    """Small, fast ablation instance. Scale up (more seeds/iters/size) for the publication run."""

    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    k_negatives: int = 8
    # graph-arm sizing / training
    d_model: int = 48
    mp_rounds: int = 3
    contrastive_iters: int = 400
    temperature: float = 0.1
    model_seed: int = 0
    modes: tuple[str, ...] = ("none", "vicreg", "oracle")


def run_en9(config: EN9Config | None = None) -> list[dict[str, float | str]]:
    """Run the EN9 contrastive ablation; return one row of metrics per anti-collapse mode."""
    import torch

    config = config or EN9Config()
    torch.set_num_threads(1)  # process-reproducibility (the EN1 discipline)
    oracle = ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)

    examples, branches = build_contrastive_dataset(
        oracle, vocab, net, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj, k_negatives=config.k_negatives,
    )

    rows: list[dict[str, float | str]] = []
    for mode in config.modes:
        model = build_graph_model(
            vocab, net, d_model=config.d_model, mp_rounds=config.mp_rounds, seed=config.model_seed
        )
        result = train_contrastive(
            model, examples, branches, mode=mode,
            steps=config.contrastive_iters, temperature=config.temperature, seed=config.model_seed,
        )
        rows.append(
            {
                "mode": mode,
                "emb_std": result.emb_std,
                "eff_rank": result.eff_rank,
                "intervention_top1": result.intervention_top1,
                "intervention_mrr": result.intervention_mrr,
                "final_loss": result.final_loss,
            }
        )
    return rows


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN9 oracle-contrastive ablation (H25/H5).")
    parser.add_argument("--contrastive-iters", type=int, default=400)
    parser.add_argument("--out", type=str, default="figures/en9_contrastive.csv")
    args = parser.parse_args()
    cfg = EN9Config(contrastive_iters=args.contrastive_iters)
    rows = run_en9(cfg)

    metrics = ("emb_std", "eff_rank", "intervention_top1", "intervention_mrr", "final_loss")
    print("EN9 — oracle hard-negative contrastive (collapse + interventional fidelity):")
    print(f"  {'mode':<8} {'emb_std':>9} {'eff_rank':>9} {'iv_top1':>9} {'iv_mrr':>9} {'loss':>9}")
    lines = ["mode,metric,value"]
    for row in rows:
        mode = str(row["mode"])
        print(
            f"  {mode:<8} {row['emb_std']:>9.4f} {row['eff_rank']:>9.3f}"
            f" {row['intervention_top1']:>9.3f} {row['intervention_mrr']:>9.3f}"
            f" {row['final_loss']:>9.4f}"
        )
        for metric in metrics:
            lines.append(f"{mode},{metric},{row[metric]:.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(rows, out.with_suffix(".png"))


def _plot(rows: list[dict[str, float | str]], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [str(r["mode"]) for r in rows]
    x = range(len(modes))
    colors = {"none": "#c66", "vicreg": "#9bd", "oracle": "#16a"}
    bar_colors = [colors.get(m, "#999") for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(list(x), [float(r["emb_std"]) for r in rows], color=bar_colors)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(modes)
    ax1.set_title("H25: embedding std (↓ = collapse)")
    ax1.set_ylabel("mean per-dim std")

    ax2.bar([i - 0.2 for i in x], [float(r["intervention_top1"]) for r in rows],
            width=0.4, color=bar_colors, label="top-1")
    ax2.bar([i + 0.2 for i in x], [float(r["intervention_mrr"]) for r in rows],
            width=0.4, color=bar_colors, alpha=0.5, label="MRR")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(modes)
    ax2.set_ylim(0, 1)
    ax2.set_title("H5: interventional fidelity (↑ = better)")
    ax2.legend(fontsize=8)

    fig.suptitle("EN9 (smoke): oracle hard-negative contrastive — collapse (H25) + intervene (H5)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":  # pragma: no cover
    main()
