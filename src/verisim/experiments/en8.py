"""Experiment EN8 — oracle-grounded SSL ablation (SPEC-8 §7, milestone OG3; SPEC-5 §12).

The first place the oracle moves from the *cherry* (EN5's RLVR) into the **self-supervised bulk**:
two pre-registered ablations on the NW8 graph+RSSM latent arm, each reusing the OG1/OG2 data factory
and the same seeded oracle data EN1/EN4 use, so the only thing that varies is the *training signal*.

  - **H24 — objective axis.** Train the graph decoder under partial observation with the raw
    next-state-likelihood objective (cross-entropy over every delta token, the supervised baseline)
    versus the **residual / bits-to-correct objective** (gradient only on the genuinely-uncertain
    bits ``R``; the oracle-decidable bits ``D`` are masked — *verify, don't learn*, SPEC-8 §4.2).
    Reported on the residual tokens, where the question actually lives.
  - **H23 — collapse axis.** JEPA latent-predictive pretraining with the target either **learned**
    (BYOL/JEPA's EMA encoder) or **oracle-anchored** (a fixed projection of the true next state — an
    external referent, SPEC-8 §4.1), crossed with the collapse-prevention machinery (EMA + VICReg)
    **on/off**. The readout is representation health (embedding std + effective rank), the standard
    collapse diagnostic: *does the oracle target keep the representation healthy with the machinery
    ablated?*

A committed **smoke** instance of the apparatus (the EN1/EN4 honesty caveat), not a tuned
publication run — whatever it shows is a datum, and the honest negative is first-class (SPEC-8 §10).
Regenerates from config + seeds with ``maxΔ = 0``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netmodel import NetVocab
from verisim.netmodel.graph_model import build_graph_model
from verisim.netmodel.grounded_train import (
    build_grounded_dataset,
    residual_token_accuracy,
    train_grounded_decoder,
    train_jepa,
)
from verisim.netoracle import ReferenceNetworkOracle


@dataclass(frozen=True)
class EN8Config:
    """Small, fast ablation instance. Scale up (more seeds/iters/size) for the publication run."""

    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    observed_fraction: float = 0.5  # partial obs makes the residual non-degenerate (§3)
    # graph-arm sizing / training
    d_model: int = 48
    mp_rounds: int = 3
    decoder_iters: int = 600
    jepa_iters: int = 400
    model_seed: int = 0
    objectives: tuple[str, ...] = ("likelihood", "residual")
    # the H23 collapse cells: (target, collapse_machinery)
    collapse_cells: tuple[tuple[str, bool], ...] = field(
        default_factory=lambda: (
            ("learned", True),  # the EMA + VICReg JEPA baseline
            ("learned", False),  # naked predictive loss → collapses (machinery is load-bearing)
            ("oracle", True),  # oracle target + machinery (belt and suspenders)
            ("oracle", False),  # the H23 cell: oracle target, machinery ablated
        )
    )


def run_en8(config: EN8Config | None = None) -> dict[str, list[dict[str, float | str]]]:
    """Run both EN8 ablations; return ``{'objective': [...rows], 'collapse': [...rows]}``."""
    import torch

    config = config or EN8Config()
    torch.set_num_threads(1)  # process-reproducibility (SPEC-2 §12; the EN1 discipline)
    oracle = ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)

    train = build_grounded_dataset(
        oracle, vocab, net, seeds=config.train_seeds, n_steps=config.train_steps_per_traj,
        observed_fraction=config.observed_fraction,
    )
    held = build_grounded_dataset(
        oracle, vocab, net, seeds=config.eval_seeds, n_steps=config.eval_steps,
        observed_fraction=config.observed_fraction,
    )

    # --- H24: objective axis ----------------------------------------------------
    # The fair metric is accuracy on the *residual* tokens R — the bits the model is responsible
    # for. The decidable bits D are offloaded to the oracle by construction (verify, don't learn,
    # §4.2), so a full-delta free-decode is a category error for the residual arm, not a result.
    objective_rows: list[dict[str, float | str]] = []
    for objective in config.objectives:
        model = build_graph_model(
            vocab, net, d_model=config.d_model, mp_rounds=config.mp_rounds, seed=config.model_seed
        )
        train_grounded_decoder(
            model, train, objective=objective, steps=config.decoder_iters, seed=config.model_seed
        )
        overall, residual = residual_token_accuracy(model, held)
        objective_rows.append(
            {"objective": objective, "overall_acc": overall, "residual_acc": residual}
        )

    # --- H23: collapse axis -----------------------------------------------------
    collapse_rows: list[dict[str, float | str]] = []
    for target, machinery in config.collapse_cells:
        model = build_graph_model(
            vocab, net, d_model=config.d_model, mp_rounds=config.mp_rounds, seed=config.model_seed
        )
        result = train_jepa(
            model, train, target=target, collapse_machinery=machinery,
            steps=config.jepa_iters, seed=config.model_seed,
        )
        collapse_rows.append(
            {
                "target": target,
                "machinery": "on" if machinery else "off",
                "emb_std": result.emb_std,
                "eff_rank": result.eff_rank,
                "jepa_loss": result.final_loss,
            }
        )

    return {"objective": objective_rows, "collapse": collapse_rows}


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN8 oracle-grounded SSL ablation (H23/H24).")
    parser.add_argument("--decoder-iters", type=int, default=600)
    parser.add_argument("--jepa-iters", type=int, default=400)
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=[100, 101, 102])
    parser.add_argument("--out", type=str, default="figures/en8_grounding.csv")
    args = parser.parse_args()
    cfg = EN8Config(
        eval_seeds=tuple(args.eval_seeds),
        decoder_iters=args.decoder_iters,
        jepa_iters=args.jepa_iters,
    )
    results = run_en8(cfg)

    lines = ["section,key,subkey,metric,value"]
    print("H24 — objective axis (partial obs; accuracy on the residual bits R):")
    print(f"  {'objective':<12} {'overall_acc':>12} {'residual_acc':>13}")
    for row in results["objective"]:
        obj = str(row["objective"])
        print(f"  {obj:<12} {row['overall_acc']:>12.4f} {row['residual_acc']:>13.4f}")
        for metric in ("overall_acc", "residual_acc"):
            lines.append(f"objective,{obj},,{metric},{row[metric]:.6f}")

    print("H23 — collapse axis (representation health):")
    print(f"  {'target':<10} {'machinery':>10} {'emb_std':>10} {'eff_rank':>10} {'jepa_loss':>10}")
    for row in results["collapse"]:
        tgt, mach = str(row["target"]), str(row["machinery"])
        print(
            f"  {tgt:<10} {mach:>10} {row['emb_std']:>10.4f} {row['eff_rank']:>10.3f}"
            f" {row['jepa_loss']:>10.4f}"
        )
        for metric in ("emb_std", "eff_rank", "jepa_loss"):
            lines.append(f"collapse,{tgt},{mach},{metric},{row[metric]:.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(results, out.with_suffix(".png"))


def _plot(
    results: dict[str, list[dict[str, float | str]]], path: Path
) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    objs = [str(r["objective"]) for r in results["objective"]]
    x = range(len(objs))
    ax1.bar([i - 0.2 for i in x], [float(r["overall_acc"]) for r in results["objective"]],
            width=0.4, color="#9bd", label="all-token acc")
    ax1.bar([i + 0.2 for i in x], [float(r["residual_acc"]) for r in results["objective"]],
            width=0.4, color="#16a", label="residual-token acc (R)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(objs)
    ax1.set_ylim(0, 1)
    ax1.set_title("H24: residual vs raw-likelihood objective")
    ax1.legend(fontsize=8)

    cells = [f"{r['target']}\n{r['machinery']}" for r in results["collapse"]]
    cx = range(len(cells))
    ranks = [float(r["eff_rank"]) for r in results["collapse"]]
    ax2.bar(list(cx), ranks, color=["#16a" if "off" in c else "#9bd" for c in cells])
    ax2.set_xticks(list(cx))
    ax2.set_xticklabels(cells, fontsize=8)
    ax2.set_title("H23: representation effective rank (↓ = collapse)")
    ax2.set_ylabel("effective rank")

    fig.suptitle("EN8 (smoke): oracle-grounded SSL — objective (H24) + collapse (H23)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":  # pragma: no cover
    main()
