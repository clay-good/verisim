"""HS3-T -- is the graph arm's `p` plateau a flat-LR trainer artifact? (SPEC-10 §4.11).

Every HS3 result (§4.6-4.10) carried the same honest caveat: the committed graph trainer plateaus at
one-step accuracy `p` ≈ 0.66, **below** the flat arm's 0.82, so part of the structured arm's pinned
`H_free` floor could be the graph arm not reaching the flat arm's per-step operating point -- a
*trainer*, not architectural, confound. The flat arm reached 0.82 with `train_batched`'s **warmup +
cosine** LR schedule (SPEC-2.1 §6), whose own docstring says "the LR schedule converges where a flat
LR stalls"; the graph trainer used a **flat LR**. So the program's standing "method-fails vs.
too-small" question, applied to the graph arm's *optimizer*, is exactly this:

    Does giving the graph arm the *same* warmup+cosine schedule that lifted the flat arm to 0.82
    raise its `p` -- and if so, does its free-running horizon `H_free` finally leave the floor?

HS3-T answers it at fixed capacity (`m`) by training the graph arm **flat-LR vs scheduled**
(``warmup_frac`` 0 vs 0.1 -- the opt-in lever added to
:func:`~verisim.netmodel.graph_train.train_graph_model`, default-off so every committed result is
byte-identical) and comparing `p` and `H_free`. A schedule that lifts `p` *and* frees the horizon
would overturn HS3 (the ceiling was a flat-LR artifact); a schedule that lifts neither shows the
plateau is the graph arm's **representation on this world, not its optimizer** -- bulletproofing the
HS3 verdict. Either is first-class, and the oracle makes it exact.

Reuses the HS3 graph cell verbatim; the only varied axis is ``warmup_frac``. CPU-local (the SPEC-9
envelope); CI runs only the smoke instance.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netmodel import NetVocab
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .horizon_graph_scaling import GraphHorizonScalingConfig, GraphScale, _cell
from .horizon_scaling import METRICS, ScaleStat

__all__ = ["CSV_HEADER", "run_graph_schedule_diag", "write_csv"]

CSV_HEADER = "warmup_frac,label,metric,mean,ci_lo,ci_hi,n"


def _arm_label(warmup_frac: float) -> str:
    return "flat-LR" if warmup_frac <= 0.0 else f"scheduled@{warmup_frac:g}"


def _arms(d: dict[str, Any]) -> tuple[float, ...]:
    """The trainer arms: each entry is a ``warmup_frac`` (0 = flat LR, >0 = warmup+cosine)."""
    return tuple(d.get("warmup_fracs", (0.0, 0.1)))


def run_graph_schedule_diag(
    config: GraphHorizonScalingConfig,
    warmup_fracs: tuple[float, ...],
    *,
    oracle: NetOracle | None = None,
) -> list[ScaleStat]:
    """Train the fixed-capacity graph arm under each ``warmup_frac``; reduce over seeds to mean+CI.

    Rows carry ``params = int(warmup_frac * 1000)`` (a sort key) and ``scale`` = the arm label
    (``flat-LR`` / ``scheduled@0.1``), so the standard reduced-record plumbing + CIs apply.
    """
    if len(config.scales) != 1:
        raise ValueError("HS3-T fixes capacity: config must specify exactly one scale")
    scale: GraphScale = config.scales[0]
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)

    stats: list[ScaleStat] = []
    for warmup_frac in warmup_fracs:
        cfg = replace(config, warmup_frac=warmup_frac)
        label = _arm_label(warmup_frac)
        per_seed = []
        for seed in config.seeds:
            cell = _cell(cfg, scale, seed, oracle, net, vocab)
            per_seed.append(cell)
            if config.verbose:  # pragma: no cover - progress for the long local sweep
                print(
                    f"  [{label} N={scale.params} model_seed={seed}] "
                    f"p={cell['one_step_acc_id']:.3f} H_free={cell['h_free_id']:.2f} "
                    f"(ood H_free={cell['h_free_ood']:.2f})",
                    flush=True,
                )
        for metric in METRICS:
            values = [c[metric] for c in per_seed]
            lo, hi = bootstrap_ci(values, seed=0)
            mean = fmean(values)
            key = int(warmup_frac * 1000)
            stats.append(ScaleStat(label, key, metric, mean, lo, hi, len(values)))
    return stats


def write_csv(stats: list[ScaleStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{s.params / 1000:g},{s.scale},{s.metric},{s.mean:.6f},{s.ci_lo:.6f},{s.ci_hi:.6f},{s.n}"
        for s in stats
    ]
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HS3-T (graph flat-LR vs warmup+cosine trainer diagnostic)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures/horizon_graph_schedule.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_graph_schedule.png")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    config = GraphHorizonScalingConfig.from_dict(raw)
    warmup_fracs = _arms(raw)
    stats = run_graph_schedule_diag(config, warmup_fracs)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    by_label: dict[str, dict[str, float]] = {}
    for s in stats:
        by_label.setdefault(s.scale, {})[s.metric] = s.mean
    for label, d in by_label.items():
        print(f"  {label:14s} [id]  p={d['one_step_acc_id']:.3f} H_free={d['h_free_id']:.2f} "
              f"eta={d['horizon_efficiency_id']:.2f}   [ood] p={d['one_step_acc_ood']:.3f} "
              f"H_free={d['h_free_ood']:.2f}")
    try:
        _plot(stats, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


def _plot(stats: list[ScaleStat], path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(s.scale for s in stats))
    x = range(len(labels))

    def vals(metric: str) -> list[float]:
        by = {s.scale: s.mean for s in stats if s.metric == metric}
        return [by.get(lbl, 0.0) for lbl in labels]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.4))
    w = 0.35
    ax_l.bar([i - w / 2 for i in x], vals("one_step_acc_id"), w, label="p  [id]", color="#1f77b4")
    ax_l.bar([i + w / 2 for i in x], vals("one_step_acc_ood"), w, label="p  [ood]", color="#d62728")
    ax_l.axhline(0.82, ls="--", color="#555", lw=1, label="flat arm's p (HS1, ≈0.82)")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels(labels)
    ax_l.set_ylim(0, 1.0)
    ax_l.set_ylabel("one-step acceptance p")
    ax_l.set_title("Does the schedule lift the graph arm's p?")
    ax_l.legend(fontsize=8)

    ax_r.bar([i - w / 2 for i in x], vals("h_free_id"), w, label="H_free  [id]", color="#1f77b4")
    ax_r.bar([i + w / 2 for i in x], vals("h_free_ood"), w, label="H_free  [ood]", color="#d62728")
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(labels)
    ax_r.set_ylabel("free-running faithful horizon  H_ε(ρ=0)")
    ax_r.set_title("…and does the horizon leave the floor?")
    ax_r.legend(fontsize=8)

    fig.suptitle("HS3-T — the graph p plateau is the representation, not the flat LR (SPEC-10)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
