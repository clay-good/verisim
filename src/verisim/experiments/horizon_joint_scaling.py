"""HS1.3 -- the joint capacity×data push (SPEC-10 §5; the compute-optimal recipe HS1.1+HS1.2 imply).

HS1.1 ([`horizon_scaling`](./horizon_scaling.py), SPEC-10 §4.2) found `H_free` is *non-monotone* in
capacity at fixed data -- it peaks at `l` then declines. HS1.2
([`horizon_data_scaling`](./horizon_data_scaling.py), §4.3) showed that decline is **data
starvation**, not a capacity wall: at fixed `xl`, more data recovers the horizon. Put together they
imply a prescription -- the Chinchilla one -- **scale data *with* capacity.** HS1.3 tests it
directly: a compute-optimal ladder where each larger model is fed a correspondingly larger coverage
set (m@4.8k -> l@9.6k -> xl@16k -> xxl@24k transitions), each cell adequately trained. The
pre-registered question:

    With BOTH levers scaled together, does `H_free` keep climbing -- exceeding the `l` peak HS1.1
    hit at a fixed data budget -- or does a real capacity wall finally bind even with matched data?

A monotone-rising `H_free` along this ladder is the clean *positive* faithfulness scaling law: not a
hump (HS1.1's fixed-data artifact) but an open climb when not starved. A plateau/decline
*despite* matched data would be the first evidence of a real wall at this world's scale. Either is
first-class, and as throughout SPEC-10 the oracle makes the verdict exact -- we measure long-horizon
fidelity against ground truth, not a proxy loss.

Reuses the HS1 harness internals verbatim (``_cell``, ``_train_scaled``, ``_evaluate``, the EN1
dataset builder); the only new structure is that each cell carries its *own* (capacity, data) pair
rather than sharing one coverage set down a pure-capacity axis. Rows are keyed by params, so the HS1
two-panel plotter renders it directly. CPU-local; CI runs only the smoke instance.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.netmodel import NetVocab, build_net_dataset
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .horizon_scaling import METRICS, HorizonScalingConfig, ModelScale, ScaleStat, _cell


def _points(d: dict[str, Any]) -> tuple[tuple[ModelScale, int], ...]:
    """Parse the ladder: each point is a (capacity, data-seed-count) pair, one cell."""
    out: list[tuple[ModelScale, int]] = []
    for p in d["points"]:
        scale = ModelScale(
            label=p["label"], n_embd=p["n_embd"], n_layer=p["n_layer"],
            n_head=p.get("n_head", 2), train_steps=p.get("train_steps", 4000),
        )
        out.append((scale, p["data_seeds"]))
    return tuple(out)


def run_joint_scaling(
    config: HorizonScalingConfig,
    points: tuple[tuple[ModelScale, int], ...],
    *,
    oracle: NetOracle | None = None,
) -> list[ScaleStat]:
    """Run the compute-optimal ladder: each cell trains at its own capacity on its own coverage set.

    Returns :class:`ScaleStat` rows keyed by ``params`` (capacity), so the HS1 plotter renders them;
    the per-cell data budget is folded into the ``scale`` label (e.g. ``xl@16000``).
    """
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)

    stats: list[ScaleStat] = []
    for scale, n_seeds in points:
        base = replace(config.base, train_seeds=tuple(range(n_seeds)))
        examples = build_net_dataset(
            oracle, vocab, net, driver=base.train_driver, seeds=base.train_seeds,
            n_steps=base.train_steps_per_traj,
        )
        n_train = len(examples)
        cfg = replace(config, base=base)
        label = f"{scale.label}@{n_train}"
        per_seed = []
        for seed in config.seeds:
            cell = _cell(cfg, scale, seed, oracle, net, vocab, examples)
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
            stats.append(ScaleStat(label, scale.params, metric, fmean(values), lo, hi, len(values)))
    return stats


def write_csv(stats: list[ScaleStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{s.scale},{s.params},{s.metric},{s.mean:.6f},{s.ci_lo:.6f},{s.ci_hi:.6f},{s.n}"
        for s in stats
    ]
    out.write_text("\n".join(["scale,params,metric,mean,ci_lo,ci_hi,n", *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run HS1.3 (joint capacity×data scaling).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures/horizon_joint_scaling.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_joint_scaling.png")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    config = HorizonScalingConfig.from_dict(raw)
    points = _points(raw)
    stats = run_joint_scaling(config, points)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    by_scale: dict[str, dict[str, float]] = {}
    for s in stats:
        by_scale.setdefault(s.scale, {"params": float(s.params)})[s.metric] = s.mean
    for label in sorted(by_scale, key=lambda k: by_scale[k]["params"]):
        d = by_scale[label]
        print(f"  {label:14s} N={int(d['params']):>8d}  [id]  p={d['one_step_acc_id']:.3f} "
              f"H_free={d['h_free_id']:.2f} eta={d['horizon_efficiency_id']:.2f}   "
              f"[ood] p={d['one_step_acc_ood']:.3f} H_free={d['h_free_ood']:.2f} "
              f"eta={d['horizon_efficiency_ood']:.2f}")
    try:
        from figures.plot_horizon_scaling import plot_horizon_scaling

        plot_horizon_scaling(
            stats, args.plot,
            suptitle="HS1.3 — the joint capacity×data scaling law (SPEC-10)",
            left_title="Scale data WITH capacity: does the horizon keep climbing?",
        )
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
