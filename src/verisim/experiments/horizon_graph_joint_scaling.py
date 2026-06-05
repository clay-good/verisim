"""HS3 (incr 4) -- the joint capacity×world-size push for the *structured* arm (SPEC-10 §5).

HS3 increments 1-3 swept the graph arm's three resource axes **one at a time** -- capacity (§4.6),
data (§4.7), world size (§4.8) -- and found `H_free = H_ε(ρ=0)` pinned at the floor on each. But the
flat arm's HS1.3 ([`horizon_joint_scaling`](./horizon_joint_scaling.py), §4.4) taught the cross-axis
lesson the marginals miss: scaling capacity *and* data **together** (a compute-optimal ladder)
lifted the flat arm's peak *above* what either marginal reached at a fixed budget. So the joint
question for the structured arm is genuinely pre-registered, not redundant:

    With BOTH capacity and world size scaled **together** -- a bigger graph arm in a bigger world,
    exactly the regime the GNN+RSSM inductive bias is meant for -- does `H_free` finally leave the
    floor, or is the structured ceiling robust to the joint push too?

A nonzero, rising `H_free` along the ladder would overturn the HS3 marginals (the ceiling was a
hold-one-axis-fixed artifact). `H_free` pinned at 0 along the whole ladder is the strongest form of
the HS3 verdict: the structured compounding ceiling survives even the joint scaling that helped the
flat arm -- a genuine wall, not a resourcing artifact on any axis or their product. Either is
first-class, and the oracle makes it exact.

Reuses the HS3 graph cell verbatim (the net-parametric ``_cell``); each ladder point carries its own
(capacity, n_hosts) pair. Rows are keyed by params so the HS1 two-panel plotter renders it directly.
CPU-local (the SPEC-9 envelope); CI runs only the smoke instance.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import scaled_net_config
from verisim.netmodel import NetVocab
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .horizon_graph_scaling import GraphHorizonScalingConfig, GraphScale, _cell
from .horizon_scaling import METRICS, ScaleStat

__all__ = ["run_graph_joint_scaling", "write_csv"]


def _points(d: dict[str, Any]) -> tuple[tuple[GraphScale, int], ...]:
    """Parse the ladder: each point is a (graph capacity, n_hosts) pair, one cell."""
    out: list[tuple[GraphScale, int]] = []
    for p in d["points"]:
        scale = GraphScale(
            label=p["label"], d_model=p["d_model"], n_layer=p.get("n_layer", 2),
            mp_rounds=p.get("mp_rounds", 2), n_head=p.get("n_head", 2),
            train_steps=p.get("train_steps", 2000),
        )
        out.append((scale, p["n_hosts"]))
    return tuple(out)


def run_graph_joint_scaling(
    config: GraphHorizonScalingConfig,
    points: tuple[tuple[GraphScale, int], ...],
    *,
    n_ports: int = 3,
    oracle: NetOracle | None = None,
) -> list[ScaleStat]:
    """Run the structured joint ladder: each cell trains at its own (capacity, world size).

    Returns :class:`ScaleStat` rows keyed by ``params`` (capacity), so the HS1 plotter renders them;
    the per-cell world size is folded into the ``scale`` label (e.g. ``l@20h``).
    """
    oracle = oracle or ReferenceNetworkOracle()

    stats: list[ScaleStat] = []
    for scale, n_hosts in points:
        net = scaled_net_config(n_hosts, n_ports)
        vocab = NetVocab(net)
        label = f"{scale.label}@{n_hosts}h"
        per_seed = []
        for seed in config.seeds:
            cell = _cell(config, scale, seed, oracle, net, vocab)
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

    parser = argparse.ArgumentParser(
        description="Run HS3 incr 4 (graph capacity×world-size ladder)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures/horizon_graph_joint_scaling.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_graph_joint_scaling.png")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    config = GraphHorizonScalingConfig.from_dict(raw)
    points = _points(raw)
    stats = run_graph_joint_scaling(config, points)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    by_scale: dict[str, dict[str, float]] = {}
    for s in stats:
        by_scale.setdefault(s.scale, {"params": float(s.params)})[s.metric] = s.mean
    for label in sorted(by_scale, key=lambda k: by_scale[k]["params"]):
        d = by_scale[label]
        print(f"  {label:10s} N={int(d['params']):>8d}  [id]  p={d['one_step_acc_id']:.3f} "
              f"H_free={d['h_free_id']:.2f} eta={d['horizon_efficiency_id']:.2f}   "
              f"[ood] p={d['one_step_acc_ood']:.3f} H_free={d['h_free_ood']:.2f} "
              f"eta={d['horizon_efficiency_ood']:.2f}")
    try:
        from figures.plot_horizon_scaling import plot_horizon_scaling

        plot_horizon_scaling(
            stats, args.plot,
            suptitle="HS3 incr 4 — the joint capacity×world-size push, STRUCTURED arm (SPEC-10)",
            left_title="Scale capacity WITH world size: does the structured floor lift?",
        )
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
