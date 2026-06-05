"""HS3 (incr 3) -- the world-size cross-axis for the *structured* (graph) arm (SPEC-10 §5).

HS3 increments 1-2 found the GNN+RSSM graph arm's free-running horizon `H_free = H_ε(ρ=0)` is a
**genuine ceiling** on the 5-host network world: it stays at the floor (≈ 0, η < 1) and moves with
*neither* model capacity (§4.6) *nor* data (§4.7). Both were measured at one **world size**. The
last open question is whether that ceiling is a property of *this small world* or survives the
**world-size axis** -- SPEC-9's `O(N²)` host-count scale ([`scaled_net_config`](../net/config.py),
SPEC-8 §7.1). The graph arm's whole reason to exist is its **inductive bias over network structure**
(the message-passing GNN + the RSSM belief, SPEC-5 §6.1-6.2), which has *more* structure to exploit
as the world grows -- so a larger world is exactly where the structured arm could, in principle,
pull ahead of its floor.

This module answers it the HS1.2/HS3-incr-2 way: **hold the graph capacity fixed** (at `m`) and
**sweep the world size** `n_hosts`. If `H_free` rises with world size, the structured ceiling was a
small-world artifact and the graph arm's bias pays off at scale; if it stays pinned at the floor
(and `η` stays below 1), the ceiling is world-size-invariant -- the genuine compounding wall
confirmed across the one axis HS3 had not yet swept.

Reuses the HS3 graph cell verbatim (the net-parametric ``_cell`` already takes the world ``net`` as
an argument); the only new axis is ``scaled_net_config(n_hosts)`` per cell. The oracle's labels are
free at every world size (SPEC-9), so this is a pure learner-compute choice. CPU-local (the SPEC-9
envelope); CI runs only the smoke instance.
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

__all__ = ["CSV_HEADER", "run_graph_world_scaling", "write_csv"]

# CSV keyed by ``n_hosts`` (the world-size x-axis) at fixed graph capacity; the per-cell metrics are
# otherwise identical to HS3's, so the same reduced-record plumbing and bootstrap CIs apply.
CSV_HEADER = "n_hosts,label,metric,mean,ci_lo,ci_hi,n"


def _world_points(d: dict[str, Any]) -> tuple[int, ...]:
    """The world-size axis: each entry is a host count (the `O(N²)` reachability-state scale)."""
    return tuple(d.get("world_sizes", (5, 10, 20, 40)))


def run_graph_world_scaling(
    config: GraphHorizonScalingConfig,
    world_sizes: tuple[int, ...],
    *,
    n_ports: int = 3,
    oracle: NetOracle | None = None,
) -> list[ScaleStat]:
    """Sweep the world size at the single fixed graph capacity ``config.scales[0]``.

    Returns :class:`ScaleStat` rows whose ``params`` field carries the **host count** (the
    world-size x-axis) and whose ``scale`` label is the same -- so the same reduced-record plumbing
    and bootstrap CIs as HS3 apply, keyed by world size, for the graph proposer at fixed capacity.
    """
    if len(config.scales) != 1:
        raise ValueError("HS3 incr 3 fixes capacity: config must specify exactly one scale")
    scale: GraphScale = config.scales[0]
    oracle = oracle or ReferenceNetworkOracle()

    stats: list[ScaleStat] = []
    for n_hosts in world_sizes:
        net = scaled_net_config(n_hosts, n_ports)
        vocab = NetVocab(net)
        per_seed = []
        for seed in config.seeds:
            cell = _cell(config, scale, seed, oracle, net, vocab)
            per_seed.append(cell)
            if config.verbose:  # pragma: no cover - progress for the long local sweep
                print(
                    f"  [n_hosts={n_hosts} model_seed={seed}] "
                    f"p={cell['one_step_acc_id']:.3f} H_free={cell['h_free_id']:.2f} "
                    f"(ood H_free={cell['h_free_ood']:.2f})",
                    flush=True,
                )
        for metric in METRICS:
            values = [c[metric] for c in per_seed]
            lo, hi = bootstrap_ci(values, seed=0)
            mean = fmean(values)
            stats.append(ScaleStat(str(n_hosts), n_hosts, metric, mean, lo, hi, len(values)))
    return stats


def write_csv(stats: list[ScaleStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{s.params},{s.scale},{s.metric},{s.mean:.6f},{s.ci_lo:.6f},{s.ci_hi:.6f},{s.n}"
        for s in stats
    ]
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run HS3 incr 3 (graph world-size cross-axis).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures/horizon_graph_world_scaling.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_graph_world_scaling.png")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    config = GraphHorizonScalingConfig.from_dict(raw)
    world_sizes = _world_points(raw)
    stats = run_graph_world_scaling(config, world_sizes)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    by_n: dict[int, dict[str, float]] = {}
    for s in stats:
        by_n.setdefault(s.params, {})[s.metric] = s.mean
    cap = config.scales[0]
    print(f"  (fixed graph capacity {cap.label}, params={cap.params})")
    for n in sorted(by_n):
        d = by_n[n]
        print(f"  n_hosts={n:>4d}  [id]  p={d['one_step_acc_id']:.3f} "
              f"H_free={d['h_free_id']:.2f} eta={d['horizon_efficiency_id']:.2f}   "
              f"[ood] p={d['one_step_acc_ood']:.3f} H_free={d['h_free_ood']:.2f} "
              f"eta={d['horizon_efficiency_ood']:.2f}")
    try:
        from figures.plot_horizon_data_scaling import plot_horizon_data_scaling

        plot_horizon_data_scaling(
            stats, cap, args.plot,  # type: ignore[arg-type]  # GraphScale has .label/.params
            suptitle="HS3 incr 3 — world-size cross-axis, the STRUCTURED graph arm (SPEC-10)",
            left_title="Does a bigger world lift the structured floor?",
            xlabel="world size (hosts)",
            right_title="Per-step accuracy vs world size",
        )
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
