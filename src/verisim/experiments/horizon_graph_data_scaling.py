"""HS3 (incr 2) -- the data cross-axis for the *structured* (graph) arm (SPEC-10 §5).

HS3 increment 1 ([`horizon_graph_scaling`](./horizon_graph_scaling.py), SPEC-10 §4.6) found the
GNN+RSSM graph arm's free-running horizon `H_free = H_ε(ρ=0)` is **≈ 0 at every capacity** with
per-step `p` **flat** in capacity (~0.66) -- the floor+cliff in its purest form, so HS1's flat-arm
capacity-lift does *not* reproduce for the structured arm. But that left the program's own
"method-fails vs. too-small" question open on the *other* axis: the graph arm's **flat `p`** (it
ceilings already at `xs`) is the signature of a **data**-limited, not capacity-limited, model -- so
the honest next test, exactly as HS1.2 was for the flat arm, is the **data cross-axis**:

  - **data starvation** -- the graph arm's inductive bias is data-efficient but early-saturating, so
    *more data* lifts `p` past the self-stabilization threshold and `H_free` leaves the floor (the
    Chinchilla reading: the graph arm's lever is data, not parameters); or
  - **an architectural / tolerance ceiling** -- the graph arm makes near-but-not-exact predictions
    whose reachability divergence exceeds ε within one step regardless of data, so `H_free` stays at
    the floor however much you feed it.

This module answers it the HS1.2 way: **hold the graph capacity fixed** (at `m`, the mid cell) and
**sweep the shared coverage-set size**. If `H_free` rises with data, the HS3-incr-1 floor was
starvation; if it stays pinned, the structured arm has a genuine ceiling at this world's tolerance.
Either verdict is first-class.

Reuses the HS3 graph harness internals verbatim (the graph ``_cell``, the ``GraphScale``/config);
the only new axis is the number of trajectory seeds in the shared coverage set. CPU-local (the
SPEC-9 envelope); CI runs only the smoke instance.
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
from verisim.netmodel.graph_train import build_graph_dataset
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

# The HS1.2 CSV schema + writer (keyed by ``n_train``) is reused verbatim -- the data axis is the
# same object, only the proposer (the graph ``_cell``) differs.
from .horizon_data_scaling import CSV_HEADER, write_csv
from .horizon_graph_scaling import GraphHorizonScalingConfig, GraphScale, _cell
from .horizon_scaling import METRICS, ScaleStat

__all__ = ["CSV_HEADER", "run_graph_data_scaling", "write_csv"]


def _data_points(d: dict[str, Any]) -> tuple[int, ...]:
    """The data axis: each entry is a trajectory-seed count (× steps_per_traj = transitions)."""
    return tuple(d.get("data_seeds", (16, 40, 80, 160)))


def run_graph_data_scaling(
    config: GraphHorizonScalingConfig,
    data_seeds: tuple[int, ...],
    *,
    oracle: NetOracle | None = None,
) -> list[ScaleStat]:
    """Sweep the coverage-set size at the single fixed graph capacity ``config.scales[0]``.

    Returns :class:`ScaleStat` rows whose ``params`` field carries the **number of transitions** in
    the coverage set (the data x-axis) and whose ``scale`` label is the seed count -- so the same
    reduced-record plumbing and bootstrap CIs as HS1.2 apply, keyed by data, for the graph proposer.
    """
    if len(config.scales) != 1:
        raise ValueError("HS3 incr 2 fixes capacity: config must specify exactly one scale")
    scale: GraphScale = config.scales[0]
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)

    stats: list[ScaleStat] = []
    for n_seeds in data_seeds:
        base = replace(config.base, train_seeds=tuple(range(n_seeds)))
        n_train = len(
            build_graph_dataset(
                oracle, vocab, net, driver=base.train_driver, seeds=base.train_seeds,
                n_steps=base.train_steps_per_traj, noise_prob=config.noise_prob,
            )
        )
        cfg = replace(config, base=base)
        per_seed = []
        for seed in config.seeds:
            cell = _cell(cfg, scale, seed, oracle, net, vocab)
            per_seed.append(cell)
            if config.verbose:  # pragma: no cover - progress for the long local sweep
                print(
                    f"  [n_train={n_train} ({n_seeds} seeds) model_seed={seed}] "
                    f"p={cell['one_step_acc_id']:.3f} H_free={cell['h_free_id']:.2f} "
                    f"(ood H_free={cell['h_free_ood']:.2f})",
                    flush=True,
                )
        for metric in METRICS:
            values = [c[metric] for c in per_seed]
            lo, hi = bootstrap_ci(values, seed=0)
            mean = fmean(values)
            stats.append(ScaleStat(str(n_seeds), n_train, metric, mean, lo, hi, len(values)))
    return stats


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run HS3 incr 2 (graph data cross-axis).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures/horizon_graph_data_scaling.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_graph_data_scaling.png")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    config = GraphHorizonScalingConfig.from_dict(raw)
    data_seeds = _data_points(raw)
    stats = run_graph_data_scaling(config, data_seeds)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    by_n: dict[int, dict[str, float]] = {}
    for s in stats:
        by_n.setdefault(s.params, {})[s.metric] = s.mean
    cap = config.scales[0]
    print(f"  (fixed graph capacity {cap.label}, params={cap.params})")
    for n in sorted(by_n):
        d = by_n[n]
        print(f"  n_train={n:>6d}  [id]  p={d['one_step_acc_id']:.3f} "
              f"H_free={d['h_free_id']:.2f} eta={d['horizon_efficiency_id']:.2f}   "
              f"[ood] p={d['one_step_acc_ood']:.3f} H_free={d['h_free_ood']:.2f} "
              f"eta={d['horizon_efficiency_ood']:.2f}")
    try:
        from figures.plot_horizon_data_scaling import plot_horizon_data_scaling

        plot_horizon_data_scaling(
            stats, cap, args.plot,  # type: ignore[arg-type]  # GraphScale has .label/.params
            suptitle="HS3 incr 2 — the data cross-axis for the STRUCTURED graph arm (SPEC-10)",
        )
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
