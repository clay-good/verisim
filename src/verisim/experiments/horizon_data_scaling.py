"""HS1.2 -- the data cross-axis at fixed capacity (SPEC-10 §5; resolves the HS1.1 confound).

HS1.1 ([`horizon_scaling`](./horizon_scaling.py), SPEC-10 §4.2) found the free-running faithful
horizon `H_free = H_ε(ρ=0)` is **non-monotone in capacity**: it peaks at `l` then *declines* at `xl`
/ `xxl`, even as one-step accuracy `p` stays flat and high. That decline is confounded between two
explanations the program must separate:

  - **a capacity wall** -- compounding genuinely worsens once the model is large (the bigger model's
    own drift carries it off-distribution faster), so no data budget rescues it; or
  - **data starvation** -- the big model simply overfits a fixed coverage set (high in-distribution
    `p`, collapsing out-of-distribution horizon is the overfit signature), so *more data* would lift
    the horizon back up. This is the Chinchilla compute-optimal lesson: capacity outran the tokens.

This is the literal "method fails vs. too small" question, now for the *data* axis. HS1.2 answers it
the only clean way: **hold capacity fixed** (at `xl`, the cell where the decline first bites and the
ood horizon efficiency `η` first crosses below 1) and **sweep the shared coverage-set size**. If
`H_free` rises monotonically with data toward (or past) the `l` peak, the decline was starvation,
not a wall -- the prescription is "scale data with capacity," not "stop scaling." If `H_free` stays
flat/low as data grows, the wall is real at this capacity. Either verdict is first-class.

Reuses the HS1 harness internals verbatim (``_train_scaled``, ``_evaluate``, the EN1 dataset
builder); the only new axis is the number of trajectory seeds in the shared coverage set. CPU-local
(the SPEC-9 envelope); CI runs only the smoke instance.
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


def _data_points(d: dict[str, Any]) -> tuple[int, ...]:
    """The data axis: each entry is a trajectory-seed count (× steps_per_traj = transitions)."""
    return tuple(d.get("data_seeds", (6, 12, 24, 48)))


# CSV is keyed by ``n_train`` (transitions in the coverage set) rather than params -- here the
# x-axis is data, with capacity held fixed. The per-cell metrics are otherwise identical to HS1's.
CSV_HEADER = "n_train,n_seeds,metric,mean,ci_lo,ci_hi,n"


def run_data_scaling(
    config: HorizonScalingConfig,
    data_seeds: tuple[int, ...],
    *,
    oracle: NetOracle | None = None,
) -> list[ScaleStat]:
    """Sweep the coverage-set size at the single fixed capacity ``config.scales[0]``.

    Returns :class:`ScaleStat` rows whose ``params`` field carries the **number of transitions** in
    the coverage set (the data x-axis) and whose ``scale`` label is the seed count -- so the same
    reduced-record plumbing and bootstrap CIs as HS1 apply, keyed by data instead of capacity.
    """
    if len(config.scales) != 1:
        raise ValueError("HS1.2 fixes capacity: config must specify exactly one scale")
    scale: ModelScale = config.scales[0]
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)

    stats: list[ScaleStat] = []
    for n_seeds in data_seeds:
        base = replace(config.base, train_seeds=tuple(range(n_seeds)))
        examples = build_net_dataset(
            oracle, vocab, net, driver=base.train_driver, seeds=base.train_seeds,
            n_steps=base.train_steps_per_traj,
        )
        n_train = len(examples)
        cfg = replace(config, base=base)
        per_seed = []
        for seed in config.seeds:
            cell = _cell(cfg, scale, seed, oracle, net, vocab, examples)
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

    parser = argparse.ArgumentParser(description="Run HS1.2 (data cross-axis at fixed capacity).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures/horizon_data_scaling.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_data_scaling.png")
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    config = HorizonScalingConfig.from_dict(raw)
    data_seeds = _data_points(raw)
    stats = run_data_scaling(config, data_seeds)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    by_n: dict[int, dict[str, float]] = {}
    for s in stats:
        by_n.setdefault(s.params, {})[s.metric] = s.mean
    cap = config.scales[0]
    print(f"  (fixed capacity {cap.label}, params={cap.params})")
    for n in sorted(by_n):
        d = by_n[n]
        print(f"  n_train={n:>6d}  [id]  p={d['one_step_acc_id']:.3f} "
              f"H_free={d['h_free_id']:.2f} eta={d['horizon_efficiency_id']:.2f}   "
              f"[ood] p={d['one_step_acc_ood']:.3f} H_free={d['h_free_ood']:.2f} "
              f"eta={d['horizon_efficiency_ood']:.2f}")
    try:
        from figures.plot_horizon_data_scaling import plot_horizon_data_scaling

        plot_horizon_data_scaling(stats, cap, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
