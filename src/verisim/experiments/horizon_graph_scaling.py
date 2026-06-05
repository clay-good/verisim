"""HS3 (incr 1) -- the faithful-horizon scaling law for the *structured* arm (SPEC-10 §5).

HS1-HS2 swept the **flat** transformer `M_θ` (the deliberate choice, §2: a clean capacity exponent
uncontaminated by message-passing depth or the RSSM belief). HS3 asks the question that choice
deferred: **does a *structured* model change the `η`-vs-capacity verdict?** Concretely -- run the
*identical* capacity axis, the same `H_free`/`p`/`H_indep`/`η` grid, the same network world and
oracle, but with the **GNN+RSSM graph arm** (SPEC-5 §6.1-6.2, the NW8 proposer that beats the flat
arm ~6.6× on delta-exact, EN4/H11) as the proposer. The flat-arm curve (HS1) is the baseline the
structured curve is read against, on the same x-axis (`params ≈ n_layer · d_model²`).

The pre-registered question (SPEC-10 §5, HS3): structure is known to lift *per-step* accuracy (EN4);
does it also (a) lift the free-running horizon `H_free` at matched capacity, and (b) change *how*
`η` scales -- i.e. is the capacity-buys-horizon verdict a property of the loop alone (HS1/HS2) or
does the inductive bias bend it? Either branch is bankable: a structured arm that buys more horizon
at matched params localizes part of the floor to the *proposer*, not just resourcing; a structured
arm with the *same* `η` trend reinforces that the verdict is the loop's.

This is HS3 increment 1 (the graph arm). The world-size cross-axis (the interaction with SPEC-9's
`O(N²)` host-count axis) is increment 2 / future. The committed sweep is local CPU (the SPEC-9
envelope discipline); CI runs only the smoke instance.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon
from verisim.net.action import NetAction
from verisim.net.config import DEFAULT_NET_CONFIG, NetConfig
from verisim.net.state import NetworkState
from verisim.netdata import NetDriver
from verisim.netdelta.edits import NetDelta
from verisim.netloop import PartialNetOracle, budget_for_rho, run_net_rollout
from verisim.netloop.model import NetModel
from verisim.netmetrics.exact import delta_exact_rate
from verisim.netmodel import NetVocab
from verisim.netmodel.graph_model import build_graph_model
from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .en1 import EN1Config, eval_actions

# Reuse HS1's independence math, per-cell stat, CSV writer, and metric grid verbatim -- HS3 is the
# *same* measurement with a different proposer, so the harness shape is shared (SPEC-10 §5).
from .horizon_scaling import (
    CSV_HEADER,
    METRICS,
    ScaleStat,
    independence_horizon,
    write_csv,
)

__all__ = [
    "CSV_HEADER",
    "DEFAULT_GRAPH_SCALES",
    "METRICS",
    "GraphHorizonScalingConfig",
    "GraphScale",
    "run_graph_horizon_scaling",
    "write_csv",
]


@dataclass(frozen=True)
class GraphScale:
    """One point on the graph-arm capacity axis. ``params ≈ n_layer · d_model²`` orders the x-axis,
    the *same* formula HS1's flat ``ModelScale`` uses, so the two curves overlay on one axis."""

    label: str
    d_model: int
    n_layer: int = 2
    mp_rounds: int = 2  # message-passing depth (the graph-specific capacity knob)
    n_head: int = 2
    train_steps: int = 1200  # graph iters; scaled up with capacity so each size converges

    @property
    def params(self) -> int:
        return self.n_layer * self.d_model * self.d_model


# The same capacity points as HS1's flat axis (matched ``params``), so the structured curve is read
# against the flat one directly; ``mp_rounds`` adds graph depth at the larger sizes.
DEFAULT_GRAPH_SCALES: tuple[GraphScale, ...] = (
    GraphScale("xs", d_model=32, n_layer=1, mp_rounds=2, train_steps=1000),
    GraphScale("s", d_model=64, n_layer=2, mp_rounds=2, train_steps=1200),
    GraphScale("m", d_model=128, n_layer=2, mp_rounds=3, train_steps=1500),
    GraphScale("l", d_model=192, n_layer=3, mp_rounds=3, n_head=4, train_steps=1800),
)


@dataclass(frozen=True)
class GraphHorizonScalingConfig:
    """HS3 config: the graph-arm analogue of :class:`HorizonScalingConfig`."""

    name: str = "hs3-graph"
    base: EN1Config = field(default_factory=EN1Config)
    scales: tuple[GraphScale, ...] = DEFAULT_GRAPH_SCALES
    seeds: tuple[int, ...] = (0, 1, 2)  # one freshly-trained model per seed; CIs are over seeds
    noise_prob: float = 0.0  # §6.3 noise-injection lever, off by default (clean capacity reading)
    warmup_frac: float = 0.0  # 0 = flat LR (the HS3 recipe); >0 = warmup+cosine (HS3-T, §4.11)
    lr: float = 3e-3
    batch_size: int = 32
    eval_driver: str = "weighted"  # in-distribution (id) free-running
    eval_driver_hard: str = "adversarial"  # harder/out-of-distribution (ood)
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 32
    one_step_seeds: tuple[int, ...] = (200, 201)
    one_step_steps: int = 32
    epsilon: float = 0.0
    verbose: bool = False

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GraphHorizonScalingConfig:
        b = GraphHorizonScalingConfig()
        scales = (
            tuple(
                GraphScale(
                    label=s["label"], d_model=s["d_model"], n_layer=s.get("n_layer", 2),
                    mp_rounds=s.get("mp_rounds", 2), n_head=s.get("n_head", 2),
                    train_steps=s.get("train_steps", 1200),
                )
                for s in d["scales"]
            )
            if "scales" in d
            else b.scales
        )
        return GraphHorizonScalingConfig(
            name=d.get("name", b.name),
            base=EN1Config.from_dict(d.get("base", {})),
            scales=scales,
            seeds=tuple(d.get("seeds", b.seeds)),
            noise_prob=d.get("noise_prob", b.noise_prob),
            warmup_frac=d.get("warmup_frac", b.warmup_frac),
            lr=d.get("lr", b.lr),
            batch_size=d.get("batch_size", b.batch_size),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_driver_hard=d.get("eval_driver_hard", b.eval_driver_hard),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            epsilon=d.get("epsilon", b.epsilon),
            verbose=d.get("verbose", b.verbose),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> GraphHorizonScalingConfig:
        return GraphHorizonScalingConfig.from_dict(json.loads(Path(path).read_text()))


def _eval_triples(
    oracle: NetOracle, net: NetConfig, seeds: tuple[int, ...], n_steps: int, driver: str
) -> list[tuple[NetworkState, NetAction, NetDelta]]:
    """Seeded held-out ``(state, action, true_delta)`` triples for the one-step `p` measurement."""
    triples: list[tuple[NetworkState, NetAction, NetDelta]] = []
    for seed in seeds:
        drv = NetDriver(name=driver, config=net, rng=random.Random(seed))
        state = NetworkState.initial(net.hosts)
        for _ in range(n_steps):
            action = drv.sample(state)
            result = oracle.step(state, action)
            triples.append((state, action, result.delta))
            state = result.state
    return triples


def _train_graph_scaled(
    config: GraphHorizonScalingConfig, scale: GraphScale, seed: int, vocab: NetVocab,
    net: NetConfig, oracle: NetOracle,
) -> NetModel:
    """Build + minibatch-train one graph arm at ``scale``/``seed`` (the EN4 graph trainer)."""
    examples = build_graph_dataset(
        oracle, vocab, net, driver=config.base.train_driver, seeds=config.base.train_seeds,
        n_steps=config.base.train_steps_per_traj, noise_prob=config.noise_prob,
    )
    wm = build_graph_model(
        vocab, net, d_model=scale.d_model, mp_rounds=scale.mp_rounds,
        n_layer=scale.n_layer, n_head=scale.n_head, seed=seed,
    )
    train_graph_model(
        wm, examples, steps=scale.train_steps, lr=config.lr, batch_size=config.batch_size,
        seed=seed, warmup_frac=config.warmup_frac,
    )
    return wm


def _evaluate(
    wm: NetModel, config: GraphHorizonScalingConfig, oracle: NetOracle, net: NetConfig, driver: str
) -> dict[str, float]:
    """One trained graph model's `p`, free-running `H_ε(ρ=0)`, the i.i.d. baseline, and `η`."""
    triples = _eval_triples(oracle, net, config.one_step_seeds, config.one_step_steps, driver)
    p = delta_exact_rate((wm.predict_delta(s, a), true) for s, a, true in triples)

    partial = PartialNetOracle(oracle)
    horizons: list[float] = []
    for eseed in config.eval_seeds:
        actions = eval_actions(oracle, net, driver, eseed, config.eval_steps)
        rollout = run_net_rollout(
            wm, partial, NetworkState.initial(net.hosts), actions,
            fixed_interval_for_rho(0.0), epsilon=config.epsilon,
            budget=budget_for_rho(0.0, len(actions)), seed=eseed,
        )
        horizons.append(float(faithful_horizon(list(rollout.divergences), config.epsilon)))

    h_free = fmean(horizons)
    h_indep = independence_horizon(p, cap=float(config.eval_steps))
    return {
        "one_step_acc": p,
        "h_free": h_free,
        "h_indep": h_indep,
        "horizon_efficiency": (h_free / h_indep) if h_indep > 0 else 0.0,
    }


def _cell(
    config: GraphHorizonScalingConfig, scale: GraphScale, seed: int, oracle: NetOracle,
    net: NetConfig, vocab: NetVocab,
) -> dict[str, float]:
    """Train one graph model at ``scale``/``seed``; evaluate it id and ood."""
    wm = _train_graph_scaled(config, scale, seed, vocab, net, oracle)
    out: dict[str, float] = {}
    for regime, driver in (("id", config.eval_driver), ("ood", config.eval_driver_hard)):
        for base, value in _evaluate(wm, config, oracle, net, driver).items():
            out[f"{base}_{regime}"] = value
    return out


def run_graph_horizon_scaling(
    config: GraphHorizonScalingConfig | None = None, *, oracle: NetOracle | None = None
) -> list[ScaleStat]:
    """Sweep the graph-arm capacity axis; reduce each metric over seeds to a mean + bootstrap CI."""
    config = config or GraphHorizonScalingConfig()
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)

    stats: list[ScaleStat] = []
    for scale in config.scales:
        per_seed = []
        for seed in config.seeds:
            cell = _cell(config, scale, seed, oracle, net, vocab)
            per_seed.append(cell)
            if config.verbose:  # pragma: no cover - progress for the long local sweep
                print(
                    f"  [{scale.label} N={scale.params} seed={seed}] "
                    f"p={cell['one_step_acc_id']:.3f} H_free={cell['h_free_id']:.2f} "
                    f"(ood H_free={cell['h_free_ood']:.2f})",
                    flush=True,
                )
        for metric in METRICS:
            values = [c[metric] for c in per_seed]
            lo, hi = bootstrap_ci(values, seed=0)
            stats.append(
                ScaleStat(scale.label, scale.params, metric, fmean(values), lo, hi, len(values))
            )
    return stats


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run HS3 (faithful-horizon scaling, graph arm).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/horizon_graph_scaling.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_graph_scaling.png")
    args = parser.parse_args()
    config = (
        GraphHorizonScalingConfig.from_json_file(args.config)
        if args.config
        else GraphHorizonScalingConfig()
    )
    stats = run_graph_horizon_scaling(config)
    path = write_csv(stats, args.out)
    print(f"wrote {len(stats)} rows to {path}")
    by_scale: dict[str, dict[str, float]] = {}
    for s in stats:
        by_scale.setdefault(s.scale, {"params": float(s.params)})[s.metric] = s.mean
    for label in sorted(by_scale, key=lambda k: by_scale[k]["params"]):
        d = by_scale[label]
        print(f"  {label:3s} N={int(d['params']):>8d}  "
              f"[id]  p={d['one_step_acc_id']:.3f} H_free={d['h_free_id']:.2f} "
              f"H_indep={d['h_indep_id']:.2f} eta={d['horizon_efficiency_id']:.2f}   "
              f"[ood] p={d['one_step_acc_ood']:.3f} H_free={d['h_free_ood']:.2f} "
              f"eta={d['horizon_efficiency_ood']:.2f}")
    try:
        from figures.plot_horizon_scaling import plot_horizon_scaling

        plot_horizon_scaling(
            stats, args.plot,
            suptitle="HS3 — the faithful-horizon scaling law, STRUCTURED graph arm (SPEC-10, H26)",
            left_title="Does structure change how capacity buys horizon?",
        )
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
