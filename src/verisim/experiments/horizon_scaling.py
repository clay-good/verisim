"""HS1 -- the faithful-horizon scaling law (SPEC-10, the prime directive scaled by capacity).

Every prior `H_ε(ρ)` curve in this program (E1, EN1, EH1) is the **floor+cliff**: free-running
faithful horizon `H_ε(ρ=0)` sits on the floor (the model drifts in ~1 step), the interior is flat,
and only `ρ=1` (full consultation) reaches the ceiling. The standing objection is **scale** -- the
models are tiny and the negatives are confounded with "too small to be interesting" (report,
*Threats to validity*). HS1 attacks that objection head-on, for the headline metric itself:

    **Does free-running faithful horizon grow with model capacity, or is the
    one-step -> horizon compounding gap fundamental?**

The design isolates the question the way the scaling-law literature (Kaplan 2020; Hoffmann 2022) and
the compounding-error literature (Ross & Bagnell, DAgger 2010 -- error compounds as O(εT²)) frame
it. For each model size on a wide capacity axis (a flat network `M_θ`, params ≈ `n_layer · n_embd²`,
swept ~30-100×) and each training seed, HS1 measures two numbers on held-out rollouts:

  - **one-step acceptance `p`** -- the teacher-forced fraction of steps whose predicted delta is
    exactly the oracle's (the per-step accuracy capacity is known to lift, SPEC-2.1);
  - **free-running horizon `H_free` = `H_ε(ρ=0)`** -- how many steps the model self-rolls before
    diverging, with no oracle.

The sharp comparison is `H_free` against the **independence (geometric) prediction** `H_indep =
p/(1-p)` -- the horizon you would get if per-step failures were i.i.d. (no compounding). Their
ratio, the **horizon efficiency** `η = H_free / H_indep`, is the scale-free headline: η → 1 means
per-step accuracy *is* the whole story and capacity buys horizon; η ≪ 1 and flat-in-scale means
compounding (the off-distribution states drift induces) is the wall, and you cannot scale out of it
-- the case for verification as a *primitive*, not a patch. H26 (SPEC-10) is the falsifiable claim;
whichever way η scales is the result.

The committed sweep is local (CPU, the SPEC-9 envelope discipline); CI runs only the smoke instance.
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
from verisim.netloop import PartialNetOracle, budget_for_rho, run_net_rollout
from verisim.netmetrics.exact import delta_exact_rate
from verisim.netmodel import NetVocab, NeuralNetworkWorldModel, build_net_dataset
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .en1 import EN1Config, eval_actions


@dataclass(frozen=True)
class ModelScale:
    """One point on the capacity axis; ``params`` (≈ `n_layer · n_embd²`) orders the x-axis."""

    label: str
    n_embd: int
    n_layer: int
    n_head: int = 2
    train_steps: int = 2500  # minibatch steps; scaled up with capacity so each size is trained

    @property
    def params(self) -> int:
        """A monotone capacity proxy: transformer params are dominated by ``n_layer · n_embd²``."""
        return self.n_layer * self.n_embd * self.n_embd


# A wide, CPU-feasible default capacity axis: ~100x params from tiny to large (the SPEC-9 envelope).
DEFAULT_SCALES: tuple[ModelScale, ...] = (
    ModelScale("xs", n_embd=32, n_layer=1, train_steps=2000),
    ModelScale("s", n_embd=64, n_layer=2, train_steps=2500),
    ModelScale("m", n_embd=128, n_layer=2, train_steps=3000),
    ModelScale("l", n_embd=192, n_layer=3, n_head=4, train_steps=3500),
)


@dataclass(frozen=True)
class HorizonScalingConfig:
    name: str = "hs1-small"
    base: EN1Config = field(default_factory=EN1Config)
    scales: tuple[ModelScale, ...] = DEFAULT_SCALES
    seeds: tuple[int, ...] = (0, 1, 2)  # one freshly-trained model per seed; CIs are over seeds
    block_size: int = 128  # decode window (host deltas are short; tighter than EN1's 256 = faster)
    batch_size: int = 64  # minibatch SGD (train_batched) -- constant per-step cost (SPEC-2.1 §6)
    lr: float = 3e-3
    num_threads: int = 1  # 1 = bit-deterministic (the repo default); 0 = use all cores (the sweep)
    eval_driver: str = "weighted"  # in-distribution (id) free-running -- the trained regime
    eval_driver_hard: str = "adversarial"  # harder/out-of-distribution (ood) regime
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    eval_steps: int = 48  # horizon cap; headroom above the largest model's free-running horizon
    one_step_seeds: tuple[int, ...] = (200, 201)  # held-out (state, action) for the p measurement
    one_step_steps: int = 40
    epsilon: float = 0.0  # exact-match acceptance / horizon (the strictest, cleanest reading)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> HorizonScalingConfig:
        b = HorizonScalingConfig()
        scales = (
            tuple(
                ModelScale(
                    label=s["label"], n_embd=s["n_embd"], n_layer=s["n_layer"],
                    n_head=s.get("n_head", 2), train_steps=s.get("train_steps", 2500),
                )
                for s in d["scales"]
            )
            if "scales" in d
            else b.scales
        )
        return HorizonScalingConfig(
            name=d.get("name", b.name),
            base=EN1Config.from_dict(d.get("base", {})),
            scales=scales,
            seeds=tuple(d.get("seeds", b.seeds)),
            block_size=d.get("block_size", b.block_size),
            batch_size=d.get("batch_size", b.batch_size),
            lr=d.get("lr", b.lr),
            num_threads=d.get("num_threads", b.num_threads),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_driver_hard=d.get("eval_driver_hard", b.eval_driver_hard),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            one_step_seeds=tuple(d.get("one_step_seeds", b.one_step_seeds)),
            one_step_steps=d.get("one_step_steps", b.one_step_steps),
            epsilon=d.get("epsilon", b.epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> HorizonScalingConfig:
        return HorizonScalingConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ScaleStat:
    """A capacity cell reduced over seeds: mean + bootstrap CI for one metric."""

    scale: str
    params: int
    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.scale},{self.params},{self.metric},"
            f"{self.mean:.6f},{self.ci_lo:.6f},{self.ci_hi:.6f},{self.n}"
        )


CSV_HEADER = "scale,params,metric,mean,ci_lo,ci_hi,n"
# Each base metric is measured on two eval regimes: ``id`` (in-distribution, the trained driver) and
# ``ood`` (the harder adversarial driver -- does the horizon lift transfer off-distribution?).
_BASE_METRICS = ("one_step_acc", "h_free", "h_indep", "horizon_efficiency")
REGIMES = ("id", "ood")
METRICS = tuple(f"{m}_{r}" for r in REGIMES for m in _BASE_METRICS)


def independence_horizon(p: float, *, cap: float) -> float:
    """The i.i.d. (geometric) faithful horizon `p/(1-p)` -- horizon with no compounding.

    `p` is per-step acceptance; the expected number of consecutive faithful steps before the first
    failure, if failures were independent. Clamped at ``cap`` (an accuracy of exactly 1 over a
    finite eval would otherwise be ``inf``); ``cap`` is the eval horizon, the most a rollout shows.
    """
    if p >= 1.0:
        return cap
    return min(p / (1.0 - p), cap)


def _eval_triples(
    oracle: NetOracle, net: NetConfig, seeds: tuple[int, ...], n_steps: int, driver: str
) -> list[tuple[NetworkState, NetAction, Any]]:
    """Seeded held-out ``(state, action, true_delta)`` triples for the one-step `p` measurement."""
    triples: list[tuple[NetworkState, NetAction, Any]] = []
    for seed in seeds:
        drv = NetDriver(name=driver, config=net, rng=random.Random(seed))
        state = NetworkState.initial(net.hosts)
        for _ in range(n_steps):
            action = drv.sample(state)
            result = oracle.step(state, action)
            triples.append((state, action, result.delta))
            state = result.state
    return triples


def _train_scaled(
    config: HorizonScalingConfig, scale: ModelScale, seed: int, vocab: NetVocab, examples: list[Any]
) -> NeuralNetworkWorldModel:
    """Build + minibatch-train one flat ``M_θ`` at ``scale``/``seed`` (the SPEC-2.1 K2 loop).

    Decoupled from EN1's full-batch single-thread ``train_model``: ``train_batched`` keeps the
    per-step cost constant (so the free large coverage set is affordable, §2) and converges where a
    flat LR stalls. Threads are governed by ``config.num_threads`` (1 = bit-deterministic).
    """
    import torch

    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_batched

    if config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
    torch.manual_seed(seed)
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=config.block_size,
            n_layer=scale.n_layer, n_head=scale.n_head, n_embd=scale.n_embd,
        )
    )
    train_batched(
        model, examples, vocab.pad, steps=scale.train_steps, lr=config.lr,
        batch_size=config.batch_size, seed=seed,
    )
    return NeuralNetworkWorldModel(model, vocab)


def _evaluate(
    world_model: NeuralNetworkWorldModel, config: HorizonScalingConfig, oracle: NetOracle,
    net: NetConfig, driver: str,
) -> dict[str, float]:
    """One trained model's `p`, free-running `H_ε(ρ=0)`, the independence baseline, and `η`."""
    triples = _eval_triples(oracle, net, config.one_step_seeds, config.one_step_steps, driver)
    p = delta_exact_rate((world_model.predict_delta(s, a), true) for s, a, true in triples)

    partial = PartialNetOracle(oracle)
    horizons: list[float] = []
    for eseed in config.eval_seeds:
        actions = eval_actions(oracle, net, driver, eseed, config.eval_steps)
        rollout = run_net_rollout(
            world_model, partial, NetworkState.initial(net.hosts), actions,
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
    config: HorizonScalingConfig, scale: ModelScale, seed: int, oracle: NetOracle, net: NetConfig,
    vocab: NetVocab, examples: list[Any],
) -> dict[str, float]:
    """Train one model at ``scale``/``seed``; evaluate it in-distribution (id) and harder (ood)."""
    world_model = _train_scaled(config, scale, seed, vocab, examples)
    out: dict[str, float] = {}
    for regime, driver in (("id", config.eval_driver), ("ood", config.eval_driver_hard)):
        for base, value in _evaluate(world_model, config, oracle, net, driver).items():
            out[f"{base}_{regime}"] = value
    return out


def run_horizon_scaling(
    config: HorizonScalingConfig | None = None, *, oracle: NetOracle | None = None
) -> list[ScaleStat]:
    """Sweep the capacity axis; reduce each metric over seeds to a mean + bootstrap CI."""
    config = config or HorizonScalingConfig()
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)
    base = config.base
    # Build the coverage set ONCE -- the oracle's labels are free (SPEC-9), and the data is shared
    # across every capacity cell so the only thing that varies down the axis is the model.
    examples = build_net_dataset(
        oracle, vocab, net, driver=base.train_driver, seeds=base.train_seeds,
        n_steps=base.train_steps_per_traj,
    )

    stats: list[ScaleStat] = []
    for scale in config.scales:
        per_seed = [
            _cell(config, scale, seed, oracle, net, vocab, examples) for seed in config.seeds
        ]
        for metric in METRICS:
            values = [c[metric] for c in per_seed]
            lo, hi = bootstrap_ci(values, seed=0)
            stats.append(
                ScaleStat(scale.label, scale.params, metric, fmean(values), lo, hi, len(values))
            )
    return stats


def write_csv(stats: list[ScaleStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run HS1 (faithful-horizon scaling law).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/horizon_scaling.csv")
    parser.add_argument("--plot", type=str, default="figures/horizon_scaling.png")
    args = parser.parse_args()
    config = (
        HorizonScalingConfig.from_json_file(args.config) if args.config else HorizonScalingConfig()
    )
    stats = run_horizon_scaling(config)
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

        plot_horizon_scaling(stats, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
