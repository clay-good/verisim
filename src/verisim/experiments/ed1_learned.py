"""ED1-learned -- the distributed `H_ε(ρ)` curve with the **real** learned `M_θ` (SPEC-7 §0, DS6).

ED1 (:mod:`verisim.experiments.ed1`) plotted the prime-directive curve and the H17 tiered-oracle
tradeoff on a *synthetic* tunable-noise proposer -- the apparatus that exists before the learned
model. This module closes the DS6-deferred *learned-model curve*: it trains the flat DS4 `M_θ`
(:class:`~verisim.distmodel.NeuralDistWorldModel`) on seeded oracle rollouts and runs it through the
**same** tiered loop, so the curve and the H17 cost are measured on a *real* error distribution
rather than a dialled one -- the network's EN1 / host's EH1 step for the distributed world.

The honest, non-obvious finding the real model makes that the synthetic one could only hypothesize:
the **constrained decoder removes the "gross" (out-of-vocab) error class by construction** (every
prediction is a grammar-valid delta -- DS4 incr 2), so the learned model's residual errors are the
*subtle*, invariant-respecting kind (a wrong value/version, a corrupted in-flight payload) that the
**cheap metamorphic tier cannot catch**. The H17 panel here therefore measures whether a cheap tier
buys faithful horizon per oracle-dollar *for the errors a real model actually makes* -- and the
expectation, which the experiment reports rather than assumes, is that it does not: you must pay the
symbolic/bit-exact tiers. A negative-for-the-cheap-tier result is first-class (the repo norm).

Two panels from one trained model:

  - **left -- `H_ε(ρ)`**: free-running → fully-consulted faithful horizon at the bit-exact tier
    (directly comparable to v0/EN1/EH1 and to ED1's synthetic curve), with a bootstrap-CI band;
  - **right -- the H17 tradeoff for the real model**: at full consultation (ρ=1), the faithful
    horizon and **oracle-dollar per faithful step** under each fixed tier (`metamorphic` /
    `symbolic` / `bit_exact`) plus the cheapest-refutation `escalate` policy -- which tier the real
    model's errors actually get caught at.

CI runs a tiny smoke instance; the committed figure comes from the local `configs/ed1_learned.json`
run (the `figures/reproduce.sh` discipline). Torch-backed (the `[model]` extra), like EN1/EH1.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distloop import (
    EscalatingTierPolicy,
    FixedTierPolicy,
    budget_for_rho,
    run_dist_rollout,
)
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci

#: The fixed tiers compared in the H17 panel (cheapest, mid, full-truth); `escalate` is added
#: separately as the cheapest-refutation policy.
H17_TIERS = ("metamorphic", "symbolic", "bit_exact")


@dataclass(frozen=True)
class ED1LearnedConfig:
    name: str = "ed1-learned"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    # training (mirrors EN1Config)
    train_driver: str = "uniform"
    train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    train_steps_per_traj: int = 40
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 512
    train_iters: int = 700
    lr: float = 3e-3
    model_seed: int = 0
    max_int: int = 256
    # evaluation (in-distribution `uniform`, the EN1 "low-difficulty" analogue -- a genuine
    # non-zero free-run floor; the H17 finding is the same on the harder `contention` eval)
    eval_driver: str = "uniform"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    n_steps: int = 32
    rhos: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    tiers: tuple[str, ...] = H17_TIERS
    epsilon: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED1LearnedConfig:
        b = ED1LearnedConfig()
        return ED1LearnedConfig(
            name=d.get("name", b.name),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            n_layer=d.get("n_layer", b.n_layer),
            n_head=d.get("n_head", b.n_head),
            n_embd=d.get("n_embd", b.n_embd),
            block_size=d.get("block_size", b.block_size),
            train_iters=d.get("train_iters", b.train_iters),
            lr=d.get("lr", b.lr),
            model_seed=d.get("model_seed", b.model_seed),
            max_int=d.get("max_int", b.max_int),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            rhos=tuple(d.get("rhos", b.rhos)),
            tiers=tuple(d.get("tiers", b.tiers)),
            epsilon=d.get("epsilon", b.epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED1LearnedConfig:
        return ED1LearnedConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED1LearnedResult:
    """The learned-model ED1 deliverables: the `H_ε(ρ)` curve points + the H17 real-model cells."""

    curve: list[dict[str, float]] = field(default_factory=list)  # per ρ: h_eps mean + CI
    h17: list[dict[str, Any]] = field(default_factory=list)  # per tier/policy: h_eps, dollars


def _eval_actions(
    oracle: DistOracle, config: DistConfig, driver: str, seed: int, n: int
) -> list[DistAction]:
    drv = DistDriver(driver, config, random.Random(seed))
    state = DistributedState.initial(config)
    actions: list[DistAction] = []
    for _ in range(n):
        action = drv.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return actions


def train_model(config: ED1LearnedConfig, oracle: DistOracle) -> Any:
    """Train the flat distributed `M_θ` -- process-reproducibly, as EN1/EH1 do."""
    import torch

    from verisim.distmodel import DistVocab, NeuralDistWorldModel, build_dist_dataset
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_supervised

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)  # process-reproducibility (SPEC-2 §12)

    vocab = DistVocab(config.dist, max_int=config.max_int)
    examples = build_dist_dataset(
        oracle, vocab, config.dist, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    # The decoder windows the context, but training collates to the longest example, so the model's
    # block_size must cover it -- clamp up if a long-prompt rollout exceeds the configured size.
    needed = max(len(p) + len(t) for p, t in examples) + 8
    block_size = max(config.block_size, needed)
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=block_size,
            n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
        )
    )
    train_supervised(
        model, examples, vocab.pad, steps=config.train_iters, lr=config.lr, seed=config.model_seed
    )
    return NeuralDistWorldModel(model, vocab)


def run_ed1_learned(
    config: ED1LearnedConfig | None = None, *, oracle: DistOracle | None = None
) -> ED1LearnedResult:
    """Train the flat `M_θ`, run it through the tiered loop: the `H_ε(ρ)` curve + H17 cells."""
    config = config or ED1LearnedConfig()
    oracle = oracle or ReferenceDistOracle(config.dist)
    world_model = train_model(config, oracle)
    result = ED1LearnedResult()

    eval_actions = {
        s: _eval_actions(oracle, config.dist, config.eval_driver, s, config.n_steps)
        for s in config.eval_seeds
    }

    def _rollout(tier: str, rho: float, seed: int, *, escalate: bool) -> tuple[int, int]:
        actions = eval_actions[seed]
        policy = EscalatingTierPolicy() if escalate else FixedTierPolicy(tier)
        record = run_dist_rollout(
            world_model, oracle, DistributedState.initial(config.dist), actions,
            fixed_interval_for_rho(rho), epsilon=config.epsilon, config=config.dist,
            tier_policy=policy, budget=budget_for_rho(rho, len(actions)), seed=seed,
        )
        return record.faithful_horizon, record.config["oracle_dollars"]

    # left panel: the H_ε(ρ) curve at the bit-exact tier (the standard prime-directive shape).
    for rho in config.rhos:
        hs = [_rollout("bit_exact", rho, s, escalate=False)[0] for s in config.eval_seeds]
        lo, hi = bootstrap_ci([float(h) for h in hs], seed=0)
        result.curve.append({"rho": rho, "h_eps": fmean(hs), "ci_lo": lo, "ci_hi": hi})

    # right panel: at full consultation, the real model's faithful horizon + oracle-$/faithful-step
    # under each fixed tier and the cheapest-refutation escalate policy -- where its errors land.
    arms: list[tuple[str, bool]] = [(t, False) for t in config.tiers] + [("escalate", True)]
    for tier, escalate in arms:
        cells = [_rollout(tier, 1.0, s, escalate=escalate) for s in config.eval_seeds]
        h = fmean(h for h, _ in cells)
        dollars = fmean(d for _, d in cells)
        result.h17.append({
            "tier": tier, "h_eps": h, "dollars": dollars,
            "dollars_per_step": (dollars / h) if h else float("inf"),
        })
    return result


CSV_HEADER = "panel,key,x,tier,h_eps,dollars,dollars_per_step,ci_lo,ci_hi"


def write_csv(result: ED1LearnedResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in result.curve:
        rows.append(f"curve,rho,{p['rho']},bit_exact,"
                    f"{p['h_eps']:.4f},,,{p['ci_lo']:.4f},{p['ci_hi']:.4f}")
    for c in result.h17:
        rows.append(f"h17,{c['tier']},,{c['tier']},"
                    f"{c['h_eps']:.4f},{c['dollars']:.4f},{c['dollars_per_step']:.4f},,")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run ED1-learned (real-model distributed H_eps).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed1_learned.csv")
    parser.add_argument("--plot", type=str, default="figures/ed1_learned.png")
    args = parser.parse_args()
    config = ED1LearnedConfig.from_json_file(args.config) if args.config else ED1LearnedConfig()
    result = run_ed1_learned(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  H_eps(rho):", [(p["rho"], round(p["h_eps"], 1)) for p in result.curve])
    for c in result.h17:
        print(f"  {c['tier']:11s} H={c['h_eps']:.1f} "
              f"$={c['dollars']:.0f} $/step={c['dollars_per_step']:.1f}")
    try:
        from figures.plot_ed1_learned import plot_ed1_learned

        plot_ed1_learned(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
