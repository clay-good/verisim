"""ED4 (consistency-level, learned arm) -- the *absolute*-predictability H20 on the real `M_θ`.

ED4-consistency (:mod:`.ed4_consistency`) sweeps the declared consistency model on the *synthetic*
proposer and measures the **H19 gap** (consistency-faithful minus bit-faithful horizon) per level.
It explicitly defers the *absolute*-predictability form of H20 to the learned model, and says why:
on the synthetic proposer the free-running horizon at equal noise is **confounded by delta
composition** -- a `put` is one local write + N async messages under `eventual` but N synchronous
writes under `linearizable`, so "equal noise" is not equal difficulty across levels. A *learned*
`M_θ`, trained on each level's own dynamics, removes that confound: each model is asked to predict
the world it was trained on, so the free-running horizon is an honest measure of **how predictable
that consistency model is** -- the H20 question SPEC-7 §10.2 actually poses ("weaker consistency is
harder to predict, because weaker models admit exponentially more legal histories to track").

This module closes that named DS7 deferral (what ED1-learned is to ED1): for each consistency level
it trains the flat DS4 `M_θ` (exactly as :mod:`.ed2_learned`, same init seed so the only difference
is the *world*) and measures its free-running (ρ=0) **bit-faithful** horizon `H_ε` -- the absolute
predictability -- alongside the **consistency-faithful** horizon and their H19 gap, with bootstrap
CIs over held-out seeds.

The H20 prediction, made absolute: **`linearizable` is more predictable than `eventual`** -- strong
consistency removes the in-flight medium (synchronous writes enqueue nothing; CP write-rejection
under partition keeps every replica current), so there is *less hidden state* for the model to track
and it free-runs further. The H19 gap, in turn, should appear under `eventual` (errors hide in the
consistency-invisible in-flight medium) and collapse under `linearizable` (no medium) -- the learned
confirmation of the synthetic ED4-consistency reading, now on a real error distribution where the
absolute horizons are comparable. Whatever it shows is the result; the honest negative (a learned
model that predicts `eventual` *better*, or shows no gap) is reported, not hidden.

CI runs a tiny smoke instance; the committed figure comes from the local
``configs/ed4_consistency_learned.json`` run. Torch-backed (the ``[model]`` extra), like every
learned arm.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distloop import FixedTierPolicy, budget_for_rho, run_dist_rollout
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.experiments.ed4_consistency import CONSISTENCY_LEVELS
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon


@dataclass(frozen=True)
class ED4ConsistencyLearnedConfig:
    name: str = "ed4-consistency-learned"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    # training (mirrors ED2LearnedConfig -- the same flat DS4 M_θ, one per level)
    train_driver: str = "contention"
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
    # evaluation (free-running ρ=0 exposes the model, not the loop)
    eval_driver: str = "contention"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105, 106, 107)
    n_steps: int = 40
    epsilon: float = 0.0
    levels: tuple[str, ...] = CONSISTENCY_LEVELS  # ("linearizable", "eventual"): strong -> weak

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED4ConsistencyLearnedConfig:
        b = ED4ConsistencyLearnedConfig()
        return ED4ConsistencyLearnedConfig(
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
            epsilon=d.get("epsilon", b.epsilon),
            levels=tuple(d.get("levels", b.levels)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED4ConsistencyLearnedConfig:
        return ED4ConsistencyLearnedConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED4ConsistencyLearnedResult:
    """Per level: the learned model's free-running bit- and consistency-faithful horizons + gap."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    #: the absolute-predictability H20 verdict: the bit-faithful horizon, strong vs weak level.
    verdict: dict[str, Any] = field(default_factory=dict)


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


def train_model_for_level(
    config: ED4ConsistencyLearnedConfig, level: str
) -> tuple[Any, DistOracle, DistConfig]:
    """Train the flat `M_θ` on ``level``'s own dynamics -- process-reproducibly, same init seed."""
    import torch

    from verisim.distmodel import DistVocab, NeuralDistWorldModel, build_dist_dataset
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_supervised

    dist = replace(config.dist, consistency_model=level)
    oracle: DistOracle = ReferenceDistOracle(dist)

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)  # process-reproducibility (SPEC-2 §12)

    vocab = DistVocab(dist, max_int=config.max_int)
    examples = build_dist_dataset(
        oracle, vocab, dist, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
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
    return NeuralDistWorldModel(model, vocab), oracle, dist


def _both_horizons_learned(
    world_model: Any, oracle: DistOracle, dist: DistConfig,
    config: ED4ConsistencyLearnedConfig, eval_seed: int,
) -> tuple[int, int]:
    """Free-running (ρ=0) ``(bit_faithful_horizon, consistency_faithful_horizon)`` for the model."""
    s0 = DistributedState.initial(dist)
    actions = _eval_actions(oracle, dist, config.eval_driver, eval_seed, config.n_steps)
    record = run_dist_rollout(
        world_model, oracle, s0, actions, fixed_interval_for_rho(0.0), epsilon=config.epsilon,
        config=dist, tier_policy=FixedTierPolicy("bit_exact"),
        budget=budget_for_rho(0.0, len(actions)), seed=eval_seed,
    )
    bit_h = record.faithful_horizon
    cons_h = faithful_horizon(record.config["consistency_divergences"], config.epsilon)
    return bit_h, cons_h


def run_ed4_consistency_learned(
    config: ED4ConsistencyLearnedConfig | None = None,
) -> ED4ConsistencyLearnedResult:
    """Train one flat `M_θ` per consistency level; measure free-running predictability (H20)."""
    config = config or ED4ConsistencyLearnedConfig()
    result = ED4ConsistencyLearnedResult()

    by_level: dict[str, dict[str, Any]] = {}
    for level in config.levels:
        world_model, oracle, dist = train_model_for_level(config, level)
        cells = [_both_horizons_learned(world_model, oracle, dist, config, s)
                 for s in config.eval_seeds]
        bit = [float(b) for b, _ in cells]
        cons = [float(c) for _, c in cells]
        gap = [c - b for b, c in zip(bit, cons, strict=True)]
        bit_lo, bit_hi = bootstrap_ci(bit, seed=0)
        gap_lo, gap_hi = bootstrap_ci(gap, seed=0)
        row = {
            "level": level,
            "bit_h": fmean(bit), "bit_lo": bit_lo, "bit_hi": bit_hi,
            "cons_h": fmean(cons),
            "gap": fmean(gap), "gap_lo": gap_lo, "gap_hi": gap_hi,
            "consistency_outlasts": gap_lo > 0.0,
        }
        result.rows.append(row)
        by_level[level] = row

    # the absolute-predictability H20 verdict: is the strong level more predictable than the weak?
    strong, weak = config.levels[0], config.levels[-1]
    bit_strong = by_level[strong]["bit_h"]
    bit_weak = by_level[weak]["bit_h"]
    result.verdict = {
        "strong_level": strong, "weak_level": weak,
        "bit_h_strong": bit_strong, "bit_h_weak": bit_weak,
        # H20: weaker consistency is harder to predict -> the strong level free-runs further.
        "strong_more_predictable": bit_strong > bit_weak,
        "gap_strong": by_level[strong]["gap"], "gap_weak": by_level[weak]["gap"],
        # H19 (learned): the gap appears under the weak level and collapses under the strong one.
        "gap_collapses_under_strong": by_level[strong]["gap"] < by_level[weak]["gap"],
    }
    return result


CSV_HEADER = "panel,level,bit_h,bit_lo,bit_hi,cons_h,gap,gap_lo,gap_hi"


def write_csv(result: ED4ConsistencyLearnedResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"row,{r['level']},{r['bit_h']:.4f},{r['bit_lo']:.4f},{r['bit_hi']:.4f},"
        f"{r['cons_h']:.4f},{r['gap']:.4f},{r['gap_lo']:.4f},{r['gap_hi']:.4f}"
        for r in result.rows
    ]
    v = result.verdict
    rows.append(f"verdict,{v['strong_level']}->{v['weak_level']},{v['bit_h_strong']:.4f},,,"
                f"{v['bit_h_weak']:.4f},{v['gap_strong']:.4f},{v['gap_weak']:.4f},")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def _print_summary(result: ED4ConsistencyLearnedResult) -> None:
    print("ED4 consistency-level (learned M_θ) — absolute free-running predictability (H20/H19):")
    for r in result.rows:
        print(f"  [{r['level']:13s}] bit H={r['bit_h']:.1f} [{r['bit_lo']:.1f},{r['bit_hi']:.1f}]"
              f"  cons H={r['cons_h']:.1f}  gap={r['gap']:.1f} "
              f"[{r['gap_lo']:.1f},{r['gap_hi']:.1f}]")
    v = result.verdict
    pred = "MORE predictable" if v["strong_more_predictable"] else "not more predictable"
    print(f"  H20 (absolute): {v['strong_level']} bit H={v['bit_h_strong']:.1f} vs "
          f"{v['weak_level']} {v['bit_h_weak']:.1f}  → strong is {pred}")


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED4 consistency-level learned arm (absolute-predictability H20 on M_θ)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed4_consistency_learned.csv")
    parser.add_argument("--plot", type=str, default="figures/ed4_consistency_learned.png")
    args = parser.parse_args()
    config = (
        ED4ConsistencyLearnedConfig.from_json_file(args.config)
        if args.config
        else ED4ConsistencyLearnedConfig()
    )
    result = run_ed4_consistency_learned(config)
    _print_summary(result)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    try:
        from figures.plot_ed4_consistency_learned import plot_ed4_consistency_learned

        plot_ed4_consistency_learned(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
