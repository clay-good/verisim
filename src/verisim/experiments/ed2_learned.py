"""ED2-learned -- equal oracle-dollar budget on the **real** learned `M_θ` (SPEC-7 §12, DS7).

ED2 (:mod:`verisim.experiments.ed2`) plots the equal-dollar-budget horizon-vs-oracle-dollar
frontier on a *synthetic* tunable-noise proposer with two dialled error modes (`gross` / `subtle`),
and reads the H17/H18 verdict off it. This module closes the DS7-deferred *learned-`M_θ`
equal-dollar arm of ED2*: it trains the flat DS4 `M_θ`
(:class:`~verisim.distmodel.NeuralDistWorldModel`, exactly as :mod:`verisim.experiments.ed1_learned`
does) and runs it through the **same** equal-dollar frontier apparatus, so the budget-form H17/H18
question is answered on a *real* error distribution rather than a dialled one. It is to ED2 what
ED1-learned is to ED1: the synthetic apparatus, re-pointed at the model that actually exists.

The non-obvious finding the real model makes -- and the reason this arm is worth its own figure --
is the **honest inverse of ED2's `gross` panel**, now in budget form. ED2's synthetic sweep showed
tiering *wins* at an equal dollar budget when the proposer makes cheaply-catchable (`gross`,
out-of-vocab) errors and *loses* when it makes bit-exact-only (`subtle`, invariant-respecting)
errors. The learned model's **constrained decoder removes the gross class by construction** (every
prediction is a grammar-valid delta -- DS4 incr 2), so a real model lives entirely in ED2's `subtle`
regime: the cheap metamorphic / symbolic tiers refute nothing the bit-exact tier would not, their
frontier is flat at the floor regardless of dollars, and the cheapest-refutation `escalate` policy
*loses* to single-tier bit-exact because it pays the cheap probes before the bit-exact correction it
always ends up needing. So at a matched oracle-dollar budget, **bit-exact buys the most horizon for
the real model** -- the H17 negative ED1-learned reported per-step, confirmed here in the sharper
equal-*dollar* form, and the H18 competitive ratio at the sub-linear quarter budget is
correspondingly low. A cheap tier helps exactly when a model makes catchable-cheaply errors; a
grammar-constrained
learned model, by design, does not -- the tiered oracle's value is *model-dependent*, reported not
assumed (the repo norm: a negative-for-the-cheap-tier result is first-class).

One panel from one trained model: the per-policy faithful-horizon-vs-oracle-dollar frontier, with
the quarter budget `B/4` marked and the H17 winner / H18 ratio read off it.

CI runs a tiny smoke instance; the committed figure comes from the local `configs/ed2_learned.json`
run (the `figures/reproduce.sh` discipline). Torch-backed (the `[model]` extra), like ED1-learned.
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
from verisim.experiments.ed2 import POLICIES, _interp_horizon
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class ED2LearnedConfig:
    name: str = "ed2-learned"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    # training (mirrors ED1LearnedConfig -- the same flat DS4 M_θ)
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
    # evaluation / frontier (mirrors ED2Config; one real model, so no error-mode dial)
    eval_driver: str = "uniform"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    n_steps: int = 32
    rhos: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    policies: tuple[str, ...] = POLICIES
    epsilon: float = 0.0
    quarter_fraction: float = 0.25  # the sub-linear-cost budget the H18 ratio is read at

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED2LearnedConfig:
        b = ED2LearnedConfig()
        return ED2LearnedConfig(
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
            policies=tuple(d.get("policies", b.policies)),
            epsilon=d.get("epsilon", b.epsilon),
            quarter_fraction=d.get("quarter_fraction", b.quarter_fraction),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED2LearnedConfig:
        return ED2LearnedConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED2LearnedResult:
    """The learned-model ED2 deliverables: the per-policy frontier + the H17/H18 verdict."""

    #: per (policy, ρ): mean faithful horizon, mean oracle-dollars, CI on horizon.
    frontier: list[dict[str, Any]] = field(default_factory=list)
    #: the single real-model verdict: the equal-budget H17 winner + the H18 ratio at B/4.
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


def train_model(config: ED2LearnedConfig, oracle: DistOracle) -> Any:
    """Train the flat distributed `M_θ` -- process-reproducibly, as ED1-learned / EN1 / EH1 do."""
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


def run_ed2_learned(
    config: ED2LearnedConfig | None = None, *, oracle: DistOracle | None = None
) -> ED2LearnedResult:
    """Train the flat `M_θ`, run it through the equal-dollar frontier: the H17/H18 verdict."""
    config = config or ED2LearnedConfig()
    oracle = oracle or ReferenceDistOracle(config.dist)
    world_model = train_model(config, oracle)
    result = ED2LearnedResult()

    eval_actions = {
        s: _eval_actions(oracle, config.dist, config.eval_driver, s, config.n_steps)
        for s in config.eval_seeds
    }

    def _rollout(policy: str, rho: float, seed: int) -> tuple[int, int]:
        tier_policy = EscalatingTierPolicy() if policy == "escalate" else FixedTierPolicy(policy)
        actions = eval_actions[seed]
        record = run_dist_rollout(
            world_model, oracle, DistributedState.initial(config.dist), actions,
            fixed_interval_for_rho(rho), epsilon=config.epsilon, config=config.dist,
            tier_policy=tier_policy, budget=budget_for_rho(rho, len(actions)), seed=seed,
        )
        return record.faithful_horizon, record.config["oracle_dollars"]

    # the frontier: per policy, one (dollars, horizon) point per ρ, averaged over seeds.
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for policy in config.policies:
        for rho in config.rhos:
            cells = [_rollout(policy, rho, s) for s in config.eval_seeds]
            hs = [float(h) for h, _ in cells]
            lo, hi = bootstrap_ci(hs, seed=0)
            point = {
                "policy": policy, "rho": rho,
                "h_eps": fmean(hs), "dollars": fmean(d for _, d in cells),
                "ci_lo": lo, "ci_hi": hi,
            }
            result.frontier.append(point)
            by_policy.setdefault(policy, []).append(point)

    # the verdict: compare policies at a matched dollar budget (equal-budget H17), and read the
    # H18 competitive ratio off the frontier at the sub-linear quarter budget. One real model, so a
    # single verdict (no error-mode dial) -- the budget-form of ED1-learned's per-step H17.
    be_points = by_policy["bit_exact"]
    b_full = max(p["dollars"] for p in be_points)  # ρ=1 full-truth dollars
    ceiling = max(p["h_eps"] for p in be_points)   # full-truth horizon (the ceiling)
    b_quarter = config.quarter_fraction * b_full
    at_quarter = {
        policy: _interp_horizon(by_policy[policy], b_quarter) for policy in config.policies
    }
    winner = max(at_quarter, key=lambda p: at_quarter[p])
    result.verdict = {
        "b_full": b_full,
        "b_quarter": b_quarter,
        "ceiling": ceiling,
        "horizon_at_quarter": at_quarter,
        "h17_winner": winner,                       # most horizon per equal dollar budget
        "h17_tiering_wins": winner != "bit_exact",  # a non-bit-exact policy beats full-truth?
        "competitive_ratio": (at_quarter[winner] / ceiling) if ceiling else float("nan"),
    }
    return result


CSV_HEADER = "panel,policy,rho,dollars,h_eps,ci_lo,ci_hi,b_quarter,ceiling,winner,ratio"


def write_csv(result: ED2LearnedResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in result.frontier:
        rows.append(f"frontier,{p['policy']},{p['rho']},{p['dollars']:.4f},"
                    f"{p['h_eps']:.4f},{p['ci_lo']:.4f},{p['ci_hi']:.4f},,,,")
    v = result.verdict
    rows.append(f"verdict,{v['h17_winner']},,,,,,"
                f"{v['b_quarter']:.4f},{v['ceiling']:.4f},{v['h17_winner']},"
                f"{v['competitive_ratio']:.4f}")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED2-learned (equal-dollar-budget H17/H18 on the real learned M_θ)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed2_learned.csv")
    parser.add_argument("--plot", type=str, default="figures/ed2_learned.png")
    args = parser.parse_args()
    config = ED2LearnedConfig.from_json_file(args.config) if args.config else ED2LearnedConfig()
    result = run_ed2_learned(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    v = result.verdict
    hq = v["horizon_at_quarter"]
    print(f"  B_full=${v['b_full']:.0f}  B/4=${v['b_quarter']:.0f}  ceiling H={v['ceiling']:.1f}")
    print("  horizon@B/4: " + "  ".join(f"{p}={hq[p]:.1f}" for p in hq))
    verdict = "tiering WINS" if v["h17_tiering_wins"] else "bit_exact wins"
    print(f"  H17 (real model) → {verdict} (winner={v['h17_winner']}), "
          f"H18 ratio@B/4={v['competitive_ratio']:.2f}")
    try:
        from figures.plot_ed2_learned import plot_ed2_learned

        plot_ed2_learned(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
