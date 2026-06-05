"""ED2-smart -- the ``π_c`` "smart-when" axis of ED2 (SPEC-7 §8.1, §10 ED2; DS7).

ED2 (:mod:`verisim.experiments.ed2`) and ED2-learned cover the ``π_w`` *which-tier* axis at an equal
oracle-dollar budget. This module closes ED2's deferred **``π_c`` "smart-when" axis** (SPEC-7 §10,
line "Deferred: belief-variance / uncertainty-gated `π_c` scheduling"): at a *fixed* interior
consultation budget ``ρ``, does spending it on the steps the model is **least sure about** earn more
faithful horizon than spreading it evenly? This is the distributed analogue of v0's E2 / the
network's EN2 / the host's EH2 -- **H9** in the §8.1 numbering.

The three §6.1 policies are compared at *equal* ``ρ`` (the runner's spend-down backstop makes every
arm spend exactly ``floor(ρ·T)`` consultations, so the comparison isolates *where* a policy spends,
not *how much*):

  - ``fixed`` -- every-``k``-steps, the dumb baseline;
  - ``uncertainty`` -- consult when the model's *instantaneous* decode entropy exceeds a budget-
    calibrated threshold ``τ``;
  - ``drift`` -- consult when *accumulated* entropy since the last consult exceeds ``τ`` (a cheap,
    oracle-free proxy for compounding drift).

The signal is the flat ``M_θ``'s **mean constrained-decode entropy** (DS4 incr 2), wired into the
loop's ``StepContext`` by the DS5 runner (the network/host runners already do this; the distributed
runner gained it here). The honest, pre-registered expectation -- which the experiment *reports*
rather than assumes -- is the **standing H2/H9 negative carried into the distributed world**: v0 and
EN2 found the flat decode-entropy signal does **not** beat ``fixed`` (SPEC-2 §7.2), because per-step
entropy is a decode-time artifact, not a calibrated belief. SPEC-7 §8.1 conjectures the *calibrated*
RSSM belief variance of the (deferred) structured ``M_θ`` is the signal that could finally flip it
-- exactly as the host's EH2 found the factored arm's belief variance beats fixed ~2.2× where the
flat arm's entropy did not. ED2-smart sets the flat-arm baseline the structured arm must beat:
a smart-``π_c`` *negative* for the flat signal is first-class (the repo norm), and it localizes the
lever to the representation, not the loop.

One panel: faithful horizon per policy across a few interior budgets ``ρ``, with bootstrap-CI bands.

CI runs a tiny smoke instance; the committed figure comes from the local `configs/ed2_smart.json`
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
from verisim.dist.delta import apply
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distloop import FixedTierPolicy, budget_for_rho, run_dist_rollout
from verisim.distloop.model import DistUncertaintyModel
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.experiments.e2 import build_policy
from verisim.metrics.aggregate import bootstrap_ci

#: The §6.1 consultation policies compared at equal ``ρ``: the dumb baseline + the two smart arms.
POLICIES: tuple[str, ...] = ("fixed", "uncertainty", "drift")


@dataclass(frozen=True)
class ED2SmartConfig:
    name: str = "ed2-smart"
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
    # evaluation: the policies are compared at each fixed interior budget ρ
    eval_driver: str = "uniform"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    n_steps: int = 32
    rhos: tuple[float, ...] = (0.25, 0.5, 0.75)
    policies: tuple[str, ...] = POLICIES
    epsilon: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED2SmartConfig:
        b = ED2SmartConfig()
        return ED2SmartConfig(
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
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED2SmartConfig:
        return ED2SmartConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED2SmartResult:
    """The ED2-smart deliverables: faithful horizon per (policy, ρ) + the smart-vs-fixed verdict."""

    #: per (policy, ρ): mean faithful horizon + CI.
    cells: list[dict[str, Any]] = field(default_factory=list)
    #: per ρ: the best smart policy, its horizon, fixed's horizon, and whether smart wins.
    verdict: list[dict[str, Any]] = field(default_factory=list)


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


def unaided_signals(
    model: DistUncertaintyModel, config: DistConfig, actions: list[DistAction]
) -> list[float]:
    """The model's per-step decode entropy along the unaided (ρ=0) rollout -- calibrates ``τ``."""
    state = DistributedState.initial(config)
    signals: list[float] = []
    for action in actions:
        delta, signal = model.predict_delta_with_uncertainty(state, action)
        signals.append(signal)
        state = apply(state, delta)
    return signals


def train_model(config: ED2SmartConfig, oracle: DistOracle) -> Any:
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


def run_ed2_smart(
    config: ED2SmartConfig | None = None, *, oracle: DistOracle | None = None
) -> ED2SmartResult:
    """Train the flat `M_θ`, compare fixed/uncertainty/drift `π_c` at each interior `ρ`."""
    config = config or ED2SmartConfig()
    oracle = oracle or ReferenceDistOracle(config.dist)
    world_model = train_model(config, oracle)
    result = ED2SmartResult()

    eval_actions = {
        s: _eval_actions(oracle, config.dist, config.eval_driver, s, config.n_steps)
        for s in config.eval_seeds
    }
    signals = {s: unaided_signals(world_model, config.dist, eval_actions[s]) for s in eval_actions}

    by_policy_rho: dict[tuple[str, float], float] = {}
    for rho in config.rhos:
        for policy_name in config.policies:
            hs: list[float] = []
            for seed in config.eval_seeds:
                actions = eval_actions[seed]
                policy = build_policy(policy_name, rho, signals[seed])
                record = run_dist_rollout(
                    world_model, oracle, DistributedState.initial(config.dist), actions, policy,
                    epsilon=config.epsilon, config=config.dist,
                    tier_policy=FixedTierPolicy("bit_exact"),
                    budget=budget_for_rho(rho, len(actions)), seed=seed,
                )
                hs.append(float(record.faithful_horizon))
            lo, hi = bootstrap_ci(hs, seed=0)
            mean_h = fmean(hs)
            by_policy_rho[(policy_name, rho)] = mean_h
            result.cells.append({
                "policy": policy_name, "rho": rho, "h_eps": mean_h, "ci_lo": lo, "ci_hi": hi,
            })

    # the verdict per ρ: does the best smart policy beat fixed at the same (equal) budget?
    for rho in config.rhos:
        fixed_h = by_policy_rho[("fixed", rho)]
        smart = {p: by_policy_rho[(p, rho)] for p in config.policies if p != "fixed"}
        best = max(smart, key=lambda p: smart[p]) if smart else "fixed"
        best_h = smart.get(best, fixed_h)
        result.verdict.append({
            "rho": rho,
            "fixed_h": fixed_h,
            "best_smart": best,
            "best_smart_h": best_h,
            "smart_wins": best_h > fixed_h,
            "lift": (best_h / fixed_h) if fixed_h else float("inf"),
        })
    return result


CSV_HEADER = "panel,policy,rho,h_eps,ci_lo,ci_hi,fixed_h,best_smart,best_smart_h,smart_wins,lift"


def write_csv(result: ED2SmartResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in result.cells:
        rows.append(f"cell,{c['policy']},{c['rho']},{c['h_eps']:.4f},"
                    f"{c['ci_lo']:.4f},{c['ci_hi']:.4f},,,,,")
    for v in result.verdict:
        rows.append(f"verdict,{v['best_smart']},{v['rho']},,,,"
                    f"{v['fixed_h']:.4f},{v['best_smart']},{v['best_smart_h']:.4f},"
                    f"{int(v['smart_wins'])},{v['lift']:.4f}")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED2-smart (π_c smart-when policy comparison on the learned M_θ; H9)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed2_smart.csv")
    parser.add_argument("--plot", type=str, default="figures/ed2_smart.png")
    args = parser.parse_args()
    config = ED2SmartConfig.from_json_file(args.config) if args.config else ED2SmartConfig()
    result = run_ed2_smart(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    for v in result.verdict:
        verdict = "smart WINS" if v["smart_wins"] else "fixed wins"
        print(f"  ρ={v['rho']:.2f}  fixed H={v['fixed_h']:.1f}  "
              f"best-smart({v['best_smart']}) H={v['best_smart_h']:.1f}  "
              f"→ {verdict} (lift {v['lift']:.2f}×)")
    try:
        from figures.plot_ed2_smart import plot_ed2_smart

        plot_ed2_smart(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
