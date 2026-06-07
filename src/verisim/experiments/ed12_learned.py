"""ED12 (partial-observation, learned arm) — the probe/consistency projections on the real `M_θ`.

ED12 (:mod:`.ed12`) measured, on the *synthetic* tunable-noise proposer, that the **observable**-
faithful horizon outlasts the **bit**-faithful one for in-flight errors (no probe can read the
replication medium), and proved the dominance ``H_ε^bit ≤ H_ε^observable``. This module is the
learned re-pointing — what ED1-learned is to ED1: it trains the flat DS4 `M_θ` (exactly as
:mod:`.ed4_consistency_learned`) and measures the three projections on its *real* error
distribution, so the question becomes **how a real model's errors distribute across the bit /
observable / consistency views**. Two panels, torch-backed (the ``[model]`` extra):

  - **Panel A — free-running horizons (the direct re-pointing of ED12).** Roll the trained model
    free (ρ=0, which exposes the model, not the loop) and read all three horizons off the *same*
    rollout at the maximal whole-cluster vantage: ``bit``, ``observable`` (replicas + reachability +
    clock, never the in-flight medium), ``consistency`` (the §9.1 per-object view). The structural
    dominance ``bit ≤ observable`` holds on every rollout by construction (a bit-faithful step is
    observably faithful); the consistency horizon is the longest. *Honest caveat, inherited from
    ED1-learned / ED4-consistency-learned:* the flat free-runner's absolute horizons are small, so
    the gaps are directional, not always disjoint — the clean separation is what the teacher-forced
    panel gives, free of the derailing the free-running horizon conflates in.

  - **Panel B — teacher-forced per-step accuracy (the clean headline).** At each step the model
    predicts the delta *from the true current state* (teacher-forced, so each step is independent
    and the measurement is not limited by the model derailing), and we score whether the predicted
    next state is correct under each projection: ``bit`` (every fact right), ``observable`` (right
    at the probe — the in-flight medium forgiven), ``consistency`` (right per-object view). A
    bit-correct step is correct under both other views (so ``bit`` lower-bounds them); the rates
    order ``bit ≤ observable ≤ consistency`` empirically and quantify **which of the model's errors
    each projection forgives** — the analogue of ED6-two-oracle's teacher-forced sufficiency.

CI runs a tiny smoke instance; the committed figure comes from the local config run.
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
from verisim.distmetrics.divergence import consistency_faithfulness, divergence
from verisim.distmetrics.observe import observable_divergence
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon


@dataclass(frozen=True)
class ED12LearnedConfig:
    name: str = "ed12-learned"
    dist: DistConfig = DEFAULT_DIST_CONFIG  # eventual: the in-flight medium projections split on
    # training (mirrors ED4ConsistencyLearnedConfig — the same flat DS4 M_θ)
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
    # evaluation
    eval_driver: str = "contention"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105, 106, 107)
    n_steps: int = 40
    epsilon: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED12LearnedConfig:
        b = ED12LearnedConfig()
        return ED12LearnedConfig(
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
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED12LearnedConfig:
        return ED12LearnedConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED12LearnedResult:
    #: Panel A: mean free-running bit / observable / consistency horizon (CIs) + dominance flag.
    horizons: dict[str, Any] = field(default_factory=dict)
    #: Panel B: teacher-forced per-step correct rate under each projection.
    accuracy: dict[str, Any] = field(default_factory=dict)


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


def train_model(config: ED12LearnedConfig) -> tuple[Any, DistOracle]:
    """Train the flat DS4 `M_θ` on the eventual world (process-reproducibly, same recipe as ED2)."""
    import torch

    from verisim.distmodel import DistVocab, NeuralDistWorldModel, build_dist_dataset
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_supervised

    dist = config.dist
    oracle: DistOracle = ReferenceDistOracle(dist)
    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)

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
    return NeuralDistWorldModel(model, vocab), oracle


def _three_horizons(
    model: Any, oracle: DistOracle, config: ED12LearnedConfig, eval_seed: int
) -> tuple[int, int, int, bool]:
    """One free-running (ρ=0) rollout: ``(bit_h, observable_h, consistency_h, dominates)``."""
    vantage = frozenset(config.dist.nodes)
    s_truth = DistributedState.initial(config.dist)
    s_pred = DistributedState.initial(config.dist)
    actions = _eval_actions(oracle, config.dist, config.eval_driver, eval_seed, config.n_steps)
    bit_div: list[float] = []
    obs_div: list[float] = []
    cons_div: list[float] = []
    dominates = True
    for action in actions:
        s_truth = oracle.step(s_truth, action).state
        s_pred = apply(s_pred, model.predict_delta(s_pred, action))
        b = divergence(s_pred, s_truth)
        o = observable_divergence(s_truth, s_pred, vantage)
        c = 1.0 - consistency_faithfulness(s_truth, s_pred)
        bit_div.append(b)
        obs_div.append(o)
        cons_div.append(c)
        if b <= config.epsilon and o > config.epsilon:  # bit-faithful must imply observ. faithful
            dominates = False
    return (
        faithful_horizon(bit_div, config.epsilon),
        faithful_horizon(obs_div, config.epsilon),
        faithful_horizon(cons_div, config.epsilon),
        dominates,
    )


def _teacher_forced_rates(
    model: Any, oracle: DistOracle, config: ED12LearnedConfig
) -> dict[str, float]:
    """Per-step (teacher-forced) correct rate under bit / observable / consistency projections."""
    vantage = frozenset(config.dist.nodes)
    bit_ok = obs_ok = cons_ok = total = 0
    for eval_seed in config.eval_seeds:
        s = DistributedState.initial(config.dist)
        actions = _eval_actions(oracle, config.dist, config.eval_driver, eval_seed, config.n_steps)
        for action in actions:
            truth = oracle.step(s, action).state
            pred = apply(s, model.predict_delta(s, action))  # predict from the TRUE current state
            total += 1
            bit_ok += divergence(pred, truth) == 0.0
            obs_ok += observable_divergence(truth, pred, vantage) == 0.0
            cons_ok += consistency_faithfulness(truth, pred) == 1.0
            s = truth  # teacher-forcing: advance along ground truth
    n = total or 1
    return {"bit": bit_ok / n, "observable": obs_ok / n, "consistency": cons_ok / n, "steps": total}


def run_ed12_learned(config: ED12LearnedConfig | None = None) -> ED12LearnedResult:
    config = config or ED12LearnedConfig()
    model, oracle = train_model(config)
    result = ED12LearnedResult()

    # --- Panel A: free-running horizons ----------------------------------------------------------
    cells = [_three_horizons(model, oracle, config, s) for s in config.eval_seeds]
    bit = [float(b) for b, _, _, _ in cells]
    obs = [float(o) for _, o, _, _ in cells]
    cons = [float(c) for _, _, c, _ in cells]
    bit_lo, bit_hi = bootstrap_ci(bit, seed=0)
    obs_lo, obs_hi = bootstrap_ci(obs, seed=0)
    cons_lo, cons_hi = bootstrap_ci(cons, seed=0)
    result.horizons = {
        "bit_h": fmean(bit), "bit_lo": bit_lo, "bit_hi": bit_hi,
        "obs_h": fmean(obs), "obs_lo": obs_lo, "obs_hi": obs_hi,
        "cons_h": fmean(cons), "cons_lo": cons_lo, "cons_hi": cons_hi,
        "observable_dominates_bit": all(ok for *_, ok in cells),
    }

    # --- Panel B: teacher-forced per-step accuracy -----------------------------------------------
    result.accuracy = _teacher_forced_rates(model, oracle, config)
    return result


CSV_HEADER = "panel,metric,value,lo,hi"


def write_csv(result: ED12LearnedResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = result.horizons
    lines = [
        CSV_HEADER,
        f"horizon,bit_h,{h['bit_h']:.4f},{h['bit_lo']:.4f},{h['bit_hi']:.4f}",
        f"horizon,obs_h,{h['obs_h']:.4f},{h['obs_lo']:.4f},{h['obs_hi']:.4f}",
        f"horizon,cons_h,{h['cons_h']:.4f},{h['cons_lo']:.4f},{h['cons_hi']:.4f}",
    ]
    a = result.accuracy
    for key in ("bit", "observable", "consistency"):
        lines.append(f"accuracy,{key},{a[key]:.4f},,")
    out.write_text("\n".join(lines) + "\n")
    return out


def _print_summary(result: ED12LearnedResult) -> None:
    h = result.horizons
    print("ED12 (learned M_θ) — partial-observation projections on the real model:")
    print(f"  Panel A free-running horizons: bit H={h['bit_h']:.2f} "
          f"[{h['bit_lo']:.2f},{h['bit_hi']:.2f}] obs H={h['obs_h']:.2f} cons H={h['cons_h']:.2f} "
          f"(bit≤obs holds on every rollout: {h['observable_dominates_bit']})")
    a = result.accuracy
    print(f"  Panel B teacher-forced per-step correct rate (n={a['steps']}): "
          f"bit={a['bit']:.2f} ≤ obs={a['observable']:.2f} ≤ cons={a['consistency']:.2f}")


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED12 learned arm (partial-observation projections on the real M_θ)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed12_learned.csv")
    parser.add_argument("--plot", type=str, default="figures/ed12_learned.png")
    args = parser.parse_args()
    config = ED12LearnedConfig.from_json_file(args.config) if args.config else ED12LearnedConfig()
    result = run_ed12_learned(config)
    _print_summary(result)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    try:
        from figures.plot_ed12_learned import plot_ed12_learned

        plot_ed12_learned(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
