"""ED6 two-oracle (learned arm) -- the consistency oracle on the **real** `M_θ` (H12, §10.1, DS8).

ED6's two-oracle slice (:mod:`.ed6_two_oracle`) runs H12 on the *synthetic* tunable-noise proposer
with two dialled error modes (``gross`` / ``subtle``): is the cheap **consistency oracle** (the §9.1
split-brain decision -- per object, converged or split, and to what value?) a *redundant* but
**decision-sufficient and cheaper** second oracle against the full **bit-exact** one? This module
closes the DS8-deferred *learned-`M_θ` re-pointing* of that slice (what ED1-learned is to ED1,
ED2-learned is to ED2): it trains the flat DS4 `M_θ`
(:class:`~verisim.distmodel.NeuralDistWorldModel`, exactly as :mod:`.ed2_learned` does) and runs the
**same** teacher-forced H12 measurement on it, so the two-oracle question is answered on a *real*
error distribution rather than a dialled one.

The non-obvious finding the real model makes -- and the reason this arm earns its own figure -- is
the **honest mirror of ED2-learned, read through the other oracle**. ED2-learned showed the cheap
*refutation* tiers (metamorphic/symbolic) are useless on a real model because its constrained
decoder removes the ``gross`` (out-of-vocab) class, leaving only ``subtle`` (invariant-respecting,
in-flight) errors no cheap tier can *catch*. The consistency oracle does not try to catch errors --
it answers a *different* question (the split-brain verdict), and a ``subtle`` in-flight error is
precisely the class that is bit-visible but **consistency-invisible** (ED5/H19). So the very errors
that defeat the cheap refutation tiers are the ones the cheap *decision* oracle is still right on:
on the real model the consistency oracle's **decision-sufficiency lands at ~0.57 -- a majority of
the bit-wrong steps, between the synthetic ``gross`` (0.0) and ``subtle`` (1.0) poles** -- because a
real error distribution is a *mixture*, predominantly but not purely the consistency-invisible
in-flight class. So it can be trusted on the question an SRE/defender actually asks (is there a
split-brain, and where?) more often than not even as the *full* prediction is wrong 87% of the time,
at a ~3.6× cheaper consult. Same model, same constrained decoder; the cheap oracle loses as a
*verifier* (ED2-learned) and is decision-sufficient on the majority of errors as a *decision oracle*
(here) -- the tiered-oracle thesis made concrete on the model that actually exists. The honest
caveat: sufficiency is partial (0.57, not the synthetic subtle pole's 1.0) because the real model
still makes some ``gross`` durable-replica errors -- reported, not hidden.

One figure from one trained model: the H12 rates (non-redundant ≈ 0, consistency-sufficient, full-
wrong) plus the consult-fact cheapness ratio, with bootstrap CIs over held-out fault-heavy seeds.

CI runs a tiny smoke instance; the committed figure comes from the local
``configs/ed6_two_oracle_learned.json`` run (the ``figures/reproduce.sh`` discipline). Torch-backed
(the ``[model]`` extra), like every learned arm.
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
from verisim.distmetrics import (
    consistency_faithfulness,
    dist_facts,
    divergence,
)
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.experiments.ed6_two_oracle import consistency_consult_facts
from verisim.metrics.aggregate import bootstrap_ci

#: the four H12 rates, aggregated over seeds (mirrors :mod:`.ed6_two_oracle`).
_RATE_KEYS = (
    "non_redundant_rate",
    "consistency_sufficient_rate",
    "full_wrong_rate",
    "consult_fact_ratio",
)


@dataclass(frozen=True)
class ED6TwoOracleLearnedConfig:
    name: str = "ed6-two-oracle-learned"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    # training (mirrors ED2LearnedConfig / ED6Config -- the same flat DS4 M_θ)
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
    # evaluation: held out, fault-heavy (the in-flight medium is where subtle errors live -- §9.1)
    eval_driver: str = "adversarial"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105, 106, 107)
    n_steps: int = 40

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED6TwoOracleLearnedConfig:
        b = ED6TwoOracleLearnedConfig()
        return ED6TwoOracleLearnedConfig(
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
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED6TwoOracleLearnedConfig:
        return ED6TwoOracleLearnedConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED6TwoOracleLearnedResult:
    """The single real-model H12 cell (the rates + CIs) and the one-line verdict."""

    cell: dict[str, Any] = field(default_factory=dict)
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


def train_model(config: ED6TwoOracleLearnedConfig, oracle: DistOracle) -> Any:
    """Train the flat distributed `M_θ` -- process-reproducibly, as ED2-learned / EN1 / EH1 do."""
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


def _score_seed(
    model: Any, oracle: DistOracle, config: ED6TwoOracleLearnedConfig, seed: int
) -> dict[str, float]:
    """Teacher-forced over one held-out trajectory: the three H12 counts + the cost ratio.

    Identical bookkeeping to :func:`verisim.experiments.ed6_two_oracle._score_seed`, but the
    proposer is the real learned `M_θ` not the synthetic ``DistNoisyModel`` (no error-mode dial).
    """
    actions = _eval_actions(oracle, config.dist, config.eval_driver, seed, config.n_steps)
    state = DistributedState.initial(config.dist)
    total = 0
    non_redundant = 0  # full exact but consistency wrong -- 0 by construction
    full_wrong = 0
    consistency_sufficient = 0  # full wrong but consistency right -- the decision payoff
    ratios: list[float] = []
    for action in actions:
        pred_delta = model.predict_delta(state, action)
        truth = oracle.step(state, action).state
        pred = apply(state, pred_delta)
        full_div = divergence(pred, truth)
        consistency_faithful = consistency_faithfulness(truth, pred) == 1.0
        total += 1
        if full_div == 0.0 and not consistency_faithful:
            non_redundant += 1
        if full_div > 0.0:
            full_wrong += 1
            if consistency_faithful:
                consistency_sufficient += 1
        full_facts = len(dist_facts(truth))
        ratios.append(
            consistency_consult_facts(truth, config.dist) / full_facts if full_facts else 0.0
        )
        state = truth  # teacher-forced
    return {
        "non_redundant_rate": non_redundant / total if total else 0.0,
        # P(consistency-right | full-wrong); vacuously 1.0 if the model was bit-exact all seed.
        "consistency_sufficient_rate": (
            consistency_sufficient / full_wrong if full_wrong else 1.0
        ),
        "full_wrong_rate": full_wrong / total if total else 0.0,
        "consult_fact_ratio": fmean(ratios) if ratios else 0.0,
    }


def run_ed6_two_oracle_learned(
    config: ED6TwoOracleLearnedConfig | None = None, *, oracle: DistOracle | None = None
) -> ED6TwoOracleLearnedResult:
    """Train the flat `M_θ`, score the consistency-vs-bit-exact H12 metrics teacher-forced."""
    config = config or ED6TwoOracleLearnedConfig()
    oracle = oracle or ReferenceDistOracle(config.dist)
    model = train_model(config, oracle)

    per_seed = [_score_seed(model, oracle, config, s) for s in config.eval_seeds]
    cell: dict[str, Any] = {}
    for key in _RATE_KEYS:
        vals = [s[key] for s in per_seed]
        lo, hi = bootstrap_ci(vals, seed=0)
        cell[key] = fmean(vals)
        cell[f"{key}_lo"] = lo
        cell[f"{key}_hi"] = hi
    cell["redundant_for_verification"] = cell["non_redundant_rate"] == 0.0

    ratio = cell["consult_fact_ratio"]
    result = ED6TwoOracleLearnedResult(cell=cell)
    result.verdict = {
        # H12 on the real model: redundant for verification, decision-sufficient, cheaper.
        "redundant_for_verification": cell["redundant_for_verification"],
        "consistency_sufficient_rate": cell["consistency_sufficient_rate"],
        "full_wrong_rate": cell["full_wrong_rate"],
        "consult_fact_ratio": ratio,
        "cheaper_factor": (1.0 / ratio) if ratio else float("inf"),
        # the decision oracle is *worth it* when the model errs often enough to matter yet stays
        # consistency-sufficient at the cheaper consult -- the real-model two-oracle payoff.
        "decision_sufficient": cell["consistency_sufficient_rate"] >= 0.5 and ratio < 1.0,
    }
    return result


CSV_HEADER = (
    "non_redundant_rate,consistency_sufficient_rate,full_wrong_rate,consult_fact_ratio,"
    "cheaper_factor,decision_sufficient"
)


def write_csv(result: ED6TwoOracleLearnedResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    c, v = result.cell, result.verdict
    row = (
        f"{c['non_redundant_rate']:.4f},{c['consistency_sufficient_rate']:.4f},"
        f"{c['full_wrong_rate']:.4f},{c['consult_fact_ratio']:.4f},"
        f"{v['cheaper_factor']:.4f},{v['decision_sufficient']}"
    )
    out.write_text("\n".join([CSV_HEADER, row]) + "\n")
    return out


def _print_summary(result: ED6TwoOracleLearnedResult) -> None:
    c, v = result.cell, result.verdict
    print("ED6 two-oracle (learned M_θ, consistency vs bit-exact, H12) — teacher-forced, CIs:")
    print(f"  non-redundant      = {c['non_redundant_rate']:.3f}  "
          f"(redundant for verification: {v['redundant_for_verification']})")
    print(f"  consistency-suff.  = {c['consistency_sufficient_rate']:.3f} "
          f"[{c['consistency_sufficient_rate_lo']:.2f}, {c['consistency_sufficient_rate_hi']:.2f}]"
          "  (the decision payoff)")
    print(f"  full-state wrong   = {c['full_wrong_rate']:.3f} "
          f"[{c['full_wrong_rate_lo']:.2f}, {c['full_wrong_rate_hi']:.2f}]")
    print(f"  consult-fact ratio = {c['consult_fact_ratio']:.3f}  "
          f"(~{v['cheaper_factor']:.1f}× cheaper)")
    verdict = (
        "decision-sufficient AND cheaper" if v["decision_sufficient"] else "not decision-sufficient"
    )
    print(f"  H12 (real model) → consistency oracle is {verdict}")


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED6 two-oracle learned (H12 consistency vs bit-exact on the real M_θ)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed6_two_oracle_learned.csv")
    parser.add_argument("--plot", type=str, default="figures/ed6_two_oracle_learned.png")
    args = parser.parse_args()
    config = (
        ED6TwoOracleLearnedConfig.from_json_file(args.config)
        if args.config
        else ED6TwoOracleLearnedConfig()
    )
    result = run_ed6_two_oracle_learned(config)
    _print_summary(result)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    try:
        from figures.plot_ed6_two_oracle_learned import plot_ed6_two_oracle_learned

        plot_ed6_two_oracle_learned(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
