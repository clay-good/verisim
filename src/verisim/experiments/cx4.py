"""Experiment CX4: exact-oracle vs learned-model counterfactual augmentation — CoDA contrast (H63).

Operationalizes the program's foundational claim (SPEC §1.1): a *learned* world model's predictions
are unverifiable and drift, so data it synthesizes inherits that drift, while the *exact* oracle's
counterfactuals are causally valid by construction. CoDA (Pitis, Creager, Garg, NeurIPS 2020) is the
SOTA for *learned* counterfactual data augmentation; CX4 contrasts it against verisim's oracle
augmentation — the same idea, one grounded by a model that can be wrong, the other by an oracle that
cannot.

On the distributed world (where counterfactual data is sharp — the near-miss partitions and
crashes), CX4 builds **one counterfactual query set** (alternative fault actions `a'` at visited
on-policy states) and labels it two ways:

  - **+oracle-aug** — the exact oracle delta ``O(s, a')`` (causally valid by construction; the
    causal-validity rate is 1.0);
  - **+learned-aug** — a *learned local model* ``M_local`` (a small `M_θ` trained on the on-policy
    trajectory, the CoDA stand-in) predicting ``M_local(s, a')`` — the same prompts, labels that
    inherit ``M_local``'s drift (its causal-validity rate is whatever it is).

Both augment the same on-policy base to the **same sample count**, differing only in the *label
source*; a `base` arm (on-policy volume control) is the no-augmentation reference. All three train
an identical downstream `M_θ` for the same budget. The reported metrics are the held-out
**intervention-exact** and **medium-recall** per arm (the augmentation payoff) and the
**causal-validity rate** of each augmenter's samples — the mechanism. H63: ``+oracle-aug`` beats
``+learned-aug`` because the learned augmenter injects causally-invalid samples (validity < 1.0)
that corrupt training. *Refuted if* they tie at equal count — the dynamics are easy enough that the
learned local model is already causally valid, so the oracle buys nothing here (a result on *which*
domains need the exact oracle). Both banked.

CPU, deterministic, torch-gated like every learned-arm experiment; multi-seed with bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import DistAction
from verisim.dist.state import DistributedState
from verisim.distdata.drivers import DistDriver
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.experiments.ed6 import ED6Config, _intervention_cases, _intervention_driver, _score
from verisim.metrics.aggregate import bootstrap_ci

ARMS = ("base", "+oracle-aug", "+learned-aug")
METRICS = ("intervention_exact", "medium_recall")


@dataclass(frozen=True)
class CX4Config:
    """The CoDA contrast over the ED6 apparatus (SPEC-17 §6). Scale via the JSON config."""

    ed6: ED6Config = field(default_factory=ED6Config)
    k_augment: int = 4  # counterfactual augmentation samples per visited on-policy state
    model_seeds: tuple[int, ...] = (0, 1, 2)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CX4Config:
        b = CX4Config()
        return CX4Config(
            ed6=ED6Config.from_dict(d.get("ed6", {})),
            k_augment=d.get("k_augment", b.k_augment),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CX4Config:
        return CX4Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ArmStat:
    """One (arm, metric) cell: mean + bootstrap CI over (model_seed × eval_seed) cells."""

    arm: str
    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.arm},{self.metric},{self.mean:.4f},{self.ci_lo:.4f},{self.ci_hi:.4f},{self.n}"
        )


CSV_HEADER = "arm,metric,mean,ci_lo,ci_hi,n"


def _on_policy(
    config: ED6Config, oracle: DistOracle, vocab: Any, seeds: tuple[int, ...]
) -> list[Any]:
    """Encoded on-policy ``(state, action) -> oracle delta`` examples over seeded trajectories."""
    from verisim.distmodel import encode_prompt, encode_target

    out: list[Any] = []
    for seed in seeds:
        drv = DistDriver(name=config.train_driver, config=config.dist, rng=random.Random(seed))
        state = DistributedState.initial(config.dist)
        for _ in range(config.train_steps_per_traj):
            action = drv.sample(state)
            result = oracle.step(state, action)
            out.append((encode_prompt(state, action, vocab), encode_target(result.delta, vocab)))
            state = result.state
    return out


def _cf_queries(config: CX4Config, oracle: DistOracle) -> list[tuple[DistributedState, DistAction]]:
    """The shared counterfactual query set: ``k`` alternative fault actions per visited state."""
    cfg = config.ed6
    queries: list[tuple[DistributedState, DistAction]] = []
    for seed in cfg.train_seeds:
        traj = DistDriver(name=cfg.train_driver, config=cfg.dist, rng=random.Random(seed))
        cf = _intervention_driver(cfg, seed ^ 0xC0DA)
        state = DistributedState.initial(cfg.dist)
        for _ in range(cfg.train_steps_per_traj):
            for _k in range(config.k_augment):
                queries.append((state, cf.sample(state)))
            state = oracle.step(state, traj.sample(state)).state
    return queries


def run_cx4(config: CX4Config | None = None, *, oracle: DistOracle | None = None) -> list[ArmStat]:
    """Contrast oracle vs learned-model counterfactual augmentation; payoff + validity (H63)."""
    import torch

    from verisim.distmodel import DistVocab, NeuralDistWorldModel, encode_prompt, encode_target
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_batched

    config = config or CX4Config()
    cfg = config.ed6
    oracle = oracle or ReferenceDistOracle(cfg.dist)
    torch.set_num_threads(1)
    vocab = DistVocab(cfg.dist, max_int=cfg.max_int)

    base = _on_policy(cfg, oracle, vocab, cfg.train_seeds)
    queries = _cf_queries(config, oracle)
    n_aug = len(queries)
    # oracle augmentation: the exact delta at each query (validity 1.0 by construction)
    oracle_aug = [
        (encode_prompt(s, a, vocab), encode_target(oracle.step(s, a).delta, vocab))
        for s, a in queries
    ]
    oracle_targets = [oracle.step(s, a).delta for s, a in queries]
    # base volume control: extra on-policy examples to match the augmented arms' count.
    extra_seeds = tuple(range(1000, 1000 + -(-n_aug // cfg.train_steps_per_traj)))
    base_pad = _on_policy(cfg, oracle, vocab, extra_seeds)[:n_aug]

    cases_by_seed = {s: _intervention_cases(oracle, cfg, s) for s in cfg.eval_seeds}

    def make_model(block_size: int) -> Any:
        return GPT(GPTConfig(vocab_size=len(vocab), block_size=block_size,
                             n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd))

    def train(examples: list[Any], block_size: int, seed: int) -> Any:
        torch.manual_seed(seed)
        model = make_model(block_size)
        train_batched(model, examples, vocab.pad, steps=cfg.train_iters, lr=cfg.lr,
                      batch_size=cfg.batch_size, seed=seed)
        return NeuralDistWorldModel(model, vocab)

    per_arm: dict[str, dict[str, list[float]]] = {a: {m: [] for m in METRICS} for a in ARMS}
    validity: list[float] = []
    for model_seed in config.model_seeds:
        # the CoDA stand-in: a learned local model trained on the on-policy base (it has never seen
        # the off-policy fault region the counterfactual queries probe — the §1.1 unverifiability).
        block_size = max(cfg.block_size, max(len(p) + len(t) for p, t in base + oracle_aug) + 8)
        m_local = train(base, block_size, model_seed)
        learned_targets = [m_local.predict_delta(s, a) for s, a in queries]
        n_valid = sum(1 for lt, ot in zip(learned_targets, oracle_targets, strict=True) if lt == ot)
        validity.append(n_valid / n_aug)
        learned_aug = [
            (encode_prompt(s, a, vocab), encode_target(lt, vocab))
            for (s, a), lt in zip(queries, learned_targets, strict=True)
        ]
        datasets = {
            "base": base + base_pad,
            "+oracle-aug": base + oracle_aug,
            "+learned-aug": base + learned_aug,
        }
        for arm in ARMS:
            wm = train(datasets[arm], block_size, model_seed)
            for seed in cfg.eval_seeds:
                scored = _score(wm, cases_by_seed[seed])
                for metric in METRICS:
                    per_arm[arm][metric].append(scored[metric])

    stats: list[ArmStat] = []
    for arm in ARMS:
        for metric in METRICS:
            vals = per_arm[arm][metric]
            lo, hi = bootstrap_ci(vals, seed=0)
            stats.append(ArmStat(arm, metric, fmean(vals), lo, hi, len(vals)))
    # the mechanism: the learned augmenter's causal-validity rate (oracle's is 1.0 by construction).
    vlo, vhi = bootstrap_ci(validity, seed=0)
    stats.append(
        ArmStat("+learned-aug", "causal_validity", fmean(validity), vlo, vhi, len(validity))
    )
    stats.append(ArmStat("+oracle-aug", "causal_validity", 1.0, 1.0, 1.0, len(validity)))
    return stats


def h63_supported(stats: list[ArmStat], metric: str = "intervention_exact") -> bool:
    """H63 holds iff +oracle-aug beats +learned-aug with **disjoint** CIs on ``metric``."""
    by = {s.arm: s for s in stats if s.metric == metric}
    o, le = by.get("+oracle-aug"), by.get("+learned-aug")
    return o is not None and le is not None and o.ci_lo > le.ci_hi


def _verdict(stats: list[ArmStat]) -> str:
    metric = "intervention_exact"
    by = {s.arm: s for s in stats if s.metric == metric}
    o, le = by.get("+oracle-aug"), by.get("+learned-aug")
    val = next(
        (s for s in stats if s.arm == "+learned-aug" and s.metric == "causal_validity"), None
    )
    if o is None or le is None or val is None:
        return "inconclusive (missing arms)"
    gap = o.mean - le.mean
    vr = val.mean
    if h63_supported(stats, metric):
        return (
            f"H63 SUPPORTED — exact-oracle augmentation beats learned-model (CoDA) augmentation: "
            f"at equal augmented-sample count, +oracle-aug > +learned-aug on intervention-exact "
            f"({o.mean:.3f} vs {le.mean:.3f}, +{gap:.3f}, disjoint CIs). The mechanism is causal "
            f"validity: the learned model's counterfactual samples are only {vr:.2f} valid (the "
            f"oracle's are 1.00 by construction), so its augmentation injects invalid data "
            f"that corrupts training — the SPEC §1.1 unverifiability, made a measured cost."
        )
    tail = (
        f"the learned model's counterfactual validity is high ({vr:.2f}), so its augmentation is "
        f"about as good as the oracle's — the distributed dynamics are locally learnable and the "
        f"exact oracle buys little augmentation value here"
        if vr >= 0.5
        else (
            f"yet the learned model's counterfactual validity is low ({vr:.2f}): its augmentation "
            f"injects invalid data but the two arms do not separate at this scale — the oracle's "
            f"validity edge is real (1.00 vs {vr:.2f}) but the downstream H_ε gap is within seed "
            f"noise (a power/scale caveat, the GPU regime the open bet)"
        )
    )
    return (
        f"H63 NOT supported here — +oracle-aug does not beat +learned-aug beyond seed noise "
        f"(+oracle-aug {o.mean:.3f} vs +learned-aug {le.mean:.3f}, {gap:+.3f}, overlapping CIs); "
        f"{tail} (a result on where the exact oracle's augmentation matters)."
    )


def _print_summary(stats: list[ArmStat]) -> None:
    print("CX4 / H63 — exact-oracle vs learned-model counterfactual augmentation (CoDA contrast):")
    print(f"  {'arm':<14} {'metric':<18} {'mean':>7} {'95% CI':>16} {'N':>5}")
    for s in stats:
        ci = f"[{s.ci_lo:.3f},{s.ci_hi:.3f}]"
        print(f"  {s.arm:<14} {s.metric:<18} {s.mean:>7.3f} {ci:>16} {s.n:>5}")
    print("  " + _verdict(stats))


def write_csv(stats: list[ArmStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    return out


def _plot(stats: list[ArmStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "#7f7f7f", "+oracle-aug": "#2ca02c", "+learned-aug": "#d62728"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    rows = [s for s in stats if s.metric == "intervention_exact"]
    x = range(len(rows))
    ax1.bar(x, [s.mean for s in rows], 0.6,
            yerr=[[s.mean - s.ci_lo for s in rows], [s.ci_hi - s.mean for s in rows]], capsize=4,
            color=[colors.get(s.arm, "#555") for s in rows])
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([s.arm for s in rows], fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("held-out intervention-exact")
    ax1.set_title("augmentation payoff: oracle vs learned (CoDA)")
    vrows = [s for s in stats if s.metric == "causal_validity"]
    vx = range(len(vrows))
    ax2.bar(vx, [s.mean for s in vrows], 0.6,
            color=[colors.get(s.arm, "#555") for s in vrows])
    ax2.set_xticks(list(vx))
    ax2.set_xticklabels([s.arm for s in vrows], fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("causal-validity rate of augmenter samples")
    ax2.set_title("the mechanism: oracle 1.0 by construction, learned drifts")
    fig.suptitle("CX4 / H63: exact-oracle vs learned-model counterfactual augmentation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CX4 CoDA contrast (oracle vs learned aug, H63).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cx4_coda_contrast.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CX4Config.from_json_file(args.config) if args.config else CX4Config()
    stats = run_cx4(cfg)
    _print_summary(stats)
    out = write_csv(stats, args.out)
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
