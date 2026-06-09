"""Experiment CX3: the matched-coverage cut — branching vs coverage (SPEC-17 §6, H62).

The confound-resolver, and the program's one genuinely-open question. ED6 (SPEC-7) found a ~2×
counterfactual lift on the distributed world: a flat `M_θ` trained on free oracle counterfactual
fault branches predicts held-out interventions better than the same model trained on matched-*count*
trajectory data. But ED6 flagged its own caveat (SPEC-7 §10.1): the counterfactual branches are
*fault-heavier* than the on-policy control, so they change the distributed **medium** (the
partition/crash hidden state) far more often — the lift conflates counterfactual *branching* with
the fault *coverage* it carries. CX3 cuts the two apart.

The cut: build a **factual** control and the **+counterfactual** arm matched on *both* example count
*and* fault-coverage — the fraction of training examples whose action changes the medium (the
[`ed6._medium`](./ed6.py) statistic, matched by [`causal/coverage.py`](../causal/coverage.py)). The
factual-matched control is a *fault-heavy on-policy trajectory* (one sequence that drifts deep into
fault-land — high coverage, no branching); the +counterfactual-matched arm is *branches* off the
light-fault on-policy states (the same states, many alternative fault futures — the abduction/
re-grounding structure), subsampled to the same coverage. The two differ in **branching** alone.
Head-to-head on held-out intervention-exact and medium-recall:

  - if +counterfactual-matched **still beats** factual-matched with disjoint CIs → counterfactual
    *branching per se* is the active ingredient (H62 supported);
  - if they **tie** → ED6 was *coverage*, not counterfactual structure; the lift re-attributes to
    H21 (fault coverage at equal volume) and the SPEC-7 caveat resolves against branching.

Both outcomes settle the named caveat cleanly and are banked (SPEC §10.1). The raw on-policy
`trajectory` baseline and the raw (unmatched) `+counterfactual` arm are reported alongside so the
original ED6 lift and the matched cut are visible on one figure.

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

from verisim.causal.coverage import coverage_rate, feasible_match, match_coverage
from verisim.dist.action import DistAction
from verisim.dist.state import DistributedState
from verisim.distdata.drivers import DistDriver
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.experiments.ed6 import (
    ED6Config,
    _intervention_cases,
    _intervention_driver,
    _medium,
    _score,
)
from verisim.metrics.aggregate import bootstrap_ci

# arm 1: raw on-policy baseline · arm 2: raw ED6 +counterfactual · arms 3/4: matched-coverage cut
ARMS = ("trajectory", "+counterfactual", "factual-matched", "+counterfactual-matched")
METRICS = ("intervention_exact", "medium_recall")


@dataclass(frozen=True)
class CX3Config:
    """A matched-coverage cut over the ED6 apparatus (SPEC-17 §6). Scale via the JSON config."""

    ed6: ED6Config = field(default_factory=ED6Config)
    target_coverage: float | None = None  # None = min natural coverage across the matched pools
    model_seeds: tuple[int, ...] = (0, 1, 2)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CX3Config:
        b = CX3Config()
        return CX3Config(
            ed6=ED6Config.from_dict(d.get("ed6", {})),
            target_coverage=d.get("target_coverage", b.target_coverage),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CX3Config:
        return CX3Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ArmStat:
    """One (arm, metric) cell: mean + bootstrap CI over (model_seed × eval_seed) cells."""

    arm: str
    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    coverage: float  # realized medium-change fraction of the arm's training set
    n_examples: int  # the arm's training-set size (matched count)
    n: int  # number of (model_seed × eval_seed) scores aggregated

    def csv_row(self) -> str:
        return (
            f"{self.arm},{self.metric},{self.mean:.4f},{self.ci_lo:.4f},{self.ci_hi:.4f},"
            f"{self.coverage:.4f},{self.n_examples},{self.n}"
        )


CSV_HEADER = "arm,metric,mean,ci_lo,ci_hi,coverage,n_examples,n"


def _build_arms(
    config: CX3Config, oracle: DistOracle, vocab: Any
) -> tuple[dict[str, list[Any]], dict[str, float]]:
    """The four training sets + their realized coverage: trajectory / +counterfactual (raw) and the
    factual-matched / +counterfactual-matched cut (matched on count *and* medium coverage)."""
    from verisim.distmodel import encode_prompt, encode_target

    cfg = config.ed6

    def example(state: DistributedState, action: DistAction) -> Any:
        delta = oracle.step(state, action).delta
        return (encode_prompt(state, action, vocab), encode_target(delta, vocab))

    def flag(state: DistributedState, action: DistAction) -> bool:
        # does this action change the distributed medium? (a partition split / crash / heal)
        return _medium(oracle.step(state, action).state) != _medium(state)

    # --- raw pools (the ED6 composition) ---------------------------------------------------------
    on_policy: list[Any] = []
    on_policy_flags: list[bool] = []
    cf_branches: list[Any] = []
    cf_flags: list[bool] = []
    for seed in cfg.train_seeds:
        traj_drv = DistDriver(name=cfg.train_driver, config=cfg.dist, rng=random.Random(seed))
        cf_drv = _intervention_driver(cfg, seed ^ 0xC0FFEE)
        state = DistributedState.initial(cfg.dist)
        for _ in range(cfg.train_steps_per_traj):
            a_traj = traj_drv.sample(state)
            on_policy.append(example(state, a_traj))
            on_policy_flags.append(flag(state, a_traj))
            for _k in range(cfg.k_counterfactual):
                a_cf = cf_drv.sample(state)  # a free counterfactual fault branch off this state
                cf_branches.append(example(state, a_cf))
                cf_flags.append(flag(state, a_cf))
            state = oracle.step(state, a_traj).state

    # --- fault-heavy *factual* pool (high coverage, NO branching): the matched control ----
    factual: list[Any] = []
    factual_flags: list[bool] = []
    for seed in cfg.train_seeds:
        fault_drv = _intervention_driver(cfg, seed ^ 0xFAC7)  # fault-heavy driver, run factually
        state = DistributedState.initial(cfg.dist)
        for _ in range(cfg.train_steps_per_traj * (1 + cfg.k_counterfactual)):
            a = fault_drv.sample(state)
            factual.append(example(state, a))
            factual_flags.append(flag(state, a))
            state = oracle.step(state, a).state  # advance factually on the drifted trajectory

    # --- the matched cut: factual vs counterfactual at equal count AND equal coverage -------------
    target_cov, target_n = feasible_match(
        [factual_flags, cf_flags], coverage=config.target_coverage
    )
    match_rng = random.Random(0xC3C0)
    factual_matched = match_coverage(
        factual, factual_flags, target_coverage=target_cov, target_count=target_n, rng=match_rng
    )
    cf_matched = match_coverage(
        cf_branches, cf_flags, target_coverage=target_cov, target_count=target_n, rng=match_rng
    )

    # --- the raw ED6 arms, count-matched to target_n (the original lift, for context) -------------
    base_n = len(cfg.train_seeds) * cfg.train_steps_per_traj
    traj_more_seeds = tuple(range(-(-target_n // cfg.train_steps_per_traj)))
    trajectory = []
    traj_flags: list[bool] = []
    for seed in traj_more_seeds:
        drv = DistDriver(name=cfg.train_driver, config=cfg.dist, rng=random.Random(1000 + seed))
        state = DistributedState.initial(cfg.dist)
        for _ in range(cfg.train_steps_per_traj):
            a = drv.sample(state)
            trajectory.append(example(state, a))
            traj_flags.append(flag(state, a))
            state = oracle.step(state, a).state
    trajectory = trajectory[:target_n]
    traj_flags = traj_flags[:target_n]
    raw_cf = (on_policy[:base_n] + cf_branches)[:target_n]
    raw_cf_flags = (on_policy_flags[:base_n] + cf_flags)[:target_n]

    datasets = {
        "trajectory": trajectory,
        "+counterfactual": raw_cf,
        "factual-matched": factual_matched,
        "+counterfactual-matched": cf_matched,
    }
    coverages = {
        "trajectory": coverage_rate(traj_flags),
        "+counterfactual": coverage_rate(raw_cf_flags),
        "factual-matched": target_cov,
        "+counterfactual-matched": target_cov,
    }
    return datasets, coverages


def run_cx3(config: CX3Config | None = None, *, oracle: DistOracle | None = None) -> list[ArmStat]:
    """Train the four arms; score held-out intervention-exact + medium-recall, matched (H62)."""
    import torch

    from verisim.distmodel import DistVocab, NeuralDistWorldModel
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_batched

    config = config or CX3Config()
    cfg = config.ed6
    oracle = oracle or ReferenceDistOracle(cfg.dist)
    torch.set_num_threads(1)

    vocab = DistVocab(cfg.dist, max_int=cfg.max_int)
    datasets, coverages = _build_arms(config, oracle, vocab)
    cases_by_seed = {s: _intervention_cases(oracle, cfg, s) for s in cfg.eval_seeds}
    needed = max(len(p) + len(t) for ds in datasets.values() for p, t in ds) + 8
    block_size = max(cfg.block_size, needed)

    stats: list[ArmStat] = []
    for arm in ARMS:
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for model_seed in config.model_seeds:
            torch.manual_seed(model_seed)
            model = GPT(
                GPTConfig(
                    vocab_size=len(vocab), block_size=block_size,
                    n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                )
            )
            train_batched(
                model, datasets[arm], vocab.pad, steps=cfg.train_iters, lr=cfg.lr,
                batch_size=cfg.batch_size, seed=model_seed,
            )
            world_model = NeuralDistWorldModel(model, vocab)
            for seed in cfg.eval_seeds:
                scored = _score(world_model, cases_by_seed[seed])
                for metric in METRICS:
                    per_metric[metric].append(scored[metric])
        for metric, vals in per_metric.items():
            lo, hi = bootstrap_ci(vals, seed=0)
            stats.append(
                ArmStat(arm, metric, fmean(vals), lo, hi, coverages[arm], len(datasets[arm]),
                        len(vals))
            )
    return stats


def h62_supported(stats: list[ArmStat], metric: str = "intervention_exact") -> bool:
    """H62 holds iff +counterfactual-matched beats factual-matched with **disjoint** CIs on the
    ``metric`` (counterfactual *branching*, net of coverage, is the active ingredient)."""
    by = {s.arm: s for s in stats if s.metric == metric}
    cf = by.get("+counterfactual-matched")
    fac = by.get("factual-matched")
    if cf is None or fac is None:
        return False
    return cf.ci_lo > fac.ci_hi


def _verdict(stats: list[ArmStat]) -> str:
    metric = "intervention_exact"
    by = {s.arm: s for s in stats if s.metric == metric}
    cf = by.get("+counterfactual-matched")
    fac = by.get("factual-matched")
    if cf is None or fac is None:
        return "inconclusive (missing matched arms)"
    gap = cf.mean - fac.mean
    if h62_supported(stats, metric):  # cf disjoint-beats fac
        return (
            f"H62 SUPPORTED — counterfactual *branching* carries the ED6 lift net of coverage: at "
            f"equal count and equal medium coverage ({cf.coverage:.2f}), +counterfactual-matched "
            f"beats factual-matched on intervention-exact ({cf.mean:.3f} vs {fac.mean:.3f}, "
            f"+{gap:.3f}, disjoint CIs). The abduction/re-grounding *structure* per se — not the "
            f"fault coverage it carried — is the active ingredient; SPEC-7 §10.1 resolves FOR it."
        )
    if fac.ci_lo > cf.ci_hi:  # fac disjoint-beats cf — the strongest anti-branching outcome
        return (
            f"H62 REFUTED (and then some) — matched on coverage, the *factual* control beats the "
            f"counterfactual arm: at equal count and equal medium coverage ({cf.coverage:.2f}), "
            f"factual-matched > +counterfactual-matched on intervention-exact ({fac.mean:.3f} vs "
            f"{cf.mean:.3f}, gap {gap:+.3f}, disjoint CIs). ED6's lift was **coverage, not "
            f"counterfactual structure** — re-attributed to H21 (fault coverage at equal volume); "
            f"branching per se not only fails to help, a fault-heavy factual sequence does better. "
            f"SPEC-7 §10.1 resolves decisively AGAINST branching. Pre-registered and banked."
        )
    return (
        f"H62 NOT supported — matched coverage erases the lift: at equal count and equal medium "
        f"coverage ({cf.coverage:.2f}), +counterfactual-matched ties factual-matched on "
        f"intervention-exact ({cf.mean:.3f} vs {fac.mean:.3f}, {gap:+.3f}, overlapping CIs). ED6 "
        f"was *coverage*, not counterfactual structure: the lift re-attributes to H21 (fault "
        f"coverage at equal volume), and SPEC-7 §10.1 resolves AGAINST branching. Both were banked."
    )


def _print_summary(stats: list[ArmStat]) -> None:
    print("CX3 / H62 — the matched-coverage cut (branching vs coverage; bootstrap CIs):")
    print(f"  {'arm':<26} {'metric':<20} {'mean':>7} {'95% CI':>16} {'cov':>6} {'N':>6}")
    for s in stats:
        ci = f"[{s.ci_lo:.3f},{s.ci_hi:.3f}]"
        print(f"  {s.arm:<26} {s.metric:<20} {s.mean:>7.3f} {ci:>16} {s.coverage:>6.2f} "
              f"{s.n_examples:>6}")
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

    colors = {
        "trajectory": "#7f7f7f", "+counterfactual": "#1f77b4",
        "factual-matched": "#ff7f0e", "+counterfactual-matched": "#2ca02c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, metric in zip(axes, METRICS, strict=True):
        rows = [s for s in stats if s.metric == metric]
        x = range(len(rows))
        means = [s.mean for s in rows]
        lo = [s.mean - s.ci_lo for s in rows]
        hi = [s.ci_hi - s.mean for s in rows]
        ax.bar(x, means, 0.6, yerr=[lo, hi], capsize=4,
               color=[colors.get(s.arm, "#555") for s in rows])
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{s.arm}\n(cov={s.coverage:.2f})" for s in rows], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"held-out {metric.replace('_', ' ')}")
    fig.suptitle("CX3 / H62: the matched-coverage cut — counterfactual branching vs fault coverage")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CX3 matched-coverage cut (H62).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cx3_matched_coverage.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CX3Config.from_json_file(args.config) if args.config else CX3Config()
    stats = run_cx3(cfg)
    _print_summary(stats)
    out = write_csv(stats, args.out)
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
