"""ED6 (two-oracle arm) -- the consistency oracle as a cheap second oracle (H12, SPEC-7 §10.1, DS8).

The distributed analogue of the network EN10 / host EH6 two-oracle slice -- the deferred half of ED6
(:mod:`.ed6`). The full **bit-exact oracle** (§5) verifies the predicted cluster state bit-for-bit;
the cheap **consistency oracle** (the §9.1 headline-new metric -- does each object read as
*converged* or *split*, and to what value?) answers only the one operationally-decisive question an
SRE/defender actually asks under partition: *is there a split-brain, and where?* H12 asks the same
three things EN10/EH6 did, teacher-forced over a fault-heavy (``adversarial``) workload:

  - **non-redundant rate** -- steps where the consistency oracle flags an error the *bit-exact*
    prediction got right (full divergence ``0`` but consistency-faithfulness ``< 1``). **0 by
    construction** -- the consistency view is a pure function of the replica state, so a bit-for-bit
    correct prediction is always consistency-correct. Redundant *for verification*.
  - **consistency-sufficient rate** -- of the steps where the bit-exact prediction is *wrong*
    (divergence ``> 0``), the fraction where the model is still **consistency-faithful**
    (consistency-faithfulness ``== 1``). The decision-relevant payoff: the model can be trusted on
    the split-brain question far more often than on the whole cluster's bytes. This is the per-step,
    teacher-forced form of ED5's free-running consistency-vs-bit *horizon* gap (H19) -- and it is
    **mode-dependent in exactly the same way**: a ``subtle`` (in-flight) error is bit-visible but
    consistency-invisible, so consistency-sufficiency is high; a ``gross`` (durable-replica) error
    is consistency-visible at once, so it is ~0 (the control).
  - **consult-fact ratio** -- the consistency answer size (the per-object converged/split view) over
    the full-state fact count (replicas + in-flight + partition/crash/clock): how much cheaper the
    decision consult is, especially **under fault**, where the in-flight medium and the partition
    structure inflate the full state but never enter the consistency view.

The H12 reframing the tiered oracle (SPEC-7 §5) rests on: the consistency oracle is *redundant* as a
verification signal but a **cheaper, decision-sufficient** consult for the question that matters --
the distributed analogue of the host's privilege second-oracle. Run on the dependency-free synthetic
``DistNoisyModel`` (mode-split), so CI runs the real instance, not a smoke; the learned-``M_θ``
re-pointing is the deferred follow-up (what ED1-learned is to ED1). Pure, GPU-free.
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
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import apply
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distloop import DistNoisyModel
from verisim.distmetrics import (
    consistency_faithfulness,
    dist_facts,
    divergence,
    object_consistency_view,
)
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.metrics.aggregate import bootstrap_ci

ERROR_MODES: tuple[str, ...] = ("gross", "subtle")


def consistency_consult_facts(state: DistributedState, config: DistConfig) -> int:
    """The size of the consistency oracle's *answer*: the per-object converged/split view.

    The decision an SRE reads is, per object, the set of distinct ``(version, value)`` across that
    object's replicas (singleton == converged, larger == split). Its size is the consult cost of the
    cheap oracle, the distributed analogue of the host's ``invariant_bits`` (procs × protected
    paths).
    """
    return sum(len(object_consistency_view(state, obj)) for obj in config.objects)


@dataclass(frozen=True)
class ED6TwoOracleConfig:
    name: str = "ed6-two-oracle"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    driver: str = "adversarial"  # fault-heavy: in-flight + partition medium inflate the full state
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105, 106, 107)
    n_steps: int = 40
    noise: float = 0.6  # the synthetic proposer's per-step corruption probability
    #: ``fallback=False`` makes the error class *exact* (the model errs only on its targeted edit,
    #: the §H20 semantics): ``gross`` is then a pure durable-replica error (consistency-visible, the
    #: clean ~0 control), ``subtle`` a pure in-flight error (consistency-invisible, the payoff).
    fallback: bool = False
    modes: tuple[str, ...] = ERROR_MODES

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED6TwoOracleConfig:
        b = ED6TwoOracleConfig()
        return ED6TwoOracleConfig(
            name=d.get("name", b.name),
            driver=d.get("driver", b.driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            noise=d.get("noise", b.noise),
            fallback=d.get("fallback", b.fallback),
            modes=tuple(d.get("modes", b.modes)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED6TwoOracleConfig:
        return ED6TwoOracleConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED6TwoOracleResult:
    """Per-mode H12 cells: the non-redundant / consistency-sufficient rates + the cost ratio."""

    per_mode: list[dict[str, Any]] = field(default_factory=list)


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


def _score_seed(
    oracle: DistOracle, config: ED6TwoOracleConfig, mode: str, seed: int
) -> dict[str, float]:
    """Teacher-forced over one held-out trajectory: the three H12 counts for ``(mode, seed)``."""
    model = DistNoisyModel(
        oracle, noise=config.noise, mode=mode, rng=random.Random(seed + 7), fallback=config.fallback
    )
    actions = _eval_actions(oracle, config.dist, config.driver, seed, config.n_steps)
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
        # a *conditional* rate P(consistency-right | full-wrong); vacuously 1.0 when the model was
        # bit-exact all seed (no wrong step to be insufficient on) -- the EN10/EH6 convention.
        "consistency_sufficient_rate": (
            consistency_sufficient / full_wrong if full_wrong else 1.0
        ),
        "full_wrong_rate": full_wrong / total if total else 0.0,
        "consult_fact_ratio": fmean(ratios) if ratios else 0.0,
    }


_RATE_KEYS = (
    "non_redundant_rate",
    "consistency_sufficient_rate",
    "full_wrong_rate",
    "consult_fact_ratio",
)


def run_ed6_two_oracle(
    config: ED6TwoOracleConfig | None = None, *, oracle: DistOracle | None = None
) -> ED6TwoOracleResult:
    """Score the consistency-vs-bit-exact two-oracle H12 metrics per error mode (teacher-forced)."""
    config = config or ED6TwoOracleConfig()
    oracle = oracle or ReferenceDistOracle(config.dist)
    result = ED6TwoOracleResult()
    for mode in config.modes:
        per_seed = [_score_seed(oracle, config, mode, s) for s in config.eval_seeds]
        cell: dict[str, Any] = {"mode": mode}
        for key in _RATE_KEYS:
            vals = [s[key] for s in per_seed]
            lo, hi = bootstrap_ci(vals, seed=0)
            cell[key] = fmean(vals)
            cell[f"{key}_lo"] = lo
            cell[f"{key}_hi"] = hi
        # H12 verdict for this mode: the cheap oracle is redundant (≈0) yet decision-sufficient
        cell["redundant_for_verification"] = cell["non_redundant_rate"] == 0.0
        result.per_mode.append(cell)
    return result


CSV_HEADER = (
    "mode,non_redundant_rate,consistency_sufficient_rate,full_wrong_rate,consult_fact_ratio"
)


def write_csv(result: ED6TwoOracleResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{c['mode']},{c['non_redundant_rate']:.4f},{c['consistency_sufficient_rate']:.4f},"
        f"{c['full_wrong_rate']:.4f},{c['consult_fact_ratio']:.4f}"
        for c in result.per_mode
    ]
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def _print_summary(result: ED6TwoOracleResult) -> None:
    print("ED6 two-oracle (consistency vs bit-exact, H12) — teacher-forced, bootstrap CIs:")
    for c in result.per_mode:
        print(f"  [{c['mode']:6s}] non-redundant={c['non_redundant_rate']:.3f}  "
              f"consistency-sufficient={c['consistency_sufficient_rate']:.3f} "
              f"[{c['consistency_sufficient_rate_lo']:.2f},{c['consistency_sufficient_rate_hi']:.2f}]"
              f"  full-wrong={c['full_wrong_rate']:.3f}  "
              f"consult-fact-ratio={c['consult_fact_ratio']:.3f}")


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="Run ED6 two-oracle (consistency vs bit-exact, H12)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed6_two_oracle.csv")
    parser.add_argument("--plot", type=str, default="figures/ed6_two_oracle.png")
    args = parser.parse_args()
    config = (
        ED6TwoOracleConfig.from_json_file(args.config) if args.config else ED6TwoOracleConfig()
    )
    result = run_ed6_two_oracle(config)
    _print_summary(result)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    try:
        from figures.plot_ed6_two_oracle import plot_ed6_two_oracle

        plot_ed6_two_oracle(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
