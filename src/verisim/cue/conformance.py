"""verisim-cue conformance: the contract checks that make the benchmark trustworthy (SPEC-21 §8).

A benchmark is only an asset if its contract holds. verisim-cue's defining contract is the one no
oracle-free computer-use benchmark can offer: **ground-truth labels** (the faithful predictor scores
*exactly* 1.0 on every task, because the oracle predicts itself) and a **well-ordered
structure->content spectrum** (the task orders are distinct and increasing, so the load-bearing
frontier is measured along a real axis). These checks are torch-free and deterministic -- they
the *battery*, not any model -- so CI runs them without the RL/torch stack.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.acd.host_integrity import make_workload, oracle_step
from verisim.hostoracle.reference import ReferenceHostOracle

from .pack import CueManifest
from .tasks import TASK_SUITE, keyed_defense_reward


@dataclass(frozen=True)
class ConformanceResult:
    """One contract check's verdict."""

    check: str
    passed: bool
    detail: str


def check_ordered_spectrum(manifest: CueManifest) -> ConformanceResult:
    """The task orders are distinct and strictly increasing (a real structure->content axis)."""
    orders = [t.order for t in manifest.tasks]
    ok = orders == sorted(orders) and len(set(orders)) == len(orders)
    return ConformanceResult(
        "ordered-spectrum", ok,
        f"orders={orders} (distinct, increasing)" if ok else f"orders={orders}",
    )


def check_ground_truth_labels(manifest: CueManifest) -> ConformanceResult:
    """The faithful predictor (oracle) scores *exactly* 1.0 per task -- labels are ground truth.

    This is the property that makes the load-bearing verdict meaningful: the faithful ceiling is
    exact, so any free-predictor shortfall is a real drift, not measurement slack. Torch-free (the
    oracle predicts itself); a few seeds suffice since the oracle is deterministic.
    """
    oracle = ReferenceHostOracle()
    step = oracle_step(oracle)
    worst = 1.0
    for task in TASK_SUITE:
        for seed in manifest.seeds[:4]:
            start, actions = make_workload(seed, manifest.horizon, driver=manifest.driver,
                                           oracle=oracle)
            r = keyed_defense_reward(step, step, start, actions, task.budget, task.key_fn)
            worst = min(worst, r)
    ok = worst == 1.0
    return ConformanceResult(
        "ground-truth-labels", ok,
        f"faithful predictor min reward = {worst:.4f} (exact == 1.0)",
    )


def check_keyed_dimensions(manifest: CueManifest) -> ConformanceResult:
    """Every task keys on a recognized host dimension (procs / fds / fs)."""
    valid = {"procs", "fds", "fs"}
    bad = [t.name for t in manifest.tasks if t.keyed_dimension not in valid]
    return ConformanceResult(
        "keyed-dimensions", not bad,
        "all keyed dimensions recognized" if not bad else f"unrecognized: {bad}",
    )


def run_conformance(manifest: CueManifest | None = None) -> list[ConformanceResult]:
    """Run the full verisim-cue conformance suite (torch-free, deterministic)."""
    manifest = manifest or CueManifest()
    return [
        check_ordered_spectrum(manifest),
        check_keyed_dimensions(manifest),
        check_ground_truth_labels(manifest),
    ]


def all_passed(results: list[ConformanceResult]) -> bool:
    return all(r.passed for r in results)
