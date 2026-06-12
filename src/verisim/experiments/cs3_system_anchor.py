"""Experiment CS3 -- the scale law survives the system oracle (SPEC-21 §6, H90).

SPEC-21's scale law (`scale_law.py`) measures the **load-bearing frontier** -- the per-task
faithful-vs-free *gap* and its *recession* with capacity -- against the deterministic reference
oracle. The standing question H90 asks is whether that object is about *real* computer-use dynamics
or only about a model of them: does the frontier (and its motion) hold when the SPEC-11
[`SandboxOracle`](../../src/verisim/oracle/sandbox.py) -- a real ``/bin/sh`` on a real kernel --
replaces the reference oracle as the reality anchor?

CS3 is the scale-law sibling of [`sy1`](./sy1.py) (the differential agreement table) and
[`pb_transfer`](./pb_transfer.py) (the sim-to-emulation horizon gap): where those measure *agree-
ment* and *faithful horizon* against the real shell, CS3 measures the scale law's headline object --
the **load-bearing gap** -- against both anchors, on the content grammar where the residue (H88) is.

The measurement (CPU, seconds, torch-free):

  - **Two tasks from the cue suite**, in the v0 filesystem world (the content slice the real shell
    can anchor): ``file-integrity`` (keyed on *which* files exist -- structure) and
    ``content-value`` (keyed on the *(path, content)* pairs -- the deep-content residue probe, CP3).
  - **A capacity-proxy ladder** -- a controlled drifting stand-in for ``M_θ`` (the trained arm
    deferred, the repo's LP7 rule), faithful on structure and drifting on *content* with probability
    ``1 - α``. ``α`` is the capacity proxy: ``α -> 1`` is a larger, less-drifting model. It
    reproduces SPEC-21's thesis -- structure (paths/tree) is learnable; content (the written tokens)
    is the high-entropy dimension a model drifts on -- so the per-task gap *recedes* as ``α`` rises,
    the same frontier motion the GPU run measures across the real capacity ladder.
  - **Both reality anchors.** Each gap is scored against the reference oracle *and* the real shell.
    The stand-in always fetches its "correct" transition from the *reference* oracle (the SY1/
    pb_transfer discipline), so it is identical across the overlay; the only variable is the oracle
    producing the ground-truth keyed set. SY1/H27 proved the two agree bit-exactly on this grammar,
    so the prediction is ``gap_sys ≈ gap_ref`` at *every* rung -- the frontier and its motion are
    anchor-invariant, i.e. the scale law is about real computer-use dynamics.

What CS3 confirms (all against the real kernel, not a model of it): the **gradient** (content gap >
structure gap) survives the real shell; the gap **recedes** with capacity identically under both
anchors; the cheap keyed-drift **forecasts** the gap under the real kernel too (H89); and the
deep-content **residue** stays load-bearing under the real shell at the top of the proxy ladder
(H88-consistency). Refuted if real-kernel nondeterminism or semantics move any of these materially.

``skipif``-guarded and disclosed when no real shell is present -- a skip is never a counted result
(the SPEC-11 §2.5 rule). Deterministic, seeded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.data.drivers import Driver
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import File, State
from verisim.metrics.calibration import spearman
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

StepFn = Callable[[State, Action], State]
KeyFn = Callable[[State], set[Any]]

# A sentinel content the v0 grammar can never produce (content tokens are lowercase words joined
# by ``append``), so a drifted file is a *guaranteed* content miss while its path stays faithful --
# the clean "structure learnable, content high-entropy" model of M_θ's drift.
DRIFT_SENTINEL = "\x00drift"


# --- the keyed-set extractors (the content slice of the cue suite, in the v0 fs world) ------------


def which_file_set(state: State) -> set[Any]:
    """Structure: *which* files exist -- the file-integrity keyed set (paths only)."""
    return {path for path, node in state.fs.items() if isinstance(node, File)}


def content_set(state: State) -> set[Any]:
    """Deep content: the *(path, content)* pairs -- the content-value residue probe (CP3/H88).

    A drifted predictor can name the right file yet mis-predict its content, so this keyed set is
    strictly harder than :func:`which_file_set`; empty files are excluded (no content to defend),
    mirroring :func:`verisim.cue.tasks.file_contents`.
    """
    return {
        (path, node.content)
        for path, node in state.fs.items()
        if isinstance(node, File) and node.content != ""
    }


@dataclass(frozen=True)
class CueAnchorTask:
    """One content-grammar task, parallel to :class:`verisim.cue.tasks.CueTask`."""

    name: str
    keyed_dimension: str
    order: int  # 0 = structure (which-file) ... 1 = deep content (path, content)
    key_fn: KeyFn
    budget: int = 4


#: The structure->content pair the real shell can anchor (the cue file/content rungs).
ANCHOR_TASKS: tuple[CueAnchorTask, ...] = (
    CueAnchorTask("file-integrity", "fs-paths", 0, which_file_set),
    CueAnchorTask("content-value", "fs-content", 1, content_set),
)


# --- the capacity-proxy stand-in for M_θ (faithful on structure, drifts on content) ---------------


class ContentDriftProposer:
    """A seeded, torch-free stand-in for ``M_θ``: faithful on structure, drifts on content.

    A content step is *learnable* with probability ``1 - irreducible`` and *effectively-unlearnable*
    with probability ``irreducible`` (SPEC-21 H88: content keyed on input the model has no way to
    learn). On a learnable content step the proposer predicts the exact reference transition with
    probability ``α`` (the capacity proxy: ``α -> 1`` is a larger model); on an unlearnable content
    step, or when capacity fails it, it keeps the *structure* (the file is at the right path) but
    corrupts the *content* -- the SPEC-21 model of where capacity helps (paths/tree, learnable) and
    where it drifts (content, high-entropy). The ``irreducible`` fraction is the residue floor: a
    gap survives even at ``α = 1``, so the deep-content task stays load-bearing at every rung (H88).
    Structure is always faithful, so the which-file task never drifts.

    It always fetches its correct transition from the *reference* oracle, so it is identical
    regardless of which oracle later scores it -- the SY1/pb_transfer invariance discipline that
    makes the only variable in the overlay the reality anchor.
    """

    def __init__(self, alpha: float, seed: int, irreducible: float = 0.0) -> None:
        self._ref = ReferenceOracle()
        self._alpha = alpha
        self._seed = seed
        self._irreducible = irreducible
        self._step = 0

    def _faithful_here(self, action: Action) -> bool:
        key = (self._seed, self._step, action.raw)
        digest = hashlib.sha256(repr(key).encode()).digest()
        capacity_coin = int.from_bytes(digest[:8], "big") / 2.0**64
        residue_coin = int.from_bytes(digest[8:16], "big") / 2.0**64
        if residue_coin < self._irreducible:
            return False  # effectively-unlearnable content -- drifts at every capacity (H88)
        return capacity_coin < self._alpha

    def step(self, state: State, action: Action) -> State:
        nxt = self._ref.step(state, action).state
        faithful = self._faithful_here(action)
        self._step += 1
        if faithful or action.name not in ("write", "append"):
            return nxt
        # drift on content: keep every path (structure faithful), corrupt the content the action
        # changed (so the keyed which-file set still matches, but the (path, content) set misses).
        out = nxt.copy()
        for path, node in nxt.fs.items():
            if not isinstance(node, File) or node.content == "":
                continue
            before = state.fs.get(path)
            changed = not isinstance(before, File) or before.content != node.content
            if changed:
                out.fs[path] = File(content=DRIFT_SENTINEL, mode=node.mode)
        return out


# --- the generic predictive-defense gap (the cue pattern, over the v0 fs world) -------------------


def _oracle_step(oracle: Oracle) -> StepFn:
    return lambda state, action: oracle.step(state, action).state


def rollout_keyed(step: StepFn, start: State, actions: Sequence[Action], key_fn: KeyFn) -> set[Any]:
    """Roll the workload under ``step``; return the **cumulative** keyed set ever touched."""
    state = start
    seen = set(key_fn(state))
    for action in actions:
        state = step(state, action)
        seen |= key_fn(state)
    return seen


def keyed_defense_reward(
    predictor: StepFn, true_step: StepFn, start: State,
    actions: Sequence[Action], budget: int, key_fn: KeyFn,
) -> float:
    """Protect the ``budget`` predicted-keyed objects; score vs the true keyed set (cue UA8)."""
    predicted = sorted(rollout_keyed(predictor, start, actions, key_fn), key=repr)
    true_set = rollout_keyed(true_step, start, actions, key_fn)
    if not true_set:
        return 1.0
    protected = set(predicted[:budget])
    return len(protected & true_set) / min(budget, len(true_set))


@dataclass(frozen=True)
class CS3Config:
    """A small, fast content-grammar scale-law-vs-real-shell instance (dependency-free core)."""

    env: EnvConfig = DEFAULT_CONFIG
    driver: str = "structural"  # the validated grammar (SY1/H27 bit-exact)
    alphas: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)  # the capacity proxy ladder
    horizon: int = 16
    seeds: tuple[int, ...] = tuple(range(800, 812))
    irreducible: float = 0.12  # the H88 residue floor: content that drifts at *every* capacity
    threshold: float = 0.05  # gap above which a task is load-bearing (the frontier contour)
    anchor_tol: float = 1e-9  # |gap_sys - gap_ref| below which the frontier is anchor-invariant

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CS3Config:
        b = CS3Config()
        return CS3Config(
            driver=d.get("driver", b.driver),
            alphas=tuple(d.get("alphas", b.alphas)),
            horizon=d.get("horizon", b.horizon),
            seeds=tuple(d.get("seeds", b.seeds)),
            irreducible=d.get("irreducible", b.irreducible),
            threshold=d.get("threshold", b.threshold),
            anchor_tol=d.get("anchor_tol", b.anchor_tol),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CS3Config:
        return CS3Config.from_dict(json.loads(Path(path).read_text()))

    @staticmethod
    def smoke() -> CS3Config:
        return CS3Config(alphas=(0.3, 0.9), horizon=10, seeds=tuple(range(800, 806)))


@dataclass(frozen=True)
class CS3Cell:
    """One (α, task) cell: the load-bearing gap against each anchor + the cheap drift forecast."""

    alpha: float
    task: str
    order: int
    gap_ref: float  # faithful-vs-free gap, reference oracle as reality anchor
    gap_sys: float  # ... real /bin/sh as reality anchor
    drift_ref: float  # cheap keyed-miss fraction (the H89 forecast), reference anchor
    drift_sys: float  # ... real shell anchor
    n: int

    @property
    def anchor_delta(self) -> float:
        return abs(self.gap_sys - self.gap_ref)


@dataclass
class CS3Result:
    """The full α-ladder × task sweep, scored against both reality anchors."""

    available: bool
    platform: str
    cells: list[CS3Cell]


def _actions(config: CS3Config, ref: Oracle, seed: int) -> list[Action]:
    """A seeded content workload, generated by stepping the reference oracle (the cue regime)."""
    import random

    drv = Driver(config.driver, config.env, random.Random(seed))
    state = State.empty()
    actions: list[Action] = []
    for _ in range(config.horizon):
        action = drv.sample(state)
        actions.append(action)
        state = ref.step(state, action).state
    return actions


def _gap(task: CueAnchorTask, alpha: float, anchor: Oracle, start: State,
         actions: list[Action], seed: int, irreducible: float) -> float:
    """The faithful-vs-free predictive-defense gap for one workload, ``anchor`` as ground truth."""
    true_step = _oracle_step(anchor)
    faithful = keyed_defense_reward(true_step, true_step, start, actions, task.budget, task.key_fn)
    proposer = ContentDriftProposer(alpha, seed, irreducible)
    free = keyed_defense_reward(
        proposer.step, true_step, start, actions, task.budget, task.key_fn
    )
    return faithful - free


def _drift(task: CueAnchorTask, alpha: float, anchor: Oracle, start: State,
           actions: list[Action], seed: int, irreducible: float) -> float:
    """The cheap keyed-miss fraction ``|true \\ free| / |true|`` (the H89 forecast of the gap)."""
    free_set = rollout_keyed(
        ContentDriftProposer(alpha, seed, irreducible).step, start, actions, task.key_fn
    )
    true_set = rollout_keyed(_oracle_step(anchor), start, actions, task.key_fn)
    return len(true_set - free_set) / len(true_set) if true_set else 0.0


def run_cs3(config: CS3Config | None = None, *, sys_oracle: Oracle | None = None) -> CS3Result:
    """Per (α, task): the load-bearing gap + cheap drift vs the reference and the real shell."""
    import sys as _sys

    config = config or CS3Config()
    ref = ReferenceOracle()
    try:
        syscle = sys_oracle or SandboxOracle()
    except SystemOracleUnavailable:
        return CS3Result(available=False, platform=_sys.platform, cells=[])

    cells: list[CS3Cell] = []
    for alpha in config.alphas:
        for task in ANCHOR_TASKS:
            gap_ref: list[float] = []
            gap_sys: list[float] = []
            drift_ref: list[float] = []
            drift_sys: list[float] = []
            for seed in config.seeds:
                start = State.empty()
                actions = _actions(config, ref, seed)
                gap_ref.append(_gap(task, alpha, ref, start, actions, seed, config.irreducible))
                gap_sys.append(_gap(task, alpha, syscle, start, actions, seed, config.irreducible))
                drift_ref.append(_drift(task, alpha, ref, start, actions, seed, config.irreducible))
                drift_sys.append(
                    _drift(task, alpha, syscle, start, actions, seed, config.irreducible)
                )
            cells.append(CS3Cell(
                alpha=alpha, task=task.name, order=task.order,
                gap_ref=fmean(gap_ref), gap_sys=fmean(gap_sys),
                drift_ref=fmean(drift_ref), drift_sys=fmean(drift_sys), n=len(config.seeds),
            ))
    return CS3Result(available=True, platform=_sys.platform, cells=cells)


# --- the verdicts (H90 anchor-invariance; the gradient/recession/residue under the real kernel) ---


def cs3_verdict(result: CS3Result, config: CS3Config | None = None) -> dict[str, Any]:
    """H90 + the frontier properties, all measured against the real shell.

    - **anchor_invariant (H90):** the largest ``|gap_sys - gap_ref|`` over every cell is within
      ``anchor_tol`` -- the load-bearing gap (and so the frontier) does not move when the real shell
      replaces the reference oracle as the reality anchor.
    - **gradient_holds:** at every α the content-value gap ≥ the file-integrity gap, vs the real
      shell (the structure->content gradient survives the real kernel).
    - **recedes:** the content-value gap is non-increasing in α, vs the real shell (the frontier
      motion -- structure-first recession's content arm -- survives the real kernel).
    - **residue_under_real_kernel (H88-consistency):** the content gap stays above ``threshold`` at
      the top α rung, vs the real shell (the residue is load-bearing on a real OS, not just in sim).
    - **forecastable (H89):** the cheap keyed drift orders the gap under the real kernel too.
    """
    config = config or CS3Config()
    if not result.available or not result.cells:
        return {"available": False}
    max_delta = max(c.anchor_delta for c in result.cells)
    alphas = sorted({c.alpha for c in result.cells})
    by = {(c.alpha, c.task): c for c in result.cells}

    gradient = all(
        by[(a, "content-value")].gap_sys >= by[(a, "file-integrity")].gap_sys - 1e-9 for a in alphas
    )
    content_by_alpha = [by[(a, "content-value")].gap_sys for a in alphas]
    recedes = all(
        content_by_alpha[i + 1] <= content_by_alpha[i] + 1e-9 for i in range(len(alphas) - 1)
    )
    residue = by[(alphas[-1], "content-value")].gap_sys > config.threshold
    drifts = [c.drift_sys for c in result.cells]
    gaps = [c.gap_sys for c in result.cells]
    rho = spearman(drifts, gaps)
    return {
        "available": True,
        "platform": result.platform,
        "n_cells": len(result.cells),
        "max_anchor_delta": max_delta,
        "anchor_invariant": max_delta <= config.anchor_tol,  # H90
        "gradient_holds": gradient,
        "content_gap_recedes": recedes,
        "content_gap_by_alpha": {
            a: round(g, 4) for a, g in zip(alphas, content_by_alpha, strict=True)
        },
        "residue_under_real_kernel": residue,  # H88-consistency
        "forecast_spearman": rho,
        "forecastable": rho > 0.6,  # H89
    }


CSV_HEADER = "alpha,task,order,gap_ref,gap_sys,anchor_delta,drift_ref,drift_sys,n"


def write_csv(result: CS3Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in result.cells:
        rows.append(
            f"{c.alpha:.4f},{c.task},{c.order},{c.gap_ref:.6f},{c.gap_sys:.6f},"
            f"{c.anchor_delta:.6e},{c.drift_ref:.6f},{c.drift_sys:.6f},{c.n}"
        )
    out.write_text("\n".join(rows) + "\n")
    return out


def _print_summary(result: CS3Result, verdict: dict[str, Any]) -> None:
    print("CS3 / H90 -- the scale law survives the system oracle (real /bin/sh):")
    if not result.available:
        print("  [system oracle UNAVAILABLE -- CS3 skipped, not counted (§2.5)]")
        return
    print("  [trained-M_θ arm DEFERRED -- proposer is the content-drifting capacity proxy]")
    print(f"  {'α':>5} {'task':>14} {'gap_ref':>8} {'gap_sys':>8} {'Δanchor':>9} {'drift_sys':>9}")
    for c in result.cells:
        print(f"  {c.alpha:>5.2f} {c.task:>14} {c.gap_ref:>8.3f} {c.gap_sys:>8.3f} "
              f"{c.anchor_delta:>9.2e} {c.drift_sys:>9.3f}")
    print(f"  H90 anchor-invariant (max Δ={verdict['max_anchor_delta']:.2e}): "
          f"{verdict['anchor_invariant']}")
    print(f"  gradient (content ≥ file) under real kernel:   {verdict['gradient_holds']}")
    print(f"  content gap recedes with capacity (real shell): {verdict['content_gap_recedes']}  "
          f"{verdict['content_gap_by_alpha']}")
    print(f"  residue load-bearing under real kernel (H88): {verdict['residue_under_real_kernel']}")
    print(f"  H89 forecast (cheap drift -> gap, real shell): spearman="
          f"{verdict['forecast_spearman']:+.3f}  forecastable={verdict['forecastable']}")


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CS3 -- the scale law survives the system oracle (SPEC-21 H90)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cs3_system_anchor.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = CS3Config.smoke() if args.smoke else (
        CS3Config.from_json_file(args.config) if args.config else CS3Config()
    )
    result = run_cs3(config)
    verdict = cs3_verdict(result, config)
    _print_summary(result, verdict)
    out = write_csv(result, args.out)
    print(f"wrote {out}")
    if result.available:
        try:
            from figures.plot_cs3 import plot_cs3

            plot_path = Path(args.plot) if args.plot else out.with_suffix(".png")
            plot_cs3(result, verdict, plot_path)
            print(f"wrote {plot_path}")
        except Exception as exc:  # pragma: no cover - plotting optional/local
            print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
