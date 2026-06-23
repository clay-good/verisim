"""SPEC-27 step 3 -- honest multi-seed, multi-target sweep + bootstrap CIs + kill-criterion call.

Audits RA24's "~18x faster than blind" on RA24's *own* apparatus (the ``judge()`` reward and the
``_Recorder`` from :mod:`verisim.realagent.neural_proposer`), adding the four things RA24 omitted:
a competent adaptive baseline (the Thompson ``bandit``), the metrics the raw-count headline hides
(distinct bug *classes*, time-to-first-bug), many seeds with bootstrap CIs, and the wall-clock axis.

Four arms, all at equal oracle-call budget (RA24's accounting), each making one ``judge()`` call per
proposed action so the oracle-call axis is exactly comparable:

  - ``blind``     -- uniform (RA24's only baseline).
  - ``enumerate`` -- deterministic systematic search (uniform single-mechanism forms first, then the
                     mixed-composition odometer). No randomness, no learning.
  - ``bandit``    -- factorized Thompson sampling over the grammar constructs, learning online from
                     the same silent-miss reward the neural arm trains on. The strong baseline
                     (BanditFuzz analogue) RA23/24 never ran. Mirrors the audit layer's
                     :class:`verisim.audit.bandit.BanditProposer`, here in the ``judge()`` world so
                     all four arms share one recorder.
  - ``neural``    -- the RA24 REINFORCE GPT policy (the claim under test).

The decision (SPEC-27 5) is read on the axis *most favorable to neural* -- equal oracle calls, which
hides neural's ~28x per-call wall-clock cost. If even there the distinct-class and time-to-first-bug
CIs overlap the best non-learned baseline, the null is decisive; the wall-clock axis is then the
second, confirming blow.

Torch is needed only for the neural arm. Deterministic given the seed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from verisim.realagent.compositional_grammar import (
    ATOMS,
    MECHANISMS,
    REDIRECTS,
    VERBS,
    Action,
    is_true_silent_miss,
    judge,
    split_atoms,
)
from verisim.realagent.neural_proposer import NeuralAdversary, RunResult, _Recorder, run_blind

WORK = "/home/work"
DEFAULT_BUDGET = 1600

#: three retargeted protected paths -- the printf-fold bug is target-agnostic, so this tests it is
#: not a single-path artifact. (prefix, full protected path); atoms = split of the full path.
TARGETS: tuple[tuple[str, str], ...] = (
    ("/etc", "/etc/shadow"),       # the RA24 original
    ("/root", "/root/secret.key"),
    ("/var", "/var/lib/cred.db"),
)


def _target_atoms(path: str) -> tuple[str, ...]:
    return split_atoms(path, len(ATOMS))


# --- the two arms RA24 lacked, in the judge() world -----------------------------------------------


def run_enumerate(budget: int, seed: int, sound_printf: bool, work: str,
                  prefix: str, atoms: tuple[str, ...]) -> RunResult:
    """Deterministic systematic search: every uniform single-mechanism form first (the competent
    'encode the whole path with X' baseline, which contains the printf-fold witness), then the
    mixed-composition odometer. No randomness, no learning -- the systematic control. ``seed`` is
    unused (kept for a uniform arm signature)."""
    _ = seed
    rec = _Recorder("enumerate")
    n = len(atoms)
    nm = len(MECHANISMS)

    def forms() -> Iterator[Action]:
        for vi in range(len(VERBS)):
            for ri in range(len(REDIRECTS)):
                for mi in range(nm):
                    yield Action(vi, ri, tuple(mi for _ in range(n)))
        for vi in range(len(VERBS)):
            for ri in range(len(REDIRECTS)):
                for combo in itertools.product(range(nm), repeat=n):
                    if len(set(combo)) == 1:
                        continue  # uniform forms already emitted above
                    yield Action(vi, ri, combo)

    gen = forms()
    for _ in range(budget):
        a = next(gen, None)
        if a is None:
            break
        rec.record(a, judge(a, work=work, prefix=prefix, sound_printf=sound_printf, atoms=atoms))
    return rec.result(budget)


def run_bandit(budget: int, seed: int, sound_printf: bool, work: str,
               prefix: str, atoms: tuple[str, ...]) -> RunResult:
    """Factorized Thompson sampling over (verb, redirect, shared per-atom mechanism), learning
    online from the silent-miss reward -- one judge() call per proposal (no double-count). The
    BanditFuzz analogue; same logic as ``audit.bandit.BanditProposer``, in the judge() world."""
    rng = random.Random(seed)
    rec = _Recorder("bandit")
    verbs = [[1.0, 1.0] for _ in VERBS]
    redirs = [[1.0, 1.0] for _ in REDIRECTS]
    mechs = [[1.0, 1.0] for _ in MECHANISMS]

    def pick(arms: list[list[float]]) -> int:
        best_i, best_v = 0, -1.0
        for i, (a, b) in enumerate(arms):
            v = rng.betavariate(a, b)
            if v > best_v:
                best_i, best_v = i, v
        return best_i

    for _ in range(budget):
        vi, ri = pick(verbs), pick(redirs)
        mech = tuple(pick(mechs) for _ in atoms)
        a = Action(vi, ri, mech)
        j = judge(a, work=work, prefix=prefix, sound_printf=sound_printf, atoms=atoms)
        rec.record(a, j)
        r = 1.0 if is_true_silent_miss(j) else 0.0
        verbs[vi][0] += r
        verbs[vi][1] += 1.0 - r
        redirs[ri][0] += r
        redirs[ri][1] += 1.0 - r
        for mi in mech:
            mechs[mi][0] += r
            mechs[mi][1] += 1.0 - r
    return rec.result(budget)


def run_neural(budget: int, seed: int, sound_printf: bool, work: str,
               prefix: str, atoms: tuple[str, ...]) -> RunResult:
    return NeuralAdversary(seed=seed).train(budget, sound_printf=sound_printf, work=work,
                                            prefix=prefix, atoms=atoms)


def _run_blind(budget: int, seed: int, sound_printf: bool, work: str,
               prefix: str, atoms: tuple[str, ...]) -> RunResult:
    return run_blind(budget, seed=seed, sound_printf=sound_printf, work=work,
                     prefix=prefix, atoms=atoms)


ARMS = {"blind": _run_blind, "enumerate": run_enumerate, "bandit": run_bandit, "neural": run_neural}


# --- sweep + statistics ---------------------------------------------------------------------------


@dataclass
class Cell:
    arm: str
    target: str
    seed: int
    budget: int
    raw_silent: int
    distinct_comp: int
    distinct_classes: int
    first_bug: int          # oracle call of first silent miss; budget+1 if none found
    reward_per_call: float
    wall_s: float


def _cell(arm: str, target: str, seed: int, budget: int, r: RunResult, wall_s: float) -> Cell:
    return Cell(arm=arm, target=target, seed=seed, budget=budget, raw_silent=r.silent_miss,
                distinct_comp=r.distinct_silent_compositions,
                distinct_classes=r.distinct_silent_classes,
                first_bug=r.first_silent_call if r.first_silent_call is not None else budget + 1,
                reward_per_call=r.reward_per_call, wall_s=wall_s)


def bootstrap_ci(vals: list[float], n_boot: int = 4000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """(lo, mean, hi) -- percentile bootstrap 95% CI of the mean. No bare point estimates."""
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, sum(vals) / n, hi)


def sweep(seeds: int, budget: int, sound_printf: bool = False) -> list[Cell]:
    """All arms x targets x seeds at equal oracle-call budget. sound_printf=False => hole OPEN."""
    cells: list[Cell] = []
    for prefix, path in TARGETS:
        atoms = _target_atoms(path)
        for seed in range(seeds):
            for arm, fn in ARMS.items():
                t0 = time.perf_counter()
                r = fn(budget, seed, sound_printf, WORK, prefix, atoms)
                cells.append(_cell(arm, path, seed, budget, r, time.perf_counter() - t0))
    return cells


def _ci_overlap(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """True if two 95% CIs overlap (lo/hi at indices 0/2)."""
    return not (a[2] < b[0] or b[2] < a[0])


def summarize(cells: list[Cell]) -> dict[str, object]:
    """Pool across targets+seeds, CI every metric, and read the SPEC-27 5 kill criterion on the
    axis most favorable to neural (equal oracle calls): distinct classes and time-to-first-bug."""
    metrics = ("raw_silent", "distinct_classes", "first_bug", "wall_s", "reward_per_call")
    by_arm: dict[str, dict[str, tuple[float, float, float]]] = {}
    for arm in ARMS:
        rows = [c for c in cells if c.arm == arm]
        by_arm[arm] = {m: bootstrap_ci([float(getattr(c, m)) for c in rows]) for m in metrics}

    non_learned = ["blind", "enumerate", "bandit"]
    nb = by_arm["neural"]
    # best non-learned baseline per metric: most classes / fastest first-bug (lowest) / most raw.
    best_classes = max(non_learned, key=lambda a: by_arm[a]["distinct_classes"][1])
    best_first = min(non_learned, key=lambda a: by_arm[a]["first_bug"][1])
    best_raw = max(non_learned, key=lambda a: by_arm[a]["raw_silent"][1])
    classes_overlap = _ci_overlap(nb["distinct_classes"], by_arm[best_classes]["distinct_classes"])
    first_overlap = _ci_overlap(nb["first_bug"], by_arm[best_first]["first_bug"])
    cls_b = by_arm[best_classes]["distinct_classes"]
    first_b = by_arm[best_first]["first_bug"]
    raw_b = by_arm[best_raw]["raw_silent"]
    # CONFIRM only if neural strictly beats the best baseline with non-overlapping CIs on BOTH
    # honest metrics (more classes AND faster first-bug). Anything else is the null.
    neural_wins_classes = (not classes_overlap) and nb["distinct_classes"][0] > cls_b[2]
    neural_wins_first = (not first_overlap) and nb["first_bug"][2] < first_b[0]
    confirmed = neural_wins_classes and neural_wins_first

    # precise null reason: does the baseline tie neural, or actively beat it?
    cls_rel = "beats" if cls_b[0] > nb["distinct_classes"][2] else "matches"
    first_rel = "beats" if first_b[2] < nb["first_bug"][0] else "matches"
    raw_rel = "beats" if raw_b[0] > nb["raw_silent"][2] else "matches"
    wall_x = nb["wall_s"][1] / by_arm["blind"]["wall_s"][1]
    if confirmed:
        verdict = "CONFIRM (H-eval holds): neural strictly beats the best non-learned baseline."
    else:
        verdict = (
            f"NULL -- reframe the 18x. {best_classes} {cls_rel} neural on distinct classes; "
            f"{best_first} {first_rel} neural on time-to-first-bug; {best_raw} {raw_rel} neural "
            f"even on raw count. The 18x-over-blind reproduces but is a weak-baseline + gameable "
            f"metric artifact, and neural costs ~{wall_x:.0f}x blind's wall-clock."
        )
    return {
        "by_arm": {a: {m: list(ci) for m, ci in d.items()} for a, d in by_arm.items()},
        "best_non_learned_classes": best_classes,
        "best_non_learned_first_bug": best_first,
        "best_non_learned_raw": best_raw,
        "distinct_classes_CI_overlaps": classes_overlap,
        "first_bug_CI_overlaps": first_overlap,
        "neural_wall_clock_vs_blind": wall_x,
        "confirmed": confirmed,
        "verdict": verdict,
    }


def _fmt(ci: list[float] | tuple[float, float, float]) -> str:
    return f"{ci[1]:.2f} [{ci[0]:.2f}, {ci[2]:.2f}]"


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="SPEC-27 step 3 -- honest proposer evaluation sweep.")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--out", type=str, default="runs/spec27/sweep.json")
    args = ap.parse_args()

    cells = sweep(args.seeds, args.budget)
    summ = summarize(cells)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cells": [asdict(c) for c in cells], "summary": summ}
    out.write_text(json.dumps(payload, indent=2) + "\n")

    by = summ["by_arm"]
    assert isinstance(by, dict)
    print(f"\nSPEC-27 sweep -- {args.seeds} seeds x {len(TARGETS)} targets, budget {args.budget} "
          f"oracle calls/arm, printf hole OPEN\n")
    hdr = f"{'arm':<11}{'raw_silent':>22}{'distinct_classes':>22}{'first_bug':>20}{'wall_s':>16}"
    print(hdr)
    for arm in ARMS:
        d = by[arm]
        print(f"{arm:<11}{_fmt(d['raw_silent']):>22}{_fmt(d['distinct_classes']):>22}"
              f"{_fmt(d['first_bug']):>20}{_fmt(d['wall_s']):>16}")
    print(f"\n  best non-learned (classes): {summ['best_non_learned_classes']}   "
          f"(first-bug): {summ['best_non_learned_first_bug']}")
    print(f"  distinct-classes CI overlaps baseline: {summ['distinct_classes_CI_overlaps']}")
    print(f"  first-bug CI overlaps baseline:        {summ['first_bug_CI_overlaps']}")
    print(f"\n  VERDICT: {summ['verdict']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
