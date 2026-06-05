"""SY2 -- the differential debugger (SPEC-11 §5; milestone SO4; hypothesis H28).

The hypothesis that turns the program's central weakness into a *tool*: every disagreement
the system oracle surfaces is a localizable model defect -- attributable to a specific
command, reproducing deterministically, and fixable in the reference oracle. SY2 is the
standing apparatus that does the localizing.

It has two jobs:

  1. **The divergence atlas.** Sweep the boundary drivers and, for each named class
     (root-protection / overwrite-policy / permission-enforcement / self-subtree), surface
     the *minimal reproducing* ``(state, action)``, the reference's predicted next state,
     and the real kernel's next state, side by side -- a committed catalog of exactly where
     and why the from-scratch POSIX model deviates from a real one. On the v0 grammar these
     are the *documented, intentional* simplifications (SPEC-11 §9), now enumerated rather
     than asserted.

  2. **The teeth control.** A debugger that never catches anything is worthless. SY2 plants
     a *seeded synthetic divergence* -- a deliberately corrupted oracle that drops one edit
     on one transition -- and asserts the differential harness detects it and localizes it
     to the right command. This proves the apparatus has teeth: the moment the grammar grows
     (Tier-B coreutils) and the first *real* divergence appears, SY2 will catch it.

The one genuine reference bug this machinery already caught -- ``mv``/``cp`` of a directory
into its own subtree producing an orphaned (invalid) state -- was fixed at the source
(``oracle/reference.py``), exactly the H28 loop: surface, localize, fix, re-run to agreement.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.data.drivers import Driver
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import State
from verisim.oracle.base import Oracle, StepResult
from verisim.oracle.differential import (
    BOUNDARY_CLASSES,
    canonical_world,
    differential_step,
)
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

BOUNDARY_DRIVERS = ("weighted", "adversarial")


@dataclass(frozen=True)
class Disagreement:
    """One localized divergence: the minimal reproducer + both oracles' verdicts."""

    divergence_class: str
    action_raw: str
    command: str
    state_fingerprint: str  # a short, human-readable digest of the input state
    ref_exit: int
    sys_exit: int
    ref_world: str
    sys_world: str
    explanation: str


_EXPLAIN = {
    "root_protection": "v0 makes '/' undeletable/uncopyable; the real kernel operates on it.",
    "overwrite_policy": "v0 mv/cp refuse to clobber an existing target; the kernel clobbers.",
    "permission_enforcement": "v0 models mode as data; the kernel enforces traversal/access.",
    "self_subtree": "v0+GNU reject copying a dir into its own subtree; BSD cp (macOS) does not.",
}


def _fingerprint(state: State) -> str:
    """A short digest of the input state: number of paths + the sorted top-level names."""
    top = sorted(p for p in state.fs if p != "/" and p.count("/") == 1)
    return f"{len(state.fs)} paths; top-level={top[:6]}"


@dataclass
class SY2Result:
    available: bool = True
    platform: str = ""
    atlas: list[Disagreement] = field(default_factory=list)  # one minimal reproducer per class
    class_counts: dict[str, int] = field(default_factory=dict)
    teeth_passed: bool = False
    teeth_detail: str = ""


def build_atlas(config_env: EnvConfig, ref: Oracle, sys: Oracle, *,
                seeds: tuple[int, ...], steps: int) -> tuple[list[Disagreement], dict[str, int]]:
    """Sweep the boundary drivers; return one minimal reproducer per class + class counts."""
    found: dict[str, Disagreement] = {}
    counts: dict[str, int] = {}
    for driver in BOUNDARY_DRIVERS:
        for seed in seeds:
            drv = Driver(driver, config_env, random.Random(seed))
            s = State.empty()
            for _ in range(steps):
                a = drv.sample(s)
                rec = differential_step(s, a, ref, sys)
                if not rec.agree:
                    cls = rec.divergence_class
                    counts[cls] = counts.get(cls, 0) + 1
                    if cls not in found:
                        r_ref: StepResult = ref.step(s, a)
                        r_sys: StepResult = sys.step(s, a)
                        found[cls] = Disagreement(
                            divergence_class=cls,
                            action_raw=a.raw,
                            command=a.name,
                            state_fingerprint=_fingerprint(s),
                            ref_exit=r_ref.exit_code,
                            sys_exit=r_sys.exit_code,
                            ref_world=canonical_world(r_ref.state)[:160],
                            sys_world=canonical_world(r_sys.state)[:160],
                            explanation=_EXPLAIN.get(cls, "unexplained -- a first-class finding"),
                        )
                s = ref.step(s, a).state
    atlas = [found[c] for c in BOUNDARY_CLASSES if c in found]
    return atlas, counts


class _FaultyOracle:
    """A deliberately corrupted oracle: on one targeted command it drops its delta (a planted
    bug). Used only as SY2's teeth control -- proving the harness catches an injected divergence.
    """

    def __init__(self, base: Oracle, corrupt_command: str) -> None:
        self._base = base
        self._corrupt = corrupt_command

    def step(self, state: State, action: Action) -> StepResult:
        r = self._base.step(state, action)
        if action.name == self._corrupt:
            # Drop the edits: pretend "nothing happened" -- a wrong next state.
            from verisim.delta.edits import SetResult
            from verisim.env.state import content_hash
            return StepResult(state=state.copy(), delta=[SetResult(0, content_hash(""))],
                              exit_code=r.exit_code, stdout=r.stdout)
        return r

    def reset(self, state: State) -> State:
        return self._base.reset(state)

    def determinism_report(self):  # type: ignore[no-untyped-def]
        return self._base.determinism_report()


def run_teeth_control(ref: Oracle, *, corrupt_command: str = "mkdir") -> tuple[bool, str]:
    """Plant a synthetic divergence and assert the harness detects + localizes it."""
    faulty = _FaultyOracle(ref, corrupt_command)
    drv = Driver("structural", DEFAULT_CONFIG, random.Random(0))
    s = State.empty()
    detected_on: list[str] = []
    false_positive = False
    for _ in range(60):
        a = drv.sample(s)
        rec = differential_step(s, a, ref, faulty)
        if not rec.agree:
            if a.name == corrupt_command:
                detected_on.append(a.raw)
            else:
                false_positive = True
        s = ref.step(s, a).state
    passed = bool(detected_on) and not false_positive
    detail = (f"planted-bug on {corrupt_command!r}: detected on {len(detected_on)} transition(s), "
              f"false positives={false_positive}")
    return passed, detail


def run_sy2(*, seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7), steps: int = 40,
            sys: Oracle | None = None) -> SY2Result:
    """Build the divergence atlas and run the teeth control (H28)."""
    import sys as _sys

    ref = ReferenceOracle()
    try:
        sys = sys or SandboxOracle()
    except SystemOracleUnavailable:
        # Even with no system oracle, the teeth control (a planted bug in the *reference*)
        # still proves the harness detects divergences -- the standing-apparatus guarantee.
        passed, detail = run_teeth_control(ref)
        return SY2Result(available=False, platform=_sys.platform,
                         teeth_passed=passed, teeth_detail=detail)
    atlas, counts = build_atlas(DEFAULT_CONFIG, ref, sys, seeds=seeds, steps=steps)
    passed, detail = run_teeth_control(ref)
    return SY2Result(available=True, platform=_sys.platform, atlas=atlas,
                     class_counts=counts, teeth_passed=passed, teeth_detail=detail)


CSV_HEADER = "divergence_class,command,action,state,ref_exit,sys_exit,count,explanation"


def write_csv(result: SY2Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for d in result.atlas:
        safe = d.explanation.replace(",", ";")
        fp = d.state_fingerprint.replace(",", ";")
        count = result.class_counts.get(d.divergence_class, 0)
        rows.append(f"{d.divergence_class},{d.command},{d.action_raw},{fp},"
                    f"{d.ref_exit},{d.sys_exit},{count},{safe}")
    verdict = "PASS" if result.teeth_passed else "FAIL"
    rows.append(f"teeth_control,_,planted-bug,_,_,_,{verdict},"
                f"{result.teeth_detail.replace(',', ';')}")
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run SY2 (system-oracle differential debugger).")
    parser.add_argument("--out", type=str, default="runs/sy2/records.jsonl")
    parser.add_argument("--csv", type=str, default="figures/sy2_disagreements.csv")
    args = parser.parse_args()
    result = run_sy2()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    recs: list[dict[str, Any]] = [
        {"class": d.divergence_class, "command": d.command, "action": d.action_raw,
         "ref_exit": d.ref_exit, "sys_exit": d.sys_exit, "explanation": d.explanation}
        for d in result.atlas
    ]
    Path(args.out).write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    path = write_csv(result, args.csv)
    print(f"wrote {path}  (platform={result.platform}, available={result.available})")
    for d in result.atlas:
        print(f"  [{d.divergence_class:22s}] {d.action_raw:24s} ref={d.ref_exit} sys={d.sys_exit}"
              f"  x{result.class_counts.get(d.divergence_class, 0)}")
    print(f"  teeth control: {'PASS' if result.teeth_passed else 'FAIL'} -- {result.teeth_detail}")
    if not result.teeth_passed:
        raise SystemExit("SY2 TEETH FAILURE: the harness did not catch a planted divergence")


if __name__ == "__main__":  # pragma: no cover
    main()
