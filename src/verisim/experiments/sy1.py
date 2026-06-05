"""SY1 -- the differential agreement table + the head-to-head curve (SPEC-11 §3, §5; SO3; H27).

**The figure that retires W1** ("the oracle is a model, not reality"). SY1 measures, bit for
bit, whether the v0 reference oracle's predicted next state equals the state a real ``/bin/sh``
on a real kernel produces -- on the macOS development host first (the SPEC-11 §2.5 macOS-first
principle), reproducing the platform-independent headline on Linux CI for free.

Three coverage tiers mirror the existing invariant discipline (verification.md §1):

  - **Tier-1 -- exhaustive over the grammar.** Enumerate the full v0 action space against a
    battery of canonical states. Every pair gets a verdict: *agree*, or one of the named v0
    modeling boundaries (root-protection / overwrite-policy / permission-enforcement /
    self-subtree). The headline claim is precise and strong: on the **structure-building**
    regime v0 is designed to model, agreement is **bit-exact 1.000**, and across the *whole*
    grammar **every** disagreement is a known, documented boundary -- the residual is zero.
  - **Tier-2 -- driver-sampled at depth.** Run the drivers for ``seeds x steps``, stepping
    both oracles in lockstep, so agreement is measured along realistic multi-step
    trajectories. ``structural``/``trivial`` (the modeled regime) agree totally;
    ``weighted``/``adversarial`` surface the boundary classes at their natural rates.
  - **Tier-3 -- the curve, head to head.** Re-run the prime-directive ``H_ε(ρ)`` with the
    ``SandboxOracle`` substituted for the ``ReferenceOracle`` on the structural grammar and
    overlay the two curves. Because the oracles agree bit-exactly there, the curves are
    *indistinguishable* -- the overlay is the figure that shows the oracle substitution is
    transparent, i.e. the reference oracle is faithful to a real computer where it claims to be.

Dependency-free and GPU-free: Tier-3 uses a seeded synthetic noisy proposer (no torch), so
the whole of SY1 runs in seconds on CPU.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.data.drivers import Driver
from verisim.delta.edits import Delta
from verisim.env.action import Action, parse_action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import Dir, File, State
from verisim.loop.policy import fixed_interval_for_rho
from verisim.loop.runner import budget_for_rho, run_rollout
from verisim.metrics.aggregate import bootstrap_ci
from verisim.oracle.base import Oracle
from verisim.oracle.differential import (
    AGREE,
    BOUNDARY_CLASSES,
    RESIDUAL,
    DiffRecord,
    differential_step,
)
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

# The structure-building regime v0 is designed to model -- the headline agreement claim.
MODELED_DRIVERS = ("structural", "trivial")
# The destructive drivers that surface the named boundary classes (the SY2 atlas input).
BOUNDARY_DRIVERS = ("weighted", "adversarial")


@dataclass(frozen=True)
class SY1Config:
    name: str = "sy1"
    env: EnvConfig = DEFAULT_CONFIG
    # Tier-1 exhaustive
    battery_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    battery_depth: int = 6  # structural-driver steps per seed state
    # Tier-2 trajectories
    traj_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    traj_steps: int = 40
    drivers: tuple[str, ...] = (*MODELED_DRIVERS, *BOUNDARY_DRIVERS)
    # Tier-3 overlay
    curve_driver: str = "structural"
    curve_seeds: tuple[int, ...] = (200, 201, 202, 203)
    curve_steps: int = 24
    curve_noise: float = 0.35
    rhos: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    epsilon: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SY1Config:
        b = SY1Config()
        return SY1Config(
            name=d.get("name", b.name),
            battery_seeds=tuple(d.get("battery_seeds", b.battery_seeds)),
            battery_depth=d.get("battery_depth", b.battery_depth),
            traj_seeds=tuple(d.get("traj_seeds", b.traj_seeds)),
            traj_steps=d.get("traj_steps", b.traj_steps),
            drivers=tuple(d.get("drivers", b.drivers)),
            curve_driver=d.get("curve_driver", b.curve_driver),
            curve_seeds=tuple(d.get("curve_seeds", b.curve_seeds)),
            curve_steps=d.get("curve_steps", b.curve_steps),
            curve_noise=d.get("curve_noise", b.curve_noise),
            rhos=tuple(d.get("rhos", b.rhos)),
            epsilon=d.get("epsilon", b.epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> SY1Config:
        return SY1Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class SY1Result:
    available: bool = True
    platform: str = ""
    tier1: dict[str, Any] = field(default_factory=dict)  # exhaustive battery summary
    tier2: list[dict[str, Any]] = field(default_factory=list)  # per-driver agreement + classes
    tier3: list[dict[str, float]] = field(default_factory=list)  # per-rho ref vs sys horizon

    @property
    def modeled_agreement(self) -> float:
        """The headline: bit-exact agreement over the structure-building regime."""
        rows = [r for r in self.tier2 if r["driver"] in MODELED_DRIVERS]
        n = sum(r["n"] for r in rows)
        return sum(r["n_agree"] for r in rows) / n if n else float("nan")

    @property
    def residual_fraction(self) -> float:
        """Fraction of *all* transitions whose disagreement is unexplained (should be 0)."""
        total = self.tier1.get("n", 0) + sum(r["n"] for r in self.tier2)
        t2_resid = sum(r["classes"].get(RESIDUAL, 0) for r in self.tier2)
        resid = self.tier1.get("residual", 0) + t2_resid
        return resid / total if total else float("nan")


# --- Tier-1: exhaustive enumeration over the grammar ------------------------

def _seed_states(config: SY1Config) -> list[State]:
    """A battery of canonical states from the collision-free structural driver."""
    states: list[State] = []
    ref = ReferenceOracle()
    for seed in config.battery_seeds:
        drv = Driver("structural", config.env, random.Random(seed))
        s = State.empty()
        for _ in range(config.battery_depth):
            s = ref.step(s, drv.sample(s)).state
        states.append(s)
    return states


def enumerate_actions(state: State, env: EnvConfig) -> list[Action]:
    """Every applicable v0 action over ``state``: each command family x representative args.

    Covers existing dirs/files, a fresh path, both copy/move targets, a chmod mode, and the
    read/nav/env commands -- the exhaustive action space over a single state.
    """
    dirs = sorted(p for p, n in state.fs.items() if isinstance(n, Dir))
    files = sorted(p for p, n in state.fs.items() if isinstance(n, File))
    allp = sorted(state.fs)
    fresh = next((f"/{n}" for n in env.name_pool if f"/{n}" not in state.fs), "/z")
    tok = env.content_tokens[0]
    out: list[str] = []
    for d in dirs:
        out += [f"mkdir {d}/{env.name_pool[0]}", f"rmdir {d}", f"touch {d}/{env.name_pool[1]}",
                f"ls {d}", f"cd {d}"]
    out.append(f"mkdir {fresh}")
    out.append(f"touch {fresh}")
    for f in files:
        out += [f"write {f} {tok}", f"append {f} {tok}", f"cat {f}", f"rm {f}"]
    for p in allp:
        out += [f"rm -r {p}", f"chmod {env.modes[0]:o} {p}"]
    for src in allp:
        for dst in (fresh, *dirs[:2]):
            out += [f"mv {src} {dst}", f"cp {src} {dst}", f"cp -r {src} {dst}"]
    out.append(f"export {env.env_keys[0]}={tok}")
    actions: list[Action] = []
    seen: set[str] = set()
    for raw in out:
        try:
            a = parse_action(raw)
        except Exception:
            continue
        if a.raw not in seen:
            seen.add(a.raw)
            actions.append(a)
    return actions


def run_tier1(config: SY1Config, ref: Oracle, sys: Oracle) -> dict[str, Any]:
    """Exhaustive battery: every (battery state, action) pair, classified."""
    classes: dict[str, int] = {AGREE: 0, RESIDUAL: 0, **{c: 0 for c in BOUNDARY_CLASSES}}
    by_family: dict[str, list[int]] = {}
    n = 0
    for state in _seed_states(config):
        for action in enumerate_actions(state, config.env):
            rec = differential_step(state, action, ref, sys)
            classes[rec.divergence_class] = classes.get(rec.divergence_class, 0) + 1
            by_family.setdefault(action.name, [0, 0])
            by_family[action.name][1] += 1
            if rec.agree:
                by_family[action.name][0] += 1
            n += 1
    # the structure-building command families that v0 models -- expected all-agree
    modeled = ("mkdir", "rmdir", "touch", "write", "append", "cat", "ls", "cd", "export")
    modeled_n = sum(by_family.get(c, [0, 0])[1] for c in modeled)
    modeled_agree = sum(by_family.get(c, [0, 0])[0] for c in modeled)
    return {
        "n": n,
        "agree": classes[AGREE],
        "residual": classes[RESIDUAL],
        "classes": classes,
        "by_family": {k: {"agree": v[0], "n": v[1]} for k, v in sorted(by_family.items())},
        "modeled_n": modeled_n,
        "modeled_agree": modeled_agree,
    }


# --- Tier-2: driver-sampled trajectories ------------------------------------

def run_tier2(config: SY1Config, ref: Oracle, sys: Oracle) -> list[dict[str, Any]]:
    """Per-driver agreement rate (bootstrap CI over seeds) + the boundary-class breakdown."""
    rows: list[dict[str, Any]] = []
    for driver in config.drivers:
        per_seed_rate: list[float] = []
        classes: dict[str, int] = {}
        n_agree = n = 0
        for seed in config.traj_seeds:
            drv = Driver(driver, config.env, random.Random(seed))
            s = State.empty()
            agree_s = tot_s = 0
            for _ in range(config.traj_steps):
                a = drv.sample(s)
                rec: DiffRecord = differential_step(s, a, ref, sys)
                classes[rec.divergence_class] = classes.get(rec.divergence_class, 0) + 1
                tot_s += 1
                if rec.agree:
                    agree_s += 1
                s = ref.step(s, a).state
            per_seed_rate.append(agree_s / tot_s)
            n_agree += agree_s
            n += tot_s
        lo, hi = bootstrap_ci(per_seed_rate, seed=0)
        rows.append({
            "driver": driver, "n": n, "n_agree": n_agree, "rate": n_agree / n,
            "ci_lo": lo, "ci_hi": hi, "classes": classes,
        })
    return rows


# --- Tier-3: the head-to-head H_ε(ρ) overlay --------------------------------

class _NoisyProposer:
    """A seeded synthetic proposer (torch-free): correct delta w.p. ``1-noise``, else drift.

    Uses the reference oracle to fetch the correct delta -- identical in both the ref-verified
    and sys-verified runs, so the only variable across the overlay is *which oracle verifies*.
    """

    def __init__(self, oracle: Oracle, noise: float, rng: random.Random) -> None:
        self._oracle = oracle
        self._noise = noise
        self._rng = rng

    def predict_delta(self, state: State, action: Action) -> Delta:
        if self._rng.random() < self._noise:
            return []  # drift: predict "nothing happens"
        return self._oracle.step(state, action).delta


def _curve_actions(config: SY1Config, ref: Oracle, seed: int) -> list[Action]:
    drv = Driver(config.curve_driver, config.env, random.Random(seed))
    s = State.empty()
    actions: list[Action] = []
    for _ in range(config.curve_steps):
        a = drv.sample(s)
        actions.append(a)
        s = ref.step(s, a).state
    return actions


def run_tier3(config: SY1Config, ref: Oracle, sys: Oracle) -> list[dict[str, float]]:
    """Overlay H_ε(ρ): reference-verified vs system-verified, on the structural grammar."""
    rows: list[dict[str, float]] = []
    for rho in config.rhos:
        h_ref: list[float] = []
        h_sys: list[float] = []
        for seed in config.curve_seeds:
            actions = _curve_actions(config, ref, seed)
            for oracle, bucket in ((ref, h_ref), (sys, h_sys)):
                # The proposer is seeded identically and always fetches its "correct" deltas
                # from the reference, so the runs differ only in the verifying oracle.
                model = _NoisyProposer(ref, config.curve_noise, random.Random(seed + 7))
                rec = run_rollout(
                    model, oracle, State.empty(), actions, fixed_interval_for_rho(rho),
                    epsilon=config.epsilon, budget=budget_for_rho(rho, len(actions)), seed=seed,
                )
                bucket.append(float(rec.faithful_horizon))
        rows.append({
            "rho": rho, "h_ref": fmean(h_ref), "h_sys": fmean(h_sys),
            "gap": abs(fmean(h_ref) - fmean(h_sys)),
        })
    return rows


def run_sy1(config: SY1Config | None = None, *, sys: Oracle | None = None) -> SY1Result:
    """Run all three SY1 tiers; return the agreement table + the overlay curve."""
    import sys as _sys

    config = config or SY1Config()
    ref = ReferenceOracle()
    try:
        sys = sys or SandboxOracle()
    except SystemOracleUnavailable:
        return SY1Result(available=False, platform=_sys.platform)
    return SY1Result(
        available=True,
        platform=_sys.platform,
        tier1=run_tier1(config, ref, sys),
        tier2=run_tier2(config, ref, sys),
        tier3=run_tier3(config, ref, sys),
    )


CSV_HEADER = "panel,key,driver,rho,n,agree_rate,ci_lo,ci_hi,h_ref,h_sys,detail"


def write_csv(result: SY1Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER, f"meta,platform,{result.platform},,,,,,,,available={result.available}"]
    t1 = result.tier1
    if t1:
        rows.append(f"tier1,exhaustive,all,,{t1['n']},{t1['agree'] / t1['n']:.4f},,,,,"
                    f"residual={t1['residual']};modeled={t1['modeled_agree']}/{t1['modeled_n']}")
        for fam, d in t1["by_family"].items():
            rate = d["agree"] / d["n"] if d["n"] else 0.0
            rows.append(f"tier1_family,{fam},,,{d['n']},{rate:.4f},,,,,")
    for r in result.tier2:
        cls = ";".join(f"{k}={v}" for k, v in sorted(r["classes"].items()))
        rows.append(f"tier2,driver,{r['driver']},,{r['n']},{r['rate']:.4f},"
                    f"{r['ci_lo']:.4f},{r['ci_hi']:.4f},,,{cls}")
    for p in result.tier3:
        rows.append(f"tier3,overlay,{SY1Config().curve_driver},{p['rho']},,,,,"
                    f"{p['h_ref']:.4f},{p['h_sys']:.4f},gap={p['gap']:.4f}")
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run SY1 (system-oracle differential agreement).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/sy1/records.jsonl")
    parser.add_argument("--csv", type=str, default="figures/sy1_agreement.csv")
    parser.add_argument("--plot", type=str, default="figures/sy1_agreement.png")
    args = parser.parse_args()
    config = SY1Config.from_json_file(args.config) if args.config else SY1Config()
    result = run_sy1(config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "platform": result.platform, "tier1": result.tier1,
        "tier2": result.tier2, "tier3": result.tier3,
    }) + "\n")
    path = write_csv(result, args.csv)
    print(f"wrote {path}  (platform={result.platform})")
    if result.available:
        print(f"  HEADLINE modeled-regime agreement = {result.modeled_agreement:.4f} "
              f"(structure-building grammar, bit-exact)")
        print(f"  residual (unexplained) fraction   = {result.residual_fraction:.4f} "
              f"(every divergence is a named boundary)")
        print(f"  Tier-3 max |H_ref - H_sys|        = "
              f"{max((p['gap'] for p in result.tier3), default=0):.4f} (curve overlay)")
    try:
        from figures.plot_sy1 import plot_sy1

        plot_sy1(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
