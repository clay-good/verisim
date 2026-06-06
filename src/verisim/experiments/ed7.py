"""ED7 -- the Tier-B system-oracle differential (SPEC-7 §5.2, DS8; the distributed W1 retirement).

The distributed analogue of the host's SY1 (SPEC-11 §3, §5): the figure that retires SPEC-3 wall
**W1** ("the oracle is a *model*, not reality") for the distributed world. Every distributed result
to date (H8/H17-H21/H5/H12) proves the loop against Tier-A -- a *single-threaded analytic
discrete-event simulator* that computes the next cluster state in closed form. ED7 validates, bit
for bit, that Tier-A's analytic shortcut equals what a **real distributed execution** produces: an
independent set of autonomous **node actors** (:class:`~verisim.distoracle.system.SystemDistOracle`)
that hold only their own replicas, exchange real replication messages, and converge under a
**seed-shuffled delivery order** -- so agreement certifies the property the analytic DES quietly
assumes, that the eventual-consistency convergence is **delivery-order-independent** (LWW by
``(version, value)`` is a commutative join).

Four coverage tiers mirror the SY1 discipline:

  - **Tier-1 -- exhaustive over the grammar.** Enumerate the full DS0 action space (every
    ``put``/``get``/``cas``/``advance``/``partition``/``heal``/``crash``/``restart`` over a battery
    of fault-rich states) and classify every ``(state, action)`` pair. The headline: agreement is
    **bit-exact 1.000** on the observable cluster, and the residual (unexplained-divergence)
    fraction is **zero**.
  - **Tier-2 -- driver-sampled at depth.** Step Tier-A and Tier-B in lockstep along the three
    workload drivers (``uniform``/``contention``/``adversarial``), bootstrap-CI over seeds, so
    agreement is measured along realistic multi-step trajectories including the fault-heavy regime
    where the in-flight medium is largest.
  - **Tier-3 -- the curve, head to head.** Re-run the prime-directive ``H_ε(ρ)`` with Tier-B
    substituted for Tier-A as the ground-truth/correction oracle and overlay the two curves. Because
    the oracles agree bit-exactly, the curves are *indistinguishable* -- the figure that shows the
    oracle substitution is transparent.
  - **Negative control (teeth, the SY3 analog).** A deliberately-broken actor that adopts deliveries
    **by arrival order** (ignoring the LWW version compare) is order-*dependent*; under the shuffled
    delivery order it disagrees with Tier-A, and the differential **catches** it as the
    ``delivery_order`` boundary -- proving the harness can detect a faithfulness break, not merely
    rubber-stamp an identical reimplementation.

A fifth, disclosed reality attestation: the same Tier-1 battery is re-run on the **threaded** tier
(actors on real OS threads + real :class:`queue.Queue` inboxes), and its availability + agreement
are reported -- the strongest "reality" claim (the distributed echo of a real ``/bin/sh`` over a
real kernel), probed and disclosed, never assumed. Dependency-free and GPU-free: the synthetic
proposer is seeded and torch-free, so all of ED7 runs in seconds on CPU.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import DistAction, parse_dist_action
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.state import DistributedState
from verisim.distdata.drivers import DIST_DRIVERS, DistDriver
from verisim.distloop.model import DistNoisyModel
from verisim.distloop.runner import budget_for_rho, run_dist_rollout
from verisim.distoracle.base import DistOracle
from verisim.distoracle.differential import (
    AGREE,
    BOUNDARY_CLASSES,
    C_DELIVERY_ORDER,
    RESIDUAL,
    DistDiffRecord,
    dist_differential_step,
)
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import (
    TIER_SIMULATED,
    TIER_THREADED,
    SystemDistOracle,
    SystemDistOracleUnavailable,
)
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class ED7Config:
    name: str = "ed7"
    config: DistConfig = DEFAULT_DIST_CONFIG
    # Tier-1 exhaustive battery
    battery_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    battery_depth: int = 12  # adversarial-driver steps per seed state (build in-flight + faults)
    # Tier-2 trajectories
    traj_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    traj_steps: int = 40
    drivers: tuple[str, ...] = DIST_DRIVERS
    # Tier-3 overlay
    curve_driver: str = "contention"
    curve_seeds: tuple[int, ...] = (200, 201, 202, 203)
    curve_steps: int = 32
    curve_noise: float = 0.35
    rhos: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    epsilon: float = 0.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED7Config:
        b = ED7Config()
        return ED7Config(
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
    def from_json_file(path: str | Path) -> ED7Config:
        return ED7Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED7Result:
    platform: str = ""
    tier1: dict[str, Any] = field(default_factory=dict)  # exhaustive battery summary
    tier2: list[dict[str, Any]] = field(default_factory=list)  # per-driver agreement + classes
    tier3: list[dict[str, float]] = field(default_factory=list)  # per-rho Tier-A vs Tier-B horizon
    negative_control: dict[str, Any] = field(default_factory=dict)  # broken-actor catch rate
    threaded: dict[str, Any] = field(default_factory=dict)  # real-OS-thread reality attestation

    @property
    def overall_agreement(self) -> float:
        """The headline: bit-exact observable-cluster agreement across Tier-1 + Tier-2."""
        n = self.tier1.get("n", 0) + sum(r["n"] for r in self.tier2)
        agree = self.tier1.get("agree", 0) + sum(r["n_agree"] for r in self.tier2)
        return agree / n if n else float("nan")

    @property
    def residual_fraction(self) -> float:
        """Fraction of *all* transitions whose disagreement is unexplained (should be 0)."""
        n = self.tier1.get("n", 0) + sum(r["n"] for r in self.tier2)
        resid = self.tier1.get("classes", {}).get(RESIDUAL, 0) + sum(
            r["classes"].get(RESIDUAL, 0) for r in self.tier2
        )
        return resid / n if n else float("nan")


# --- battery + action enumeration -----------------------------------------------------------------

def _battery_states(config: ED7Config) -> list[DistributedState]:
    """Fault-rich battery states from the adversarial driver (in-flight, partitions, crashes)."""
    states: list[DistributedState] = []
    ref = ReferenceDistOracle(config.config)
    for seed in config.battery_seeds:
        drv = DistDriver("adversarial", config.config, random.Random(seed))
        s = DistributedState.initial(config.config)
        for _ in range(config.battery_depth):
            s = ref.step(s, drv.sample(s)).state
        states.append(s)
    return states


def enumerate_dist_actions(state: DistributedState, config: DistConfig) -> list[DistAction]:
    """Every applicable DS0 action over ``state``: each family x representative args.

    Covers a write/read/cas at each up node holding each object, the time engine (``advance``), and
    the full fault medium (``partition`` of every node from the rest, ``heal``,
    ``crash``/``restart`` of each node) -- the exhaustive action space over a single cluster state.
    """
    raws: list[str] = []
    up_nodes = [n for n in config.nodes if state.is_up(n)]
    for node in up_nodes:
        for obj in config.objects:
            raws.append(f"put {node} {obj} {config.values[0]}")
            raws.append(f"get {node} {obj}")
            cur = state.replicas.get((obj, node))
            old = cur.value if cur is not None else config.default_value
            raws.append(f"cas {node} {obj} {old} {config.values[1]}")
            raws.append(f"cas {node} {obj} {config.values[2]} {config.values[1]}")  # conflict
    raws += ["advance 1", "advance 2"]
    for node in config.nodes:
        rest = [n for n in config.nodes if n != node]
        if rest:
            raws.append(f"partition {node} | {' '.join(rest)}")
        raws.append(f"crash {node}")
        raws.append(f"restart {node}")
    raws.append("heal")
    actions: list[DistAction] = []
    seen: set[str] = set()
    for raw in raws:
        try:
            a = parse_dist_action(raw)
        except Exception:
            continue
        if a.raw not in seen:
            seen.add(a.raw)
            actions.append(a)
    return actions


# --- Tier-1: exhaustive enumeration ---------------------------------------------------------------

def run_tier1(config: ED7Config, ref: DistOracle, sys: DistOracle) -> dict[str, Any]:
    classes: dict[str, int] = {AGREE: 0, RESIDUAL: 0, **{c: 0 for c in BOUNDARY_CLASSES}}
    by_family: dict[str, list[int]] = {}
    n = 0
    for state in _battery_states(config):
        for action in enumerate_dist_actions(state, config.config):
            rec = dist_differential_step(state, action, ref, sys)
            classes[rec.divergence_class] = classes.get(rec.divergence_class, 0) + 1
            by_family.setdefault(action.name, [0, 0])
            by_family[action.name][1] += 1
            if rec.agree:
                by_family[action.name][0] += 1
            n += 1
    return {
        "n": n,
        "agree": classes[AGREE],
        "classes": classes,
        "by_family": {k: {"agree": v[0], "n": v[1]} for k, v in sorted(by_family.items())},
    }


# --- Tier-2: driver-sampled trajectories ----------------------------------------------------------

def run_tier2(config: ED7Config, ref: DistOracle, sys: DistOracle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for driver in config.drivers:
        per_seed_rate: list[float] = []
        classes: dict[str, int] = {}
        n_agree = n = 0
        for seed in config.traj_seeds:
            drv = DistDriver(driver, config.config, random.Random(seed))
            s = DistributedState.initial(config.config)
            agree_s = tot_s = 0
            for _ in range(config.traj_steps):
                a = drv.sample(s)
                rec: DistDiffRecord = dist_differential_step(s, a, ref, sys)
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


# --- Tier-3: the head-to-head H_ε(ρ) overlay ------------------------------------------------------

def _curve_actions(config: ED7Config, ref: DistOracle, seed: int) -> list[DistAction]:
    drv = DistDriver(config.curve_driver, config.config, random.Random(seed))
    s = DistributedState.initial(config.config)
    actions: list[DistAction] = []
    for _ in range(config.curve_steps):
        a = drv.sample(s)
        actions.append(a)
        s = ref.step(s, a).state
    return actions


def run_tier3(config: ED7Config, ref: DistOracle, sys: DistOracle) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    s0 = DistributedState.initial(config.config)
    for rho in config.rhos:
        h_ref: list[float] = []
        h_sys: list[float] = []
        for seed in config.curve_seeds:
            actions = _curve_actions(config, ref, seed)
            for oracle, bucket in ((ref, h_ref), (sys, h_sys)):
                # The proposer always fetches its deltas from Tier-A, so the runs differ ONLY in
                # which oracle supplies ground truth + corrections (exactly the SY1 design).
                model = DistNoisyModel(
                    ref, noise=config.curve_noise, mode="gross",
                    rng=random.Random(seed + 7),
                )
                rec = run_dist_rollout(
                    model, oracle, s0, actions, fixed_interval_for_rho(rho),
                    epsilon=config.epsilon, config=config.config,
                    budget=budget_for_rho(rho, len(actions)), seed=seed,
                )
                bucket.append(float(rec.faithful_horizon))
        rows.append({
            "rho": rho, "h_ref": fmean(h_ref), "h_sys": fmean(h_sys),
            "gap": abs(fmean(h_ref) - fmean(h_sys)),
        })
    return rows


# --- negative control + threaded attestation ------------------------------------------------------

def run_negative_control(config: ED7Config, ref: DistOracle) -> dict[str, Any]:
    """A broken-arrival Tier-B (order-dependent): the differential must CATCH it (SY3 analog)."""
    broken = SystemDistOracle(config.config, broken_arrival=True)
    classes: dict[str, int] = {}
    n_agree = n = 0
    for seed in config.traj_seeds:
        drv = DistDriver("adversarial", config.config, random.Random(seed))
        s = DistributedState.initial(config.config)
        for _ in range(config.traj_steps):
            a = drv.sample(s)
            rec = dist_differential_step(s, a, ref, broken)
            classes[rec.divergence_class] = classes.get(rec.divergence_class, 0) + 1
            n += 1
            if rec.agree:
                n_agree += 1
            s = ref.step(s, a).state
    caught = classes.get(C_DELIVERY_ORDER, 0)
    return {
        "n": n, "n_agree": n_agree, "n_caught": caught, "classes": classes,
        "detects_break": caught > 0,
    }


def run_threaded_attestation(config: ED7Config, ref: DistOracle) -> dict[str, Any]:
    """Re-run the Tier-1 battery on real OS threads; report availability + agreement (disclosed)."""
    try:
        threaded = SystemDistOracle(config.config, tier=TIER_THREADED)
    except SystemDistOracleUnavailable as exc:
        return {"available": False, "reason": str(exc)}
    t1 = run_tier1(config, ref, threaded)
    return {
        "available": True, "tier": TIER_THREADED, "n": t1["n"], "agree": t1["agree"],
        "rate": t1["agree"] / t1["n"] if t1["n"] else float("nan"),
    }


def run_ed7(config: ED7Config | None = None) -> ED7Result:
    """Run all four ED7 tiers + the negative control + the threaded attestation."""
    import sys as _sys

    config = config or ED7Config()
    ref = ReferenceDistOracle(config.config)
    sys_oracle = SystemDistOracle(config.config, tier=TIER_SIMULATED)
    return ED7Result(
        platform=_sys.platform,
        tier1=run_tier1(config, ref, sys_oracle),
        tier2=run_tier2(config, ref, sys_oracle),
        tier3=run_tier3(config, ref, sys_oracle),
        negative_control=run_negative_control(config, ref),
        threaded=run_threaded_attestation(config, ref),
    )


CSV_HEADER = "panel,key,driver,rho,n,agree_rate,ci_lo,ci_hi,h_ref,h_sys,detail"


def write_csv(result: ED7Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        CSV_HEADER,
        f"meta,platform,{result.platform},,,,,,,,overall={result.overall_agreement:.4f}",
    ]
    t1 = result.tier1
    if t1:
        rate = t1["agree"] / t1["n"] if t1["n"] else 0.0
        cls = ";".join(f"{k}={v}" for k, v in sorted(t1["classes"].items()))
        rows.append(f"tier1,exhaustive,all,,{t1['n']},{rate:.4f},,,,,{cls}")
        for fam, d in t1["by_family"].items():
            fr = d["agree"] / d["n"] if d["n"] else 0.0
            rows.append(f"tier1_family,{fam},,,{d['n']},{fr:.4f},,,,,")
    for r in result.tier2:
        cls = ";".join(f"{k}={v}" for k, v in sorted(r["classes"].items()))
        rows.append(f"tier2,driver,{r['driver']},,{r['n']},{r['rate']:.4f},"
                    f"{r['ci_lo']:.4f},{r['ci_hi']:.4f},,,{cls}")
    for p in result.tier3:
        rows.append(f"tier3,overlay,{ED7Config().curve_driver},{p['rho']},,,,,"
                    f"{p['h_ref']:.4f},{p['h_sys']:.4f},gap={p['gap']:.4f}")
    nc = result.negative_control
    if nc:
        rows.append(f"negctl,broken_arrival,adversarial,,{nc['n']},"
                    f"{nc['n_agree'] / nc['n']:.4f},,,,,caught={nc['n_caught']};"
                    f"detects_break={nc['detects_break']}")
    th = result.threaded
    if th.get("available"):
        rows.append(f"threaded,real_os_threads,,,{th['n']},{th['rate']:.4f},,,,,available=True")
    else:
        rows.append("threaded,real_os_threads,,,,,,,,,available=False")
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED7 (the Tier-B system-oracle differential; distributed W1 retirement)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed7.csv")
    parser.add_argument("--plot", type=str, default="figures/ed7.png")
    args = parser.parse_args()
    config = ED7Config.from_json_file(args.config) if args.config else ED7Config()
    result = run_ed7(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}  (platform={result.platform})")
    print(f"  HEADLINE overall agreement     = {result.overall_agreement:.4f} "
          f"(observable cluster, bit-exact)")
    print(f"  residual (unexplained) fraction= {result.residual_fraction:.4f} "
          f"(every divergence is a named boundary)")
    print(f"  Tier-3 max |H_A - H_B|         = "
          f"{max((p['gap'] for p in result.tier3), default=0):.4f} (curve overlay)")
    nc = result.negative_control
    print(f"  negative control (broken actor): caught {nc['n_caught']} delivery-order breaks "
          f"-> detects_break={nc['detects_break']}")
    th = result.threaded
    th_status = f"agreement {th['rate']:.4f}" if th.get("available") else "unavailable (disclosed)"
    print(f"  threaded (real OS threads)     = {th_status}")
    try:
        from figures.plot_ed7 import plot_ed7

        plot_ed7(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
