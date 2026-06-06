"""ED12 -- partial observation: the probe-faithful horizon + crash/partition indistinguishability.

The DS3-increment-4 experiment for SPEC-7 §5.4 (partial observation & the probe oracle), on the
existing dependency-free apparatus (the DS0-increment-1 replicated-KV world + the synthetic
tunable-noise proposer :class:`~verisim.distloop.model.DistNoisyModel`). GPU-free; CI runs the smoke
instance. Two findings, both downstream of one fact: **no observer ever holds the global state.**

  - **Panel A -- the probe-faithful horizon outlasts the bit-faithful horizon, for in-flight
    errors.** A probe at any vantage reads replicas + reachability + the clock, but **never the
    in-flight replication medium** (:func:`~verisim.dist.observe.observe`). So a model that
    mispredicts only a message-in-transit (the ``subtle`` error class) is *observably faithful* --
    zero observable divergence even at the maximal whole-cluster vantage -- until ``advance``
    delivers the message and the error surfaces in a replica the probe can read. This is the
    partial-observation form of ED5/H19: where ED5 reads the gap through the *consistency view* (an
    abstraction that collapses node placement), ED12 reads it through *physical observability* (what
    a perfect monitor sees), and the gap is the same hidden medium. The ``gross`` (durable-replica)
    class is the control where the probe sees the corruption immediately, so the bit and observable
    horizons coincide. ED12 reports all three horizons -- bit, observable, consistency -- on the
    *same* free-running rollout (ρ=0); the structural half (a bit-faithful step is necessarily
    observably faithful, so the observable horizon dominates the bit horizon) is pinned as a
    property, not just asserted.

  - **Panel B -- crash and partition are indistinguishable from one vantage.** A node that is
    ``down`` and a node that is partitioned away project to the *same* ``unreachable`` fact -- the
    canonical failure-detector limit (the epistemic core of FLP). ED12 runs a battery: build a base
    trajectory, then either crash a node or isolate it, and ask whether a probe can tell which
    happened. From a **single external vantage** the two are
    :func:`~verisim.distmetrics.observe.observably_indistinguishable` (rate 1.0 -- the probe cannot
    localize the fault); from a **paired vantage** that reaches the node's side of the split the
    partition case exposes the (live, isolated) replica while the crash case does not, so the two
    separate (indistinguishable rate 0.0). One probe cannot tell a crash from a partition; a quorum
    of probes can -- the operational reason distributed failure detection needs more than one
    observer.

The committed sweep runs in milliseconds on CPU (no torch).
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
from verisim.dist.delta import apply
from verisim.dist.state import DistributedState
from verisim.distdata import DistDriver
from verisim.distloop import DistNoisyModel
from verisim.distmetrics.divergence import consistency_faithfulness, divergence
from verisim.distmetrics.observe import observable_divergence, observably_indistinguishable
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon

ERROR_MODES: tuple[str, ...] = ("gross", "subtle")


@dataclass(frozen=True)
class ED12Config:
    name: str = "ed12-dist"
    dist: DistConfig = DEFAULT_DIST_CONFIG
    driver: str = "contention"
    eval_seeds: tuple[int, ...] = (200, 201, 202, 203, 204, 205, 206, 207)
    n_steps: int = 40
    epsilon: float = 0.0
    #: Panel A -- the per-mode free-running noise (the model is exposed, not the loop).
    noise: float = 0.4
    modes: tuple[str, ...] = ERROR_MODES
    #: Panel B -- the indistinguishability battery: how many base trajectories, of what length.
    battery_seeds: tuple[int, ...] = (300, 301, 302, 303, 304, 305, 306, 307)
    battery_prefix: int = 6

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED12Config:
        b = ED12Config()
        return ED12Config(
            name=d.get("name", b.name),
            driver=d.get("driver", b.driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            n_steps=d.get("n_steps", b.n_steps),
            epsilon=d.get("epsilon", b.epsilon),
            noise=d.get("noise", b.noise),
            modes=tuple(d.get("modes", b.modes)),
            battery_seeds=tuple(d.get("battery_seeds", b.battery_seeds)),
            battery_prefix=d.get("battery_prefix", b.battery_prefix),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED12Config:
        return ED12Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED12Result:
    """The ED12 deliverables: the probe-horizon ordering (A) + indistinguishability (B)."""

    #: per mode: mean bit / observable / consistency free-running horizon, the obs-vs-bit gap, CIs.
    horizons: list[dict[str, Any]] = field(default_factory=list)
    #: the structural property: the observable horizon dominates the bit horizon on every rollout.
    observable_dominates_bit: bool = True
    #: Panel B: single-vantage indistinguishable rate (expect 1.0) + paired rate (expect 0.0).
    single_vantage_indistinguishable: float = 0.0
    paired_vantage_indistinguishable: float = 0.0
    battery_n: int = 0


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


def _three_horizons(
    oracle: DistOracle, cfg: ED12Config, mode: str, eval_seed: int
) -> tuple[int, int, int, bool]:
    """One free-running (ρ=0) rollout: ``(bit_h, observable_h, consistency_h, dominates)``.

    The model proposes every step and is *trusted* every step (ρ=0 exposes the model, not the
    loop), exactly as ED5's H19 arm does. The maximal whole-cluster vantage is the probe -- so the
    observable gap is purely the unobservable in-flight medium, not a vantage being too small.
    """
    vantage = frozenset(cfg.dist.nodes)
    s_truth = DistributedState.initial(cfg.dist)
    s_pred = DistributedState.initial(cfg.dist)
    actions = _eval_actions(oracle, cfg.dist, cfg.driver, eval_seed, cfg.n_steps)
    model = DistNoisyModel(oracle, noise=cfg.noise, mode=mode, rng=random.Random(eval_seed + 7))

    bit_div: list[float] = []
    obs_div: list[float] = []
    cons_div: list[float] = []
    dominates = True
    for action in actions:
        s_truth = oracle.step(s_truth, action).state
        s_pred = apply(s_pred, model.predict_delta(s_pred, action))
        b = divergence(s_pred, s_truth)
        o = observable_divergence(s_truth, s_pred, vantage)
        c = 1.0 - consistency_faithfulness(s_truth, s_pred)
        bit_div.append(b)
        obs_div.append(o)
        cons_div.append(c)
        # The rigorous structural claim (un-normalized, so denominator-independent): a bit-faithful
        # step is necessarily observably faithful, because bit-equal states are equal everywhere the
        # probe can read. Hence the observable horizon dominates the bit horizon on every rollout.
        # The consistency horizon is *empirically* the longest but not structurally ordered against
        # observable (a crashed node's replica is consistency-visible yet probe-invisible), so it is
        # reported, not claimed. At ε=0, faithful means divergence 0.
        if b <= cfg.epsilon and o > cfg.epsilon:
            dominates = False
    return (
        faithful_horizon(bit_div, cfg.epsilon),
        faithful_horizon(obs_div, cfg.epsilon),
        faithful_horizon(cons_div, cfg.epsilon),
        dominates,
    )


def _indistinguishability_battery(oracle: DistOracle, cfg: ED12Config) -> tuple[float, float, int]:
    """Panel B: crash vs partition, seen from one vantage vs a paired vantage.

    For each base trajectory we pick the last node ``dark`` of the cluster, build two end-states --
    one where ``dark`` is **crashed**, one where it is **partitioned** into its own group -- and ask
    a probe to tell them apart. The external vantage is the cluster's first node; the paired vantage
    adds ``dark`` itself, the observer that *can* tell a live-but-isolated replica from a dead one.
    """
    nodes = cfg.dist.nodes
    dark = nodes[-1]
    external = (nodes[0],)
    paired = (nodes[0], dark)
    single_hits = 0
    paired_hits = 0
    n = 0
    for seed in cfg.battery_seeds:
        base_actions = _eval_actions(oracle, cfg.dist, cfg.driver, seed, cfg.battery_prefix)
        base = DistributedState.initial(cfg.dist)
        for a in base_actions:
            base = oracle.step(base, a).state
        # Heal + bring ``dark`` up so the two branches differ in exactly one thing: why ``dark`` is
        # gone afterward. Without the heal a base partition among the other nodes would leak into
        # the comparison; without the restart a base trajectory that already crashed ``dark`` makes
        # "partition it" a no-op (partitioning a dead node changes nothing -- itself a true and even
        # stronger indistinguishability, but not the live-crash-vs-isolate contrast under test).
        base = oracle.step(base, parse_dist_action("heal")).state
        base = oracle.step(base, parse_dist_action(f"restart {dark}")).state
        others = tuple(node for node in nodes if node != dark)
        crashed = oracle.step(base, parse_dist_action(f"crash {dark}")).state
        partitioned = oracle.step(
            base, parse_dist_action("partition " + " ".join(others) + f" | {dark}")
        ).state
        single_hits += observably_indistinguishable(crashed, partitioned, external)
        paired_hits += observably_indistinguishable(crashed, partitioned, paired)
        n += 1
    return (single_hits / n if n else 0.0, paired_hits / n if n else 0.0, n)


def run_ed12(config: ED12Config | None = None, *, oracle: DistOracle | None = None) -> ED12Result:
    """Run ED12: the probe-faithful horizon ordering (A) + indistinguishability (B)."""
    config = config or ED12Config()
    oracle = oracle or ReferenceDistOracle(config.dist)
    result = ED12Result()

    # --- Panel A: bit vs observable vs consistency free-running horizon (ρ=0) --------------------
    for mode in config.modes:
        cells = [_three_horizons(oracle, config, mode, s) for s in config.eval_seeds]
        bit = [float(b) for b, _, _, _ in cells]
        obs = [float(o) for _, o, _, _ in cells]
        cons = [float(c) for _, _, c, _ in cells]
        if not all(ok for *_, ok in cells):
            result.observable_dominates_bit = False
        bit_lo, bit_hi = bootstrap_ci(bit, seed=0)
        obs_lo, obs_hi = bootstrap_ci(obs, seed=0)
        cons_lo, cons_hi = bootstrap_ci(cons, seed=0)
        gap = [o - b for b, o in zip(bit, obs, strict=True)]
        gap_lo, gap_hi = bootstrap_ci(gap, seed=0)
        result.horizons.append({
            "mode": mode,
            "bit_h": fmean(bit), "bit_lo": bit_lo, "bit_hi": bit_hi,
            "obs_h": fmean(obs), "obs_lo": obs_lo, "obs_hi": obs_hi,
            "cons_h": fmean(cons), "cons_lo": cons_lo, "cons_hi": cons_hi,
            "gap": fmean(gap), "gap_lo": gap_lo, "gap_hi": gap_hi,
            # the probe outlasts bytes for this mode iff observable materially outlasts bit (CI > 0)
            "observable_outlasts": gap_lo > 0.0,
        })

    # --- Panel B: crash/partition indistinguishability -------------------------------------------
    single, paired, n = _indistinguishability_battery(oracle, config)
    result.single_vantage_indistinguishable = single
    result.paired_vantage_indistinguishable = paired
    result.battery_n = n
    return result


CSV_HEADER = "panel,mode,bit_h,obs_h,cons_h,gap,gap_lo,gap_hi,single_indist,paired_indist"


def write_csv(result: ED12Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in result.horizons:
        rows.append(f"horizon,{r['mode']},{r['bit_h']:.4f},{r['obs_h']:.4f},{r['cons_h']:.4f},"
                    f"{r['gap']:.4f},{r['gap_lo']:.4f},{r['gap_hi']:.4f},,")
    rows.append(f"indist,,,,,,,,"
                f"{result.single_vantage_indistinguishable:.4f},"
                f"{result.paired_vantage_indistinguishable:.4f}")
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED12 (probe-faithful horizon + crash/partition indistinguishability)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed12.csv")
    parser.add_argument("--plot", type=str, default="figures/ed12.png")
    args = parser.parse_args()
    config = ED12Config.from_json_file(args.config) if args.config else ED12Config()
    result = run_ed12(config)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A (free-running bit vs observable vs consistency horizon):")
    for r in result.horizons:
        verdict = "probe OUTLASTS bytes" if r["observable_outlasts"] else "coincide (control)"
        print(f"    [{r['mode']:6s}] bit H={r['bit_h']:.1f}  obs H={r['obs_h']:.1f}  "
              f"cons H={r['cons_h']:.1f}  gap={r['gap']:.1f} "
              f"[{r['gap_lo']:.1f},{r['gap_hi']:.1f}] → {verdict}")
    print(f"    observable horizon dominates bit horizon on every rollout: "
          f"{result.observable_dominates_bit}")
    print(f"  Panel B (crash/partition indistinguishability, n={result.battery_n}):")
    print(f"    single external vantage: indistinguishable rate "
          f"{result.single_vantage_indistinguishable:.2f} (expect 1.0 — one probe can't localize)")
    print(f"    paired vantage:          indistinguishable rate "
          f"{result.paired_vantage_indistinguishable:.2f} (expect 0.0 — a quorum can)")
    try:
        from figures.plot_ed12 import plot_ed12

        plot_ed12(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
