"""ED13 -- causal consistency: the effect-before-cause anomaly, forbidden without over-syncing.

The DS0-increment-5 experiment for SPEC-7 §3.4 -- the third `CONSISTENCY_MODELS` end, **`causal`**,
between `eventual` (weakest) and `linearizable` (strongest). Causal consistency is "eventual
delivery plus a guarantee: **if write B causally depends on write A, no replica ever observes B
before A**" -- the cross-object ordering that keeps a defender/SRE from seeing an effect whose cause
is still invisible. It is implemented as a delivery-order refinement
([`reference.py`](../../src/verisim/distoracle/reference.py)): a write carries the versions its
origin node had already observed (`Message.deps`), and `advance` defers a message until the
destination has applied those dependencies -- a version-vector slice, deterministic and dep-free.

The mechanism (and the scenario both panels use). A partition is toggled across time to route the
*effect* to the observer while the *cause* is still blocked -- the only way to manufacture
out-of-causal-order delivery in a group-partition model, since disjoint groups are transitive at any
single instant:

    put n0 x a               # n0: x=a@1; replication messages x@1 -> n1, n2 enqueued
    partition n0 n1 | n2     # isolate the observer n2
    advance 1                # x@1 -> n1 delivers; x@1 -> n2 is blocked (n2 isolated from n0)
    put n1 y b               # n1 has now observed x@1, so its write y=b@1 causally depends on x@1
    partition n0 | n1 n2     # re-route: n1 and n2 together, n0 isolated
    advance 1                # y@1 -> n2 is reachable; x@1 -> n2 is still blocked

Two panels, dependency-free and GPU-free:

  - **Panel A -- the anomaly is forbidden.** In the scenario above n2 is about to learn the *effect*
    (y=b) while the *cause* (x=a) is still in flight. Under **`eventual`** the message is delivered
    greedily, so n2 reads ``y=b, x=nil`` -- an effect before its cause (anomaly rate **1.0**). Under
    **`causal`** the message carries ``deps={x@1}``, unmet at n2, so it is **held**: n2 reads
    ``y=nil, x=nil`` and the anomaly cannot occur (rate **0.0**). The battery sweeps object pairs.

  - **Panel B -- causal does not over-synchronize, and stays live.** A minimal refinement must (i)
    order only *causally-linked* writes, not independent ones, and (ii) still converge. The
    **independent control** writes ``y`` at n1 *before* n1 receives x@1, so y does **not** depend on
    x; under `causal` that message is **delivered freely** (it carries no deps), so independent
    writes keep their concurrency -- causal blocks the dependent message and *only* that message.
    And **convergence is preserved**: after a `heal` + `advance`, every scenario reaches the
    *identical* final cluster state under `eventual` and `causal` -- causal is a delivery-*order*
    refinement, not a different outcome, and the held message is delivered once its cause arrives.

The committed sweep runs in milliseconds on CPU (no torch).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig, scaled_dist_config
from verisim.dist.state import DistributedState
from verisim.distmetrics.divergence import divergence
from verisim.distoracle.reference import ReferenceDistOracle

CONSISTENCY_MODELS: tuple[str, ...] = ("eventual", "causal")


@dataclass(frozen=True)
class ED13Config:
    name: str = "ed13-causal"
    n_nodes: int = 3
    n_objects: int = 4
    #: the (cause, effect) object index pairs the battery sweeps (one scenario per model each).
    object_pairs: tuple[tuple[int, int], ...] = (
        (0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3),
    )

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED13Config:
        b = ED13Config()
        return ED13Config(
            name=d.get("name", b.name),
            n_nodes=d.get("n_nodes", b.n_nodes),
            n_objects=d.get("n_objects", b.n_objects),
            object_pairs=tuple(tuple(p) for p in d.get("object_pairs", b.object_pairs)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED13Config:
        return ED13Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED13Result:
    #: per model: the effect-before-cause anomaly rate over the dependent battery (A).
    anomaly: list[dict[str, Any]] = field(default_factory=list)
    #: causal does not block independent writes (B-i): held rate, dependent vs independent.
    over_sync: dict[str, Any] = field(default_factory=dict)
    #: convergence preserved (B-ii): eventual and causal reach the identical final state.
    convergence: dict[str, Any] = field(default_factory=dict)


def _config(cfg: ED13Config, model: str) -> DistConfig:
    return scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, consistency_model=model)


def _dependent_script(x: str, y: str) -> list[str]:
    """The effect-before-cause scenario: n1 observes x@1, *then* writes y (so y depends on x)."""
    return [
        f"put n0 {x} a",            # cause: x=a@1 at n0
        "partition n0 n1 | n2",     # isolate the observer n2
        "advance 1",                # x@1 -> n1 delivers; x@1 -> n2 blocked
        f"put n1 {y} b",            # n1 has seen x@1, so effect y=b@1 depends on x@1
        "partition n0 | n1 n2",     # re-route so y can reach n2 while x cannot
        "advance 1",                # y@1 -> n2 reachable; x@1 -> n2 still blocked
    ]


def _independent_script(x: str, y: str) -> list[str]:
    """The control: n1 writes y *before* receiving x@1, so y does not depend on x."""
    return [
        f"put n0 {x} a",            # cause: x=a@1 at n0
        f"put n1 {y} b",            # n1 writes y=b@1 BEFORE seeing x@1 -> no dependency on x
        "partition n0 | n1 n2",     # n1, n2 together; n0 isolated
        "advance 1",                # y@1 -> n2 reachable; x@1 -> n2 blocked (n0 isolated)
    ]


def _run(config: DistConfig, script: list[str]) -> DistributedState:
    oracle = ReferenceDistOracle(config)
    state = DistributedState.initial(config)
    for cmd in script:
        state = oracle.step(state, parse_dist_action(cmd)).state
    return state


def _sees_effect_without_cause(state: DistributedState, x: str, y: str, observer: str) -> bool:
    """``True`` iff ``observer`` has adopted the effect (y=b) but not its cause (x=a)."""
    xr = state.replicas.get((x, observer))
    yr = state.replicas.get((y, observer))
    return yr is not None and yr.value == "b" and (xr is None or xr.value != "a")


def run_ed13(cfg: ED13Config | None = None) -> ED13Result:
    cfg = cfg or ED13Config()
    result = ED13Result()

    # --- Panel A: the effect-before-cause anomaly over the dependent battery ----------------------
    for model in CONSISTENCY_MODELS:
        config = _config(cfg, model)
        anomalies = 0
        for xi, yi in cfg.object_pairs:
            state = _run(config, _dependent_script(f"o{xi}", f"o{yi}"))
            if _sees_effect_without_cause(state, f"o{xi}", f"o{yi}", "n2"):
                anomalies += 1
        n = len(cfg.object_pairs)
        result.anomaly.append({
            "model": model,
            "scenarios": n,
            "anomalies": anomalies,
            "anomaly_rate": anomalies / n if n else 0.0,
        })

    # --- Panel B-i: causal blocks the dependent message but NOT the independent one ---------------
    causal = _config(cfg, "causal")
    dep_held = 0
    indep_held = 0
    for xi, yi in cfg.object_pairs:
        dep = _run(causal, _dependent_script(f"o{xi}", f"o{yi}"))
        indep = _run(causal, _independent_script(f"o{xi}", f"o{yi}"))
        # the y@1 -> n2 message is held under causal iff n2 has not adopted y=b
        if dep.replicas[(f"o{yi}", "n2")].value != "b":
            dep_held += 1
        if indep.replicas[(f"o{yi}", "n2")].value != "b":
            indep_held += 1
    n = len(cfg.object_pairs)
    result.over_sync = {
        "dependent_held_rate": dep_held / n if n else 0.0,       # expect 1.0: dependent is ordered
        "independent_held_rate": indep_held / n if n else 0.0,   # expect 0.0: independent is free
        "scenarios": n,
    }

    # --- Panel B-ii: convergence preserved (eventual and causal reach the identical durable state)
    # We compare the *durable live cluster state* (replicas + in-flight + partition/crash/clock, the
    # `divergence` fact-set) rather than the raw serialization: the two legitimately differ only in
    # the transient `last_result` (causal's final `advance` delivers one more message -- the one it
    # earlier deferred -- so its delivered-count is higher), not part of the converged state.
    ev = _config(cfg, "eventual")
    identical = 0
    causal_inflight_after_heal = 0
    for xi, yi in cfg.object_pairs:
        script = [*_dependent_script(f"o{xi}", f"o{yi}"), "heal", "advance 5"]
        ev_state = _run(ev, script)
        ca_state = _run(causal, script)
        if divergence(ev_state, ca_state) == 0.0:
            identical += 1
        causal_inflight_after_heal += len(ca_state.inflight)
    result.convergence = {
        "identical_final_state_rate": identical / n if n else 0.0,  # expect 1.0
        "causal_inflight_after_heal": causal_inflight_after_heal,    # expect 0: all deps arrived
        "scenarios": n,
    }
    return result


CSV_HEADER = "panel,model,metric,value,detail"


def write_csv(result: ED13Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.anomaly:
        lines.append(f"anomaly,{r['model']},anomaly_rate,{r['anomaly_rate']:.4f},"
                     f"{r['anomalies']}/{r['scenarios']}")
    o = result.over_sync
    lines.append(f"over_sync,causal,dependent_held_rate,{o['dependent_held_rate']:.4f},"
                 f"of {o['scenarios']}")
    lines.append(f"over_sync,causal,independent_held_rate,{o['independent_held_rate']:.4f},"
                 f"of {o['scenarios']}")
    c = result.convergence
    lines.append(f"convergence,both,identical_final_state_rate,{c['identical_final_state_rate']:.4f},"
                 f"causal_inflight_after_heal={c['causal_inflight_after_heal']}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED13 (causal consistency: the effect-before-cause anomaly)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed13.csv")
    parser.add_argument("--plot", type=str, default="figures/ed13.png")
    args = parser.parse_args()
    cfg = ED13Config.from_json_file(args.config) if args.config else ED13Config()
    result = run_ed13(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — effect-before-cause anomaly rate (observer n2 sees y=b but x=nil):")
    for r in result.anomaly:
        verdict = "ANOMALY admitted" if r["anomaly_rate"] > 0 else "anomaly forbidden"
        print(f"    [{r['model']:10s}] {r['anomaly_rate']:.2f} "
              f"({r['anomalies']}/{r['scenarios']}) → {verdict}")
    o = result.over_sync
    print(f"  Panel B-i — causal blocks only causally-linked writes (n={o['scenarios']}):")
    print(f"    dependent   message held: {o['dependent_held_rate']:.2f} (expect 1.0 — ordered)")
    print(f"    independent msg held: {o['independent_held_rate']:.2f} (expect 0.0, concurrent)")
    c = result.convergence
    print(f"  Panel B-ii — convergence preserved (n={c['scenarios']}):")
    print(f"    eventual ≡ causal final state after heal: "
          f"{c['identical_final_state_rate']:.2f} (expect 1.0); "
          f"causal in-flight after heal={c['causal_inflight_after_heal']} (expect 0)")
    try:
        from figures.plot_ed13 import plot_ed13

        plot_ed13(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
