"""ED21 — clock skew: a per-node timing shift that convergence is immune to.

The DS0-increment-14 experiment for SPEC-7 §3.2 / §3.4 — the `clock_skew` fault, the **last** of the
§3.4 medium faults ("partition, crash, message loss, reorder, **clock skew**"), now complete.
`clock_skew node delta` offsets a node's local clock by a signed `delta`, which shifts the
`deliver_after` it stamps on every message it sends (via `DistributedState.sender_clock`) — a
positive offset (a clock running ahead) defers the node's sends, a negative one (behind) rushes
them. It adds one omitted-when-empty `skew` map (no per-message state), so a synchronized cluster is
byte-identical to the pre-increment-14 form. Two findings, dependency-free, on a 3-node cluster
where every node replicates every object:

  - **Panel A — skew is a persistent, per-node timing shift.** A node skewed by `delta` stamps its
    replication messages with `deliver_after` shifted by exactly `delta` (every send, not one
    message — a persistent property of the node, unlike the one-shot `delay`). So a peer converges
    only once the global clock passes that shifted time: under a short `advance`, a negatively/zero
    skewed node's write has already landed but a positively-skewed (ahead) node's write is still in
    flight — clock skew deferring delivery exactly as a persistent per-node delay would.

  - **Panel B — convergence is clock-independent (the version-LWW immunity).** Sweep the writer's
    skew across a wide range and run the same write+`advance` script; the **converged state is
    byte-identical** at every skew (invariance rate **1.0**). Because the protocol resolves
    conflicts by last-writer-wins on `(version, value)` — never on a wall-clock timestamp — a node's
    wrong
    clock shifts *when* its write is delivered but never *which* write wins. This is the exact
    property deterministic-simulation testing (FoundationDB, madsim) injects clock skew to verify: a
    correct replicated store does not secretly depend on synchronized clocks. (A timestamp-LWW store
    would diverge here — a fast-clock node's stale write could win; version-LWW cannot be fooled.)

Both panels confirm **Tier-B** agreement: clock skew is a medium change (a per-node offset on the
send timestamp), so the autonomous-actor system oracle computes a byte-identical skew via the shared
helper and then, delivering on its own seed-shuffled schedule, reproduces the shifted-but-invariant
convergence bit-for-bit (the W1 retirement, §5.2). The `clock_skew` action is purely additive — no
prior golden, hash, or tokenization changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.serialize import to_canonical
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


@dataclass(frozen=True)
class ED21Config:
    name: str = "ed21-clock-skew"
    nodes: tuple[str, ...] = ("n0", "n1", "n2")
    key: str = "x"
    value: str = "b"
    skews: tuple[int, ...] = (-4, -2, 0, 2, 4)  # the per-node clock offsets swept
    short_advance: int = 2  # Panel A: a short advance — defers a positively-skewed send

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED21Config:
        b = ED21Config()
        return ED21Config(
            name=d.get("name", b.name),
            nodes=tuple(d.get("nodes", b.nodes)),
            key=d.get("key", b.key),
            value=d.get("value", b.value),
            skews=tuple(d.get("skews", b.skews)),
            short_advance=int(d.get("short_advance", b.short_advance)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED21Config:
        return ED21Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED21Result:
    #: Panel A: per skew, the deliver_after the node stamped + whether a peer converged after a
    #: short advance (a positive skew defers delivery).
    timing: list[dict[str, Any]] = field(default_factory=list)
    #: Panel A: the shift is by exactly `delta` for every send (persistent, not one-shot).
    shift_equals_delta: bool = True
    #: Panel B: fraction of swept skews whose converged state matches the zero-skew baseline.
    converged_invariant_rate: float = 0.0
    n_skews: int = 0
    #: Tier-B reproduces the skewed send + delivery bit-for-bit over every scenario.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0


def _config(cfg: ED21Config) -> DistConfig:
    return DistConfig(
        name=cfg.name, nodes=cfg.nodes, objects=(cfg.key, "y"),
        values=("a", cfg.value, "c", "d"), replication_factor=len(cfg.nodes),
    )


def _replica(state: DistributedState, key: str, node: str) -> str:
    r = state.replicas.get((key, node))
    return r.value if r is not None else ""


def run_ed21(cfg: ED21Config | None = None) -> ED21Result:
    cfg = cfg or ED21Config()
    config = _config(cfg)
    ref = ReferenceDistOracle(config)
    sysb = SystemDistOracle(config)
    coord = cfg.nodes[0]
    peer = cfg.nodes[1]
    result = ED21Result(n_skews=len(cfg.skews))
    tier_b_agree = True
    tier_b_steps = 0

    def run(script: list[str]) -> DistributedState:
        """Run a script through both oracles, checking Tier-B agreement at every step."""
        nonlocal tier_b_agree, tier_b_steps
        sa = sb = DistributedState.initial(config)
        for cmd in script:
            action = parse_dist_action(cmd)
            ra, rb = ref.step(sa, action), sysb.step(sb, action)
            if cluster_view(ra.state) != cluster_view(rb.state) or (ra.status, ra.value) != (
                rb.status, rb.value
            ):
                tier_b_agree = False
            tier_b_steps += 1
            sa, sb = ra.state, rb.state
        return sa

    # --- Panel A: skew shifts the send timing by exactly delta (persistent, per-node) -------------
    baseline = run([f"put {coord} {cfg.key} {cfg.value}"])
    base_da = min(
        (m.deliver_after for m in baseline.inflight.values() if m.src == coord and m.dst == peer),
        default=0,
    )
    shift_ok = True
    for skew in cfg.skews:
        after_skew = run([f"clock_skew {coord} {skew}", f"put {coord} {cfg.key} {cfg.value}"])
        msgs = [
            m for m in after_skew.inflight.values() if m.src == coord and m.dst == peer
        ]
        deliver_after = msgs[0].deliver_after if msgs else 0
        if deliver_after != base_da + skew:  # shifted by exactly the skew offset
            shift_ok = False
        # a short advance: a positively-skewed (deferred) send has not landed yet, others have.
        short = run([
            f"clock_skew {coord} {skew}",
            f"put {coord} {cfg.key} {cfg.value}",
            f"advance {cfg.short_advance}",
        ])
        converged = _replica(short, cfg.key, peer) == cfg.value
        result.timing.append({
            "skew": skew,
            "deliver_after": deliver_after,
            "converged_after_short": converged,
        })
    result.shift_equals_delta = shift_ok

    # --- Panel B: the converged state is invariant to skew (version-LWW clock-independence) -------
    far = max(abs(s) for s in cfg.skews) + 10
    invariant = 0
    base_final = to_canonical(run([
        f"put {coord} {cfg.key} {cfg.value}", f"advance {far}",
    ]))["replicas"]
    for skew in cfg.skews:
        final = to_canonical(run([
            f"clock_skew {coord} {skew}",
            f"put {coord} {cfg.key} {cfg.value}",
            f"advance {far}",
        ]))["replicas"]
        if final == base_final:
            invariant += 1
    result.converged_invariant_rate = invariant / len(cfg.skews) if cfg.skews else 0.0

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,skew,value,detail"


def write_csv(result: ED21Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for t in result.timing:
        lines.append(f"timing,{t['skew']},{t['deliver_after']},"
                     f"converged_after_short={t['converged_after_short']}")
    lines.append(f"shift,all,{1.0 if result.shift_equals_delta else 0.0:.4f},"
                 f"deliver_after_shifted_by_exactly_delta")
    lines.append(f"invariance,all,{result.converged_invariant_rate:.4f},"
                 f"converged_state_clock_independent")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps};skews={result.n_skews}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED21 (clock skew: a per-node timing shift convergence is immune to)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed21.csv")
    parser.add_argument("--plot", type=str, default="figures/ed21.png")
    args = parser.parse_args()
    cfg = ED21Config.from_json_file(args.config) if args.config else ED21Config()
    result = run_ed21(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — skew shifts the send timing (deliver_after) by exactly delta:")
    for t in result.timing:
        verdict = "delivered" if t["converged_after_short"] else "DEFERRED"
        print(f"    [skew {t['skew']:+d}] deliver_after {t['deliver_after']:+d} "
              f"→ after short advance: {verdict}")
    print(f"    shift == delta for every send (persistent): {result.shift_equals_delta}")
    print("  Panel B — convergence is clock-independent (version-LWW immunity):")
    verdict = "clock-independent" if result.converged_invariant_rate == 1.0 else "CLOCK-DEPENDENT!"
    print(f"    converged-state invariance over the skew sweep: "
          f"{result.converged_invariant_rate:.2f} → {verdict}")
    print(f"  Tier-B reproduces the skewed send + delivery bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed21 import plot_ed21

        plot_ed21(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
