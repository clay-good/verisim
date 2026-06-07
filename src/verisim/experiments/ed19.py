"""ED19 — anti-entropy / read-repair: convergence restored after message loss.

The DS0-increment-12 experiment for SPEC-7 §3.2 / §5.1 — the ``anti_entropy`` protocol op, the
**read-repair** mechanism real eventually-consistent stores (Dynamo, Cassandra) use to converge
*despite* lost messages, and the SPEC-7 §4 ``ReplicaConverge`` op the spec named but had not
implemented. ``anti_entropy node`` pulls each object to the winning ``(version, value)`` among the
node's **reachable** replicas. It is the counterpart to ED18's ``drop``: where message loss *breaks*
the eventual-consistency convergence guarantee, anti-entropy is how a real system *restores* it
without a fresh write. Two findings, dependency-free, on a 3-node cluster where every node holds
every object:

  - **Panel A — anti-entropy repairs a dropped write; advance cannot.** Drop a write's replication
    message to one peer, then heal: the peer is permanently stale (ED18). Under **advance-only** the
    stale replica never recovers (rate **0.0**) — there is no in-flight message left to deliver.
    Under **anti_entropy** the peer pulls the latest value directly from its now-reachable replicas
    and recovers (rate **1.0**). Read-repair converges what message loss broke, with no new write —
    the mechanism that makes "eventual consistency" actually eventual under an unreliable network.

  - **Panel B — anti-entropy is bounded by reachability.** The *same* repair op succeeds or fails on
    nothing but connectivity. With the stale peer **partitioned away**, ``anti_entropy`` reaches
    only its own side, so it cannot pull the value held across the split (repair rate **0.0**); once
    the partition **heals**, the same op reaches every replica and repairs (rate **1.0**).
    Anti-entropy converges only the reachable set — it is gossip, not magic, so full convergence
    still needs the network to come back.

Both panels confirm the **Tier-B** agreement: the read-repair is a coordinator-level reconciliation
whose reachable set is read from the medium, so the autonomous-actor system oracle computes a
byte-identical repair and reaches the same converged (or still-partitioned) cluster bit-for-bit (the
W1 retirement, §5.2). ``anti_entropy`` is purely additive (it reuses the ``ReplicaWrite`` edit and
adds no state field), so every prior golden/hash/tokenization is unchanged, and it composes with
every consistency model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


@dataclass(frozen=True)
class ED19Config:
    name: str = "ed19-anti-entropy"
    nodes: tuple[str, ...] = ("n0", "n1", "n2")
    key: str = "x"
    v1: str = "b"  # the write that gets dropped to a peer

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED19Config:
        b = ED19Config()
        return ED19Config(
            name=d.get("name", b.name),
            nodes=tuple(d.get("nodes", b.nodes)),
            key=d.get("key", b.key),
            v1=d.get("v1", b.v1),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED19Config:
        return ED19Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED19Result:
    #: Panel A: per repair regime (advance_only | anti_entropy), the post-heal convergence rate.
    convergence: list[dict[str, Any]] = field(default_factory=list)
    #: Panel B: anti_entropy repair rate while the peer is partitioned away vs after heal.
    bounded_rate: float = 0.0   # anti_entropy while partitioned — cannot cross the split
    reachable_rate: float = 0.0  # anti_entropy after heal — reaches every replica
    n_peers: int = 0
    #: Tier-B reproduces the read-repair + (broken/repaired) convergence bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0


def _config(cfg: ED19Config) -> DistConfig:
    return DistConfig(
        name=cfg.name, nodes=cfg.nodes, objects=(cfg.key, "y"),
        values=("a", cfg.v1, "c", "d"), replication_factor=len(cfg.nodes),
    )


def _replica(state: DistributedState, key: str, node: str) -> str:
    r = state.replicas.get((key, node))
    return r.value if r is not None else ""


def run_ed19(cfg: ED19Config | None = None) -> ED19Result:
    cfg = cfg or ED19Config()
    config = _config(cfg)
    ref = ReferenceDistOracle(config)
    sysb = SystemDistOracle(config)
    coord = cfg.nodes[0]
    peers = cfg.nodes[1:]
    result = ED19Result(n_peers=len(peers))
    tier_b_agree = True
    tier_b_steps = 0

    def run(script: list[str]) -> DistributedState:
        nonlocal tier_b_agree, tier_b_steps
        sa = sb = DistributedState.initial(config)
        for cmd in script:
            action = parse_dist_action(cmd)
            ra, rb = ref.step(sa, action), sysb.step(sb, action)
            if cluster_view(ra.state) != cluster_view(rb.state):
                tier_b_agree = False
            tier_b_steps += 1
            sa, sb = ra.state, rb.state
        return sa

    # --- Panel A: anti-entropy repairs a dropped write; advance cannot ----------------------------
    for regime in ("advance_only", "anti_entropy"):
        converged = 0
        for peer in peers:
            # drop the write off from `peer`, advance, heal: `peer` is permanently stale (ED18).
            base = [
                f"put {coord} {cfg.key} {cfg.v1}",
                f"drop {coord} {peer}",
                "advance 2",
                "heal",
            ]
            repair = "advance 2" if regime == "advance_only" else f"anti_entropy {peer}"
            final = run([*base, repair])
            if _replica(final, cfg.key, peer) == cfg.v1:
                converged += 1
        result.convergence.append({
            "regime": regime,
            "converged": converged,
            "scenarios": len(peers),
            "rate": converged / len(peers) if peers else 0.0,
        })

    # --- Panel B: anti-entropy is bounded by reachability -----------------------------------------
    bounded = 0
    reachable = 0
    for peer in peers:
        others = " ".join(n for n in cfg.nodes if n != peer)
        stale = [f"put {coord} {cfg.key} {cfg.v1}", f"drop {coord} {peer}", "advance 2"]
        # bounded: anti_entropy while `peer` is partitioned away — it cannot pull across the split.
        partitioned = run([*stale, f"partition {peer} | {others}", f"anti_entropy {peer}"])
        if _replica(partitioned, cfg.key, peer) != cfg.v1:
            bounded += 1  # correctly did NOT repair (no reachable copy of the value)
        # reachable: the same op after heal reaches every replica and repairs.
        healed = run([*stale, "heal", f"anti_entropy {peer}"])
        if _replica(healed, cfg.key, peer) == cfg.v1:
            reachable += 1

    n = len(peers) or 1
    result.bounded_rate = (len(peers) - bounded) / n  # repair rate while partitioned (expected 0.0)
    result.reachable_rate = reachable / n              # repair rate after heal (expected 1.0)
    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,regime,value,detail"


def write_csv(result: ED19Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.convergence:
        lines.append(f"convergence,{r['regime']},{r['rate']:.4f},"
                     f"converged={r['converged']}/{r['scenarios']}")
    lines.append(f"reachability,partitioned,{result.bounded_rate:.4f},anti_entropy_cannot_cross")
    lines.append(f"reachability,healed,{result.reachable_rate:.4f},anti_entropy_reaches_all")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps};peers={result.n_peers}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED19 (anti-entropy / read-repair: convergence restored after loss)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed19.csv")
    parser.add_argument("--plot", type=str, default="figures/ed19.png")
    args = parser.parse_args()
    cfg = ED19Config.from_json_file(args.config) if args.config else ED19Config()
    result = run_ed19(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — convergence of a dropped-then-healed write:")
    for r in result.convergence:
        verdict = "repairs" if r["rate"] > 0 else "STAYS STALE"
        print(f"    [{r['regime']:12s}] rate {r['rate']:.2f} "
              f"({r['converged']}/{r['scenarios']}) → {verdict}")
    print("  Panel B — anti-entropy bounded by reachability:")
    print(f"    [partitioned] repair rate {result.bounded_rate:.2f} → "
          f"{'repairs' if result.bounded_rate > 0 else 'cannot cross the split'}")
    print(f"    [after heal ] repair rate {result.reachable_rate:.2f} → "
          f"{'repairs' if result.reachable_rate > 0 else 'never repairs'}")
    print(f"  Tier-B reproduces the read-repair bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed19 import plot_ed19

        plot_ed19(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
