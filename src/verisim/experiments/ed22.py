"""ED22 — pairwise gossip: bidirectional anti-entropy and epidemic convergence.

The DS0-increment-15 experiment for SPEC-7 §4 (the multi-node `ReplicaConverge` form) — the `gossip`
protocol op, the **pairwise, bidirectional** sibling of `anti_entropy`. Where `anti_entropy node`
(ED19) is a one-directional pull — it repairs the *one* named node from its reachable peers —
`gossip a b` is the Merkle-tree anti-entropy real eventually-consistent stores (Dynamo, Cassandra)
run in the background between *pairs* of nodes: for every object both replicate, **both** adopt the
per-object winner of their two replicas. It reuses the `ReplicaWrite` edit (no new state) and is a
pure coordinator-level reconciliation, so Tier-A and Tier-B compute byte-identical deltas. Two
findings, dependency-free, on a cluster where every node replicates every object:

  - **Panel A — one pairwise gossip reconciles BOTH endpoints; one anti-entropy fixes only one.**
    Cut `a` off from object `x`'s write and `b` off from object `y`'s, so `a` is stale on `y` and
    `b` is stale on `x` (complementary holes). A single `gossip a b` fills *both* holes at once (`a`
    gets `y`, `b` gets `x`) — both nodes fully reconciled. A single `anti_entropy a` fills only
    `a`'s hole (it pulls `y` to `a`) and leaves `b` still stale on `x`: the one-directional vs
    bidirectional distinction, the reason real systems run pairwise anti-entropy, not just
    read-repair.

  - **Panel B — a chain of pairwise gossips converges the whole reachable component (epidemic
    spread).** A write lands at one node and is dropped to all the others (every peer stale). Gossip
    along a chain (`n0`↔`n1`, `n1`↔`n2`, …) and the value spreads hop by hop until the **entire
    reachable component** holds it (convergence rate **1.0**) — the gossip-protocol epidemic, where
    `anti_entropy` would need every node to pull. It is **bounded by reachability**: a node
    partitioned away from the chain never receives the value (gossip across a cut link is
    `unavailable`), so full convergence still needs the network whole.

Both panels confirm **Tier-B** agreement: pairwise gossip is a coordinator-level reconciliation (it
reads both replicas directly, no in-flight message), so the autonomous-actor system oracle computes
a byte-identical sync and reproduces the bidirectional / epidemic convergence bit-for-bit (the W1
retirement, §5.2). The `gossip` action is purely additive — no prior golden, hash, or tokenization
changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


@dataclass(frozen=True)
class ED22Config:
    name: str = "ed22-gossip"
    nodes: tuple[str, ...] = ("n0", "n1", "n2", "n3")
    key_x: str = "x"
    key_y: str = "y"
    vx: str = "b"
    vy: str = "c"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED22Config:
        b = ED22Config()
        return ED22Config(
            name=d.get("name", b.name),
            nodes=tuple(d.get("nodes", b.nodes)),
            key_x=d.get("key_x", b.key_x),
            key_y=d.get("key_y", b.key_y),
            vx=d.get("vx", b.vx),
            vy=d.get("vy", b.vy),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED22Config:
        return ED22Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED22Result:
    #: Panel A: one `gossip a b` fills both complementary holes (a stale on y, b stale on x).
    gossip_reconciles_both: bool = False
    #: Panel A: one `anti_entropy a` fills only a's hole, leaving b stale (the one-sided contrast).
    anti_entropy_reconciles_one: bool = False
    #: Panel B: fraction of a stale chain that converges via a chain of pairwise gossips.
    epidemic_converged_rate: float = 0.0
    #: Panel B: a node partitioned off the chain stays stale (gossip cannot cross the cut).
    partition_blocks_epidemic: bool = False
    n_chain: int = 0
    #: Tier-B reproduces the bidirectional / epidemic convergence bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0


def _config(cfg: ED22Config) -> DistConfig:
    return DistConfig(
        name=cfg.name, nodes=cfg.nodes, objects=(cfg.key_x, cfg.key_y),
        values=("a", cfg.vx, cfg.vy, "d"), replication_factor=len(cfg.nodes),
    )


def _replica(state: DistributedState, key: str, node: str) -> str:
    r = state.replicas.get((key, node))
    return r.value if r is not None else ""


def run_ed22(cfg: ED22Config | None = None) -> ED22Result:
    cfg = cfg or ED22Config()
    config = _config(cfg)
    ref = ReferenceDistOracle(config)
    sysb = SystemDistOracle(config)
    a, b = cfg.nodes[0], cfg.nodes[1]
    result = ED22Result(n_chain=len(cfg.nodes))
    tier_b_agree = True
    tier_b_steps = 0

    def run(script: list[str]) -> DistributedState:
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

    # --- Panel A: complementary holes — gossip fills both, anti_entropy fills one -----------------
    # a writes x (dropped to b → b stale on x); b writes y (dropped to a → a stale on y).
    setup = [
        f"put {a} {cfg.key_x} {cfg.vx}", f"drop {a} {b}",
        f"put {b} {cfg.key_y} {cfg.vy}", f"drop {b} {a}",
        "advance 2",
    ]
    g = run([*setup, f"gossip {a} {b}"])
    result.gossip_reconciles_both = (
        _replica(g, cfg.key_x, a) == cfg.vx and _replica(g, cfg.key_y, a) == cfg.vy
        and _replica(g, cfg.key_x, b) == cfg.vx and _replica(g, cfg.key_y, b) == cfg.vy
    )
    pulled = run([*setup, f"anti_entropy {a}"])
    # b is still stale on x — anti_entropy is one-sided (it repairs only the named node).
    result.anti_entropy_reconciles_one = (
        _replica(pulled, cfg.key_x, a) == cfg.vx and _replica(pulled, cfg.key_y, a) == cfg.vy
        and _replica(pulled, cfg.key_x, b) != cfg.vx
    )

    # --- Panel B: epidemic convergence via a chain of pairwise gossips ----------------------------
    coord = cfg.nodes[0]
    drops = [f"drop {coord} {n}" for n in cfg.nodes[1:]]  # cut the write off from every peer
    chain = [f"gossip {cfg.nodes[i]} {cfg.nodes[i + 1]}" for i in range(len(cfg.nodes) - 1)]
    epidemic = run([f"put {coord} {cfg.key_x} {cfg.vx}", *drops, "advance 2", *chain])
    converged = sum(1 for n in cfg.nodes if _replica(epidemic, cfg.key_x, n) == cfg.vx)
    result.epidemic_converged_rate = converged / len(cfg.nodes)

    # bounded by reachability: partition the last node off, the chain cannot reach it.
    last = cfg.nodes[-1]
    rest = " ".join(cfg.nodes[:-1])
    blocked = run([
        f"put {coord} {cfg.key_x} {cfg.vx}", *drops, "advance 2",
        f"partition {last} | {rest}", *chain,
    ])
    result.partition_blocks_epidemic = _replica(blocked, cfg.key_x, last) != cfg.vx

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED22Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"bidirectional,gossip_reconciles_both,"
                 f"{1.0 if result.gossip_reconciles_both else 0.0:.4f},both_endpoints_synced")
    lines.append(f"bidirectional,anti_entropy_reconciles_one,"
                 f"{1.0 if result.anti_entropy_reconciles_one else 0.0:.4f},only_named_node_synced")
    lines.append(f"epidemic,converged_rate,{result.epidemic_converged_rate:.4f},"
                 f"chain_of_pairwise_gossips_over_{result.n_chain}_nodes")
    lines.append(f"epidemic,partition_blocks,"
                 f"{1.0 if result.partition_blocks_epidemic else 0.0:.4f},bounded_by_reachability")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED22 (pairwise gossip: bidirectional anti-entropy + epidemic convergence)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed22.csv")
    parser.add_argument("--plot", type=str, default="figures/ed22.png")
    args = parser.parse_args()
    cfg = ED22Config.from_json_file(args.config) if args.config else ED22Config()
    result = run_ed22(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — one pairwise gossip reconciles BOTH; one anti-entropy fixes only one:")
    print(f"    [gossip a b   ] both endpoints reconciled: {result.gossip_reconciles_both}")
    print(f"    [anti_entropy ] only the named node reconciled (peer stays stale): "
          f"{result.anti_entropy_reconciles_one}")
    print("  Panel B — a chain of pairwise gossips converges the reachable component (epidemic):")
    print(f"    converged fraction over {result.n_chain} nodes: "
          f"{result.epidemic_converged_rate:.2f}")
    print(f"    bounded by reachability (partitioned node stays stale): "
          f"{result.partition_blocks_epidemic}")
    print(f"  Tier-B reproduces the bidirectional / epidemic convergence bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed22 import plot_ed22

        plot_ed22(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
