"""ED18 — message loss: the broken-convergence anomaly + the lost write only a newer write heals.

The DS0-increment-11 experiment for SPEC-7 §3.2 / §3.4 — the ``drop`` fault, the unreliable-
network ``BUGGIFY`` primitive (§2.1) the deterministic-simulation-testing data factory needs and
the delta vocabulary already anticipated (``MsgDrop``). ``drop src dst`` loses every in-flight
replication message from ``src`` to ``dst``. Two findings, dependency-free, on a 3-node cluster
where every node replicates every object (so a write at ``n0`` enqueues one message to each peer):

  - **Panel A — drop breaks convergence; partition does not.** A write, then the coordinator's link
    to one peer is cut, then time advances and the link is restored. Under **partition** the
    replication message is *held* (stuck in-flight across the split) and **delivered on
    ``heal``+``advance``**, so the peer reconverges (convergence rate **1.0**). Under **drop** the
    message is *destroyed*, so ``heal``+``advance`` has nothing to deliver and the peer stays
    **permanently stale** (convergence rate **0.0**). Same observable symptom — a stale replica —
    from two media: a *recoverable* delay (partition) and an *unrecoverable* loss (drop) — the
    eventual-consistency convergence guarantee's hidden premise made visible: it assumes delivery
    is *reliable, if delayed*; lift that and convergence is gone.

  - **Panel B — only a newer write heals the lost write.** After the drop leaves the peer stale,
    ``heal``+``advance`` alone never repairs it (rate **0.0**), but a *subsequent* write to the same
    key (a higher MVCC version) does — its fresh replication message is delivered and overwrites the
    stale replica (rate **1.0**). And the dropped value is **never observed** by that peer: its
    replica goes boot → the *new* value, skipping the lost one (a lost update at the network layer —
    the write is gone, not delayed, recoverable only by being superseded).

Both panels confirm the **Tier-B** agreement: message loss is a medium change (the message simply
never arrives), so the autonomous-actor system oracle computes a byte-identical drop and then,
running its own independent delivery, reproduces the broken (and overwrite-repaired) convergence
bit-for-bit (the W1 retirement, §5.2). The ``drop`` action is purely additive — no prior golden,
hash, or tokenization changes — so it composes with every consistency model unchanged.
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
class ED18Config:
    name: str = "ed18-drop"
    nodes: tuple[str, ...] = ("n0", "n1", "n2")
    key: str = "x"
    v1: str = "b"  # the write that gets cut off from a peer
    v2: str = "c"  # the later overwrite that heals it

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED18Config:
        b = ED18Config()
        return ED18Config(
            name=d.get("name", b.name),
            nodes=tuple(d.get("nodes", b.nodes)),
            key=d.get("key", b.key),
            v1=d.get("v1", b.v1),
            v2=d.get("v2", b.v2),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED18Config:
        return ED18Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED18Result:
    #: Panel A: per regime (partition | drop), the post-heal+advance convergence rate over peers.
    convergence: list[dict[str, Any]] = field(default_factory=list)
    #: Panel B: after a drop, the heal-only vs overwrite repair rates + whether the lost value is
    #: ever observed by the cut-off peer.
    drop_heal_only_rate: float = 0.0
    drop_overwrite_rate: float = 0.0
    lost_value_never_observed: bool = True
    n_peers: int = 0
    #: Tier-B reproduces the drop + (broken / repaired) convergence bit-for-bit over every scenario.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0


def _config(cfg: ED18Config) -> DistConfig:
    # full replication: a write at n0 sends one message to each peer (every node holds every object)
    return DistConfig(
        name=cfg.name, nodes=cfg.nodes, objects=(cfg.key, "y"),
        values=("a", cfg.v1, cfg.v2, "d"), replication_factor=len(cfg.nodes),
    )


def _replica(state: DistributedState, key: str, node: str) -> str:
    r = state.replicas.get((key, node))
    return r.value if r is not None else ""


def run_ed18(cfg: ED18Config | None = None) -> ED18Result:
    cfg = cfg or ED18Config()
    config = _config(cfg)
    ref = ReferenceDistOracle(config)
    sysb = SystemDistOracle(config)
    coord = cfg.nodes[0]
    peers = cfg.nodes[1:]  # the replicas the coordinator must reach
    result = ED18Result(n_peers=len(peers))
    tier_b_agree = True
    tier_b_steps = 0

    def run(script: list[str], track_peer: str | None = None) -> tuple[DistributedState, set[str]]:
        nonlocal tier_b_agree, tier_b_steps
        sa = sb = DistributedState.initial(config)
        seen: set[str] = set()
        for cmd in script:
            action = parse_dist_action(cmd)
            ra, rb = ref.step(sa, action), sysb.step(sb, action)
            if cluster_view(ra.state) != cluster_view(rb.state):
                tier_b_agree = False
            tier_b_steps += 1
            sa, sb = ra.state, rb.state
            if track_peer is not None:
                seen.add(_replica(sa, cfg.key, track_peer))
        return sa, seen

    # --- Panel A: partition (recoverable) vs drop (unrecoverable) convergence ---------------------
    for regime in ("partition", "drop"):
        converged = 0
        for peer in peers:
            others = " ".join(n for n in cfg.nodes if n != peer)
            cut = (
                f"partition {peer} | {others}" if regime == "partition"
                else f"drop {coord} {peer}"
            )
            script = [
                f"put {coord} {cfg.key} {cfg.v1}",  # write at the coordinator
                cut,                                  # cut the link to one peer
                "advance 2",                          # deliver what can be delivered
                "heal",                               # restore the network
                "advance 2",                          # ... and try again
            ]
            final, _ = run(script)
            if _replica(final, cfg.key, peer) == cfg.v1:
                converged += 1
        result.convergence.append({
            "regime": regime,
            "converged": converged,
            "scenarios": len(peers),
            "rate": converged / len(peers) if peers else 0.0,
        })

    # --- Panel B: after a drop, heal-only never repairs; a newer write does -----------------------
    heal_only = 0
    overwrite = 0
    lost_never_seen = True
    for peer in peers:
        drop_heal = [
            f"put {coord} {cfg.key} {cfg.v1}",
            f"drop {coord} {peer}",
            "advance 2",
            "heal",
            "advance 2",
        ]
        final_heal, _ = run(drop_heal)
        if _replica(final_heal, cfg.key, peer) == cfg.v1:
            heal_only += 1

        drop_overwrite = [*drop_heal, f"put {coord} {cfg.key} {cfg.v2}", "advance 2"]
        final_ow, seen_ow = run(drop_overwrite, track_peer=peer)
        if _replica(final_ow, cfg.key, peer) == cfg.v2:
            overwrite += 1
        if cfg.v1 in seen_ow:  # the cut-off peer must never have observed the dropped value
            lost_never_seen = False

    n = len(peers) or 1
    result.drop_heal_only_rate = heal_only / n
    result.drop_overwrite_rate = overwrite / n
    result.lost_value_never_observed = lost_never_seen
    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,regime,value,detail"


def write_csv(result: ED18Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.convergence:
        lines.append(f"convergence,{r['regime']},{r['rate']:.4f},"
                     f"converged={r['converged']}/{r['scenarios']}")
    lines.append(f"recovery,heal_only,{result.drop_heal_only_rate:.4f},"
                 f"newer_write_required")
    lines.append(f"recovery,overwrite,{result.drop_overwrite_rate:.4f},"
                 f"lost_value_never_observed={result.lost_value_never_observed}")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps};peers={result.n_peers}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED18 (message loss: broken convergence + a lost write only a write heals)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed18.csv")
    parser.add_argument("--plot", type=str, default="figures/ed18.png")
    args = parser.parse_args()
    cfg = ED18Config.from_json_file(args.config) if args.config else ED18Config()
    result = run_ed18(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — convergence after heal+advance (write cut off from a peer):")
    for r in result.convergence:
        verdict = "converges" if r["rate"] > 0 else "STAYS STALE"
        print(f"    [{r['regime']:10s}] rate {r['rate']:.2f} "
              f"({r['converged']}/{r['scenarios']}) → {verdict}")
    print("  Panel B — repairing a dropped write:")
    print(f"    [heal only ] rate {result.drop_heal_only_rate:.2f} → "
          f"{'repairs' if result.drop_heal_only_rate > 0 else 'NEVER repairs'}")
    print(f"    [overwrite ] rate {result.drop_overwrite_rate:.2f} → "
          f"{'repairs' if result.drop_overwrite_rate > 0 else 'never repairs'} "
          f"(lost value never observed: {result.lost_value_never_observed})")
    print(f"  Tier-B reproduces drop + (broken/repaired) convergence bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed18 import plot_ed18

        plot_ed18(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
