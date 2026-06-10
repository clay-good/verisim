"""ED20 — message timing: the recoverable delay and the reorder-invariant convergence.

The DS0-increment-13 experiment for SPEC-7 §3.2 / §3.4 — the ``delay`` and ``reorder`` faults, the
message-timing half of the medium SPEC-7 §3.4 names ("partition, crash, message loss, **reorder**,
clock skew") but had deferred since increment 1. Both edit only the existing
``Message.deliver_after`` field, so they add no state and compose with every consistency model. Two
findings, dependency-free,
on a 3-node cluster where every node replicates every object (a write at ``n0`` enqueues one message
to each peer):

  - **Panel A — delay is recoverable; drop is not.** A write, then the coordinator's link to one
    peer is either **delayed** (its replication message deferred by ``dt``) or **dropped**. Under
    ``delay`` the peer is stale until the clock passes the deferral, then the message is delivered
    and the peer **converges** (rate **1.0**) — a *recoverable* delay. Under ``drop`` the message is
    destroyed, so the peer stays **permanently stale** (rate **0.0**) — an *unrecoverable* loss.
    Same observable symptom (a stale replica), two media — completing the contrast ED18 opened: the
    eventual-consistency convergence guarantee assumes delivery is *reliable if delayed*, and
    ``delay`` exercises exactly the "if delayed" half it depends on.

  - **Panel B — reorder flips the transit observation, never the converged value.** Two writes to
    one key are staggered (the first deferred, so the *newer* write is scheduled to arrive first).
    Normally a peer transiently shows the newer value; after ``reorder`` the schedule reverses and
    the peer transiently shows the *older* value — a genuinely different in-transit observation —
    yet **both converge to the newer write** (rate of converged-value-invariance **1.0**).
    Last-writer-wins by ``(version, value)`` is a commutative join, so delivery order changes what
    you can catch in flight but never where the cluster lands: the §5.2 order-independence Tier-B's
    shuffled scheduler certifies, here made a *controllable input* rather than only a scheduler
    property.

Both panels confirm **Tier-B** agreement: message timing is a medium change, so the autonomous-actor
system oracle computes a byte-identical reschedule and then, delivering on its own seed-shuffled
schedule, reproduces the delayed (and reordered) convergence bit-for-bit (the W1 retirement, §5.2).
The ``delay``/``reorder`` actions are purely additive — no prior golden, hash, or tokenization
changes.
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
class ED20Config:
    name: str = "ed20-timing"
    nodes: tuple[str, ...] = ("n0", "n1", "n2")
    key: str = "x"
    v1: str = "b"  # the first write (scheduled to arrive last after the delay)
    v2: str = "c"  # the second, newer write (scheduled to arrive first)
    delay_dt: int = 5  # Panel A: the recoverable deferral
    stagger: int = 100  # Panel B: how far the first write is deferred to stagger the schedule

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED20Config:
        b = ED20Config()
        return ED20Config(
            name=d.get("name", b.name),
            nodes=tuple(d.get("nodes", b.nodes)),
            key=d.get("key", b.key),
            v1=d.get("v1", b.v1),
            v2=d.get("v2", b.v2),
            delay_dt=int(d.get("delay_dt", b.delay_dt)),
            stagger=int(d.get("stagger", b.stagger)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED20Config:
        return ED20Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED20Result:
    #: Panel A: per regime (delay | drop), the post-advance convergence rate over peers.
    convergence: list[dict[str, Any]] = field(default_factory=list)
    #: Panel A: under delay, the peer is stale before the deferral elapses and converges after.
    delay_stale_then_converges: bool = True
    #: Panel B: fraction of peers whose *in-transit* value changed under reorder (the flip).
    reorder_transit_flip_rate: float = 0.0
    #: Panel B: fraction of peers whose *converged* value is unchanged by reorder (LWW invariance).
    reorder_converged_invariant_rate: float = 0.0
    n_peers: int = 0
    #: Tier-B reproduces the delay/reorder + delivery bit-for-bit over every scenario.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0


def _config(cfg: ED20Config) -> DistConfig:
    # full replication: a write at n0 sends one message to each peer (every node holds every object)
    return DistConfig(
        name=cfg.name, nodes=cfg.nodes, objects=(cfg.key, "y"),
        values=("a", cfg.v1, cfg.v2, "d"), replication_factor=len(cfg.nodes),
    )


def _replica(state: DistributedState, key: str, node: str) -> str:
    r = state.replicas.get((key, node))
    return r.value if r is not None else ""


def run_ed20(cfg: ED20Config | None = None) -> ED20Result:
    cfg = cfg or ED20Config()
    config = _config(cfg)
    ref = ReferenceDistOracle(config)
    sysb = SystemDistOracle(config)
    coord = cfg.nodes[0]
    peers = cfg.nodes[1:]
    result = ED20Result(n_peers=len(peers))
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

    # --- Panel A: delay (recoverable) vs drop (unrecoverable) -------------------------------------
    for regime in ("delay", "drop"):
        converged = 0
        for peer in peers:
            cut = (
                f"delay {coord} {peer} {cfg.delay_dt}" if regime == "delay"
                else f"drop {coord} {peer}"
            )
            # advance short of the deferral (still stale), then well past it (delay converges).
            final = run([
                f"put {coord} {cfg.key} {cfg.v1}",
                cut,
                "advance 2",
                f"advance {cfg.delay_dt + 10}",
            ])
            if _replica(final, cfg.key, peer) == cfg.v1:
                converged += 1
        result.convergence.append({
            "regime": regime,
            "converged": converged,
            "scenarios": len(peers),
            "rate": converged / len(peers) if peers else 0.0,
        })

    # Panel A detail: under delay the peer is stale before the deferral, converged after.
    stale_then_converges = True
    for peer in peers:
        mid = run([f"put {coord} {cfg.key} {cfg.v1}", f"delay {coord} {peer} {cfg.delay_dt}",
                   "advance 2"])
        late = run([f"put {coord} {cfg.key} {cfg.v1}", f"delay {coord} {peer} {cfg.delay_dt}",
                    "advance 2", f"advance {cfg.delay_dt + 10}"])
        if _replica(mid, cfg.key, peer) == cfg.v1 or _replica(late, cfg.key, peer) != cfg.v1:
            stale_then_converges = False
    result.delay_stale_then_converges = stale_then_converges

    # --- Panel B: reorder flips the transit observation, not the converged value ------------------
    transit_flips = 0
    converged_invariant = 0
    far = cfg.stagger + 10
    for peer in peers:
        setup = [
            f"put {coord} {cfg.key} {cfg.v1}",
            f"delay {coord} {peer} {cfg.stagger}",  # defer v1 so v2 is scheduled to arrive first
            f"put {coord} {cfg.key} {cfg.v2}",
        ]
        sched_transit = _replica(run([*setup, "advance 2"]), cfg.key, peer)
        reord_transit = _replica(
            run([*setup, f"reorder {coord} {peer}", "advance 2"]), cfg.key, peer
        )
        sched_final = _replica(run([*setup, "advance 2", f"advance {far}"]), cfg.key, peer)
        reord_final = _replica(
            run([*setup, f"reorder {coord} {peer}", "advance 2", f"advance {far}"]), cfg.key, peer
        )
        if sched_transit != reord_transit:
            transit_flips += 1
        if sched_final == reord_final == cfg.v2:
            converged_invariant += 1
    n = len(peers) or 1
    result.reorder_transit_flip_rate = transit_flips / n
    result.reorder_converged_invariant_rate = converged_invariant / n

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,regime,value,detail"


def write_csv(result: ED20Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.convergence:
        lines.append(f"convergence,{r['regime']},{r['rate']:.4f},"
                     f"converged={r['converged']}/{r['scenarios']}")
    lines.append(f"delay_detail,stale_then_converges,"
                 f"{1.0 if result.delay_stale_then_converges else 0.0:.4f},recoverable")
    lines.append(f"reorder,transit_flip,{result.reorder_transit_flip_rate:.4f},"
                 f"in_transit_value_changes")
    lines.append(f"reorder,converged_invariant,{result.reorder_converged_invariant_rate:.4f},"
                 f"lww_commutative_join")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps};peers={result.n_peers}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED20 (message timing: recoverable delay + reorder-invariant convergence)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed20.csv")
    parser.add_argument("--plot", type=str, default="figures/ed20.png")
    args = parser.parse_args()
    cfg = ED20Config.from_json_file(args.config) if args.config else ED20Config()
    result = run_ed20(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — convergence after advance (write cut off from a peer):")
    for r in result.convergence:
        verdict = "converges" if r["rate"] > 0 else "STAYS STALE"
        print(f"    [{r['regime']:6s}] rate {r['rate']:.2f} "
              f"({r['converged']}/{r['scenarios']}) → {verdict}")
    print(f"    delay is stale-then-converges (recoverable): {result.delay_stale_then_converges}")
    print("  Panel B — reorder flips the transit, not the converged value:")
    flip = "changes what is seen in flight" if result.reorder_transit_flip_rate > 0 else "no flip"
    inv = ("LWW lands the same value regardless of order"
           if result.reorder_converged_invariant_rate > 0 else "order-dependent!")
    print(f"    [transit flip      ] rate {result.reorder_transit_flip_rate:.2f} → {flip}")
    print(f"    [converged invariant] rate {result.reorder_converged_invariant_rate:.2f} → {inv}")
    print(f"  Tier-B reproduces delay/reorder + delivery bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed20 import plot_ed20

        plot_ed20(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
