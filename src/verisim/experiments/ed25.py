"""ED25 — leader leases: local linearizable reads, and the lease-vs-election safety tension.

The DS0-increment-18 experiment for SPEC-7 §3.2/§4 (the consensus family) — the Raft **leader
lease**, a read optimization layered on the `elect`/`propose`/`step_down` core. `lease node dt` lets
the *current* leader take a read lease through global clock `+ dt`; `lread node key` then serves a
**local linearizable read with no quorum round-trip** while the lease holds. Two properties,
measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — local reads without a quorum.** Across a cluster-size sweep, a leader holding a live
    lease serves `lread` (rate **1.0**), and the sharp case: a leader **partitioned into the
    minority can still `lread`** locally (rate **1.0**) where its `propose` there is `no_quorum`
    (the control, **1.0**) — the read-availability the lease buys, safe because a live lease keeps
    the leader's term uncontested. Once the clock advances past the deadline the same `lread` is
    rejected `lease_expired` (rate **1.0**): the leader must renew or fall back to a quorum read.

  - **Panel B — the lease/election safety tension, and step_down releases it.** While the
    incumbent's lease is live, a fresh `elect` is rejected `lease_held` (a successor must **wait out
    the lease** — rate **1.0**); after the clock advances past expiry the election succeeds
    (**1.0**). This is what makes the lease read safe: leadership cannot change hands under a live
    lease, so no two leaders ever serve reads at once. The contrast that ties back to ED24: a
    voluntary `step_down` **releases the lease immediately**, so a graceful handoff elects a
    successor with no wait (**1.0**) — where a *crashed* leader forces the cluster to outlast the
    lease. Power handed back is free; power lost is waited out.

`lease`/`lread` touch no replica (leadership/lease are coordinator-level cluster metadata, like
`term`/`leader`), so the autonomous-actor system oracle (Tier-B) reproduces every lease/read/fence
transition bit-for-bit (the W1 retirement, §5.2). The lease deadline is omitted from the canonical
form until the first `lease`, so the op family is purely additive — no prior golden/hash/token
changes. (Honest scope: the lease is a global-clock deadline read as cluster metadata, and `lread`
is linearizable w.r.t. consensus `propose` writes — the real per-node lease timer under bounded
clock *drift* is a Tier-B refinement; our `clock_skew` is a constant offset, not a rate, so it
shifts neither grant nor expiry here.)
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
class ED25Config:
    name: str = "ed25-lease"
    #: cluster sizes for the local-read sweep (odd → a clean strict majority).
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "x"
    val: str = "b"  # the value the leader commits, then serves via lread
    lease_dt: int = 5  # the lease duration (in clock units)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED25Config:
        b = ED25Config()
        return ED25Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            val=d.get("val", b.val),
            lease_dt=d.get("lease_dt", b.lease_dt),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED25Config:
        return ED25Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED25Result:
    #: Panel A: a live lease serves a local read.
    valid_lease_read_rate: float = 0.0
    #: Panel A: a minority-stranded leader can still lread locally (no quorum needed).
    minority_lread_rate: float = 0.0
    #: Panel A: the control — that same minority leader's propose is no_quorum.
    minority_propose_blocked_rate: float = 0.0
    #: Panel A: once the lease expires the local read is rejected.
    expired_lease_read_rate: float = 0.0
    n_sizes: int = 0
    #: Panel B: a fresh elect is blocked while the incumbent's lease is live (wait it out).
    elect_blocked_under_lease: bool = False
    #: Panel B: the election succeeds once the clock advances past the lease deadline.
    elect_after_expiry_ok: bool = False
    #: Panel B: step_down releases the lease, so a successor elects immediately (the fast handoff).
    stepdown_releases_lease: bool = False
    #: Tier-B reproduces every lease/read/fence transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, str]] = field(default_factory=list)  # (n, leader)


def _config(n: int, key: str, vals: tuple[str, ...]) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(
        name=f"ed25-{n}n", nodes=nodes, objects=(key,),
        values=vals, replication_factor=n,
    )


def run_ed25(cfg: ED25Config | None = None) -> ED25Result:
    cfg = cfg or ED25Config()
    result = ED25Result()
    vals = ("a", cfg.val, "c", "d")
    tier_b_agree = True
    tier_b_steps = 0

    def step_both(
        ref: ReferenceDistOracle, sysb: SystemDistOracle,
        sa: DistributedState, sb: DistributedState, cmd: str,
    ) -> tuple[DistributedState, DistributedState, str, str]:
        nonlocal tier_b_agree, tier_b_steps
        action = parse_dist_action(cmd)
        ra, rb = ref.step(sa, action), sysb.step(sb, action)
        if cluster_view(ra.state) != cluster_view(rb.state) or (ra.status, ra.value) != (
            rb.status, rb.value
        ):
            tier_b_agree = False
        tier_b_steps += 1
        return ra.state, rb.state, ra.status, ra.value

    # --- Panel A: local linearizable reads without a quorum ---------------------------------------
    valid_read = 0
    minority_read = 0
    minority_blocked = 0
    expired_read = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key, vals)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        leader = nodes[0]
        others = [x for x in nodes if x != leader]
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {leader}")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"propose {leader} {cfg.key} {cfg.val}")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"lease {leader} {cfg.lease_dt}")
        # a live lease serves a local read
        _, _, s_read, _ = step_both(ref, sysb, sa, sb, f"lread {leader} {cfg.key}")
        # isolate the leader alone (a strict minority for n >= 3); it cannot reach a majority to
        # propose, but its lease still holds, so it can still serve a local lread without a quorum.
        sa, sb, _, _ = step_both(ref, sysb, sa, sb,
                                 f"partition {leader} | {' '.join(others)}")
        _, _, s_minprop, _ = step_both(ref, sysb, sa, sb, f"propose {leader} {cfg.key} c")
        _, _, s_minread, _ = step_both(ref, sysb, sa, sb, f"lread {leader} {cfg.key}")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
        # advance past the lease deadline: the local read is now rejected
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"advance {cfg.lease_dt + 1}")
        _, _, s_expired, _ = step_both(ref, sysb, sa, sb, f"lread {leader} {cfg.key}")
        sizes += 1
        if s_read == "ok":
            valid_read += 1
        if s_minread == "ok":
            minority_read += 1
        if s_minprop == "no_quorum":
            minority_blocked += 1
        if s_expired == "lease_expired":
            expired_read += 1
        result.per_size.append((n, leader))
    result.valid_lease_read_rate = valid_read / sizes if sizes else 0.0
    result.minority_lread_rate = minority_read / sizes if sizes else 0.0
    result.minority_propose_blocked_rate = minority_blocked / sizes if sizes else 0.0
    result.expired_lease_read_rate = expired_read / sizes if sizes else 0.0
    result.n_sizes = sizes

    # --- Panel B: the lease/election safety tension, and step_down releases it --------------------
    config = _config(3, cfg.key, vals)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)

    # A live lease blocks a fresh election; advancing past expiry unblocks it.
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"lease n0 {cfg.lease_dt}")
    _, _, s_blocked, _ = step_both(ref, sysb, sa, sb, "elect n1")  # lease_held
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"advance {cfg.lease_dt + 1}")
    _, _, s_after, _ = step_both(ref, sysb, sa, sb, "elect n1")  # now elected
    result.elect_blocked_under_lease = s_blocked == "lease_held"
    result.elect_after_expiry_ok = s_after == "elected"

    # step_down releases the lease, so a successor elects immediately (no wait).
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"lease n0 {cfg.lease_dt}")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "step_down n0")  # releases the lease
    _, _, s_fast, _ = step_both(ref, sysb, sa, sb, "elect n1")  # elected with no wait
    result.stepdown_releases_lease = s_fast == "elected"

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED25Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"local_read,valid_lease_read,{result.valid_lease_read_rate:.4f},"
                 f"served_over_{result.n_sizes}_clusters")
    lines.append(f"local_read,minority_lread,{result.minority_lread_rate:.4f},"
                 f"minority_leader_reads_locally")
    lines.append(f"local_read,minority_propose_blocked,{result.minority_propose_blocked_rate:.4f},"
                 f"propose_no_quorum_control")
    lines.append(f"local_read,expired_lease_read,{result.expired_lease_read_rate:.4f},"
                 f"rejected_after_expiry")
    lines.append(f"safety,elect_blocked_under_lease,"
                 f"{1.0 if result.elect_blocked_under_lease else 0.0:.4f},lease_held_wait_it_out")
    lines.append(f"safety,elect_after_expiry,"
                 f"{1.0 if result.elect_after_expiry_ok else 0.0:.4f},unblocked_past_deadline")
    lines.append(f"safety,stepdown_releases_lease,"
                 f"{1.0 if result.stepdown_releases_lease else 0.0:.4f},fast_handoff_no_wait")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED25 (leader leases: local reads without a quorum + the lease tension)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed25.csv")
    parser.add_argument("--plot", type=str, default="figures/ed25.png")
    args = parser.parse_args()
    cfg = ED25Config.from_json_file(args.config) if args.config else ED25Config()
    result = run_ed25(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — local linearizable reads without a quorum (over the cluster-size sweep):")
    print(f"    valid lease serves read: {result.valid_lease_read_rate:.2f} "
          f"| minority leader reads locally: {result.minority_lread_rate:.2f} "
          f"(propose no_quorum: {result.minority_propose_blocked_rate:.2f}) "
          f"| expired lease rejected: {result.expired_lease_read_rate:.2f}")
    print("  Panel B — the lease/election safety tension:")
    print(f"    elect blocked under live lease: {result.elect_blocked_under_lease} "
          f"| elect after expiry: {result.elect_after_expiry_ok} "
          f"| step_down releases lease (fast handoff): {result.stepdown_releases_lease}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed25 import plot_ed25

        plot_ed25(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
