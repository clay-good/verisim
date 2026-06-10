"""ED33 — the tombstone delete: versioned removal and the resurrection problem under partition.

The DS0-increment-26 experiment for SPEC-7 §3.2 — `delete`, the fundamental KV remove the grammar
lacked, and a canonical distributed hazard. A `delete node key` is a **versioned write of a
tombstone** (it reuses the `put` replication path with the `TOMBSTONE` value), *not* a removal of
the replica: the deleted key keeps a replica at a bumped version, so last-writer-wins orders the
delete against concurrent and stale writes by version. That is exactly what avoids the
**resurrection problem** — a "deleted" key reappearing because a stale replica's old value
out-versions an absence. Both panels are measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — delete is a versioned tombstone (LWW).** Across a cluster-size sweep (linearizable),
    a `put` then `delete` leaves **every** replica reading `deleted` (rate **1.0**). The tombstone
    is a real versioned write: it **out-versions the put it deleted** (**1.0**), and a *genuinely
    newer* `put` (higher version than the tombstone) **legitimately brings the key back** (1.0) — a
    new write, not a resurrection.

  - **Panel B — the resurrection problem under partition, and the repair (eventual).** A `delete` on
    the majority side under a partition leaves the partitioned **minority still reading the old
    value** (the deleted item is "still there" — the danger, **1.0**). The repair: after `heal`,
    both convergence ops carry the tombstone — and because its version is **higher** than the stale
    value, the minority converges to `deleted` rather than resurrecting it: `anti_entropy` repairs
    (**1.0**) and pairwise `gossip` repairs (**1.0**). A naive removal (no version) would let the
    stale replica's value win the merge — the bug the versioned tombstone exists to prevent.

`delete` reuses the `put` write path (same consistency-model replication, same in-flight medium), so
the autonomous-actor system oracle (Tier-B) reproduces every transition — including the
divergence-under-partition and the version-ordered convergence — bit-for-bit (§5.2). It adds no
state field and no edit type (the tombstone is just a replica value), so the op is purely additive.
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
class ED33Config:
    name: str = "ed33-delete"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "x"
    val: str = "a"
    val2: str = "b"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED33Config:
        b = ED33Config()
        return ED33Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            val=d.get("val", b.val),
            val2=d.get("val2", b.val2),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED33Config:
        return ED33Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED33Result:
    #: Panel A: a put then delete leaves every replica reading deleted (cluster-size sweep).
    delete_removes_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: the tombstone out-versions the put it deleted (a real versioned write).
    tombstone_outversions_put: bool = False
    #: Panel A: a genuinely newer put (higher version than the tombstone) legitimately resurrects.
    newer_put_resurrects: bool = False
    #: Panel B: under partition, the partitioned minority still reads the old value (the danger).
    minority_reads_deleted_item: bool = False
    #: Panel B: heal + anti_entropy converges the minority to deleted (tombstone wins, no comeback).
    anti_entropy_no_resurrection: bool = False
    #: Panel B: heal + pairwise gossip also converges the minority to deleted (no resurrection).
    gossip_no_resurrection: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, deleted everywhere)


def _config(n: int, key: str, model: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed33-{n}n", nodes=nodes, objects=(key,), values=("a", "b"),
                      replication_factor=n, consistency_model=model)


def run_ed33(cfg: ED33Config | None = None) -> ED33Result:
    cfg = cfg or ED33Config()
    result = ED33Result()
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

    # --- Panel A: delete is a versioned tombstone (linearizable, synchronous to all) --------------
    removed_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key, "linearizable")
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"put n0 {cfg.key} {cfg.val}")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"delete n0 {cfg.key}")
        # every replica reads deleted (synchronous linearizable delete reaches all)
        deleted_everywhere = True
        for nd in nodes:
            _, _, s_get, _ = step_both(ref, sysb, sa, sb, f"get {nd} {cfg.key}")
            if s_get != "deleted":
                deleted_everywhere = False
        sizes += 1
        if deleted_everywhere:
            removed_count += 1
        result.per_size.append((n, deleted_everywhere))
    result.delete_removes_rate = removed_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # the versioned-tombstone properties, on a 3-node linearizable cluster.
    config = _config(3, cfg.key, "linearizable")
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"put n0 {cfg.key} {cfg.val}")
    put_ver = sa.replicas[(cfg.key, "n0")].version
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"delete n0 {cfg.key}")
    tomb_ver = sa.replicas[(cfg.key, "n0")].version
    result.tombstone_outversions_put = tomb_ver > put_ver
    # a genuinely newer put (a higher version than the tombstone) legitimately brings the key back
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"put n0 {cfg.key} {cfg.val2}")
    _, _, s_back, v_back = step_both(ref, sysb, sa, sb, f"get n0 {cfg.key}")
    result.newer_put_resurrects = s_back == "ok" and v_back == cfg.val2

    # --- Panel B: the resurrection problem under partition + repair (eventual) --------------------
    def resurrection_run(repair_cmd: str) -> bool:
        """True iff the minority converges to `deleted` (no resurrection) after `repair_cmd`."""
        config = _config(5, cfg.key, "eventual")
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"put n0 {cfg.key} {cfg.val}")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, "advance 5")  # replicate the value everywhere
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"delete n0 {cfg.key}")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, "advance 5")  # tombstone reaches n1,n2 only
        # the danger: the partitioned minority (n3) still reads the old value
        _, _, s_min, v_min = step_both(ref, sysb, sa, sb, f"get n3 {cfg.key}")
        result.minority_reads_deleted_item = s_min == "ok" and v_min == cfg.val
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, repair_cmd)
        _, _, s_after, _ = step_both(ref, sysb, sa, sb, f"get n3 {cfg.key}")
        return s_after == "deleted"  # the tombstone's higher version wins — no resurrection

    result.anti_entropy_no_resurrection = resurrection_run("anti_entropy n3")
    result.gossip_no_resurrection = resurrection_run("gossip n0 n3")

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED33Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"tombstone,delete_removes,{result.delete_removes_rate:.4f},"
                 f"deleted_on_all_over_{result.n_sizes}_clusters")
    lines.append(f"tombstone,outversions_put,"
                 f"{1.0 if result.tombstone_outversions_put else 0.0:.4f},tombstone_version_gt_put")
    lines.append(f"tombstone,newer_put_resurrects,"
                 f"{1.0 if result.newer_put_resurrects else 0.0:.4f},higher_version_write_back")
    lines.append(f"resurrection,minority_reads_item,"
                 f"{1.0 if result.minority_reads_deleted_item else 0.0:.4f},"
                 f"partitioned_minority_still_reads_old_value")
    lines.append(f"resurrection,anti_entropy_repairs,"
                 f"{1.0 if result.anti_entropy_no_resurrection else 0.0:.4f},"
                 f"tombstone_wins_no_resurrection")
    lines.append(f"resurrection,gossip_repairs,"
                 f"{1.0 if result.gossip_no_resurrection else 0.0:.4f},"
                 f"pairwise_repair_no_resurrection")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED33 (the tombstone delete: versioned removal + the resurrection problem)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed33.csv")
    parser.add_argument("--plot", type=str, default="figures/ed33.png")
    args = parser.parse_args()
    cfg = ED33Config.from_json_file(args.config) if args.config else ED33Config()
    result = run_ed33(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — delete is a versioned tombstone:")
    print(f"    deleted on all replicas: {result.delete_removes_rate:.2f} "
          f"| tombstone out-versions put: {result.tombstone_outversions_put} "
          f"| newer put resurrects: {result.newer_put_resurrects}")
    print("  Panel B — the resurrection problem under partition + repair:")
    print(f"    minority reads deleted item: {result.minority_reads_deleted_item} "
          f"| anti_entropy repairs (no resurrection): {result.anti_entropy_no_resurrection} "
          f"| gossip repairs: {result.gossip_no_resurrection}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed33 import plot_ed33

        plot_ed33(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
