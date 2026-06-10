"""ED32 — the quorum-confirmed linearizable read: Raft ReadIndex, the partner to the lease read.

The DS0-increment-25 experiment for SPEC-7 §5.1 — `read_index`, the *other* way Raft serves a
linearizable read (the partner to `lread`, the lease read of ED25/incr 18). Where `lread` skips the
quorum round-trip by relying on a time **lease**, `read_index` keeps no clock assumption and instead
**confirms leadership with a majority** before serving the read (the ReadIndex heartbeat round). The
two reads have **opposite availability/safety profiles**, and that contrast is the experiment. Both
panels are measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — the two linearizable reads, opposite availability.** Across a cluster-size sweep, a
    `read_index` at the leader with full connectivity **serves the read** (rate **1.0**); a
    `read_index` by a **non-leader** is `not_leader` (**1.0**); a leader **stranded in a minority**
    is `no_quorum` (**1.0**) — it cannot confirm it is still leader, so it refuses. The sharp
    contrast: that same minority leader, holding a **live lease**, *can* serve `lread` locally
    (**1.0**) where its `read_index` is `no_quorum` — the read-availability the lease buys and the
    quorum read declines (and the clock dependence the quorum read avoids in return).

  - **Panel B — linearizable safety + freshness.** A `read_index` reflects the **latest committed
    value** after an `append` (**1.0**). The safety: a leader **deposed** by a higher-term
    election — partitioned away while the majority elected a new leader and committed a newer — is
    `not_leader` on `read_index` **even after `heal`** (**1.0**), *refusing* to serve its now-stale
    local replica, where a plain `get` from that same node **serves the stale value** (the read
    `read_index` exists to prevent). The new leader's `read_index` returns the **fresh committed
    value** (**1.0**).

`read_index` reads its majority-confirmation from the partition/down medium (a coordinator-level
decision, like `propose`/`append`) and touches no replica (a pure read), so the autonomous-actor
system oracle (Tier-B) reproduces every read verdict bit-for-bit (§5.2). It adds no state field and
no edit type — purely additive, no prior golden/hash/tokenization changes.
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
class ED32Config:
    name: str = "ed32-read-index"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "x"
    val: str = "b"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED32Config:
        b = ED32Config()
        return ED32Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            val=d.get("val", b.val),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED32Config:
        return ED32Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED32Result:
    #: Panel A: a read_index at the leader with full connectivity serves the read.
    read_index_ok_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: a read_index by a non-leader is fenced (`not_leader`).
    nonleader_fenced: bool = False
    #: Panel A: a minority-stranded leader's read_index is `no_quorum`.
    minority_no_quorum: bool = False
    #: Panel A: the lease/quorum contrast — a minority leader with a live lease serves lread where
    #: read_index is no_quorum (the read-availability the lease buys, the quorum read declines).
    lease_serves_where_quorum_refuses: bool = False
    #: Panel B: read_index reflects the latest committed value after an append.
    reflects_committed: bool = False
    #: Panel B: a deposed leader's read_index is `not_leader` even after heal (refuses stale read).
    deposed_read_index_fenced: bool = False
    #: Panel B: the safety contrast — a plain `get` on that deposed node serves the stale value
    #: where read_index refuses, and the new leader's read_index returns the fresh committed value.
    stale_get_vs_safe_read_index: bool = False
    #: Tier-B reproduces every read verdict bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, read served at leader)


def _config(n: int, key: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed32-{n}n", nodes=nodes, objects=(key,), values=("a", "b"),
                      replication_factor=n, consistency_model="quorum")


def run_ed32(cfg: ED32Config | None = None) -> ED32Result:
    cfg = cfg or ED32Config()
    result = ED32Result()
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

    # --- Panel A: the two linearizable reads, opposite availability -------------------------------
    ok_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"elect {nodes[0]}")
        sa, sb, s_read, _ = step_both(ref, sysb, sa, sb, f"read_index {nodes[0]} {cfg.key}")
        served = s_read == "ok"
        sizes += 1
        if served:
            ok_count += 1
        result.per_size.append((n, served))
    result.read_index_ok_rate = ok_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # the fence + the lease/quorum contrast, on a 5-node cluster.
    config = _config(5, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    nodes = config.nodes  # n0..n4
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, s_nonleader, _ = step_both(ref, sysb, sa, sb, f"read_index n1 {cfg.key}")
    result.nonleader_fenced = s_nonleader == "not_leader"
    # strand the leader n0 in the 2-of-5 minority {n0,n1}
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 | n2 n3 n4")
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"read_index n0 {cfg.key}")
    result.minority_no_quorum = s_min == "no_quorum"
    # the same minority leader takes a lease and serves lread locally where read_index is no_quorum
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "lease n0 100")
    sa, sb, s_lread, _ = step_both(ref, sysb, sa, sb, f"lread n0 {cfg.key}")
    sa, sb, s_min2, _ = step_both(ref, sysb, sa, sb, f"read_index n0 {cfg.key}")
    result.lease_serves_where_quorum_refuses = s_lread == "ok" and s_min2 == "no_quorum"

    # --- Panel B: linearizable safety + freshness ------------------------------------------------
    config = _config(5, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n0")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"append n0 {cfg.key} a")  # commit a value first
    sa, sb, s_fresh, v_fresh = step_both(ref, sysb, sa, sb, f"read_index n0 {cfg.key}")
    result.reflects_committed = s_fresh == "ok" and v_fresh == "a"
    # n0 partitioned ALONE; the majority side elects n2 and commits a *newer* value n0 never sees
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 | n1 n2 n3 n4")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "elect n2")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"append n2 {cfg.key} {cfg.val}")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    # n0 is deposed and its local replica is stale; read_index refuses, plain get serves it stale.
    sa, sb, s_deposed, _ = step_both(ref, sysb, sa, sb, f"read_index n0 {cfg.key}")
    result.deposed_read_index_fenced = s_deposed == "not_leader"
    _, _, s_get, v_get = step_both(ref, sysb, sa, sb, f"get n0 {cfg.key}")  # serves stale local
    _, _, s_leader_read, v_leader_read = step_both(ref, sysb, sa, sb, f"read_index n2 {cfg.key}")
    result.stale_get_vs_safe_read_index = (
        s_get == "ok" and v_get != cfg.val            # the plain get returns the *stale* value
        and s_leader_read == "ok" and v_leader_read == cfg.val  # the leader's read_index is fresh
    )

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED32Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"reads,read_index_ok,{result.read_index_ok_rate:.4f},"
                 f"served_at_leader_over_{result.n_sizes}_clusters")
    lines.append(f"reads,nonleader_fenced,"
                 f"{1.0 if result.nonleader_fenced else 0.0:.4f},non_leader_read_index_not_leader")
    lines.append(f"reads,minority_no_quorum,"
                 f"{1.0 if result.minority_no_quorum else 0.0:.4f},minority_leader_cannot_confirm")
    lines.append(f"reads,lease_serves_where_quorum_refuses,"
                 f"{1.0 if result.lease_serves_where_quorum_refuses else 0.0:.4f},"
                 f"lread_ok_in_minority_where_read_index_no_quorum")
    lines.append(f"safety,reflects_committed,"
                 f"{1.0 if result.reflects_committed else 0.0:.4f},read_index_returns_committed")
    lines.append(f"safety,deposed_fenced,"
                 f"{1.0 if result.deposed_read_index_fenced else 0.0:.4f},"
                 f"deposed_leader_read_index_not_leader_after_heal")
    lines.append(f"safety,stale_get_vs_safe_read_index,"
                 f"{1.0 if result.stale_get_vs_safe_read_index else 0.0:.4f},"
                 f"get_serves_stale_where_read_index_refuses_and_leader_fresh")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED32 (read_index: the quorum-confirmed linearizable read, Raft ReadIndex)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed32.csv")
    parser.add_argument("--plot", type=str, default="figures/ed32.png")
    args = parser.parse_args()
    cfg = ED32Config.from_json_file(args.config) if args.config else ED32Config()
    result = run_ed32(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — the two linearizable reads, opposite availability:")
    print(f"    read_index served at leader: {result.read_index_ok_rate:.2f} "
          f"| non-leader fenced: {result.nonleader_fenced} "
          f"| minority no_quorum: {result.minority_no_quorum} "
          f"| lease serves where quorum refuses: {result.lease_serves_where_quorum_refuses}")
    print("  Panel B — linearizable safety + freshness:")
    print(f"    reflects committed: {result.reflects_committed} "
          f"| deposed read_index fenced: {result.deposed_read_index_fenced} "
          f"| stale get vs safe read_index: {result.stale_get_vs_safe_read_index}")
    print(f"  Tier-B reproduces every read verdict bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed32 import plot_ed32

        plot_ed32(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
