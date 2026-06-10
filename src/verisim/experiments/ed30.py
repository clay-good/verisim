"""ED30 — the embedded host: each cluster node runs a real SPEC-6 host.

The DS0-increment-23 experiment for SPEC-7 §3.1/§4 — `host`, the compositional vision the spec names
since increment 1 (`HostDelta(...)` on an embedded subsystem). A cluster node is no longer just a
bag of KV replicas: it runs a real SPEC-6 host (a process table + per-process fd tables + an
embedded v0 filesystem), and `host node <syscall>` delegates to the SPEC-6 `ReferenceHostOracle` on
that node's own host. Two properties, measured dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — composition + per-node isolation.** Across a cluster-size sweep, a `fork` on a node
    spawns a process on **that node's host only** — never on another's (per-node isolation, rate
    **1.0**). A node serves a KV `put` *and* a host `fork` independently (the two subsystems coexist
    on one node, **1.0**), and a host `open` + `write` materializes the file in **that node's
    embedded v0 filesystem** (the composition runs all the way down to the FS sub-oracle, **1.0**).

  - **Panel B — the cross-layer crash linkage.** Host ops respect the node's up/down: a `host`
    syscall on a **crashed** node is `unavailable` (**1.0**) — the same gate the KV client ops obey,
    now reaching the embedded host. `restart` restores host ops (**1.0**), and the host **state
    survives the crash** (a process forked before the crash is still there after restart, and the
    next `fork` allocates the following pid — **1.0**): a crash pauses the node, it does not wipe.

`host` delegates to the SPEC-6 host oracle on the node's own state (a node-local computation, no
medium interaction), so the autonomous-actor system oracle (Tier-B) reproduces every host transition
bit-for-bit (the W1 retirement, §5.2). The embedded hosts join the observable `cluster_view`, and
the `hosts` map is omitted from the canonical form until the first `host` op, so the op is purely
additive — no prior golden/hash/tokenization changes.
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
class ED30Config:
    name: str = "ed30-host"
    cluster_sizes: tuple[int, ...] = (3, 4, 5)
    key: str = "x"
    path: str = "/f"  # a top-level file (parent / exists in the boot FS)
    token: str = "a"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED30Config:
        b = ED30Config()
        return ED30Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            path=d.get("path", b.path),
            token=d.get("token", b.token),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED30Config:
        return ED30Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED30Result:
    #: Panel A: a fork on a node creates the process only on that node's host (per-node isolation).
    fork_isolated_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: a node serves a KV put and a host fork independently (the subsystems coexist).
    kv_and_host_coexist: bool = False
    #: Panel A: open + write materializes the file in the node's embedded v0 filesystem.
    embedded_fs_works: bool = False
    #: Panel B: a host syscall on a crashed node is unavailable (the cross-layer crash gate).
    crashed_host_unavailable: bool = False
    #: Panel B: host ops work again after restart.
    restart_restores: bool = False
    #: Panel B: the host state survives the crash (a pre-crash proc persists; pids keep counting).
    host_state_survives_crash: bool = False
    #: Tier-B reproduces every host transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, isolated)


def _config(n: int, key: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed30-{n}n", nodes=nodes, objects=(key,), replication_factor=n)


def run_ed30(cfg: ED30Config | None = None) -> ED30Result:
    cfg = cfg or ED30Config()
    result = ED30Result()
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

    # --- Panel A: composition + per-node isolation ------------------------------------------------
    isolated_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        sa = sb = DistributedState.initial(config)
        # fork a child on n0's host; it must appear on n0's host and on no other node's host.
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"host {nodes[0]} fork 1")
        leader_host = sa.hosts.get(nodes[0])
        leader_has_child = leader_host is not None and 2 in leader_host.procs
        others_clean = all(nd not in sa.hosts for nd in nodes[1:])  # no host created elsewhere
        isolated = leader_has_child and others_clean
        sizes += 1
        if isolated:
            isolated_count += 1
        result.per_size.append((n, isolated))
    result.fork_isolated_rate = isolated_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # KV + host coexist on one node, and the embedded FS works (open + write -> file in host FS).
    config = _config(3, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, s_put, _ = step_both(ref, sysb, sa, sb, f"put n0 {cfg.key} a")
    sa, sb, s_fork, _ = step_both(ref, sysb, sa, sb, "host n0 fork 1")
    result.kv_and_host_coexist = (
        s_put == "ok" and s_fork == "ok"
        and sa.replicas[(cfg.key, "n0")].value == "a"  # the KV write landed
        and 2 in sa.hosts["n0"].procs  # the fork landed, on the same node
    )
    sa, sb, s_open, fd = step_both(ref, sysb, sa, sb, f"host n0 open 1 {cfg.path}")
    sa, sb, s_write, _ = step_both(ref, sysb, sa, sb, f"host n0 write 1 {fd} {cfg.token}")
    fs_node = sa.hosts["n0"].fs.fs.get(cfg.path)  # the v0 State's path->Node map
    result.embedded_fs_works = (
        s_open == "ok" and s_write == "ok"
        and fs_node is not None and getattr(fs_node, "content", None) == cfg.token
    )

    # --- Panel B: the cross-layer crash linkage ---------------------------------------------------
    config = _config(3, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, pid_before = step_both(ref, sysb, sa, sb, "host n0 fork 1")  # pid 2 before the crash
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "crash n0")
    _, _, s_crashed, _ = step_both(ref, sysb, sa, sb, "host n0 fork 1")  # crashed -> unavailable
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "restart n0")
    sa, sb, s_after, pid_after = step_both(ref, sysb, sa, sb, "host n0 fork 1")  # works again
    result.crashed_host_unavailable = s_crashed == "unavailable"
    result.restart_restores = s_after == "ok"
    # the pre-crash process (pid 2) is still present, and the post-restart fork allocates pid 3 —
    # host state (process table) survived the crash rather than being reset.
    result.host_state_survives_crash = (
        2 in sa.hosts["n0"].procs and pid_before == "2" and pid_after == "3"
    )

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED30Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"compose,fork_isolated,{result.fork_isolated_rate:.4f},"
                 f"per_node_over_{result.n_sizes}_clusters")
    lines.append(f"compose,kv_and_host_coexist,"
                 f"{1.0 if result.kv_and_host_coexist else 0.0:.4f},kv_put_plus_host_fork")
    lines.append(f"compose,embedded_fs_works,"
                 f"{1.0 if result.embedded_fs_works else 0.0:.4f},open_write_into_host_fs")
    lines.append(f"crash,crashed_host_unavailable,"
                 f"{1.0 if result.crashed_host_unavailable else 0.0:.4f},host_op_gated_by_up_down")
    lines.append(f"crash,restart_restores,"
                 f"{1.0 if result.restart_restores else 0.0:.4f},host_ops_work_after_restart")
    lines.append(f"crash,host_state_survives_crash,"
                 f"{1.0 if result.host_state_survives_crash else 0.0:.4f},proc_table_persists")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED30 (the embedded host: each cluster node runs a real SPEC-6 host)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed30.csv")
    parser.add_argument("--plot", type=str, default="figures/ed30.png")
    args = parser.parse_args()
    cfg = ED30Config.from_json_file(args.config) if args.config else ED30Config()
    result = run_ed30(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — composition + per-node isolation:")
    print(f"    fork isolated per node: {result.fork_isolated_rate:.2f} "
          f"| KV + host coexist: {result.kv_and_host_coexist} "
          f"| embedded FS works: {result.embedded_fs_works}")
    print("  Panel B — the cross-layer crash linkage:")
    print(f"    crashed host unavailable: {result.crashed_host_unavailable} "
          f"| restart restores: {result.restart_restores} "
          f"| host state survives crash: {result.host_state_survives_crash}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed30 import plot_ed30

        plot_ed30(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
