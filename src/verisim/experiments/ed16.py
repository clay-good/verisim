"""ED16 — read-committed isolation: the lost-update anomaly + the cost of preventing it (DS0 inc9).

The experiment that pins the *weakest* transaction isolation level (SPEC-7 §3.2, §3.4) against the
two stronger ones ED9 already covers:

  - **read_committed** — the real-world default of Postgres / Oracle / SQL-Server — does **no**
    commit-time concurrency validation. Reads still see only committed data (the MVCC ``tget`` gives
    no dirty reads), but with no write-write check two read-modify-write transactions on the same
    key both commit and the later silently overwrites the earlier: the classic **lost-update**
    anomaly.
  - **snapshot** / **serializable** — both validate the write-set (first-committer-wins), so a
    same-key write-write conflict aborts the second committer: lost update cannot occur.

Two panels (mirroring ED9's write-skew / price-of-serializability shape):

  - **The lost-update anomaly (the headline).** The textbook scenario: two transactions both read
    `x` at the same version, then both write `x` (a read-modify-write). Under **read_committed**
    both commit (no validation) and only the later write survives — one update is lost. Under
    **snapshot**/**serializable** the second committer's write-set validation sees `x`'s version
    bumped by the first, so it aborts — the update is preserved. ED16 reports the **anomaly rate**
    per level: ≈1.0 under read_committed, 0.0 under snapshot and serializable.
  - **The price of preventing lost update.** Under a read-modify-write contended workload (each of
    `K` concurrent transactions reads one key and writes it back), read_committed commits *every*
    transaction (abort rate 0) — maximal apparent throughput — exactly because it admits the lost
    updates of panel 1; snapshot and serializable pay aborts to preserve every update. ED16 reports
    the abort rate per level with bootstrap CIs: the throughput read_committed buys is the
    correctness it sells.

All three levels compose with Tier-B: the autonomous-actor system oracle reproduces Tier-A on every
scenario (transaction bookkeeping is coordinator-local). Dependency-free, GPU-free.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import scaled_dist_config
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle
from verisim.metrics.aggregate import bootstrap_ci

# Ordered strong -> weak; read_committed is the new (weakest) level this increment adds.
ISOLATION_LEVELS = ("serializable", "snapshot", "read_committed")


@dataclass(frozen=True)
class ED16Config:
    name: str = "ed16"
    n_nodes: int = 3
    # lost-update panel: one same-key read-modify-write scenario per object index
    lost_update_objects: tuple[int, ...] = (0, 1, 2)
    # abort-rate panel: K concurrent read-modify-write txns over M objects (hot-key contention)
    n_txns: int = 8
    n_objects: int = 4
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    values: tuple[str, ...] = ("a", "b", "c", "d")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED16Config:
        b = ED16Config()
        return ED16Config(
            name=d.get("name", b.name),
            n_nodes=d.get("n_nodes", b.n_nodes),
            lost_update_objects=tuple(d.get("lost_update_objects", b.lost_update_objects)),
            n_txns=d.get("n_txns", b.n_txns),
            n_objects=d.get("n_objects", b.n_objects),
            seeds=tuple(d.get("seeds", b.seeds)),
            values=tuple(d.get("values", b.values)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED16Config:
        return ED16Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED16Result:
    lost_update: list[dict[str, Any]] = field(default_factory=list)  # per isolation level
    abort_rate: list[dict[str, Any]] = field(default_factory=list)  # per isolation level


# --- the lost-update panel ------------------------------------------------------------------------

def _lost_update_trajectory(node: str, key: str) -> list[str]:
    """Two txns both read ``key`` at the same version, then both write it (read-modify-write)."""
    return [
        f"begin {node} A", f"begin {node} B",
        f"tget {node} A {key}", f"tget {node} B {key}",  # both read the same pre-version
        f"tput {node} A {key} b", f"tput {node} B {key} c",  # both write back distinct values
        f"commit {node} A", f"commit {node} B",
    ]


def _run_lost_update(cfg: ED16Config, isolation: str) -> dict[str, Any]:
    config = scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, txn_isolation=isolation)
    ref = ReferenceDistOracle(config)
    sys_oracle = SystemDistOracle(config)
    node = config.nodes[0]
    anomalies = 0
    agree = True
    for oi in cfg.lost_update_objects:
        key = f"o{oi}"
        cmds = _lost_update_trajectory(node, key)
        s = DistributedState.initial(config)
        s_sys = DistributedState.initial(config)
        commits = 0
        for cmd in cmds:
            action = parse_dist_action(cmd)
            r = ref.step(s, action)
            r_sys = sys_oracle.step(s_sys, action)
            if cluster_view(r.state) != cluster_view(r_sys.state):
                agree = False
            if r.status == "committed":
                commits += 1
            s, s_sys = r.state, r_sys.state
        # lost update = BOTH txns commit after reading the same version (the earlier write is gone).
        if commits == 2:
            anomalies += 1
    n = len(cfg.lost_update_objects)
    return {
        "isolation": isolation,
        "scenarios": n,
        "anomalies": anomalies,
        "anomaly_rate": anomalies / n if n else 0.0,
        "tier_b_agrees": agree,
    }


# --- the abort-rate panel -------------------------------------------------------------------------

def _contention_trajectory(cfg: ED16Config, node: str, rng: random.Random) -> list[str]:
    """K read-modify-write txns on hot keys: each reads one key then writes it back (RMW)."""
    keys = [f"o{i}" for i in range(cfg.n_objects)]
    rmw_keys = [rng.choice(keys) for _ in range(cfg.n_txns)]
    cmds = [f"begin {node} t{i}" for i in range(cfg.n_txns)]
    for i in range(cfg.n_txns):  # all reads happen before any commit (the concurrency race)
        cmds.append(f"tget {node} t{i} {rmw_keys[i]}")
    cmds += [f"tput {node} t{i} {rmw_keys[i]} {rng.choice(cfg.values)}" for i in range(cfg.n_txns)]
    cmds += [f"commit {node} t{i}" for i in range(cfg.n_txns)]
    return cmds


def _run_abort_rate(cfg: ED16Config, isolation: str) -> dict[str, Any]:
    config = scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, txn_isolation=isolation)
    ref = ReferenceDistOracle(config)
    node = config.nodes[0]
    rates: list[float] = []
    for seed in cfg.seeds:
        rng = random.Random(seed * 7919 + len(isolation))
        cmds = _contention_trajectory(cfg, node, rng)
        s = DistributedState.initial(config)
        aborts = 0
        for cmd in cmds:
            r = ref.step(s, parse_dist_action(cmd))
            if r.status == "conflict":
                aborts += 1
            s = r.state
        rates.append(aborts / cfg.n_txns)
    lo, hi = bootstrap_ci(rates, seed=0)
    return {"isolation": isolation, "abort_rate": fmean(rates), "ci_lo": lo, "ci_hi": hi}


def run_ed16(cfg: ED16Config | None = None) -> ED16Result:
    cfg = cfg or ED16Config()
    return ED16Result(
        lost_update=[_run_lost_update(cfg, iso) for iso in ISOLATION_LEVELS],
        abort_rate=[_run_abort_rate(cfg, iso) for iso in ISOLATION_LEVELS],
    )


CSV_HEADER = "panel,isolation,metric,value,ci_lo,ci_hi,detail"


def write_csv(result: ED16Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.lost_update:
        lines.append(f"lost_update,{r['isolation']},anomaly_rate,{r['anomaly_rate']:.4f},,,"
                     f"anomalies={r['anomalies']}/{r['scenarios']};tier_b_agrees={r['tier_b_agrees']}")
    for r in result.abort_rate:
        lines.append(f"abort_rate,{r['isolation']},abort_rate,{r['abort_rate']:.4f},"
                     f"{r['ci_lo']:.4f},{r['ci_hi']:.4f},")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED16 (read-committed isolation: lost update + the price of preventing it)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed16.csv")
    parser.add_argument("--plot", type=str, default="figures/ed16.png")
    args = parser.parse_args()
    cfg = ED16Config.from_json_file(args.config) if args.config else ED16Config()
    result = run_ed16(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  lost-update anomaly rate (both RMW txns commit the same key):")
    for r in result.lost_update:
        verdict = "LOST UPDATE admitted" if r["anomaly_rate"] > 0 else "lost update forbidden"
        print(f"    [{r['isolation']:14s}] {r['anomaly_rate']:.2f} "
              f"({r['anomalies']}/{r['scenarios']}) -> {verdict}  "
              f"Tier-B agrees={r['tier_b_agrees']}")
    print("  abort rate under read-modify-write contention (the price of preventing it):")
    for r in result.abort_rate:
        print(f"    [{r['isolation']:14s}] {r['abort_rate']:.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    try:
        from figures.plot_ed16 import plot_ed16

        plot_ed16(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
