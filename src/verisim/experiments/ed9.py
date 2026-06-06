"""ED9 — transaction isolation: the write-skew anomaly + the price of serializability (DS0 incr 3).

The experiment that pins the difference between the two transaction isolation levels (SPEC-7 §3.2):

  - **serializable** — OCC backward validation of the **read-set** (a committed transaction's read
    versions must be unchanged). Catches the read another transaction's write invalidated, so it
    **forbids write skew**.
  - **snapshot** — validation of **write-write** conflicts only (the write-set versions,
    first-committer-wins). A read another transaction wrote is *not* checked, so two transactions
    with disjoint write-sets both commit — **write skew**, the classic SI anomaly.

Two panels:

  - **The write-skew anomaly (the headline).** The textbook scenario: two transactions both read
    `{x, y}`, then `A` writes `x` and `B` writes `y`. Under **snapshot** both commit (the write-sets
    `{x}` and `{y}` are disjoint) — a serial schedule could never produce that pair of outcomes, so
    the cross-object invariant they each checked is silently violated. Under **serializable** the
    second committer's read of the key the first wrote is invalidated, so it aborts and the anomaly
    cannot occur. ED9 reports the **anomaly rate** per level: ≈1.0 under snapshot, 0.0 under
    serializable.
  - **The price of serializability.** Under a read-heavy contended workload (each of `K` concurrent
    transactions reads two keys and writes one, over `M` objects), serializable aborts strictly more
    than snapshot — it pays extra aborts to buy the stronger guarantee. ED9 reports both abort rates
    with bootstrap CIs.

Both levels compose with Tier-B: the autonomous-actor system oracle reproduces Tier-A on every
scenario. Dependency-free, GPU-free.
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

ISOLATION_LEVELS = ("serializable", "snapshot")


@dataclass(frozen=True)
class ED9Config:
    name: str = "ed9"
    n_nodes: int = 3
    # write-skew panel: pairs of constraint objects (each pair gives one write-skew scenario)
    skew_object_pairs: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2))
    # abort-rate panel: K concurrent txns, each reads 2 keys + writes 1, over M objects
    n_txns: int = 8
    n_objects: int = 4
    reads_per_txn: int = 2
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    values: tuple[str, ...] = ("a", "b", "c", "d")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED9Config:
        b = ED9Config()
        return ED9Config(
            name=d.get("name", b.name),
            n_nodes=d.get("n_nodes", b.n_nodes),
            skew_object_pairs=tuple(
                tuple(p) for p in d.get("skew_object_pairs", b.skew_object_pairs)
            ),
            n_txns=d.get("n_txns", b.n_txns),
            n_objects=d.get("n_objects", b.n_objects),
            reads_per_txn=d.get("reads_per_txn", b.reads_per_txn),
            seeds=tuple(d.get("seeds", b.seeds)),
            values=tuple(d.get("values", b.values)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED9Config:
        return ED9Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED9Result:
    write_skew: list[dict[str, Any]] = field(default_factory=list)  # per isolation level
    abort_rate: list[dict[str, Any]] = field(default_factory=list)  # per isolation level


# --- the write-skew panel -------------------------------------------------------------------------

def _write_skew_trajectory(node: str, key_a: str, key_b: str) -> list[str]:
    """Two txns both read {key_a, key_b}; A writes key_a, B writes key_b (disjoint write-sets)."""
    return [
        f"begin {node} A", f"begin {node} B",
        f"tget {node} A {key_a}", f"tget {node} A {key_b}",
        f"tget {node} B {key_a}", f"tget {node} B {key_b}",
        f"tput {node} A {key_a} a", f"tput {node} B {key_b} b",
        f"commit {node} A", f"commit {node} B",
    ]


def _run_write_skew(cfg: ED9Config, isolation: str) -> dict[str, Any]:
    config = scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, txn_isolation=isolation)
    ref = ReferenceDistOracle(config)
    sys_oracle = SystemDistOracle(config)
    node = config.nodes[0]
    anomalies = 0
    agree = True
    for ai, bi in cfg.skew_object_pairs:
        cmds = _write_skew_trajectory(node, f"o{ai}", f"o{bi}")
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
        # write skew = BOTH committed (a serial schedule forbids that disjoint-write pair)
        if commits == 2:
            anomalies += 1
    n = len(cfg.skew_object_pairs)
    return {
        "isolation": isolation,
        "scenarios": n,
        "anomalies": anomalies,
        "anomaly_rate": anomalies / n if n else 0.0,
        "tier_b_agrees": agree,
    }


# --- the abort-rate panel -------------------------------------------------------------------------

def _contention_trajectory(cfg: ED9Config, node: str, rng: random.Random) -> list[str]:
    """K txns, each reads `reads_per_txn` distinct keys then writes one; all read pre-state."""
    keys = [f"o{i}" for i in range(cfg.n_objects)]
    read_sets = [rng.sample(keys, min(cfg.reads_per_txn, len(keys))) for _ in range(cfg.n_txns)]
    write_keys = [rng.choice(read_sets[i]) for i in range(cfg.n_txns)]  # write a key it read
    cmds = [f"begin {node} t{i}" for i in range(cfg.n_txns)]
    for i in range(cfg.n_txns):  # all reads happen before any commit (the OCC race)
        cmds += [f"tget {node} t{i} {k}" for k in read_sets[i]]
    cmds += [
        f"tput {node} t{i} {write_keys[i]} {rng.choice(cfg.values)}" for i in range(cfg.n_txns)
    ]
    cmds += [f"commit {node} t{i}" for i in range(cfg.n_txns)]
    return cmds


def _run_abort_rate(cfg: ED9Config, isolation: str) -> dict[str, Any]:
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


def run_ed9(cfg: ED9Config | None = None) -> ED9Result:
    cfg = cfg or ED9Config()
    return ED9Result(
        write_skew=[_run_write_skew(cfg, iso) for iso in ISOLATION_LEVELS],
        abort_rate=[_run_abort_rate(cfg, iso) for iso in ISOLATION_LEVELS],
    )


CSV_HEADER = "panel,isolation,metric,value,ci_lo,ci_hi,detail"


def write_csv(result: ED9Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.write_skew:
        lines.append(f"write_skew,{r['isolation']},anomaly_rate,{r['anomaly_rate']:.4f},,,"
                     f"anomalies={r['anomalies']}/{r['scenarios']};tier_b_agrees={r['tier_b_agrees']}")
    for r in result.abort_rate:
        lines.append(f"abort_rate,{r['isolation']},abort_rate,{r['abort_rate']:.4f},"
                     f"{r['ci_lo']:.4f},{r['ci_hi']:.4f},")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED9 (transaction isolation: write-skew + the price of serializability)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed9.csv")
    parser.add_argument("--plot", type=str, default="figures/ed9.png")
    args = parser.parse_args()
    cfg = ED9Config.from_json_file(args.config) if args.config else ED9Config()
    result = run_ed9(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  write-skew anomaly rate (both txns commit a disjoint-write pair):")
    for r in result.write_skew:
        verdict = "WRITE SKEW admitted" if r["anomaly_rate"] > 0 else "write skew forbidden"
        print(f"    [{r['isolation']:12s}] {r['anomaly_rate']:.2f} "
              f"({r['anomalies']}/{r['scenarios']}) -> {verdict}  "
              f"Tier-B agrees={r['tier_b_agrees']}")
    print("  abort rate under read-heavy contention (the price of serializability):")
    for r in result.abort_rate:
        print(f"    [{r['isolation']:12s}] {r['abort_rate']:.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    try:
        from figures.plot_ed9 import plot_ed9

        plot_ed9(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
