"""ED10 — Elle: black-box serializability checking of the transaction history (DS3 incr 2).

ED9 detected the write-skew anomaly the way an *omniscient* observer would: it counted how many of a
pair of disjoint-write transactions the oracle let commit. ED10 asks the harder, more useful
question: can a checker that sees **only the client-visible history** — what each committed
transaction read and wrote, no oracle, no cluster state — recover the same verdict? This is the
Jepsen/Elle thesis (Kingsbury & Alvaro, VLDB 2020), and it is the reference-free, *stronger*-
consistency form of the per-step ``cycle`` tier the DS3 milestone deferred. The checker
(:mod:`verisim.distoracle.elle`) reconstructs Adya's Direct Serialization Graph from the history and
reports a cycle iff the schedule was non-serializable.

Two panels, mirroring ED9:

  - **The write-skew anomaly, recovered black-box (the headline).** For each isolation level, run
    the textbook write-skew scenarios, record only the committed transactions' read/write versions,
    and hand that history to Elle. Under **snapshot** both transactions commit and Elle reports a
    **G2 anti-dependency cycle** (``A →rw B →rw A``) — the write skew, found with no oracle. Under
    **serializable** the second committer aborts, so the history has one txn and Elle reports no
    cycle. ED10's ``elle_g2_rate`` equals ED9's oracle-side ``anomaly_rate`` (1.0 vs 0.0) — the
    free black-box checker recovers the exact anomaly the expensive oracle sees, and we report the
    per-scenario agreement (``elle_matches_oracle``) to prove it.
  - **Elle certifies the serializable level (the verifier result).** Under a read-heavy contended
    workload, Elle flags a positive fraction of **snapshot** histories non-serializable (the
    anomalies that level admits) and **zero serializable** histories (the guarantee that level
    enforces, certified independently of the oracle that enforces it), with bootstrap CIs.

Pure standard library, dependency-free, GPU-free.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig, scaled_dist_config
from verisim.dist.state import DistributedState
from verisim.distoracle.elle import TxnObservation, check_serializable
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.metrics.aggregate import bootstrap_ci

ISOLATION_LEVELS = ("serializable", "snapshot")


@dataclass(frozen=True)
class ED10Config:
    name: str = "ed10"
    n_nodes: int = 3
    # write-skew panel: pairs of constraint objects (each pair = one write-skew scenario)
    skew_object_pairs: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2))
    # contention panel: K concurrent txns, each reads 2 keys + writes 1, over M objects
    n_txns: int = 8
    n_objects: int = 4
    reads_per_txn: int = 2
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    values: tuple[str, ...] = ("a", "b", "c", "d")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED10Config:
        b = ED10Config()
        return ED10Config(
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
    def from_json_file(path: str | Path) -> ED10Config:
        return ED10Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED10Result:
    write_skew: list[dict[str, Any]] = field(default_factory=list)  # per isolation level
    contention: list[dict[str, Any]] = field(default_factory=list)  # per isolation level


# --- the observable history (only committed txns, only their read/installed versions) -------------

def collect_history(
    config: DistConfig, cmds: list[str]
) -> tuple[list[TxnObservation], int]:
    """Run ``cmds`` through the reference oracle and return (committed history, #commits).

    The history is everything Elle is allowed to see: each committed transaction's read versions
    (the ``(key, version)`` it pinned on a genuine replica read — read-your-writes reads are not
    cross-transaction dependencies and are excluded) and the MVCC version it installed per written
    key. The oracle's internal state is *not* part of the returned history; only the client-visible
    read/write footprint is, exactly the signal a black-box checker has.
    """
    ref = ReferenceDistOracle(config)
    s = DistributedState.initial(config)
    history: list[TxnObservation] = []
    commits = 0
    for cmd in cmds:
        action = parse_dist_action(cmd)
        pre = s
        r = ref.step(s, action)
        if action.name == "commit" and r.status == "committed":
            commits += 1
            node, txn_id = action.args[0], action.args[1]
            txn = pre.txns[txn_id]
            written_keys = {k for k, _ in txn.writes}
            writes = tuple(
                sorted((k, r.state.replicas[(k, node)].version) for k in written_keys)
            )
            history.append(TxnObservation(txn_id, reads=txn.reads, writes=writes))
        s = r.state
    return history, commits


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


def _run_write_skew(cfg: ED10Config, isolation: str) -> dict[str, Any]:
    config = scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, txn_isolation=isolation)
    node = config.nodes[0]
    g2 = 0
    matches = 0
    n = len(cfg.skew_object_pairs)
    for ai, bi in cfg.skew_object_pairs:
        cmds = _write_skew_trajectory(node, f"o{ai}", f"o{bi}")
        history, commits = collect_history(config, cmds)
        report = check_serializable(history)
        is_g2 = (not report.serializable) and report.anomaly == "G2"
        if is_g2:
            g2 += 1
        # the oracle-side write-skew flag (ED9): both disjoint-write txns committed
        oracle_anomaly = commits == 2
        if is_g2 == oracle_anomaly:  # black-box Elle agrees with the omniscient oracle
            matches += 1
    return {
        "isolation": isolation,
        "scenarios": n,
        "g2_cycles": g2,
        "elle_g2_rate": g2 / n if n else 0.0,
        "elle_matches_oracle": matches == n,
    }


# --- the contention panel -------------------------------------------------------------------------

def _contention_trajectory(cfg: ED10Config, node: str, rng: random.Random) -> list[str]:
    """K txns, each reads `reads_per_txn` distinct keys then writes one it read (the OCC race)."""
    keys = [f"o{i}" for i in range(cfg.n_objects)]
    read_sets = [rng.sample(keys, min(cfg.reads_per_txn, len(keys))) for _ in range(cfg.n_txns)]
    write_keys = [rng.choice(read_sets[i]) for i in range(cfg.n_txns)]
    cmds = [f"begin {node} t{i}" for i in range(cfg.n_txns)]
    for i in range(cfg.n_txns):  # all reads before any commit
        cmds += [f"tget {node} t{i} {k}" for k in read_sets[i]]
    cmds += [
        f"tput {node} t{i} {write_keys[i]} {rng.choice(cfg.values)}" for i in range(cfg.n_txns)
    ]
    cmds += [f"commit {node} t{i}" for i in range(cfg.n_txns)]
    return cmds


def _run_contention(cfg: ED10Config, isolation: str) -> dict[str, Any]:
    config = scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, txn_isolation=isolation)
    node = config.nodes[0]
    flags: list[float] = []  # 1.0 if Elle flagged this history non-serializable
    anomalies: dict[str, int] = {}
    for seed in cfg.seeds:
        rng = random.Random(seed * 7919 + len(isolation))
        cmds = _contention_trajectory(cfg, node, rng)
        history, _ = collect_history(config, cmds)
        report = check_serializable(history)
        flags.append(0.0 if report.serializable else 1.0)
        if not report.serializable:
            anomalies[report.anomaly] = anomalies.get(report.anomaly, 0) + 1
    lo, hi = bootstrap_ci(flags, seed=0)
    return {
        "isolation": isolation,
        "nonserializable_rate": fmean(flags),
        "ci_lo": lo,
        "ci_hi": hi,
        "anomalies": anomalies,
    }


def run_ed10(cfg: ED10Config | None = None) -> ED10Result:
    cfg = cfg or ED10Config()
    return ED10Result(
        write_skew=[_run_write_skew(cfg, iso) for iso in ISOLATION_LEVELS],
        contention=[_run_contention(cfg, iso) for iso in ISOLATION_LEVELS],
    )


CSV_HEADER = "panel,isolation,metric,value,ci_lo,ci_hi,detail"


def write_csv(result: ED10Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.write_skew:
        lines.append(
            f"write_skew,{r['isolation']},elle_g2_rate,{r['elle_g2_rate']:.4f},,,"
            f"g2={r['g2_cycles']}/{r['scenarios']};matches_oracle={r['elle_matches_oracle']}"
        )
    for r in result.contention:
        anomalies = ";".join(f"{k}={v}" for k, v in sorted(r["anomalies"].items())) or "none"
        lines.append(
            f"contention,{r['isolation']},nonserializable_rate,{r['nonserializable_rate']:.4f},"
            f"{r['ci_lo']:.4f},{r['ci_hi']:.4f},{anomalies}"
        )
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED10 (Elle: black-box serializability checking of the txn history)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed10.csv")
    parser.add_argument("--plot", type=str, default="figures/ed10.png")
    args = parser.parse_args()
    cfg = ED10Config.from_json_file(args.config) if args.config else ED10Config()
    result = run_ed10(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  write-skew: Elle's G2-cycle rate, recovered black-box from the history:")
    for r in result.write_skew:
        verdict = "G2 write-skew cycle" if r["elle_g2_rate"] > 0 else "no cycle (serializable)"
        print(f"    [{r['isolation']:12s}] {r['elle_g2_rate']:.2f} "
              f"({r['g2_cycles']}/{r['scenarios']}) -> {verdict}  "
              f"agrees with oracle={r['elle_matches_oracle']}")
    print("  contention: fraction of histories Elle flags non-serializable:")
    for r in result.contention:
        anomalies = ", ".join(f"{k}:{v}" for k, v in sorted(r["anomalies"].items())) or "none"
        print(f"    [{r['isolation']:12s}] {r['nonserializable_rate']:.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  anomalies: {anomalies}")
    try:
        from figures.plot_ed10 import plot_ed10

        plot_ed10(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
