"""ED15 — optimistic (OCC) vs pessimistic (2PL) concurrency control: the cost of aborting.

The DS0-increment-8 experiment for SPEC-7 §3.2 — the `concurrency_control` dial (`occ` | `2pl`).
Both reach the *same* serializable guarantee, by opposite strategies, so the interesting axis is
**when they pay for a conflict**:

  - **OCC** (optimistic, first-committer-wins): a txn buffers its reads and writes and *validates at
    commit*. A conflict is detected **late** — at the very end, after the txn has done all its
    work — so an aborted txn has wasted every operation it executed.
  - **2PL** (pessimistic, strict two-phase locking with **wound-wait**): a txn acquires shared/
    exclusive locks as it reads/writes, held to commit; a conflict is detected **early**, at
    lock-acquisition. Wound-wait makes it deterministic and deadlock-free *without a scheduler* (the
    older txn — lexicographically smaller id — preempts the younger; the younger aborts rather than
    waiting), the deterministic 2PL the core can pin (DD-D3 deferred the *blocking* 2PL, whose
    victim selection injects nondeterminism).

Two findings, dependency-free and GPU-free, on the ED9-style contention workload (K transactions,
each reads two keys then writes one, over M objects, all at one coordinator):

  - **Panel A — wasted work (fail-fast).** The mean number of data operations (`tget`/`tput`) an
    *aborted* transaction completed before it aborted. Under **OCC** an aborted txn completed
    **all** its operations (it only fails at commit) — maximal wasted work; under **2PL** it fails
    at the conflicting lock-acquisition, **earlier**, so it wastes strictly less. Pessimistic
    locking trades the optimist's wasted work for upfront blocking-by-abort — the classic
    optimistic/pessimistic tradeoff, made measurable.

  - **Panel B — same serializable guarantee, both deadlock-free.** Both `occ` (serializable) and
    `2pl` **forbid write skew** (the ED9 anomaly: rate 0.0), reaching serializability by opposite
    routes — OCC validates the read-set late, 2PL locks the read-set early. Both are deterministic
    and deadlock-free, and **Tier-B reproduces both bit-for-bit** (the W1 retirement, §5.2), so the
    behavior is a property of a real execution, not just the analytic DES.
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
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle

CONCURRENCY = ("occ", "2pl")


@dataclass(frozen=True)
class ED15Config:
    name: str = "ed15-cc"
    n_nodes: int = 3
    n_objects: int = 4
    n_txns: int = 8
    reads_per_txn: int = 2
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    values: tuple[str, ...] = ("a", "b", "c", "d")
    # the write-skew panel: constraint object pairs (each gives one write-skew scenario, as in ED9)
    skew_object_pairs: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED15Config:
        b = ED15Config()
        return ED15Config(
            name=d.get("name", b.name),
            n_nodes=d.get("n_nodes", b.n_nodes),
            n_objects=d.get("n_objects", b.n_objects),
            n_txns=d.get("n_txns", b.n_txns),
            reads_per_txn=d.get("reads_per_txn", b.reads_per_txn),
            seeds=tuple(d.get("seeds", b.seeds)),
            values=tuple(d.get("values", b.values)),
            skew_object_pairs=tuple(
                tuple(p) for p in d.get("skew_object_pairs", b.skew_object_pairs)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED15Config:
        return ED15Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED15Result:
    #: per CC scheme: mean wasted ops per aborted txn, commit/abort counts (Panel A).
    wasted: list[dict[str, Any]] = field(default_factory=list)
    #: per CC scheme: the write-skew anomaly rate (Panel B) — both forbid it (serializable).
    write_skew: list[dict[str, Any]] = field(default_factory=list)
    #: Tier-B reproduces both schemes bit-for-bit (the W1 retirement).
    tier_b_agrees: bool = True
    tier_b_steps: int = 0


def _config(cfg: ED15Config, cc: str) -> DistConfig:
    return scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, concurrency_control=cc)


def _contention_ops(cfg: ED15Config, node: str, rng: random.Random) -> list[tuple[str, str]]:
    """K txns, each reads ``reads_per_txn`` distinct keys then writes one it read (ED9's workload).

    Returns ``(txn_id, command)`` pairs in execution order: all begins, all reads, all writes, all
    commits — so the read/write contention is exposed (OCC validates it at commit, 2PL locks it).
    """
    keys = [f"o{i}" for i in range(cfg.n_objects)]
    read_sets = [rng.sample(keys, min(cfg.reads_per_txn, len(keys))) for _ in range(cfg.n_txns)]
    write_keys = [rng.choice(read_sets[i]) for i in range(cfg.n_txns)]
    ids = [f"t{i}" for i in range(cfg.n_txns)]
    ops: list[tuple[str, str]] = [(ids[i], f"begin {node} {ids[i]}") for i in range(cfg.n_txns)]
    for i in range(cfg.n_txns):
        ops += [(ids[i], f"tget {node} {ids[i]} {k}") for k in read_sets[i]]
    ops += [(ids[i], f"tput {node} {ids[i]} {write_keys[i]} {rng.choice(cfg.values)}")
            for i in range(cfg.n_txns)]
    ops += [(ids[i], f"commit {node} {ids[i]}") for i in range(cfg.n_txns)]
    return ops


def _run_wasted(cfg: ED15Config, cc: str) -> dict[str, Any]:
    """Replay the contention workload under ``cc`` and measure wasted work per aborted txn."""
    config = _config(cfg, cc)
    ref = ReferenceDistOracle(config)
    sys_oracle = SystemDistOracle(config)
    node = config.nodes[0]
    wasted_means: list[float] = []
    commits_total = 0
    aborts_total = 0
    agree = True
    steps = 0
    for seed in cfg.seeds:
        rng = random.Random(seed * 31 + len(cc))
        ops = _contention_ops(cfg, node, rng)
        s = DistributedState.initial(config)
        s_sys = DistributedState.initial(config)
        work_ops: dict[str, int] = {}    # completed tget/tput per txn
        committed: set[str] = set()
        for txn_id, cmd in ops:
            action = parse_dist_action(cmd)
            r = ref.step(s, action)
            r_sys = sys_oracle.step(s_sys, action)
            if cluster_view(r.state) != cluster_view(r_sys.state):
                agree = False
            steps += 1
            verb = cmd.split()[0]
            if verb in ("tget", "tput") and r.status == "ok":
                work_ops[txn_id] = work_ops.get(txn_id, 0) + 1
            if verb == "commit" and r.status == "committed":
                committed.add(txn_id)
            s, s_sys = r.state, r_sys.state
        aborted = [t for t in (f"t{i}" for i in range(cfg.n_txns)) if t not in committed]
        commits_total += len(committed)
        aborts_total += len(aborted)
        if aborted:
            wasted_means.append(fmean(work_ops.get(t, 0) for t in aborted))
    return {
        "cc": cc,
        "wasted_ops_per_abort": fmean(wasted_means) if wasted_means else 0.0,
        "commits": commits_total,
        "aborts": aborts_total,
        "agree": agree,
        "steps": steps,
    }


def _write_skew_trajectory(node: str, key_a: str, key_b: str) -> list[str]:
    return [
        f"begin {node} A", f"begin {node} B",
        f"tget {node} A {key_a}", f"tget {node} A {key_b}",
        f"tget {node} B {key_a}", f"tget {node} B {key_b}",
        f"tput {node} A {key_a} a", f"tput {node} B {key_b} b",
        f"commit {node} A", f"commit {node} B",
    ]


def _run_write_skew(cfg: ED15Config, cc: str) -> dict[str, Any]:
    """Both read {x, y}; A writes x, B writes y. Write skew = both commit (serial-impossible)."""
    config = _config(cfg, cc)
    ref = ReferenceDistOracle(config)
    node = config.nodes[0]
    anomalies = 0
    for ai, bi in cfg.skew_object_pairs:
        s = DistributedState.initial(config)
        commits = 0
        for cmd in _write_skew_trajectory(node, f"o{ai}", f"o{bi}"):
            r = ref.step(s, parse_dist_action(cmd))
            if r.status == "committed":
                commits += 1
            s = r.state
        if commits == 2:  # both disjoint-write txns committed -> write skew
            anomalies += 1
    n = len(cfg.skew_object_pairs)
    return {"cc": cc, "anomalies": anomalies, "scenarios": n,
            "anomaly_rate": anomalies / n if n else 0.0}


def run_ed15(cfg: ED15Config | None = None) -> ED15Result:
    cfg = cfg or ED15Config()
    result = ED15Result()
    agree = True
    steps = 0
    for cc in CONCURRENCY:
        w = _run_wasted(cfg, cc)
        agree = agree and w["agree"]
        steps += w["steps"]
        result.wasted.append({k: w[k] for k in ("cc", "wasted_ops_per_abort", "commits", "aborts")})
        result.write_skew.append(_run_write_skew(cfg, cc))
    result.tier_b_agrees = agree
    result.tier_b_steps = steps
    return result


CSV_HEADER = "panel,cc,metric,value,detail"


def write_csv(result: ED15Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.wasted:
        lines.append(f"wasted,{r['cc']},wasted_ops_per_abort,{r['wasted_ops_per_abort']:.4f},"
                     f"commits={r['commits']};aborts={r['aborts']}")
    for r in result.write_skew:
        lines.append(f"write_skew,{r['cc']},anomaly_rate,{r['anomaly_rate']:.4f},"
                     f"anomalies={r['anomalies']}/{r['scenarios']}")
    lines.append(f"tier_b,both,agrees,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED15 (optimistic OCC vs pessimistic 2PL concurrency control)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed15.csv")
    parser.add_argument("--plot", type=str, default="figures/ed15.png")
    args = parser.parse_args()
    cfg = ED15Config.from_json_file(args.config) if args.config else ED15Config()
    result = run_ed15(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — wasted work per aborted txn (data ops completed before abort):")
    for r in result.wasted:
        print(f"    [{r['cc']:4s}] {r['wasted_ops_per_abort']:.2f}  "
              f"(commits={r['commits']}, aborts={r['aborts']})")
    print("  Panel B — write-skew anomaly rate (both serializable → 0.0):")
    for r in result.write_skew:
        verdict = "WRITE SKEW" if r["anomaly_rate"] > 0 else "forbidden"
        print(f"    [{r['cc']:4s}] {r['anomaly_rate']:.2f} → {verdict}")
    print(f"  Tier-B reproduces both schemes bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed15 import plot_ed15

        plot_ed15(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
