"""ED17 — read-uncommitted isolation: the dirty-read anomaly + its black-box recovery (DS0 inc10).

The experiment that pins the *weakest* transaction isolation level (SPEC-7 §3.2, §3.4) — the bottom
of the standard SQL hierarchy (``read_uncommitted ⊂ read_committed ⊂ snapshot ⊂ serializable``) —
against the three stronger ones ED9/ED16 already cover. ``read_uncommitted`` drops even read-
committed's last guarantee: an OCC ``tget`` may observe another active transaction's **uncommitted**
buffered write, so when that writer later aborts the reader saw a value that *never committed* — the
classic **dirty-read** anomaly (Adya G1a).

Two panels (mirroring ED10's oracle-anomaly / Elle-recovery shape):

  - **The dirty-read anomaly (the oracle side, the headline).** The textbook scenario: transaction
    ``A`` writes ``x`` (uncommitted), ``B`` reads ``x``, then ``A`` **aborts** — so ``A``'s write is
    rolled back and never committed. Under **read_uncommitted** ``B`` observed ``A``'s uncommitted
    value (the dirty read); under **read_committed** / **snapshot** / **serializable** the MVCC
    ``tget`` gives ``B`` only the committed value (the boot default). ED17 reports the **anomaly
    rate** per level: ≈1.0 under read_uncommitted, 0.0 under the three stronger levels.

  - **Elle recovers the dirty read black-box (the reference-free side).** The §5.3 value oracle
    (DS3 increment 3) reconstructs the dirty read from the *client-visible history alone* — no
    oracle, no cluster state. Encoding the run as a list-append history (the aborted writer ``A``
    contributes **nothing** to the committed appends; ``B``'s observed read becomes its list-read),
    :func:`~verisim.distoracle.elle.recover_versions` sees ``B`` read a value no committed
    transaction appended and reports the **``dirty-read``** recovery anomaly — at exactly the rate
    the oracle-side panel admits. The cheap black-box verifier agrees with the expensive oracle on
    the question it answers (the dirty-read echo of ED10/ED16's write-skew + lost-update recovery).

All four levels compose with Tier-B: the autonomous-actor system oracle reproduces Tier-A on every
scenario (transaction bookkeeping is coordinator-local). Dependency-free, GPU-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import scaled_dist_config
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.elle import AppendObservation, check_serializable_appends
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle

# Ordered strong -> weak; read_uncommitted is the new (weakest) level this increment adds.
ISOLATION_LEVELS = ("serializable", "snapshot", "read_committed", "read_uncommitted")


@dataclass(frozen=True)
class ED17Config:
    name: str = "ed17"
    n_nodes: int = 3
    # one dirty-read scenario per object index; each writes a distinct value so the dirty read is
    # unambiguous (a non-default value can only have come from the uncommitted writer).
    dirty_objects: tuple[int, ...] = (0, 1, 2)
    n_objects: int = 4
    # the distinct uncommitted value each scenario's writer buffers (cycled over the objects).
    values: tuple[str, ...] = ("a", "b", "c", "d")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED17Config:
        b = ED17Config()
        return ED17Config(
            name=d.get("name", b.name),
            n_nodes=d.get("n_nodes", b.n_nodes),
            dirty_objects=tuple(d.get("dirty_objects", b.dirty_objects)),
            n_objects=d.get("n_objects", b.n_objects),
            values=tuple(d.get("values", b.values)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED17Config:
        return ED17Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED17Result:
    dirty_read: list[dict[str, Any]] = field(default_factory=list)  # oracle side, per level
    recovery: list[dict[str, Any]] = field(default_factory=list)  # Elle black-box side, per level


def _dirty_read_trajectory(node: str, key: str, value: str) -> list[str]:
    """``A`` writes ``key`` (uncommitted), ``B`` reads it, then ``A`` **aborts** (the rollback)."""
    return [
        f"begin {node} A", f"begin {node} B",
        f"tput {node} A {key} {value}",  # A buffers an uncommitted write
        f"tget {node} B {key}",          # B reads: dirty under read_uncommitted, else committed
        f"abort {node} A",               # A rolls back -> the value B may have read never commits
        f"commit {node} B",
    ]


def _run_level(cfg: ED17Config, isolation: str, default_value: str) -> dict[str, Any]:
    """Run the dirty-read scenarios for one isolation level; return both panels' per-level facts."""
    config = scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, txn_isolation=isolation)
    ref = ReferenceDistOracle(config)
    sys_oracle = SystemDistOracle(config)
    node = config.nodes[0]
    n = len(cfg.dirty_objects)
    dirty_anomalies = 0  # oracle side: B observed A's uncommitted value
    elle_anomalies = 0  # black-box side: Elle flags a dirty-read recovery anomaly
    matches = 0  # the two verdicts agree on this scenario
    tier_b_agrees = True

    for i, oi in enumerate(cfg.dirty_objects):
        key = f"o{oi}"
        value = cfg.values[i % len(cfg.values)]
        s = DistributedState.initial(config)
        s_sys = DistributedState.initial(config)
        b_read = ""
        for cmd in _dirty_read_trajectory(node, key, value):
            action = parse_dist_action(cmd)
            r = ref.step(s, action)
            r_sys = sys_oracle.step(s_sys, action)
            if cluster_view(r.state) != cluster_view(r_sys.state) or r.status != r_sys.status:
                tier_b_agrees = False
            if cmd.startswith(f"tget {node} B"):
                b_read = r.value
            s, s_sys = r.state, r_sys.state

        # oracle side: a dirty read = B observed the (non-default) uncommitted value A buffered.
        oracle_dirty = b_read == value and b_read != default_value
        if oracle_dirty:
            dirty_anomalies += 1

        # black-box side: encode the client-visible history as a list-append history and let Elle's
        # value oracle recover it. A aborted, so it contributes NO committed append; B's observed
        # read becomes its list-read (a non-default value = a 1-element list of the uncommitted
        # value; the boot default = the empty committed log). Elle then sees B read a value no
        # committed txn appended -> the `dirty-read` recovery anomaly (Adya G1a), reference-free.
        b_list: tuple[str, ...] = (b_read,) if b_read != default_value else ()
        history = [
            AppendObservation("A", appends=(), list_reads=()),  # aborted: no committed effect
            AppendObservation("B", appends=(), list_reads=((key, b_list),)),
        ]
        report = check_serializable_appends(history)
        elle_dirty = report.anomaly == "dirty-read"
        if elle_dirty:
            elle_anomalies += 1
        if elle_dirty == oracle_dirty:
            matches += 1

    return {
        "isolation": isolation,
        "scenarios": n,
        "dirty_anomalies": dirty_anomalies,
        "anomaly_rate": dirty_anomalies / n if n else 0.0,
        "elle_anomalies": elle_anomalies,
        "recovery_rate": elle_anomalies / n if n else 0.0,
        "matches_oracle": matches == n,
        "tier_b_agrees": tier_b_agrees,
    }


def run_ed17(cfg: ED17Config | None = None) -> ED17Result:
    cfg = cfg or ED17Config()
    default_value = scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects).default_value
    rows = [_run_level(cfg, iso, default_value) for iso in ISOLATION_LEVELS]
    return ED17Result(
        dirty_read=[{k: r[k] for k in ("isolation", "scenarios", "dirty_anomalies",
                                       "anomaly_rate", "tier_b_agrees")} for r in rows],
        recovery=[{k: r[k] for k in ("isolation", "scenarios", "elle_anomalies",
                                     "recovery_rate", "matches_oracle")} for r in rows],
    )


CSV_HEADER = "panel,isolation,metric,value,detail"


def write_csv(result: ED17Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.dirty_read:
        lines.append(f"dirty_read,{r['isolation']},anomaly_rate,{r['anomaly_rate']:.4f},"
                     f"anomalies={r['dirty_anomalies']}/{r['scenarios']};"
                     f"tier_b_agrees={r['tier_b_agrees']}")
    for r in result.recovery:
        lines.append(f"recovery,{r['isolation']},recovery_rate,{r['recovery_rate']:.4f},"
                     f"elle={r['elle_anomalies']}/{r['scenarios']};"
                     f"matches_oracle={r['matches_oracle']}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED17 (read-uncommitted isolation: dirty read + its black-box recovery)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed17.csv")
    parser.add_argument("--plot", type=str, default="figures/ed17.png")
    args = parser.parse_args()
    cfg = ED17Config.from_json_file(args.config) if args.config else ED17Config()
    result = run_ed17(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  dirty-read anomaly rate (B observed an aborted txn's uncommitted write):")
    for r in result.dirty_read:
        verdict = "DIRTY READ admitted" if r["anomaly_rate"] > 0 else "dirty read forbidden"
        print(f"    [{r['isolation']:16s}] {r['anomaly_rate']:.2f} "
              f"({r['dirty_anomalies']}/{r['scenarios']}) -> {verdict}  "
              f"Tier-B agrees={r['tier_b_agrees']}")
    print("  Elle value-oracle recovery rate (dirty read recovered black-box from the history):")
    for r in result.recovery:
        print(f"    [{r['isolation']:16s}] {r['recovery_rate']:.2f} "
              f"({r['elle_anomalies']}/{r['scenarios']}) -> matches oracle={r['matches_oracle']}")
    try:
        from figures.plot_ed17 import plot_ed17

        plot_ed17(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
