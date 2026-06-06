"""ED11 — Elle's version oracle: serializability from values + the split-brain fork (DS3 incr 3).

ED10 recovered the write-skew anomaly *black-box* — but it still let the store hand Elle the integer
MVCC version each transaction read and installed (``collect_history`` peeks at
``state.replicas[(k, node)].version``). That is the one cooperation Jepsen's Elle removes, and the
reason it works against a true black box. Over a **list-append** register — every write appends a
globally-unique value, every read returns the whole list — the per-key version order is recoverable
from the read *values* alone (Kingsbury & Alvaro, VLDB 2020, the "version oracle"). ED11 closes that
last gap and shows the strict gain it buys, in two panels:

  - **The version oracle is sound (the headline).** Translate every ED10 write-skew/contention
    history into its list-append form — value ``k#v`` *is* the value installed at version ``v`` of
    key ``k``, and a read of version ``v`` observed the prefix ``[k#1 … k#v]`` — hand Elle only the
    *values*, and let :func:`recover_versions` reconstruct the order. The recovered version history
    is **bit-identical** to the store-supplied one (``recovery_sound``), so the G2 write-skew rate
    the certify-serializable verdict are exactly ED10's — recovered with *zero* store cooperation.
  - **The split-brain fork only value-recovery can see (the strict gain).** Build the black-box
    signature of split-brain: a partition lets two sides extend one key divergently, so a later read
    sees ``[a, b]`` while another sees ``[a, c]``. Neither list is a prefix of the other, so the
    version oracle reports an **incompatible-order** anomaly — non-serializable, detected before any
    cycle search, from the client-visible history alone. ED10's integer-version mode *structurally
    cannot represent this* (it receives one non-contradictory version sequence per key), so this is
    the consistency anomaly the §9.1 split-brain view exists to catch, now caught reference-free
    from the client history alone. A clean (un-forked) control reports 0.

Pure standard library, dependency-free, GPU-free.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.config import scaled_dist_config
from verisim.distoracle.elle import (
    AppendObservation,
    TxnObservation,
    appends_to_version_history,
    check_serializable,
    check_serializable_appends,
    recover_versions,
)
from verisim.experiments.ed10 import _write_skew_trajectory, collect_history
from verisim.metrics.aggregate import bootstrap_ci

ISOLATION_LEVELS = ("serializable", "snapshot")


@dataclass(frozen=True)
class ED11Config:
    name: str = "ed11"
    n_nodes: int = 3
    skew_object_pairs: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2))
    n_txns: int = 8
    n_objects: int = 4
    reads_per_txn: int = 2
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    values: tuple[str, ...] = ("a", "b", "c", "d")
    # fork panel: how many split-brain scenarios to build (each forks one key two ways)
    n_forks: int = 6

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED11Config:
        b = ED11Config()
        return ED11Config(
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
            n_forks=d.get("n_forks", b.n_forks),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED11Config:
        return ED11Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED11Result:
    recovery: list[dict[str, Any]] = field(default_factory=list)  # per isolation level (sound + G2)
    fork: dict[str, Any] = field(default_factory=dict)  # the split-brain panel


# --- the version-supplied → list-append translation (the only thing ED11 adds over ED10) ----------

def to_append_history(history: list[TxnObservation]) -> list[AppendObservation]:
    """Translate a version-supplied history into its list-append form, hiding the integer versions.

    Value ``k#v`` is *defined* to be the value installed at version ``v`` of key ``k``; a read of
    version ``v`` observed the prefix ``[k#1, …, k#v]`` (version 0 = the empty boot list). This is
    the list-append datatype Elle's version oracle is built for — the only signal it gets is the
    values. Recovering versions from those values must reproduce the originals (``recovery_sound``).
    """
    out: list[AppendObservation] = []
    for t in history:
        appends = tuple((k, f"{k}#{v}") for k, v in t.writes)
        list_reads = tuple(
            (k, tuple(f"{k}#{i}" for i in range(1, v + 1))) for k, v in t.reads
        )
        out.append(AppendObservation(t.txn_id, appends=appends, list_reads=list_reads))
    return out


# --- panel A: the version oracle is sound (recovers exactly ED10's store-supplied versions) -------

def _run_recovery(cfg: ED11Config, isolation: str) -> dict[str, Any]:
    config = scaled_dist_config(cfg.n_nodes, n_objects=cfg.n_objects, txn_isolation=isolation)
    node = config.nodes[0]
    g2 = 0
    sound = 0
    agrees = 0
    n = len(cfg.skew_object_pairs)
    for ai, bi in cfg.skew_object_pairs:
        cmds = _write_skew_trajectory(node, f"o{ai}", f"o{bi}")
        supplied, _ = collect_history(config, cmds)  # ED10's store-supplied (key, version) history
        appends = to_append_history(supplied)  # hide the versions
        recovered = recover_versions(appends)
        # soundness: re-deriving versions from values reproduces the store's exact version history
        if recovered.ok and appends_to_version_history(appends, recovered) == supplied:
            sound += 1
        report_val = check_serializable_appends(appends)  # value-recovery verdict
        report_ver = check_serializable(supplied)  # ED10's version-supplied verdict
        if (report_val.serializable, report_val.anomaly) == (
            report_ver.serializable,
            report_ver.anomaly,
        ):
            agrees += 1
        if (not report_val.serializable) and report_val.anomaly == "G2":
            g2 += 1
    return {
        "isolation": isolation,
        "scenarios": n,
        "elle_g2_rate": g2 / n if n else 0.0,
        "recovery_sound": sound == n,
        "agrees_supplied": agrees == n,
    }


# --- panel B: the split-brain fork only value-recovery can represent ------------------------------

def _fork_history(base: str, fork_left: str, fork_right: str) -> list[AppendObservation]:
    """Split-brain on key ``x``: both sides see ``[base]`` then append divergently, two reads fork.

    A appends ``base``; under a partition B (one side) appends ``fork_left`` and C (the other side)
    appends ``fork_right``; a healing read R1 observes ``[base, fork_left]`` and R2 observes
    ``[base, fork_right]`` — neither a prefix of the other. The version oracle reports
    ``incompatible-order`` (a fork), the black-box signature of split-brain.
    """
    return [
        AppendObservation("A", appends=(("x", base),)),
        AppendObservation("B", appends=(("x", fork_left),), list_reads=(("x", (base,)),)),
        AppendObservation("C", appends=(("x", fork_right),), list_reads=(("x", (base,)),)),
        AppendObservation("R1", list_reads=(("x", (base, fork_left)),)),
        AppendObservation("R2", list_reads=(("x", (base, fork_right)),)),
    ]


def _clean_history(base: str, follow: str) -> list[AppendObservation]:
    """The un-forked control: one append log ``[base, follow]``, both reads agree — serializable."""
    return [
        AppendObservation("A", appends=(("x", base),)),
        AppendObservation("B", appends=(("x", follow),), list_reads=(("x", (base,)),)),
        AppendObservation("R1", list_reads=(("x", (base, follow)),)),
        AppendObservation("R2", list_reads=(("x", (base,)),)),
    ]


def _run_fork(cfg: ED11Config) -> dict[str, Any]:
    fork_flags: list[float] = []
    clean_flags: list[float] = []
    for i in range(cfg.n_forks):
        rng = random.Random(i * 7919)
        base, left, right = (f"v{rng.randrange(1000)}" for _ in range(3))
        fr = check_serializable_appends(_fork_history(base, left, right))
        # 1.0 iff caught as the split-brain class the integer-version mode cannot represent
        caught = not fr.serializable and fr.anomaly == "incompatible-order"
        fork_flags.append(1.0 if caught else 0.0)
        cr = check_serializable_appends(_clean_history(base, left))
        clean_flags.append(0.0 if cr.serializable else 1.0)
    lo, hi = bootstrap_ci(fork_flags, seed=0)
    return {
        "scenarios": cfg.n_forks,
        "incompatible_order_rate": fmean(fork_flags),
        "ci_lo": lo,
        "ci_hi": hi,
        "clean_control_flag_rate": fmean(clean_flags),
    }


def run_ed11(cfg: ED11Config | None = None) -> ED11Result:
    cfg = cfg or ED11Config()
    return ED11Result(
        recovery=[_run_recovery(cfg, iso) for iso in ISOLATION_LEVELS],
        fork=_run_fork(cfg),
    )


CSV_HEADER = "panel,key,metric,value,ci_lo,ci_hi,detail"


def write_csv(result: ED11Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in result.recovery:
        lines.append(
            f"recovery,{r['isolation']},elle_g2_rate,{r['elle_g2_rate']:.4f},,,"
            f"sound={r['recovery_sound']};agrees_supplied={r['agrees_supplied']}"
        )
    f = result.fork
    lines.append(
        f"fork,split_brain,incompatible_order_rate,{f['incompatible_order_rate']:.4f},"
        f"{f['ci_lo']:.4f},{f['ci_hi']:.4f},clean_control={f['clean_control_flag_rate']:.4f}"
    )
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED11 (Elle's version oracle: serializability from values + the fork)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed11.csv")
    parser.add_argument("--plot", type=str, default="figures/ed11.png")
    args = parser.parse_args()
    cfg = ED11Config.from_json_file(args.config) if args.config else ED11Config()
    result = run_ed11(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  recovery: Elle's G2 write-skew rate, recovered from values alone (no store versions):")
    for r in result.recovery:
        print(f"    [{r['isolation']:12s}] g2={r['elle_g2_rate']:.2f}  "
              f"sound={r['recovery_sound']}  agrees-supplied={r['agrees_supplied']}")
    f = result.fork
    print("  fork: the split-brain anomaly only value-recovery can represent:")
    print(f"    incompatible-order rate {f['incompatible_order_rate']:.2f} "
          f"[{f['ci_lo']:.2f}, {f['ci_hi']:.2f}]  clean control {f['clean_control_flag_rate']:.2f}")
    try:
        from figures.plot_ed11 import plot_ed11

        plot_ed11(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
