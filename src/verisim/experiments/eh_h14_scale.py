"""Experiment EH-H14-scale -- does the concurrency cost scale with concurrency *width*? (§3.4).

EH-H14 confirmed H14 at one workload width (5 threads): free-running ``H_ε`` collapses as
interleaving entropy rises. The natural scaling question -- the host analogue of the SPEC-9
free-oracle scaling work -- is whether the *cost of concurrency grows with the amount of
concurrency*: as a host runs more concurrent threads (more shared-file contention, more interleaved
forks), does the ``H_ε`` collapse **steepen** and the floor **drop**, or does it saturate?

This reruns the EH-H14 dial (:func:`verisim.experiments.eh_h14.run_eh_h14`) at several thread counts
and overlays the curves. Each thread count trains its *own* factored arm on its *own* recorded
(sequential) regime, so the comparison is fair: every width is given a model fit to its sequential
schedules, and the question is purely how chaos degrades it. Whatever it shows is a datum: more
threads steepening the collapse would quantify "concurrency width is a difficulty axis"; saturation
would say the cost is paid by the *first* few interleavings and plateaus -- both sharpen H14.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from .eh_h14 import EHH14Config, InterleavePoint, run_eh_h14


@dataclass(frozen=True)
class EHH14ScaleConfig:
    thread_counts: tuple[int, ...] = (2, 4, 6, 8)
    epsilon: float = 0.0  # the strictest tolerance, where the collapse is clearest
    base: EHH14Config = field(default_factory=EHH14Config)

    @staticmethod
    def from_dict(d: dict[str, object]) -> EHH14ScaleConfig:
        b = EHH14ScaleConfig()
        base = d.get("base")
        return EHH14ScaleConfig(
            thread_counts=tuple(d.get("thread_counts", b.thread_counts)),  # type: ignore[arg-type]
            epsilon=float(d.get("epsilon", b.epsilon)),  # type: ignore[arg-type]
            base=EHH14Config.from_dict(base) if isinstance(base, dict) else b.base,
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EHH14ScaleConfig:
        return EHH14ScaleConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ScalePoint:
    """One (thread-count, interleave) cell: the realized interleaving entropy + ``H_ε`` at ``ε``."""

    n_threads: int
    interleave: float
    mean_entropy: float
    mean_h: float
    ci_low: float
    ci_high: float

    def csv_row(self) -> str:
        return (
            f"{self.n_threads},{self.interleave},{self.mean_entropy:.4f},"
            f"{self.mean_h:.4f},{self.ci_low:.4f},{self.ci_high:.4f}"
        )


CSV_HEADER = "n_threads,interleave,mean_entropy,mean_h,ci_low,ci_high"


def run_eh_h14_scale(config: EHH14ScaleConfig | None = None) -> list[ScalePoint]:
    """Run the H14 dial at each thread count; one point per (thread-count, interleave) at ε."""
    config = config or EHH14ScaleConfig()
    points: list[ScalePoint] = []
    for n in config.thread_counts:
        # reuse the EH-H14 dial verbatim, just at this workload width
        from dataclasses import replace

        cfg = replace(config.base, n_threads=n)
        cells: list[InterleavePoint] = run_eh_h14(cfg)
        for p in cells:
            if abs(p.epsilon - config.epsilon) < 1e-9:
                points.append(
                    ScalePoint(n, p.interleave, p.mean_entropy, p.mean_h, p.ci_low, p.ci_high)
                )
    return points


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH-H14-scale: concurrency cost vs thread count.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh_h14_scale.csv")
    args = parser.parse_args()
    config = EHH14ScaleConfig.from_json_file(args.config) if args.config else EHH14ScaleConfig()
    points = run_eh_h14_scale(config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    print(f"wrote {out}  (ε={config.epsilon})")
    for n in config.thread_counts:
        cells = sorted((p for p in points if p.n_threads == n), key=lambda p: p.interleave)
        lo = cells[0].mean_h if cells else 0.0
        hi = cells[-1].mean_h if cells else 0.0
        print(f"  {n} threads: H_ε {lo:.1f} (recorded) -> {hi:.1f} (chaos)")
    _plot(points, out.with_suffix(".png"), config)


def _plot(  # pragma: no cover
    points: list[ScalePoint], path: Path, config: EHH14ScaleConfig
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for n in config.thread_counts:
        cells = sorted((p for p in points if p.n_threads == n), key=lambda p: p.mean_entropy)
        xs = [p.mean_entropy for p in cells]
        ys = [p.mean_h for p in cells]
        lo = [p.ci_low for p in cells]
        hi = [p.ci_high for p in cells]
        (line,) = ax.plot(xs, ys, marker="o", label=f"{n} threads")
        ax.fill_between(xs, lo, hi, alpha=0.12, color=line.get_color())
    ax.set_xlabel("interleaving entropy  (thread context-switch rate)")
    ax.set_ylabel(f"free-running faithful horizon  H_ε  (ρ=0, ε={config.epsilon})")
    ax.set_title("Verisim EH-H14-scale — does the concurrency collapse steepen with more threads?")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
