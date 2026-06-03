"""Experiment EH-H13-scale -- does the composition coupling deepen with concurrency width? (H13 ×
H14).

EH1/EH4 found the host composition **coupled** (H13): the composed per-step acceptance sits *below*
the multiplicative/independence prediction ``∏ aᵢ`` -- subsystem failures are anti-correlated. H14
(EH-H14) found concurrency is a measurable dial. EH-H13-scale crosses the two: as a host runs more
concurrent threads (more shared-file contention, more interleaved forks coupling the proc/fd/fs
subsystems through the schedule), does the composition become **more coupled** -- composed sinking
further below the independence floor?

It trains one factored arm on the general (forky) workload, then -- per thread count --
the composition law *teacher-forced on chaos-scheduled workloads* of that width
(:mod:`verisim.hostdata.scheduler`). The reported quantity is the **independence gap** ``∏ aᵢ −
composed`` (positive = below the floor = coupled/anti-correlated) as a function of thread count.
*More threads widening the gap* would say concurrency manufactures coupling; *a flat gap* would say
the
coupling is intrinsic to the bundle and width-invariant -- both sharpen H13. Whatever it shows is a
datum.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.hostdata import HostScheduler, interleaving_entropy, make_workload
from verisim.hostmetrics.composition import composition_law
from verisim.hostoracle.reference import ReferenceHostOracle

from .eh1 import EH1Config, teacher_forced_faithful


@dataclass(frozen=True)
class EHH13ScaleConfig:
    base: EH1Config = field(default_factory=EH1Config)
    thread_counts: tuple[int, ...] = (2, 4, 6, 8)
    interleave: float = 1.0  # chaos schedules, so width is the variable under test
    eval_workload_seeds: tuple[int, ...] = (10, 11, 12)
    eval_chaos_seeds: tuple[int, ...] = (0, 1, 2, 3)
    composition_epsilon: float = 0.05
    max_pid: int = 64
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 800
    graph_batch: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EHH13ScaleConfig:
        b = EHH13ScaleConfig()
        return EHH13ScaleConfig(
            base=EH1Config.from_dict(d.get("base", {})),
            thread_counts=tuple(d.get("thread_counts", b.thread_counts)),
            interleave=d.get("interleave", b.interleave),
            eval_workload_seeds=tuple(d.get("eval_workload_seeds", b.eval_workload_seeds)),
            eval_chaos_seeds=tuple(d.get("eval_chaos_seeds", b.eval_chaos_seeds)),
            composition_epsilon=d.get("composition_epsilon", b.composition_epsilon),
            max_pid=d.get("max_pid", b.max_pid),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_batch=d.get("graph_batch", b.graph_batch),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EHH13ScaleConfig:
        return EHH13ScaleConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ScalePoint:
    """One thread-count cell: the composition law on chaos schedules of that width."""

    n_threads: int
    mean_entropy: float
    composed: float
    multiplicative: float
    weakest_link: float
    independence_gap: float  # ∏ aᵢ − composed; positive = below the independence floor (coupled)
    verdict: str

    def csv_row(self) -> str:
        return (
            f"{self.n_threads},{self.mean_entropy:.4f},{self.composed:.4f},"
            f"{self.multiplicative:.4f},{self.weakest_link:.4f},{self.independence_gap:.4f},{self.verdict}"
        )


CSV_HEADER = "n_threads,mean_entropy,composed,multiplicative,weakest_link,independence_gap,verdict"


def run_eh_h13_scale(config: EHH13ScaleConfig | None = None) -> list[ScalePoint]:
    """Train the factored arm; measure the composition law on chaos schedules at each thread
    count."""
    from verisim.hostmodel import HostVocab
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, train_host_graph_model

    config = config or EHH13ScaleConfig()
    base = config.base
    oracle = ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)

    examples = build_host_graph_dataset(
        oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
        n_steps=base.train_steps_per_traj,
    )
    model = build_host_graph_model(
        vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
        mp_rounds=config.graph_mp_rounds, seed=base.model_seed,
    )
    train_host_graph_model(
        model, examples, steps=config.graph_iters, lr=base.lr,
        batch_size=config.graph_batch, seed=base.model_seed,
    )

    points: list[ScalePoint] = []
    for n in config.thread_counts:
        steps: list[dict[str, bool]] = []
        entropies: list[float] = []
        for ws in config.eval_workload_seeds:
            wl = make_workload(host, n, seed=ws)
            for cs in config.eval_chaos_seeds:
                sched = HostScheduler(host, interleave=config.interleave).schedule(
                    wl, chaos_seed=cs
                )
                entropies.append(interleaving_entropy(sched.thread_ids))
                steps.extend(
                    teacher_forced_faithful(
                        model, oracle, sched.actions, config.composition_epsilon
                    )
                )
        law = composition_law(steps)
        points.append(
            ScalePoint(
                n_threads=n,
                mean_entropy=fmean(entropies) if entropies else 0.0,
                composed=law.composed_acceptance,
                multiplicative=law.multiplicative_prediction,
                weakest_link=law.weakest_link_prediction,
                independence_gap=law.multiplicative_prediction - law.composed_acceptance,
                verdict=law.verdict,
            )
        )
    return points


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH-H13-scale: coupling vs thread count.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh_h13_scale.csv")
    args = parser.parse_args()
    config = EHH13ScaleConfig.from_json_file(args.config) if args.config else EHH13ScaleConfig()
    points = run_eh_h13_scale(config)
    print(f"{'threads':>7} {'entropy':>8} {'composed':>9} {'mult(∏)':>8} {'gap':>7}  verdict")
    for p in points:
        print(f"{p.n_threads:>7} {p.mean_entropy:>8.3f} {p.composed:>9.3f} "
              f"{p.multiplicative:>8.3f} {p.independence_gap:>7.3f}  {p.verdict}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    print(f"wrote {out}")
    _plot(points, out.with_suffix(".png"))


def _plot(points: list[ScalePoint], path: Path) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [p.n_threads for p in points]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(xs, [p.multiplicative for p in points], marker="s", label="∏ aᵢ (independence floor)")
    ax1.plot(xs, [p.composed for p in points], marker="o", label="composed a (measured)")
    ax1.plot(xs, [p.weakest_link for p in points], marker="^", label="min aᵢ (weakest-link)")
    ax1.set_xlabel("concurrent threads")
    ax1.set_ylabel("per-step acceptance")
    ax1.set_title("composition law vs concurrency width")
    ax1.legend(fontsize="small")
    ax2.plot(xs, [p.independence_gap for p in points], marker="o", color="#c33")
    ax2.axhline(0.0, color="#888", lw=0.8, ls="--")
    ax2.set_xlabel("concurrent threads")
    ax2.set_ylabel("independence gap  ∏ aᵢ − composed")
    ax2.set_title("does coupling deepen with width?  (gap > 0 = coupled)")
    fig.suptitle("Verisim EH-H13-scale — composition coupling vs concurrency width (H13 × H14)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
