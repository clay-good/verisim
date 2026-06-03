"""Experiment EH-H14 -- the concurrency dial: ``H_ε`` vs interleaving entropy (SPEC-6 §3.4, H14).

Concurrency is the host world's defining differentiator and the record/replay literature's named
unsolved source (HW-1). SPEC-6 does not solve it; it makes it a **measured dial** (H14): under
chaos-mode scheduling, faithful horizon should degrade monotonically with **interleaving entropy**,
and a recorded (sequential) schedule should recover it.

EH-H14 measures exactly that. It builds a concurrent workload (independent threads sharing files,
:mod:`verisim.hostdata.scheduler`), trains the factored arm on the **recorded** regime
(``interleave=0``, near-sequential schedules -- the determinized baseline), then evaluates
free-running (``ρ=0``) across the chaos dial ``interleave ∈ {0 … 1}`` over many chaos seeds. The
x-axis is the
*realized* interleaving entropy (the thread context-switch rate, a property of the emitted schedule,
not the knob); the y-axis is the composed ``H_ε``.

  - *Confirmed* → ``H_ε`` falls as interleaving entropy rises: concurrency is a continuum the chaos
    seed sweeps, the first quantification of HW-1's cost, and the recorded (low-entropy) end
    recovers horizon -- a knob, not a wall.
  - *Refuted* → ``H_ε`` is flat: the model learns schedule-invariant effects and concurrency is a
    non-issue at this scale -- a surprising result that would simplify every downstream world.

Both branches are first-class (the all-data-is-good-data stance). The committed config is a small,
fast instance of the apparatus, not a tuned publication run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.state import HostState
from verisim.hostdata import HostScheduler, interleaving_entropy, make_workload
from verisim.hostloop import PartialHostOracle, run_host_rollout
from verisim.hostmodel import HostVocab, build_host_graph, encode_target
from verisim.hostmodel.graph_train import GraphExample
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import Never
from verisim.metrics.aggregate import bootstrap_ci
from verisim.metrics.horizon import faithful_horizon


@dataclass(frozen=True)
class EHH14Config:
    name: str = "eh-h14-small"
    # model / training
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    mp_rounds: int = 3
    max_pid: int = 64
    graph_iters: int = 800
    graph_batch: int = 32
    lr: float = 3e-3
    model_seed: int = 0
    # workload
    n_threads: int = 5
    train_workload_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    train_interleave: float = 0.0  # the recorded/sequential regime the model is trained on
    # sweep
    eval_workload_seeds: tuple[int, ...] = (10, 11, 12)
    eval_interleaves: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    eval_chaos_seeds: tuple[int, ...] = (0, 1, 2, 3)
    epsilons: tuple[float, ...] = (0.0, 0.05, 0.1)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EHH14Config:
        b = EHH14Config()
        return EHH14Config(
            name=d.get("name", b.name),
            n_layer=d.get("n_layer", b.n_layer), n_head=d.get("n_head", b.n_head),
            n_embd=d.get("n_embd", b.n_embd), mp_rounds=d.get("mp_rounds", b.mp_rounds),
            max_pid=d.get("max_pid", b.max_pid), graph_iters=d.get("graph_iters", b.graph_iters),
            graph_batch=d.get("graph_batch", b.graph_batch), lr=d.get("lr", b.lr),
            model_seed=d.get("model_seed", b.model_seed),
            n_threads=d.get("n_threads", b.n_threads),
            train_workload_seeds=tuple(d.get("train_workload_seeds", b.train_workload_seeds)),
            train_interleave=d.get("train_interleave", b.train_interleave),
            eval_workload_seeds=tuple(d.get("eval_workload_seeds", b.eval_workload_seeds)),
            eval_interleaves=tuple(d.get("eval_interleaves", b.eval_interleaves)),
            eval_chaos_seeds=tuple(d.get("eval_chaos_seeds", b.eval_chaos_seeds)),
            epsilons=tuple(d.get("epsilons", b.epsilons)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EHH14Config:
        return EHH14Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class InterleavePoint:
    """One chaos-dial cell: the mean realized interleaving entropy + ``H_ε`` (with CI) per ε."""

    interleave: float
    mean_entropy: float
    epsilon: float
    mean_h: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "interleave": self.interleave, "mean_entropy": self.mean_entropy,
            "epsilon": self.epsilon, "mean_h": self.mean_h,
            "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n,
        }


def _graph_examples(
    oracle: HostOracle, vocab: HostVocab, config: HostConfig, actions: list[HostAction]
) -> list[GraphExample]:
    """Replay ``actions`` from boot, building ``(graph, target)`` per step (a scheduled dataset)."""
    state = HostState.initial()
    examples: list[GraphExample] = []
    for action in actions:
        result = oracle.step(state, action)
        examples.append(
            (build_host_graph(state, action, config, vocab.max_pid),
             encode_target(result.delta, vocab))
        )
        state = result.state
    return examples


def run_eh_h14(
    config: EHH14Config | None = None, *, oracle: HostOracle | None = None
) -> list[InterleavePoint]:
    """Train on recorded schedules; sweep the chaos dial; return ``H_ε`` vs interleaving entropy."""
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import train_host_graph_model

    config = config or EHH14Config()
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)

    # --- train on the recorded (sequential) regime -----------------------------
    train_examples: list[GraphExample] = []
    for ws in config.train_workload_seeds:
        wl = make_workload(host, config.n_threads, seed=ws)
        sched = HostScheduler(host, interleave=config.train_interleave).schedule(wl, chaos_seed=ws)
        train_examples += _graph_examples(oracle, vocab, host, sched.actions)
    model = build_host_graph_model(
        vocab, host, max_pid=config.max_pid, d_model=config.n_embd, mp_rounds=config.mp_rounds,
        n_layer=config.n_layer, n_head=config.n_head, seed=config.model_seed,
    )
    train_host_graph_model(
        model, train_examples, steps=config.graph_iters, lr=config.lr,
        batch_size=config.graph_batch, seed=config.model_seed,
    )

    # --- sweep the chaos dial, free-running (ρ=0) ------------------------------
    partial = PartialHostOracle(oracle)
    points: list[InterleavePoint] = []
    for il in config.eval_interleaves:
        entropies: list[float] = []
        horizons: dict[float, list[float]] = {e: [] for e in config.epsilons}
        for ws in config.eval_workload_seeds:
            wl = make_workload(host, config.n_threads, seed=ws)
            for cs in config.eval_chaos_seeds:
                sched = HostScheduler(host, interleave=il).schedule(wl, chaos_seed=cs)
                entropies.append(interleaving_entropy(sched.thread_ids))
                rollout = run_host_rollout(
                    model, partial, HostState.initial(), sched.actions, Never(),
                    epsilon=config.epsilons[0], budget=0, seed=cs,
                )
                for e in config.epsilons:
                    horizons[e].append(float(faithful_horizon(rollout.divergences, e)))
        mean_entropy = fmean(entropies) if entropies else 0.0
        for e in config.epsilons:
            lo, hi = bootstrap_ci(horizons[e], n_resamples=2000, seed=0)
            points.append(
                InterleavePoint(
                    interleave=il, mean_entropy=mean_entropy, epsilon=e,
                    mean_h=fmean(horizons[e]) if horizons[e] else 0.0,
                    ci_low=lo, ci_high=hi, n=len(horizons[e]),
                )
            )
    return points


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run EH-H14 (H_eps vs interleaving entropy).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh_h14_interleaving.csv")
    args = parser.parse_args()
    config = EHH14Config.from_json_file(args.config) if args.config else EHH14Config()
    points = run_eh_h14(config)

    lines = ["interleave,mean_entropy,epsilon,mean_h,ci_low,ci_high,n"]
    print(f"{'interleave':>10} {'entropy':>8} {'eps':>5} {'mean_H':>8}  CI")
    for p in points:
        lines.append(
            f"{p.interleave},{p.mean_entropy:.6f},{p.epsilon},{p.mean_h:.6f},"
            f"{p.ci_low:.6f},{p.ci_high:.6f},{p.n}"
        )
        print(f"{p.interleave:>10} {p.mean_entropy:>8.3f} {p.epsilon:>5} {p.mean_h:>8.3f}  "
              f"[{p.ci_low:.2f},{p.ci_high:.2f}]")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(points, out.with_suffix(".png"))


def _plot(points: list[InterleavePoint], path: Path) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[float, list[InterleavePoint]] = {}
    for p in points:
        series.setdefault(p.epsilon, []).append(p)

    fig, ax = plt.subplots(figsize=(7, 5))
    for epsilon, pts in sorted(series.items()):
        pts = sorted(pts, key=lambda p: p.mean_entropy)
        xs = [p.mean_entropy for p in pts]
        ys = [p.mean_h for p in pts]
        lo = [p.ci_low for p in pts]
        hi = [p.ci_high for p in pts]
        line = ax.plot(xs, ys, marker="o", label=f"ε={epsilon}")[0]
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("interleaving entropy  (thread context-switch rate)")
    ax.set_ylabel("free-running faithful horizon  H_ε  (ρ=0)")
    ax.set_title("Verisim EH-H14 — does concurrency lower faithful horizon? (H14)")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
