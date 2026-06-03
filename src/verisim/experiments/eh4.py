"""Experiment EH4 -- the factored-vs-flat comparison (SPEC-6 §6.1, DD-H1; the H13 follow-up).

Trains the flat HC4 transformer ``M_θ`` and the factored interaction-graph arm (GNN+RSSM) on the
**same** seeded oracle data, then scores both with the **same** eval primitives, so the only
difference is the proposer's architecture. Two complementary lenses:

  - **delta-exact rate** -- the fraction of held-out steps where the model's free constrained-decode
    assembles the oracle's *exact* bundle delta (the strict one-step metric, EN4's headline);
  - **the composition law (H13)** -- per arm, the teacher-forced composed per-step acceptance vs the
    multiplicative (independence) and weakest-link predictions. The flat baseline read **coupled**
    with composed *below* the independence floor (EH1/H13). The factored arm models the
    cross-subsystem references the flat arm flattens, so the question is whether it **raises
    composed acceptance and moves the verdict off `coupled` toward `multiplicative`** (less anti-
    failures) -- i.e. whether modeling the composition explicitly buys faithfulness it could not buy
    flattened. Whatever it shows is a datum (the all-data-is-good-data stance): a smoke-scale
    instance of the apparatus, not a tuned publication run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.state import HostState
from verisim.hostloop.model import HostModel
from verisim.hostmetrics.composition import composition_law
from verisim.hostmodel import HostVocab
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

from .eh1 import EH1Config, eval_actions, teacher_forced_faithful
from .eh1 import train_model as train_flat


@dataclass(frozen=True)
class EH4Config:
    name: str = "eh4-small"
    base: EH1Config = field(default_factory=EH1Config)
    max_pid: int = 64
    graph_iters: int = 800
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_batch: int = 32
    composition_epsilon: float = 0.05

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH4Config:
        b = EH4Config()
        return EH4Config(
            name=d.get("name", b.name),
            base=EH1Config.from_dict(d.get("base", {})),
            max_pid=d.get("max_pid", b.max_pid),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_batch=d.get("graph_batch", b.graph_batch),
            composition_epsilon=d.get("composition_epsilon", b.composition_epsilon),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH4Config:
        import json

        return EH4Config.from_dict(json.loads(Path(path).read_text()))


def _delta_exact_rate(model: HostModel, oracle: HostOracle, actions: list[HostAction]) -> float:
    """Fraction of (teacher-forced) steps where free decode == the oracle's exact bundle delta."""
    state = HostState.initial()
    exact = 0
    for action in actions:
        truth = oracle.step(state, action)
        if model.predict_delta(state, action) == truth.delta:
            exact += 1
        state = truth.state
    return exact / len(actions) if actions else 1.0


def _arm_metrics(
    model: HostModel, oracle: HostOracle, host: HostConfig, base: EH1Config, epsilon: float
) -> dict[str, Any]:
    """Score one arm: delta-exact rate + the composition law, pooled over the eval rollouts."""
    exact: list[float] = []
    steps: list[dict[str, bool]] = []
    for difficulty, driver in base.difficulties.items():  # noqa: B007 - driver used below
        for seed in base.eval_seeds:
            actions = eval_actions(oracle, host, driver, seed, base.eval_steps)
            exact.append(_delta_exact_rate(model, oracle, actions))
            steps.extend(teacher_forced_faithful(model, oracle, actions, epsilon))
    law = composition_law(steps)
    return {
        "delta_exact": sum(exact) / len(exact) if exact else 0.0,
        "composed": law.composed_acceptance,
        "multiplicative": law.multiplicative_prediction,
        "weakest_link": law.weakest_link_prediction,
        "verdict": law.verdict,
        "subsystem_acceptance": law.subsystem_acceptance,
    }


def run_eh4(config: EH4Config | None = None, *, oracle: HostOracle | None = None) -> dict[str, Any]:
    """Train the flat + factored arms on identical data; return ``{arm: {metric: value}}``."""
    from verisim.hostmodel import NeuralHostWorldModel
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, train_host_graph_model

    config = config or EH4Config()
    base = config.base
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)

    # flat arm (HC4 incr-1), trained exactly as EH1/EH3 do
    flat_gpt = train_flat(base, vocab, oracle, host)
    flat = NeuralHostWorldModel(flat_gpt, vocab)

    # factored arm (HC4 incr-2) on the same seeded data
    graph_examples = build_host_graph_dataset(
        oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
        n_steps=base.train_steps_per_traj,
    )
    factored = build_host_graph_model(
        vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
        mp_rounds=config.graph_mp_rounds, seed=base.model_seed,
    )
    train_host_graph_model(
        factored, graph_examples, steps=config.graph_iters, lr=base.lr,
        batch_size=config.graph_batch, seed=base.model_seed,
    )

    return {
        "flat": _arm_metrics(flat, oracle, host, base, config.composition_epsilon),
        "factored": _arm_metrics(factored, oracle, host, base, config.composition_epsilon),
    }


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH4 factored-vs-flat host comparison.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh4_factored_vs_flat.csv")
    args = parser.parse_args()
    config = EH4Config.from_json_file(args.config) if args.config else EH4Config()
    results = run_eh4(config)

    arms = ("flat", "factored")
    cols = ("delta_exact", "composed", "multiplicative", "weakest_link", "verdict")
    print(f"{'arm':<10} {'delta_exact':>12} {'composed':>10} {'mult':>8} {'weak':>8}  verdict")
    lines = ["arm," + ",".join(cols)]
    for arm in arms:
        r = results[arm]
        print(f"{arm:<10} {r['delta_exact']:>12.4f} {r['composed']:>10.4f} "
              f"{r['multiplicative']:>8.4f} {r['weakest_link']:>8.4f}  {r['verdict']}")
        lines.append(
            f"{arm},{r['delta_exact']:.6f},{r['composed']:.6f},"
            f"{r['multiplicative']:.6f},{r['weakest_link']:.6f},{r['verdict']}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(results, arms, out.with_suffix(".png"))


def _plot(results: dict[str, Any], arms: tuple[str, ...], path: Path) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(arms))
    ax1.bar(list(x), [results[a]["delta_exact"] for a in arms], width=0.5, color="#16a")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(arms)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("delta-exact rate (free decode)")
    ax1.set_title("one-step exactness: factored vs flat")

    w = 0.26
    ax2.bar([i - w for i in x], [results[a]["multiplicative"] for a in arms], w,
            label="∏ aᵢ (independent)", color="#4c72b0")
    ax2.bar(list(x), [results[a]["composed"] for a in arms], w,
            label="composed a (measured)", color="#dd8452")
    ax2.bar([i + w for i in x], [results[a]["weakest_link"] for a in arms], w,
            label="min aᵢ (weakest-link)", color="#55a868")
    for i, a in enumerate(arms):
        ax2.annotate(results[a]["verdict"], (i, results[a]["weakest_link"] + 0.02),
                     ha="center", fontsize="small", fontweight="bold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(arms)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("per-step acceptance a")
    ax2.set_title("composition law (H13): does factoring uncouple it?")
    ax2.legend(fontsize="small")
    fig.suptitle("Verisim EH4 — factored interaction-graph vs flat M_θ (DD-H1)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
