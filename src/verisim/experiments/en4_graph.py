"""Experiment EN4 (graph arm) -- the H11 graph-vs-flat comparison (SPEC-5 §12, §10.2 H11).

Trains the flat NW4 transformer ``M_θ`` and the NW8 GNN+RSSM graph arm on the **same** seeded
oracle data, then scores both with the **same** eval primitives EN1 uses, so the only difference
is the proposer's architecture (H22's model-invariance setup, used here to test H11: does the
graph+belief arm beat the flat-Markov one?). Two complementary numbers per arm:

  - **one-step held-out teacher-forced accuracy** -- the generalization gap on never-trained
    eval seeds (the K-series diagnostic);
  - **free-running faithful horizon ``H_0`` at ρ=0** -- the real drift measure: how many leading
    steps the unaided model stays bit-exact (SPEC §5.1, first-exceedance).

This is the machinery EN4 needs, run here as a small, fast instance (the EN1 honesty caveat: a
committed *smoke* comparison of the apparatus, not a tuned publication run). Whatever it shows is
a datum -- if the graph arm does not beat the flat arm at this scale, that bounds where structure
starts to pay and is reported as-is (SPEC-5 §12.1, the all-data-is-good-data stance).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from verisim.experiments.en1 import EN1Config, eval_actions
from verisim.experiments.en1 import train_model as train_flat
from verisim.loop.policy import fixed_interval_for_rho
from verisim.net.action import NetAction
from verisim.net.config import DEFAULT_NET_CONFIG, NetConfig
from verisim.net.state import NetworkState
from verisim.netdata.drivers import NetDriver
from verisim.netdelta.edits import NetDelta
from verisim.netloop import PartialNetOracle, budget_for_rho, run_net_rollout
from verisim.netloop.model import NetModel
from verisim.netmetrics.exact import delta_exact_rate
from verisim.netmodel import NetVocab, NeuralNetworkWorldModel, build_net_dataset
from verisim.netmodel.graph_model import build_graph_model
from verisim.netmodel.graph_train import (
    build_graph_dataset,
    graph_teacher_forced_accuracy,
    train_graph_model,
    train_graph_model_self_forced,
)
from verisim.netoracle import ReferenceNetworkOracle
from verisim.train.supervised import teacher_forced_accuracy


@dataclass(frozen=True)
class EN4Config:
    """Small, fast comparison instance. Scale up (more seeds/iters) for the publication run."""

    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    difficulties: tuple[tuple[str, str], ...] = (("low", "weighted"), ("high", "adversarial"))
    epsilons: tuple[float, ...] = (0.0, 0.05, 0.1)
    # graph-arm sizing / training
    d_model: int = 64
    mp_rounds: int = 3
    graph_iters: int = 800
    graph_noise_prob: float = 0.3
    # §6.3 self-forcing / scheduled-sampling lever
    selfforce_sample_prob: float = 0.5
    selfforce_refresh_every: int = 200
    model_seed: int = 0


def _faithful_horizon(divergences: list[float], epsilon: float) -> int:
    """Leading steps within ``epsilon`` (first-exceedance H_ε, SPEC §5.1)."""
    h = 0
    for d in divergences:
        if d <= epsilon:
            h += 1
        else:
            break
    return h


EvalTriple = tuple[NetworkState, NetAction, NetDelta]


def _eval_triples(
    oracle: ReferenceNetworkOracle,
    net: NetConfig,
    seeds: tuple[int, ...],
    n_steps: int,
    driver: str = "weighted",
) -> list[EvalTriple]:
    """Seeded ``(state, action, true_delta)`` triples for the held-out delta-exact eval."""
    triples: list[EvalTriple] = []
    for seed in seeds:
        driver_obj = NetDriver(name=driver, config=net, rng=random.Random(seed))
        state = NetworkState.initial(net.hosts)
        for _ in range(n_steps):
            action = driver_obj.sample(state)
            result = oracle.step(state, action)
            triples.append((state, action, result.delta))
            state = result.state
    return triples


def _delta_exact_rate(wm: NetModel, triples: list[EvalTriple]) -> float:
    """Fraction of held-out steps whose freely decoded delta exactly matches the oracle's."""
    return delta_exact_rate((wm.predict_delta(s, a), true) for s, a, true in triples)


def _mean_horizons(
    wm: NetModel, oracle: ReferenceNetworkOracle, net: NetConfig, config: EN4Config
) -> dict[float, float]:
    """Mean free-running (ρ=0) faithful horizon per ε over the eval cells (SPEC §5.1)."""
    partial = PartialNetOracle(oracle)
    policy = fixed_interval_for_rho(0.0)
    sums = {e: 0.0 for e in config.epsilons}
    cells = 0
    for _diff, driver in config.difficulties:
        for seed in config.eval_seeds:
            actions = eval_actions(oracle, net, driver, seed, config.eval_steps)
            roll = run_net_rollout(
                wm, partial, NetworkState.initial(net.hosts), actions, policy,
                epsilon=0.0, budget=budget_for_rho(0.0, len(actions)), seed=seed,
            )
            divs = list(roll.divergences)
            for e in config.epsilons:
                sums[e] += _faithful_horizon(divs, e)
            cells += 1
    return {e: sums[e] / cells for e in config.epsilons}


def run_en4_graph(config: EN4Config | None = None) -> dict[str, dict[str, float]]:
    """Train flat + graph (+ graph+noise) on identical data; return ``{arm: {metric: value}}``."""
    config = config or EN4Config()
    oracle = ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)

    # --- flat arm (reuse EN1's exact trainer) -----------------------------------
    flat_cfg = EN1Config(
        train_seeds=config.train_seeds,
        train_steps_per_traj=config.train_steps_per_traj,
        model_seed=config.model_seed,
    )
    flat_model = train_flat(flat_cfg, vocab, oracle, net)
    flat_wm = NeuralNetworkWorldModel(flat_model, vocab)

    # --- graph arms: clean, and with the §6.3 noise-injection lever --------------
    def _train_graph(noise_prob: float) -> NetModel:
        examples = build_graph_dataset(
            oracle, vocab, net, driver=flat_cfg.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj, noise_prob=noise_prob,
        )
        wm = build_graph_model(
            vocab, net, d_model=config.d_model, mp_rounds=config.mp_rounds, seed=config.model_seed
        )
        train_graph_model(wm, examples, steps=config.graph_iters, seed=config.model_seed)
        return wm

    graph_wm = _train_graph(0.0)
    graph_noise_wm = _train_graph(config.graph_noise_prob)

    # --- graph arm with the §6.3 self-forcing / scheduled-sampling lever ---------
    selfforce_wm = build_graph_model(
        vocab, net, d_model=config.d_model, mp_rounds=config.mp_rounds, seed=config.model_seed
    )
    train_graph_model_self_forced(
        selfforce_wm, oracle, vocab, net, driver=flat_cfg.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj, steps=config.graph_iters,
        refresh_every=config.selfforce_refresh_every, max_sample_prob=config.selfforce_sample_prob,
        seed=config.model_seed,
    )

    # --- held-out one-step teacher-forced accuracy ------------------------------
    flat_eval = build_net_dataset(
        oracle, vocab, net, driver="weighted", seeds=config.eval_seeds, n_steps=config.eval_steps
    )
    graph_eval = build_graph_dataset(
        oracle, vocab, net, driver="weighted", seeds=config.eval_seeds, n_steps=config.eval_steps
    )
    accs = {
        "flat": teacher_forced_accuracy(flat_model, flat_eval, vocab.pad),
        "graph": graph_teacher_forced_accuracy(graph_wm, graph_eval),  # type: ignore[arg-type]
        "graph+noise": graph_teacher_forced_accuracy(graph_noise_wm, graph_eval),  # type: ignore[arg-type]
        "graph+selfforce": graph_teacher_forced_accuracy(selfforce_wm, graph_eval),
    }

    # --- held-out per-step delta-exact rate (free decode, all arms via NetModel) ---
    triples = _eval_triples(oracle, net, config.eval_seeds, config.eval_steps)

    out: dict[str, dict[str, float]] = {}
    arms_wm: list[tuple[str, NetModel]] = [
        ("flat", flat_wm), ("graph", graph_wm), ("graph+noise", graph_noise_wm),
        ("graph+selfforce", selfforce_wm),
    ]
    for arm, wm in arms_wm:
        horizons = _mean_horizons(wm, oracle, net, config)
        row = {"onestep_acc": accs[arm], "delta_exact": _delta_exact_rate(wm, triples)}
        row.update({f"h@{e}": horizons[e] for e in config.epsilons})
        out[arm] = row
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN4 graph-vs-flat comparison (H11).")
    parser.add_argument("--graph-iters", type=int, default=800)
    parser.add_argument("--noise-prob", type=float, default=0.3)
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=[100, 101, 102])
    parser.add_argument("--out", type=str, default="figures/en4_graph_vs_flat.csv")
    args = parser.parse_args()
    cfg = EN4Config(
        eval_seeds=tuple(args.eval_seeds),
        graph_iters=args.graph_iters,
        graph_noise_prob=args.noise_prob,
    )
    results = run_en4_graph(cfg)

    arms = ("flat", "graph", "graph+noise", "graph+selfforce")
    eps = cfg.epsilons
    header = (
        f"{'arm':<12} {'onestep_acc':>12} {'delta_exact':>12}"
        + "".join(f"{'H@'+str(e):>10}" for e in eps)
    )
    print(header)
    lines = ["arm,onestep_acc,delta_exact," + ",".join(f"h@{e}" for e in eps)]
    for arm in arms:
        r = results[arm]
        print(
            f"{arm:<12} {r['onestep_acc']:>12.4f} {r['delta_exact']:>12.4f}"
            + "".join(f"{r[f'h@{e}']:>10.3f}" for e in eps)
        )
        lines.append(
            f"{arm},{r['onestep_acc']:.6f},{r['delta_exact']:.6f},"
            + ",".join(f"{r[f'h@{e}']:.6f}" for e in eps)
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(results, arms, eps, out.with_suffix(".png"))


def _plot(
    results: dict[str, dict[str, float]],
    arms: tuple[str, ...],
    eps: tuple[float, ...],
    path: Path,
) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(arms))
    ax1.bar([i - 0.2 for i in x], [results[a]["onestep_acc"] for a in arms],
            width=0.4, color="#2a7", label="one-step token acc")
    ax1.bar([i + 0.2 for i in x], [results[a]["delta_exact"] for a in arms],
            width=0.4, color="#16a", label="delta-exact rate")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
    ax1.set_title("one-step held-out: token acc vs delta-exact")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    for a in arms:
        ax2.plot(eps, [results[a][f"h@{e}"] for e in eps], marker="o", label=a)
    ax2.set_title("free-running faithful horizon H_ε (ρ=0)")
    ax2.set_xlabel("ε")
    ax2.set_ylabel("mean H_ε")
    ax2.legend()
    fig.suptitle("EN4 (smoke): graph+RSSM vs flat-Markov M_θ (H11)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":  # pragma: no cover
    main()
