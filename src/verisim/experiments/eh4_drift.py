"""Experiment EH4-drift -- the §6.3 drift-lever ablation on the factored arm (SPEC-6 §6.3, HC7).

EH4 showed the factored arm is far more one-step-accurate than flat, yet (like every arm in the
program) it still drifts at ``ρ=0``: good per-step prediction does not buy free-running horizon (the
standing one-step→horizon wall). The §6.3 drift levers are the standard attack on that gap, and
they are **required ablation arms** here. EH4-drift trains the factored arm three ways on
*identical* seeds and scores each with the same eval primitives:

  - **clean** -- pure teacher forcing (the EH4 baseline);
  - **+noise** -- oracle-relabeled state-noise augmentation (the GNS lever, §6.3): broaden the input
    distribution toward random off-trajectory states, with exact targets;
  - **+self-forcing** -- scheduled sampling (§6.3): re-roll on the model's *own* predictions and
    oracle-relabel, broadening toward the actual deploy (drift) distribution.

Two numbers per arm: **delta-exact rate** (free decode == oracle delta, teacher-forced) and the
**free-running faithful horizon ``H_ε`` at ρ=0** (the real drift measure). *Does either lever turn
the factored arm's one-step accuracy into horizon?* The network found a banked negative (a small
one-step dip, no horizon); whatever the host shows at this smoke scale is a datum, reported as-is.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.state import HostState
from verisim.hostloop import PartialHostOracle, run_host_rollout
from verisim.hostloop.model import HostModel
from verisim.hostmodel import HostVocab
from verisim.hostmodel.graph_model import GraphHostWorldModel, build_host_graph_model
from verisim.hostmodel.graph_train import (
    build_host_graph_dataset,
    train_host_graph_model,
    train_host_graph_model_self_forced,
)
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import Never
from verisim.metrics.horizon import faithful_horizon

from .eh1 import EH1Config, eval_actions

_ARMS = ("clean", "noise", "self_forcing")


@dataclass(frozen=True)
class EH4DriftConfig:
    name: str = "eh4-drift-small"
    base: EH1Config = field(default_factory=EH1Config)
    max_pid: int = 64
    graph_iters: int = 800
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_batch: int = 32
    noise_prob: float = 0.3
    sf_rounds: int = 4
    sf_sample_prob: float = 0.5

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH4DriftConfig:
        b = EH4DriftConfig()
        return EH4DriftConfig(
            name=d.get("name", b.name),
            base=EH1Config.from_dict(d.get("base", {})),
            max_pid=d.get("max_pid", b.max_pid),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_batch=d.get("graph_batch", b.graph_batch),
            noise_prob=d.get("noise_prob", b.noise_prob),
            sf_rounds=d.get("sf_rounds", b.sf_rounds),
            sf_sample_prob=d.get("sf_sample_prob", b.sf_sample_prob),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH4DriftConfig:
        import json

        return EH4DriftConfig.from_dict(json.loads(Path(path).read_text()))


def _new_model(config: EH4DriftConfig, vocab: HostVocab) -> GraphHostWorldModel:
    return build_host_graph_model(
        vocab, DEFAULT_HOST_CONFIG, max_pid=config.max_pid, d_model=config.graph_d_model,
        mp_rounds=config.graph_mp_rounds, seed=config.base.model_seed,
    )


def _train_arm(
    arm: str, config: EH4DriftConfig, vocab: HostVocab, oracle: HostOracle, host: HostConfig
) -> GraphHostWorldModel:
    """Train one drift-lever arm to the same total step budget (``graph_iters``)."""
    base = config.base
    model = _new_model(config, vocab)
    if arm == "clean":
        examples = build_host_graph_dataset(
            oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
            n_steps=base.train_steps_per_traj,
        )
        train_host_graph_model(
            model, examples, steps=config.graph_iters, lr=base.lr,
            batch_size=config.graph_batch, seed=base.model_seed,
        )
    elif arm == "noise":
        examples = build_host_graph_dataset(
            oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
            n_steps=base.train_steps_per_traj, noise_prob=config.noise_prob,
            noise_seed=base.model_seed,
        )
        train_host_graph_model(
            model, examples, steps=config.graph_iters, lr=base.lr,
            batch_size=config.graph_batch, seed=base.model_seed,
        )
    else:  # self_forcing -- rounds * steps_per_round == graph_iters
        train_host_graph_model_self_forced(
            model, oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
            n_steps=base.train_steps_per_traj, rounds=config.sf_rounds,
            steps_per_round=config.graph_iters // config.sf_rounds,
            sample_prob=config.sf_sample_prob, lr=base.lr, batch_size=config.graph_batch,
            seed=base.model_seed,
        )
    return model


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
    model: HostModel, oracle: HostOracle, host: HostConfig, base: EH1Config
) -> dict[str, float]:
    """Score one arm: delta-exact + free-running ``H_ε`` at ρ=0 per ε (the drift measure)."""
    partial = PartialHostOracle(oracle)
    exact: list[float] = []
    horizons: dict[float, list[float]] = {e: [] for e in base.epsilons}
    for _difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            actions = eval_actions(oracle, host, driver, seed, base.eval_steps)
            exact.append(_delta_exact_rate(model, oracle, actions))
            rollout = run_host_rollout(
                model, partial, HostState.initial(), actions, Never(),
                epsilon=base.epsilons[0], budget=0, seed=seed,
            )
            for e in base.epsilons:
                horizons[e].append(float(faithful_horizon(rollout.divergences, e)))
    out = {"delta_exact": fmean(exact) if exact else 0.0}
    for e in base.epsilons:
        out[f"h@{e}"] = fmean(horizons[e]) if horizons[e] else 0.0
    return out


def run_eh4_drift(
    config: EH4DriftConfig | None = None, *, oracle: HostOracle | None = None
) -> dict[str, dict[str, float]]:
    """Train the three drift-lever arms on identical data; return ``{arm: {metric: value}}``."""
    config = config or EH4DriftConfig()
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)
    return {
        arm: _arm_metrics(_train_arm(arm, config, vocab, oracle, host), oracle, host, config.base)
        for arm in _ARMS
    }


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH4-drift (§6.3 drift-lever ablation).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh4_drift.csv")
    args = parser.parse_args()
    config = EH4DriftConfig.from_json_file(args.config) if args.config else EH4DriftConfig()
    results = run_eh4_drift(config)
    eps = config.base.epsilons

    print(f"{'arm':<14} {'delta_exact':>12}" + "".join(f"{'H@'+str(e):>10}" for e in eps))
    lines = ["arm,delta_exact," + ",".join(f"h@{e}" for e in eps)]
    for arm in _ARMS:
        r = results[arm]
        print(
            f"{arm:<14} {r['delta_exact']:>12.4f}"
            + "".join(f"{r[f'h@{e}']:>10.3f}" for e in eps)
        )
        lines.append(
            f"{arm},{r['delta_exact']:.6f}," + ",".join(f"{r[f'h@{e}']:.6f}" for e in eps)
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(results, eps, out.with_suffix(".png"))


def _plot(  # pragma: no cover
    results: dict[str, dict[str, float]], eps: tuple[float, ...], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(_ARMS))
    ax1.bar(list(x), [results[a]["delta_exact"] for a in _ARMS], width=0.5, color="#16a")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(_ARMS)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("delta-exact rate (free decode)")
    ax1.set_title("one-step exactness by drift lever")
    for a in _ARMS:
        ax2.plot(eps, [results[a][f"h@{e}"] for e in eps], marker="o", label=a)
    ax2.set_xlabel("ε")
    ax2.set_ylabel("free-running H_ε at ρ=0 (steps)")
    ax2.set_title("does a drift lever buy horizon?")
    ax2.legend(fontsize="small")
    fig.suptitle("Verisim EH4-drift — §6.3 levers on the factored arm (noise / self-forcing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
