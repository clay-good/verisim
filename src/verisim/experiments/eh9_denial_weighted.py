"""Experiment EH9 -- the denial-weighted objective: can we close the EH8 denial-recall gap? (§9.4).

EH8 found the sharp negative: a host model can look ~94% privilege-faithful yet have **denied recall
≈ 0** -- it almost never predicts the EPERM/EBADF *failures* a defender most needs, because denials
are rare in the data and the cross-entropy is dominated by the common success case. EH9 attacks that
negative head-on with the program's standard lever -- *the oracle labels for free, so reweight the
data*: oversample the training transitions whose oracle outcome is a **denial** by a factor
``denial_oversample`` (the cheapest, trainer-agnostic intervention -- it touches the data, not
loss). It sweeps the factor for both arms and reads three numbers:

  - **denied recall** -- of the transitions truth says failed, the fraction the model predicts
    (the EH8 metric we are trying to lift);
  - **allowed specificity** -- of the transitions truth says succeeded, the fraction the model still
    predicts succeeded (the cost: does upweighting denials make the model cry wolf?);
  - **privilege-faithfulness** -- overall denied/allowed agreement (the headline that hid the gap).

*Does oversampling denials lift recall, and at what specificity cost?* A recall lift with little
specificity loss would say the gap is a data-balance artifact the free oracle can fix; a recall lift
that tanks specificity would say denials need a smarter signal than reweighting. Either is a datum.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.state import HostState
from verisim.hostloop.model import HostModel
from verisim.hostmetrics.privilege import privilege_faithfulness
from verisim.hostoracle.base import EXIT_OK, HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

from .eh1 import EH1Config, eval_actions
from .eh8_privilege import predicted_exit

_ARMS = ("flat", "factored")


@dataclass(frozen=True)
class EH9Config:
    base: EH1Config = field(default_factory=EH1Config)
    train_drivers: tuple[str, ...] = ("forky", "adversarial")  # adversarial carries the denials
    denial_oversamples: tuple[int, ...] = (1, 4, 16)
    max_pid: int = 64
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 800
    graph_batch: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH9Config:
        b = EH9Config()
        return EH9Config(
            base=EH1Config.from_dict(d.get("base", {})),
            train_drivers=tuple(d.get("train_drivers", b.train_drivers)),
            denial_oversamples=tuple(d.get("denial_oversamples", b.denial_oversamples)),
            max_pid=d.get("max_pid", b.max_pid),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_batch=d.get("graph_batch", b.graph_batch),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH9Config:
        return EH9Config.from_dict(json.loads(Path(path).read_text()))


def _denial_steps(oracle: HostOracle, config: HostConfig, drivers: tuple[str, ...],
                  seeds: tuple[int, ...], n_steps: int) -> list[tuple[HostState, Any, Any, bool]]:
    """Roll each (driver, seed); return ``(state, action, result, is_denial)`` per step."""
    import random as _random

    from verisim.hostdata.drivers import HostDriver

    steps: list[tuple[HostState, Any, Any, bool]] = []
    for driver in drivers:
        for seed in seeds:
            drv = HostDriver(name=driver, config=config, rng=_random.Random(seed))
            state = HostState.initial()
            for _ in range(n_steps):
                action = drv.sample(state)
                result = oracle.step(state, action)
                steps.append((state, action, result, result.exit_code != EXIT_OK))
                state = result.state
    return steps


def _eval_arm(model: HostModel, oracle: HostOracle, base: EH1Config) -> dict[str, float]:
    """Teacher-forced denied-recall / allowed-specificity / privilege-faithfulness for one arm."""
    host = DEFAULT_HOST_CONFIG
    pred: list[int] = []
    true: list[int] = []
    for _difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            state = HostState.initial()
            for action in eval_actions(oracle, host, driver, seed, base.eval_steps):
                pe = predicted_exit(model.predict_delta(state, action))
                result = oracle.step(state, action)
                pred.append(pe)
                true.append(result.exit_code)
                state = result.state
    denied = [i for i, t in enumerate(true) if t != EXIT_OK]
    allowed = [i for i, t in enumerate(true) if t == EXIT_OK]
    return {
        "denied_recall": fmean(float(pred[i] != EXIT_OK) for i in denied) if denied else 1.0,
        "allowed_specificity": (
            fmean(float(pred[i] == EXIT_OK) for i in allowed) if allowed else 1.0
        ),
        "privilege_faithfulness": privilege_faithfulness(pred, true),
        "n_denied": float(len(denied)),
    }


def run_eh9(
    config: EH9Config | None = None, *, oracle: HostOracle | None = None
) -> dict[str, dict[str, float]]:
    """Train each arm at each denial-oversample factor; return ``{f'{arm}@{k}': metrics}``."""
    import torch

    from verisim.hostmodel import (
        HostVocab,
        NeuralHostWorldModel,
        build_host_graph,
        encode_prompt,
        encode_target,
    )
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import train_host_graph_model
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_supervised

    config = config or EH9Config()
    base = config.base
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)
    steps = _denial_steps(
        oracle, host, config.train_drivers, base.train_seeds, base.train_steps_per_traj
    )

    results: dict[str, dict[str, float]] = {}
    for k in config.denial_oversamples:
        flat_examples = []
        graph_examples = []
        for state, action, result, is_denial in steps:
            reps = k if is_denial else 1
            flat_e = (encode_prompt(state, action, vocab), encode_target(result.delta, vocab))
            graph_e = (build_host_graph(state, action, host, vocab.max_pid),
                       encode_target(result.delta, vocab))
            flat_examples.extend([flat_e] * reps)
            graph_examples.extend([graph_e] * reps)

        torch.manual_seed(base.model_seed)
        torch.set_num_threads(1)
        gpt = GPT(GPTConfig(vocab_size=len(vocab), block_size=base.block_size,
                            n_layer=base.n_layer, n_head=base.n_head, n_embd=base.n_embd))
        train_supervised(gpt, flat_examples, vocab.pad, steps=base.train_iters, lr=base.lr,
                         seed=base.model_seed)
        results[f"flat@{k}"] = _eval_arm(NeuralHostWorldModel(gpt, vocab), oracle, base)

        factored = build_host_graph_model(vocab, host, max_pid=config.max_pid,
                                          d_model=config.graph_d_model,
                                          mp_rounds=config.graph_mp_rounds,
                                          seed=base.model_seed)
        train_host_graph_model(factored, graph_examples, steps=config.graph_iters, lr=base.lr,
                              batch_size=config.graph_batch, seed=base.model_seed)
        results[f"factored@{k}"] = _eval_arm(factored, oracle, base)
    return results


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH9 denial-weighted objective.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh9_denial_weighted.csv")
    args = parser.parse_args()
    config = EH9Config.from_json_file(args.config) if args.config else EH9Config()
    results = run_eh9(config)
    cols = ("denied_recall", "allowed_specificity", "privilege_faithfulness")
    print(f"{'arm@oversample':<16} {'denied_recall':>13} {'allowed_spec':>13} {'priv_faith':>11}")
    lines = ["arm,oversample," + ",".join(cols)]
    for arm in _ARMS:
        for k in config.denial_oversamples:
            r = results[f"{arm}@{k}"]
            print(f"{arm + '@' + str(k):<16} {r['denied_recall']:>13.3f} "
                  f"{r['allowed_specificity']:>13.3f} {r['privilege_faithfulness']:>11.3f}")
            lines.append(f"{arm},{k}," + ",".join(f"{r[c]:.6f}" for c in cols))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(results, config, out.with_suffix(".png"))


def _plot(  # pragma: no cover
    results: dict[str, dict[str, float]], config: EH9Config, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(config.denial_oversamples)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for arm in _ARMS:
        ax1.plot(ks, [results[f"{arm}@{k}"]["denied_recall"] for k in ks], marker="o", label=arm)
        ax2.plot(ks, [results[f"{arm}@{k}"]["allowed_specificity"] for k in ks],
                 marker="o", label=arm)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("denial oversample factor")
    ax1.set_ylabel("denied recall")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("does upweighting denials lift recall?")
    ax1.legend()
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("denial oversample factor")
    ax2.set_ylabel("allowed specificity")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("the cost: does it cry wolf on successes?")
    ax2.legend()
    fig.suptitle("Verisim EH9 — the denial-weighted objective (attacking the EH8 gap)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
