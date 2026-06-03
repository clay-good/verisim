"""Experiment EH8 -- privilege-faithfulness: does the model get security-critical *failures* right?
(§9.4).

A defender trusts a host simulator only if it predicts *denials* correctly: that a ``setuid`` by a
non-root process is **EPERM**, that a write to a closed fd is **EBADF** -- the failures that mean
"the
attacker did not get the privilege". This is the security core of SPEC-6 (§9.4, §3.2): success is
easy, denial is the safety-relevant prediction. EH8 measures privilege-faithfulness for the flat and
factored arms on a denial-heavy workload (the ``adversarial`` driver is
``setuid``/``exit``/``close``
heavy, so EPERM/EBADF abound), teacher-forced, reading the exit code off each predicted bundle
delta:

  - **overall** privilege-faithfulness -- denied/allowed agreement over every transition (§9.4);
  - **setuid** privilege-faithfulness -- the same over ``setuid`` transitions only (the privilege
  axis);
  - **denied recall** -- of the transitions truth says *failed*, the fraction the model also
  predicts
    failed: the security-critical number (missing a denial is the dangerous error -- the model would
    tell a defender an unprivileged action *succeeded* when it did not).

*Does a more faithful model (factored ≫ flat per EH4) translate its one-step accuracy into better
privilege prediction, especially denial recall?* Whatever it shows is a datum; the apparatus makes
the
security-relevant slice of faithfulness first-class.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.delta import HostDelta, SetExit
from verisim.host.state import HostState
from verisim.hostloop.model import HostModel
from verisim.hostmetrics.privilege import privilege_faithfulness
from verisim.hostoracle.base import EXIT_OK
from verisim.hostoracle.reference import ReferenceHostOracle

from .eh1 import EH1Config, eval_actions

_ARMS = ("flat", "factored")


@dataclass(frozen=True)
class EH8Config:
    base: EH1Config = field(default_factory=EH1Config)
    max_pid: int = 64
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 800
    graph_batch: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH8Config:
        b = EH8Config()
        return EH8Config(
            base=EH1Config.from_dict(d.get("base", {})),
            max_pid=d.get("max_pid", b.max_pid),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_batch=d.get("graph_batch", b.graph_batch),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH8Config:
        return EH8Config.from_dict(json.loads(Path(path).read_text()))


def predicted_exit(delta: HostDelta) -> int:
    """The exit code the model's predicted bundle delta reports (its last ``SetExit``; else
    EXIT_OK)."""
    code = EXIT_OK
    for edit in delta:
        if isinstance(edit, SetExit):
            code = edit.exit_code
    return code


def _arm_metrics(
    model: HostModel, oracle: ReferenceHostOracle, base: EH1Config
) -> dict[str, float]:
    """Teacher-forced privilege-faithfulness for one arm, pooled over the eval rollouts."""
    host = DEFAULT_HOST_CONFIG
    pred_all: list[int] = []
    true_all: list[int] = []
    pred_su: list[int] = []
    true_su: list[int] = []
    for _difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            state = HostState.initial()
            for action in eval_actions(oracle, host, driver, seed, base.eval_steps):
                pe = predicted_exit(model.predict_delta(state, action))
                result = oracle.step(state, action)
                pred_all.append(pe)
                true_all.append(result.exit_code)
                if action.name == "setuid":
                    pred_su.append(pe)
                    true_su.append(result.exit_code)
                state = result.state  # teacher-forced
    denied_idx = [i for i, t in enumerate(true_all) if t != EXIT_OK]
    denied_recall = (
        sum(1 for i in denied_idx if pred_all[i] != EXIT_OK) / len(denied_idx)
        if denied_idx else 1.0
    )
    return {
        "privilege_faithfulness": privilege_faithfulness(pred_all, true_all),
        "setuid_faithfulness": privilege_faithfulness(pred_su, true_su),
        "denied_recall": denied_recall,
        "n_denied": float(len(denied_idx)),
        "n_transitions": float(len(true_all)),
    }


def run_eh8(
    config: EH8Config | None = None, *, oracle: ReferenceHostOracle | None = None
) -> dict[str, dict[str, float]]:
    """Train the flat + factored arms; return privilege-faithfulness metrics for each."""
    from verisim.hostmodel import HostVocab, NeuralHostWorldModel
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, train_host_graph_model

    from .eh1 import train_model as train_flat

    config = config or EH8Config()
    base = config.base
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)

    flat = NeuralHostWorldModel(train_flat(base, vocab, oracle, host), vocab)
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
    models: dict[str, HostModel] = {"flat": flat, "factored": factored}
    return {arm: _arm_metrics(models[arm], oracle, base) for arm in _ARMS}


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH8 privilege-faithfulness (flat vs factored).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh8_privilege.csv")
    args = parser.parse_args()
    config = EH8Config.from_json_file(args.config) if args.config else EH8Config()
    results = run_eh8(config)
    cols = ("privilege_faithfulness", "setuid_faithfulness", "denied_recall", "n_denied")
    print(
        f"{'arm':<10} {'priv_faith':>11} {'setuid_faith':>13} "
        f"{'denied_recall':>14} {'n_denied':>9}"
    )
    lines = ["arm," + ",".join(cols)]
    for arm in _ARMS:
        r = results[arm]
        print(f"{arm:<10} {r['privilege_faithfulness']:>11.3f} {r['setuid_faithfulness']:>13.3f} "
              f"{r['denied_recall']:>14.3f} {r['n_denied']:>9.0f}")
        lines.append(f"{arm}," + ",".join(f"{r[c]:.6f}" for c in cols))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    _plot(results, out.with_suffix(".png"))


def _plot(results: dict[str, dict[str, float]], path: Path) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("privilege_faithfulness", "setuid_faithfulness", "denied_recall")
    labels = ("overall", "setuid only", "denied recall")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = range(len(metrics))
    w = 0.38
    for i, arm in enumerate(_ARMS):
        ax.bar([j + (i - 0.5) * w for j in x], [results[arm][m] for m in metrics], w, label=arm)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("privilege-faithfulness (denied/allowed agreement)")
    ax.set_title("Verisim EH8 — does the model get security-critical denials right?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
