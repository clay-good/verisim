"""K0 diagnostics — *where* does the v0 model fail? (SPEC-2.1 §4).

Before tuning anything, measure the failure modes, so the fix is targeted rather than a
blind sweep. On held-out trajectories (teacher-forced, so each step's error is uncompounded)
this collects:

  - **per-command accuracy** — which commands the model gets right/wrong (the hypothesis:
    structure-building writes are easy, cascading ``mv``/``rm -r`` are hard);
  - **per-edit-type precision/recall** — is the predicted delta missing edits, inventing
    them, or mis-typing them?
  - **accuracy by trajectory position** — does competence decay with depth (memory/context)
    or hold flat (a pure coverage problem)?
  - **train-vs-val gap** — the direct test of the SPEC-2.1 §1 diagnosis (memorize train,
    fail to generalize), the signature of an under-data/under-training regime;
  - **mean bits-to-correct** — the smooth gate (``metrics.bits``), reported as a baseline.

Everything is records-only (SPEC-2 §7.3): :func:`run_diagnostics` returns a structured
report regenerable from a config + seeds.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from verisim.delta.apply import apply
from verisim.delta.edits import Delta
from verisim.delta.serialize import edit_to_dict
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop.model import Model
from verisim.metrics.bits import bits_to_correct
from verisim.metrics.divergence import divergence, state_facts
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle

from .e1 import E1Config, eval_actions, train_model


def _edit_keys_by_op(delta: Delta) -> dict[str, list[str]]:
    """Group canonical edit identities by op type (for precision/recall)."""
    out: dict[str, list[str]] = {}
    for edit in delta:
        d = edit_to_dict(edit)
        out.setdefault(str(d["op"]), []).append(
            json.dumps(d, sort_keys=True, separators=(",", ":"))
        )
    return out


def _fact_type(fact: tuple[object, ...]) -> str:
    """Classify a divergence fact for the per-type residual breakdown."""
    tag = fact[0]
    named = {"\x00cwd": "cwd", "\x00env": "env", "\x00exit": "exit", "\x00stdout": "stdout"}
    if isinstance(tag, str) and tag in named:
        return named[tag]
    return str(fact[1])  # "file" or "dir"


@dataclass
class DiagnosticReport:
    per_command: dict[str, list[int]]  # command -> [correct, total]
    per_edit_pr: dict[str, list[int]]  # op -> [true_positive, predicted, true]
    accuracy_by_position: list[list[int]]  # position -> [correct, total]
    divergence_by_fact_type: dict[str, int]  # fact type -> count of divergent facts
    mean_bits_to_correct: float
    train_accuracy: float
    val_accuracy: float

    def to_dict(self) -> dict[str, object]:
        return {
            "per_command": self.per_command,
            "per_edit_pr": self.per_edit_pr,
            "accuracy_by_position": self.accuracy_by_position,
            "divergence_by_fact_type": self.divergence_by_fact_type,
            "mean_bits_to_correct": self.mean_bits_to_correct,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }


class _Accumulator:
    def __init__(self) -> None:
        self.per_command: dict[str, list[int]] = {}
        self.per_edit_pr: dict[str, list[int]] = {}
        self.position: list[list[int]] = []
        self.bits: list[float] = []
        self.div_by_type: Counter[str] = Counter()

    def add_state_divergence(self, predicted_state: State, true_state: State) -> None:
        """Tally divergent state facts by type (which fact class is the residual?)."""
        for fact in state_facts(predicted_state) ^ state_facts(true_state):
            self.div_by_type[_fact_type(fact)] += 1

    def add_step(self, position: int, action: Action, predicted: Delta, true: Delta) -> None:
        exact = predicted == true
        cmd = self.per_command.setdefault(action.name, [0, 0])
        cmd[0] += int(exact)
        cmd[1] += 1
        while len(self.position) <= position:
            self.position.append([0, 0])
        self.position[position][0] += int(exact)
        self.position[position][1] += 1
        self.bits.append(bits_to_correct(predicted, true))
        pred_keys = _edit_keys_by_op(predicted)
        true_keys = _edit_keys_by_op(true)
        for op in set(pred_keys) | set(true_keys):
            pr = self.per_edit_pr.setdefault(op, [0, 0, 0])
            pk = list(pred_keys.get(op, []))
            tk = list(true_keys.get(op, []))
            tp = 0
            remaining = list(tk)
            for k in pk:
                if k in remaining:
                    remaining.remove(k)
                    tp += 1
            pr[0] += tp
            pr[1] += len(pk)
            pr[2] += len(tk)


def _accuracy(model: Model, oracle: Oracle, s0: State, actions: list[Action]) -> float:
    """Teacher-forced fraction of steps whose predicted delta exactly reproduces the truth."""
    if not actions:
        return 1.0
    state = s0
    correct = 0
    for action in actions:
        result = oracle.step(state, action)
        if divergence(apply(state, model.predict_delta(state, action)), result.state) == 0.0:
            correct += 1
        state = result.state
    return correct / len(actions)


def run_diagnostics(
    config: E1Config | None = None, *, oracle: Oracle | None = None
) -> DiagnosticReport:
    """Train the baseline model and diagnose *where* it fails on held-out trajectories."""
    config = config or E1Config()
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    model: Model = NeuralWorldModel(train_model(config, vocab, oracle, env), vocab)

    acc = _Accumulator()
    for driver in config.difficulties.values():
        for seed in config.eval_seeds:
            actions = eval_actions(oracle, env, driver, seed, config.eval_steps)
            state = State.empty()
            for pos, action in enumerate(actions):
                result = oracle.step(state, action)
                predicted = model.predict_delta(state, action)
                acc.add_step(pos, action, predicted, result.delta)
                acc.add_state_divergence(apply(state, predicted), result.state)
                state = result.state

    # Train-vs-val gap: accuracy on the *training* action sequences vs held-out eval seeds.
    train_acc_vals: list[float] = []
    for seed in config.train_seeds:
        actions = eval_actions(oracle, env, config.train_driver, seed, config.train_steps_per_traj)
        train_acc_vals.append(_accuracy(model, oracle, State.empty(), actions))
    val_acc_vals: list[float] = []
    for driver in config.difficulties.values():
        for seed in config.eval_seeds:
            actions = eval_actions(oracle, env, driver, seed, config.eval_steps)
            val_acc_vals.append(_accuracy(model, oracle, State.empty(), actions))

    mean_bits = sum(acc.bits) / len(acc.bits) if acc.bits else 0.0
    return DiagnosticReport(
        per_command=acc.per_command,
        per_edit_pr=acc.per_edit_pr,
        accuracy_by_position=acc.position,
        divergence_by_fact_type=dict(acc.div_by_type.most_common()),
        mean_bits_to_correct=mean_bits,
        train_accuracy=sum(train_acc_vals) / len(train_acc_vals) if train_acc_vals else 0.0,
        val_accuracy=sum(val_acc_vals) / len(val_acc_vals) if val_acc_vals else 0.0,
    )


def write_report(report: DiagnosticReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run the K0 diagnostics battery (SPEC-2.1 §4).")
    parser.add_argument("--config", type=str, default=None, help="path to an E1-style config")
    parser.add_argument("--out", type=str, default="runs/k0/diagnostics.json")
    args = parser.parse_args()
    config = E1Config.from_json_file(args.config) if args.config else E1Config()
    report = run_diagnostics(config)
    path = write_report(report, args.out)
    print(
        f"wrote diagnostics to {path}; train_acc={report.train_accuracy:.3f} "
        f"val_acc={report.val_accuracy:.3f} mean_bits={report.mean_bits_to_correct:.2f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
