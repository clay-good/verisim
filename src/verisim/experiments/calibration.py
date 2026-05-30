"""Uncertainty-calibration diagnostic (SPEC-2 §7.2, §17.2).

Collects per-step ``(signal, divergence)`` pairs from the neural model and reports
how well its decode-entropy confidence predicts its actual per-step error -- the
diagnostic that explains the E2 (H2) negative (an *un*calibrated signal cannot beat
even-spacing, M7). Pairs are collected **teacher-forced** (the rollout advances
along the oracle's truth, not the model's prediction), so each pair is one step's
confidence vs. that same step's error, uncompounded -- exactly the §7.2 "reliability
of confidence vs. actual per-step divergence".

The figure (`figures/plot_calibration.py`) and CSV are produced from the committed
pairs only (SPEC-2 §7.3, §12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.delta.apply import apply
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.loop.model import UncertaintyModel
from verisim.metrics.calibration import CalibrationReport, Pair, calibration_report
from verisim.metrics.divergence import divergence
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle

from .e1 import E1Config, eval_actions, train_model


@dataclass(frozen=True)
class CalibrationConfig:
    name: str = "calibration-small"
    base: E1Config = field(default_factory=E1Config)
    n_bins: int = 10

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CalibrationConfig:
        base = CalibrationConfig()
        return CalibrationConfig(
            name=d.get("name", base.name),
            base=E1Config.from_dict(d.get("base", {})),
            n_bins=d.get("n_bins", base.n_bins),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CalibrationConfig:
        return CalibrationConfig.from_dict(json.loads(Path(path).read_text()))


def collect_pairs(
    model: UncertaintyModel, oracle: Oracle, s0: State, actions: list[Action]
) -> list[Pair]:
    """Teacher-forced per-step ``(signal, divergence)`` pairs along ``actions``."""
    state = s0
    pairs: list[Pair] = []
    for action in actions:
        delta, signal = model.predict_delta_with_uncertainty(state, action)
        truth = oracle.step(state, action).state
        pairs.append((signal, divergence(apply(state, delta), truth)))
        state = truth  # teacher-forced: per-step error, not compounded
    return pairs


def run_calibration(
    config: CalibrationConfig | None = None, *, oracle: Oracle | None = None
) -> tuple[CalibrationReport, list[Pair]]:
    """Train the model and report its uncertainty calibration over the eval set."""
    config = config or CalibrationConfig()
    base = config.base
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    model = train_model(base, vocab, oracle, env)
    world_model = NeuralWorldModel(model, vocab)

    pairs: list[Pair] = []
    for driver in base.difficulties.values():
        for seed in base.eval_seeds:
            actions = eval_actions(oracle, env, driver, seed, base.eval_steps)
            pairs += collect_pairs(world_model, oracle, State.empty(), actions)
    return calibration_report(pairs, n_bins=config.n_bins), pairs


def write_pairs(pairs: list[Pair], path: str | Path) -> Path:
    """Write ``(signal, divergence)`` pairs as JSONL (the figure's source records)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps({"signal": s, "divergence": d}) for s, d in pairs) + "\n"
    )
    return out


def read_pairs(path: str | Path) -> list[Pair]:
    """Read a pairs JSONL written by :func:`write_pairs`."""
    text = Path(path).read_text().strip()
    return [
        (float(r["signal"]), float(r["divergence"]))
        for line in text.splitlines()
        if line
        for r in [json.loads(line)]
    ]


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run the uncertainty-calibration diagnostic.")
    parser.add_argument("--config", type=str, default=None, help="path to a calibration config")
    parser.add_argument("--out", type=str, default="runs/calibration/pairs.jsonl")
    args = parser.parse_args()
    config = CalibrationConfig.from_json_file(args.config) if args.config else CalibrationConfig()
    report, pairs = run_calibration(config)
    path = write_pairs(pairs, args.out)
    print(
        f"wrote {len(pairs)} pairs to {path}; "
        f"pearson={report.pearson:.3f} spearman={report.spearman:.3f} "
        f"mean_divergence={report.mean_divergence:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
