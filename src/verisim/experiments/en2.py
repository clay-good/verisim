"""Experiment EN2 -- consultation-policy comparison (SPEC-5 §12, H9, milestone NW7).

The network analogue of v0's E2 (:mod:`verisim.experiments.e2`). Fixes the budget ``ρ`` at
the EN1 interior and compares the §8.1 consultation policies ``π_c`` -- ``fixed`` vs.
``uncertainty_triggered`` vs. ``drift_triggered`` -- at *equal* ``ρ`` (the runner's spend-down
backstop makes every arm spend exactly ``floor(ρ·T)`` consultations, so the comparison
isolates *where* a policy spends its budget). This is **H9**: does spending the budget on the
steps the model is least sure about *earn* more faithful horizon than spreading it evenly?

The triggered policies read the flat ``M_θ``'s mean decode entropy as the per-step
uncertainty signal, with thresholds ``τ`` calibrated per rollout to the budget (the v0 E2
recipe, reused verbatim via :func:`verisim.experiments.e2.build_policy`). EN2 runs in
**full-consultation** mode so it is directly comparable to v0's E2 and tests H9 cleanly.

Scope note (SPEC-5 §8.2, H10): EN2's *other* axis -- the probe-selection policy ``π_o``
(*what* to observe) -- needs a model that localizes its belief uncertainty per host, which is
the RSSM belief the NW7 graph arm supplies (§6.2). The NW5 loop already ships the probe-policy
*interface* and the dependency-free baselines (``RandomProbe``/``RoundRobinProbe``); the smart
information-gain ``π_o`` that could *beat* them -- the real H10 test -- lands with that arm. So
EN2 here is the H9 (when) result on the flat arm; H10 (what) is deferred with its model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.experiments.e2 import build_policy
from verisim.metrics.record import RunRecord, write_records
from verisim.net.action import NetAction
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.net.state import NetworkState
from verisim.netdelta.apply import apply
from verisim.netloop import PartialNetOracle, budget_for_rho, run_net_rollout
from verisim.netloop.model import NetUncertaintyModel
from verisim.netmodel import NetVocab, NeuralNetworkWorldModel
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .en1 import EN1Config, eval_actions, train_model


@dataclass(frozen=True)
class EN2Config:
    name: str = "en2-small"
    base: EN1Config = field(default_factory=EN1Config)
    rho: float = 0.3  # the EN1 interior the policies are compared at
    policies: tuple[str, ...] = ("fixed", "uncertainty", "drift")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EN2Config:
        base = EN2Config()
        return EN2Config(
            name=d.get("name", base.name),
            base=EN1Config.from_dict(d.get("base", {})),
            rho=d.get("rho", base.rho),
            policies=tuple(d.get("policies", base.policies)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EN2Config:
        return EN2Config.from_dict(json.loads(Path(path).read_text()))


def unaided_signals(
    model: NetUncertaintyModel, s0: NetworkState, actions: list[NetAction]
) -> list[float]:
    """The model's per-step uncertainty along the unaided (ρ=0) network rollout."""
    state = s0
    signals: list[float] = []
    for action in actions:
        delta, signal = model.predict_delta_with_uncertainty(state, action)
        signals.append(signal)
        state = apply(state, delta)
    return signals


def run_en2(config: EN2Config | None = None, *, oracle: NetOracle | None = None) -> list[RunRecord]:
    """Train the model and compare the consultation policies at the fixed interior ``ρ``."""
    config = config or EN2Config()
    base = config.base
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG
    vocab = NetVocab(net)
    model = train_model(base, vocab, oracle, net)
    world_model = NeuralNetworkWorldModel(model, vocab)
    partial = PartialNetOracle(oracle)

    records: list[RunRecord] = []
    for difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            actions = eval_actions(oracle, net, driver, seed, base.eval_steps)
            signals = unaided_signals(world_model, NetworkState.initial(net.hosts), actions)
            budget = budget_for_rho(config.rho, len(actions))
            for policy_name in config.policies:
                policy = build_policy(policy_name, config.rho, signals)
                rollout = run_net_rollout(
                    world_model, partial, NetworkState.initial(net.hosts), actions, policy,
                    epsilon=base.epsilons[0], budget=budget, seed=seed,
                )
                for epsilon in base.epsilons:
                    records.append(
                        RunRecord(
                            config={
                                "experiment": config.name,
                                "policy": policy_name,
                                "difficulty": difficulty,
                                "driver": driver,
                                "rho": config.rho,
                                "n_steps": len(actions),
                                "oracle_bits": rollout.config["oracle_bits"],
                            },
                            seed=seed,
                            epsilon=epsilon,
                            divergences=list(rollout.divergences),
                            consultation_schedule=list(rollout.consultation_schedule),
                        )
                    )
    return records


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment EN2 (network policy comparison).")
    parser.add_argument("--config", type=str, default=None, help="path to an EN2 config JSON")
    parser.add_argument("--out", type=str, default="runs/en2/records.jsonl")
    args = parser.parse_args()
    config = EN2Config.from_json_file(args.config) if args.config else EN2Config()
    records = run_en2(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
