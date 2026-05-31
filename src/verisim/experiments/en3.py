"""Experiment EN3 -- correction/belief-operator comparison (SPEC-5 §12, §8.3, milestone NW7).

The network analogue of v0's E3 (:mod:`verisim.experiments.e3`), and the experiment that
cashes in the one thing partial observability buys that v0 could not show. Fixes the
consultation policy and budget ``ρ`` and compares the §8.3 operators at *equal* ``ρ``:

  - **full-consultation** operators -- ``hard_reset`` / ``residual`` / ``projection`` -- which
    (as in v0) all snap the coupled state to the complete one-step truth, so their headline
    ``H_ε`` is **identical** (the full-truth identity, reported with CIs); they differ only in
    the diagnostic they expose.
  - **probe** + ``belief_filter`` -- a cheap one-host probe corrects only the observed
    subgraph, so it corrects *strictly less* than a full consult. Its ``H_ε`` is therefore
    **genuinely lower** at equal consultation count -- the v0 identity collapse is **broken**
    (SPEC-5 §8.3, wall W5). The honest read is the cost lens: the probe spends far fewer
    oracle-bits per consult (recorded as ``oracle_bits``), so the operator that *earns* the
    most faithful horizon per oracle-bit -- not per consult -- is the real question, the §9.4
    probe-efficiency framing the smart-sensing NW7 work (EN2/H10) then optimizes.

Output: faithful horizon and the per-operator diagnostic + mean oracle-bits by operator. The
figure (``figures/plot_comparison.py --key operator``) and CSV come from these records only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.experiments.e2 import build_policy
from verisim.metrics.record import RunRecord, write_records
from verisim.net.config import DEFAULT_NET_CONFIG
from verisim.net.state import NetworkState
from verisim.netloop import (
    BeliefFilter,
    HardReset,
    PartialNetOracle,
    Projection,
    Residual,
    RoundRobinProbe,
    budget_for_rho,
    run_net_rollout,
)
from verisim.netmodel import NetVocab, NeuralNetworkWorldModel
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .en1 import EN1Config, eval_actions, train_model
from .en2 import unaided_signals

# The compared operators: the three full-consult operators (which coincide on H_ε) plus the
# probe-mode belief filter (which does not -- the no-identity-collapse result).
_FULL_OPERATORS = ("hard_reset", "residual", "projection")
_PROBE_OPERATOR = "belief_filter"


@dataclass(frozen=True)
class EN3Config:
    name: str = "en3-small"
    base: EN1Config = field(default_factory=EN1Config)
    rho: float = 0.3
    policy: str = "fixed"  # the policy held fixed across operators
    operators: tuple[str, ...] = (*_FULL_OPERATORS, _PROBE_OPERATOR)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EN3Config:
        base = EN3Config()
        return EN3Config(
            name=d.get("name", base.name),
            base=EN1Config.from_dict(d.get("base", {})),
            rho=d.get("rho", base.rho),
            policy=d.get("policy", base.policy),
            operators=tuple(d.get("operators", base.operators)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EN3Config:
        return EN3Config.from_dict(json.loads(Path(path).read_text()))


def run_en3(config: EN3Config | None = None, *, oracle: NetOracle | None = None) -> list[RunRecord]:
    """Train the model and compare the operators at the fixed policy + interior ``ρ``."""
    config = config or EN3Config()
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
            policy = build_policy(config.policy, config.rho, signals)
            budget = budget_for_rho(config.rho, len(actions))
            for operator_name in config.operators:
                diagnostic, rollout = _run_operator(
                    operator_name, world_model, partial, net, actions, policy, budget,
                    epsilon=base.epsilons[0], seed=seed,
                )
                for epsilon in base.epsilons:
                    records.append(
                        RunRecord(
                            config={
                                "experiment": config.name,
                                "operator": operator_name,
                                "policy": config.policy,
                                "difficulty": difficulty,
                                "driver": driver,
                                "rho": config.rho,
                                "n_steps": len(actions),
                                "diagnostic": diagnostic,
                                "oracle_bits": rollout.config["oracle_bits"],
                            },
                            seed=seed,
                            epsilon=epsilon,
                            divergences=list(rollout.divergences),
                            consultation_schedule=list(rollout.consultation_schedule),
                        )
                    )
    return records


def _run_operator(
    name: str, world_model: NeuralNetworkWorldModel, partial: PartialNetOracle,
    net: Any, actions: list[Any], policy: Any, budget: int, *, epsilon: float, seed: int,
) -> tuple[float | None, RunRecord]:
    """Run one operator's rollout; return its mean per-correction diagnostic + the record."""
    s0 = NetworkState.initial(net.hosts)
    if name == "hard_reset":
        rollout = run_net_rollout(
            world_model, partial, s0, actions, policy, epsilon=epsilon,
            operator=HardReset(), budget=budget, seed=seed,
        )
        return None, rollout
    if name == "residual":
        res = Residual()
        rollout = run_net_rollout(
            world_model, partial, s0, actions, policy, epsilon=epsilon,
            operator=res, budget=budget, seed=seed,
        )
        return (fmean(res.discrepancies) if res.discrepancies else 0.0), rollout
    if name == "projection":
        proj = Projection()
        rollout = run_net_rollout(
            world_model, partial, s0, actions, policy, epsilon=epsilon,
            operator=proj, budget=budget, seed=seed,
        )
        return (fmean(proj.repaired_fractions) if proj.repaired_fractions else 0.0), rollout
    if name == "belief_filter":
        bop = BeliefFilter()
        rollout = run_net_rollout(
            world_model, partial, s0, actions, policy, epsilon=epsilon,
            probe_policy=RoundRobinProbe(net.hosts), belief_op=bop, budget=budget, seed=seed,
        )
        return (fmean(bop.repaired_fractions) if bop.repaired_fractions else 0.0), rollout
    raise ValueError(f"unknown operator {name!r}")


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment EN3 (network operators).")
    parser.add_argument("--config", type=str, default=None, help="path to an EN3 config JSON")
    parser.add_argument("--out", type=str, default="runs/en3/records.jsonl")
    args = parser.parse_args()
    config = EN3Config.from_json_file(args.config) if args.config else EN3Config()
    records = run_en3(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
