"""Experiment EH3 -- correction/operator comparison at equal budget (SPEC-6 §8.3, HC7).

The host analogue of v0's E3 and the network EN3, and the experiment that cashes in the new host
axis (`π_w`, §8.2: *which subsystem's truth to buy*). Fixes the consultation policy and budget ``ρ``
and compares the §8.3 operators at *equal* ``ρ``:

  - **full-consultation** operators -- ``hard_reset`` / ``residual`` / ``projection`` -- which (as
    in v0/EN3) all snap the coupled state to the complete one-step truth, so their headline ``H_ε``
    is **identical** (the full-truth identity, with CIs); they differ only in the diagnostic they
    expose.
  - **per-subsystem** ``subsystem_filter`` -- a cheap probe corrects only one subsystem (proc / fd /
    round-robin), so it corrects *strictly less* than a full consult. Its ``H_ε`` is therefore
    **genuinely lower** at equal consultation count -- the v0 identity collapse **breaks** (§8.3).
    The honest read is the cost lens: a per-subsystem consult spends far fewer oracle-bits (recorded
    as ``oracle_bits``), so the operator that *earns* the most faithful horizon per oracle-bit --
    not per consult -- is the real question (§9.4). H13 found the composition **coupled** and
    dominated by proc/fd, so *targeting the weakest subsystem* (``subsystem_proc``) is the static
    heuristic the smart-``π_w`` work (HC7) is built to beat -- so it is an arm here.

Output: faithful horizon + the per-operator diagnostic + mean oracle-bits by operator. The figure
(``figures/plot_comparison.py --key operator``) and CSV come from these records only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.state import HostState
from verisim.hostloop import (
    HardReset,
    PartialHostOracle,
    Projection,
    Residual,
    RoundRobinSubsystem,
    SubsystemFilter,
    budget_for_rho,
    run_host_rollout,
)
from verisim.hostloop.subsystem import FixedSubsystem
from verisim.hostmetrics.record import HostRunRecord
from verisim.hostmodel import HostVocab, NeuralHostWorldModel
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.record import RunRecord, write_records

from .eh1 import EH1Config, eval_actions, train_model

# The compared operators: the three full-consult operators (which coincide on H_ε) plus the
# per-subsystem filters (which do not -- the no-identity-collapse result, §8.3).
_FULL_OPERATORS = ("hard_reset", "residual", "projection")
_SUBSYSTEM_OPERATORS = ("subsystem_rr", "subsystem_proc", "subsystem_fd")


@dataclass(frozen=True)
class EH3Config:
    name: str = "eh3-small"
    base: EH1Config = field(default_factory=EH1Config)
    rho: float = 0.3
    policy: str = "fixed"  # the policy held fixed across operators
    operators: tuple[str, ...] = (*_FULL_OPERATORS, *_SUBSYSTEM_OPERATORS)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH3Config:
        base = EH3Config()
        return EH3Config(
            name=d.get("name", base.name),
            base=EH1Config.from_dict(d.get("base", {})),
            rho=d.get("rho", base.rho),
            policy=d.get("policy", base.policy),
            operators=tuple(d.get("operators", base.operators)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH3Config:
        return EH3Config.from_dict(json.loads(Path(path).read_text()))


def run_eh3(
    config: EH3Config | None = None, *, oracle: HostOracle | None = None
) -> list[RunRecord]:
    """Train the model and compare the operators at the fixed policy + interior ``ρ``."""
    config = config or EH3Config()
    base = config.base
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host)
    model = train_model(base, vocab, oracle, host)
    world_model = NeuralHostWorldModel(model, vocab)
    partial = PartialHostOracle(oracle)

    records: list[RunRecord] = []
    for difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            actions = eval_actions(oracle, host, driver, seed, base.eval_steps)
            policy = fixed_interval_for_rho(config.rho)
            budget = budget_for_rho(config.rho, len(actions))
            for operator_name in config.operators:
                diagnostic, rollout = _run_operator(
                    operator_name, world_model, partial, actions, policy, budget,
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
    name: str, world_model: NeuralHostWorldModel, partial: PartialHostOracle,
    actions: list[HostAction], policy: Any, budget: int, *, epsilon: float, seed: int,
) -> tuple[float | None, HostRunRecord]:
    """Run one operator's rollout; return its mean per-correction diagnostic + the record."""
    s0 = HostState.initial()

    def _run(**kw: Any) -> HostRunRecord:
        return run_host_rollout(
            world_model, partial, s0, actions, policy, epsilon=epsilon, budget=budget, seed=seed,
            **kw,
        )

    if name == "hard_reset":
        return None, _run(operator=HardReset())
    if name == "residual":
        res = Residual()
        rollout = _run(operator=res)
        return (fmean(res.discrepancies) if res.discrepancies else 0.0), rollout
    if name == "projection":
        proj = Projection()
        rollout = _run(operator=proj)
        return (fmean(proj.repaired_fractions) if proj.repaired_fractions else 0.0), rollout
    if name in {"subsystem_rr", "subsystem_proc", "subsystem_fd"}:
        sub_policy = (
            RoundRobinSubsystem() if name == "subsystem_rr"
            else FixedSubsystem("proc") if name == "subsystem_proc"
            else FixedSubsystem("fd")
        )
        sop = SubsystemFilter()
        rollout = _run(subsystem_policy=sub_policy, subsystem_op=sop)
        return (fmean(sop.repaired_fractions) if sop.repaired_fractions else 0.0), rollout
    raise ValueError(f"unknown operator {name!r}")


def efficiency_by_operator(records: list[RunRecord]) -> dict[str, dict[str, float]]:
    """Mean faithful horizon, oracle-bits, and **horizon-per-oracle-bit** by operator (§9.4).

    The cost lens that makes the per-subsystem vs full comparison honest: a per-subsystem consult
    buys less horizon but at far fewer bits, so horizon-per-bit -- not horizon-per-consult -- is the
    real efficiency question. Computed at the smallest ε (the strictest horizon).
    """
    eps = min(r.epsilon for r in records) if records else 0.0
    by_op: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        if r.epsilon == eps:
            by_op.setdefault(str(r.config["operator"]), []).append(
                (float(r.faithful_horizon), float(r.config["oracle_bits"]))
            )
    out: dict[str, dict[str, float]] = {}
    for op, pairs in sorted(by_op.items()):
        mean_h = fmean(h for h, _ in pairs)
        mean_bits = fmean(b for _, b in pairs)
        out[op] = {
            "mean_h": mean_h,
            "mean_bits": mean_bits,
            "h_per_bit": (mean_h / mean_bits) if mean_bits else 0.0,
        }
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run EH3 (host correction-operator comparison).")
    parser.add_argument("--config", type=str, default=None, help="path to an EH3 config JSON")
    parser.add_argument("--out", type=str, default="runs/eh3/records.jsonl")
    args = parser.parse_args()
    config = EH3Config.from_json_file(args.config) if args.config else EH3Config()
    records = run_eh3(config)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")
    for op, stats in efficiency_by_operator(records).items():
        print(f"  {op:16s} H_ε={stats['mean_h']:.2f}  bits={stats['mean_bits']:.1f}  "
              f"H_ε/bit={stats['h_per_bit']:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
