"""EH5-heads -- trained per-subsystem heads vs bucketed decode entropy (SPEC-6 §8.2, HC7).

EH5 measured the smart which-subsystem policy ``π_w`` using the factored arm's **bucketed decode
entropy**: each decoded token's masked-distribution entropy is attributed to the subsystem of the
op being emitted (§5.4). That signal is *post-hoc* (it reads the ambiguity of a grammar-constrained
decode) and *sparse* (a subsystem whose ops do not appear in this step's delta gets entropy 0, so it
is invisible to ``π_w`` even if the model is quietly wrong about it). The open HC7 work was the
calibrated alternative: a **trained per-subsystem head** that predicts, per subsystem, the decoder's
*own* per-subsystem error -- a dense, learned signal regressed against the realized teacher-forced
loss the free oracle supplies (§9.4).

EH5-heads compares the two π_w signals **confound-free**: it trains a *single* heads-enabled
factored arm, which exposes *both* signals on the *identical* proposer, and runs the
``UncertaintySubsystem`` policy twice -- once reading the trained head (``uncertainty_heads``),
once reading the bucketed entropy
(``uncertainty_entropy``) -- alongside the EH5 baselines (``round_robin``, ``fixed_fd``). All four
share the same trained model, so any difference among the uncertainty arms is the *signal*, not the
proposer, and the difference vs the baselines is the *policy*.

Reported per policy: faithful horizon ``H_ε``, oracle-bits, and horizon-per-oracle-bit (§9.4). The
question is whether the calibrated head localizes the leaking subsystem better than the entropy
bucket. Whatever it shows is a datum: a smoke-scale instance of the apparatus, not a tuned run.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.delta import HostDelta
from verisim.host.delta import apply as apply_host_delta
from verisim.host.state import HostState
from verisim.hostdata import HostDriver
from verisim.hostloop import (
    FixedSubsystem,
    PartialHostOracle,
    RoundRobinSubsystem,
    SubsystemFilter,
    UncertaintySubsystem,
    budget_for_rho,
    run_host_rollout,
)
from verisim.hostloop.subsystem import SubsystemPolicy
from verisim.hostmetrics.divergence import SUBSYSTEMS, divergence_by_subsystem
from verisim.hostmodel import HostVocab
from verisim.hostmodel.graph_model import GraphHostWorldModel
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.loop.policy import fixed_interval_for_rho
from verisim.metrics.calibration import pearson, spearman
from verisim.metrics.record import RunRecord, write_records

from .eh1 import EH1Config, eval_actions
from .eh5 import efficiency_by_policy

_POLICIES = ("fixed_fd", "round_robin", "uncertainty_entropy", "uncertainty_heads")


class _EntropySignalView:
    """Expose a heads-arm's **bucketed-entropy** π_w signal through the loop protocol.

    Delegates the proposal to the underlying arm (byte-identical predictions) but, where the runner
    reads per-subsystem uncertainty, reports the *entropy* signal instead of the trained head. This
    is what makes the EH5-heads comparison confound-free: ``uncertainty_entropy`` and
    ``uncertainty_heads`` run the *same* trained model, differing only in the signal ``π_w`` reads.
    """

    def __init__(self, model: GraphHostWorldModel) -> None:
        self.model = model

    def predict_delta(self, state: HostState, action: HostAction) -> HostDelta:
        return self.model.predict_delta(state, action)

    def predict_delta_with_subsystem_uncertainty(
        self, state: HostState, action: HostAction
    ) -> tuple[HostDelta, dict[str, float]]:
        return self.model.predict_delta_with_subsystem_entropy(state, action)


@dataclass(frozen=True)
class EH5HeadsConfig:
    name: str = "eh5-heads-small"
    base: EH1Config = field(default_factory=EH1Config)
    rho: float = 0.3
    policies: tuple[str, ...] = _POLICIES
    max_pid: int = 64
    graph_iters: int = 800
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_batch: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH5HeadsConfig:
        b = EH5HeadsConfig()
        return EH5HeadsConfig(
            name=d.get("name", b.name),
            base=EH1Config.from_dict(d.get("base", {})),
            rho=d.get("rho", b.rho),
            policies=tuple(d.get("policies", b.policies)),
            max_pid=d.get("max_pid", b.max_pid),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_batch=d.get("graph_batch", b.graph_batch),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH5HeadsConfig:
        return EH5HeadsConfig.from_dict(json.loads(Path(path).read_text()))


def _make_policy(name: str) -> SubsystemPolicy:
    if name == "fixed_fd":
        return FixedSubsystem("fd")
    if name == "round_robin":
        return RoundRobinSubsystem()
    if name in ("uncertainty_entropy", "uncertainty_heads"):
        return UncertaintySubsystem()
    raise ValueError(f"unknown π_w policy {name!r}")


def _train_heads_arm(
    config: EH5HeadsConfig, vocab: HostVocab, oracle: HostOracle
) -> GraphHostWorldModel:
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, train_host_graph_model

    base = config.base
    host = DEFAULT_HOST_CONFIG
    examples = build_host_graph_dataset(
        oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
        n_steps=base.train_steps_per_traj,
    )
    model = build_host_graph_model(
        vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
        mp_rounds=config.graph_mp_rounds, per_subsystem_heads=True, seed=base.model_seed,
    )
    train_host_graph_model(
        model, examples, steps=config.graph_iters, lr=base.lr,
        batch_size=config.graph_batch, seed=base.model_seed,
    )
    return model


def run_eh5_heads(
    config: EH5HeadsConfig | None = None,
    *,
    oracle: HostOracle | None = None,
    model: GraphHostWorldModel | None = None,
) -> list[RunRecord]:
    """Compare trained-head vs bucketed-entropy π_w at the fixed interior ``ρ`` (one heads arm).

    A pre-trained heads arm may be supplied as ``model`` (so the CLI can reuse it for the §9.4
    calibration diagnostic without retraining); otherwise one is trained from ``config``.
    """
    config = config or EH5HeadsConfig()
    base = config.base
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)
    heads_arm = model or _train_heads_arm(config, vocab, oracle)
    entropy_view = _EntropySignalView(heads_arm)
    partial = PartialHostOracle(oracle)

    records: list[RunRecord] = []
    for difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            actions = eval_actions(oracle, host, driver, seed, base.eval_steps)
            budget = budget_for_rho(config.rho, len(actions))
            for policy_name in config.policies:
                # The entropy arm reads the *entropy* signal via the view; every other policy reads
                # the heads arm directly (the static baselines ignore the signal). Same proposer.
                proposer = entropy_view if policy_name == "uncertainty_entropy" else heads_arm
                rollout = run_host_rollout(
                    proposer, partial, HostState.initial(), actions,
                    fixed_interval_for_rho(config.rho),
                    epsilon=base.epsilons[0], budget=budget,
                    subsystem_policy=_make_policy(policy_name), subsystem_op=SubsystemFilter(),
                    seed=seed,
                )
                for epsilon in base.epsilons:
                    records.append(
                        RunRecord(
                            config={
                                "experiment": config.name,
                                "policy": policy_name,
                                "difficulty": difficulty,
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


def signal_calibration(
    model: GraphHostWorldModel, oracle: HostOracle, config: EH5HeadsConfig
) -> dict[str, dict[str, float]]:
    """Does each π_w signal predict actual per-subsystem error? The §9.4 calibration claim.

    Over held-out eval rollouts, pair each step's per-subsystem signal (trained head and bucketed
    entropy, from the *same* heads arm) with the realized one-step per-subsystem divergence between
    the model's predicted next state and the oracle's truth. The signal that better predicts where
    the model is wrong is the better ``π_w`` driver -- the reason the heads exist (§8.2). Returns
    ``{"head": {pearson, spearman, n}, "entropy": {...}}`` pooled across subsystems and steps.
    """
    host = DEFAULT_HOST_CONFIG
    base = config.base
    head_sig: list[float] = []
    ent_sig: list[float] = []
    err: list[float] = []
    for driver in base.difficulties.values():
        for seed in base.eval_seeds:
            state = HostState.initial()
            drv = HostDriver(driver, host, random.Random(seed))
            for _ in range(base.eval_steps):
                action = drv.sample(state)
                delta, _, entropy_map, head_map = model._decode(
                    state, action, max_edits=64, max_new_tokens=4096
                )
                truth = oracle.step(state, action).state
                predicted = apply_host_delta(state, delta)
                per_sub_err = divergence_by_subsystem(truth, predicted)
                for sub in SUBSYSTEMS:
                    err.append(per_sub_err[sub])
                    ent_sig.append(entropy_map[sub])
                    head_sig.append(head_map[sub] if head_map is not None else 0.0)
                state = truth  # advance on the true next state (teacher forced)
    return {
        "head": {"pearson": pearson(head_sig, err), "spearman": spearman(head_sig, err),
                 "n": float(len(err))},
        "entropy": {"pearson": pearson(ent_sig, err), "spearman": spearman(ent_sig, err),
                    "n": float(len(err))},
    }


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run EH5-heads (trained per-subsystem head vs bucketed-entropy π_w)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/eh5_heads/records.jsonl")
    args = parser.parse_args()
    config = EH5HeadsConfig.from_json_file(args.config) if args.config else EH5HeadsConfig()
    oracle = ReferenceHostOracle()
    vocab = HostVocab(DEFAULT_HOST_CONFIG, max_pid=config.max_pid)
    model = _train_heads_arm(config, vocab, oracle)
    records = run_eh5_heads(config, oracle=oracle, model=model)
    path = write_records(records, args.out)
    print(f"wrote {len(records)} records to {path}")
    for p, stats in efficiency_by_policy(records).items():
        print(f"  {p:20s} H_ε={stats['mean_h']:.2f}  bits={stats['mean_bits']:.1f}  "
              f"H_ε/bit={stats['h_per_bit']:.4f}")
    cal = signal_calibration(model, oracle, config)
    print("calibration (signal vs per-subsystem error, §9.4):")
    for sig, stats in cal.items():
        print(f"  {sig:10s} pearson={stats['pearson']:+.3f}  spearman={stats['spearman']:+.3f}  "
              f"n={int(stats['n'])}")


if __name__ == "__main__":  # pragma: no cover
    main()
