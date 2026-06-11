"""UA1+UA2 -- learn-in-imagination and the grounding ablation (SPEC-20 §5, H73/H74).

The experiment SPEC-20 exists for. Train the *same* defender three ways -- in `E_oracle` (reality /
the expensive baseline), `E_grounded` (the cheap, oracle-corrected flagship model), and `E_free`
(the same model, uncorrected) -- and evaluate **all three in `E_oracle`** (reality). Two reads.

  - **UA1 / H73 (learn-in-imagination):** the `E_grounded`-trained defender reaches reality
    containment competitive with the `E_oracle`-trained one, at far lower training oracle cost --
    learning in the cheap faithful model transfers.
  - **UA2 / H74 (the money hypothesis -- grounding is load-bearing):** the `E_grounded`-trained
    defender transfers to reality *materially better* than the `E_free`-trained one at matched
    rollout budget. If `E_free` transfers as well, faithfulness does not convert into downstream
    usefulness in this world -- the deep bankable negative (SPEC-20 §7).

The training oracle-call cost is the honest cost axis: `E_oracle` spends one oracle call per step,
`E_grounded` spends ρ per step, `E_free` spends none. Reuses the UA0 env + the UA1 REINFORCE engine
verbatim; only the backend differs across arms (SPEC-20 §3). CPU-local; CI runs the smoke instance
with a perfect (oracle-backed) stand-in model that exercises the orchestration without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verisim.acd.containment import (
    ContainmentConfig,
    ContainmentEnv,
    FreeBackend,
    GroundedBackend,
    OracleBackend,
)
from verisim.acd.train import TrainConfig, evaluate, reinforce
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

if TYPE_CHECKING:
    from verisim.netloop.model import NetModel


@dataclass(frozen=True)
class UATransferConfig:
    """The UA1/UA2 instance: the task, the REINFORCE schedule, the grounding budget, eval seeds."""

    containment: ContainmentConfig = field(default_factory=ContainmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rho: float = 0.5  # E_grounded training-time consultation budget
    eval_seeds: tuple[int, ...] = tuple(range(500, 520))

    @staticmethod
    def smoke() -> UATransferConfig:
        return UATransferConfig(
            containment=ContainmentConfig.smoke(), train=TrainConfig.smoke(),
            eval_seeds=(500, 501, 502, 503),
        )


@dataclass(frozen=True)
class ArmResult:
    """One training backend's outcome: reality containment + the training oracle-call cost."""

    backend: str  # "oracle" | "grounded" | "free"
    reality_containment: float
    train_oracle_calls: int

    def csv_row(self) -> str:
        return f"{self.backend},{self.reality_containment:.6f},{self.train_oracle_calls}"


CSV_HEADER = "backend,reality_containment,train_oracle_calls"


def _train_oracle_calls(backend: str, config: UATransferConfig) -> int:
    """Nominal oracle calls spent *training* in a backend (the honest cost axis, SPEC-20 §5)."""
    steps = config.train.episodes * config.containment.episode_steps
    if backend == "oracle":
        return steps
    if backend == "grounded":
        return round(steps * config.rho)
    return 0  # free spends no oracle calls during training


def run_ua_transfer(
    model: NetModel, config: UATransferConfig | None = None, *, oracle: NetOracle | None = None,
) -> dict[str, ArmResult]:
    """Train the defender in each backend; evaluate every arm in `E_oracle` (reality)."""
    config = config or UATransferConfig()
    oracle = oracle or ReferenceNetworkOracle()

    def reality_env() -> ContainmentEnv:
        return ContainmentEnv(config.containment, OracleBackend(oracle))

    backends = {
        "oracle": lambda: ContainmentEnv(config.containment, OracleBackend(oracle)),
        "grounded": lambda: ContainmentEnv(
            config.containment, GroundedBackend(model=model, oracle=oracle, rho=config.rho)
        ),
        "free": lambda: ContainmentEnv(config.containment, FreeBackend(model=model)),
    }

    results: dict[str, ArmResult] = {}
    for name, make_train_env in backends.items():
        policy = reinforce(make_train_env, config.train)
        reality = evaluate(reality_env, policy, seeds=config.eval_seeds)  # ALWAYS tested in reality
        results[name] = ArmResult(name, reality, _train_oracle_calls(name, config))
    return results


def ua_verdict(
    results: dict[str, ArmResult], *, h73_frac: float = 0.9, h73_cost_ratio: float = 5.0,
) -> dict[str, Any]:
    """UA1/H73 (imagination transfers cheaply) + UA2/H74 (grounding is load-bearing)."""
    oracle = results["oracle"]
    grounded = results["grounded"]
    free = results["free"]
    cost_ratio = (
        oracle.train_oracle_calls / grounded.train_oracle_calls
        if grounded.train_oracle_calls > 0
        else float("inf")
    )
    return {
        "oracle_reality": oracle.reality_containment,
        "grounded_reality": grounded.reality_containment,
        "free_reality": free.reality_containment,
        "cost_ratio_oracle_over_grounded": cost_ratio,
        # H73: grounded reaches ≥ h73_frac of oracle's reality success at ≥ h73_cost_ratio cheaper
        "h73_supported": (
            grounded.reality_containment >= h73_frac * oracle.reality_containment
            and cost_ratio >= h73_cost_ratio
        ),
        # H74 (the money hypothesis): grounded transfers materially better than free
        "grounding_advantage": grounded.reality_containment - free.reality_containment,
        "h74_supported": grounded.reality_containment > free.reality_containment,
    }


def transfer_gap(
    model: NetModel, config: UATransferConfig | None = None, *, oracle: NetOracle | None = None,
) -> dict[str, Any]:
    """UA3/H75: the sim-to-emulation *policy* gap -- in-model success vs reality success.

    Train the defender in `E_grounded`, then evaluate it BOTH in `E_grounded` (the model it trained
    in) and in `E_oracle` (reality). The gap Δ = in_model − reality is the policy-level analogue of
    SPEC-18's `ΔH=0` faithfulness transfer (SPEC-20 §5): small ⇒ the faithful model is an honest
    proxy for reality; large ⇒ faithful one-step dynamics did not suffice for policy transfer.
    """
    config = config or UATransferConfig()
    oracle = oracle or ReferenceNetworkOracle()

    def grounded_env() -> ContainmentEnv:
        return ContainmentEnv(
            config.containment, GroundedBackend(model=model, oracle=oracle, rho=config.rho)
        )

    def reality_env() -> ContainmentEnv:
        return ContainmentEnv(config.containment, OracleBackend(oracle))

    policy = reinforce(grounded_env, config.train)
    in_model = evaluate(grounded_env, policy, seeds=config.eval_seeds)
    reality = evaluate(reality_env, policy, seeds=config.eval_seeds)
    return {
        "in_model_containment": in_model,
        "reality_containment": reality,
        "transfer_gap": in_model - reality,
        "abs_transfer_gap": abs(in_model - reality),
    }


def budget_sweep(
    model: NetModel, rhos: tuple[float, ...], config: UATransferConfig | None = None,
    *, oracle: NetOracle | None = None,
) -> list[dict[str, Any]]:
    """UA4/H76: the grounding advantage (H74) vs the training-time consultation budget ρ.

    Re-runs the UA2 grounded-vs-free contrast at each ρ. H76 predicts the advantage is *monotone* in
    ρ -- more grounding during the agent's rollouts buys more transfer -- recovering `E_oracle` as
    ρ→1 and `E_free` as ρ→0. A flat-in-ρ advantage refutes H76 (SPEC-20 §4).
    """
    oracle = oracle or ReferenceNetworkOracle()
    rows: list[dict[str, Any]] = []
    for rho in rhos:
        from dataclasses import replace

        cfg = replace(config or UATransferConfig(), rho=rho)
        results = run_ua_transfer(model, cfg, oracle=oracle)
        v = ua_verdict(results)
        rows.append({"rho": rho, "grounding_advantage": v["grounding_advantage"]})
    return rows


def write_csv(results: dict[str, ArmResult], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [results[k].csv_row() for k in ("oracle", "grounded", "free") if k in results]
    out.write_text("\n".join([CSV_HEADER, *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="UA1/UA2 -- imagination + grounding (SPEC-20).")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/ua2_grounding_ablation.csv")
    parser.add_argument("--rho", type=float, default=None, help="E_grounded budget (H73/H76)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = UATransferConfig.smoke() if args.smoke else UATransferConfig()
    if args.rho is not None:
        from dataclasses import replace

        config = replace(config, rho=args.rho)
    if args.smoke:
        from verisim.netloop.model import NetOracleBackedModel

        model: NetModel = NetOracleBackedModel(ReferenceNetworkOracle())
    else:
        from verisim.experiments.flagship import load_checkpoint

        model = load_checkpoint(args.checkpoint).world_model

    results = run_ua_transfer(model, config)
    path = write_csv(results, args.out)
    print(f"wrote {len(results)} arms to {path}")
    for k in ("oracle", "grounded", "free"):
        r = results[k]
        print(f"  {k:9s} reality_containment={r.reality_containment:.3f} "
              f"train_oracle_calls={r.train_oracle_calls}")
    verdict = ua_verdict(results)
    print(f"H73 (imagination transfers): {'SUPPORTED' if verdict['h73_supported'] else 'no'}")
    print(f"H74 (grounding load-bearing): {'SUPPORTED' if verdict['h74_supported'] else 'no'}  "
          f"(advantage={verdict['grounding_advantage']:+.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()
