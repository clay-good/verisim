"""UA6 -- the task-taxonomy fork: when is faithfulness load-bearing? (SPEC-20 §7, H78).

UA2 found grounding null (H74) because the containment policy keys on drift-invariant features. UA6
is the pre-registered redirect: contrast the **drift-robust** task (UA0: local features, unlimited
isolation) with a **drift-sensitive** task (UA6: the marginal-cut structural feature + a tight
isolation budget, so the optimal policy depends on the multi-hop reachability the model drifts on),
and measure the grounding advantage on each. The contrast is the result:

  - if the advantage is ~0 on the robust task (reproducing H74) but **positive** on the sensitive
    task, faithfulness is load-bearing *exactly when the optimal policy depends on the dynamics the
    model drifts on* — the boundary the H74 negative pointed to (H78 SUPPORTED);
  - if grounding stays null even on the sensitive task, that is a deeper negative: faithfulness
    does not convert even when the policy must read multi-hop structure (H78 refuted), which
    would say the flat model's drift does not corrupt the marginal-cut signal enough to matter.

Reuses the UA0 env (now with `cut_budget`), the UA1 REINFORCE engine (now featurizer-pluggable), and
the UA2 three-backend contrast — only the featurizer + budget change between the two task variants.
CPU-local; CI runs the smoke instance with a perfect stand-in (advantage ~0 on both, the control).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verisim.acd.containment import (
    ContainmentConfig,
    ContainmentEnv,
    FreeBackend,
    GroundedBackend,
    OracleBackend,
)
from verisim.acd.policy import N_ACTION_FEATURES, action_features
from verisim.acd.structural import N_STRUCTURAL_FEATURES, structural_action_features
from verisim.acd.train import Featurizer, TrainConfig, evaluate, reinforce
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

if TYPE_CHECKING:
    from verisim.netloop.model import NetModel


@dataclass(frozen=True)
class TaxonomyConfig:
    """The UA6 contrast: the robust task, the sensitive task, the shared schedule + eval seeds."""

    robust: ContainmentConfig = field(default_factory=ContainmentConfig)
    sensitive: ContainmentConfig = field(
        default_factory=lambda: ContainmentConfig(cut_budget=2)
    )
    train: TrainConfig = field(default_factory=TrainConfig)
    rho: float = 0.5
    eval_seeds: tuple[int, ...] = tuple(range(500, 520))

    @staticmethod
    def smoke() -> TaxonomyConfig:
        return TaxonomyConfig(
            robust=ContainmentConfig.smoke(),
            sensitive=replace(ContainmentConfig.smoke(), cut_budget=1),
            train=TrainConfig.smoke(), eval_seeds=(500, 501, 502, 503),
        )


def _grounding_advantage(
    model: NetModel, containment: ContainmentConfig, train: TrainConfig, rho: float,
    eval_seeds: tuple[int, ...], featurizer: Featurizer, n_features: int, oracle: NetOracle,
) -> dict[str, float]:
    """Train in `E_grounded` and `E_free`; eval both in reality; return the advantage."""
    def reality() -> ContainmentEnv:
        return ContainmentEnv(containment, OracleBackend(oracle))

    def grounded() -> ContainmentEnv:
        return ContainmentEnv(containment, GroundedBackend(model=model, oracle=oracle, rho=rho))

    def free() -> ContainmentEnv:
        return ContainmentEnv(containment, FreeBackend(model=model))

    pol_g = reinforce(grounded, train, featurizer=featurizer, n_features=n_features)
    pol_f = reinforce(free, train, featurizer=featurizer, n_features=n_features)
    g = evaluate(reality, pol_g, seeds=eval_seeds, featurizer=featurizer)
    f = evaluate(reality, pol_f, seeds=eval_seeds, featurizer=featurizer)
    return {"grounded": g, "free": f, "advantage": g - f}


def run_taxonomy(
    model: NetModel, config: TaxonomyConfig | None = None, *, oracle: NetOracle | None = None,
) -> dict[str, dict[str, float]]:
    """Measure the grounding advantage on the drift-robust and drift-sensitive tasks (UA6)."""
    config = config or TaxonomyConfig()
    oracle = oracle or ReferenceNetworkOracle()
    robust = _grounding_advantage(
        model, config.robust, config.train, config.rho, config.eval_seeds,
        action_features, N_ACTION_FEATURES, oracle,
    )
    sensitive = _grounding_advantage(
        model, config.sensitive, config.train, config.rho, config.eval_seeds,
        structural_action_features, N_STRUCTURAL_FEATURES, oracle,
    )
    return {"robust": robust, "sensitive": sensitive}


def feature_drift_diagnostic(
    model: NetModel, containment: ContainmentConfig, *, oracle: NetOracle | None = None,
    n_episodes: int = 30, seed: int = 700,
) -> dict[str, float]:
    """How much does the model's free-running drift corrupt the marginal-cut feature?

    The load-bearing precondition for H78: the structural feature must actually *differ* between the
    drifted (`E_free`) state and the true state, or there is nothing for grounding to fix. This
    free-runs the model alongside the oracle and reports the fraction of (step, host) marginal-cut
    values that *agree* — a high agreement means the feature is drift-invariant in practice (the
    model's drift is connectivity-structure-preserving), which *explains* an H78 null without
    appealing to "faithfulness doesn't matter".
    """
    import random

    from verisim.acd.containment import seed_topology
    from verisim.acd.structural import marginal_cut
    from verisim.netdata import NetDriver
    from verisim.netdelta.apply import apply

    oracle = oracle or ReferenceNetworkOracle()
    net_cfg = containment.net()
    agree = 0
    total = 0
    for ep in range(n_episodes):
        true_s, comp = seed_topology(containment, random.Random(seed + ep))
        free_s = true_s.copy()
        drv = NetDriver(name="weighted", config=net_cfg, rng=random.Random(seed + 1000 + ep))
        for _ in range(containment.episode_steps):
            action = drv.sample(true_s)
            for host in [h for h in true_s.hosts if true_s.hosts[h].up]:
                total += 1
                if marginal_cut(true_s, comp, host) == marginal_cut(free_s, comp, host):
                    agree += 1
            true_s = oracle.step(true_s, action).state
            free_s = apply(free_s, model.predict_delta(free_s, action))
    return {
        "marginal_cut_agreement": (agree / total) if total else 1.0,
        "n_compared": float(total),
    }


def h78_verdict(results: dict[str, dict[str, float]]) -> dict[str, Any]:
    """H78: grounding is load-bearing on the drift-sensitive task but not the drift-robust one."""
    robust_adv = results["robust"]["advantage"]
    sensitive_adv = results["sensitive"]["advantage"]
    return {
        "robust_advantage": robust_adv,
        "sensitive_advantage": sensitive_adv,
        "delta": sensitive_adv - robust_adv,
        # H78 supported = grounding helps materially MORE on the sensitive task (the boundary)
        "h78_supported": sensitive_adv > robust_adv and sensitive_adv > 0.0,
    }


def write_csv(results: dict[str, dict[str, float]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{task},{r['grounded']:.6f},{r['free']:.6f},{r['advantage']:.6f}"
        for task, r in results.items()
    ]
    out.write_text("\n".join(["task,grounded,free,advantage", *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="UA6 -- the task-taxonomy fork (SPEC-20, H78).")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/ua6_taxonomy.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = TaxonomyConfig.smoke() if args.smoke else TaxonomyConfig()
    if args.smoke:
        from verisim.netloop.model import NetOracleBackedModel

        model: NetModel = NetOracleBackedModel(ReferenceNetworkOracle())
    else:
        from verisim.experiments.flagship import load_checkpoint

        model = load_checkpoint(args.checkpoint).world_model

    results = run_taxonomy(model, config)
    write_csv(results, args.out)
    for task in ("robust", "sensitive"):
        r = results[task]
        print(f"  {task:9s} grounded={r['grounded']:.3f} free={r['free']:.3f} "
              f"advantage={r['advantage']:+.3f}")
    v = h78_verdict(results)
    print(f"H78 (faithfulness load-bearing on the drift-sensitive task): "
          f"{'SUPPORTED' if v['h78_supported'] else 'no'}  (Δ advantage = {v['delta']:+.3f})")
    diag = feature_drift_diagnostic(model, config.sensitive)
    print(f"diagnostic: marginal-cut free-vs-true agreement = {diag['marginal_cut_agreement']:.3f} "
          f"(if ~1.0, the feature is drift-invariant -> the task is not actually drift-sensitive)")


if __name__ == "__main__":  # pragma: no cover
    main()
