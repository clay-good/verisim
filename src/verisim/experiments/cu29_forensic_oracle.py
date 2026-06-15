"""Experiment CU29 -- the forensic oracle: the posterior dual of the targeting arc (SPEC-22, H122).

The targeting arc (CU10-CU28) is a-priori and preventive. CU29 turns the same exact oracle around to
the forensic question after an incident: given a breached trace, which step caused it, and what was
the root cause? Two findings, on all four unified worlds (net exfil / host / dist / segmentation):

  1. THE HEADLINE -- the exact oracle attributes every breach (localization 1.000); the omitting
     world model is blind (localization 0.000, it cannot even tell an incident happened). And this
     is not a strawman: the *real* trained network ``M_theta`` (torch-gated, frozen flagship) has
     forensic localization 0.000 too -- it omits the very step that breached (CU8: recall 0.02).
     "You can't ask the omitter where it breached."

  2. THE ROOT CAUSE -- the realizing step is not always the cause. The oracle is an exact SCM
     (SPEC-17), so a counterfactual ``do(remove an earlier action)`` finds the earliest averting
     intervention. In the genesis-separated worlds (host ``open``->``write``, dist
     ``put``->stale-``get``) it precedes the breach by several steps in a majority of incidents; the
     four genesis-grammar flavors reappear as root-cause structures, read backward.

Torch-free core; the real-``M_theta`` forensic point is torch-gated, reusing the frozen checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.closed_loop_net import _new_flows
from verisim.acd.forensic_oracle import (
    CU29Config,
    NetFlowModelDefender,
    cu29_verdict,
    detection_rate,
    localization_accuracy,
    run_cu29,
    write_csv,
)
from verisim.acd.targeted_verification import CU10Config
from verisim.acd.unified_targeting import State, net_flow_arm
from verisim.netdelta.apply import apply


def _real_model_forensic(checkpoint: str, config: CU29Config) -> tuple[float, float] | None:
    """The torch-gated real-``M_theta`` forensic point on the network arm: (localization, detect).

    Grounds the worst-case omitter against the real trained model -- if the real model is blind too,
    the omitter is not a strawman. Reuses the frozen flagship (no retrain).
    """
    if not Path(checkpoint, "model.pt").exists():
        return None
    from verisim.experiments.flagship import load_checkpoint

    model = load_checkpoint(checkpoint).world_model

    def predict_new_flows(m: object, s: State, a: object, jewels: frozenset[str]) -> bool:
        pred = apply(s, m.predict_delta(s, a))  # type: ignore[attr-defined]
        return bool(_new_flows(s, pred, jewels))

    arm = net_flow_arm(
        CU10Config(horizon=config.horizon, n_seeds=config.n_seeds, max_episodes=config.max_episodes)
    )
    defender = NetFlowModelDefender(model, frozenset(("h0", "h4")), predict_new_flows)
    return localization_accuracy(arm, defender), detection_rate(arm, defender)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU29 -- the forensic oracle (SPEC-22 H122).")
    parser.add_argument("--out", type=str, default="figures/cu29_forensic_oracle.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--checkpoint", type=str, default="runs/flagship/net-l",
        help="reuse a frozen flagship checkpoint for the real-M_theta forensic point (no retrain).",
    )
    parser.add_argument("--no-model", action="store_true", help="skip the torch-gated real model.")
    args = parser.parse_args()

    config = CU29Config.smoke() if args.smoke else CU29Config()
    result = run_cu29(config)
    verdict = cu29_verdict(result)

    print("CU29 / H122 -- the forensic oracle (the posterior dual of the targeting arc):")
    print(f"  {'world':<24} {'breach':>6} {'oracle':>7} {'model':>7} {'detect':>7} "
          f"{'covers':>7} {'lag':>6} {'prec':>6}")
    for a in result.arms:
        print(f"  {a.world_name:<24} {a.n_breached:>6} "
              f"{a.oracle_localization_accuracy:>7.3f} {a.omitter_localization_accuracy:>7.3f} "
              f"{a.omitter_detection_rate:>7.3f} {a.forensic_surface_is_covering!s:>7} "
              f"{a.mean_root_cause_lag:>6.2f} {a.fraction_cause_precedes_breach:>6.2f}")
    print(f"  oracle attributes every breach: {verdict['oracle_attributes_every_breach']}")
    print(f"  model forensically blind: {verdict['model_is_forensically_blind']}; "
          f"cannot detect incident: {verdict['model_cannot_detect_incident']}")
    print(f"  forensic<->prevention surfaces converge: "
          f"{verdict['forensic_and_prevention_surfaces_converge']}")
    print(f"  upstream cause dominates the genesis-separated worlds: "
          f"{verdict['upstream_cause_dominates_separated_worlds']}")

    real: tuple[float, float] | None = None
    if not args.no_model:
        print(f"\n  grounding the omitter against the real network M_theta ({args.checkpoint})...")
        real = _real_model_forensic(args.checkpoint, config)
        if real is None:
            print("  [no checkpoint -- real-M_theta point skipped]")
        else:
            print(f"  real M_theta forensic on the network arm: localization {real[0]:.3f} / "
                  f"detection {real[1]:.3f}  (CU8 omission, not a strawman)")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu29 import plot_cu29

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu29(result, verdict, plot_path, real_model=real)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
