"""Experiment CU30 -- the remediation oracle: the recovery dual of the forensic oracle (H123).

CU29 *diagnosed* an incident (which step realized it, what determined it). CU30 closes the loop with
the defender's next *action* -- recovery: compute a remediation that undoes the breach, and prove it
on the exact oracle. On all four unified worlds (net exfil / host / dist / segmentation), five
remediation policies are scored on two oracle-certified axes -- *avert rate* (does the breach not
recur in the counterfactual re-run?) and *mean collateral* (how much benign mission work the fix
sacrificed):

  1. THE HEADLINE -- undoing the action that *realized* the breach does not undo the breach. The
     ``surgical`` undo (delete the realizing action) averts in net/dist but FAILS in the
     genesis-separated host world (~0.40) -- a redundant consumer just re-triggers the corruption.
     The robust fix removes the genesis (CU29's root cause), and the oracle's counterfactual is what
     finds it.

  2. THE WINNER -- only ``min_certified`` (the smallest averting set, searched against the oracle)
     averts *every* incident at minimal collateral, world by world; ``sledgehammer`` (disable the
     capability) averts but at 3-5x the collateral; the ``model`` fix is empty -- blind to the
     incident it never saw. This is not a strawman: the real network ``M_theta`` (torch-gated,
     frozen flagship) remediates ~0 of incidents (CU8 omission; CU29 localization 0.000).

Torch-free core; the real-``M_theta`` remediation point is torch-gated, reusing the frozen flagship.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.closed_loop_net import _new_flows
from verisim.acd.forensic_oracle import NetFlowModelDefender
from verisim.acd.remediation_oracle import (
    CU30Config,
    cu30_verdict,
    real_model_avert_rate,
    run_cu30,
    write_csv,
)
from verisim.acd.targeted_verification import CU10Config
from verisim.acd.unified_targeting import State, net_flow_arm
from verisim.netdelta.apply import apply


def _real_model_remediation(checkpoint: str, config: CU30Config) -> float | None:
    """The torch-gated real-``M_theta`` remediation point on the network arm: its avert rate.

    Grounds the empty omitter fix against the real trained model -- if the real model's remediation
    averts ~0, the omitter's empty fix is not a strawman. Reuses the frozen flagship; no retrain.
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
    return real_model_avert_rate(arm, defender)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU30 -- the remediation oracle (SPEC-22 H123).")
    parser.add_argument("--out", type=str, default="figures/cu30_remediation.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--checkpoint", type=str, default="runs/flagship/net-l",
        help="reuse a frozen flagship checkpoint for the real-M_theta remediation point.",
    )
    parser.add_argument("--no-model", action="store_true", help="skip the torch-gated real model.")
    args = parser.parse_args()

    config = CU30Config.smoke() if args.smoke else CU30Config()
    result = run_cu30(config)
    verdict = cu30_verdict(result)

    policies = ("model", "surgical", "root_cause", "min_certified", "sledgehammer")
    print("CU30 / H123 -- the remediation oracle (the recovery dual of the forensic oracle):")
    print(f"  {'world':<22} {'inc':>4} | " + " ".join(f"{p:>14}" for p in policies))
    print(f"  {'':<22} {'':>4} | " + " ".join(f"{'avert/collat':>14}" for _ in policies))
    for a in result.arms:
        cells = []
        for name in policies:
            p = a.by_name(name)
            cells.append(f"{p.avert_rate:.2f}/{p.mean_collateral:>4.1f}")
        print(f"  {a.world_name:<22} {a.n_incidents:>4} | " + " ".join(f"{c:>14}" for c in cells))
    print(f"  min_certified averts every incident: "
          f"{verdict['min_certified_averts_every_incident']} "
          f"(cheaper than sledgehammer: {verdict['min_certified_cheaper_than_sledgehammer']})")
    print(f"  surgical undo fails in a genesis-separated world: "
          f"{verdict['surgical_undo_fails_in_separated_world']}; "
          f"collateral is the redundancy tax: {verdict['collateral_is_redundancy_tax']}")
    print(f"  model remediation is empty (blind): {verdict['model_remediation_is_empty']}")

    real: float | None = None
    if not args.no_model:
        print(f"\n  grounding the empty omitter fix against the real network M_theta "
              f"({args.checkpoint})...")
        real = _real_model_remediation(args.checkpoint, config)
        if real is None:
            print("  [no checkpoint -- real-M_theta point skipped]")
        else:
            print(f"  real M_theta remediation avert rate on the network arm: {real:.3f}  "
                  f"(CU8 omission, not a strawman)")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu30 import plot_cu30

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu30(result, verdict, plot_path, real_model=real)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
