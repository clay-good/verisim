"""Experiment CU21 -- the unified target: one model-free rule across all three worlds (H114).

The targeting arc shipped four hand-built defenses (CU10/CU11 network exfil, CU16 host corruption,
CU17 network segmentation, CU18 distributed staleness), each of which looked bespoke. CU21 proves
they are ONE rule: "consult the oracle iff the action is on the danger's model-free surface ``D``,"
where the arc's whole headline -- safe, cheap, un-gameable -- follows from one property, *coverage*
(``D.realizes => target``). The un-gameability is then a theorem (a covering, model-free target
blocks every adversarial placement, independent of the model), and the CU17/CU18 boundary is one
mechanism (a target that breaks coverage leaks exactly the danger it fails to cover).

This runs the unified schedule on all four arms under the worst-case omitter (the CU16/CU17/CU18
substrate; the per-world trained arms CU5-net/CU8/CU19/CU20 already closed the rigor gap). For each
world it reports the covering target (safe, cheap, un-gameable), the model self-targeting failure,
the gameable uniform knee, the perfect-model control (self-governs), and -- where a shortcut from
another world exists -- the non-covering shortcut that leaks. Torch-free; runs in under a minute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.segmentation_targeting import CU17Config
from verisim.acd.targeted_verification import CU10Config
from verisim.acd.unified_targeting import (
    arm_verdict,
    cu21_verdict,
    dist_arm,
    host_arm,
    net_flow_arm,
    net_reach_arm,
    run_cu21,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU21 -- the unified target (H114).")
    parser.add_argument("--out", type=str, default="figures/cu21_unified_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rhos: tuple[float, ...]
    if args.smoke:
        from verisim.acd.dist_targeting import CU18Config
        from verisim.acd.host_targeting import CU16Config

        arms = [
            net_flow_arm(CU10Config.smoke()), host_arm(CU16Config.smoke()),
            dist_arm(CU18Config.smoke()), net_reach_arm(CU17Config.smoke()),
        ]
        rhos = (0.0, 0.5, 1.0)
    else:
        arms = [net_flow_arm(), host_arm(), dist_arm(), net_reach_arm()]
        rhos = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)

    result = run_cu21(arms, rhos)
    verdict = cu21_verdict(result)

    print("\nCU21 / H114 -- the unified target (worst-case omitter; one rule across all worlds):")
    for arm in result.arms:
        v = arm_verdict(arm)
        print(f"\n  [{arm.world_name}]  danger = {arm.danger_name}  ({arm.n_scenarios} deps)")
        print(f"    target ({arm.target_name}): random {v['target_random_breach']:.3f}  "
              f"adversarial {v['target_adversarial_breach']:.3f}  "
              f"{v['target_calls']:.2f} calls  "
              f"= {v['target_call_saving']:.1f}x cheaper than full oracle "
              f"({v['full_oracle_calls']:.1f})  covers={v['target_covers']}")
        print(f"    model self-targeting: random {v['model_random_breach']:.3f} "
              f"(fails={v['model_self_targeting_fails']})   "
              f"perfect model self-governs: {v['oracle_self_governs']} "
              f"(breach {v['oracle_free_breach']:.3f})")
        print(f"    uniform knee ρ={v['uniform_knee_rho']:g}: random "
              f"{v['uniform_knee_random_breach']:.3f} -> adversarial "
              f"{v['uniform_knee_adversarial_breach']:.3f} (gameable={v['uniform_is_gameable']})")
        if "shortcut_leaks" in v:
            print(f"    shortcut ({arm.shortcut_name}): random {v['shortcut_random_breach']:.3f}  "
                  f"adversarial {v['shortcut_adversarial_breach']:.3f}  "
                  f"covers={v['shortcut_covers']}  leaks={v['shortcut_leaks']}")

    print("\n  UNIFIED VERDICT (H114):")
    print(f"    covering target safe in every world:        "
          f"{verdict['unified_target_safe_everywhere']}")
    print(f"    covering target un-gameable in every world: "
          f"{verdict['unified_target_ungameable_everywhere']}")
    print(f"    covering target cheaper in every world:     "
          f"{verdict['unified_target_cheaper_everywhere']}")
    print(f"    coverage holds for every target:            "
          f"{verdict['unified_target_covers_everywhere']}")
    print(f"    non-covering shortcuts leak everywhere:     "
          f"{verdict['shortcuts_leak_everywhere']}")
    print(f"    shortcuts break coverage everywhere:        "
          f"{verdict['shortcuts_break_coverage_everywhere']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu21 import plot_cu21

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu21(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
