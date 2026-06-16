"""Experiment CU39 -- the redundant verifier: defense in depth requires failure independence (H132).

CU38 tiled the CIA triad with verifier members held EXACT on their channel. CU35's premise is the
opposite -- a real deployed verifier drifts (faithful on its surface only with probability phi < 1).
CU39 asks what CU38's tiling theorem does when the members are imperfect, and recovers the oldest
principle in security with a coverage-theoretic statement: stack redundant monitors so a danger one
misses another catches -- but it works only if they fail INDEPENDENTLY. A single imperfect monitor
is adversarially gameable (CU35's cliff); HOMOGENEOUS redundancy (copies sharing a blind spot) flat
in stack height (the same scanner twice buys nothing); HETEROGENEOUS redundancy (independent blind
spots) drives adversarial breach to the oracle's 0 at a knee, because the members' faithful surfaces
tile the leg. Depth costs a compounding false-block tax (1 - psi**m). The CU38 tiling theorem
operating within one danger leg at the sub-action granularity. Torch-free (CU21 net/host arms + the
CU35 imperfect-verifier dial, salt-keyed for independent vs shared blind spots).
"""

from __future__ import annotations

import argparse

from verisim.acd.verifier_redundancy import (
    CU39Config,
    arm_verdict,
    cu39_verdict,
    run_cu39,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU39 -- the redundant verifier (defense in depth, H132)."
    )
    parser.add_argument("--out", type=str, default="figures/cu39_verifier_redundancy.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU39Config.smoke() if args.smoke else CU39Config()
    result = run_cu39(config)
    v = cu39_verdict(result)

    print("\nCU39 / H132 -- the redundant verifier (defense in depth requires independence):")
    for arm in result.arms:
        a = arm_verdict(arm, result.headline_phi)
        ph = a["headline_phi"]
        print(f"\n  {arm.world_name} (phi={ph}, psi={arm.psi}, n={arm.n_scenarios}):")
        print(f"    {'m':>3s}  {'homo adv':>9s}  {'hetero adv':>11s}  {'hetero false-blocks':>19s}")
        homo = arm.homogeneous[ph]  # type: ignore[index]
        hetero = arm.heterogeneous[ph]  # type: ignore[index]
        for ph_pt, pe in zip(homo, hetero, strict=True):
            print(f"    {pe.m:3d}  {ph_pt.adversarial_breach:9.3f}  {pe.adversarial_breach:11.3f}  "
                  f"{pe.mean_false_blocks:19.3f}")
        print(f"    single monitor gameable = {a['single_monitor_gameable']}  |  "
              f"homogeneous flat in m = {a['homogeneous_flat_in_m']}")
        print(f"    heterogeneous -> oracle (breach 0) at knee m = {a['defense_in_depth_knee']}  "
              f"|  depth costs utility = {a['depth_costs_utility']}")

    print(f"\n  single monitor gameable everywhere       = "
          f"{v['single_monitor_gameable_everywhere']}")
    print(f"  homogeneous redundancy flat everywhere   = {v['homogeneous_flat_everywhere']}")
    print(f"  heterogeneous reaches the oracle (all)   = "
          f"{v['heterogeneous_reaches_oracle_everywhere']}")
    print(f"  INDEPENDENCE is load-bearing everywhere  = "
          f"{v['independence_load_bearing_everywhere']}")
    print(f"  depth costs a compounding false-block tax = {v['depth_costs_utility_somewhere']}")

    out = write_csv(result, args.out)
    print(f"\n  wrote {out}")

    if args.plot:
        from figures.plot_cu39 import plot_cu39

        path = plot_cu39(result, args.plot)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
