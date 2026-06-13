"""Experiment CU6 -- the replanning agent (SPEC-22 §5, H98).

CU5 gave each goal one route. CU6 gives it several and lets the agent **replan** -- try another way
when the gate blocks the first. Replanning is the capability that makes an agent capable; this asks
what it costs. The result that matters for deploying capable agents: a *free* agent that replans is
**more capable AND more dangerous** (its retry loop searches its own gate's blind spots), while an
*oracle-grounded* replanner is capable and **safe** -- persistence becomes pure benefit.

Two findings (H98): (1) replanning lifts capability (it recovers the goals a one-shot agent
abandons) but **for a free agent that capability is danger** -- replanning amplifies the harm rate;
(2) **only a verified agent is both capable and safe** -- the oracle aborts every dangerous route,
so no number of retries can execute one, and ρ is the path from the capable-but-dangerous free
corner to the capable-and-safe oracle corner. CPU-only, torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.closed_loop_replan import (
    CU6Config,
    cells_for,
    cu6_verdict,
    run_cu6,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU6 -- the replanning agent (SPEC-22 H98).")
    parser.add_argument("--out", type=str, default="figures/cu6_closed_loop_replan.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU6Config.smoke() if args.smoke else CU6Config()
    result = run_cu6(config)
    verdict = cu6_verdict(result)

    print(f"CU6 / H98 -- the replanning agent (phi={result.phi}, {result.n_goals} goals, "
          f"{result.n_routes} routes/goal):")
    for agent in ("one_shot", "replanner"):
        print(f"  {agent}:")
        print(f"    {'rho':>5} {'success':>8} {'harm':>8} {'lost':>8}")
        for c in cells_for(result, agent):
            print(f"    {c.rho:>5.2f} {c.success_rate:>8.3f} {c.harm_rate:>8.3f} "
                  f"{c.lost_rate:>8.3f}")

    print(f"\n  replanning lifts capability (recovers abandoned goals): "
          f"{verdict['replanning_lifts_capability']}")
    print(f"  but for a free agent that capability is danger -- replanning amplifies harm: "
          f"{verdict['free_replanning_amplifies_harm']} "
          f"(+{verdict['amplification_at_free']:.3f} harm vs one-shot at ρ=0)")
    print(f"  the oracle makes persistence pure benefit (success≥0.99, harm=0): "
          f"{verdict['grounded_replanning_is_pure_benefit']}")
    print(f"  only a verified agent is both capable and safe: "
          f"{verdict['only_verified_is_capable_and_safe']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu6 import plot_cu6

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu6(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
