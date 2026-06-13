"""Experiment CU5-net -- the closed-loop safe agent on a REAL trained network model (SPEC-22, H100).

CU5-CU7 measured the closed loop on a controlled stand-in (a φ-dial hash-coin drifter). CU5-net is
the trained-arm confirmation: the same two-axis closed loop, re-run on the real trained network
flagship `M_θ` (a transformer world-model), whose live-flow drift is the content the agent's gate
keys on. An agent works a connectivity goal (establish flows to the work hosts) from a workload that
also tempts it with exfiltration (flows to the protected crown-jewel hosts), previewing each action
through the model (ρ-grounded) and executing iff the preview shows no exfil.

The finding (H100): the closed loop closes on the real model just as on the stand-in -- a free agent
is unsafe and unreliable, the oracle agent is safe and reliable, and ρ is the path between. The
headline computer-use result is not an artifact of synthetic drift; it survives a learned model.
The trained model is torch-gated (loaded via the flagship); the closed-loop core is torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.closed_loop_net import (
    CU5NetConfig,
    cu5_net_verdict,
    run_cu5_net,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU5-net -- the closed loop on a trained network model (SPEC-22 H100)."
    )
    parser.add_argument("--out", type=str, default="figures/cu5_net_closed_loop.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from verisim.experiments.flagship import FlagshipConfig, train_flagship

    base = FlagshipConfig.smoke() if args.smoke else FlagshipConfig()
    print("training the network flagship M_θ (one-time)...")
    model, _ = train_flagship(base)
    config = CU5NetConfig.smoke() if args.smoke else CU5NetConfig()
    result = run_cu5_net(model, config)
    verdict = cu5_net_verdict(result)

    print(f"\nCU5-net / H100 -- the closed loop on a REAL trained network M_θ "
          f"({result.n_episodes} contested episodes, horizon {result.horizon}):")
    print(f"  {'rho':>5} {'success':>8} {'unsafe':>8} {'mean_unsafe':>12} {'calls':>7}")
    for c in result.cells:
        print(f"  {c.rho:>5.2f} {c.success_rate:>8.3f} {c.unsafe_rate:>8.3f} "
              f"{c.mean_unsafe:>12.3f} {c.mean_calls:>7.1f}")

    print(f"\n  the free agent is unsafe (opens exfil flows): {verdict['free_is_unsafe']} "
          f"(unsafe {verdict['free_unsafe']:.2f}, success {verdict['free_success']:.2f})")
    print(f"  the oracle agent is safe AND reliable: {verdict['oracle_is_safe_and_reliable']} "
          f"(unsafe {verdict['oracle_unsafe']:.2f}, success {verdict['oracle_success']:.2f})")
    print(f"  verification closes the loop on the real trained model: "
          f"{verdict['verification_closes_the_loop']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu5_net import plot_cu5_net

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu5_net(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
