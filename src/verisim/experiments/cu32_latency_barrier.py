"""Experiment CU32 -- the verification-latency barrier: the THROUGHPUT cost of safety (H125).

The CU arc prices verification in oracle CALLS; CU32 prices it in WALL-CLOCK. A real verifier has a
latency ``L`` (a sandbox/kernel replay, a SOC/SOAR approval gate). The obvious way to hide it is
to PIPELINE -- commit the action speculatively, reconcile the verdict ``L`` steps later -- but that
is a trap on the IRREVERSIBLE slice (the send has left before the verdict confirms it). The latency
theorem: safety on the irreversible slice needs a SYNCHRONOUS BARRIER (stall ``L``), and the routing
CU27 derived from reversibility -- stall the irreversible, pipeline the reversible -- is also the
routing that minimizes throughput. Safety costs ``L`` x the irreversible danger rate; you can defer
verification you can undo, you must stall for verification you can't. Torch-free; ~1s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.latency_barrier import (
    CU32Config,
    cu32_verdict,
    run_cu32,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU32 -- the verification-latency barrier (H125).")
    parser.add_argument("--out", type=str, default="figures/cu32_latency_barrier.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU32Config.smoke() if args.smoke else CU32Config()
    result = run_cu32(config)
    v = cu32_verdict(result)

    print("\nCU32 / H125 -- the verification-latency barrier (the throughput cost of safety):")
    print(f"  {result.n_reversible} reversible (exposure) + {result.n_irreversible} irreversible "
          f"(exfil) deployments, horizon {result.horizon}, reference latency L={result.latency}\n")
    print(f"  {'policy':14s} {'class':13s} {'rand':>6s} {'adv':>6s} {'consult':>8s} "
          f"{'stalls':>7s} {'tput@L':>7s}")
    for c in result.cells:
        print(f"  {c.policy:14s} {c.danger_class:13s} {c.random_breach:6.3f} "
              f"{c.adversarial_breach:6.3f} {c.mean_consults:8.2f} {c.mean_stalls:7.2f} "
              f"{c.throughput:7.3f}")

    print("\n  THE LATENCY-HIDING TRAP: pipeline_all is safe on the reversible class "
          f"(adv breach {result.cell('pipeline_all', 'reversible').adversarial_breach:.3f}) but "
          f"FAILS on irreversible (adv breach {v['pipeline_irreversible_adv_breach']:.3f}) -- the "
          "send leaves before the verdict returns.")
    print(f"  ROUTED is safe everywhere={v['routed_safe_everywhere']} and never stalls the "
          f"reversible class (mean stalls {result.cell('routed', 'reversible').mean_stalls:.2f}); "
          f"on a 50/50 mix at L={result.latency} throughput {v['routed_throughput_mixed']:.3f} = "
          f"{v['throughput_saving_vs_barrier']:.2f}x barrier_all's "
          f"({v['barrier_throughput_mixed']:.3f}).")

    print("\n  SAFE-THROUGHPUT vs LATENCY (50/50 mix; pipeline is fast-but-unsafe):")
    print(f"    {'L':>4s} {'pipeline':>9s} {'barrier':>8s} {'routed':>7s}")
    for p in result.latency_curve:
        print(f"    {p.latency:4d} {p.pipeline_all_throughput:9.3f} "
              f"{p.barrier_all_throughput:8.3f} {p.routed_throughput:7.3f}")
    print(f"    routed beats barrier for L>0={v['routed_beats_barrier_for_latency']}  "
          f"routed decays with L={v['routed_throughput_decays_with_latency']}")

    print(f"\n  THE MIX LAW at L={result.latency} (safety is FREE in a reversible world):")
    print(f"    {'f_irr':>6s} {'routed':>7s} {'barrier':>8s} {'pipeline':>9s} {'pipe_breach':>12s}")
    for q in result.fraction_law:
        print(f"    {q.irreversible_fraction:6.2f} {q.routed_throughput:7.3f} "
              f"{q.barrier_all_throughput:8.3f} {q.pipeline_all_throughput:9.3f} "
              f"{q.pipeline_all_breach:12.3f}")
    print(f"    routed free when fully reversible={v['routed_free_when_fully_reversible']}  "
          f"pipeline breach grows with f={v['pipeline_breach_grows_with_irreversibility']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu32 import plot_cu32

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu32(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
