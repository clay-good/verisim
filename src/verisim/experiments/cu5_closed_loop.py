"""Experiment CU5 -- the closed-loop safe agent (SPEC-22 §5, H97).

CU1-CU4 scored the gate's verdict on a *fixed plan pool*. CU5 closes the loop: an agent works a task
(a set of benign files it must write) from a candidate queue salted with dangerous traps, previewing
each action through its world model and executing the safe ones / aborting the rest -- and is scored
on **both axes at once**: did it finish the job (``success_rate``), and did it ever do the
irreversible bad thing (``unsafe_rate``)?

Two findings (H97): (1) a *free* agent is in the bad corner -- **unsafe and unreliable** -- while
the oracle agent is safe and reliable, and ρ is the path between them; (2) **where you spend the
budget matters** -- a stakes-aware schedule (consult the actions the model is most uncertain about)
reaches the safe-and-reliable corner at a fraction of a uniform schedule's budget (the knee).
CPU-only, torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.closed_loop_agent import (
    CU5Config,
    cu5_verdict,
    run_cu5,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU5 -- the closed-loop safe agent (SPEC-22 H97).")
    parser.add_argument("--out", type=str, default="figures/cu5_closed_loop.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU5Config.smoke() if args.smoke else CU5Config()
    result = run_cu5(config)
    verdict = cu5_verdict(result)

    print(f"CU5 / H97 -- the closed-loop safe agent (phi={result.phi}, "
          f"{result.n_episodes} episodes, {result.n_actions} actions/episode):")
    for schedule in ("uniform", "prioritized"):
        print(f"  {schedule} consultation:")
        print(f"    {'rho':>5} {'success':>8} {'unsafe':>8} {'mean_unsafe':>12} {'mean_lost':>10}")
        for c in result.cells:
            if c.schedule == schedule:
                print(f"    {c.rho:>5.2f} {c.success_rate:>8.3f} {c.unsafe_rate:>8.3f} "
                      f"{c.mean_unsafe:>12.3f} {c.mean_lost:>10.3f}")

    print(f"\n  the free agent is unsafe AND unreliable: {verdict['free_unsafe_and_unreliable']} "
          f"(success {verdict['free_success']:.2f}, unsafe-episode {verdict['free_unsafe']:.2f})")
    print(f"  the oracle agent is safe AND reliable: {verdict['oracle_safe_and_reliable']}")
    print(f"  stakes-aware consultation buys the knee: {verdict['stakes_aware_buys_the_knee']} "
          f"(safe-and-reliable at ρ={verdict['prioritized_knee_rho']:.2f} vs uniform's "
          f"ρ={verdict['uniform_knee_rho']:.2f})")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu5 import plot_cu5

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu5(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
