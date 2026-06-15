"""Experiment CU26 -- the low-and-slow danger: target the accumulator, not the action (H119).

Every danger the CU arc studied (CU10-CU25) is realized by a single action. CU26 is the fourth
genesis-grammar flavor and a third generative test: a **cumulative** danger no single action
realizes -- mass collection / data hoarding, the canonical low-and-slow exfiltration. The breach is
the AGGREGATE crossing a budget B (concurrent flows to >= B distinct hosts). The framework's
``covers`` invariant -- extended to **cumulative coverage** (the target must fire on every crossing
a multi-action adversary can stage, not just the benign trajectory's) -- predicts every candidate a
priori: the real-world magnitude / DLP heuristic (alarm on high-value targets) leaks to
low-and-slow; the grammar target (watch every sensitive flow) covers but overpays; the derived
accumulator-closure (watch the budget boundary) covers, is un-gameable, and is cheapest.
Torch-free; ~2s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.cumulative_targeting import (
    CU26Config,
    cu26_verdict,
    run_cu26,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU26 -- the low-and-slow danger (H119).")
    parser.add_argument("--out", type=str, default="figures/cu26_cumulative_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU26Config.smoke() if args.smoke else CU26Config()
    result = run_cu26(config)
    v = cu26_verdict(result)

    print("\nCU26 / H119 -- the low-and-slow danger (cumulative exfil, worst-case omitter):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}, budget B={result.budget}, "
          f"free random {v['free_random_breach']:.3f} / "
          f"adversarial {v['free_adversarial_breach']:.3f}\n")
    print("  candidate targets (covers = cumulative coverage, the a-priori prediction):")
    for c in result.candidates:
        print(f"    {c.label:46s}  covers={c.covers!s:5s}  "
              f"random {c.random_breach:.3f}  low-and-slow {c.adversarial_breach:.3f}  "
              f"{c.mean_calls:.2f} calls")
    print(f"\n  derived accumulator-closure: safe={v['closure_is_safe']}  "
          f"un-gameable={v['closure_is_ungameable']}  "
          f"{v['closure_call_saving_vs_grammar']:.1f}x cheaper than grammar, "
          f"{v['closure_call_saving_vs_full']:.1f}x cheaper than full oracle "
          f"({result.full_oracle.mean_calls:.0f} calls)")
    print(f"  magnitude / DLP heuristic leaks={v['magnitude_leaks']} "
          f"(low-and-slow breach {v['magnitude_adversarial_breach']:.3f} at "
          f"{v['magnitude_calls']:.2f} calls -- false security)")
    print(f"  uniform knee rho={v['uniform_knee_rho']:g}: low-and-slow "
          f"{v['uniform_knee_adversarial_breach']:.3f} (gameable={v['uniform_is_gameable']})  "
          f"model self-targeting fails={v['model_self_targeting_fails']}  "
          f"perfect model self-governs={v['oracle_self_governs']}")
    print("\n  cost law (closure boundary gets rarer as B rises; grammar surface unchanged):")
    for b in sorted(result.cost_by_budget):
        costs = result.cost_by_budget[b]
        ratio = costs["grammar"] / costs["closure"] if costs["closure"] > 0 else float("inf")
        print(f"    B={b}:  magnitude {costs['magnitude']:.2f}  grammar {costs['grammar']:.2f}  "
              f"closure {costs['closure']:.2f}  (ratio {ratio:.1f}x)")
    print(f"    closure cost falls with B={v['closure_cost_falls_with_budget']}  "
          f"ratio grows with B={v['cost_ratio_grows_with_budget']}")
    print(f"\n  GENERATIVE HEADLINE (H119): cumulative covers predicted every candidate; "
          f"confirmed = {v['framework_predicts_every_candidate']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu26 import plot_cu26

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu26(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
