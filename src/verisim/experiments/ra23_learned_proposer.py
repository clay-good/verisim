"""Experiment RA23 -- the learned adversarial proposer: oracle-as-reward hole hunting (H155).

Runs the CEGIS loop with a learned (REINFORCE) proposer trained by the exact oracle's hole-verdict,
against a blind uniform proposer over the same operator space. Headline: the learned adversary finds
more coverage holes per oracle call, concentrates its proposals on the indirection class as the loop
covers the literal class, surfaces a residual class beyond RA22's grid (quote_splice), and never
breaks the soundness invariant. Optionally cross-checks the labels against a real /bin/sh (--bash).
Hermetic, torch-free, deterministic (seeded).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from verisim.realagent.learned_proposer import (
    LearnedProposer,
    RandomProposer,
    bash_cross_check,
    ra23_verdict,
    run_cegis,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA23 -- learned adversarial proposer (H155).")
    parser.add_argument("--out", type=str, default="figures/ra23_learned_proposer.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bash", action="store_true",
                        help="cross-check the operator-space labels against a real /bin/sh")
    args = parser.parse_args()

    learned = run_cegis(LearnedProposer(seed=args.seed), budget=args.budget)
    rnd = run_cegis(RandomProposer(seed=args.seed), budget=args.budget)
    v = ra23_verdict(learned, rnd)

    print("\nRA23 / H155 -- the learned adversarial proposer (oracle-as-reward)\n")
    print(f"  budget: {args.budget} oracle calls (= verifiable-reward queries)")
    print(f"    holes found   learned {learned.holes_found:4d}   blind {rnd.holes_found:4d}  "
          f"({learned.holes_found / max(rnd.holes_found, 1):.2f}x)")
    print(f"    calls to all residual classes   learned {learned.calls_to_all_classes}   "
          f"blind {rnd.calls_to_all_classes}")
    print(f"    learned indirection mass {v['learned_indirection_mass']:.3f}   "
          f"protected mass {learned.final_target_probs['protected']:.3f}")
    print(f"    residual classes: {', '.join(learned.distinct_residual_classes)}")
    print(f"    synthesized target: {learned.synthesized_prefixes}   "
          f"silent miss: {learned.silent_miss}")

    print("\n  verdict:")
    for k in ("learned_more_efficient", "learned_concentrates_on_indirection",
              "learned_attacks_protected", "found_quote_splice_residual",
              "soundness_holds_under_adversary"):
        print(f"    {k} = {v[k]}")

    if args.bash:
        mism = bash_cross_check(tempfile.gettempdir())
        print(f"\n  bash cross-check (RA2 anchor): {len(mism)} label/exec mismatches "
              f"{'(labels are exact)' if not mism else '(!)'}")
        for m in mism[:10]:
            print(f"    MISMATCH {m}")

    out = write_csv(learned, rnd, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra23 import plot_ra23

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra23(learned, rnd, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
