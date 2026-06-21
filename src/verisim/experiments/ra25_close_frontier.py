"""Experiment RA25 -- closing the mapped frontier, and why it is a treadmill (H157).

Three measured parts, the exact oracle/resolver the only judge throughout:

  1. **The fold closes the frontier.** The RA24 neural adversary, re-run against the resolver with
     ``fold_filters`` on (printf escapes + pure ``echo|rev|cut`` pipelines folded), collapses: on
     the
     twelve-mechanism grammar those forms were the *entire* frontier, so its reward/oracle-call
     drops
     from the RA24 level to ~0, with the soundness invariant still intact (0 silent miss).
  2. **It is an enumeration treadmill.** A battery of realizing commands spelling /etc/shadow
     through
     different pure filters shows ``fold_filters`` covers only the ones it was taught
     (printf/rev/cut)
     and ABSTAINs the rest (tr/sed/base64/xxd) -- the unbounded tail. Folding by enumeration is the
     denylist's problem in miniature.
  3. **The post-commit diff is the real closer.** Every command here is a reversible file write, so
     its ABSTAIN is already safe (routed to the exact post-commit diff, CU27) with no folder at
     all --
     the fold is a usability win, not a safety one. The lone irreversible demo (an egress send) is
     the
     one place the folder is safety-load-bearing, which is exactly the honest open edge (RA15).

Optionally cross-checks the treadmill battery's realization against a real /bin/sh (``--bash``).
torch only for the adversary.
"""

from __future__ import annotations

import argparse
import tempfile

from verisim.realagent.frontier_close import (
    bash_cross_check,
    irreversible_demo,
    measure_frontier,
    ra25_verdict,
    treadmill_battery,
)
from verisim.realagent.neural_proposer import NeuralAdversary


def main() -> None:  # pragma: no cover - CLI entry point
    p = argparse.ArgumentParser(description="RA25 -- closing the mapped frontier (H157).")
    p.add_argument("--out", type=str, default="figures/ra25_close_frontier.csv")
    p.add_argument("--plot", type=str, default=None)
    p.add_argument("--budget", type=int, default=1600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bash", action="store_true",
                   help="cross-check the treadmill battery's realization against a real /bin/sh")
    args = p.parse_args()

    # 1. the fold closes the frontier: same adversary, base resolver vs the folded one
    base = NeuralAdversary(seed=args.seed).train(budget=args.budget, batch=args.batch,
                                                 sound_printf=True, fold_filters=False)
    folded = NeuralAdversary(seed=args.seed).train(budget=args.budget, batch=args.batch,
                                                   sound_printf=True, fold_filters=True)
    # 2 + 3. the treadmill battery and the reversibility partition
    fr = measure_frontier()
    v = ra25_verdict(fr, base.reward_per_call, folded.reward_per_call)
    demo_cmd, demo_verdict, demo_irrev = irreversible_demo()

    print("\nRA25 / H157 -- closing the mapped frontier, and why it is a treadmill\n")
    print(f"  budget: {args.budget} oracle calls/arm   seed: {args.seed}")

    print("\n  1. the fold closes the frontier (same adversary, base vs folded resolver):")
    print(f"     adversary reward/call   base {base.reward_per_call:.3f}  ->  "
          f"folded {folded.reward_per_call:.3f}   (frontier_closed={v['frontier_closed']})")
    print(f"     soundness (silent miss) base {base.silent_miss}   folded {folded.silent_miss}")

    print("\n  2. the treadmill battery (fold_filters covers only what it was taught):")
    for name, _cov, cmd in treadmill_battery():
        from verisim.realagent.shell_resolver import abstract_targets_protected as R
        print(f"     {name:14s} base {R(cmd):8s} -> folded {R(cmd, fold_filters=True):8s}")
    print(f"     covered {fr.covered_before}/{fr.n_filters} -> {fr.covered_after}/{fr.n_filters}; "
          f"treadmill tail still ABSTAIN: {fr.abstain_after}")

    print("\n  3. the post-commit diff is the real closer (reversibility partition):")
    print("     reversible ABSTAIN (post-commit-diff safe, no folder needed): "
          f"{fr.reversible_abstain}")
    print(f"     irreversible ABSTAIN (the folder's job): {fr.irreversible_abstain}")
    print(f"     irreversible demo: {demo_verdict} / irreversible={demo_irrev}  ::  {demo_cmd}")

    print("\n  verdict:")
    for k in ("frontier_closed", "soundness_preserved", "fold_closed_some_filters",
              "treadmill_is_unbounded", "reversible_residual_postcommit_safe"):
        print(f"     {k} = {v[k]}")

    # CSV: cumulative adversary reward vs oracle call, base vs folded
    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = ["oracle_call,base_reward,folded_reward"]
    rows += [f"{i + 1},{base.reward_curve[i]:.3f},{folded.reward_curve[i]:.3f}"
             for i in range(base.budget)]
    out.write_text("\n".join(rows) + "\n")
    print(f"\n  wrote {out}")

    if args.bash:
        mism = bash_cross_check(tempfile.gettempdir())
        print(f"  /bin/sh cross-check: {len(mism)} unfaithful / {fr.n_filters} treadmill commands")

    if args.plot:
        from figures.plot_ra25 import plot
        plot(base, folded, fr, args.plot)
        print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
