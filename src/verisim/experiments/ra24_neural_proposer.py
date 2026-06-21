"""Experiment RA24 -- a neural compositional adversary vs the real RA18 resolver (H156).

Two lanes, both with the exact oracle as the only reward (no learned reward model):

  - **discovery** (``sound_printf=False``, the pre-RA24 resolver): the learned adversary hunts the
    printf-format-escape silent miss the hand-run RA18 red team missed, far faster than blind.
  - **hardened** (default): the RA24 fix routes that class to ABSTAIN; the soundness invariant holds
    (silent_miss == 0) under the learned adversary, which now maps the folder-incompleteness
    frontier
    (string-resolvable ``cut`` / ``rev`` pipe-filters) the resolver cannot fold.

In both lanes the neural policy is compared to a blind-uniform proposer over the same full
compositional space (sample efficiency) and to a single-transform proposer with RA23's architecture
(one mechanism for the whole path -- the compositionality control). Optionally cross-checks every
sampled command's realization label against a real /bin/sh (``--bash``). torch only for the policy.
"""

from __future__ import annotations

import argparse
import tempfile

from verisim.realagent.neural_proposer import (
    NeuralAdversary,
    RunResult,
    ra24_verdict,
    run_blind,
    run_single_transform,
    write_csv,
)


def _lane(budget: int, batch: int, seed: int,
          sound_printf: bool) -> tuple[RunResult, RunResult, RunResult]:
    neural = NeuralAdversary(seed=seed).train(budget=budget, batch=batch, sound_printf=sound_printf)
    blind = run_blind(budget, seed=seed, sound_printf=sound_printf)
    single = run_single_transform(budget, seed=seed, sound_printf=sound_printf)
    return neural, blind, single


def main() -> None:  # pragma: no cover - CLI entry point
    p = argparse.ArgumentParser(description="RA24 -- neural compositional adversary (H156).")
    p.add_argument("--out", type=str, default="figures/ra24_neural_proposer.csv")
    p.add_argument("--plot", type=str, default=None)
    p.add_argument("--budget", type=int, default=1600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bash", action="store_true",
                   help="cross-check sampled commands' realization labels against a real /bin/sh")
    args = p.parse_args()

    # discovery lane (pre-RA24 resolver): the learned adversary finds the soundness bug
    d_neural, d_blind, d_single = _lane(args.budget, args.batch, args.seed, sound_printf=False)
    # hardened lane (default resolver): soundness holds; map the frontier
    h_neural, h_blind, h_single = _lane(args.budget, args.batch, args.seed, sound_printf=True)
    v = ra24_verdict(h_neural, h_blind, h_single)

    print("\nRA24 / H156 -- a neural compositional adversary vs the RA18 resolver\n")
    print(f"  budget: {args.budget} oracle calls/arm   batch: {args.batch}   seed: {args.seed}")

    print("\n  -- discovery lane (pre-RA24 resolver; printf-format-escape silent miss live) --")
    print(f"    silent misses found    neural {d_neural.silent_miss:4d}   "
          f"blind {d_blind.silent_miss:4d}"
          f"   single {d_single.silent_miss:4d}")
    print(f"    min silent-miss depth  neural {d_neural.min_silent_depth}   "
          f"single {d_single.min_silent_depth}"
          f"   (single confined to uniform depth {len(d_single.silent_curve) and 6})")
    print(f"    distinct silent comps  neural {d_neural.distinct_silent_compositions}"
          f"   blind {d_blind.distinct_silent_compositions}")

    print("\n  -- hardened lane (RA24 fix; default resolver) --")
    print(f"    silent misses          neural {h_neural.silent_miss}   blind {h_blind.silent_miss}"
          f"   single {h_single.silent_miss}   (soundness_holds={v['soundness_holds']})")
    print(f"    reward / oracle call   neural {h_neural.reward_per_call:.3f}   "
          f"blind {h_blind.reward_per_call:.3f}   single {h_single.reward_per_call:.3f}")
    print(f"    distinct frontier comps  neural {h_neural.distinct_frontier_compositions}   "
          f"blind {h_blind.distinct_frontier_compositions}   "
          f"single {h_single.distinct_frontier_compositions}")
    print(f"    min frontier depth     neural {h_neural.min_frontier_depth}   "
          f"single {h_single.min_frontier_depth}  (single is uniform-only)")

    print("\n  verdict:")
    for k in ("neural_more_efficient_than_blind", "neural_beats_single_transform",
              "soundness_holds", "neural_explores_mixed_compositions"):
        print(f"    {k} = {v[k]}")

    path = write_csv(h_neural, h_blind, h_single, args.out)
    print(f"\n  wrote {path}")

    if args.bash:
        import random

        from verisim.realagent.compositional_grammar import (
            ATOMS,
            MECHANISMS,
            VERBS,
            Action,
            bash_cross_check,
        )
        rng = random.Random(args.seed)
        sample = [Action(rng.randrange(len(VERBS)), rng.randrange(2),
                         tuple(rng.randrange(len(MECHANISMS)) for _ in ATOMS)) for _ in range(200)]
        mism = bash_cross_check(tempfile.gettempdir(), sample)
        print(f"  /bin/sh cross-check: {len(mism)} unfaithful / {len(sample)} sampled commands")

    if args.plot:
        from figures.plot_ra24 import plot
        plot(d_neural, d_blind, d_single, h_neural, h_blind, h_single, args.plot)
        print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
