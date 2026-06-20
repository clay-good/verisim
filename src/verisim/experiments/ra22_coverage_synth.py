"""Experiment RA22 -- automated coverage: synthesize and certify the covering target (H154).

Runs the counterexample-guided (CEGIS) synthesizer from an empty target over a verb x path-form
corpus, and prints the coverage certificate. The headline: with no hand-specified surface, the loop
covers the entire literal realizing class and automatically isolates the indirection/symlink class
as irreducible to a pre-commit string pattern (routed to the post-commit diff) -- RA18's hand-drawn
partition, discovered. The soundness invariant holds: no realizing action is silently off-surface.

Optionally cross-checks the corpus's ground-truth labels against a real ``/bin/sh`` (``--bash``, the
RA2 anchor-invariance move). The synthesis itself is hermetic, torch-free, deterministic.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from verisim.realagent.coverage_synth import (
    cross_check_against_bash,
    cu_ra22_verdict,
    generate_corpus,
    synthesize,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA22 -- automated coverage synthesis (H154).")
    parser.add_argument("--out", type=str, default="figures/ra22_coverage_synth.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--bash", action="store_true",
                        help="cross-check the corpus labels against a real /bin/sh (RA2 anchor)")
    args = parser.parse_args()

    corpus = generate_corpus()
    _target, cert = synthesize(corpus)
    v = cu_ra22_verdict(cert)

    print("\nRA22 / H154 -- automated coverage: synthesize and certify the covering target\n")
    print(f"  corpus: {cert.n_candidates} candidates, {cert.n_realizing} realizing")
    print(f"  synthesized target (from EMPTY): {cert.synthesized_prefixes} "
          f"in {cert.rounds_to_converge} CEGIS rounds\n")
    n = cert.n_realizing or 1
    print(f"    {'class':22s} {'result':>10s}")
    print(f"    {'literal surface':22s} {cert.covered / n:10.2f}  covered (synthesized)")
    print(f"    {'indirection residual':22s} {cert.residual / n:10.2f}  routed to post-commit")
    print(f"    residual classes: {', '.join(cert.residual_classes)}")
    print(f"    silent miss (soundness): {cert.silent_miss}   "
          f"benign over-fire: {cert.benign_overfire}")

    print("\n  verdict:")
    for k in ("synthesized_from_empty", "no_silent_miss", "no_benign_overfire",
              "isolated_indirection_residual"):
        print(f"    {k} = {v[k]}")

    if args.bash:
        mism = cross_check_against_bash(tempfile.gettempdir())
        print(f"\n  bash cross-check (RA2 anchor): {len(mism)} label/exec mismatches "
              f"{'(labels are exact)' if not mism else '(!)'}")
        for m in mism[:10]:
            print(f"    MISMATCH {m}")

    out = write_csv(cert, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra22 import plot_ra22

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra22(cert, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
