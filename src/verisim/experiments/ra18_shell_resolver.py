"""Experiment RA18 -- the abstract shell-path resolver: the open edge, split and closed (H150).

Measures the RA4 regex target vs the RA18 abstract resolver across explicit attacks, benign idioms,
and an adversarial indirection battery. The headline: the regex silently misses the paper's named
open edge (3/3); the resolver closes the string-resolvable slice (FIRES), routes the runtime-
dependent slice (ABSTAIN -> reversibility routing, never a silent miss), and names the symlink
class as the provably-irreducible residual handled by the post-commit diff -- all with zero attack-
regression and zero benign over-fire. Hermetic, torch-free, deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.shell_resolver_eval import cu_ra18_verdict, run_ra18, write_csv


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA18 -- the abstract shell-path resolver (H150).")
    parser.add_argument("--out", type=str, default="figures/ra18_shell_resolver.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    r = run_ra18()
    v = cu_ra18_verdict(r)

    print("\nRA18 / H150 -- the abstract shell-path resolver: the open edge, split and closed\n")
    print(f"  corpora: {r.n_attacks} attacks, {r.n_benign} benign, {r.n_evasions} evasions\n")
    print(f"    {'metric':30s} {'regex':>8s} {'resolver':>10s}")
    print(f"    {'attack fire (no regression)':30s} {r.regex_attack_fire:8.2f} "
          f"{r.resolver_attack_fire:10.2f}")
    print(f"    {'benign false-fire':30s} {r.regex_benign_false_fire:8.2f} "
          f"{r.resolver_benign_false_fire:10.2f}")
    print(f"    {'named-edge silent miss':30s} {r.regex_named_edge_miss:8.2f} "
          f"{r.resolver_silent_miss:10.2f}")

    print("\n  the open edge, partitioned by the resolver:")
    print(f"    string-resolvable slice CLOSED (FIRES) = {r.resolver_string_resolvable_caught:.2f}")
    print(f"    runtime slice ROUTED (ABSTAIN)          = {r.resolver_runtime_abstained:.2f}  "
          f"(fail-closed {r.abstain_irreversible_failclosed}, post-commit "
          f"{r.abstain_reversible_postcommit})")
    print(f"    symlink residual (CLEAR, post-commit)   = {r.resolver_symlink_residual:.2f}  "
          f"(provably irreducible to string analysis)")
    print(f"    resolver silent-miss on string-visible commands = {r.resolver_silent_miss:.2f}")

    print("\n  verdict:")
    for k in ("no_attack_regression", "no_benign_false_fire", "regex_missed_named_edge",
              "resolver_closes_string_slice", "resolver_routes_runtime_slice",
              "resolver_zero_silent_miss", "symlink_is_named_residual"):
        print(f"    {k} = {v[k]}")

    out = write_csv(r, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra18 import plot_ra18

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra18(r, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
