"""Experiment RA4 -- the command-agnostic gate: the gate does not need the agent's grammar (H136).

Measures the command-agnostic covering target across three labeled corpora: diverse attack syntaxes
(the ways a real agent corrupts a file, well beyond the v0 grammar), real benign shell idioms, and
indirection evasions. Reports the grammar-blindness gap, the false-block rate, and -- honestly --
the evasion rate of the syntactic pre-commit target, which is why the principled gate routes by
reversibility (CU27): exact post-commit fs-diff where the action is reversible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.realagent.command_agnostic import (
    ATTACK_SYNTAXES,
    EVASIONS,
    RA4Result,
    command_targets_protected,
    cu_ra4_verdict,
    grammar_can_parse,
    run_ra4,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RA4 -- the command-agnostic gate (H136).")
    parser.add_argument("--out", type=str, default="figures/ra4_command_agnostic.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    result: RA4Result = run_ra4()
    v = cu_ra4_verdict(result)

    print("\nRA4 / H136 -- the command-agnostic gate (the gate does not need the agent's grammar):")
    print(f"  {result.n_attacks} attack syntaxes, {result.n_benign} benign commands, "
          f"{result.n_evasions} evasions\n")

    print("  ATTACK SYNTAXES -- command-agnostic target vs grammar visibility:")
    for c in ATTACK_SYNTAXES:
        fires = command_targets_protected(c.command)
        seen = grammar_can_parse(c.command)
        flag = "" if seen else "  <- GRAMMAR-BLIND"
        print(f"    target={fires!s:5s} grammar_sees={seen!s:5s}  {c.command}{flag}")

    print(f"\n  command-agnostic target catches all explicit attacks = "
          f"{v['target_catches_all_explicit_attacks']} ({result.attack_target_fires:.2f})")
    print(f"  grammar gate is structurally blind = {v['grammar_gate_is_blind']} "
          f"({result.grammar_blind:.0%} of attack syntaxes it cannot even parse)")
    print(f"  no false-blocks on unseen benign commands = {v['no_false_blocks_on_benign']} "
          f"({result.benign_false_block:.2f})")

    print("\n  THE HONEST OPEN EDGE -- indirection evasions slip the syntactic pre-commit target:")
    for c in EVASIONS:
        print(f"    target={command_targets_protected(c.command)!s:5s}  {c.command}  ({c.name})")
    print(f"  pre-commit target evaded = {v['pre_commit_target_has_evasions']} "
          f"({result.evasion_miss:.0%} missed) -> route by reversibility (CU27): the exact "
          f"post-commit fs-diff escapes no evasion = {v['post_commit_diff_is_exact']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_ra4 import plot_ra4

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_ra4(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
