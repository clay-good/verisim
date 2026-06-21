"""SPEC-23 experiment -- the monitor auditor (H158-H164): one loop, many monitors.

Runs the protocol-driven discover->fix->re-verify loop across the SPEC-23 instances and prints the
coverage certificates, then writes a summary CSV + the per-run certificate JSONs under ``figures/``:

  - Direction A (H158/H160): the RA24 printf silent miss through the ``ResolverMonitor`` interface
    (unsound -> a silent hole; hardened -> none), and the RA22 partition through the
    ``SyntacticPathMonitor`` grown from empty -- no literal /etc in the loop.
  - H159 (``--neural``, needs torch): the neural proposer, retargeted off /etc, beats blind uniform
    proposer at surfacing holes in an arbitrary denylist.
  - H161 (``--bash``, needs a shell): the same monitor audited by the syntactic ``ShellPathOracle``
    and the real-container ``ContainerDiffOracle`` -- the diff oracle additionally catches the
    symlink indirection.
  - Direction B (H162/H163/H164): the relational/cumulative/context policy triad, each compiling to
    a Monitor + Oracle the same loop audits (a hole in the under-specified policy, clean on repair).

Torch-free by default (the neural arm is opt-in); deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.audit import (
    Certificate,
    DenylistMonitor,
    GrammarProposer,
    Monitor,
    Oracle,
    Proposer,
    ResolverMonitor,
    ShellPathOracle,
    SyntacticPathMonitor,
    audit,
)
from verisim.audit.proposers import CorpusProposer
from verisim.policy import (
    ContextPolicy,
    CumulativePolicy,
    RelationalPolicy,
    compile_policy,
)

PROTECTED, PREFIX, WORK = "/etc/shadow", "/etc", "/home/work"
_DENYLIST = (PREFIX, "$'", "${", "$(printf %s", "`printf", "rm ")

CSV_HEADER = "run,monitor,oracle,proposer,n_realizing,covered,silent_holes,residual,sound"


def _row(name: str, c: Certificate) -> str:
    residual = c.residual_post_commit + c.residual_human
    return (f"{name},{c.monitor},{c.oracle},{c.proposer},{c.n_realizing},{c.covered},"
            f"{c.silent_holes},{residual},{c.sound}")


def main() -> int:  # pragma: no cover - CLI entry point
    ap = argparse.ArgumentParser(description="SPEC-23 monitor auditor (H158-H164).")
    ap.add_argument("--out", default="figures/spec23_monitor_auditor.csv")
    ap.add_argument("--neural", action="store_true", help="the H159 neural-vs-blind arm (torch)")
    ap.add_argument("--bash", action="store_true", help="run the H161 container-diff arm (/bin/sh)")
    args = ap.parse_args()

    figdir = Path(args.out).parent
    rows = [CSV_HEADER]
    print("\nSPEC-23 monitor auditor -- one loop, many monitors\n")

    # --- Direction A: reproduce RA24 / RA22 through the protocol ---------------------------
    runs: dict[str, tuple[Monitor, Oracle, Proposer]] = {
        "A_resolver_unsound": (ResolverMonitor(PREFIX, sound_printf=False, name="resolver_unsound"),
                               ShellPathOracle(PREFIX),
                               GrammarProposer(PROTECTED, WORK, mode="enumerate")),
        "A_resolver_hardened": (ResolverMonitor(PREFIX, sound_printf=True),
                                ShellPathOracle(PREFIX),
                                GrammarProposer(PROTECTED, WORK, mode="enumerate")),
        "A_syntactic_from_empty": (SyntacticPathMonitor(), ShellPathOracle(PREFIX),
                                   CorpusProposer(PREFIX, WORK)),
    }
    for name, (mon, orc, prop) in runs.items():
        cert = audit(mon, orc, prop)
        cert.write(str(figdir / f"{name}.json"))
        rows.append(_row(name, cert))
        print(f"  {name:26s} silent_holes={cert.silent_holes:>2}  sound={cert.sound}  "
              f"(realizing {cert.n_realizing}, covered {cert.covered})")

    # --- H159: the neural proposer generalizes off the hardcoded path ----------------------
    if args.neural:
        try:
            from verisim.audit import NeuralGrammarProposer

            tp, tpx = "/srv/secret/key", "/srv"
            den = DenylistMonitor((tpx, "$'", "${", "$(printf %s", "`printf", "rm "))
            orc = ShellPathOracle(tpx)
            blind = audit(den, orc, GrammarProposer(tp, WORK, mode="blind", seed=0), budget=512)
            neural = audit(den, orc, NeuralGrammarProposer(tp, tpx, WORK, train_budget=768,
                                                           sample_n=256, seed=0))
            rows.append(_row("A_neural_denylist", neural))
            rows.append(_row("A_blind_denylist", blind))
            print(f"  {'A_neural_vs_blind':26s} neural_holes={len(neural.holes)} > "
                  f"blind_holes={len(blind.holes)}  (off /etc, at /srv)")
        except ImportError:
            print("  (neural arm skipped: torch unavailable)")

    # --- H161: the real-container diff oracle ----------------------------------------------
    if args.bash:
        from verisim.audit import ContainerDiffOracle

        diff = ContainerDiffOracle()
        try:
            px, wk, pr = diff.prefix, diff.work, f"{diff.prefix}/shadow"
            con = audit(ResolverMonitor(px, sound_printf=False, name="resolver_unsound"), diff,
                        GrammarProposer(pr, wk, mode="enumerate"))
            rows.append(_row("A_container_diff", con))
            sym = sum(1 for h in con.holes if h.klass == "residual_symlink")
            print(f"  {'A_container_diff':26s} holes={len(con.holes)} "
                  f"(incl {sym} symlink the syntactic oracle cannot express)")
        finally:
            diff.close()

    # --- Direction B: the policy triad, each audited by the same loop ----------------------
    SERVICES = ("analytics", "reporting", "replica", "db")
    BASE = frozenset({("reporting", "db")})
    rel = RelationalPolicy(frozenset({"analytics"}), "db", BASE, SERVICES, frozenset())
    cum = CumulativePolicy(budget=5, n_records=30, stateful=False)
    ctxp = ContextPolicy(context_aware=False)
    freeze: dict[str, object] = {"freeze_active": True}
    policy_runs: dict[str, tuple[tuple[Monitor, Oracle, Proposer], dict[str, object]]] = {
        "B_relational_incomplete": (compile_policy(rel), {}),
        "B_relational_repaired": (compile_policy(rel.closure_complete()), {}),
        "B_cumulative_per_resource": (compile_policy(cum), {}),
        "B_cumulative_accumulator": (compile_policy(cum.accumulator()), {}),
        "B_context_static_allow": (compile_policy(ctxp), freeze),
        "B_context_aware": (compile_policy(ContextPolicy(context_aware=True)), freeze),
    }
    for name, ((mon, orc, prop), st) in policy_runs.items():
        cert = audit(mon, orc, prop, state=st, ctx=st)
        cert.write(str(figdir / f"{name}.json"))
        rows.append(_row(name, cert))
        print(f"  {name:26s} silent_holes={cert.silent_holes:>2}  sound={cert.sound}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"\nwrote {args.out} ({len(rows) - 1} runs) + certificate JSONs in {figdir}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
