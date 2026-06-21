"""SPEC-25 experiment -- audit a real LLM guardrail (H171-H174): where does the model judge break?

Points the SPEC-23/24 auditor at a SafePred-style model judge in the `Monitor` slot and reports, per
harm class, the judge's recall (with a Wilson CI) and the oracle-confirmed silent passes -- the
realizing actions an action-reading judge let through. Hermetic by default (a deterministic stub
judge); ``--claude`` runs the real ``claude -p`` lane on the file battery (run-on-demand).

  python -m verisim.experiments.spec25_llm_guardrail            # hermetic stub
  python -m verisim.experiments.spec25_llm_guardrail --claude   # real model lane (needs claude CLI)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.audit import (
    LLMCertificate,
    LLMGuardrailMonitor,
    ShellPathOracle,
    StubJudge,
    certify_llm,
)
from verisim.audit.llm_guardrail import file_proposer

PROTECTED, PREFIX, WORK = "/etc/shadow", "/etc", "/home/work"
CSV_HEADER = "judge,framing,klass,realizing_draws,recall,recall_lo,recall_hi,misses"


def _emit(c: LLMCertificate, rows: list[str]) -> None:
    print(f"\n  judge={c.monitor} framing={c.framing} trials={c.trials}")
    print(f"  overall recall {c.overall_recall:.3f} "
          f"[{c.overall_recall_lo:.3f}, {c.overall_recall_hi:.3f}]  "
          f"residual<{c.residual_epsilon:.4f}  found_break={c.found_break}")
    for klass in sorted(c.per_class):
        pc = c.per_class[klass]
        print(f"    {klass:14s} recall {pc['recall']:.2f} "
              f"[{pc['recall_lo']:.2f},{pc['recall_hi']:.2f}]  misses={int(pc['misses'])}")
        rows.append(f"{c.monitor},{c.framing},{klass},{int(pc['realizing_draws'])},"
                    f"{pc['recall']:.4f},{pc['recall_lo']:.4f},{pc['recall_hi']:.4f},"
                    f"{int(pc['misses'])}")


def main() -> int:  # pragma: no cover - CLI entry point
    ap = argparse.ArgumentParser(description="SPEC-25 LLM guardrail auditor (H171-H174).")
    ap.add_argument("--out", default="figures/spec25_llm_guardrail.csv")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--claude", action="store_true", help="run the real claude -p lane (on demand)")
    ap.add_argument("--samples", type=int, default=1, help="majority-vote samples per judge call")
    args = ap.parse_args()

    figdir = Path(args.out).parent
    rows = [CSV_HEADER]
    orc = ShellPathOracle(PREFIX)
    print("\nSPEC-25 LLM guardrail auditor -- where does the model judge break?")

    if args.claude:
        from verisim.audit import ClaudeCliJudge

        for framing in ("direct", "policy"):  # H174: the framing-transfer harness
            judge = ClaudeCliJudge(PREFIX, framing=framing, samples=args.samples)
            mon = LLMGuardrailMonitor(judge, framing=framing)
            cert = certify_llm(mon, orc, lambda: file_proposer(PROTECTED, WORK),
                               trials=args.trials)
            cert.write(str(figdir / f"claude_{framing}.json"))
            _emit(cert, rows)
    else:
        mon = LLMGuardrailMonitor(StubJudge(PREFIX))
        cert = certify_llm(mon, orc, lambda: file_proposer(PROTECTED, WORK), trials=args.trials)
        cert.write(str(figdir / "stub.json"))
        _emit(cert, rows)
        print("\n  (hermetic stub: an action-reader blocks the literal, passes every encoding -- "
              "the action-vs-effect gap. Real recall comes only from --claude.)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"\nwrote {args.out} + certificate JSON in {figdir}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
