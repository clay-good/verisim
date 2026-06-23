"""SPEC-28 -- the Coverage Certifier CLI: ``python -m verisim.certify audit --hook <path>``.

Audits a deployed PreToolUse hook and prints the bypasses + a coverage certificate. M1 output is the
terminal summary + the certificate JSON; the human-readable compliance report (OWASP ASI mapping) is
M2.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC

from .core import certify_denylist, certify_hook


def _audit(args: argparse.Namespace) -> int:
    if args.denylist is not None:
        patterns = [p.strip() for p in args.denylist.split(",") if p.strip()]
        result = certify_denylist(patterns, protected_path=args.protected, proposer=args.proposer,
                                  budget=args.budget, seed=args.seed)
    else:
        result = certify_hook(
            args.hook,
            protected_path=args.protected,
            proposer=args.proposer,
            budget=args.budget,
            oracle=args.oracle,
            seed=args.seed,
        )
    cert = result.certificate

    print(f"\nverisim-certify — auditing {result.monitor}")
    print(f"  protected path : {result.protected_path}")
    print(f"  oracle         : {result.oracle}")
    print(f"  proposer       : {result.proposer}  (budget {args.budget})")
    print(f"  sampled space  : {cert.n_proposed} actions, {cert.n_realizing} realizing")
    print(f"\n  {result.verdict_line()}")

    if result.bypasses:
        shown = result.bypasses[: args.max_show]
        print("\n  bypasses (commands the guardrail ALLOWS but that realize harm):")
        for h in shown:
            print(f"    [{h.klass:>12}] {h.command}")
        if len(result.bypasses) > len(shown):
            print(f"    … and {len(result.bypasses) - len(shown)} more (--max-show to see more)")

    if args.out:
        path = cert.write(args.out)
        print(f"\n  certificate JSON   : {path}")
    if args.report:
        from datetime import datetime

        from .report import render_report

        target = (f"--denylist '{args.denylist}'" if args.denylist is not None
                  else f"--hook {args.hook} --oracle {args.oracle}")
        cmd = (f"python -m verisim.certify audit {target} "
               f"--proposer {args.proposer} --budget {args.budget}")
        md = render_report(result, reproduce_cmd=cmd,
                           generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
        from pathlib import Path

        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(md)
        print(f"  certificate report : {args.report}")
    print()
    return 1 if result.bypasses else 0  # nonzero exit on a leak -> usable as a CI gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verisim-certify",
        description="Prove an agent guardrail's completeness, don't assert it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="audit a deployed guardrail (a PreToolUse hook or a denylist)")
    target = a.add_mutually_exclusive_group(required=True)
    target.add_argument("--hook", help="path to the hook script (a PreToolUse hook)")
    target.add_argument("--denylist", help="comma-separated deny patterns to audit directly")
    a.add_argument("--protected", default="/etc/shadow", help="protected path to probe")
    a.add_argument("--proposer", choices=("enumerate", "bandit", "both"), default="enumerate")
    a.add_argument("--budget", type=int, default=512, help="oracle-call budget")
    a.add_argument("--oracle", choices=("syntactic", "real"), default="syntactic",
                   help="syntactic (fast) or real (prove vs /bin/sh in a sandbox)")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--max-show", type=int, default=15, help="max bypasses to print")
    a.add_argument("--out", default=None, help="write the certificate JSON here")
    a.add_argument("--report", default=None, help="write a human-readable Markdown report here")
    a.set_defaults(func=_audit)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
