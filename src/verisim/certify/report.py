"""SPEC-28 M2 -- the human-readable coverage certificate.

Turns a :class:`~verisim.certify.core.CertifyResult` into a Markdown report a security engineer (or
a compliance reviewer) can read without the code: the verdict, the coverage number with a confidence
interval, the bypasses grouped by class, the OWASP Agentic Top 10 mapping, and the honest contract
note (a "bypass" is what the guardrail *allows* at this layer; if you defer some to a downstream
control, verify it — reversibility is advisory, not a guarantee). Deterministic: the timestamp is
injected, never read from the clock, so the report is reproducible in tests.
"""

from __future__ import annotations

from collections import OrderedDict

from verisim.audit.guarantee import residual_epsilon, wilson_lower, wilson_upper
from verisim.audit.protocols import Hole

from .core import CertifyResult

#: the OWASP Top 10 for Agentic Applications (2026) items a protected-path corruption maps to.
OWASP_ASI = (
    ("ASI02", "Tool Misuse & Exploitation",
     "the agent's Bash tool is induced to corrupt a protected file off the checked surface"),
    ("ASI03", "Identity & Privilege Abuse",
     "the protected path is a credential/identity store; a bypass writes, deletes, or chmods it"),
    ("ASI05", "Unexpected Code Execution",
     "bypasses use shell substitution ($(...), backticks, printf) the gate did not fold"),
)


def _by_class(bypasses: list[Hole]) -> OrderedDict[str, list[Hole]]:
    groups: OrderedDict[str, list[Hole]] = OrderedDict()
    for h in sorted(bypasses, key=lambda x: (x.klass, x.command)):
        groups.setdefault(h.klass, []).append(h)
    return groups


def render_report(result: CertifyResult, *, reproduce_cmd: str | None = None,
                  generated: str | None = None) -> str:
    """Render the certificate as Markdown. ``generated`` is the timestamp string (injected, not read
    from the clock); ``reproduce_cmd`` is the exact command to re-run the audit."""
    cert = result.certificate
    n = cert.n_realizing
    b = len(result.bypasses)
    lo = 100.0 * wilson_lower(b, n)
    hi = 100.0 * wilson_upper(b, n)
    eps = 100.0 * residual_epsilon(b, n)
    reversible = sum(1 for h in result.bypasses if h.reversible)

    L: list[str] = []
    L.append(f"# Coverage Certificate — {result.monitor}")
    L.append("")
    badge = "🔴 LEAKS" if not result.sound else "🟢 SOUND"
    L.append(f"**Verdict:** {badge} — {result.verdict_line()}")
    if generated:
        L.append(f"**Generated:** {generated}")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| field | value |")
    L.append("|---|---|")
    L.append(f"| guardrail | `{result.monitor}` |")
    L.append(f"| protected path | `{result.protected_path}` |")
    L.append(f"| oracle | `{result.oracle}` |")
    L.append(f"| proposer / budget | `{result.proposer}` / {result.budget} |")
    L.append(f"| sampled space | {cert.n_proposed} actions, {n} realizing |")
    L.append(f"| coverage | {result.coverage_pct:.1f}% ({cert.covered}/{n}) |")
    L.append(f"| bypasses (in-contract gaps) | **{b}** |")
    L.append(f"| routed residual (deferred, symlink/runtime) | {result.routed_residual} |")
    L.append(f"| 95% CI on bypass rate (Wilson) | [{lo:.1f}%, {hi:.1f}%] |")
    L.append(f"| 95% upper bound on uncovered rate | ≤ {eps:.2f}% |")
    L.append("")
    L.append("## What this certifies (and what it does not)")
    L.append("")
    L.append("- **Certifies:** integrity / state-change harm on the protected path, over the "
             "sampled shell-encoding algebra. A bypass is an oracle-confirmed realizing command "
             "the guardrail **allows**.")
    L.append("- **Does not certify:** confidentiality / exfiltration (no sparse surface — out of "
             "scope by design), and it is **empirical over the sampled space**, not a proof over "
             "all of bash. Read the upper bound above as the honest residual.")
    L.append("")

    if result.bypasses:
        L.append("## Bypasses by class")
        L.append("")
        L.append("| class | count | example (a command the guardrail allows that realizes harm) |")
        L.append("|---|---|---|")
        for klass, holes in _by_class(result.bypasses).items():
            example = holes[0].command
            if len(example) > 88:
                example = example[:85] + "…"
            L.append(f"| `{klass}` | {len(holes)} | `{example}` |")
        L.append("")

    L.append("## OWASP Agentic Top 10 (2026) mapping")
    L.append("")
    for code, name, why in OWASP_ASI:
        L.append(f"- **{code} {name}** — {why}.")
    L.append("")

    L.append("## Honest notes")
    L.append("")
    L.append("- A *bypass* is a command this guardrail **allows** that realizes harm and is in the "
             "guardrail's contract — not the symlink/runtime residual it explicitly defers.")
    if result.bypasses:
        L.append(f"- {reversible} of {b} bypasses are classified **reversible** by the same "
                 "indirection-aware check the gate uses. If your deployment has a downstream "
                 "control (e.g. a post-commit filesystem diff) for reversible actions, that subset "
                 "*may* be caught there — **verify it.** The irreversible subset has no recourse.")
        L.append("- Reversibility is **advisory, not a guarantee**: the classifier can be fooled "
                 "by the same obfuscation (an obfuscated `rm` can read as reversible). Treat "
                 "allowed irreversible-looking commands as the highest severity.")
    L.append("")

    L.append("## Reproduce")
    L.append("")
    L.append("```")
    L.append(reproduce_cmd or f"python -m verisim.certify audit --hook <path> "
             f"--proposer {result.proposer} --budget {result.budget}")
    L.append("```")
    L.append("")
    return "\n".join(L)
