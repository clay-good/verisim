# SPEC-28 runbook — the Coverage Certifier MVP

> Why this file exists: so next session can start cold at M1 with the full context — what we're
> building, why now, what's already built that we assemble, and what NOT to do. The spec
> [`docs/specs/SPEC-28.md`](../docs/specs/SPEC-28.md) is the vision; this is the build.

## The decision (2026-06-23 session)

After a top-down review and four web-diligence passes, we concluded: the *research* program is done
(mechanism frontier closed for a solo researcher; SPEC-27 was the right capstone). The durable asset is
the certify-don't-assert loop, and the highest-value, solo-achievable, genuinely-meaningful next thing
is to **ship it as a tool security engineers run against their own agent guardrails to catch real
coverage gaps.** Not a startup (crowded, $8.7B funded), not another paper — an OSS tool + a public
writeup that plants a flag. The user chose outcome (a): "security engineers run my tool against their
agents and it catches real coverage gaps."

## Market context (so we position correctly, sourced 2026-06-23)

- 337 agentic-AI-security vendors, ~$8.7B raised, 174 in runtime/guardrails (prompt.security map).
- Two camps, empty middle: red-teamers FIND prompt attacks (Promptfoo→OpenAI, garak, General Analysis,
  DeepTeam, Lakera, Mindgard); guardrails ASSERT completeness (TrueFoundry Cedar/OPA, Microsoft AGT,
  LlamaFirewall, Galileo, APort). Nobody CERTIFIES a deployed rail's coverage with a free-oracle proof.
- Incident data: 88% of agent-running orgs had an incident; 97% of breached missing access controls;
  "most 'agent did something it shouldn't' incidents trace to a missing tool-call rail."
- Standards/timing: OWASP Agentic Top 10 (ASI02 Tool Misuse, ASI05 Unexpected Code Exec, ASI03 Privilege
  Abuse); EU AI Act fully applicable 2 Aug 2026 (adversarial testing + cybersecurity assessment; note
  the AI Omnibus pushed parts of Art. 6(2) high-risk to 2027/28 — a tailwind, not a hard deadline);
  NIST Feb 2026 agent standards; CSA STAR for AI.
- Moat: the free exact oracle (prove the bypass against a real /bin/sh, emit a coverage number with a
  CI) — not "an LLM thinks it's risky."

## What's already built (assemble, don't rebuild)

- `src/verisim/audit/auditor.py` — `audit(monitor, oracle, proposer, budget) -> Certificate`. The engine.
- `src/verisim/audit/monitors.py::SubprocessMonitor` — drives an opaque external hook over stdin/stdout
  in the Claude Code PreToolUse contract. **The product's front door — already exists.**
- `src/verisim/audit/oracles.py` — `ShellPathOracle` (fast), `ContainerDiffOracle` (proves realization
  vs real /bin/sh).
- `src/verisim/audit/bandit.py::BanditProposer` + `proposers.py::GrammarProposer(mode="enumerate")` —
  the competent proposers SPEC-27 validated. (Do NOT use the neural proposer — SPEC-27 retired it.)
- `src/verisim/audit/protocols.py::Certificate` + `audit/guarantee.py` — certificate + CI + good-Turing
  residual.
- `scripts/claude_code_coverage_hook.py` — the repo's own PreToolUse hook; use it as the FIRST
  target-under-audit in M1 (we audit our own hook end-to-end before anyone else's).

## Build order (each milestone has a verify check)

1. **M1 — end-to-end on a real hook.** A CLI entry (`python -m verisim.certify audit --hook <path>`)
   that: wraps the hook in `SubprocessMonitor`, runs `audit()` with the bandit+enumerate proposers and
   `ContainerDiffOracle` (real-shell proof), prints the off-surface realizing commands + the
   certificate. First target: `scripts/claude_code_coverage_hook.py`.
   → verify: surfaces ≥1 oracle-confirmed off-surface realizing command; certificate JSON written.
2. **M2 — the certificate report.** Human-readable Markdown/terminal report: coverage %, residual CI,
   the bypass list, OWASP ASI02/05/03 mapping, a reproduce block.
   → verify: a non-author can read the report and know what leaked and what to fix.
3. **M3 — a second target + BYO-hook docs.** Audit a plain denylist and document the `Monitor` interface
   so a user can wrap their own hook/policy.
   → verify: the same CLI audits a denylist target and a user-supplied hook with no code change.
4. **M4 — package + plant the flag.** OSS front-door README (simplify, lead with the product; archive +
   link the research log — delete nothing), and a public writeup auditing N shipping hooks.
   → verify: README leads with "run this against your hook"; research preserved + linked.

## Anti-goals (don't drift)

- No new research mechanism. SPEC-28 is a product assembly of audited components.
- Don't resurrect the neural proposer. SPEC-27 retired it; the bandit/enumerate are the product search.
- Don't claim confidentiality/exfil coverage. Integrity/state-change only; name the boundary loudly.
- Don't delete research. Archive and link; the completed work is the evidence the certificate stands on.
- Don't build the hosted/SaaS layer in the MVP. Free OSS CLI first; catch a real gap on a real hook.

## Definition of done (MVP)

A security engineer who is not us can run `verisim-certify` against their own Claude Code hook (or
denylist) and get (a) concrete, oracle-proven bypasses and (b) a coverage certificate with a CI mapped
to OWASP ASI items — and it surfaces a real gap they did not know about.
