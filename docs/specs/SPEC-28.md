# SPEC-28 — The Coverage Certifier: prove your agent guardrail's completeness, don't assert it

> **✅ ACTIVE PRIORITY — set 2026-06-23. THIS IS THE ONLY ACTIVE SPEC.** The research program
> (SPEC-1…27, CU1–35, RA1–26) is **DONE** — its honest result is the essay
> [`docs/essay-state-not-capability.md`](../essay-state-not-capability.md), the SPEC-27 null, and the
> one durable, battle-tested asset underneath it all: the **certify-don't-assert** loop (`audit()`,
> SPEC-23). SPEC-28 turns that asset into the thing the program was always pointing at — a tool a
> security engineer runs against the guardrail they already shipped, that catches real coverage gaps
> and proves them. No prior spec is active work. The program status ledger is §8.

> **✅ MVP BUILT — M1–M4 done, 2026-06-23.** The certifier ships in
> [`src/verisim/certify/`](../../src/verisim/certify/) (`python -m verisim.certify audit`), green on
> ruff + strict mypy with 13 tests; it audits a real `PreToolUse` hook and a denylist, emits bypasses
> + a coverage certificate + a Markdown report, and found real gaps in both targets it was pointed at
> (see [`docs/certifier-findings.md`](../certifier-findings.md)). The README now leads with the
> product; the research log is preserved below the divider. The open goal is distribution — a
> non-author running it against their own hook (§7 success criterion).

---

## 1. The vision, in one sentence

A security engineer points a CLI at the guardrail they already shipped — a Claude Code `PreToolUse`
hook, an MCP-gateway policy, a denylist — and it returns **the concrete commands that realize harm
*off* their checked surface, each proven against a real shell, plus a coverage certificate with a
confidence interval.** It catches real gaps, and it produces evidence a CISO or an auditor will accept.

The product is not another guardrail (there are 174) and not another prompt-injection red-teamer (there
are a dozen). It is the missing **measurement that connects them**: *is the tool-call rail you deployed
actually complete, and here is the adversarial proof and the number.*

## 2. Why now, why this — the outside view (June 2026, sourced)

The agentic-AI-security market is large and bifurcated, and the certifier sits in its empty middle.

- **The market is split into "find" and "assert," with nothing certifying coverage.** Red-team tools
  *find* prompt attacks — [Promptfoo (acquired by OpenAI, Mar 2026)](https://generalanalysis.com/guides/best-ai-red-teaming-tools),
  garak, [General Analysis](https://generalanalysis.com/guides/best-ai-red-teaming-tools), DeepTeam,
  Lakera, Mindgard. Guardrails *assert* completeness —
  [TrueFoundry (Cedar/OPA at the PreTool hook)](https://www.truefoundry.com/blog/claude-code-enterprise-mcp-gateway),
  [Microsoft Agent Governance Toolkit](https://developer.microsoft.com/blog/securing-mcp-a-control-plane-for-agent-tool-execution),
  LlamaFirewall, Galileo, APort, NeMo Guardrails. The
  [AI-security startup map tracks 337 vendors, ~$8.7B raised, 174 in runtime/guardrails](https://startups.prompt.security/).
  **None certifies whether a *specific deployed* tool-call rail is complete, with adversarial proof.**
- **The incident data says the gap is real.**
  [88% of orgs running agents had a security incident; 97% of breached orgs were "missing proper access
  controls"](https://www.truefoundry.com/blog/enterprise-ai-agent-security-solutions); the market's own
  framing is that *"most 'the agent did something it shouldn't' incidents trace to a missing tool-call
  rail."* That missing rail — and the inability to prove a rail is complete — is exactly what SPEC-28
  measures.
- **Standards now demand evidence, not assertions.** The
  [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
  names the surface directly — **ASI02 Tool Misuse & Exploitation, ASI05 Unexpected Code Execution,
  ASI03 Identity & Privilege Abuse.** The EU AI Act becomes fully applicable
  [2 August 2026, with adversarial testing and a cybersecurity assessment among the obligations](https://www.mckennaconsultants.com/eu-ai-act-high-risk-compliance-a-technical-readiness-guide-for-august-2026/)
  (honest note: the AI Omnibus postponed parts of the Article 6(2) high-risk obligations to Dec 2027 /
  Aug 2028, so this is a tailwind, not a gun to the head). NIST's Feb 2026 agent-standards initiative
  and CSA's STAR for AI / AICM push the same way. **A coverage certificate is the evidence artifact
  these regimes want.**
- **Our unique moat is the free exact oracle.** Every red-teamer says "an LLM judge thinks this is
  risky." We do not. We **execute the candidate action in a sandbox and read the true effect against a
  real `/bin/sh`** — so a reported bypass is *proven* to realize harm, not guessed, and the coverage
  number is grounded in ground truth, not a model's opinion. This is the certify-don't-assert thesis
  the research validated, and SPEC-27 already stress-tested it honestly (it even corrected our own
  overclaim — the method is trustworthy *because* it caught us).

## 3. The user and the job-to-be-done

- **User:** a platform / security engineer at a company deploying coding or computer-use agents (Claude
  Code, an MCP-based agent) who has shipped *a* guardrail and now must answer, to a CISO, an EU-AI-Act
  technical file, or a SOC2 auditor: *"is it actually complete?"*
- **Job:** "Tell me, with proof, which dangerous actions my guardrail lets through, and give me a
  coverage number I can defend in a compliance document." Today they cannot answer this; they assert.

## 4. The MVP

**Input** — a deployed guardrail behind the existing `Monitor` protocol (SPEC-23). The MVP front door
is **already built**: `SubprocessMonitor` (SPEC-24 H168) drives an opaque external hook over
stdin/stdout, speaking the [Claude Code `PreToolUse` `hookSpecificOutput.permissionDecision` contract](https://code.claude.com/docs/en/permissions)
(allow/ask/deny). Target #1 is a Claude Code `PreToolUse` hook; target #2 is a denylist / MCP-gateway
policy.

**Engine** — the `audit()` loop (SPEC-23) with the **bandit + enumerate** proposers (SPEC-27 retired
the neural one: it lost to the bandit on every honest metric at 16× the cost, so the product uses the
cheap, competent, defensible search) and the **free oracle** (`ShellPathOracle` for fast triage,
`ContainerDiffOracle` to *prove* realization against a real `/bin/sh`), producing a `Certificate` with
the good-Turing residual bound + CI (SPEC-24).

**Output** —
1. a ranked list of **concrete bypasses**: the exact command that realizes harm off the checked
   surface, each oracle-confirmed against a real shell (the "here is the gap" the engineer acts on);
2. a **coverage certificate**: covered / silent / residual counts, the residual-rate confidence
   interval, the sampled space, the oracle named — as JSON for CI + a human-readable report that maps
   findings to OWASP **ASI02 / ASI05 / ASI03** (the "here is my evidence" for compliance).

**Packaging** — a CLI, working name `verisim-certify` (final name open; candidates: *Coverage*,
*RailProof*, *Certify*). Free OSS core (`certify audit --hook ./my_hook.py`); the hosted report,
CI-gate integration, and broader target catalogue are the post-MVP upsell, not the MVP.

## 5. What's reused — the research assets become product components

Nothing here is greenfield; the MVP is **assembly** of audited, tested code:

| component | what it is | spec |
|---|---|---|
| `audit()` loop | the discover→confirm→certify engine | SPEC-23 |
| `SubprocessMonitor` | audits an opaque external hook over stdin/stdout — **the product's front door** | SPEC-24 (H168) |
| `ShellPathOracle` / `ContainerDiffOracle` | the free verifier; the latter proves realization vs a real `/bin/sh` | SPEC-23 |
| `BanditProposer` / `GrammarProposer(enumerate)` | the competent proposers (SPEC-27: match/beat neural, cheap) | SPEC-27 |
| `Certificate` + `guarantee` | the coverage certificate with CI + good-Turing residual | SPEC-23/24 |
| the sparse-surface measurement | 1.1% of 123,195 real Claude Code commands touch the danger surface — "checking is nearly free" | RA21 |

## 6. Honest scope and limits — stated up front, because honesty *is* the brand

- **Integrity / state-change harms only.** Confidentiality / exfiltration is **out** (RA15 honest
  negative — no sparse surface to cover; any allowed channel leaks a bit). The certificate says so
  precisely: *"this certifies your integrity rail, not your DLP."* Naming the boundary is the product's
  credibility, not its weakness.
- **One harm family today** (file/credential corruption on protected paths via shell). The grammar is a
  realistic-but-finite attacker toolkit, not all of bash. The certificate is **empirical over the
  sampled space**, with a stated residual bound — not a proof over all inputs. We report the bound; we
  do not hide behind it.
- **We measure a gate; we do not replace it.** The certifier is the measurement, not the guardrail. It
  makes whatever rail you shipped honest.

## 7. Milestones (build starts next session — NO code this session)

- **M0: the vision.** ✅ DONE — SPEC-28 + the runbook.
- **M1: end-to-end on a real hook.** ✅ DONE — `verisim.certify` audits a `PreToolUse` hook via
  `SubprocessMonitor` and emits bypasses + a certificate. Verified: 66 bypasses on a weak denylist
  hook (8.3% coverage), 18 on the repo's own hardened resolver hook (75%, the `is_irreversible`
  blind spot). ([`src/verisim/certify/`](../../src/verisim/certify/), `tests/test_certify.py`.)
- **M2: the certificate report.** ✅ DONE — Markdown report with coverage %, Wilson CI + residual
  upper bound, bypasses by class, OWASP ASI02/03/05 mapping, the reversibility caveat, reproduce
  block. (`report.py`, `--report`.)
- **M3: a second target type + BYO docs.** ✅ DONE — `--denylist` audits a denylist in-process; the
  one-method `Monitor` protocol is the BYO interface ([tool README](../../src/verisim/certify/README.md)).
- **M4: package + plant the flag.** ✅ DONE (this session's scope) — the top-level README leads with
  the product; the findings writeup is [`docs/certifier-findings.md`](../certifier-findings.md). The
  *external* writeup ("we audited N **third-party** shipping hooks") is the open follow-up (N=2 here,
  both our own hooks).
- **Success criterion (the open goal):** a security engineer who is **not us** runs the tool against
  their own hook and it surfaces a real coverage gap they did not know about. The MVP is built and
  green; this is now a distribution/outreach step, not a build step.

## 8. Program status ledger — everything else is DONE or NOT DOING

| arc | status | artifact / why |
|---|---|---|
| SPEC-1…21 (world models, scaling, oracle self-supervision, the four worlds) | **DONE** | the research log — `docs/report.md` |
| SPEC-22 / RA1–26 (the oracle gate, complete mediation, the live arc) | **DONE** | `docs/paper.md` + the essay |
| SPEC-23–25 (the monitor-auditor, the certificate, the LLM-guardrail audit) | **DONE → reused** | this is SPEC-28's engine |
| SPEC-26 (cumulative-harm hunt) | **DONE** | null; folded into the essay |
| SPEC-27 (honest proposer evaluation) | **DONE** | null; the 18× retracted, the bandit baseline kept |
| the neural proposer | **NOT DOING** | retired by SPEC-27 (lost to the bandit at 16× the cost) |
| confidentiality / covert-channel frontier | **NOT DOING** | RA15 honest negative + owned externally; no sparse surface |
| more *mechanism* research | **NOT DOING** | frontier closed for a solo researcher (4 diligence passes) |
| certified-composition-safety research | **PARKED** | interesting, incremental; revisit only if it feeds the product |
| a venture-scale startup | **NOT DOING** | solo vs a crowded, $8.7B-funded field; OSS-tool + reputation is the right shape |
| **SPEC-28 — the Coverage Certifier** | **✅ ACTIVE** | the only active work |

## 9. Next

Build order, market context, reused-component paths, and anti-goals:
[`plans/SPEC-28-certifier-mvp.md`](../../plans/SPEC-28-certifier-mvp.md). Start next session at **M1**.
