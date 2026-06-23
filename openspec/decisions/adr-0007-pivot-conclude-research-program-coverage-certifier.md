# ADR-0007: Pivot: conclude research program; Coverage Certifier (SPEC-28) is the only active priority

## Status

accepted

**Domains**: unknown

## Context

After top-down review and market diligence (June 2026), the mechanism research frontier is closed for a solo researcher and SPEC-27 was the right capstone. The agentic-AI-security market (~$8.7B, 337 vendors) has an empty middle: nobody certifies whether a deployed tool-call rail is actually complete with adversarial proof. Incident data (88% had incidents; 97% of breached missing access controls) plus standards demand (OWASP Agentic Top 10 ASI02/05/03; EU AI Act; NIST; CSA STAR) make a coverage certificate a real, timed need. The repo's durable asset — the certify-don't-assert audit() loop with a free exact oracle (prove the bypass vs real /bin/sh, emit a number with a CI) — is the unique fit. Shape: OSS tool + public writeup, not a venture-scale startup. SPEC-28 productizes existing audited components (audit(), SubprocessMonitor, ShellPathOracle/ContainerDiffOracle, BanditProposer/enumerate, Certificate); the neural proposer is retired (SPEC-27), confidentiality out of scope (RA15).

## Decision

Pivot: conclude research program; Coverage Certifier (SPEC-28) is the only active priority

## Consequences

SPEC-28 is the only active spec; all prior specs marked DONE or NOT DOING (ledger in SPEC-28 section 8). New docs: docs/specs/SPEC-28.md (vision), plans/SPEC-28-certifier-mvp.md (runbook + market context + build order M1-M4). No code this session; build starts next session at M1 (audit a real Claude Code PreToolUse hook end-to-end). Repo cleanup deferred to M4 — nothing deleted, all research preserved as evidence.

> Recorded by openlore decisions on 2026-06-23
> Decision ID: 90c85637
