# Verisim docs — index

Navigation for the write-ups, run records, and specs. The front door is the
[top-level README](../README.md); the complete preserved history is [RESEARCH-LOG.md](../RESEARCH-LOG.md).

## Start here (distilled)

- [essay-state-not-capability.md](essay-state-not-capability.md) — the 5-minute thesis: your agent
  guardrail is stateless, and that is the whole problem.
- [paper.md](paper.md) — the ~8-page technical preprint.
- [review.md](review.md) — the hostile self-review (three reviewers) that forced the honesty.
- [related-work.md](related-work.md) — the maintained bibliography and positioning.
- [lineage.md](lineage.md) — the intellectual lineage (CEGIS, complete mediation, RLVR).

## The product

- [../src/verisim/certify/README.md](../src/verisim/certify/README.md) — the Coverage Certifier tool docs.
- [certifier-findings.md](certifier-findings.md) — what it caught on real hooks.
- [specs/SPEC-28.md](specs/SPEC-28.md) — the product vision and milestones.

## Run records (the experiments, newest-relevant first)

- [learned-proposer-eval-run.md](learned-proposer-eval-run.md) — SPEC-27, the pre-registered null that
  retracted the 18× claim.
- [cumulative-horizon-run.md](cumulative-horizon-run.md) — the cumulative-harm hunt (a null).
- [llm-guardrail-audit-run.md](llm-guardrail-audit-run.md) — the real-Claude-judge guardrail audit.
- [monitor-auditor-run.md](monitor-auditor-run.md) · [monitor-auditor-depth-run.md](monitor-auditor-depth-run.md)
  — the certified monitor-auditor.
- [neural-proposer-run.md](neural-proposer-run.md) · [learned-proposer-run.md](learned-proposer-run.md)
  — the learned adversarial proposers (efficiency claims corrected by SPEC-27).
- [coverage-synthesis-run.md](coverage-synthesis-run.md) · [frontier-close-run.md](frontier-close-run.md)
  — automated coverage synthesis and the mapped-frontier closer.
- [cc-corpus-run.md](cc-corpus-run.md) — the §8 operating point on 123,195 real Claude Code commands.
- [terminal-bench-run.md](terminal-bench-run.md) · [live-run-2026-06-20.md](live-run-2026-06-20.md)
  — the live Docker + real-Claude-agent arc and the official Terminal-Bench number.

## Semantics (the four worlds)

- [host-semantics.md](host-semantics.md) · [network-semantics.md](network-semantics.md) ·
  [distributed-semantics.md](distributed-semantics.md) · [semantics.md](semantics.md) ·
  [model-representation.md](model-representation.md) · [verification.md](verification.md)

## Specifications

- [specs/](specs/) — SPEC-1 … SPEC-28. [specs/SPEC.md](specs/SPEC.md) is the master science spec;
  [specs/SPEC-28.md](specs/SPEC-28.md) is the only active one.

## Meta / integration

- [openlore-integration-viability.md](openlore-integration-viability.md) ·
  [openlore-ingest-contract.md](openlore-ingest-contract.md) — the openlore architectural-memory integration.
- [essays/](essays/) — long-form drafts.
