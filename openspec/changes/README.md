# OpenSpec changes — Verisim ↔ OpenLore unified verification & CD prototype

These changes implement the viable core defined in
[docs/openlore-integration-viability.md](../../docs/openlore-integration-viability.md) §7, with
the scope locked in that document's §9 Resolution. Read the findings doc first — it explains
*why* the integration takes this shape and what it deliberately does **not** do (no direct
cross-process DB writes, no tracing of M_θ rollouts, no autonomous git/deploy).

## North star

A small, real, **human-gated** prototype that proves the research claim:

> A dynamic, oracle-grounded simulator (Verisim) and a static structural call-graph
> (OpenLore) can be unified so that **runtime reality corrects the static graph** through a
> safe provenance contract, and the whole loop runs locally, network-isolated, and end-to-end
> from intent to a *prepared, human-confirmed* delivery.

## Invariants every change in this set must hold

- **One-way DB access.** Verisim reads OpenLore's `call-graph.db` as a regenerable cache and
  never writes it. All write-back is a *structured payload* OpenLore ingests on its own terms.
- **Trace the real surface, not the model.** Dynamic tracing attaches to the `SandboxOracle`'s
  real execution (SPEC-11), never to M_θ imagination rollouts.
- **Human-gated irreversibility.** No autonomous `git reset`/file delete/commit/push/deploy. The
  loop *prepares* and *recommends*; a human confirms.
- **Determinism preserved.** Nothing perturbs the sandbox hermeticity (SPEC-11 §2.3) or the
  torch-free determinism discipline (SPEC-5 §13).
- **Platform-honest.** macOS-first; eBPF is never a hard default (Linux-only). See
  [memory: macos-first-testing].

## The six changes (dependency order)

| # | Change | Domain | Depends on | One line |
|---|--------|--------|------------|----------|
| 1 | `add-defanged-codebase-fixture` | `fixture` | — | Copy a real local repo, de-fang `.git`, deterministic selection — the subject under test. |
| 2 | `add-openlore-graph-adapter` | `openlore-bridge` | 1 | Read OpenLore's call graph (read-only) and expose it as a typed substrate Verisim can ground on. |
| 3 | `add-sandbox-trace-oracle` | `trace-oracle` | 1 | Wrap the real `SandboxOracle` to capture exec/syscall/bind/file-mutation traces as typed records. |
| 4 | `add-synthesized-edge-feedback` | `openlore-bridge` | 2,3 | Turn trace/invariant discrepancies into a `synthesized_by`-conformant payload OpenLore can ingest. |
| 5 | `add-trajectory-oscillation-breaker` | `safety-breaker` | — | Detect oscillation/repetitive-edit loops; freeze + drop speculative rollout; recommend rollback (human-gated). |
| 6 | `add-headless-cd-pipeline` | `cd-pipeline` | 1–5 | The headless entry point chaining intent → simulate → trace → evaluate → breaker → prepare delivery (human-confirmed). |

## Spec status convention

Every `proposal.md` carries a `> Status:` line. **All six are now `IMPLEMENTED`** (2026-06-14) —
the full chain ships: fixture → graph adapter → trace oracle → synthesized-edge feedback →
oscillation breaker → the headless CD pipeline (`src/verisim/pipeline/`, console script
`verisim-cd`) that composes them behind one ordered, fail-safe, network-isolated, human-gated entry
point. Each `specs/<domain>/spec.md`
uses the standard OpenSpec delta headers (`## ADDED Requirements`, `### Requirement:`,
`#### Scenario:` with GIVEN/WHEN/THEN). Tasks are checklists with explicit verification steps.
