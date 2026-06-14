# Contract-mediated synthesized-edge feedback

> Status: IMPLEMENTED — `src/verisim/bridge/feedback.py`, `schemas/verisim-feedback-v1.json`,
> `docs/openlore-ingest-contract.md`, tests in `tests/test_feedback.py` (24 tests, all green;
> full suite + bare `mypy` + `ruff` clean). Both discrepancy classes now ship: the primary
> missed-edge payload **and** the secondary architectural-invariant **findings** path
> (`detect_findings` + `LayerInvariant`/`RuntimeFinding`, populating the previously-reserved
> `findings[]` slot via `build_feedback_payload(..., invariants=...)`; empty when no invariants are
> declared, so the format is unchanged for the default missed-edge payload). OpenLore-side ingest
> remains a cross-repo dependency, stood in for by the local `validate_payload`.
> One sentence: **when Verisim's runtime traces reveal a call the static graph missed, emit a
> versioned, `synthesized_by`-conformant payload that OpenLore can ingest and integrate on its
> own terms — never a direct write into OpenLore's database.**

## Why

This is the payoff of the whole arc and the literal answer to the requester's data-path
decision (findings §9.2): *decoupled, with contract-mediated bidirectional sync.* Verisim reads
the static graph (Change 2) and observes runtime reality (Change 3). The interesting events are
the **discrepancies**:

- a **dynamically dispatched / indirect call** that actually fired at runtime but has no edge in
  the static graph (the exact blind spot OpenLore's own synthesis pass targets — callbacks,
  registries, framework routing, reflection);
- (secondary) an **architectural invariant violation** observed at runtime (e.g. a layer
  boundary crossed only via a runtime-resolved path).

OpenLore already has the right home for such edges: the `edges` table carries
`confidence = 'synthesized'` + a `synthesizedBy` rule name, and the analyzer keeps synthesized
edges **additive and provenance-labeled, never silently mixed** with directly-resolved edges
(`call-graph.ts`). Verisim's runtime evidence is just a new *source* of synthesized edges —
labeled, say, `synthesizedBy: 'verisim-runtime'` — distinguished from AST-synthesized ones.

## The honest constraint

There is **no external ingestion path in OpenLore today** — synthesis is internal AST
pattern-matching, and no `import-edges` API exists (verified: no `ingest`/`applyEdges` path in
`OpenLore/src`). So contract-mediated write-back has **two halves**:

1. **Verisim side (this change):** produce a **versioned JSON payload** of candidate edges, each
   carrying the full provenance OpenLore's `CallEdge` requires
   (`callerId`, `calleeId`/`calleeName`, `file`, `line`, `kind`, `callType`,
   `confidence: 'synthesized'`, `synthesizedBy: 'verisim-runtime'`) plus the **runtime evidence**
   (the trace id, action, fixture source sha) that justifies it. The payload conforms to a new
   schema in the style of `OpenLore/schemas/openlore-manifest-v1.json`.
2. **OpenLore side (cross-repo dependency, not built here):** an ingestion entry point that
   validates the payload, maps it onto `nodes` by stable id, and writes the edges through the
   *existing* synthesized-edge path so every consumer sees them as `synthesized` with Verisim
   provenance — and so OpenLore can **reject** anything that does not resolve to known nodes.

This change specifies (1) fully and pins (2) as an explicit interface contract Verisim emits
against, so the prototype is end-to-end the moment OpenLore implements the ingest counterpart
(which can be stubbed by a local validator for the prototype's own tests).

## What changes

1. **Discrepancy detector** — diff each `RuntimeTrace` (Change 3) against the static `CodeGraph`
   (Change 2): a runtime call with no corresponding static edge becomes a *candidate synthesized
   edge*; a runtime path that violates a declared invariant becomes a *candidate finding*.
2. **Node resolution** — map runtime exec/call sites to OpenLore node ids (by file+symbol), and
   **drop** candidates that cannot be resolved to known nodes (fail-safe: never propose an edge
   OpenLore cannot anchor).
3. **Payload producer** — emit the versioned `verisim-feedback-v1` JSON payload, idempotent and
   deduplicated (never re-propose an edge that already exists in the static graph, including
   pre-existing `synthesized` ones).
4. **Local validator** — a Verisim-side validator that checks payloads against the schema and
   against a fixture's node set, standing in for OpenLore's ingest during prototype tests.

## Contract / boundaries

- **Verisim never writes `call-graph.db`.** It writes a payload file; OpenLore (or the local
  validator stub) decides whether to ingest.
- **Additive + labeled only.** Candidates are always `confidence: 'synthesized'`,
  `synthesizedBy: 'verisim-runtime'`; they never modify or remove directly-resolved edges.
- **Evidence-bearing.** Every candidate carries the runtime trace id + fixture sha that justify
  it, so OpenLore (and a human) can audit *why* the edge is claimed.

## Risks & honest limits

- The OpenLore-side ingest does not exist yet; until it does, write-back lands in the validator
  stub, not the real DB. This is called out as a cross-repo dependency, not hidden.
- Symbol→node resolution across languages is imperfect; unresolved candidates are dropped, not
  forced — the prototype prefers precision over recall on write-back.
