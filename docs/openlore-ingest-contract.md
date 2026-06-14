# OpenLore-side ingest contract for Verisim runtime feedback

This is the **cross-repo interface** the Verisim `verisim-feedback-v1` payload
([schema](../schemas/verisim-feedback-v1.json)) is emitted against. Verisim builds and validates
the payload (`src/verisim/bridge/feedback.py`); the counterpart described here lives in **OpenLore**
and is **not built in this repository**. Until it exists, write-back lands in Verisim's local
validator stand-in (`validate_payload`), not the real database — called out as a cross-repo
dependency, not hidden.

## Why a payload, not a database write

OpenLore's `call-graph.db` is OpenLore's store of record; Verisim treats it as a regenerable,
**read-only** cache (Change 2). OpenLore already keeps synthesized edges additive and
provenance-labeled — the `edges` table carries `confidence = 'synthesized'` plus a `synthesizedBy`
rule name, and the analyzer never silently mixes them with directly-resolved edges. Verisim's
runtime evidence is just a **new source** of synthesized edges, labeled
`synthesizedBy = 'verisim-runtime'`. So the contract is: Verisim emits a payload of candidate
edges; OpenLore decides, on its own terms, whether to ingest each one.

There is **no external ingestion path in OpenLore today** (synthesis is internal AST
pattern-matching; no `import-edges` / `applyEdges` API exists). This document pins the entry point
OpenLore would add so the prototype is end-to-end the moment OpenLore implements it.

## Inputs

A single `verisim-feedback-v1` JSON document conforming to
[`schemas/verisim-feedback-v1.json`](../schemas/verisim-feedback-v1.json):

- `version` — must be `"verisim-feedback-v1"`; reject any other.
- `generatedAgainst.dbContentHash` / `.schemaVersion` — the static graph state the payload was
  diffed against. Ingest **should** verify these match the current database and refuse (or re-diff)
  a stale payload, rather than apply edges computed against a different graph.
- `edges[]` — candidate synthesized edges, each in OpenLore `CallEdge` shape
  (`callerId`, `calleeId`, `calleeName`, `file`, `line`, `kind`, `callType`, `confidence`,
  `synthesizedBy`) plus an `evidence` block (`traceAction`, `fixtureSourceSha`, `execCommand`,
  `fidelity`).
- `findings[]` — secondary **architectural-invariant findings**: runtime paths that crossed a
  declared forbidden layer boundary. Each carries the rule name (`invariant`), the offending
  invocation (`callerId`, `calleeId`, `calleeName`, `callerFile`, `calleeFile`), `kind`
  (`"layer-violation"`), `synthesizedBy = 'verisim-runtime'`, and the same `evidence` block as an
  edge. Empty unless invariants were declared. A finding is **not** an edge — it reports a violation
  for human/agent review, never a new call relationship to synthesize — so it is ingested (if at all)
  as a separate concern, never written into the `edges` table.
- `dropped` — count of runtime invocations Verisim could not anchor to a node (informational).

## Validation (reject, never best-effort)

The ingest **must fail closed**, mirroring Verisim's `validate_payload` stand-in:

1. **Version** — exactly `verisim-feedback-v1`, else reject.
2. **Provenance label** — every edge must have `confidence == 'synthesized'` and
   `synthesizedBy == 'verisim-runtime'`. An edge claiming a direct-resolution confidence is an
   over-claim and must be rejected. Every **finding** must likewise carry
   `synthesizedBy == 'verisim-runtime'`.
3. **Node anchoring** — every edge **and finding** `callerId` / `calleeId` must resolve to a
   **known node** by stable id. A reference to an unknown node is rejected (the unknown-node error).
   This is the load-bearing guard: OpenLore only accepts edges and findings it can anchor.
4. **Graph-state match** — `generatedAgainst` should agree with the live graph (see above).

## Additive write (through the existing synthesized-edge path)

Accepted edges are written through OpenLore's **existing** synthesized-edge insertion path, so:

- they appear to every consumer (agents, queries, exports) as `confidence = 'synthesized'` with
  `synthesizedBy = 'verisim-runtime'` — distinguishable from AST-synthesized edges and from
  directly-resolved ones;
- they are **purely additive** — a Verisim edge never modifies or removes a directly-resolved edge,
  and re-ingesting an edge OpenLore already has is a no-op (Verisim already dedups against the
  static graph, including pre-existing synthesized edges, so the payload should not contain
  duplicates, but ingest must be idempotent regardless).

## Rejection rules (summary)

| Condition | Result |
|---|---|
| `version != "verisim-feedback-v1"` | reject whole payload |
| edge `confidence != "synthesized"` or `synthesizedBy != "verisim-runtime"` | reject edge (mislabel) |
| finding `synthesizedBy != "verisim-runtime"` | reject finding (mislabel) |
| edge/finding `callerId` or `calleeId` not a known node | reject (unknown-node) |
| `generatedAgainst` does not match the live graph | reject or re-diff (stale) |
| duplicate of an existing edge | no-op (idempotent) |

## Boundaries

- Verisim **never** writes `call-graph.db`; it writes a payload file. OpenLore owns the write.
- The trace evidence (`fidelity = 'full'` only — Verisim never proposes from a degraded trace) and
  `fixtureSourceSha` let OpenLore and a human audit *why* each edge is claimed before accepting it.
