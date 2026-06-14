# Read-only OpenLore call-graph adapter

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/bridge/` + `tests/test_bridge.py` (17 tests
> green; ruff + bare mypy clean). Change 2 of the six is done; Changes 3–6 remain DRAFT.
> Verified against OpenLore **2.0.18** (latest).
> One sentence: **let Verisim read OpenLore's static call graph for a fixture as a typed,
> read-only substrate it can ground a world model and architectural invariants on, treating the
> database as the regenerable cache it actually is.**
>
> **The MCP surface, settled on the latest OpenLore (2.0.18, ran e2e to confirm).** The MCP contract
> has *two* faces, and the cross-check uses both:
> - `get_subgraph` is the **real call graph** — actual per-edge `caller`/`callee`/`kind`/`callType`
>   relationships. It reads an in-memory graph index that an OpenLore *version upgrade resets*, so a
>   query must first **(re)build the index e2e** via the `analyze_codebase` tool in the same MCP
>   session (the "run it to generate" step). The strong parity test confirms these real MCP edges
>   **equal** the SQLite read's internal edges, exactly (`subgraph_via_mcp`).
> - `get_call_graph` is an **aggregate summary** (counts/hubs/entry-points), a *derived* view whose
>   `total_edges` is an expanded count (1496 vs raw 840 on a 36-file fixture), used for the cheap
>   count-level invariants.
>
> **Two honest findings that hold even on the latest OpenLore:**
> 1. **Per-edge provenance lives only in SQLite.** Neither MCP face carries `confidence` /
>    `synthesized_by` — `get_subgraph` omits them, `get_call_graph` is an aggregate. So the
>    **read-only SQLite open is the canonical loader** (the only faithful source of the provenance
>    Change 4 needs), and the MCP surface is a cross-check, not a replacement. This is exactly the
>    schema-drift hazard the schema guard exists for: the graph is read, not reconstructed.
> 2. **The schema drifts *within* the supported range.** OpenLore 2.0.15 authored `schema_version`
>    5; 2.0.18 (and older committed analyses) author 7, and `edges.synthesized_by` exists in v7 but
>    not v5. The loader is therefore **column-defensive** across the supported 5–7 range (a known
>    in-range column absent degrades to its default) while still failing closed on an *unknown*
>    version. `file_hashes` is populated only on some runs, so staleness anchors on a
>    Verisim-computed source tree hash, not the DB's own `file_hashes`.

## Why

OpenLore already produces the exact artifact the prototype needs: a deterministic, locally
computed **static call graph** of a real codebase, in `.openlore/analysis/call-graph.db`
(tables `nodes`, `edges`, `classes`, `cfg_overlay`, `decisions`, …). The research claim is that
**runtime reality can correct this static picture**; to do that, Verisim must first *read* the
static picture faithfully.

The findings doc (§2, §6) established two hard constraints this change must obey:

- **The DB is a cache, not a store of record** — `.openlore/` is gitignored and rebuilt on
  analyze. Verisim must treat it as **read-only and ephemeral**, re-deriving whenever OpenLore
  regenerates it, and must never write it.
- **The supported contract is the MCP tool surface** (`get_call_graph`, `get_subgraph`,
  `find_path`, `analyze_impact`, `search_code`, …); raw SQL is an internal, versioned schema
  (`schema_version`). The adapter SHOULD prefer the MCP/CLI surface and treat direct read-only
  SQL as a fast path guarded by a schema-version check.

## What changes

1. **Fixture analysis trigger** — ensure the fixture (Change 1) has a fresh OpenLore analysis,
   invoking OpenLore's own analyzer (CLI/MCP) so the DB is OpenLore-authored, not Verisim-authored.
2. **A read-only graph adapter** that loads the call graph into a typed, in-memory Verisim
   structure (`CodeGraph`: code nodes, call edges with `confidence`/`kind`/`synthesizedBy`
   provenance, classes, and the cfg overlay if present). Access is read-only and obtained via:
   - primary: OpenLore MCP/CLI tool output, or
   - fast path: a read-only SQLite open (WAL or snapshot copy) **gated by a
     `schema_version` compatibility check** that fails closed on mismatch.
3. **Provenance preservation** — every edge keeps its `confidence` label
   (`import`/`type_inference`/`same_file`/`name_only`/`synthesized`) and `synthesizedBy` rule, so
   downstream code (Change 4) can distinguish "directly resolved" from "already synthesized" and
   never proposes an edge OpenLore already has.
4. **Cache-staleness awareness** — the adapter records the DB's `file_hashes`/graph version it
   read at, and exposes a `is_stale_against(fixture)` check so a re-analysis invalidates a cached
   `CodeGraph` rather than silently serving a stale graph (the honest analogue of OpenLore's own
   epistemic-lease freshness, used *read-only*, not wired into OpenLore's TS lease).

## Contract / boundaries

- **One-way.** The adapter opens the DB read-only (and/or reads MCP output). It performs no
  writes, no schema migration, no `PRAGMA` that mutates the file.
- **Schema-pinned.** A `schema_version` outside the supported range fails closed with a clear
  message ("re-run OpenLore analyze / adapter needs update"), never a best-effort parse.
- **No mapping back into OpenLore's node space here.** This change only *reads*. The grounding of
  Verisim's simulated assets *onto* code structure is a Verisim-internal projection
  (consumed by Changes 3–4), never a write into OpenLore's `nodes`.

## Risks & honest limits

- The MCP path depends on OpenLore being reachable as an MCP server; the SQL fast path depends on
  schema stability. The adapter supports both and degrades to a clear error, not silent wrong data.
- A fixture in a language OpenLore analyzes weakly yields a sparse graph; that is a property of
  the subject, surfaced in adapter stats, not hidden.
