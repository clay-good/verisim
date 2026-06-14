# Read-only OpenLore call-graph adapter

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/bridge/` + `tests/test_bridge.py` (16 tests
> green; ruff + bare mypy clean). Change 2 of the six is done; Changes 3–6 remain DRAFT.
> One sentence: **let Verisim read OpenLore's static call graph for a fixture as a typed,
> read-only substrate it can ground a world model and architectural invariants on, treating the
> database as the regenerable cache it actually is.**
>
> **Implementation notes (two honest findings only running the real CLI could surface):**
> 1. **The supported MCP surface is a *derived summary*, not the call graph.** OpenLore's
>    `get_call_graph` returns aggregates (internal-node/edge counts, entry points, hubs), not the
>    provenance-bearing edge set — and its `total_edges` is an *expanded* count that diverges from
>    the raw `edges` table (e.g. 1361 vs 723 on a 36-file fixture). So the **read-only SQLite open
>    is the canonical loader** (the only faithful source of per-edge `confidence`/`synthesized_by`),
>    and the MCP surface is used as an independent **cross-check** on the invariants that *are*
>    clean (internal-node count, entry-point count). This divergence is exactly the schema-drift
>    hazard the schema guard exists for: the graph must be read, not reconstructed from a summary.
> 2. **The schema drifts *within* the supported range.** The installed CLI authors `schema_version`
>    5; older committed analyses are v7, and `edges.synthesized_by` exists in v7 but not v5. The
>    loader is therefore **column-defensive** across the supported 5–7 range (a known in-range
>    column absent degrades to its default) while still failing closed on an *unknown* version.
>    `file_hashes` is populated only on some runs, so staleness anchors on a Verisim-computed source
>    tree hash, not the DB's own `file_hashes`.

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
