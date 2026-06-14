# Read-only OpenLore call-graph adapter

> Status: DRAFT — proposal + spec delta. No code yet.
> One sentence: **let Verisim read OpenLore's static call graph for a fixture as a typed,
> read-only substrate it can ground a world model and architectural invariants on, treating the
> database as the regenerable cache it actually is.**

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
