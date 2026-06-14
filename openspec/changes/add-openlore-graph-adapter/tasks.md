# Tasks — Read-only OpenLore call-graph adapter

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/bridge/` + `tests/test_bridge.py` (16 tests
> green; ruff + bare mypy clean). Change 2 of the six is done.

## 1. Analysis trigger
- [x] Implement a step that runs OpenLore's analyzer over a fixture (CLI/MCP), so the DB is
      OpenLore-authored; capture the resulting graph version / `file_hashes` snapshot.
      (`bridge.analyze_fixture` runs `openlore init` + `analyze --force`; records the OpenLore
      `fingerprint.json` hash and `file_hashes` at load.)
- [x] Assert the analysis produced `.openlore/analysis/call-graph.db` with a supported
      `schema_version`. (`analyze_fixture` loads it through `load_code_graph`, which fails closed.)

## 2. Typed read model
- [x] Define `CodeGraph` types: code node (id, name, file, language, fan_in/out, is_entry_point,
      is_hub), call edge (caller_id, callee_id, line, `confidence`, `kind`, `call_type`,
      `synthesizedBy`), class, optional cfg overlay. (`bridge/graph.py`: `CodeNode`, `CallEdge`,
      `CodeClass`, `CfgEntry`, `CodeGraph`.)
- [x] Implement the MCP/CLI loader path (primary). **Honest scope:** the supported MCP surface
      (`get_call_graph`) returns a *derived aggregate summary*, not the provenance-bearing edge
      set, so it cannot be the full-graph loader; it is implemented as
      `call_graph_summary_via_mcp` and used as the cross-check below. The provenance-bearing graph
      is read from SQLite (the supported "fast path"), which is the only faithful source.
- [x] Implement the read-only SQLite fast path (read-only open), guarded by a `schema_version`
      compatibility check that fails closed on mismatch. (`load_code_graph` + `_open_readonly` +
      `_read_schema_version`; supported range 5–7, column-defensive across in-range schema drift.)
- [x] Preserve all edge provenance fields verbatim (no lossy normalization).
      (`CallEdge.confidence/kind/call_type/synthesized_by` read exactly; tested.)

## 3. Freshness
- [x] Record the graph version / `file_hashes` the `CodeGraph` was read at.
      (`schema_version`, `db_content_hash`, `source_tree_hash`, `fingerprint`, `db_file_hashes`.)
- [x] Implement `is_stale_against(fixture)` comparing recorded hashes to current source hashes.
      (Recomputes the source tree hash, excluding `.git`/caches/`.openlore`.)
- [x] Invalidate / re-load on staleness rather than serving a stale graph. (Any source edit flips
      `is_stale_against` to true; the caller reloads.)

## 4. Safety + verification
- [x] Assert the adapter opens the DB read-only and performs zero writes (verify content hash
      unchanged across a load). (`test_loading_does_not_mutate_the_database`.)
- [x] Schema-mismatch test: an unsupported `schema_version` fails closed with a clear error.
      (`test_unsupported_schema_above_range_fails_closed`, `..._below_range_...`,
      `test_non_openlore_db_fails_closed`.)
- [x] Parity test: MCP-path graph and SQL-path graph agree on counts for a fixture.
      (`test_sql_read_agrees_with_supported_mcp_surface` — asserts the clean invariants:
      internal-node count and entry-point count. The surface's `total_edges` is a derived expanded
      count that diverges from the raw `edges` table and is recorded, not asserted — the finding
      that motivates reading the provenance-bearing graph from SQLite, not the summary.)
- [x] Provenance test: a known `synthesized` edge round-trips with its `synthesizedBy` rule intact.
      (`test_synthesized_edge_provenance_is_preserved`,
      `test_all_edge_provenance_fields_preserved_verbatim`.)
- [x] Staleness test: re-analyzing the fixture flips `is_stale_against` to true and forces reload.
      (`test_modifying_the_source_invalidates_the_graph`,
      `test_staleness_ignores_openlore_cache`, `test_staleness_without_source_anchor_raises`.)
