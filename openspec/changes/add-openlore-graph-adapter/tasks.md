# Tasks — Read-only OpenLore call-graph adapter

## 1. Analysis trigger
- [ ] Implement a step that runs OpenLore's analyzer over a fixture (CLI/MCP), so the DB is
      OpenLore-authored; capture the resulting graph version / `file_hashes` snapshot.
- [ ] Assert the analysis produced `.openlore/analysis/call-graph.db` with a supported
      `schema_version`.

## 2. Typed read model
- [ ] Define `CodeGraph` types: code node (id, name, file, language, fan_in/out, is_entry_point,
      is_hub), call edge (caller_id, callee_id, line, `confidence`, `kind`, `call_type`,
      `synthesizedBy`), class, optional cfg overlay.
- [ ] Implement the MCP/CLI loader path (primary).
- [ ] Implement the read-only SQLite fast path (WAL read or snapshot copy), guarded by a
      `schema_version` compatibility check that fails closed on mismatch.
- [ ] Preserve all edge provenance fields verbatim (no lossy normalization).

## 3. Freshness
- [ ] Record the graph version / `file_hashes` the `CodeGraph` was read at.
- [ ] Implement `is_stale_against(fixture)` comparing recorded hashes to current source hashes.
- [ ] Invalidate / re-load on staleness rather than serving a stale graph.

## 4. Safety + verification
- [ ] Assert the adapter opens the DB read-only and performs zero writes (verify file mtime/hash
      unchanged across a load).
- [ ] Schema-mismatch test: an unsupported `schema_version` fails closed with a clear error.
- [ ] Parity test: MCP-path graph and SQL-path graph agree on node/edge counts for a fixture.
- [ ] Provenance test: a known `synthesized` edge in the fixture round-trips with its
      `synthesizedBy` rule intact.
- [ ] Staleness test: re-analyzing the fixture flips `is_stale_against` to true and forces reload.
