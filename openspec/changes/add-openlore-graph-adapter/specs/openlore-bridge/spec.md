# openlore-bridge spec delta (read side)

## ADDED Requirements

### Requirement: ReadOnlyCallGraphAccess

The system SHALL load an OpenLore call graph for a fixture into a typed in-memory `CodeGraph`
without ever writing OpenLore's database. Access SHALL be obtained via OpenLore's MCP/CLI surface
or via a read-only SQLite open, and the database file SHALL be unchanged (identical content hash)
across any load. The loader SHALL preserve every edge's provenance fields
(`confidence`, `kind`, `call_type`, `synthesizedBy`) without lossy normalization.

#### Scenario: Loading does not mutate the database
- **GIVEN** a fixture with an OpenLore-authored `call-graph.db` and its content hash
- **WHEN** Verisim loads the call graph
- **THEN** the database file's content hash is unchanged after the load

#### Scenario: Synthesized-edge provenance is preserved
- **GIVEN** a fixture whose graph contains an edge with `confidence = 'synthesized'` and a
  `synthesizedBy` rule name
- **WHEN** the graph is loaded into `CodeGraph`
- **THEN** that edge retains `confidence = 'synthesized'` and the same `synthesizedBy` value

### Requirement: SchemaVersionGuard

The system SHALL check the database `schema_version` against a supported range before reading via
the SQL fast path and SHALL fail closed with an explicit, actionable error on mismatch rather
than attempt a best-effort parse.

#### Scenario: Unsupported schema fails closed
- **GIVEN** a `call-graph.db` whose `schema_version` is outside the supported range
- **WHEN** the SQL fast path is used
- **THEN** the load fails with an explicit version-mismatch error and returns no partial graph

### Requirement: CacheStalenessAwareness

The system SHALL record the graph version / source `file_hashes` a `CodeGraph` was read at and
SHALL expose a staleness check against the current fixture, so a regenerated database invalidates
the cached graph instead of serving stale structure.

#### Scenario: Re-analysis invalidates the cached graph
- **GIVEN** a loaded `CodeGraph` and its recorded source hashes
- **WHEN** the fixture is modified and re-analyzed by OpenLore
- **THEN** the staleness check reports stale and a subsequent access reloads the graph
