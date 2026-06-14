# openlore-bridge spec delta (write side / feedback contract)

## ADDED Requirements

### Requirement: RuntimeDiscrepancyDetection

The system SHALL diff each `RuntimeTrace` against the static `CodeGraph` and SHALL classify a
runtime call that has no corresponding static edge as a candidate synthesized edge. The detector
SHALL deduplicate candidates against the static graph, including edges already labeled
`confidence = 'synthesized'`, so an edge OpenLore already has is never re-proposed.

#### Scenario: A runtime-only dynamic dispatch becomes a candidate
- **GIVEN** a fixture call that fires at runtime via dynamic dispatch and has no static edge
- **WHEN** the trace is diffed against the static graph
- **THEN** exactly one candidate synthesized edge is produced for that call

#### Scenario: An already-known edge is not re-proposed
- **GIVEN** a runtime call that already has a corresponding edge in the static graph
- **WHEN** the trace is diffed
- **THEN** no candidate edge is produced for that call

### Requirement: ContractConformantFeedbackPayload

The system SHALL emit candidate edges as a versioned `verisim-feedback-v1` JSON payload, never as
a direct write to OpenLore's database. Each candidate edge SHALL carry the provenance OpenLore's
`CallEdge` requires — including `confidence = 'synthesized'` and `synthesizedBy = 'verisim-runtime'`
— plus the runtime trace id and fixture source sha that justify it. The payload SHALL be
idempotent for a fixed set of traces and graph.

#### Scenario: Verisim does not write the database
- **GIVEN** a set of candidate edges and the fixture's `call-graph.db` content hash
- **WHEN** feedback is produced
- **THEN** a payload file is written, and the database content hash is unchanged

#### Scenario: Every candidate is evidence-bearing and labeled
- **GIVEN** an emitted payload
- **WHEN** any candidate edge is inspected
- **THEN** it has `confidence = 'synthesized'`, `synthesizedBy = 'verisim-runtime'`, and a
  resolvable runtime trace id and fixture source sha

### Requirement: NodeResolutionFailClosed

The system SHALL resolve each candidate's call/callee sites to OpenLore node ids and SHALL drop
any candidate that does not resolve to a known node, preferring precision over recall. Dropped
candidates SHALL be counted, not silently discarded.

#### Scenario: Unresolvable candidate is dropped
- **GIVEN** a runtime call whose endpoints do not map to any node in the static graph
- **WHEN** the payload is produced
- **THEN** no candidate edge for that call appears in the payload and the drop is counted

### Requirement: PayloadValidation

The system SHALL provide a validator that checks a `verisim-feedback-v1` payload against its
schema and against a fixture's node set, standing in for OpenLore's ingestion during prototype
tests, and SHALL reject payloads that are malformed, mislabeled, or reference unknown nodes.

#### Scenario: Over-claiming payload is rejected
- **GIVEN** a payload whose edge references a node id not present in the fixture graph
- **WHEN** the validator runs
- **THEN** validation fails with an explicit unknown-node error
