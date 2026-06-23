# ADR-0006: Fold the distributed world (ED1) into the cross-world floor+cliff synthesis as a fourth world

## Status

accepted

**Domains**: experiments, docs

## Context

The cross-world synthesis (§22) claimed the floor+cliff shape across three worlds (filesystem/network/host). Adding ED1's distributed world makes the claim strongest: the shape survives even in the one world where bit-exact global truth is intractable (NP-complete consistency checking), measured against a tiered cost-bounded oracle rather than a cheap exact one — so the shape is not an artifact of having an exact oracle. world_curve() is made schema-aware (detects ED1 panel schema, reads panel=='curve' rows) rather than emitting a second synthesis-shaped CSV, keeping one DEFAULT_WORLDS map and one reader.

## Decision

The system SHALL include the distributed world (ED1) as a fourth world in the cross-world floor+cliff synthesis, reading its panel-schema curve alongside the three existing worlds.

## Consequences

synthesis.py reads two curve schemas; DEFAULT_WORLDS gains 'distributed (ED1)'; committed synthesis figure/CSV are now four worlds; README §22 + intro + sibling-thesis tiles and docs/report.md updated from three to four worlds. No change to ED1 itself or the other three worlds' curves.

> Recorded by openlore decisions on 2026-06-23
> Decision ID: f2816710
