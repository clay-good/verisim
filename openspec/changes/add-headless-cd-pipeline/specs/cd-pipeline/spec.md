# cd-pipeline spec delta

## ADDED Requirements

### Requirement: HeadlessOrderedPipeline

The system SHALL provide a headless entry point that runs, in order, against a fixture: intent
graph construction, speculative rollout simulation, runtime side-effect tracing, architectural
invariant evaluation, feedback payload emission, oscillation interception, and delivery
preparation. Each stage SHALL write a typed artifact, and the run SHALL be deterministic for a
fixed (fixture revision, intent, seed).

#### Scenario: Full run produces a prepared, undelivered change
- **GIVEN** a fixture and a benign intent
- **WHEN** the pipeline runs to completion
- **THEN** it produces a signed run report and a prepared, unpushed commit/patch on the fixture,
  and no commit, push, or deploy has been performed

#### Scenario: Deterministic stages
- **GIVEN** a fixed fixture revision, intent, and seed
- **WHEN** the pipeline is run twice
- **THEN** the per-stage artifacts are identical

### Requirement: FailSafeGating

The system SHALL enforce fail-safe ordering: a failed or halted stage SHALL block all later
stages, and the irreversible delivery step SHALL be reachable only after every prior stage passes
and a human explicitly confirms. A `critical` oscillation-breaker trip SHALL halt the run before
delivery.

#### Scenario: Breaker trip halts before delivery
- **GIVEN** a run whose trajectory reaches the `critical` tier
- **WHEN** the breaker fires
- **THEN** the pipeline halts before the delivery-preparation stage and surfaces a human-gated
  rollback recommendation, with no commit/push/deploy performed

#### Scenario: Delivery requires explicit confirmation
- **GIVEN** a completed run with a prepared commit/patch
- **WHEN** no human confirmation is provided
- **THEN** no commit, push, or deploy occurs

### Requirement: LocalNetworkIsolatedExecution

The system SHALL execute the pipeline locally with no cloud model lookups and no required
outbound network beyond loopback, SHALL operate only on the de-fanged fixture, and SHALL honor
Verisim's existing local commit gate without bypassing it.

#### Scenario: No outbound network and source untouched
- **GIVEN** a pipeline run
- **WHEN** the run completes
- **THEN** it required no outbound network beyond loopback and the original source repository is
  byte-identical to before the run

#### Scenario: Even a confirmed commit cannot reach the original
- **GIVEN** a completed run and explicit human confirmation to commit
- **WHEN** the commit (and any attempted push) executes
- **THEN** the commit lands only on the de-fanged fixture and any push fails because the fixture
  has no remote and a blocking pre-push hook
