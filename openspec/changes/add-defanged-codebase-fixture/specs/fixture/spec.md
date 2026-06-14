# fixture spec delta

## ADDED Requirements

### Requirement: IsolatedFixtureMaterialization

The system SHALL materialize a real source repository into an isolated, Verisim-owned working
copy without modifying the source. The materializer SHALL accept a source path, SHALL reject any
source path not inside a configured roots allowlist, and SHALL place the copy under a gitignored
scratch root that is never inside the source tree nor inside any `.openlore/` directory. The copy
SHALL be byte-deterministic for a fixed `(source, options)` pair (excluding volatile `.git`
internals), and the materializer SHALL fail loudly rather than produce an incomplete fixture.

#### Scenario: Source outside the allowlist is rejected
- **GIVEN** a source path that is not under any configured root
- **WHEN** materialization is requested
- **THEN** the request fails with an explicit allowlist error and no files are copied

#### Scenario: Materialization leaves the source untouched
- **GIVEN** a source repo and a content hash of its tree
- **WHEN** the repo is materialized, used, and torn down
- **THEN** the source tree's content hash is byte-identical to the pre-materialization hash

#### Scenario: Deterministic copy
- **GIVEN** a fixed source repo and fixed options
- **WHEN** the repo is materialized twice into fresh scratch locations
- **THEN** the two copies have identical file sets and identical per-file content hashes

### Requirement: GitDefanging

The system SHALL neutralize the materialized copy's `.git` so the prototype cannot commit to the
original's history or push to any remote. De-fanging SHALL, by construction, (1) remove all
remotes, (2) set a sentinel fixture identity, (3) install a `pre-push` hook that always fails,
and (4) record the actions in a `FIXTURE.json` manifest carrying the source path and the source
HEAD sha at copy time. A push attempt inside a fixture SHALL fail.

#### Scenario: No remote remains
- **GIVEN** a freshly de-fanged fixture
- **WHEN** the fixture's configured remotes are listed
- **THEN** there are zero remotes

#### Scenario: Push is structurally blocked
- **GIVEN** a de-fanged fixture
- **WHEN** a `git push` is attempted from inside it
- **THEN** the operation fails (no remote to resolve and the pre-push hook exits non-zero)

#### Scenario: Manifest is traceable to a source revision
- **GIVEN** a materialized fixture
- **WHEN** `FIXTURE.json` is read
- **THEN** its recorded source HEAD sha equals the source repo's HEAD at copy time

### Requirement: FixtureTeardown

The system SHALL provide teardown that removes a fixture in full and SHALL verify, as part of the
prototype's safety tests, that a complete materialize→use→teardown cycle leaves the source repo
unchanged.

#### Scenario: Teardown removes the fixture
- **GIVEN** an existing fixture directory
- **WHEN** teardown runs
- **THEN** the fixture directory no longer exists and no Verisim-owned scratch state for it remains
