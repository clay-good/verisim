# safety-breaker spec delta

## ADDED Requirements

### Requirement: OscillationMetric

The system SHALL compute an oscillation metric over the planning loop's own state-transition
stream — repeated state/edit bigrams divided by total transitions in a sliding window — together
with a repetitive-file-modification counter. The metric SHALL be a pure, deterministic function
of the transition stream.

#### Scenario: Oscillating trajectory scores high
- **GIVEN** a transition stream that flip-flops between two states (A→B→A→B…)
- **WHEN** the metric is computed
- **THEN** the oscillation score is high enough to reach the `critical` tier

#### Scenario: Converging trajectory scores low
- **GIVEN** a transition stream of distinct, progressing states
- **WHEN** the metric is computed
- **THEN** the oscillation score stays in the `ok` tier

### Requirement: DetectAndHaltBreaker

The system SHALL define tiers `ok` / `degraded` / `critical` with explicit thresholds, and on
reaching `critical` SHALL freeze the loop and drop the in-memory speculative rollout. These
actions SHALL be automatic because they are internal and reversible. The breaker SHALL NOT
perform any git or filesystem mutation as part of tripping.

#### Scenario: Critical trips the safe automatic actions only
- **GIVEN** a run that reaches the `critical` tier
- **WHEN** the breaker fires
- **THEN** the loop is frozen and the speculative rollout is dropped, and no git/filesystem
  mutation has occurred

### Requirement: HumanGatedRollback

The system SHALL, on a breaker trip, compute a `RollbackRecommendation` (target known-good
baseline, diff preview, reason) and SHALL require explicit human confirmation before executing
any rollback. Any executed rollback SHALL operate only on the fixture, SHALL snapshot first so no
uncommitted work is lost, and SHALL leave the original source repository untouched.

#### Scenario: Rollback is recommended, not executed
- **GIVEN** a breaker trip
- **WHEN** the recommendation is produced and no human confirmation is given
- **THEN** a `RollbackRecommendation` exists and the fixture working tree is unchanged

#### Scenario: Confirmed rollback is safe and scoped
- **GIVEN** a breaker trip and explicit human confirmation
- **WHEN** the rollback executes
- **THEN** the fixture is restored to the baseline with a pre-rollback snapshot retained, and the
  original source repository is byte-identical to before the run
