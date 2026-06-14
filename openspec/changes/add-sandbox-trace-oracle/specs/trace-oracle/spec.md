# trace-oracle spec delta

## ADDED Requirements

### Requirement: RuntimeTraceCapture

The system SHALL capture the real runtime effects of each `SandboxOracle` step as a typed
`RuntimeTrace` containing at minimum process exec events, file mutations under the throwaway
tree, and network bind/connect attempts, each linked to the originating action and the fixture's
source sha. Tracing SHALL attach to the real sandbox execution surface and SHALL NOT instrument
M_θ imagination rollouts.

#### Scenario: A file write and an exec are recorded
- **GIVEN** a fixture action that writes a known file and runs a known command in the sandbox
- **WHEN** the step executes under the tracing oracle
- **THEN** the resulting `RuntimeTrace` contains the file mutation and the exec event, linked to
  the action and the fixture source sha

### Requirement: DeterminismPreservedUnderTracing

The system SHALL ensure tracing is observationally pure with respect to the oracle: a traced step
and an untraced step SHALL produce an identical canonical `StepResult`, and tracing SHALL NOT
alter the `DeterminismSeal` (environment scrub, umask, resource limits).

#### Scenario: Traced and untraced steps agree
- **GIVEN** a fixed fixture state and a fixed action
- **WHEN** the action is executed once with tracing and once without
- **THEN** the two canonical `StepResult`s are identical

### Requirement: ExplicitTracerFidelity

The system SHALL select a platform-appropriate tracer at runtime and SHALL record a fidelity tier
(`full` or `degraded`) in every trace. eBPF SHALL NOT be a default. When privileged tracing is
unavailable, a degraded tracer SHALL still record exec events, file mutations, and binds, and
SHALL mark the trace `degraded` so downstream consumers do not treat it as authoritative.

#### Scenario: Degraded tier remains useful and labeled
- **GIVEN** a host where privileged syscall tracing is unavailable
- **WHEN** a step is traced
- **THEN** the trace is tagged `degraded` and still contains exec events, file mutations, and binds
