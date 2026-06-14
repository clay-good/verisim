# trace-oracle spec delta

## ADDED Requirements

### Requirement: RuntimeTraceCapture

The system SHALL capture the real runtime effects of each `SandboxOracle` step as a typed
`RuntimeTrace` containing at minimum process exec events, file mutations under the throwaway
tree, and network bind/connect attempts, each linked to the originating action and the fixture's
source sha. At the `full` fidelity tier the trace SHALL additionally contain the observed syscall
stream. Tracing SHALL attach to the real sandbox execution surface and SHALL NOT instrument
M_θ imagination rollouts.

#### Scenario: A file write and an exec are recorded
- **GIVEN** a fixture action that writes a known file and runs a known command in the sandbox
- **WHEN** the step executes under the tracing oracle
- **THEN** the resulting `RuntimeTrace` contains the file mutation and the exec event, linked to
  the action and the fixture source sha

#### Scenario: The full tier records the real syscall stream
- **GIVEN** a host where a ptrace-based tracer (`strace`) is available and permitted
- **WHEN** a sandbox step that execs a real command is traced at the `full` tier
- **THEN** the trace is tagged `full`, the exec event carries the real argv, and the syscall stream
  contains the `execve` and the file syscalls the command actually issued

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
(`full` or `degraded`) in every trace. The `full` tier SHALL be a ptrace-based tracer (`strace`)
selected only on a host where it is available and permitted **and** when the wrapped oracle exposes
an exec-instrumentation seam; eBPF SHALL NOT be a default. When privileged tracing is unavailable,
a degraded tracer SHALL still record exec events, file mutations, and binds, and SHALL mark the
trace `degraded` so downstream consumers do not treat it as authoritative. Selecting or running the
`full` tier SHALL NOT weaken the sandbox's grammar-allowlist or filesystem-confinement guarantees:
the tracer prepends only a constant, trusted instrumentation prefix to the already-confined rendered
argv, and its own trace log is harness observability (a Verisim-owned artifact), not a write by the
sandboxed command.

#### Scenario: Degraded tier remains useful and labeled
- **GIVEN** a host where privileged syscall tracing is unavailable (e.g. macOS under SIP)
- **WHEN** a step is traced
- **THEN** the trace is tagged `degraded` and still contains exec events, file mutations, and binds

#### Scenario: The full tier is chosen only when it can actually work
- **GIVEN** a tracer selection on a host without a permitted `strace`, or over an oracle with no
  exec-instrumentation seam
- **WHEN** the tracer is selected
- **THEN** the degraded tracer is chosen, never a `full` tracer that would emit an empty syscall
  stream while claiming `full` fidelity
