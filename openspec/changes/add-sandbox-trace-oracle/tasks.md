# Tasks — Sandbox runtime-trace oracle

## 1. Trace model
- [ ] Define `RuntimeTrace` (action id, fixture source sha, exec events, optional syscall stream,
      file mutations, net bind/connect events, tracer fidelity tier, timing).
- [ ] Define the `Tracer` interface (start/stop around a step, emit `RuntimeTrace`).

## 2. Tracing wrapper
- [ ] Implement a `TracingOracle` decorator over the `Oracle` protocol that wraps any
      `SandboxOracle.step`, reusing the sandbox's existing structural delta for file mutations.
- [ ] Tie each trace to the originating action and `FIXTURE.json` source sha.

## 3. Platform tracers
- [ ] Implement the macOS path (native profiler / ptrace / DTrace) with a clean
      capability probe.
- [ ] Implement the always-available **degraded** tracer (exec + file delta + binds from
      observable signals), tagged `degraded`.
- [ ] (Scale) Implement an optional Linux eBPF/strace path — never selected by default.
- [ ] Make every trace self-report its fidelity tier.

## 4. Determinism + budget
- [ ] Prove tracing does not alter `StepResult`: traced vs untraced step produce identical
      canonical state (golden test).
- [ ] Prove tracing does not perturb the `DeterminismSeal` (env scrub, umask, rlimits unchanged).
- [ ] Account tracer overhead within the existing step wall-clock budget; fail the step rather
      than overrun silently.

## 5. Artifacts + verification
- [ ] Write traces to Verisim-owned scratch only; typed + versioned.
- [ ] Golden trace test on a fixture action that performs a known file write + a known exec.
- [ ] Fidelity-tier test: when privileged tracing is unavailable, the degraded tier still records
      exec + file delta + binds and is marked `degraded`.
- [ ] Cross-POSIX: degraded tier works on macOS and Linux CI.
