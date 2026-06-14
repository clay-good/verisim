# Tasks — Sandbox runtime-trace oracle

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/trace/` + `tests/test_trace.py` (11 tests
> green; ruff + bare mypy clean). Change 3 of the six is done.

## 1. Trace model
- [x] Define `RuntimeTrace` (action id, fixture source sha, exec events, optional syscall stream,
      file mutations, net bind/connect events, tracer fidelity tier, timing).
      (`trace/model.py`: `RuntimeTrace` + `ExecEvent`/`FileMutation`/`NetEvent`, `schema_version`,
      `elapsed_s`, `to_json`. The optional full-tier syscall stream is represented by the fidelity
      tier; the degraded floor records exec/file/net.)
- [x] Define the `Tracer` interface (start/stop around a step, emit `RuntimeTrace`).
      (`trace/tracer.py`: `Tracer` protocol with `begin`/`finish`.)

## 2. Tracing wrapper
- [x] Implement a `TracingOracle` decorator over the `Oracle` protocol that wraps any
      `SandboxOracle.step`, reusing the sandbox's existing structural delta for file mutations.
      (`trace/oracle.py`; file mutations projected from `StepResult.delta`.)
- [x] Tie each trace to the originating action and `FIXTURE.json` source sha.
      (`action_name`/`action_args` + `fixture_source_sha` constructor arg.)

## 3. Platform tracers
- [ ] ~~Implement the macOS path (native profiler / ptrace / DTrace)~~ — **out of scope, disclosed.**
      Not achievable as a *pure decorator* (cannot instrument the sandbox's internal subprocess
      spawn) and unavailable on the SIP-locked dev host. `full_tracing_available()` returns `False`
      with the reason; the degraded floor is shipped instead (the spec's mandated tier).
- [x] Implement the always-available **degraded** tracer (exec + file delta + binds from
      observable signals), tagged `degraded`. (`DegradedTracer`.)
- [ ] (Scale) Optional Linux eBPF/strace path — not implemented; the spec forbids it as a default
      and the pure-decorator architecture precludes wiring it here.
- [x] Make every trace self-report its fidelity tier. (`RuntimeTrace.fidelity`, `is_degraded()`.)

## 4. Determinism + budget
- [x] Prove tracing does not alter `StepResult`: traced vs untraced step produce identical
      canonical state (golden test). (`test_traced_and_untraced_steps_agree` on a real-shell
      sequence; `test_traced_result_is_inner_result_verbatim` asserts the same object is returned.)
- [x] Prove tracing does not perturb the `DeterminismSeal` — the wrapper never touches env/umask/
      rlimits; it returns the inner `StepResult` verbatim, so the seal is untouched by construction.
- [x] Account tracer overhead within the existing step wall-clock budget; fail the step rather
      than overrun silently. (`overhead_budget_s` → `TraceBudgetExceeded`;
      `test_tracing_overhead_budget_fails_loudly`.)

## 5. Artifacts + verification
- [x] Write traces to Verisim-owned scratch only; typed + versioned. (`write_trace` refuses any
      path under the source-roots allowlist; files carry `schema_version`.)
- [x] Golden trace test on a fixture action that performs a known file write + a known exec.
      (`test_file_write_and_exec_are_recorded_in_the_sandbox`.)
- [x] Fidelity-tier test: when privileged tracing is unavailable, the degraded tier still records
      exec + file delta + binds and is marked `degraded`. (`test_degraded_tier_is_useful_and_labeled`.)
- [x] Cross-POSIX: degraded tier works on macOS and Linux CI — the suite uses `SandboxOracle`
      (pure POSIX) and `ReferenceOracle` (pure Python); the shell-dependent cases skip, disclosed,
      when no shell is present.
