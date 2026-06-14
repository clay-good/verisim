# Tasks — Sandbox runtime-trace oracle

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/trace/` + `tests/test_trace.py` (16 tests
> green; ruff + bare mypy clean). Change 3 of the six is done. Both tiers ship; the full (`strace`)
> tier verified end-to-end against real strace in a Linux container, and CI installs strace.

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
- [x] Implement the **full** tier via a ptrace-based tracer (`strace`). Added a small additive
      `exec_wrapper` seam to `SandboxOracle` (default `None`); `StraceTracer` wraps the real
      subprocess (`strace -f -qq -e trace=… -o <log> --`), parses the log into the real argv +
      syscall stream, and `TracingOracle` installs/restores the seam per step. Verified e2e against
      real strace in a Linux container (`test_full_tier_with_real_strace`).
      (macOS path: privileged DTrace needs SIP-disable/entitlements the dev host lacks, so macOS
      degrades — the probe detects this. A privileged dtruss tracer is a future extension.)
- [x] Implement the always-available **degraded** tracer (exec + file delta + binds from
      observable signals), tagged `degraded`. (`DegradedTracer`.)
- [x] The full tier uses ptrace-based `strace` (not eBPF — the spec forbids eBPF as a default);
      `select_tracer` picks it only where strace is permitted **and** the oracle exposes the seam.
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
      exec + file delta + binds and is marked `degraded` (`test_degraded_tier_is_useful_and_labeled`);
      the full tier is selected only when it can actually run
      (`test_full_tier_is_chosen_only_when_it_can_actually_work`), the strace parser is pinned
      (`test_strace_output_parses_into_typed_events`), and the full chain is exercised by a fake
      strace on macOS (`test_full_tier_wiring_end_to_end_with_a_fake_strace`) and real strace on
      Linux (`test_full_tier_with_real_strace`).
- [x] Cross-POSIX: degraded tier works on macOS and Linux CI — the suite uses `SandboxOracle`
      (pure POSIX) and `ReferenceOracle` (pure Python); the shell-dependent cases skip, disclosed,
      when no shell is present.
