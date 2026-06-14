# Sandbox runtime-trace oracle

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/trace/` + `tests/test_trace.py` (16 tests
> green; ruff + bare mypy clean). Change 3 of the six is done; Changes 4–6 remain DRAFT.
> **Both tiers ship.** The `full` tier was verified end-to-end against **real `strace` in a Linux
> container** (and CI now installs `strace` so the full-tier e2e runs on every push); the degraded
> tier is verified on the macOS dev host.
> One sentence: **capture what the code actually does at runtime by tracing the real
> `SandboxOracle` execution — exec, syscalls, file mutations, network binds — as typed records,
> so dynamic reality can later correct the static call graph.**
>
> **Implementation notes (the two tiers):**
> - **Full tier — a real ptrace-based `strace` wraps the real subprocess.** `SandboxOracle` gained
>   a small additive `exec_wrapper` seam (default `None`, behavior-identical when unused); the
>   `StraceTracer` installs `strace -f -qq -e trace=… -o <log> --` in front of the *already-confined*
>   rendered argv (a constant trusted prefix — the grammar allowlist is untouched) and parses the
>   log into the real argv + syscall stream. Selected only where `strace` is present and permitted
>   (Linux) **and** the oracle exposes the seam, so a `full` tracer is never an empty-stream
>   pretender. The strace log is harness observability written to a Verisim temp file (like
>   capturing stdout); the v0 command stays confined, as the snapshot/delta proves.
> - **Degraded tier — the always-available floor.** Where privileged tracing is unavailable (the
>   macOS dev host: DTrace needs SIP-disable/entitlements; any host without `strace`), the
>   `DegradedTracer` records exec + file delta + binds post-hoc and tags the trace `degraded`.
> - **File mutations are reused from the oracle's structural delta** (`StepResult.delta`) at both
>   tiers — the proposal's explicit instruction; the full tier *adds* the syscall stream.
> - **Net events** come from observed `connect`/`bind` syscalls at the full tier (expected empty
>   for the v0 grammar — an egress attempt would be a real finding) and are honestly empty at the
>   degraded tier.
> - **Determinism is preserved.** `TracingOracle.step` returns the inner oracle's `StepResult`
>   verbatim; even under real `strace` the canonical state matches an untraced run (verified on
>   Linux). The `exec_wrapper` is installed only for the duration of the step and restored after.

## Why

The findings doc (§3) corrected the original request's central mistake: there is nothing to trace
in an M_θ *imagination* rollout — it is tensor math. The **real** execution surface is
[oracle/sandbox.py](../../../src/verisim/oracle/sandbox.py)'s `SandboxOracle`, which runs a real
`/bin/sh` over a real kernel in a throwaway tree under a `DeterminismSeal` (SPEC-11). That is the
only place real processes run, real files mutate, and real binds happen — exactly the dynamic
facts that static analysis cannot see (dynamic dispatch, indirect calls, runtime-resolved paths,
sockets opened at runtime).

Tracing *there* is honest and on-mission: each `SandboxOracle.step` is already a "reversible
experiment in a vacuum," so a trace is a faithful, reproducible record of one action's real
effects, tied to a known v0 action and a known fixture revision.

## What changes

1. **A tracing wrapper** around `SandboxOracle` (a decorator over the `Oracle` protocol, not a
   fork) that, for each step, records a typed `RuntimeTrace`:
   - process exec events (argv, exit code) and, where the platform supports it, the syscall
     stream;
   - file mutations under the throwaway tree (created/modified/deleted paths, already computed by
     the sandbox's structural delta — reused, not recomputed);
   - network bind/connect attempts;
   - a stable link back to the originating action and the fixture's `FIXTURE.json` source sha.
2. **Platform-honest tracers** behind one interface, selected at runtime:
   - macOS (primary): a native profiler / `ptrace`-style or DTrace-with-fallback driver; if
     privileged tracing is unavailable, a **degraded tracer** that still records exec + file
     delta + binds from observable signals, clearly marked `degraded`.
   - Linux (CI/scale): may use eBPF/strace where available — **never a hard default**.
   The tracer reports its own fidelity tier in every trace.
3. **Determinism preservation** — tracing SHALL NOT change the oracle's `StepResult` or perturb
   the `DeterminismSeal`; a traced step and an untraced step produce identical canonical state.
4. **Trace artifacts** written only to Verisim-owned scratch, typed and versioned, ready for
   Change 4 to diff against the static `CodeGraph`.

## Contract / boundaries

- **Wrapper, not rewrite.** The `SandboxOracle` semantics are unchanged; tracing is additive and
  removable. The existing hermeticity contract (SPEC-11 §2.3) holds.
- **Fidelity is explicit.** Every trace carries its tracer tier (`full` / `degraded`) so
  downstream code never treats a degraded trace as authoritative.
- **No tracing of M_θ.** This change deliberately does not instrument imagination rollouts.

## Risks & honest limits

- Privileged syscall tracing on macOS may require entitlements the dev environment lacks; the
  degraded tier is the always-available floor, and the spec mandates the prototype is *useful*
  (exec + file delta + binds) even at the degraded tier.
- Tracing overhead must not break the sandbox wall-clock timeout budget; the tracer runs within,
  and is accounted against, the existing step budget.
