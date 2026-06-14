# Sandbox runtime-trace oracle

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/trace/` + `tests/test_trace.py` (11 tests
> green; ruff + bare mypy clean). Change 3 of the six is done; Changes 4–6 remain DRAFT.
> One sentence: **capture what the code actually does at runtime by tracing the real
> `SandboxOracle` execution — exec, syscalls, file mutations, network binds — as typed records,
> so dynamic reality can later correct the static call graph.**
>
> **Implementation notes (honest scope of the tiers):**
> - **Degraded tier is what ships, and it is the spec's mandated floor — by design, not omission.**
>   The tracer is a *pure decorator* over the `Oracle` protocol (the "wrapper, not rewrite"
>   constraint), so it cannot inject `ptrace`/DTrace into the subprocess `SandboxOracle` spawns
>   *inside* its own `step`; a `full`-tier syscall tracer would have to instrument that spawn. On
>   the macOS dev host, privileged process tracing also needs SIP-disable/entitlements the
>   environment lacks (the proposal's stated risk). So `full_tracing_available()` honestly returns
>   `False` and `select_tracer()` returns the degraded tracer, which still records the floor the
>   spec requires: **exec + file delta + binds**, every trace tagged `degraded`.
> - **File mutations are reused from the oracle's structural delta** (`StepResult.delta`), not
>   recomputed — the proposal's explicit instruction.
> - **Net events are honestly empty.** The v0 filesystem grammar exposes no network action and the
>   sandbox blocks egress by allowlist, so binds are `()` — recorded, not omitted, so a future
>   grammar with a network action has a place to land.
> - **Determinism is preserved by construction.** `TracingOracle.step` returns the inner oracle's
>   `StepResult` verbatim and builds the trace purely from the action + result, touching neither the
>   execution nor the `DeterminismSeal`; a traced step is bit-identical to an untraced one (tested
>   on a real-shell sequence).

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
