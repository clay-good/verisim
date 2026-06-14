# Headless CD pipeline (human-gated delivery)

> Status: DRAFT — proposal + spec delta. No code yet.
> One sentence: **chain the whole stack behind one local, network-isolated headless entry point —
> intent → simulate → trace → evaluate invariants → oscillation check → prepare an unpushed
> commit/patch + report — and stop for a human to confirm delivery.**

## Why

This is the deliverable that proves the research (findings §9.1: *a working CD prototype*). The
original request wanted a fully autonomous loop that commits and deploys with the user's real git
credentials. The requester chose **human-gated everywhere** (findings §4 resolution), and the
findings doc (§5) showed an unattended deployer is irreversible, fights the existing commit gate,
and overreaches for a research repo. So the prototype runs the full pipeline *up to* the
irreversible step and then **prepares and recommends**, leaving the final commit/push/deploy as a
human action.

The value is end-to-end: it demonstrates that a static call-graph and a dynamic oracle, unified
through a safe contract, can drive a closed loop from intent to a vetted, ready-to-deliver change
— locally, deterministically, and network-isolated.

## What changes

A single headless entry point (a `console_scripts`-style command + a callable API) that runs, in
order, against a fixture (Change 1):

1. **Intent graph construction** — load the read-only `CodeGraph` (Change 2) and the declared
   architectural invariants; build the run's intent (what change/goal is being attempted) and the
   set of invariants to evaluate.
2. **Speculative rollout simulation** — run the loop's M_θ imagination rollout for the intent
   (existing `loop/speculative.py` machinery).
3. **Trace runtime side-effects** — execute the plan's verifying steps through the tracing
   `SandboxOracle` (Change 3), producing `RuntimeTrace`s.
4. **Evaluate architectural invariants** — check the runtime traces + the (statically known +
   runtime-discovered) graph against the declared invariants; produce findings.
5. **Feedback** — emit the `verisim-feedback-v1` payload for any runtime-discovered edges
   (Change 4) for OpenLore to ingest on its own terms.
6. **Intercept oscillation loops** — the breaker (Change 5) runs throughout; a `critical` trip
   freezes the pipeline and surfaces a human-gated rollback recommendation.
7. **Prepare delivery (human-gated)** — assemble a **signed run report** + a **prepared, unpushed
   commit/patch** on the fixture. The pipeline **stops here**: commit/push/deploy require explicit
   human confirmation, and even on confirmation operate only on the de-fanged fixture (which
   cannot push to the original).

## Execution contract

- **Local + network-isolated.** No cloud model lookups; the run asserts no outbound network
  (beyond loopback) is required, consistent with the simulator's offline discipline.
- **Deterministic + resumable.** Given a fixed fixture revision + intent + seed, stages are
  reproducible; each stage writes a typed artifact so the run is inspectable and re-runnable.
- **Honors the commit gate.** Any prepared commit respects Verisim's existing local commit gate;
  the pipeline never bypasses it ([memory: commit-gate]).
- **Fail-safe ordering.** A failed/halted earlier stage prevents later stages; nothing
  irreversible is ever reached without passing every prior gate and a human confirmation.

## Risks & honest limits

- "Deliver to the target environment" in this prototype means *prepare on the de-fanged fixture
  and stop for confirmation* — it deliberately cannot reach the real repo/remotes. Promoting to a
  real target is a separate, later, explicitly-authorized change.
- End-to-end depends on Changes 1–5; the pipeline spec assumes their requirements hold and tests
  the *composition* (ordering, gating, artifacts), not their internals.
