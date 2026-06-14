# Tasks — Headless CD pipeline

## 1. Entry point + run model
- [ ] Define a headless entry point (console-script command + callable API) taking a fixture +
      intent + seed.
- [ ] Define a `RunReport` artifact aggregating every stage's typed output.

## 2. Stage wiring (in order)
- [ ] Stage 1 — Intent graph: load read-only `CodeGraph` (Change 2) + declared invariants; build intent.
- [ ] Stage 2 — Speculative rollout: run M_θ imagination for the intent (existing machinery).
- [ ] Stage 3 — Trace side-effects: execute verifying steps via the tracing `SandboxOracle` (Change 3).
- [ ] Stage 4 — Evaluate invariants: check traces + graph against declared invariants → findings.
- [ ] Stage 5 — Feedback: emit `verisim-feedback-v1` payload (Change 4).
- [ ] Stage 6 — Breaker: run the oscillation monitor (Change 5) throughout; `critical` halts the run.
- [ ] Stage 7 — Prepare delivery: signed report + prepared, unpushed commit/patch on the fixture.

## 3. Gating + isolation
- [ ] Enforce fail-safe ordering: a halted/failed stage blocks all later stages.
- [ ] Require explicit human confirmation before commit/push/deploy; default is prepare-and-stop.
- [ ] Assert local + network-isolated execution (no outbound network beyond loopback).
- [ ] Honor the existing commit gate; never bypass it.

## 4. Determinism + artifacts
- [ ] Make stages deterministic for fixed (fixture revision, intent, seed); write a typed artifact
      per stage.
- [ ] Make the run resumable/inspectable from artifacts.

## 5. Verification
- [ ] Happy-path e2e: a fixture + benign intent runs all stages and stops at "prepared, unpushed
      commit + report" with no commit/push performed.
- [ ] Gating test: a `critical` breaker trip halts before delivery and emits a rollback recommendation.
- [ ] Isolation test: the run completes with no outbound network and never touches the source repo.
- [ ] Confirmation test: only with explicit confirmation does a commit happen — and only on the
      de-fanged fixture, which cannot push to the original.
- [ ] Determinism test: same (fixture revision, intent, seed) → same stage artifacts.
