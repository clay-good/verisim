# Tasks — Headless CD pipeline

> All implemented 2026-06-14. Code: `src/verisim/pipeline/model.py` (typed run model),
> `src/verisim/pipeline/cd.py` (ordered fail-safe orchestrator + `verisim-cd` console entry).
> Tests: `tests/test_pipeline.py` (10). Entry point registered in `pyproject.toml [project.scripts]`.

## 1. Entry point + run model
- [x] Headless entry point — `run_pipeline(fixture, intent, ...)` callable + `main()` /
      `verisim-cd` console script, taking a fixture + `Intent` (goal/actions/invariants/change) + seed.
- [x] `RunReport` artifact (`model.py`) aggregating every stage's typed `StageResult` plus findings,
      feedback counts, rollback recommendation, prepared delivery, and a content seal (`signature`).

## 2. Stage wiring (in order)
- [x] Stage 1 — Intent graph: `_stage_intent_graph` loads the read-only `CodeGraph` (Change 2;
      `load_code_graph` / `analysis_db_path`) or accepts a passed graph, and records the intent.
- [x] Stage 2 — Speculative rollout: `_stage_speculative` runs `loop/speculative.py`
      `speculative_rollout` (oracle-as-drafter stand-in for M_θ) and fills the droppable rollout.
- [x] Stage 3 — Trace side-effects: `_stage_trace` executes the verifying steps via `TracingOracle`
      (Change 3), threading state forward into the breaker's trajectory.
- [x] Stage 4 — Evaluate invariants: `_stage_evaluate` checks static + runtime-discovered edges
      against the declared `ArchInvariant`s → `InvariantFinding`s.
- [x] Stage 5 — Feedback: `_stage_feedback` emits + validates the `verisim-feedback-v1` payload
      (Change 4; `build_feedback_payload` / `write_feedback` / `validate_payload`).
- [x] Stage 6 — Breaker: `TrajectoryBreaker` (Change 5) observes every transition during stage 3;
      `_stage_breaker` is the gate — a `critical` status halts the run before delivery.
- [x] Stage 7 — Prepare delivery: `_stage_prepare_delivery` writes a content-sealed report + a
      prepared, *unpushed* commit/patch on the fixture.

## 3. Gating + isolation
- [x] Fail-safe ordering: the `guarded` runner marks a stage `failed` on exception and every later
      stage `skipped`; the breaker may `halt` before delivery (tested: missing graph + breaker trip).
- [x] Explicit human confirmation before commit: `confirm_delivery` defaults `False` (prepare and
      stop); only `True` commits. Push/deploy never performed.
- [x] Local + network-isolated: `_network_isolated` attests via the oracle's hermeticity; v0 grammar
      exposes no network surface. Test asserts isolation + the original source is byte-identical.
- [x] Honor the commit gate: `git commit` is run plainly, never `--no-verify` (tested: a refusing
      fixture pre-commit hook fails the commit instead of being bypassed).

## 4. Determinism + artifacts
- [x] Deterministic for fixed (fixture revision, intent, seed): every artifact is canonical JSON
      with wall-clock/volatile fields projected out (`_project_trace`); the report carries a content
      seal. Test: same source materialized twice → identical signatures + artifacts.
- [x] Resumable/inspectable: each stage writes a numbered typed artifact under `.verisim-run/`
      (Verisim-owned scratch, outside the subject tree) plus a final `run-report.json`.

## 5. Verification
- [x] Happy-path e2e: `test_full_run_prepares_an_undelivered_change` — all 7 stages OK, prepared
      unpushed commit + report, HEAD unchanged (no commit/push performed).
- [x] Gating test: `test_breaker_trip_halts_before_delivery` — a flip-flop trajectory trips
      `critical`, delivery is skipped, a rollback recommendation is surfaced, no commit.
- [x] Isolation test: `test_source_untouched_and_network_isolated` — completes network-isolated and
      leaves the original source repo byte-identical.
- [x] Confirmation test: `test_confirmed_commit_lands_only_on_fixture_and_cannot_push` — only with
      confirmation does a commit happen, on the fixture, and a push fails (no remote + pre-push hook).
- [x] Determinism test: `test_deterministic_stages` — same (fixture revision, intent, seed) →
      identical stage artifacts and report signature.
