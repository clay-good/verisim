# Tasks — Trajectory-oscillation circuit breaker

All code lands in `src/verisim/safety/` (`breaker.py` + `__init__.py`); verification in
`tests/test_safety_breaker.py` (12 tests). Full-project `mypy` and `ruff` are clean.

## 1. Metric
- [x] Define the planning-state transition stream the monitor consumes (from the loop runner).
      (`PlanningTransition(from_state, to_state, files_modified)` — opaque hashable state labels +
      repo-relative modified paths; the Change-6 loop is the producer, kept decoupled.)
- [x] Implement the oscillation metric: repeated state/edit bigrams ÷ total transitions over a
      sliding window; plus a repetitive-file-modification counter. (`compute_metric`:
      `oscillation = (n − distinct bigrams) / n` over the last `window` transitions;
      `file_repeat`/`worst_file` = max modifications to any single file in the window.)
- [x] Make the metric a pure function of the stream (deterministic). (`compute_metric`/`classify`/
      `evaluate` are pure; `test_metric_is_deterministic` pins same-stream → same result.)

## 2. Tiers + breaker
- [x] Define tiers `ok` / `degraded` / `critical` with explicit default thresholds (configurable).
      (`BreakerConfig`: `window=12`, `degraded/critical_oscillation=0.34/0.50`,
      `degraded/critical_file_repeat=3/5`; tier = worse of the two sub-metric tiers, named honestly.)
- [x] On `critical`: freeze the loop; drop the in-memory speculative rollout.
      (`TrajectoryBreaker._trip`: sets `frozen`, calls `SpeculativeRollout.drop()`;
      `test_critical_trips_safe_automatic_actions_only`, `test_frozen_loop_ignores_further_transitions`.)
- [x] Compute a `RollbackRecommendation` (target baseline, diff preview, reason) — do NOT execute it.
      (`_build_recommendation` against the latest known-good checkpoint; `A/D/M` diff preview from a
      tree-map diff; `test_recommendation_has_target_and_diff`.)
- [x] Gate any git/filesystem mutation behind explicit human confirmation.
      (`confirm_and_rollback(*, confirmed)` raises `RollbackNotConfirmed` and mutates nothing unless
      `confirmed=True`; `test_rollback_is_recommended_not_executed`.)

## 3. Known-good snapshots
- [x] Maintain proactive known-good checkpoints of the fixture working tree (snapshot/stash).
      (`checkpoint(label)` copies the working tree into `<fixture.root>/.verisim-snapshots/` — a
      sibling of `repo/`, outside the subject tree, under scratch; `auto_baseline=True` takes one at
      construction.)
- [x] Ensure a confirmed rollback restores a checkpoint without losing uncommitted work (snapshot
      first), and operates only on the fixture. (`confirm_and_rollback` snapshots the *current* tree
      (`pre-rollback-*`) before `_restore`; both operate only on `fixture.repo_path`;
      `test_confirmed_rollback_is_safe_and_scoped` asserts the source repo is byte-identical after.)

## 4. Verification
- [x] Test: an oscillating transition stream (A→B→A→B…) trips `critical`; a converging one does not.
      (`test_oscillating_stream_reaches_critical`, `test_converging_stream_stays_ok`; plus
      `test_partial_oscillation_is_degraded` and `test_sliding_window_forgets_old_oscillation` for the
      gradient and window behavior.)
- [x] Test: repetitive edits to the same file trip the repetitive-modification path.
      (`test_repetitive_file_modification_trips_critical` — distinct states, oscillation 0, file path
      trips via five edits to one file.)
- [x] Test: on trip, the loop freezes and the speculative rollout is dropped — and NO git/fs
      mutation occurs without confirmation. (`test_critical_trips_safe_automatic_actions_only`
      asserts `frozen`, `rollout.dropped`, and `tree_hash` unchanged across the trip.)
- [x] Test: a `RollbackRecommendation` is produced with a correct target + diff preview.
      (`test_recommendation_has_target_and_diff`.)
- [x] Test: confirmed rollback restores the baseline on the fixture and leaves the source repo
      untouched. (`test_confirmed_rollback_is_safe_and_scoped`: tree restored to baseline, stray file
      removed, pre-rollback snapshot retains it, source `tree_hash` unchanged.)
- [x] Determinism test: identical streams yield identical tiers. (`test_metric_is_deterministic`.)
