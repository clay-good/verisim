# Tasks — Trajectory-oscillation circuit breaker

## 1. Metric
- [ ] Define the planning-state transition stream the monitor consumes (from the loop runner).
- [ ] Implement the oscillation metric: repeated state/edit bigrams ÷ total transitions over a
      sliding window; plus a repetitive-file-modification counter.
- [ ] Make the metric a pure function of the stream (deterministic).

## 2. Tiers + breaker
- [ ] Define tiers `ok` / `degraded` / `critical` with explicit default thresholds (configurable).
- [ ] On `critical`: freeze the loop; drop the in-memory speculative rollout.
- [ ] Compute a `RollbackRecommendation` (target baseline, diff preview, reason) — do NOT execute it.
- [ ] Gate any git/filesystem mutation behind explicit human confirmation.

## 3. Known-good snapshots
- [ ] Maintain proactive known-good checkpoints of the fixture working tree (snapshot/stash).
- [ ] Ensure a confirmed rollback restores a checkpoint without losing uncommitted work (snapshot
      first), and operates only on the fixture.

## 4. Verification
- [ ] Test: an oscillating transition stream (A→B→A→B…) trips `critical`; a converging one does not.
- [ ] Test: repetitive edits to the same file trip the repetitive-modification path.
- [ ] Test: on trip, the loop freezes and the speculative rollout is dropped — and NO git/fs
      mutation occurs without confirmation.
- [ ] Test: a `RollbackRecommendation` is produced with a correct target + diff preview.
- [ ] Test: confirmed rollback restores the baseline on the fixture and leaves the source repo
      untouched.
- [ ] Determinism test: identical streams yield identical tiers.
