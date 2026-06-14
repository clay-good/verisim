# Trajectory-oscillation circuit breaker (detect-and-halt)

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/safety/` + `tests/test_safety_breaker.py`
> (12 tests, all green; full `mypy`/`ruff` clean). The breaker consumes a `PlanningTransition`
> stream (the Change-6 loop is its producer); shipped and verified standalone.
> One sentence: **watch the planning loop's own state transitions for oscillation and repetitive
> edits; on breach, freeze the loop and drop the in-memory speculative rollout, then surface a
> recommended workspace rollback for a human to confirm — never an autonomous git/file reset.**

## Why

The original request wanted an "epistemic circuit breaker" that extends OpenLore's Epistemic
Lease and *autonomously rolls back the workspace*. The findings doc (§4) showed two problems:
OpenLore's lease is unrelated machinery (it measures a coding agent's staleness about *source*,
exposes no cross-process API, and has no `Stale [Critical]` tier), and autonomous `git reset` /
file deletion behind a heuristic is destructive. The requester chose **human-gated everywhere**.

What *is* sound and valuable is the **detection** idea, computed over Verisim's **own** planning
trajectory. The simulator already reasons in terms of state transitions, drift, and faithful
horizon; an oscillation metric over those transitions is a natural sibling of the existing drift
metrics and a genuine safety signal: an agent stuck flip-flopping between two states, or
re-editing the same files in a cycle, is in a degenerate loop the prototype should *stop*.

We borrow the lease *concept* (confidence decay → escalating response) **by analogy only**; we do
not wire into OpenLore's TS object.

## What changes

1. **An oscillation metric** over the planning loop's state-transition stream: repeated
   state/edit bigrams ÷ total transitions in a sliding window (the same shape as OpenLore's own
   `oscillation` field, but over Verisim planning states, not MCP tool accesses), plus a
   repetitive-file-modification counter.
2. **Decay tiers** (`ok` → `degraded` → `critical`) with explicit, documented thresholds — named
   honestly (no fictional `Stale [Critical]`); `critical` is the breaker trip point.
3. **The breaker action, on `critical`:**
   - **freeze** the loop (stop issuing further actions);
   - **drop the in-memory speculative rollout** (cheap, reversible, internal — this is *not* a
     filesystem operation);
   - **compute and surface a recommended rollback** to the last known-good baseline (the fixture
     snapshot / last green checkpoint), as a *recommendation object* with a diff preview;
   - **require human confirmation** before any `git`/filesystem mutation runs. Absent
     confirmation, nothing irreversible happens.
4. **Snapshot-before-anything** — the loop maintains known-good checkpoints (a stash/snapshot of
   the fixture working tree) so that *if* a human confirms a rollback, no uncommitted work is
   lost; the snapshot is taken proactively, not at trip time.

## Contract / boundaries

- **Operates on the fixture only** (Change 1), never the real source repo.
- **Reversible actions are automatic; irreversible actions are human-gated.** Dropping a
  speculative rollout and freezing the loop are automatic. Touching the filesystem/git is not.
- **Deterministic thresholds.** The metric and tiers are pure functions of the transition stream;
  same stream → same tier.

## Risks & honest limits

- Threshold tuning is empirical; the spec fixes the *shape* and *defaults* and makes them
  configurable, with tests pinning behavior at the boundaries.
- A breaker that only halts (never auto-fixes) can leave a frozen run needing human attention —
  that is the intended, safe failure mode for a prototype.
