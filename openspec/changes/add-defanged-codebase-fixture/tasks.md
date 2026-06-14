# Tasks — De-fanged real-codebase fixture

> Posture: human-gated everywhere irreversible. This change only writes inside Verisim-owned
> scratch and never mutates a source repo.
>
> Status: IMPLEMENTED — `src/verisim/fixture/` (`materialize.py`, `__init__.py`) +
> `tests/test_fixture.py` (18 tests). ruff + bare mypy (710 files) + pytest all green.

## 1. Selection + config
- [x] Define a `FixtureConfig` (source-roots allowlist, default root
      `/Users/user/Documents/development/public/`, fixture scratch root, copy options).
      → `materialize.FixtureConfig` (`source_roots`, `scratch_root`, `exclude`).
- [x] Implement deterministic selection: given a repo name/path, validate it is (a) inside the
      allowlist and (b) a git repo; reject otherwise with a clear error.
      → `materialize.validate_source` (allowlist + `.git` + submodule/LFS loud-fail).
- [x] Define selection criteria doc for "prototype-suitable" repos (small, single dominant
      language, real call graph) and list the on-disk candidates by file count.
      → module docstring "Selection criteria" section (worldify ~36 … proxilion ~321k out-of-scope).

## 2. Materialize
- [x] Copy the working tree into `<scratch>/<name>-<source-head-short-sha>/` (collision-safe).
      → into `<scratch>/<name>-<short-sha>/repo`; existing dest is rejected, not clobbered.
- [x] Exclude volatile/irrelevant dirs from the copy (`node_modules`, `.venv`, build caches) but
      preserve everything OpenLore needs to analyze. → `DEFAULT_EXCLUDE`; `.git` deliberately kept.
- [x] Verify the copy is complete (file count + content-hash manifest) or fail loudly.
      → post-copy `tree_hash(copy) == tree_hash(source)` check; `file_count` in the manifest.

## 3. De-fang `.git`
- [x] Remove every remote from the copy. → `_defang` step 1.
- [x] Set a sentinel fixture git identity (name/email) in the copy's local config.
      → `Verisim Fixture <fixture@verisim.invalid>`.
- [x] Install a `pre-push` hook in the copy that exits non-zero unconditionally. → `_PRE_PUSH_HOOK`.
- [x] Set a guard config flag and assert no remote URL resolves.
      → `verisim.fixture=true`; `_defang` step 5 asserts zero remotes or raises.
- [x] Write `FIXTURE.json` manifest (source path, source HEAD sha, timestamp, defang actions).
      → `FixtureManifest` at `<fixture>/FIXTURE.json` (also `tree_hash`, `file_count`).

## 4. Teardown + safety
- [x] Implement teardown that removes a fixture directory entirely. → `materialize.teardown`.
- [x] Round-trip safety test: hash the source tree before materialize and after teardown — assert
      byte-identical (the original is provably untouched).
      → `test_materialization_leaves_source_untouched`.
- [x] Negative test: attempting a `git push` inside a fixture fails (hook + no-remote).
      → `test_push_is_structurally_blocked` (both no-remote and forced-remote paths).
- [x] Negative test: a source path outside the allowlist is rejected.
      → `test_source_outside_allowlist_rejected`.

## 5. Verification
- [x] Deterministic-materialize test: same `(source, options)` → identical file set twice.
      → `test_deterministic_copy`.
- [x] Manifest-traceability test: `FIXTURE.json` HEAD sha matches the source's HEAD at copy time.
      → `test_manifest_traceable_to_source_revision`.
- [x] Cross-POSIX: copy/hook logic works on macOS (primary) and Linux CI; no Linux-only calls.
      → pure `pathlib`/`shutil`/`subprocess git`; `#!/bin/sh` hook; no Linux-only syscalls.
