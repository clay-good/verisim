# De-fanged real-codebase fixture

> Status: IMPLEMENTED (2026-06-14) — `src/verisim/fixture/` + `tests/test_fixture.py` (18 tests
> green; ruff + bare mypy clean). Change 1 of the six is done; Changes 2–6 remain DRAFT.
> One sentence: **give the prototype a real, local codebase to operate on by copying one into an
> isolated fixture and neutralizing its `.git` so nothing can ever commit or push to the
> original.**

## Why

The integration's whole point (findings doc §9) is to ground Verisim's simulated
host/filesystem/network world in a **real** codebase, and to let a **human-gated CD loop**
prepare commits against it. That requires a subject under test that is:

- **real** — an actual repository with a non-trivial call graph OpenLore can analyze, so the
  static↔dynamic correction the research claims is exercised on real structure, not a toy;
- **isolated** — a throwaway copy, so the loop's filesystem writes, speculative rollbacks, and
  prepared commits never touch a repo the user cares about;
- **inert with respect to delivery** — its `.git` must be **de-fanged** so that even a bug in
  the CD pipeline (Change 6) cannot commit to or push to the original's history or remotes.

There are many candidates on disk under `/Users/user/Documents/development/public/`. Small,
real, multi-file repos are ideal for a prototype that must also *scale* later: e.g.
`agent-replay` (~65 files), `armorly` (~68), `securifine` (~42), `worldify` (~36),
`proxilion-mcp` (~130). This change does **not** hardcode one — it specifies a deterministic
*selection + materialization* procedure parameterized by a source path, with a documented
default, so the same harness scales from one fixture to many.

## What changes

1. **A fixture materializer** that, given a source repo path under a configured roots allowlist,
   produces an isolated working copy under a Verisim-owned scratch location (never inside the
   source tree, never inside `.openlore/`).
2. **`.git` de-fanging** applied to the copy so the prototype is structurally incapable of
   reaching the original's history or any remote:
   - remove all configured remotes;
   - rewrite `user`/`commit` identity to a sentinel fixture identity;
   - install a fixture-local `pre-push` hook (and a guard config) that hard-fails any push;
   - record a `FIXTURE.json` manifest (source path, source HEAD sha, copy timestamp, defang
     actions applied) so every downstream artifact is traceable to an exact source revision.
3. **Determinism + selection contract** — the same `(source path, options)` yields a
   byte-identical file set (minus volatile `.git` internals), and selection from the roots
   allowlist is deterministic and explicit (no implicit "pick any repo").
4. **A teardown** that removes a fixture completely and verifies the original is untouched
   (content hash of the source tree unchanged across a full materialize→use→teardown cycle).

## Contract / boundaries

- Source roots are an explicit allowlist (default: `/Users/user/Documents/development/public/`).
  A path outside the allowlist is rejected.
- The fixture root is Verisim-owned scratch and is gitignored.
- De-fanging is enforced **by construction** (no remotes exist to push to; the hook hard-fails)
  rather than by trusting downstream code — mirroring the SPEC-11 hermeticity-by-construction
  discipline.
- The materializer copies the working tree and a *neutralized* `.git`; it does not preserve the
  original's reflog, remotes, or hooks.

## Risks & honest limits

- A repo with submodules or LFS pointers may copy incompletely; the manifest records this and
  the materializer fails loudly rather than producing a half-fixture.
- Very large repos (e.g. `invariant` ~339k files, `proxilion` ~321k) are out of scope for the
  prototype default; selection criteria steer toward small repos first.
