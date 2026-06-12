# Datasheet — verisim-cue@0.1.0+53b53fefba0d267e

## Motivation
`verisim-cue` is the one **computer-use** world-model benchmark whose labels are *exact*: a
deterministic host oracle (the shell/file/process slice of computer use — the slice that *admits* a
ground-truth oracle, unlike GUI) supplies the true next state at every step. Each task is scored not
only on success but on whether **faithfulness was load-bearing** for it — a verdict no oracle-free
computer-use benchmark can produce. Built for defensive autonomous-cyber-defense and world-model
faithfulness research.

## Composition
- **Environment:** the SPEC-6 host world (`host`): processes, file descriptors, a
  filesystem, commands, under a deterministic reference oracle and a real-`/bin/sh` system anchor
  (SPEC-11).
- **Tasks:** 4 predictive-defense tasks ordered structure→content
  (process-control, fd-control, file-integrity, content-value), each a faithful-vs-free gap over a keyed-set
  extractor.
- **Workload:** the `forky` driver; 24 seeds, 16-step
  episodes per task.
- **Capacity ladder:** xs, s, m, l (CPU rungs; the GPU run extends the top) — the
  benchmark is the substrate of a *scale law*, not a single number.
- **Records are oracle-generated**, not scraped — no PII, no copyright surface.

## Collection process
Deterministic and seeded: every record is a pure function of (task, driver, seed), regenerable from
this manifest (hash `53b53fefba0d267e`). No human annotation.

## Intended use
Defensive ACD environments and computer-use world-model faithfulness research. **Not** for offensive
automation (SPEC.md §13).

## Limits
The reference oracle is a *model* of host semantics, validated bit-exact against a real shell on the
structure-building grammar (SPEC-11 H27) but not across all of POSIX. Computer use here is
shell/file/process, **not** GUI — that is the oracle-grounded slice and the point. Load-bearing
verdicts are comparable only within a fixed manifest hash and at a stated scale.
