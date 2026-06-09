# Datasheet — verisim-bench@0.1.0+8147113f0407082f

## Motivation
The one world-model faithfulness benchmark whose labels are **exact**: a deterministic oracle
supplies the true next state at every step, so `H_ε(ρ)` (faithful horizon at oracle budget ρ) is
measured against ground truth rather than eyeballed. Built for defensive autonomous-cyber-defense
and world-model faithfulness research.

## Composition
- **Worlds:** network, host, filesystem (3 worlds).
- **Drivers:** weighted, forky, structural (one per world).
- **Seeds:** 16 per (world, driver); rollouts of 80 steps.
- **Labels:** bit-exact next-state from the reference oracle; `ε = 1.0·δ` per world
  (δ = the world's single-edit divergence granularity, SPEC-13).
- **Records are oracle-generated**, not scraped — no PII, no copyright surface.

## Collection process
Deterministic and seeded: every record is a pure function of (world, driver, seed), regenerable from
this manifest (hash `8147113f0407082f`). No human annotation.

## Intended use
Defensive ACD environments and world-model faithfulness research. **Not** for offensive automation.

## Limits
The reference oracle is a *model* of POSIX/network/host semantics, validated bit-exact against a
real shell on the structure-building grammar (SPEC-11 H27) but not across all of POSIX. Scores are
comparable only within a fixed manifest hash.
