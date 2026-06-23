# ADR-0004: SPEC-10 / HS1: scale faithful horizon along model capacity to separate under-resourcing from fundamental compounding wall

## Status

accepted

**Domains**: experiments, specs, figures

## Context

Prior negatives (floor+cliff H_ε in v0/EN1/EH1) were confounded with model under-resourcing. SPEC-10 scales the headline metric H_free along a 108x parameter range (flat network arm), holding the world fixed. Key implementation choices: (1) use train_batched (SPEC-2.1 minibatch loop) instead of full-batch to keep per-step cost constant over the large coverage set; (2) build coverage dataset once and share across capacity cells (oracle labels are free, SPEC-9 regime); (3) num_threads as config knob (1 for deterministic smoke test, 0/all-cores for the committed sweep). Horizon efficiency eta = H_free / H_indep is the scale-free headline. Result: H_free lifts ~9x (1.75→15.8 steps) then saturates; lift transfers to adversarial regime; eta > 1 throughout (no compounding penalty at this scale). H26 supported with nuance: the floor was substantially an under-resourcing artifact, and the open question moves to whether a favorable consultation knee exists once the floor is high.

## Decision

The system SHALL measure free-running faithful horizon H_free and horizon efficiency eta across a capacity sweep of at least two orders of magnitude, reporting bootstrap confidence intervals over independent seeds.

## Consequences

New spec docs/specs/SPEC-10.md; new experiment horizon_scaling.py with ModelScale capacity axis, HorizonScalingConfig, independence_horizon, bootstrap CI reduction over seeds; plotting script, config, deterministic smoke test; committed figures. SPEC.md §9 gains H26 verdict; report.md gains Scale section; README gains SPEC-10 row and HS1 result cell. Future work: HS2/HS3 for universality on v0/host, graph arm, world-size cross-axis, and the consultation-knee-at-high-floor question.

> Recorded by openlore decisions on 2026-06-23
> Decision ID: af24fa07
