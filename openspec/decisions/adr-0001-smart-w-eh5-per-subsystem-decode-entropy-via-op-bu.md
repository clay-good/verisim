# ADR-0001: Smart π_w (EH5): per-subsystem decode entropy via op-bucketing drives an information-gain which-subsystem policy

## Status

accepted

**Domains**: hostmodel, hostloop

## Context

The EH2 smart-π_c positive licensed the smart π_w axis (§8.2: verify the subsystem whose predicted delta is least certain). Per-subsystem uncertainty is obtained by bucketing each decoded token's masked-distribution entropy into the subsystem of the op currently being decoded (OP_SUBSYSTEM map in grammar.py; the factored arm's _decode accumulates it). This reuses the existing single-decoder forward pass with no architecture change. Exposed via predict_delta_with_subsystem_uncertainty and a HostSubsystemUncertaintyModel protocol (extends HostModel, runtime_checkable). SubsystemPolicy.select gains an optional uncertainty map; new UncertaintySubsystem policy picks argmax with canonical-order tie-breaking and round-robin fallback. The runner supplies uncertainty only when the proposer implements the protocol (isinstance check). Result at smoke scale: uncertainty matches the best fixed baseline (fixed_proc) and beats round_robin per-bit, but cheapest-fixed (fd) still wins pure bit-efficiency — the EH3 cost-vs-consequence tension persists and raw-horizon CIs overlap.

## Decision

The system SHALL support an uncertainty-driven which-subsystem policy that selects the subsystem with highest per-subsystem decode entropy for consultation.

## Consequences

grammar.py gains OP_SUBSYSTEM; graph_model._decode returns (delta, belief_var, per_subsystem_entropy) and exposes predict_delta_with_subsystem_uncertainty; hostloop/model.py adds HostSubsystemUncertaintyModel protocol; subsystem.py gains UncertaintySubsystem policy and SubsystemPolicy.select gets optional uncertainty arg; runner passes per-subsystem uncertainty to π_w. New files: experiments/eh5.py, configs/eh5.json, tests/test_eh5.py, figure eh5_subsystem_policy.png wired into reproduce.sh. Future work: ideal cost-AND-consequence π_w, per-subsystem decode heads (vs entropy-bucketing signal), §6.3 drift levers, scheduler/H14.

> Recorded by openlore decisions on 2026-06-03
> Decision ID: 2806f009
