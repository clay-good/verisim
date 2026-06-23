# ADR-0005: SPEC-7 DS0 increment 1: replicated-KV-under-partition deterministic core (dist/ + distoracle/)

## Status

accepted

**Domains**: dist, distoracle, specs

## Context

The distributed world (SPEC-7) begins with a deterministic-core-first approach mirroring the host and network worlds. DS0 increment 1 ships the eventual-consistency core: DistributedState = per-(object,node) MVCC replicas + causal event log + in-flight messages + partition/crash/clock (no stored global state). The Tier-A ReferenceDistOracle is a pure deterministic DES where put writes locally and enqueues async replication messages delivering on advance only if due+reachable; convergence is last-writer-wins by (version, value); partition keeps messages stuck in-flight producing stale reads. Deferred to later DS0 increments: Raft-subset consensus, txn/lock table, and embedding SPEC-6 HostState / SPEC-5 network inside each node.

## Decision

The system SHALL model distributed state as per-(object,node) MVCC replicas with no stored global state, where replication messages deliver only when due and reachable, producing stale reads under partition.

## Consequences

New packages src/verisim/dist/ (config, state, action, delta, serialize) and src/verisim/distoracle/ (base, reference). New tests tests/test_dist_core.py (14 property/semantics tests) and tests/test_dist_goldens.py (2 golden canonical states). New docs/distributed-semantics.md (normative DS0 semantics). SPEC-7 status flipped PAUSED→ACTIVE. README updated for four worlds. All dependency-free, GPU-free. Later DS increments extend with HostDelta/NetDelta embedding, drivers, tiered oracle, M_theta, the loop, and ED1 distributed H_eps(rho) curve.

> Recorded by openlore decisions on 2026-06-23
> Decision ID: a31483cf
