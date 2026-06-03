# ADR-0003: HC8 §16 trustless verified-contribution protocol + the composed-host faithfulness benchmark/Inspect adapter

## Status

accepted

**Domains**: contrib, hosteval

## Context

SPEC-6 §16 specifies the program's open/decentralized intent in research terms: the deterministic oracle makes contributed host data *trustless by construction*. Where Prime Intellect's INTELLECT-2 must verify untrusted rollouts heuristically (TOPLOC locality-sensitive hashing, §2.9), verisim re-runs the oracle and checks contributions bit-for-bit. Separately, SPEC-6 §1.4 argues the computer-use field lacks the *metrology* for a whole-machine world model — OSWorld/TheAgentCompany grade the agent, never a simulator of the host's predicted next state — so the host world owes a packaged faithfulness benchmark (the host analogue of v0's verisim.eval). Both are dependency-free, GPU-free HC8 packaging items that the deterministic core already supports; neither needs the Tier-B oracle, torch, or the experience stream. Verifying an arbitrary contributed (state, action, next_state) by re-execution requires reconstructing the contributed HostState, which needed the serialization inverse `from_canonical_host` (to_canonical_host previously had no inverse — the HC2 dataset only ever replayed forward from boot).

## Decision

The system SHALL provide (1) a verisim.contrib verification protocol that accepts a contributed transition/trajectory iff re-running the deterministic oracle reproduces it bit-for-bit — verify_transition checks next_state plus optional delta/observation; verify_trajectory additionally requires steps to chain (next_state[i] == state[i+1]) so individually-valid transitions cannot be spliced; content_address gives a tamper-evident SHA-256 manifest hash; hostile/malformed input is rejected, never raised — and (2) a verisim.hosteval composed-host faithfulness benchmark mirroring verisim.eval: score_host_model grades any HostModel through the HC5 composed loop (composed H_ε, oracle calls), host_step_labels/grade_host_prediction are the single-step QA form (composed-divergence grader), and a lazily-imported host_faithfulness_task packages it as an inspect_ai Task. verisim.host gains the round-trippable from_canonical_host inverse the protocol rests on.

## Consequences

New dependency-free verisim.contrib (protocol.py: VerificationReport, content_address, verify_transition, verify_trajectory) and verisim.hosteval (faithfulness.py + lazily-imported inspect_task.py) modules; tests/test_contrib.py and tests/test_hosteval.py (torch-free; the Inspect core grader tested directly since inspect_ai is absent in CI). verisim.host.state gains from_canonical_host (round-trip tested). pyproject relaxes strict typing for verisim.hosteval.inspect_task (same rationale as verisim.eval.inspect_task). Realizes the *verification* half of §16 (free and certain); DiLoCo-style low-communication merge stays deferred and gated (§16 v1 scope). Open HC8 work remains: per-subsystem decode heads (HC7), the Tier-B system oracle (rr/Hermit/gVisor), and the technical report.

> Recorded manually on 2026-06-03 (sibling to ADR-0001/0002; openlore sync to follow)
> Decision ID: a3d0c916
