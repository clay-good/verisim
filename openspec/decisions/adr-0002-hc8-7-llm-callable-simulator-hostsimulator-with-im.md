# ADR-0002: HC8 §7 LLM-callable simulator: HostSimulator with imagine/verify protocol and task-oracle Goals

## Status

accepted

**Domains**: hostsim

## Context

The world model's payoff is as a cheap, faithful, verifiable machine an LLM agent reasons over. New verisim.hostsim packages the loop's M_θ (HostModel) + reference oracle into a §7 protocol object. HostSimulator exposes two interfaces: predict_next (the existing loop interface) and the new agent interface — imagine(state, plan) rolls M_θ over syscall strings with no oracle (cheap draft, Dreamer-style 'plan in imagination'), while verify(state, plan, epsilon, goal) rolls model imagination alongside oracle truth step-by-step, returning a PlanReport with predicted vs true final state, per-step divergence, plan-faithful-horizon (§17.8 H_ε-for-a-plan = first step where divergence > epsilon), oracle cost, trusted flag, and goal agreement. Goals are named predicates over final HostState (file_content/file_absent/proc_running/proc_killed/proc_uid/all_of), composing the §7 'third oracle': state oracle judges faithfulness, task oracle (Goal predicate) judges plan success, and their agreement yields task-level faithfulness for a verifiable-reward trainer. No changes to host state/action/delta/oracle/model — the simulator is a thin agent-facing wrapper. Tested torch-free with null (drifts) and oracle-backed (perfect, trusted, goal-agrees) baselines.

## Decision

The system SHALL provide an LLM-callable HostSimulator that exposes imagine (oracle-free plan rollout) and verify (oracle-grounded plan evaluation with per-step divergence, faithful horizon, and goal agreement) over any HostModel.

## Consequences

New dependency-free-except-model verisim.hostsim module (HostSimulator, PlanReport, PlanRollout, Plan; goal.py: Goal + constructors) plus tests/test_hostsim.py (torch-free). Realizes §17.8 H_ε-for-a-plan as plan-faithful-horizon and §7 third-oracle as goal_agreement. run_record exposes plans as HostRunRecord so plans reuse existing figures-from-records machinery. Open HC8 work remains: experience-stream + plasticity probe (H15), counterfactual replay (EH6), Tier-B system oracle (rr/Hermit/gVisor), Inspect benchmark + verifiers-spec host RL env, §16 decentralized protocol, and technical report.

> Recorded by openlore decisions on 2026-06-03
> Decision ID: d21cec22
