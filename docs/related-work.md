# Related work — a maintained bibliography

> The neighbors Verisim positions against (SPEC.md §8) and the state-of-the-art it
> cites (SPEC.md §16), each with a one-line take on the relationship. Entries are
> by canonical name, venue/org, and year; references are current as of **May 2026**.
>
> **On links:** to avoid fabricated or rotted URLs this file deliberately cites by
> name + venue rather than pasting IDs. When a canonical URL is verified it is added
> inline; until then the name + venue is enough to find the source. Do not invent
> links to fill the gaps.

## Generative / visual world models

- **Genie 3** (Google DeepMind, 2025) — interactive environment generation; holds
  visual coherence for minutes then drifts. *Take:* same failure mode (long-horizon
  drift), opposite domain (no oracle). Verisim attacks their open problem where it is
  measurable.
- **V-JEPA 2** (Meta, 2025) — predictive world model in latent space for embodied
  planning. *Take:* prediction branch of the 2025 split; no inference-time oracle.
- **NVIDIA Cosmos** (NVIDIA, 2025) — world foundation models for physical-AI
  synthesis. *Take:* visual-synthesis branch; competes on fidelity Verisim does not.
- **World Labs** (2025) — large-scale spatial/world models. *Take:* same no-oracle
  regime; cited as the frontier Verisim is *not* on.

## Model-based RL world models

- **Dreamer (v1–v3)** (Hafner et al.) — learn a latent world model and plan in it.
  *Take:* closest on the "roll out a learned model" mechanic, but the env-as-oracle is
  the thing being modeled *away*, not kept in the loop. Verisim keeps the oracle and
  asks how little of it suffices.

## Neuro-symbolic world models and verifier-gated reasoning

- **Neuro-symbolic world models** (2025–26 line: symbolic transition structure +
  neural fill-in; train neural only where symbolic rules don't cover). *Take:* closest
  prior art on *method*. Verisim's differences: the symbolic half is the real
  system/oracle (not a hand-built abstraction), the oracle is budgeted and optimized
  over at inference (RQ2), and the domain is computer environments specifically.
- **VIRF, EIDOKU, SymCode** — deterministic verifier gating stochastic generation at
  the level of a single reasoning step / answer. *Take:* same instinct, lifted by
  Verisim from one answer to an entire temporal rollout where errors compound — a
  strictly harder regime.

## RL with verifiable rewards

- **RLVR (RL with verifiable rewards)** — reward from a checkable signal rather than a
  learned reward model. *Take:* Verisim instantiates it with the oracle as the reward
  and faithful horizon as the return (SPEC.md §6.3); realized as the
  `verisim.rl.WorldModelEnv` environment.
- **Constitutional AI / RLAIF** (Anthropic) — an LLM critiques an LLM (probabilistic
  checking probabilistic). *Take:* Verisim replaces the probabilistic critic with a
  deterministic oracle wherever the domain admits one.

## Autonomous cyber defense simulators (the application field)

- **CybORG, CyGIL, PrimAITE** — ACD training/eval simulators; the emulation–simulation
  gap is their standing problem. *Take:* the application field. Verisim contributes a
  learned-but-faithful world model as a candidate fix for their sim-to-emulation gap
  (SPEC.md §4.1; hypothesis H4/Phase 1).

## Evaluation & RL-environment ecosystems (distribution targets)

- **Inspect / `inspect_evals`** (UK AI Safety Institute) — the eval framework labs use.
  *Take:* Verisim's faithfulness benchmark ships as an `inspect_ai` task
  ([`verisim.eval.inspect_task`](../src/verisim/eval/inspect_task.py)).
- **Prime Intellect Environments Hub / `verifiers`** — community RL environments with a
  verifiable-reward spec. *Take:* Verisim's oracle-as-reward env conforms to the
  `load_environment` entrypoint ([`verisim.rl`](../src/verisim/rl/)).
