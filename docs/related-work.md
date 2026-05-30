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

## SPEC-3 / SPEC-4 era — depth and the autonomous engine (May 2026)

> The references that ground [SPEC-3.md](../SPEC-3.md) (depth) and [SPEC-4.md](../SPEC-4.md)
> (the autonomous research engine). Name + venue + year, per this file's no-fabricated-links
> policy; arXiv IDs are given only where independently verified.

- **Speculative decoding** — Leviathan et al. (Google, 2023); Chen et al. (DeepMind, 2023);
  **Medusa**; **EAGLE / EAGLE-2** (2024, arXiv:2406.16858). *Take:* the draft+verify pattern
  is structurally identical to Verisim's neural-proposer + oracle-verifier loop (SPEC-3 §8);
  EAGLE-2's confidence↔acceptance calibration (≈0.05→0.04, ≈0.95→0.98) is the theory behind a
  confidence-gated consultation policy (RQ2/H9), and the ~0.5-acceptance net-negative floor
  bounds the useful ρ regime (SPEC-3 HW-3).
- **Test-time training / adaptation** — TTT layers; **TTT for ARC** (Akyürek et al., MIT, 2024);
  **LoRA-TTT** (2025); PETAL, AR-TTA, ReservoirTTA (TTA continual-learning line, 2025). *Take:*
  the recipe for the §6 self-healing loop — small-lr, replay, low-rank/selective updates, EMA +
  trust-region revert; per-step gradient updates without these forget catastrophically (SPEC-3 HW-2).
- **Autonomous research agents** — Karpathy **autoresearch** (2026, single-GPU nanochat, `val_bpb`
  gate, keep-if-better, fixed shard+budget anti-gaming); Sakana **AI Scientist v1/v2** (2025,
  arXiv:2504.08066) and its documented eval-hacking (smaller/synthetic datasets); Google AI
  co-scientist; DiscoPOP / population-based training / Bayesian opt. *Take:* the blueprint and the
  cautionary tale for SPEC-4 — the deterministic oracle is the anti-hacking asset, but only on a
  frozen, proposer-independent eval (SPEC-4 §5, §9).
- **RL with verifiable rewards (extended)** — **DeepSeek-R1** (2025; rule-based verifiable rewards,
  *refused* neural reward models for hacking risk); **GRPO** (Shao et al., 2024, critic-free).
  *Take:* the oracle is the purest verifiable reward; prefer GRPO (no learned critic to hack) and
  keep the reward strictly oracle-derived (SPEC-4 §10). Gap: RLVR is charted for reasoners, not
  next-state predictors — dense per-field reward likely needed.
- **Information-theoretic evaluation** — bits-per-byte; **MDL** (Rissanen, 1978); **prequential /
  description length of deep learning models** (Blier & Ollivier, NeurIPS 2018). *Take:* the basis
  for the §7 bits-to-correct metric — scale-free, tokenizer-independent, 0 iff faithful, computed
  prequentially over a rollout.
- **Small-model data-centrism** — **phi-1 … phi-4** ("Textbooks Are All You Need", Microsoft,
  2023–2024). *Take:* the H1/H6 lever — clean accuracy ~0.1 is a data/curriculum problem, not a
  parameter-count one (consistent with the committed E4 finding); fix with oracle-generated,
  curriculum-ordered, coverage-balanced synthetic transitions and distillation (SPEC-3 §5).
- **Partial observability / belief-state world models** — **RSSM** (PlaNet/Dreamer, ICLR 2020);
  TransDreamer; Mamba-based variants. *Take:* the §4 hybrid — keep the explicit checkable state for
  the observed part, add an RSSM latent for the unobserved part; oracle consults are belief resets.
- **Deterministic system-oracle infrastructure** — **rr** (record/replay); **Hermit** (Facebook,
  hermetic determinism over time/threads/RNG); **gVisor** (user-space syscall sandbox); CRIU;
  mininet / containerlab. *Take:* the §2 system oracle is built on these; concurrency/thread
  interleaving is the one hard wall they tame only at cost (SPEC-3 HW-1).
- **Autonomous cyber defense (updated)** — **CAGE-4** (2024); **CybORG++** (2024); **CyGIL**;
  **FARLAND** (unified emulation–simulation). *Take:* the sim-to-emulation transfer the field
  *asserts* (e.g. CyGIL "full transferability") is **unquantified** — Verisim's opening is to lead
  with a measured `H_ε(ρ)` against a real emulator oracle (SPEC-3 §3.3).
