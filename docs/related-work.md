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
- **V-JEPA 2 / JEPA** (Meta; LeCun, 2025) — an *energy-based* predictive world model:
  predict the future's *representation* (not raw pixels), discarding whatever is
  unpredictable; plan by minimizing a learned goal energy via MPC. *Take:* the cleanest
  oracle-free foil. Its energy is a *learned* surrogate, so a rollout can drift in latent
  space uncaught — Verisim's divergence is the same energy *computed against ground truth*,
  i.e. an energy-based world model whose energy is the oracle. Dual role: also a **component
  source** — its latent-belief + collapse-prevention (VICReg/EMA) are borrowed for the
  partial-observation arm (SPEC-5 §6.2), confined to the *unobserved* part so the checkable
  part stays exact (SPEC.md §8; DD-3, SPEC-3 §4.2). And the question JEPA answers blind —
  *what is safe to discard?* — Verisim answers by intervention against the oracle (SPEC.md §4).
- **NVIDIA Cosmos** (NVIDIA, 2025) — world foundation models for physical-AI
  synthesis. *Take:* visual-synthesis branch; competes on fidelity Verisim does not.
- **World Labs** (2025) — large-scale spatial/world models. *Take:* same no-oracle
  regime; cited as the frontier Verisim is *not* on.

## Model-based RL world models

- **Dreamer (v1–v3)** (Hafner et al.) — learn a latent world model and plan in it.
  *Take:* closest on the "roll out a learned model" mechanic, but the env-as-oracle is
  the thing being modeled *away*, not kept in the loop. Verisim keeps the oracle and
  asks how little of it suffices.

## Landmark-graph planning (the SPEC-12 architecture)

> The goal-conditioned-RL lineage [SPEC-12.md](../specs/SPEC-12.md) adopts: represent the
> world model as a sparse graph of waypoint states, plan landmark-to-landmark, execute short
> hops with a low-level controller. Verisim's contribution is the oracle-verified edge.

- **L3P — World Model as a Graph: Learning Latent Landmarks for Planning** (Zhang et al.,
  ICML 2021, arXiv:2011.12491). Latent landmarks clustered in a learned latent space; edges =
  reachability distilled from Q-learning; hierarchical plan (high-level landmark hops + low-level
  controller); re-plan only when reachability says a subgoal is reached/failed. *Take:* the
  architecture SPEC-12 instantiates — but its edges are *distilled and hoped* because
  continuous-control has no oracle; verisim's are *verified*, the one move that line cannot make.
- **SoRB — Search on the Replay Buffer** (Eysenbach et al., NeurIPS 2019, arXiv:1906.05253).
  Waypoints = buffer states; edge weight = value-function distance; Dijkstra for the route.
  *Take:* the cheap-baseline graph (LP2/LP3) — landmarks from visited states, classical search on top.
- **SPTM — Semi-Parametric Topological Memory** (Savinov et al., ICLR 2018) and **SFL — Successor
  Feature Landmarks** (Hoang et al., NeurIPS 2021, arXiv:2111.09858). Non-parametric graph + a learned
  reachability/similarity metric deciding which states become nodes. *Take:* the landmark-*admission*
  rule (LP5) is a measured knob in verisim, not a fixed heuristic.

## LLMs on graphs (why the LLM stays at the leaves)

- **NLGraph** (Wang et al., NeurIPS 2023, arXiv:2305.10037) and **Talk like a Graph** (Fatemi et al.,
  Google, ICLR 2024). LLM graph-reasoning accuracy degrades with traversal depth and node degree;
  advanced prompting helps less as problems deepen; a positive-answer bias makes models assert paths
  that do not exist. *Take:* the empirical basis for SPEC-12's division of labor — graph search does the
  traversal, the LLM only ever sees one leaf (LP7 measures the boundary under the oracle).

## Attack graphs / network reachability (the defensive application)

- **MulVAL** (Ou et al., USENIX Security 2005) and the lateral-movement-as-graph-walk line. Reachability
  + exploit preconditions compose into multi-host attack paths; the field's standing pain is *unsound*
  graphs (paths the real network does not permit), because the graph derives from a static config model.
  *Take:* SPEC-12's oracle-grounded landmark graph is an attack/reachability graph with **zero false
  paths by construction** (every edge data-plane-verified) — the SPEC §4.1 ACD contribution, defensive
  and human-out-of-loop.

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

## SPEC-13–18 era — the research backlog (June 2026)

> The references that ground the six design-stage specs mapped in June 2026: SPEC-13 (speculative
> rollout), SPEC-14 (neural algorithmic reasoning), SPEC-15 (conformal consultation), SPEC-16
> (rollout-stability training), SPEC-17 (causal/counterfactual), SPEC-18 (the product). Name + venue
> + year per this file's policy; arXiv IDs given only where web-verified.

### Speculative decoding (the SPEC-13 consultation-policy lens)

- **Speculative decoding / speculative sampling** — Leviathan et al. (Google, ICML 2023); Chen et al.
  (DeepMind, 2023). A cheap drafter proposes `k` tokens, the target verifies in one pass and accepts
  the longest matching prefix; output distribution provably unchanged. *Take:* this *is* verisim's
  propose/verify/correct loop lifted from tokens to world-states — with the oracle as an *exact*
  verifier, so "accept the longest correct prefix" is the longest *faithful* prefix (SPEC-13 H39).
- **EAGLE / EAGLE-2 / EAGLE-3** — Li et al. (EAGLE, ICML 2024, arXiv:2401.15077; EAGLE-2,
  arXiv:2406.16858; EAGLE-3, arXiv:2503.01840). Feature-level drafting; draft confidence approximates
  acceptance rate and grows the draft only where confidence is high. *Take:* the theory of when to
  trust a longer draft — verisim calibrates draft length `k` from `belief_var`/decode-entropy against
  ground truth (SPEC-13 H41).
- **SpecInfer** — Miao et al. (ASPLOS 2024, arXiv:2305.09781). Token-tree verification: many candidate
  continuations checked against the target in one shared pass via tree attention. *Take:* a tree of
  candidate rollouts verified against one oracle trajectory raises the accepted prefix at near-zero
  extra oracle cost (SPEC-13 H42).
- **Medusa** — Cai et al. (ICML 2024, arXiv:2401.10774). Parallel draft heads, no separate draft
  model. *Take:* the multi-draft source for SPEC-13's best-of-`m` branch.
- **Lookahead decoding** — Fu et al. (ICML 2024, arXiv:2402.02057). Exact parallel decoding via Jacobi
  iteration, no auxiliary model. *Take:* the self-speculative variant; motivates the two-tier cascade
  and the cost-inversion question (oracle expensive, not GPU) of SPEC-13 H43.

### Neural Algorithmic Reasoning (the SPEC-14 lens)

- **Neural Execution of Graph Algorithms** — Veličković et al. (DeepMind, ICLR 2020, arXiv:1910.10593).
  Encode-process-decode GNNs trained to imitate *every step* of BFS/Bellman–Ford via intermediate
  "hint" supervision. *Take:* end-to-end GNNs do not compose multi-step — step-wise hints are the fix —
  exactly the HS3 symptom (great one-step `p`, `H_free`=0); the oracle emits these hints for free
  (SPEC-14 NA1/H46).
- **The CLRS-30 Algorithmic Reasoning Benchmark** — Veličković et al. (DeepMind, 2022,
  arXiv:2205.15659). Full hint trajectories of 30 algorithms; hints prevent non-generalizable
  single-train-set shortcuts. *Take:* names the HS3 failure as a shortcut and the cure as trajectory
  supervision; verisim's reachability hints are a free, exact CLRS-style trajectory.
- **What Can Neural Networks Reason About?** — Xu et al. (MIT/NTT, ICLR 2020, arXiv:1905.13211).
  "Algorithmic alignment": a net learns efficiently when its computation mirrors the task's algorithm.
  *Take:* the theory for aligning `mp_update`'s aggregation to the oracle's min-hop relaxation (SPEC-14
  NA2/H47).
- **A Generalist Neural Algorithmic Learner (Triplet-GMPNN)** — Ibarz et al. (DeepMind, LoG 2022,
  arXiv:2209.11142). One hint-supervised processor executing many algorithms OOD. *Take:* the
  architectural choices that make a hint-supervised processor OOD-robust.
- **Neural Algorithmic Reasoning Without Intermediate Supervision** — Rodionov & Prokhorenkova (Yandex,
  NeurIPS 2023, arXiv:2306.13411). A self-supervised regularizer *because hints are costly and usually
  unavailable.* *Take:* the inversion that makes SPEC-14 NA1 possible — verisim's hints are free and
  exact, the experiment that field could not cheaply run.
- **Deep Equilibrium Models** — Bai, Kolter & Koltun (NeurIPS 2019, arXiv:1909.01377). *Take:*
  reachability closure is a fixed point, so the processor can "think longer" to convergence at
  inference (SPEC-14 NA3/H48).

### Conformal prediction & calibration (the SPEC-15 lens)

- **Algorithmic Learning in a Random World** — Vovk, Gammerman & Shafer (Springer 2005, 2nd ed. 2022).
  The foundational text; split/inductive conformal gives distribution-free finite-sample coverage for
  exchangeable data. *Take:* the calibrator SPEC-15 wires the consultation threshold to.
- **A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification** —
  Angelopoulos & Bates (2021, arXiv:2107.07511). *Take:* the modern recipe and the marginal-vs-
  conditional caveat behind SPEC-15's depth-conditional coverage reporting.
- **Conformalized Quantile Regression** — Romano, Patterson & Candès (NeurIPS 2019, arXiv:1905.03222).
  *Take:* adaptive-width intervals motivate a region-conditional threshold `τ_α(r)`, since faithful
  horizon is a region property (EN7/H22).
- **Distribution-Free, Risk-Controlling Prediction Sets** — Bates et al. (2021, arXiv:2101.02703) and
  **Conformal Risk Control** — Angelopoulos et al. (2022, arXiv:2208.02814). *Take:* bound *expected
  loss* (undetected faithful-horizon breach), not mere miscoverage (SPEC-15 CF3/H54).
- **Adaptive Conformal Inference Under Distribution Shift** — Gibbs & Candès (NeurIPS 2021,
  arXiv:2106.00170) and **Conformal PID Control for Time Series** — Angelopoulos et al. (NeurIPS 2023,
  arXiv:2307.16895). *Take:* the load-bearing fix for autoregressive rollout drift — rollout states are
  non-exchangeable, so vanilla conformal can lose coverage; ACI/PID restore it (SPEC-15 CF2/H52).

### Rollout-stability training & exposure bias (the SPEC-16 lens)

- **DAgger — A Reduction of Imitation Learning to No-Regret Online Learning** — Ross, Gordon & Bagnell
  (AISTATS 2011, arXiv:1011.0686). Aggregate the *learner's* drifted states with expert labels; cuts
  compounding error `O(εT²)→O(εT)`. *Take:* SPEC-16 RS1 is DAgger with the oracle as expert — the
  expensive-expert bottleneck vanishes because `O(s,a)` is free and exact at any drifted state.
- **Scheduled Sampling** — Bengio et al. (NeurIPS 2015, arXiv:1506.03099) and **Professor Forcing** —
  Lamb et al. (NeurIPS 2016, arXiv:1610.09038). Match train/sample dynamics to fight exposure bias.
  *Take:* the shipped `train_graph_model_self_forced` lever (SPEC-5 §6.3), finally on the exact
  faithful-horizon axis (SPEC-16 RS2).
- **Learning to Simulate Complex Physics with Graph Networks (GNS)** — Sanchez-Gonzalez et al. (ICML
  2020, arXiv:2002.09405) and **MeshGraphNets** — Pfaff et al. (ICLR 2021, arXiv:2010.03409). The
  decisive long-rollout trick is training-time noise injection so the model corrects its own drift.
  *Take:* the lever made *exact* by the free oracle — relabel the noisy state with `O(s̃,a)` (SPEC-16 RS3).
- **Learned Coarse Models for Efficient Turbulence Simulation** — Stachenfeld et al. (ICLR 2022,
  arXiv:2112.15275). *Take:* noise magnitude is a measured knob, read on `η` (the compounding penalty
  the oracle measures exactly).
- **Message Passing Neural PDE Solvers (the pushforward trick)** — Brandstetter, Worrall & Welling
  (ICLR 2022, arXiv:2202.03376) and **PDE-Refiner** — Lippe et al. (NeurIPS 2023, arXiv:2308.05732).
  *Take:* SPEC-16 RS4's multi-step unrolled loss is the pushforward made exact — supervise every
  drifted unrolled step against the oracle, removing the stop-gradient approximation.

### Causal & counterfactual modeling (the SPEC-17 lens)

- **Causality** (Pearl, 2nd ed. 2009) and **The Book of Why** (Pearl & Mackenzie, 2018); **On Pearl's
  Hierarchy and the Foundations of Causal Inference** — Bareinboim et al. (2020). *Take:* the ladder
  (association / intervention / counterfactual) verisim climbs *exactly* — the oracle is an SCM, the
  seed is the exogenous `U`, reset+replay is abduction.
- **Toward Causal Representation Learning** — Schölkopf et al. (Proc. IEEE 2021, arXiv:2102.11107).
  *Take:* the foil — the field *discovers* causal structure; verisim is *given* it by the oracle, so
  the structure-learning bottleneck is skipped (SPEC-17 §2).
- **CoPhy — Counterfactual Learning of Physical Dynamics** — Baradel et al. (ICLR 2020,
  arXiv:1909.12000). *Take:* CoPhy approximates rung 3 with learned confounders; verisim abducts the
  latent *exactly* from the seed (SPEC-17 CX-rung3).
- **Counterfactual Data Augmentation using Locally Factored Dynamics (CoDA)** — Pitis, Creager & Garg
  (NeurIPS 2020, arXiv:2007.02863). *Take:* CoDA's samples are valid *in a learned model*; SPEC-17 CX4
  contrasts that against exact-oracle augmentation valid by construction.
- **Invariant Risk Minimization** — Arjovsky et al. (2019, arXiv:1907.02893). *Take:* the off-policy
  hidden-state axis of H61 is IRM's "environment" axis — which verisim *constructs* exactly by
  intervening on the fault medium rather than hoping it appears in data.

### Benchmark governance & sim-to-emulation transfer (the SPEC-18 lens)

- **BetterBench** — Reuel et al. (NeurIPS 2024, arXiv:2411.12990). 46 lifecycle best practices; the
  implementation/maintenance stages are weakest (most benchmarks ship no replication script or
  standardized metadata). *Take:* the spec sheet for `verisim-bench`; the `reproduce.sh`/seeded
  discipline already meets the replication criterion (SPEC-18 PB-pack).
- **CyGIL — Unified Emulation-Simulation Training Environment** — Li et al. (2023, arXiv:2304.01244).
  Reports a sim-trained agent "transferrable directly… full transferability" to the emulated network.
  *Take:* the canonical **unquantified** transfer assertion — SPEC-18 PB-transfer (H66) replaces it
  with a measured `ΔH` and CI against the SPEC-11 real-OS oracle.
- **CybORG++ / MiniCAGE** — Emerson et al. (2024, arXiv:2410.16324). A CAGE-2 gym with fixes and a
  ~1000× faster sim; sim/emulation behavior reported correlated but not numerically bounded. *Take:*
  the ACD-environment baseline `verisim-acd` improves on by being both cheap *and* oracle-checkable.
- **Croissant — ML-ready dataset metadata** — Akhtar et al. (MLCommons, 2024, arXiv:2407.16883).
  *Take:* the machine-readable metadata standard PB-pack emits for the frozen battery.
- **Inspect / `inspect_evals`** (UK AI Safety Institute) — framework + community evals with a published
  submission standard (scorer unit tests, resource checks). *Take:* the distribution target and quality
  bar `verisim-bench`'s Inspect task is hardened to (already wired in `eval/inspect_task.py`).

## AI control — the frame for the RA arc (SPEC-22 RA16, June 2026)

The home discipline for "deploy a powerful-but-untrusted agent safely" is **AI control**, and RA16
positions the oracle coverage gate in it explicitly. The contribution there is one operating point the
domain allows: a *trusted monitor* that is the environment run as an oracle, so it has no model blind
spot, and whose safety-usefulness frontier is invariant to whether the red team is adaptive.

- **AI Control: Improving Safety Despite Intentional Subversion** — Greenblatt, Shlegeris et al.
  (Redwood Research, arXiv:2312.06942, 2023). Defines the red-team/blue-team control evaluation and the
  safety-usefulness frontier at a given audit budget; trusted monitoring, defer-to-trusted, and
  trusted editing are the canonical blue-team protocols. *Take:* the vocabulary RA16 adopts; the oracle
  gate is a trusted monitor whose verdict is read, not predicted.
- **Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols** — arXiv:2510.09462 (2025). The
  load-bearing recent result: a *model* monitor has a blind spot an adaptive attacker aims at, so no
  audit budget short of auditing-everything reaches safety 1. *Take:* this is RA5/RA11's learned-
  guardrail failure named in the control literature, and exactly the failure the exact-oracle monitor
  (RA16) does not have — the reason its frontier does not move under an adaptive adversary.
- **Evaluating Control Protocols for Untrusted AI Agents** (arXiv:2511.02997, 2025) and **The sum of
  its parts: composing AI control protocols** (Redwood, 2025). *Take:* the protocol-composition and
  evaluation methodology; the oracle gate is one composable trusted check, strongest where exact state
  is on tap.
- **Anthropic Alignment Science** — "adversarial robustness and AI control" is a named research area
  (alignment.anthropic.com; Fellows 2026). *Take:* the audience for the RA16/paper-§8 product reading,
  where a permission system is a control protocol and approval prompts are its audit budget.
