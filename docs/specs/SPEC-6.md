# SPEC-6 — The Host World

**Engineering & experiment specification for the running computer: a process tree
making syscalls over a filesystem and a network — the world that *composes* the
others, where concurrency makes truth hard and a deterministic kernel still makes it
free.**

> **▶ HC0-HC8 STARTED — the deterministic core, both learned arms (flat + factored interaction-graph), the composed loop, the EH1–EH7 figures (both consultation axes π_c × π_w), the concurrency scheduler + the **H14 dial (CONFIRMED)**, **EH7/H22 model-invariance (CONFIRMED in the composed world)**, and HC8's packaging — the §7 LLM-callable simulator, the oracle-as-reward RL env, EH-stream (H15)/EH6-counterfactual (H16), the §16 trustless verified-contribution protocol, and the composed-host Inspect benchmark (2026-06).** The network world is now exhaustively explored (EN1–EN10; the "knee" came back as the honest floor+cliff negative, EN1/H8), and per the program's "a negative licenses the next step" principle (SPEC-2.1 §10) the host world's **deterministic, no-GPU foundation** is buildable now — it makes no learning claim and gates nothing. **HC0 increment 1 + HC1 ship:** the bundle `HostState` (process table + per-process fd tables + the **embedded v0 filesystem**), the syscall grammar (`fork`/`exit`/`setuid`/`open`/`write`/`close`), canonical serialization, the **Tier-A reference host oracle composing the v0 FS sub-oracle**, and the **compositional bundle `Delta` + the `apply == oracle` invariant** (the M1/NW1-analogue: a `write`'s delta embeds the v0 FS sub-oracle's own `Delta` verbatim) ([`verisim.host`](../../src/verisim/host/), [`verisim.hostoracle`](../../src/verisim/hostoracle/), [`docs/host-semantics.md`](../host-semantics.md), property-tested + golden + invariant-tested). **HC2 increment 1 ships** the data factory ([`verisim.hostdata`](../../src/verisim/hostdata/)): three state-reading workload drivers (`uniform`/`forky`/`adversarial`), the `HostConfig` vocabulary, and regenerable trajectory JSONL + manifests/splits — the recorded bundle deltas replay to every next_state (HC1 carried into the dataset). **HC3 ships** the deterministic metric core ([`verisim.hostmetrics`](../../src/verisim/hostmetrics/), [`docs/host-semantics.md`](../host-semantics.md), metric-tested): composed **and per-subsystem** divergence (`proc`/`fd`/`fs`/`global`, the embedded fs reusing v0's `state_facts`), per-subsystem bits-to-correct (the `fs` residual delegated to v0's gate), the **composition-faithfulness diagnostic** (`composition_law` — multiplicative ↔ weakest-link ↔ coupled, the §9.2 headline-new metric / H13), privilege-faithfulness (§9.4), and the `HostRunRecord` schema carrying composed + per-subsystem trajectories — `H_ε` reused verbatim from v0. The interleaving-entropy/chaos dial waits on the scheduler. **HC4 increment 1 ships** the *learned* model's **flat-serializer baseline arm** ([`verisim.hostmodel`](../../src/verisim/hostmodel/)): the closed `HostVocab`, the `(bundle_state, action) → bundle_delta` tokenizer (the embedded FS write delta flattened and reconstructed verbatim, §5.1), the LL(1) grammar + constrained decode reusing v0's `GPT`, the supervised dataset adapter (v0's generic trainer reused), and `NeuralHostWorldModel` — the DD-H1 floor the factored arm must beat, property/overfit-tested + grammar-valid by construction. **HC5 increment 1 ships** the **composed propose-verify-correct loop** ([`verisim.hostloop`](../../src/verisim/hostloop/), the rest of the deterministic core): the model-agnostic runner, the two-mode partial-observation oracle (full / per-subsystem probe, §5.3), the **`π_w` "which-subsystem" policy** (§8.2) and the per-subsystem **`SubsystemFilter`** operator (correct only the observed subsystem — the EH3 no-identity-collapse lever, §8.3), the `HostNullModel`/`HostOracleBackedModel` baselines, populating HC3's composed + per-subsystem `HostRunRecord` — loop-invariant tested (equal-`ρ` budget, spend-down backstop, per-subsystem ≤ full horizon), `π_c` reused verbatim from v0. **HC6 ships the prime directive** ([`eh1.py`](../../src/verisim/experiments/eh1.py), figures [`eh1_curve.png`](../../figures/eh1_curve.png) + [`eh1_composition.png`](../../figures/eh1_composition.png)): the **composed-host `H_ε(ρ)` curve** is the **floor+cliff** shape (drift in <1 step at `ρ=0`, near-flat interior, cliff to `H_ε=T` only at `ρ=1`) — the composed world reproduces v0/EN1's no-favorable-knee result (H8/H22); and the **composition-law measurement (H13) reads `coupled`** — the teacher-forced composed acceptance sits *below* the multiplicative/independence prediction, so the flat baseline's subsystem failures are anti-correlated (coupling is load-bearing, §10.2), the honest negative that licenses the factored arm. **HC7's EH3 ships** the equal-budget operator comparison ([`eh3.py`](../../src/verisim/experiments/eh3.py), figure [`eh3_operators.png`](../../figures/eh3_operators.png)): the full-consult operators coincide on `H_ε` (the full-truth identity), the per-subsystem `SubsystemFilter` arms break that collapse (§8.3), and **per-subsystem consultation earns up to ~3.7× more faithful horizon per oracle-bit** — but the *cheapest* subsystem (fd) wins, **not** the H13-weakest (proc), so the static "target the weakest" heuristic is the thing **smart `π_w`** must beat. **HC4 increment 2 ships the factored interaction-graph arm** ([`hostmodel/graph_model.py`](../../src/verisim/hostmodel/graph_model.py)) — a masked message-passing GNN + RSSM over the process spine's lineage + shared-file edges, graph-conditioned decode under the same grammar — and the **EH4 comparison** ([`eh4.py`](../../src/verisim/experiments/eh4.py), figure [`eh4_factored_vs_flat.png`](../../figures/eh4_factored_vs_flat.png)): the factored arm **beats flat ~6.6× on delta-exact and ~5.3× on composed acceptance** (structure helps — the host H11 echo), **yet both stay `coupled`**, so the H13 coupling is genuine, not a flat-arm artifact. **EH2 ships the program's first smart-`π_c` positive** ([`eh2.py`](../../src/verisim/experiments/eh2.py), figure [`eh2_policies.png`](../../figures/eh2_policies.png)): the flat arm reproduces the standing H2-negative, but the factored arm's **RSSM belief variance makes uncertainty-triggered consultation beat fixed ~2.2×** at equal `ρ` — the §6.2 calibrated signal is the better-localized one (§8.1 confirmed). **EH5 then ships the smart `π_w` axis** ([`eh5.py`](../../src/verisim/experiments/eh5.py)): the factored arm's **per-subsystem decode entropy** drives an `UncertaintySubsystem` policy that verifies the least-certain subsystem — a modest but real edge over round-robin (matches the best raw horizon at lower cost), though the cheapest-fixed still wins pure bit-efficiency. With EH1–EH5 both consultation axes (when `π_c` × which `π_w`) are measured. **The §6.3 drift levers ship** ([`eh4_drift.py`](../../src/verisim/experiments/eh4_drift.py)) and reproduce the network's banked negative — neither noise nor self-forcing buys free-running horizon at this scale, so the one-step→horizon gap stays open (budget routes to scale/objective). **The concurrency scheduler + H14 dial ship and CONFIRM H14** ([`hostdata/scheduler.py`](../../src/verisim/hostdata/scheduler.py), [`eh_h14.py`](../../src/verisim/experiments/eh_h14.py), figure [`eh_h14_interleaving.png`](../../figures/eh_h14_interleaving.png)): a chaos-seeded scheduler interleaves a multi-thread workload (shared-file + interleaved-fork order-sensitivity), and **free-running `H_ε` degrades monotonically with interleaving entropy** (~8× from the recorded regime to chaos, recovered at the low-entropy end) — concurrency is a *measurable dial, not a binary wall*, the first quantification of HW-1's cost and the host world's defining result. **HC8 increment 1 ships the §7 LLM-callable simulator protocol** ([`verisim.hostsim`](../../src/verisim/hostsim/)): `HostSimulator` both predicts the next state (the loop) and *simulates a plan* for an agent — `imagine` (Dreamer-style, oracle-free) + `verify` (a `PlanReport` with the plan-level faithful horizon, the §17.8 `H_ε`-for-a-plan, plus the composing **task oracle** `Goal`, the §7 third oracle). **EH7 confirms H22 in the composed world** ([`eh7.py`](../../src/verisim/experiments/eh7.py), figure [`eh7_invariance.png`](../../figures/eh7_invariance.png)): four proposers in the same loop share the floor+cliff `H_ε(ρ)` shape (proposer sets the floor height, loop sets the shape) — the model-agnostic-primitive claim holds in the hardest world. **Two cross-cutting findings ([`synthesis.py`](../../src/verisim/experiments/synthesis.py), [`eh_h14_scale.py`](../../src/verisim/experiments/eh_h14_scale.py)):** the **cross-world synthesis** overlays all three worlds' normalized `H_ε(ρ)` onto **one floor+cliff** (model- *and* world-agnostic — the thesis in one figure), and **EH-H14-scale** shows the concurrency collapse **steepens with thread count** (~2.5×→~12× from 2→8 threads — concurrency's cost scales with its width). **HC8 then ships its packaging and continual-loop science:** the **oracle-as-reward RL env** ([`hostrl/`](../../src/verisim/hostrl/), the episode return *is* the composed `H_ε`), **EH-stream** ([`eh_stream.py`](../../src/verisim/experiments/eh_stream.py)) — **H15 refuted** but replay load-bearing and the plasticity probe localizes **HW-4** — **EH6-counterfactual** ([`eh6_counterfactual.py`](../../src/verisim/experiments/eh6_counterfactual.py)) — **H16 refuted beyond volume**, the network's EN6/H5 null reproduced world-agnostically — the **§16 decentralized verified-contribution protocol** ([`contrib/`](../../src/verisim/contrib/): contributions are **trustless by re-execution** — accepted iff the oracle reproduces them bit-for-bit, chained and content-addressed), and the **composed-host faithfulness benchmark + Inspect adapter** ([`hosteval/`](../../src/verisim/hosteval/): the §1.4 missing metrology for a *whole machine*, packaged where labs already look). **EH5-heads closes the per-subsystem decode-heads question with a negative** ([`eh5_heads.py`](../../src/verisim/experiments/eh5_heads.py), figure [`eh5_heads.png`](../../figures/eh5_heads.png)): a trained per-subsystem error head is *uncalibrated* to held-out error (Spearman −0.02) where the bucketed decode entropy it was meant to replace is strongly calibrated (+0.57) — the per-subsystem echo of v0's H2 negative, the open HC7 item resolved. Only the **Tier-B system oracle** (rr/Hermit/gVisor; a real-OS dep, deferred under the no-egress posture §15) remains future, gated as ever (DD-4) on a committed figure. Canonical build order: [SPEC §12](./SPEC.md#12-research-roadmap).

This is to the *host* what [SPEC-2](./SPEC-2.md) is to the filesystem and
[SPEC-5](./SPEC-5.md) is to the network. SPEC-2 proved the *method*
(propose-verify-correct against a free deterministic oracle) on a single filesystem;
SPEC-5 promoted it to a multi-host network. SPEC-6 builds the layer *between and
beneath* them — the **operating-system host**: processes, syscalls, file descriptors,
signals, pipes, and the scheduler — because that layer is (a) the literal "computer"
in the user's "network *and computer*", (b) the substrate every computer-use and
cyber-defense agent actually acts on, and (c) the one place where the program's two
existing worlds **compose into a single machine** with a single, still-deterministic
oracle.

It does not restate the science. Read [SPEC.md](./SPEC.md) first for *why*
oracle-grounded world models of computer environments are the one domain where
long-horizon faithfulness is measurable. This document is *how* — for a whole host.

> **Reading order.** Prereqs: [SPEC.md](./SPEC.md) (science), [SPEC-2.md](./SPEC-2.md)
> (filesystem v0), [SPEC-5.md](./SPEC-5.md) (network world). Companions:
> [SPEC-3.md](./SPEC-3.md) (the depth roadmap whose walls **W1/W4** and hard wall
> **HW-1** this spec makes concrete) and [SPEC-4.md](./SPEC-4.md) (the autonomous
> research engine that builds this with the human at the boundary).

**License:** MIT. **Status:** design spec, pre-implementation. **Audience:** the
author, collaborators, reviewers, and the autonomous research engine (SPEC-4), which
reads this file as part of its operating context.

---

## 0. Prime directive

> **SPEC-6's only job is to plot the composed-host `H_ε(ρ)` curve once, cleanly, and to
> measure whether the faithful horizon of a *whole machine* is predictable from the
> faithful horizons of its *parts* — the composition law (H13) — reporting honestly if
> it is not.**

Everything below serves that. SPEC-2's prime directive was "plot `H_ε(ρ)` for a
filesystem"; SPEC-5's was "plot it for a network." SPEC-6's is "plot it for the host
that *contains* both, and find out whether composition is lawful." The difference is
not a bigger world for its own sake — it is the first world where the central object is
**how faithfulness composes**, which is the question that decides whether oracle-grounded
world models scale to whole systems or only to isolated subsystems.

This spec is large because it is exhaustive by request, but the program is staged
(HC0–HC8, §13) so the deterministic core ships and is fully tested **with no runtime
dependencies and no GPU** before any learned model — exactly as M0–M3 did in v0 and
NW0–NW3 did for the network.

---

## 1. Why the host world, and why now

### 1.1 What SPEC-2 and SPEC-5 left between them

SPEC-2 modeled a filesystem; SPEC-5 modeled a network of hosts. Neither modeled the
thing that *creates* filesystem and network state in the first place: a **running
process making system calls**. A `write` to a file, in reality, is a process calling
`openat`/`write`/`close`; a network flow is a process calling `socket`/`connect`/`send`.
The filesystem and the network are the *consequences* of syscalls; the host world is
where the causes live. Three properties of the host world are absent from both prior
worlds and are exactly the properties that make the science bite next:

| Property | v0 filesystem | SPEC-5 network | Host world (SPEC-6) |
|---|---|---|---|
| **Concurrency** | none (one command at a time) | coarse (`advance Δt` only) | **native** — multiple processes interleave; the scheduler is a nondeterminism source |
| **Composition** | one subsystem | one subsystem (replicated across hosts) | **multi-subsystem** — process table + fd table + the FS it touches + the sockets it opens, under one state |
| **Causality depth** | one-step effects | temporally extended | **deep** — a `fork` at step 5 spawns a subtree that touches files and sockets at step 500 |

These are not incremental. Concurrency is the single nondeterminism source SPEC-3
flagged as **the unsolved one** (HW-1: "thread-interleaving is the one source that
record/replay only tames at cost"). Composition is a *new kind of object* — the host
state is not one graph (SPEC-5) or one tree (SPEC-2) but a *bundle of coupled
subsystems*. SPEC-6 is the spec that confronts both.

### 1.2 The deterministic oracle survives — even for concurrency

The reason to go here next rather than to vision/biology/robotics is unchanged
(SPEC.md §2): the oracle stays **free, real, and exact**. A syscall is a deterministic
function of `(state, args)` *given a fixed schedule*. The 2024–2026 record/replay tooling
makes whole-process determinism a solved engineering problem for the single-threaded and
recorded-schedule cases, and a *controllable* one for the concurrent case:

- **rr** records all kernel inputs and nondeterministic CPU effects of a group of
  Linux user-space processes and replays them bit-deterministically (instruction flow,
  registers, memory, syscall results, PIDs).
- **Hermit** (Facebook) *hermetically* isolates time, thread interleavings, and RNG by
  syscall interception, turning nondeterministic execution into repeatable execution —
  and crucially ships a **chaos mode** that explores interleavings *deterministically
  given a seed*. That chaos seed is not noise to be suppressed; it is the **difficulty
  dial** for concurrency (H14).
- **gVisor** gives a user-space syscall-interposition point (a clean place to *be* the
  oracle), and **CRIU** gives process checkpoint/restore (snapshot/reset).

So the host world keeps the one property that makes computer environments special, and
it keeps it *for the hardest nondeterminism source*, by making that source a seeded,
replayable, controllable input rather than an uncontrolled leak. The `determinism_report`
discipline (SPEC-3 §2.2) is mandatory here and declares, per figure, whether scheduling
was single-threaded, recorded, or chaos-seeded.

### 1.3 The foil that proves the thesis: NeuralOS

In 2025, **NeuralOS** (arXiv:2507.08800) did exactly the thing Verisim argues you should
*not* do: it simulates an operating system **generatively**, predicting *screen frames*
with an RNN state-tracker feeding a diffusion renderer, trained on Ubuntu XFCE
recordings. It is impressive and it is the perfect anti-pattern: it models the OS as
*pixels*, has **no oracle**, and therefore can only be eyeballed for faithfulness and
must drift over long horizons exactly as every other pixel-space world model does
(SPEC.md §1). Verisim's host world is the mirror image of NeuralOS along every axis:

| | NeuralOS (2025) | Verisim host world (SPEC-6) |
|---|---|---|
| Prediction target | screen pixels (full-frame) | structured **syscall-effect delta** on checkable kernel state |
| Faithfulness | eyeballed; no ground truth | **bit-for-bit vs. a syscall oracle** (`d`, `H_ε`, bits-to-correct) |
| Drift control | none (autoregressive pixels) | oracle-in-the-loop correction on a budget `ρ` |
| What it's *for* | rendering an interface | being the cheap faithful machine an agent reasons over |

NeuralOS answers "can a neural net *render* an OS?" Verisim answers "can a neural net
*be a faithful, checkable model of what the OS state actually becomes*, and how cheaply
can an oracle keep it so?" The contrast is the cleanest possible statement of the whole
program, now at the OS layer the field's flagship 2025 result chose to model the wrong
way. And the contrast is *about the loop, not the network*: NeuralOS's diffusion renderer
could be swapped for any other generator and it would still drift, because it has no oracle;
Verisim's host proposer can be swapped (factored model, flat serializer, a JEPA-style latent
arm, an LLM) and the composed-host curve and composition law (H13) stay the object of study,
because the oracle — not the proposer — is what holds the result. That proposer-invariance is
the host-layer instance of the project's model-agnostic primitive (SPEC.md §6 commitment 4,
H22).

### 1.4 Why it is worth building for the community, not just for us

A deterministic, oracle-grounded, **composed** host world is a benchmark the field does
not have: a place to measure long-horizon faithfulness of a world model over *a whole
running machine* — process lifecycle, fd state, IPC, and the FS/network it touches —
with ground truth, under controllable concurrency. Computer-use agent benchmarks
(OSWorld, TheAgentCompany) evaluate *task success* with execution-based checkers but
have **no model of, and no metric for, the host's predicted next state** — they grade
the agent, not a simulator of the host. Verisim's contribution is the missing
metrology and the cheap faithful host simulator the agents could call (§7), packaged
where researchers already look (Inspect benchmark + `verifiers`-spec RL env, SPEC-2 §15).
This is the non-competitive contribution: not a bigger agent, but a free, honest
measuring instrument for host-state faithfulness and the verified simulator beneath it.

---

## 2. Lessons folded in (and the design choice each one forces)

The "browse far and wide" deliverable, host-world edition. Each lesson is stated as
design guidance with its source; skeptical notes flag where the literature is hype.
(Citations consolidated in [`docs/related-work.md`](./docs/related-work.md), kept current
by the engine; name + venue + year per that file's no-fabricated-links policy.)

### 2.1 Generative OS simulation — the anti-pattern to avoid

- **NeuralOS (arXiv:2507.08800, 2025).** RNN state + diffusion renderer predicting OS
  screen frames; no oracle (§1.3). → **Design choice:** never predict pixels or raw
  bytes; predict the **structured, canonical kernel state** the OS actually maintains
  (process/fd/signal tables), which is exactly what is checkable and serializable. This
  is the host-layer restatement of v0's E4 finding that *delta* beats *full-state* beats
  *raw reconstruction* — and it is the entire reason the host oracle is possible.

### 2.2 Deterministic record/replay & the concurrency wall

- **rr (rr-project.org), Hermit (Facebook), gVisor, CRIU.** Whole-process determinism by
  interposition and recorded/replayed scheduling; Hermit's **chaos mode** explores
  interleavings deterministically per seed. → **Design choices:** the Tier-B host oracle
  (§5.2) is built on these; concurrency is a **seeded, controllable input** (the chaos
  seed), not an uncontrolled leak; `determinism_report` declares the scheduling regime on
  every figure. **Hard wall (HW-1, inherited):** bit-exact concurrent ground truth costs
  recorded scheduling — so the *science* of concurrency (H14) is run in chaos mode where
  the interleaving entropy is a measured independent variable, not pretended away.

### 2.3 Compositional / factored / object-centric world models

- **COMBO: Compositional World Models (ICLR 2025); Dyn-O (NeurIPS 2025); the
  object-centric / factored-MDP line (Boutilier-style factored MDPs; FIOC-WM).**
  Factorizing dynamics into modular components with a *sparse interaction graph* beats
  monolithic models on compositional generalization, and inferring the per-step
  interaction structure is the active lever. → **Design choices:** the host state is
  modeled as a **factored bundle of objects** (processes, fds, files, sockets, signals)
  with a sparse, per-step **interaction graph** (which process touches which fd touches
  which file); `M_θ` is a factored/object-centric predictor (§6), and the *composition
  law* (H13, §10) is the host-world test of whether faithfulness factors the way the
  state does.

### 2.4 The Era of Experience (continual streams, grounded rewards)

- **Silver & Sutton, "Welcome to the Era of Experience" (DeepMind, 2025).** The next
  capability frontier is agents that learn from **continuous streams of their own
  experience** with **grounded** (environment-derived, not human-labeled) rewards, as
  human data saturates. → **Design choice:** the host world is the natural home of an
  **experience stream** (§8.5) — an endless run of sandboxed host activity from which the
  model predicts, the oracle verifies, and the model heals, forever. The decisive Verisim
  twist on the manifesto: the "grounded reward" is a **deterministic oracle**, not a
  hand-designed or learned reward — the strongest possible grounding, and the one Silver &
  Sutton's physical-world examples cannot have (SPEC.md §2). H15 makes "the stream beats
  the batch" falsifiable.

### 2.5 Streaming / continual learning that actually works

- **"Streaming Deep RL Finally Works" (stream-x, 2024–2025, arXiv:2410.14606);
  "Experience Replay Addresses Loss of Plasticity in Continual Learning"
  (arXiv:2503.20018, 2025).** Streaming/online updates are now viable with the right
  normalization and step-size discipline; the dominant failure mode of long online
  training is **loss of plasticity** (the net slowly stops learning), and *replay* is a
  documented fix. → **Design choices:** the experience-stream loop (§8.5) is built on the
  stream-x recipe and an oracle-correction **replay buffer**; **plasticity loss is named
  as the new hard wall (HW-4)** and probed directly (a frozen plasticity probe alongside
  the divergence probe of SPEC-3 DD-5b). This extends SPEC-3 §6 self-healing from
  *per-rollout TTT* to a *never-ending stream*.

### 2.6 Computer-use agent environments (the payoff, defensively framed)

- **OSWorld (NeurIPS 2024); TheAgentCompany; (Visual)WebArena.** Real-computer tasks with
  reproducible initial states and **execution-based checkers** — a partial,
  task-completion oracle, not a state oracle. → **Design choices:** the downstream framing
  (§7, §15) is **defensive** — SRE, change-safety, incident response, autonomous *defense*
  — never offense; the host world is the substrate these agents act in, and the
  execution-based checker is a *third* (task-level) oracle that composes with the state
  oracle (§2.7, §5). Verisim supplies what these benchmarks lack: a faithfulness metric
  and a cheap verified simulator of the host the agent is about to change.

### 2.7 Symbolic second-oracles for sub-subsystems

- **Formal SQL/transaction semantics (operational semantics under ANSI isolation levels;
  versioned-KV transaction models verified in Coq).** Transactional store state is
  *formally checkable* — a process that does DB/KV I/O has a deterministic symbolic
  oracle for that subsystem, exactly as SPEC-5 used Batfish as a control-plane oracle. →
  **Design choice:** where a subsystem admits a *cheaper symbolic oracle* than running it
  (a KV store's get/put semantics, a permission check, a reachability query), expose it as
  a **second, differently-priced oracle** and let the consultation policy choose which
  oracle to spend — the multi-oracle generalization of SPEC-5 §5.1, and the basis of H13's
  per-subsystem decomposition.

### 2.8 Causal world models (the named open problem, uniquely solvable here)

- **Counterfactual calculus generalizing do-calculus (PMLR 2025); Causal-JEPA; CausalVAE
  for counterfactual dynamics (2025).** Counterfactual validity requires interventional
  *data/structure*, not just predictive accuracy; without it, counterfactual rollouts
  leave the valid manifold. → **Design choice:** the host oracle can *actually run the
  counterfactual* — re-execute a process tree from step `t` with one syscall changed and
  read back the true alternative state — which physical-domain causal world models cannot
  do. This makes the host world the cleanest possible testbed for RQ4/H5; SPEC-6
  operationalizes it as **counterfactual replay** (§9, H16), the host analogue of
  SPEC-5's EN6.

### 2.9 Decentralized verifiable contribution (open-source, trustless)

- **Prime Intellect INTELLECT-2 / PRIME-RL / TOPLOC / DiLoCo (2025, arXiv:2505.07291).**
  Globally distributed RL across *untrusted* contributors works because rollouts are
  **verified** (TOPLOC's locality-sensitive hashing detects tampering) and updates are
  low-communication (DiLoCo). → **Design choice:** the oracle makes contributed host
  trajectories **trustless by construction** — what TOPLOC verifies *heuristically*,
  Verisim verifies *exactly* (re-run the deterministic oracle on the contributed
  `(state, action)` and check the contributed next state bit-for-bit). §16 specifies the
  open, decentralized contribution protocol this enables: anyone can contribute
  oracle-verified host trajectories or whole `Environment`/`Oracle` pairs, and the
  verification is free and certain. This is the concrete form of the user's "open,
  freely-available, decentralized" intent — framed strictly as research methodology and
  community infrastructure, not a service.

### 2.10 Agent memory (long horizons need it)

- **The 2025–2026 agent-memory line (episodic / hierarchical / externalized memory
  surveys).** Long-horizon competence needs explicit memory beyond a context window. →
  **Design choice:** long host rollouts demand memory of **process lineage and file/fd
  provenance** (who forked whom, which fd points at which file across 500 steps). The
  canonical state *is* the memory, but as it grows the model gets a **retrieval/memory
  over prior state** (the deferred SPEC-3 §5.3 option, made concrete here because the host
  world is the first world where it binds). Memory architecture is an EH4 ablation lever,
  not a default.

---

## 3. The world (environment)

### 3.1 State as a bundle of coupled subsystems

The host state `s` is **not** a single tree (SPEC-2) or a single graph (SPEC-5). It is a
*bundle* of typed subsystems sharing references, plus a small global block. This is the
structural novelty: the host state is where the FS and the network become two subsystems
of one machine, joined by the process/fd layer.

```
HostState = {
  procs:   map[pid -> Process]        # the process table (the new core subsystem)
  fds:     map[(pid, fd) -> FdEntry]  # per-process file-descriptor table
  fs:      FilesystemState            # ⊇ the v0 SPEC-2 State (files an fd can point at)
  sockets: map[sock_id -> Socket]     # endpoints into the SPEC-5 network (optional)
  ipc:     { pipes, signals_pending, shm }   # inter-process communication
  global:  { sched_seed, clock_virtual, rng_cursor, last_result }
}
Process = { pid, ppid, state: RUNNING|SLEEPING|ZOMBIE|STOPPED, creds: (uid,gid),
            cwd, env, exit_code? }
FdEntry = File(path_ref) | Pipe(pipe_id, end) | Socket(sock_id) | Dev(name)
```

- **References are the composition.** An `FdEntry` points *into* the FS subsystem (a file)
  or *into* the network subsystem (a socket). The process table is the spine; the FS and
  network hang off it through fds. This is the factored/object-centric structure §2.3
  prescribes, and the sparse interaction graph (which process → which fd → which file) is
  recomputed each step.
- **Canonicalization** (sorted, hashed, volatile-IDs normalized) is mandatory, exactly as
  in SPEC-2 §2.1 and SPEC-5 §3.1. PIDs/inodes/fd numbers are canonicalized so the
  divergence metric measures *competence*, not nondeterministic identifier churn (SPEC-3
  DD-1).
- **The v0 filesystem is embedded verbatim.** `fs: FilesystemState` *is* SPEC-2's `State`.
  SPEC-6 does not refork it; it composes it. Likewise an optional `sockets` subsystem
  composes SPEC-5's network. SPEC-6 is the integrating spec, not a third parallel one.

### 3.2 Actions: the syscall grammar

Actions are **syscalls** (and the scheduler event), in a constrained grammar paired with
the oracle, exactly as v0 pairs a shell grammar with `ReferenceOracle`. The pinned v0
subset (extend only with cause — §17.1 is the boundary call):

```
process:   fork, execve(prog), exit(code), wait(pid), kill(pid, sig)
files:     openat(path, flags) -> fd, read(fd, n), write(fd, token), close(fd),
           dup(fd), lseek(fd, off)            # effects land in the embedded FS subsystem
ipc:       pipe() -> (rfd, wfd), signal-delivery, (later) shmget/shmat
sockets:   socket() -> fd, connect(sock, dst), send(fd, n), recv(fd, n)   # optional; into the net subsystem
creds:     setuid(uid), setgid(gid)           # the privilege-state axis (security core)
sched:     yield / advance(quantum)           # the asynchrony primitive (cf. SPEC-5 advance Δt)
```

- `fork`/`exit`/`wait`/`kill` are the long-range, branching, compounding-state core: a
  `fork` creates a subtree whose later syscalls depend on everything the parent did
  (SPEC-2 §2.4's "long-range dependencies", now via process lineage).
- `sched: yield/advance` is the asynchrony primitive — the host analogue of SPEC-5's
  `advance Δt` and the thing that makes interleaving a variable (H14).
- `setuid/setgid` make **privilege state** first-class, which is where cyber-defense
  faithfulness actually lives (does the model correctly predict that an operation *fails*
  for lack of privilege?).

### 3.3 Determinism contract

Deterministic given `(initial host snapshot, sched_seed, rng_cursor)`. Tier-A (§5.1) is a
pure function. Tier-B (§5.2) pins kernel + image hash, seeds the scheduler (Hermit-style),
disables real egress (§15); residual scheduler nondeterminism is **made a controlled
variable** (chaos seed), not suppressed and forgotten — and `determinism_report` declares
which (§2.2, §17.2). Single-threaded execution is the floor regime; recorded and
chaos-seeded scheduling are the harder regimes, each labeled.

### 3.4 Scale & curriculum

Difficulty is a small set of dials (mirroring SPEC-2 §2.4, SPEC-5 §3.4): number of
concurrent processes, fork depth, fraction of cross-subsystem syscalls (fds spanning
FS *and* sockets), IPC density (pipes/signals), privilege transitions, and — the new
one — **interleaving entropy** (the chaos-mode schedule space the oracle explores). The
curriculum starts at **one process, a handful of file syscalls, single-threaded** (the
host analogue of v0's smallest world and SPEC-5's "2 hosts, 1 link") and ratchets up to
**concurrent process trees doing IPC and I/O under chaos scheduling**. Composition depth
(how many subsystems a trajectory touches) and interleaving entropy are explicit axes
because they are exactly what H13/H14 measure.

---

## 4. The state delta `Δ`

`M_θ` predicts a structured **bundle delta**, not a full state and not raw bytes (§2.1,
and v0's E4 result). The delta vocabulary composes the v0 and SPEC-5 edit types with the
new process/fd/IPC types:

```
ProcSpawn(pid, ppid, prog) / ProcExit(pid, code) / ProcState(pid, st)   # process table
FdOpen(pid, fd, target) / FdClose(pid, fd) / FdDup(pid, fd, newfd) / FdSeek(pid, fd, off)
FileEdit(...)            # the v0 SPEC-2 delta ops, unchanged, on the embedded fs subsystem
SockEvent(...)           # the SPEC-5 connection/flow delta ops, on the embedded net subsystem
PipeWrite(pipe_id, n) / PipeRead(pipe_id, n) / SignalDeliver(pid, sig)   # IPC
CredChange(pid, uid, gid)                                                # privilege
SchedAdvance(quantum) / ContextSwitch(from_pid, to_pid)                  # scheduling event
SetResult(exit_code, errno, stdout_token)                               # observation, as in v0
```

The **M1-analogue invariant** is required and tested: `apply(state, oracle.delta) ==
oracle.next_state` for every transition, by construction, and `apply` is *compositional*
— a bundle delta applies to each subsystem independently and the embedded `fs`/`sockets`
deltas reuse SPEC-2/SPEC-5's `apply` verbatim. Delta↔serialization is round-trippable.
This invariant is the contract that keeps the loop (§8) model-agnostic, as in every prior
world.

---

## 5. The oracle `O`

Two tiers (SPEC-3's system-oracle plan), plus the multi-oracle composition that is the
host world's distinctive feature.

### 5.1 Tier A — reference host oracle (the deterministic core)

A **from-scratch deterministic model of the pinned syscall semantics**: a process table,
per-process fd tables, the embedded v0 filesystem (`ReferenceOracle` from SPEC-2, reused
verbatim as the FS sub-oracle), pipes/signals, optional sockets (SPEC-5's reference DES as
the net sub-oracle), and a **deterministic scheduler** parameterized by `sched_seed`. It
has **no runtime dependencies and needs no GPU**, like M0–M3 and NW0–NW3. It is the
executable truth, paired with a normative `docs/host-semantics.md` (the analogue of
`docs/semantics.md` / `docs/network-semantics.md`). Golden trajectories pin the semantics
and are denylisted from the autoresearch engine (§14).

**The host oracle is a composition of sub-oracles.** This is the structural point: the
host oracle does not reimplement filesystem or network semantics — it *delegates* a
`write(fd→file)` to the FS sub-oracle (SPEC-2) and a `send(fd→socket)` to the net
sub-oracle (SPEC-5), and owns only the process/fd/IPC/scheduling glue. Each sub-oracle is
independently deterministic, free, and already tested. The host oracle's correctness is
the correctness of the parts *plus* the correctness of the composition glue — which is
exactly what H13 measures on the *learned* side.

### 5.2 Tier B — system oracle (reality check)

Real Linux processes under **Hermit/rr-class deterministic execution** (time, RNG, and —
the hard part — thread interleaving sealed or chaos-seeded), syscalls captured at a
**gVisor**-style interposition point, snapshots via **CRIU**, pinned kernel + image hash,
**no real-internet egress** (§15). Tier B attacks SPEC-3 wall **W1** ("the oracle is a
model, not reality") for the host domain and is the *only* place the concurrency wall
(HW-1) is confronted against a real kernel. The Tier-A↔Tier-B gap is itself a reportable
result (the host analogue of SPEC-3 §2.4's differential test), and a curriculum signal
(where our syscall model diverges from Linux is where the data should concentrate).

### 5.3 Partial observation & the multi-oracle menu

The host world inherits SPEC-5's two consultation *modes* and adds the *multi-oracle*
choice:

- **Probe (cheap):** a localized observation — one process's state, one fd's target,
  `/proc`-style sampling, a single subsystem's slice. Partial truth, low cost.
- **Full (expensive):** the complete next bundle state, as in v0.
- **Sub-oracle selection (new):** *which* oracle to spend — the FS sub-oracle, the net
  sub-oracle, the process/scheduling oracle, or a cheaper **symbolic second-oracle** for a
  formally-checkable subsystem (§2.7: a KV-store get/put oracle, a permission-check
  oracle). Each is differently priced. This makes consultation a *two-dimensional* choice
  (when × which-oracle), generalizing SPEC-5's (when × what-to-observe), and it is what
  H13's per-subsystem decomposition needs.

### 5.4 Bits-to-correct over bundle deltas

The scale-free gate metric (SPEC-3 §7, SPEC-4 §5) generalizes directly and, importantly,
**decomposes by subsystem**: `bits-to-correct(Δ̂, Δ) = Σ_subsystem MDL(correction on that
subsystem)`. 0 iff the prediction equals truth on every subsystem; smooth and unfakeable
otherwise. The per-subsystem breakdown is the diagnostic that tells the engine (and H13)
*where* the model's faithfulness is leaking — process table, fds, FS, sockets, or
scheduling — which is the lever a monolithic scalar hides.

---

## 6. The model `M_θ`

### 6.1 Architecture: factored, object-centric, interaction-graph-conditioned

A **factored predictor over the host object bundle** (§2.3), action- and
schedule-conditioned, emitting a structured bundle delta under grammar-constrained
decoding (the constrained-decode machinery from v0's M4 and SPEC-5's NW4 carries over).
Concretely: per-object encoders for processes / fds / files / sockets; a **sparse
interaction graph** recomputed each step (which process touches which fd touches which
file/socket — the COMBO/FIOC-WM bet, §2.3) carrying message passing along *only* the
active references; and a recurrent carry between `SchedAdvance`/context-switch events for
the asynchronous dynamics (the SPEC-5 recurrent-between-events template, §6.1 there).
Heterogeneous magnitudes (byte counters vs. pids vs. flags) pass through a **symlog
transform** (DreamerV3, reused from SPEC-5 §6.1).

> **Design decision (DD-H1): model the composition, do not flatten it.** The host model
> predicts a *bundle* delta with per-subsystem heads sharing the interaction graph, **not**
> a single flat token sequence over a concatenated state. *Rationale:* flattening discards
> the factorization that H13 is about and that the composition literature (§2.3) shows is
> the lever; per-subsystem heads also give the per-subsystem bits-to-correct (§5.4) for
> free. *Alternative considered:* one flat serializer (the v0/SPEC-5 default) — kept only
> as the EH4 baseline arm `M_θ` must beat (H11's host analogue).

### 6.2 Memory over lineage (long-horizon, §2.10)

A retrieval/memory over prior bundle states keyed by **process lineage and fd/file
provenance**, so a syscall at step 500 that depends on a `fork`/`openat` at step 5 can
attend to it without holding all 500 steps in context. The canonical state is the
substrate; the memory is the index over it. Memory presence/size is an EH4 lever, not a
default — adopted only if long-horizon accuracy demands it (DD-4 discipline: measure
before scaling).

### 6.3 Drift mitigations (required ablation levers, inherited)

The SPEC-5 §6.3 levers carry over and are mandatory ablation arms because the host world's
deep causality (§1.1) makes drift compound hard: **noise-injected rollout training** (GNS,
the cheapest highest-leverage one), **self-forcing / scheduled sampling** (close the
train/deploy exposure-bias gap), and the **multi-step / latent-overshoot objective**
(penalize compounding error directly). Each is an on/off lever in EH4 so any horizon gain
is attributable to a specific lesson.

### 6.4 Size & specialization (SLM)

`M_θ` is small and host-specialized by design (the SLM thesis, SPEC-5 §2.1): a fast
faithful *host* simulator, not a generalist. EH4's size axis re-asks v0's capacity
question (v0 answer: *no*, size is not the floor) in the harder, composed, concurrent
world — where the answer may differ, and the honest measurement is the point.

---

## 7. SLM/LLM complementarity — the whole-machine simulator an agent calls

This section answers the user's "complement/integrate SLM/LLM" intent at the layer that
matters: a computer-use or cyber-defense agent acts on a **whole host**, so the host world
is exactly the simulator such an agent needs. The world model is not a competitor to the
LLM; it is the cheap, faithful, verifiable machine the LLM reasons *over*.

- **World model as a sandboxed "what-if" for the agent.** An LLM agent proposes a plan
  (a sequence of host changes — "kill this process, remount this fs read-only, drop this
  firewall rule"). `M_θ` simulates the consequences across all subsystems fast; the oracle
  verifies on a budget. This is propose-verify-correct lifted from the *syscall* level to
  the *plan* level — Dreamer's "plan in imagination," made honest by a real oracle, for the
  thing OSWorld-class agents actually do (§2.6).
- **The execution-checker as a third oracle.** OSWorld-style task checkers (§2.6) are a
  *task-level* oracle that composes with the *state-level* oracle (§5): the state oracle
  says "is the predicted host state faithful?", the task oracle says "would the task have
  succeeded?". The agent can be scored on both, and the model trained against both
  (verifiable reward, §2.7).
- **Distillation from an infinite, perfect teacher.** The oracle emits unlimited correctly
  labeled host trajectories → distill into the SLM. Ground truth at scale is the unfair
  advantage of this domain (SPEC-5 §2.1).
- **Speculative execution / routing.** `M_θ` drafts host dynamics; the oracle verifies only
  when the consultation policy fires (`ρ` is the verification rate — SPEC-3's
  speculative-decoding framing). The LLM is escalated to only for natural-language intent →
  syscall-plan translation, never for simulating dynamics it is bad at.

The integration is a **protocol**, specified here and built in HC8: a `Model` that
implements both "predict next host state" (for the loop) and "simulate a plan" (for an LLM
caller), packaged like v0's `eval`/`rl` modules so external agents can call the verified
host simulator. Open question §17.8 (how to define `H_ε` for a *plan* unit) is inherited
from SPEC-5 §17.8 and is the same problem in this world.

---

## 8. The loop (composed, partially-observed propose-verify-correct)

Same skeleton as SPEC.md §5.2, with the host-world extensions (multi-oracle, per-subsystem
correction, the experience stream).

```
for each step t:
    Δ̂ ← M_θ(belief, a_t)                       # PROPOSE: predict the bundle delta
    ŝ' ← apply(ŝ, Δ̂); belief ← update(belief, Δ̂)
    if π_c decides to consult (budget ρ):                # WHEN to consult
        o ← O(s, a_t, oracle = π_w(belief))              # WHICH oracle/subsystem (§8.2)
        d ← divergence(o, ŝ')                            # partial, per-subsystem, or full
        ŝ', belief ← C(ŝ', belief, o)                    # CORRECT / belief-update (§8.3)
        replay.add(s_t, a_t, o.delta)                    # accumulate the experience stream (§8.5)
        if healer: θ ← heal(θ, replay)                   # SELF-HEAL: gated TTT step (§8.4)
    s ← O.true_next(s, a_t)                              # the host advances regardless
```

### 8.1 Consultation policy `π_c` (when)
The v0/SPEC-5 policies carry over (`fixed`, `drift_triggered`, `uncertainty_triggered`,
`learned`), now reading the factored model's per-subsystem uncertainty (§5.4) — a richer,
better-localized signal than v0's single decode entropy (the H2-negative fix continues).

### 8.2 Oracle/subsystem policy `π_w` (which) — new axis
Given a consultation, *which* oracle/subsystem to spend on: the FS sub-oracle, the net
sub-oracle, the process/scheduling oracle, or a cheap symbolic second-oracle (§2.7, §5.3).
The information-gain choice — verify the subsystem whose predicted delta is least certain
and most consequential — is active oracle-selection. It generalizes SPEC-5's probe-policy
`π_o` (what-to-observe) to *which-truth-source-to-buy*, and it is what makes H13's
per-subsystem budgeting measurable.

### 8.3 Correction / belief operators `C`
`hard_reset`, `residual`, `projection` (onto the nearest *invariant-consistent* host
state — e.g. no fd points at a closed file, no zombie has a live child), and the
SPEC-5 `belief-filter` for the unobserved subsystems. Per-subsystem correction (fix only
the subsystem the oracle was spent on) is native here and is why operators differ (no v0
identity collapse) — EH3.

### 8.4 Self-healing online TTT (optional, gated)
Each consultation is a free, perfectly-labeled example; take a gated gradient step
(SPEC-3 §6.3's recipe: small-lr, replay, PEFT, EMA + trust-region revert under the
keep-if-better referee, DD-5b). Off by default so the static baseline is clean (DD-5).

### 8.5 The experience stream (new — the Era-of-Experience loop)
The host world is long and generative enough that the loop need never stop: a continuous
sandboxed stream of host activity from which the model predicts, the oracle verifies on a
budget, and the model heals from the replay buffer — indefinitely (§2.4). This is the
middle loop of SPEC-3 §6.1 promoted from *one rollout* to *an endless stream*, and it is
the first world large enough to study **plasticity loss (HW-4)**: does an
oracle-grounded model that learns forever keep learning, or does it ossify? The stream
runs the stream-x recipe (§2.5) with the oracle-correction replay buffer fighting both
forgetting *and* plasticity loss. H15 makes "the stream beats the equal-compute batch"
falsifiable; the plasticity probe makes HW-4 measurable, not assumed.

---

## 9. Metrics

### 9.1 Composed divergence `d(s, ŝ)`
Normalized symmetric difference over the *union* of typed subsystem tuples (process, fd,
file, socket, ipc) — i.e. the v0 FS divergence and the SPEC-5 graph divergence summed with
the new process/fd/ipc tuples, canonicalized. `d = 0` iff every subsystem agrees. Reported
**both** as the composed scalar **and** per-subsystem (the §5.4 decomposition), because the
per-subsystem breakdown is what H13 needs and what a single scalar hides.

### 9.2 Composition-faithfulness diagnostic (the headline-new metric)
For each subsystem `i`, the *component* faithful horizon `H_ε^i` (run the loop with only
subsystem `i` active / observed) and the *composed* `H_ε`. The diagnostic is the relation
between them — does `H_ε ≈ min_i H_ε^i` (weakest-link), `≈` a product of per-subsystem
acceptance rates (multiplicative, the speculative-decoding-acceptance view of SPEC-3 §8),
or neither (coupling dominates)? This is the object the prime directive (§0) is about.

### 9.3 Faithful horizon `H_ε(ρ)`
Unchanged in definition (max steps within `ε`), now over composed host state under
controllable concurrency. The headline curve (EH1, HC6).

### 9.4 Bits-to-correct, plasticity, calibration, privilege-faithfulness
- **Bits-to-correct**, per-subsystem (§5.4): the scale-free gate.
- **Plasticity probe** (HW-4, §8.5): on a frozen held-out transition set, has the
  streaming model's *ability to fit a fresh batch* decayed? The continual-learning health
  metric.
- **Calibration**: does the factored model's per-subsystem uncertainty predict its
  per-subsystem error? (Extends SPEC-2 §7.2 / SPEC-5 §9.4; the diagnostic that decides
  whether smart `π_c`/`π_w` *can* work.)
- **Privilege-faithfulness** (security-relevant, §3.2): does the model correctly predict
  permission-denied outcomes? Graded first-class, because a defender's trust in the
  simulator hinges on it getting *failures* right, not just successes.

---

## 10. Hypotheses

SPEC-6 operationalizes two **already-stated** hypotheses that needed a composed,
concurrent, long world to bite, and adds four **new** ones (H13–H16) unique to a host that
composes subsystems under controllable scheduling. Each is falsifiable and names its
honest negative (SPEC.md §9: "the favorable curve might not exist").

### 10.1 Existing hypotheses this spec operationalizes (do not re-coin)
- **H7 (SPEC-3 §13) — correction teaches online.** Self-healing over the *experience
  stream* (§8.5) reduces post-correction divergence within a run, beating `hard_reset` —
  tested in a world long enough, and over a stream long enough, for it to bite (EH5).
- **H5 (SPEC.md §9) — counterfactual lift.** Oracle-grounding improves interventional
  fidelity on identical data, trained on **counterfactual replay** the host oracle makes
  free (re-run a process tree with one syscall changed) — EH6, §2.8.

### 10.2 New hypotheses (H13–H16, non-colliding with H1–H12)
- **H13 — faithfulness composes lawfully.** The composed-host `H_ε` is *predictable* from
  the per-subsystem horizons via a simple law — the candidate is the **multiplicative
  acceptance law**: composed per-step acceptance ≈ ∏_i a_i where `a_i` is subsystem `i`'s
  per-step faithfulness (a step is faithful only if *every* subsystem is). If so, the
  weakest subsystem and the *number* of subsystems both shorten the horizon, and the
  optimal `π_w` (§8.2) spends the oracle on the subsystem with the lowest `a_i` — a
  derivable policy. *Honest negative:* composition is *not* lawful — cross-subsystem
  coupling (an fd that ties FS and process state) makes `H_ε` un-predictable from the
  parts, in which case "model subsystems independently" is the wrong bet and the finding
  reshapes the architecture.
- **H14 — concurrency is the determinism wall, and it is a measurable dial.** Under
  chaos-mode scheduling (§2.2), faithful horizon degrades monotonically with **interleaving
  entropy**, and recorded-schedule determinization recovers it. I.e. concurrency is not a
  binary "deterministic or not" but a continuum the chaos seed sweeps, and the curve
  `H_ε(interleaving-entropy)` is the first quantification of HW-1's cost. *Honest negative:*
  interleaving entropy does not affect `H_ε` at this scale — the model learns
  schedule-invariant effects and concurrency is a non-issue here (which would itself be a
  useful, surprising result).
- **H15 — the experience stream beats the batch.** A streaming, oracle-grounded,
  continually-healing model (§8.5) sustains longer `H_ε` than a batch-trained model at
  **equal total compute** (the Era-of-Experience claim, §2.4, made falsifiable and
  oracle-grounded). *Honest negative:* the stream loses to the batch because of plasticity
  loss / forgetting (HW-4) — the manifesto's promise does not survive contact with this
  oracle at this scale, and the plasticity probe (§9.4) localizes why.
- **H16 — the host oracle uniquely trains counterfactual fidelity.** Training on
  counterfactual replays (the oracle re-running history with one syscall changed —
  something physical-domain causal world models cannot do, §2.8) yields strictly higher
  interventional fidelity than training on factual rollouts alone, at equal data.
  *Honest negative:* counterfactual data adds nothing over factual — the factual
  distribution already covers the interventions that matter.

### 10.3 Outcome → implication: where each host result routes the program

Per the epistemic engine (SPEC.md §10.1), each hypothesis is pre-registered to a forward move on *both*
branches. The host world's distinctive feature is that **its honest negatives are often the more valuable
result**, because they characterize *coupling* and *concurrency* — the two phenomena the simpler worlds
could not exhibit — and the oracle makes those characterizations bankable.

- **H13 (composition law).** *Confirmed* → the multiplicative acceptance law holds, the optimal `π_w`
  (spend the oracle on the weakest subsystem) is *derivable*, and "model subsystems independently" is
  validated — a clean, reusable result. *Refuted* → composition is **not** lawful because cross-subsystem
  coupling (an fd tying FS and process state) makes `H_ε` un-predictable from the parts. This is not a
  setback: it is the **discovery that coupling is the load-bearing structure**, which *reshapes the
  architecture* toward interaction-graph conditioning (§6.1) — the refutation literally tells us what to
  build next. The limitation is the contribution.
- **H14 (concurrency is a measurable dial). ✓ CONFIRMED (HC7, [`eh_h14`](../../src/verisim/experiments/eh_h14.py)):**
  `H_ε(interleaving-entropy)` falls monotonically (~8×, recorded→chaos) and the recorded/low-entropy
  schedule recovers horizon — the first quantification of HW-1's cost, a knob not a wall. (The refuted
  branch — schedule-invariant effects, concurrency a non-issue — would have been a *surprising* result
  that would simplify every downstream world.) Both branches turn the field's
  vague "concurrency is hard" into a number.
- **H15 (experience stream beats batch). ✗ REFUTED at smoke scale (HC8, [`eh_stream`](../../src/verisim/experiments/eh_stream.py)),
  with the mechanism localized.** At equal compute the stream loses to the batch (one-step exact 0.47 vs
  0.54, free-running `H_ε` 1.7 vs 4.0) — the Era-of-Experience claim does not survive contact with the
  oracle here. *But the controlled arms turn the negative into the more valuable result:* **experience
  replay is decisively load-bearing** (it rescues the stream from collapse, 0.47 vs the no-replay 0.10),
  and the **plasticity probe (§9.4) localizes HW-4** — the no-replay stream's plasticity decays to 0.77
  vs 0.95 for the batch/replay arms, the first *grounded* measurement that the forgetting-prone regime
  ossifies and that replay (§2.5) is its fix. The precise negative the continual-learning field needs.
- **H16 (counterfactual training is unique to the oracle). ✗ REFUTED beyond volume (HC8, [`eh6_counterfactual`](../../src/verisim/experiments/eh6_counterfactual.py)),
  world-agnostically.** Training on free oracle counterfactual branches *does* beat the base trajectory
  on held-out intervention-exactness (0.46 vs 0.34) but **loses to a matched-volume control** (0.59) — so
  the lift is data volume, not counterfactual structure: for plain next-state supervision a counterfactual
  is just another labeled transition, and the factual distribution already covers the interventions that
  matter. This *bounds* how much counterfactual augmentation buys, and — crucially — it **reproduces the
  network world's identical EN6/H5 null** (counterfactual *examples* don't lift plain *supervision*,
  though counterfactual *negatives* lifted the *contrastive* representation, EN9), making the H16 null a
  property of the oracle-grounded method, not a host quirk.

The throughline: in the host world a wall, once hit, does not stop the program — it *names the next
problem* (coupling → interaction graphs; concurrency → determinization dials; forgetting → plasticity
probes). We climb by metabolizing the obstacle, never by pretending it isn't there.

---

## 11. Walls (relative to SPEC-3 / SPEC-5)

SPEC-6 makes concrete SPEC-3's wall **W4** ("the model is static") via the experience
stream (§8.5) and **W1** ("the oracle is a model") via Tier-B (§5.2), and it inherits
SPEC-5's **W5** ("effects are synchronous/one-step") — now sharpened by true concurrency.
It adds one genuinely new wall and names two hard walls:

- **W6 — the state is a single subsystem.** Both prior worlds modeled one kind of thing (a
  tree; a graph). The host world's state is a *bundle of coupled subsystems*, and whether
  faithfulness composes (H13) is the new central question. W6 is what the prime directive
  attacks.
- **HW-1 (inherited, now central) — concurrency.** The unsolved determinism source
  (SPEC-3). SPEC-6 does not pretend to solve it; it makes it a *measured dial* (H14).
- **HW-4 (new) — loss of plasticity.** A model that learns forever (§8.5) may stop
  learning (§2.5). The new hard wall of the continual loop; probed, not assumed (§9.4).

---

## 12. Experiments (EH-series)

Non-colliding with E1–E4, the reserved E5/E6 (SPEC-2 §9), and EN1–EN9 (SPEC-5). The host
suite is its own namespace, **EH1–EH6**. Each mirrors a prior experiment's role and names
the hypotheses it tests (§10). Every figure regenerates from config + seeds (the
`figures/reproduce.sh` discipline) and **negative results are first-class** (the repo norm).

- **EH1 — the composed-host `H_ε(ρ)` curve** (role of E1/EN1; the prime directive, HC6).
  Sweep `ρ × ε × difficulty × interleaving-entropy`. Bootstrap-CI aggregation. *Does the
  knee appear, and where does composition put the floor?*
- **EH2 — when × which-oracle policies** (role of E2/EN2): cross `π_c` (when) with `π_w`
  (which subsystem/oracle, §8.2), at equal budget. *Does spending the oracle on the
  weakest subsystem (the H13-derived policy) beat fixed/uniform?*
- **EH3 — composed correction operators** (role of E3/EN3): `hard_reset` vs `residual` vs
  `projection` (invariant-consistent) vs `belief-filter`, per-subsystem. *Do operators
  differ now that correction is per-subsystem and observation partial (no v0 identity), and
  does correction teach over the stream — H7?*
- **EH4 — composition & drift ablation** (role of E4/EN4): factored/object-centric vs flat
  serializer (H11's host analogue, DD-H1); interaction-graph on/off; memory-over-lineage
  on/off; noise-injection / self-forcing / multi-step on/off; size; **composition depth**
  (1 → all subsystems) and **interleaving entropy** (single-threaded → chaos) as the two
  new sweep axes. *Which lesson buys horizon, and how does `H_ε` fall with composition
  depth — H13 — and with interleaving entropy — H14?*
- **EH5 — the experience stream** (role of E4 objective axis): batch-supervised vs
  +streaming-self-heal vs +RLVR, with the **plasticity probe** (§9.4) tracked throughout.
  *Does the stream beat the batch at equal compute — H15 — or does plasticity loss
  (HW-4) sink it?*
- **EH6 — counterfactual & two-oracle grounding** (H5, H16): train with counterfactual
  replays (§2.8); add a symbolic second-oracle for a formally-checkable subsystem (§2.7).
  *Does counterfactual data lift interventional fidelity (H16), and does a cheap symbolic
  sub-oracle lower bits-to-correct more than the data-plane oracle alone (the host analogue
  of SPEC-5's H12)?*

**External harness (optional, for community legibility).** Where cheap, EH1/EH3/EH5 are
additionally reported against an **OSWorld / TheAgentCompany**-class change-safety or
incident-response scenario (§2.6) — a host-task gym the computer-use community already
trusts — with the speedup-vs-fidelity Pareto (model alone → model + budgeted oracle → full
oracle) the way learned simulators report speedup vs. ground truth. Defensive tasks only
(§15).

---

## 13. Milestones (HC0–HC8)

SPEC-6 is the buildable expansion of SPEC-3 §15's host/composition concern and the
integration of SPEC-2 and SPEC-5. The `HC` series is to the host what `M0–M8` were to the
filesystem and `NW0–NW8` to the network: the same staging discipline, deterministic core
first (HC0–HC3, **no runtime deps, no GPU**), learned model after. It does not collide with
`M0–M8`, `S1–S6`, `AR0–AR5`, or `NW0–NW8`.

| Milestone | What | Gate |
|---|---|---|
| **HC0** | Host env: bundle `State` (procs/fds/fs/sockets/ipc), syscall grammar, canonical serialization, **Tier-A reference host oracle** composing the SPEC-2 FS sub-oracle + deterministic scheduler + `docs/host-semantics.md` + golden trajectories | property tests + goldens — ◐ **increment 1 shipped** ([`host/`](../../src/verisim/host/), [`hostoracle/`](../../src/verisim/hostoracle/), [`test_host_oracle.py`](../../tests/test_host_oracle.py)): procs (fork/exit) + per-process fds (open/write/close) + setuid privilege gating + the FS-composition write-through, property-tested + golden. Sockets/IPC/scheduler/cwd deferred |
| **HC1** | Bundle `Delta` types, compositional `apply`, delta↔serialization; the `apply == oracle` invariant (reusing SPEC-2/SPEC-5 `apply` for embedded subsystems) | invariant tests — ✓ **shipped** ([`host/delta.py`](../../src/verisim/host/delta.py), [`test_host_delta.py`](../../tests/test_host_delta.py)): the `ProcSpawn`/`ProcExit`/`FdOpen`/`FdClose`/`CredChange`/`SetExit` edits + the composition `FsDelta` (wraps a v0 `Delta`, applied by the v0 `apply`); compositional `apply` builds a fresh state; the `apply == oracle` M1-analogue + serialization round-trip pinned over a mixed trajectory |
| **HC2** | Drivers (workload generators: fork trees, I/O, IPC, privilege), trajectory JSONL, manifests/splits, the **interleaving-entropy / chaos dial** | data tests — ◐ **increment 1 shipped** ([`hostdata/`](../../src/verisim/hostdata/), [`host/config.py`](../../src/verisim/host/config.py), [`test_host_data.py`](../../tests/test_host_data.py)): three state-reading workload drivers (`uniform`/`forky`/`adversarial`) over the HC0 grammar, `HostConfig` vocabulary, trajectory JSONL + regenerable manifests/splits (by-trajectory, no leakage), the recorded bundle deltas replay to every next_state (HC1-in-the-data). IPC/socket drivers + the interleaving-entropy/chaos dial deferred with the scheduler |
| **HC3** | Composed divergence `d` (+ per-subsystem), composition-faithfulness diagnostic, `H_ε`, bits-to-correct (per-subsystem), privilege-faithfulness, run-record schema | metric tests — ✓ **shipped** ([`hostmetrics/`](../../src/verisim/hostmetrics/), [`test_host_metrics.py`](../../tests/test_host_metrics.py)): composed + per-subsystem divergence (`proc`/`fd`/`fs`/`global`, the embedded fs reusing v0's `state_facts`); per-subsystem bits-to-correct (the `fs` residual delegated to v0's gate); the **composition-law diagnostic** (`composition_law` — multiplicative ↔ weakest-link ↔ coupled, §9.2/H13); privilege-faithfulness (denied/allowed agreement, §9.4); `HostRunRecord` carrying composed + per-subsystem trajectories; `H_ε` reused verbatim from v0. The chaos/interleaving axis arrives with the scheduler |
| **HC4** | `M_θ`: factored/object-centric model + interaction graph + (optional) lineage memory, constrained bundle-delta decode, supervised training (SLM-sized) | model tests (torch extra) — ◐ **increment 1 shipped** ([`hostmodel/`](../../src/verisim/hostmodel/), [`test_host_model.py`](../../tests/test_host_model.py)): the **flat-serializer baseline arm** (the DD-H1 floor `M_θ` must beat) — the closed `HostVocab` over the bundle DSL (pids/fds/uids/paths/content/exits, the unbounded pid/fd families bounded by sized pools), the `(bundle_state, action) → bundle_delta` tokenizer with the embedded FS write delta **flattened** (and reconstructed verbatim, §5.1), the LL(1) `HostDeltaGrammar` + constrained decode (grammar-validity is structural, not learned) reusing v0's `GPT`, the supervised dataset adapter (the generic v0 trainer reused), and `NeuralHostWorldModel` (the HC5 loop's drop-in). Tested: tokenizer round-trip over all six syscalls, decode is always grammar-valid from an untrained model, the tiny host overfits to ~0 loss and free-runs the training deltas, and `apply(state, M_θ(state, a))` stays a valid bundle state (the M1-analogue survives the learned proposer). **Increment 2 shipped** ([`hostmodel/graph.py`](../../src/verisim/hostmodel/graph.py), [`graph_model.py`](../../src/verisim/hostmodel/graph_model.py), [`graph_train.py`](../../src/verisim/hostmodel/graph_train.py), [`test_host_graph_model.py`](../../tests/test_host_graph_model.py)): the **factored interaction-graph arm** (the DD-H1 alternative) — a torch-free process-interaction featurization (process-indexed nodes + a validity mask for the variable-size table, the **lineage** fork-tree edges + the **shared-file** coupling edges that fold the fd/fs subsystems onto the process spine), a masked message-passing GNN + RSSM belief (the §6.2 calibrated uncertainty), and a graph-conditioned decoder under the **same** `HostDeltaGrammar`/`parse_target` as the flat arm, dropping into the HC5 loop unchanged. The **EH4 comparison** ([`eh4.py`](../../src/verisim/experiments/eh4.py), figure [`eh4_factored_vs_flat.png`](../../figures/eh4_factored_vs_flat.png)) is the H13 follow-up: **the factored arm beats flat ~6.6× on delta-exact (0.058→0.388) and ~5.3× on composed acceptance (0.075→0.396)** — structure helps (the host EN4/H11 echo) — **but both stay `coupled`** (factored composed still below its independence floor), so the coupling H13 found is genuine, not a flat-arm artifact. Lineage memory remains future; the §6.3 drift levers and the (opt-in) per-subsystem decode heads ship in HC7 |
| **HC5** | Composed propose-verify-correct loop with multi-oracle consultation (`π_c` × `π_w`), per-subsystem operators, experience-stream + replay scaffolding, baselines | loop invariants — ◐ **increment 1 shipped** ([`hostloop/`](../../src/verisim/hostloop/), [`test_host_loop.py`](../../tests/test_host_loop.py)): the model-agnostic composed runner (two rollouts in lockstep), the **two-mode partial-observation oracle** (full / per-subsystem probe, §5.3), the **`π_w` subsystem policy** (which truth-source to buy — `FixedSubsystem`/`RoundRobinSubsystem`, §8.2), the full operators (`HardReset`/`Residual`/`Projection`) **and the per-subsystem `SubsystemFilter`** (correct only the observed subsystem — the EH3 no-identity-collapse lever, §8.3), and the `HostNullModel`/`HostOracleBackedModel` baselines. `π_c` reused verbatim from v0; the run records populate HC3's composed + per-subsystem `HostRunRecord`. Loop invariants tested: `ρ=1` full-consult reproduces the oracle (`H_ε=T`), the perfect model never drifts at `ρ=0`, the budget is never exceeded and the spend-down backstop spends it exactly, and a per-subsystem consult corrects strictly less than a full one (horizon no greater at equal `ρ`). The **experience-stream + replay scaffolding** (the §8.5 Era-of-Experience loop, paired with gated self-healing TTT) and the **info-gain `π_w`** remain future |
| **HC6** | **EH1 composed-host `H_ε(ρ)` curve** + the **composition-law measurement (H13)** + bootstrap-CI aggregation + figure | **the prime directive** — ✓ **shipped** ([`eh1.py`](../../src/verisim/experiments/eh1.py), [`plot_eh1.py`](../../figures/plot_eh1.py), [`test_eh1.py`](../../tests/test_eh1.py); figures [`eh1_curve.png`](../../figures/eh1_curve.png) + [`eh1_composition.png`](../../figures/eh1_composition.png)): trains the flat HC4 `M_θ`, sweeps `ρ × ε × difficulty × seed` through the HC5 loop (full mode), and reads two results off the records. **(1) The composed `H_ε(ρ)` curve is the floor+cliff shape** — `ρ=0` drifts in <1 step (the honest floor), a near-flat interior, and the cliff to `H_ε=T` only at `ρ=1`: the composed host world reproduces the no-favorable-knee result of v0/EN1 (H8/H22 holds in the bundle). **(2) The composition law (H13) is `coupled`** — the per-step (teacher-forced) composed acceptance (0.04–0.11) sits *below* the multiplicative/independence prediction (0.21–0.24), so the flat baseline's subsystem failures are **anti-correlated**: modeling subsystems independently is the wrong bet, coupling is load-bearing (§10.2). This is the negative that licenses the factored interaction-graph arm (HC4 incr-2). Bootstrap-CI aggregation reuses v0's verbatim |
| **HC7** | Smart `π_w` + per-subsystem operators + drift mitigations + the **chaos/interleaving experiment (H14)**; EH2/EH3/EH4 (equal-budget, CIs) | comparison figures — ◐ **EH3 shipped** ([`eh3.py`](../../src/verisim/experiments/eh3.py), [`test_eh3.py`](../../tests/test_eh3.py); figure [`eh3_operators.png`](../../figures/eh3_operators.png)): the equal-budget operator comparison at fixed `ρ`. The three full-consult operators (`hard_reset`/`residual`/`projection`) **coincide on `H_ε`** (the full-truth identity); the per-subsystem `SubsystemFilter` arms correct strictly less (lower `H_ε`) at strictly fewer oracle-bits, so the v0 identity collapse **breaks** (§8.3) and the cost lens is the headline: **per-subsystem consultation earns up to ~3.7× more faithful horizon per oracle-bit** than full (`subsystem_fd` 0.054 vs full 0.015). Honest nuance — the *cheapest* subsystem (fd) wins on bits, **not** the H13-*weakest* (proc): the static "target the weakest" heuristic loses, which is exactly what the **smart `π_w`** (cost-vs-consequence, the remaining HC7 work) must beat. **EH4** (flat-vs-factored, H11's host analogue) shipped with HC4 increment 2 above. **EH2 shipped** ([`eh2.py`](../../src/verisim/experiments/eh2.py), [`test_eh2.py`](../../tests/test_eh2.py); figure [`eh2_policies.png`](../../figures/eh2_policies.png)): the consultation-policy comparison (H9's host analogue) at equal `ρ`, both arms × {`fixed`,`uncertainty`,`drift`}. **The result is the program's first smart-`π_c` positive**: the **flat** arm reproduces the standing **H2-negative** (uncertainty/drift *worse* than fixed — decode entropy mis-localizes error), but the **factored** arm's **RSSM belief variance fixes it** — uncertainty-triggered earns **~2.2× more faithful horizon than fixed** (5.8 vs 2.6), confirming the §8.1 conjecture that the calibrated-by-construction signal (§6.2) is the better-localized one. **EH5 ships the smart `π_w` axis** ([`eh5.py`](../../src/verisim/experiments/eh5.py), [`test_eh5.py`](../../tests/test_eh5.py); figure [`eh5_subsystem_policy.png`](../../figures/eh5_subsystem_policy.png)): the factored arm now exposes **per-subsystem decode entropy** (bucketed by the op's subsystem, §5.4), feeding an `UncertaintySubsystem` policy that verifies the least-certain subsystem (the §8.2 information-gain choice). At equal `ρ` it matches the best raw horizon (ties `fixed_proc`, beats `round_robin`) and beats `round_robin` per-bit — a **modest but real edge for adaptive targeting** — though the *cheapest*-fixed (`fd`) still wins pure bit-efficiency (EH3's cost-vs-consequence tension persists; raw-horizon CIs overlap at smoke scale). **The §6.3 drift levers ship** ([`graph_train.py`](../../src/verisim/hostmodel/graph_train.py): oracle-relabeled noise injection + self-forcing/scheduled sampling; [`eh4_drift.py`](../../src/verisim/experiments/eh4_drift.py), figure [`eh4_drift.png`](../../figures/eh4_drift.png)) — and reproduce the network's **banked negative**: neither lever buys free-running `H_ε` at this scale (flat ~2–3 steps), and noise slightly *lowers* one-step delta-exact (0.39→0.33), so the one-step→horizon gap is not closed by input-distribution patches (it routes budget to scale/objective, as the network found). **The H14 chaos experiment ships and CONFIRMS H14** ([`hostdata/scheduler.py`](../../src/verisim/hostdata/scheduler.py): the concurrency scheduler + interleaving-entropy dial; [`eh_h14.py`](../../src/verisim/experiments/eh_h14.py), figure [`eh_h14_interleaving.png`](../../figures/eh_h14_interleaving.png)): a workload of independent threads (shared files → fs order-sensitivity; interleaved forks → proc order-sensitivity) is interleaved by a chaos-seeded scheduler, the factored arm is trained on the recorded (sequential) regime, and free-running `H_ε` is swept across the chaos dial. **Faithful horizon degrades monotonically with interleaving entropy** — `H_ε` falls ~8× from the recorded regime (12.5 steps) to chaos (1.5), and the low-entropy/recorded end recovers it: **concurrency is a measurable dial, not a binary wall — the first quantification of HW-1's cost.** **EH7 confirms H22 in the composed world** ([`eh7.py`](../../src/verisim/experiments/eh7.py), [`test_eh7.py`](../../tests/test_eh7.py); figure [`eh7_invariance.png`](../../figures/eh7_invariance.png)): four proposers (null / flat / factored / oracle-backed) dropped into the **same** HC5 loop share the **floor+cliff `H_ε(ρ)` shape** — the proposer sets the floor *height* (factored 2.3 > flat 0.4 > null 0.0 at `ρ=0`, the EH4 ordering) while the loop sets the *shape* (flat interior, cliff to `H_ε=T` only at `ρ=1`); oracle-backed is the flat ceiling. The program's deepest claim — **deterministic verification as a model-agnostic primitive** — holds in the hardest (coupled, concurrent) world too. **EH5-heads closes the per-subsystem decode-*heads* question with a negative** ([`eh5_heads.py`](../../src/verisim/experiments/eh5_heads.py), [`test_eh5_heads.py`](../../tests/test_eh5_heads.py); figure [`eh5_heads.png`](../../figures/eh5_heads.png)): a trained per-subsystem head (opt-in `per_subsystem_heads`, regressing the decoder's *own* per-subsystem teacher-forced error so it predicts which subsystem it will get wrong directly — the calibrated alternative to bucketing decode entropy post-hoc) is **uncalibrated to held-out per-subsystem error** (Spearman −0.02) where the **bucketed entropy it was meant to replace is strongly calibrated** (Spearman +0.57), robustly across noise levels; so in the equal-`ρ` `π_w` comparison the entropy-driven `uncertainty` arm earns the most faithful horizon while the head-driven arm spends the most bits for the least. Mechanism: the head's CE target collapses to ~0 on the (overfit) training distribution, so it learns nothing about the deploy-time divergence the entropy — measured *on the actual constrained decode* — tracks directly. The per-subsystem echo of v0's H2 negative; the next lever is a head trained on the deploy-time (drift) divergence, or scale |
| **HC8** | Experience-stream + plasticity probe (EH5/H15), counterfactual replay (EH6/H5/H16), **Tier-B system oracle** (rr/Hermit/gVisor), the **LLM-callable whole-machine simulator protocol** (§7), the **decentralized verified-contribution protocol** (§16), Inspect benchmark + `verifiers`-spec host RL env, technical report | packaging + report — ◐ **increment 1 shipped** ([`hostsim/`](../../src/verisim/hostsim/), [`test_hostsim.py`](../../tests/test_hostsim.py)): the **LLM-callable whole-machine simulator protocol** (§7). `HostSimulator` packages the loop's `M_θ` + the oracle into the object an agent calls — it both *predicts the next host state* (the loop interface, reused) and *simulates a plan*: `imagine` rolls `M_θ` over a proposed syscall plan with **no oracle** (Dreamer-style "plan in imagination", the cheap draft), and `verify` runs the model's imagination against the oracle step-by-step, returning a `PlanReport` with the predicted-vs-true final state, the **plan-level faithful horizon** (the §17.8 `H_ε`-for-a-plan — how many leading plan steps the agent can trust the model), the oracle cost, and — composing the **task oracle** (`Goal`, the §7 "third oracle") — whether the plan achieves a goal and whether the model *agrees with the oracle* on that. Propose-verify-correct lifted from the syscall level to the plan level, dependency-free except the model. **Increment 2 ships the oracle-as-reward RL env** ([`hostrl/`](../../src/verisim/hostrl/), [`test_hostrl.py`](../../tests/test_hostrl.py)): the host analogue of v0's `rl/` — a `verifiers`-spec reset/step env whose reward *is* a faithful step and whose episode **return equals the composed `H_ε`** (no learned reward model in the loop), with the `load_environment` discovery entrypoint. **Increment 3 ships EH-stream (H15 / HW-4)** ([`eh_stream.py`](../../src/verisim/experiments/eh_stream.py), [`test_eh_stream.py`](../../tests/test_eh_stream.py), figure [`eh_stream.png`](../../figures/eh_stream.png)): the §8.5 experience-stream-vs-batch comparison at equal compute, with the plasticity probe — **H15 refuted** (the stream loses to the batch), **but replay is load-bearing** (rescues it from collapse) and the **plasticity probe localizes HW-4** (no-replay plasticity 0.77 vs 0.95). **Increment 4 ships EH6 counterfactual (H16)** ([`eh6_counterfactual.py`](../../src/verisim/experiments/eh6_counterfactual.py), [`test_eh6_counterfactual.py`](../../tests/test_eh6_counterfactual.py), figure [`eh6_counterfactual.png`](../../figures/eh6_counterfactual.png)): the free oracle counterfactual-replay arm — **H16 refuted beyond volume** (counterfactual training beats the base trajectory but loses to a matched-volume control), reproducing the network's EN6/H5 null world-agnostically. **Increment 5 ships the §16 decentralized verified-contribution protocol** ([`contrib/`](../../src/verisim/contrib/), [`test_contrib.py`](../../tests/test_contrib.py)): the concrete form of the program's open/decentralized intent — the oracle makes contributed host data **trustless by construction**. `verify_transition` / `verify_trajectory` accept a contributed `(state, action, next_state[, delta, observation])` iff **re-running the deterministic oracle reproduces it bit-for-bit**; trajectories must also *chain* (`next_state[i] == state[i+1]`) so individually-valid transitions cannot be spliced; `content_address` gives the corpus its tamper-evident SHA-256 manifest hash; and hostile input is *rejected, never raised*. What TOPLOC verifies *heuristically* (§2.9), the oracle verifies *exactly* — the part of §16 that is free and certain (DiLoCo merging stays deferred, §16). Required the new `from_canonical_host` serialization inverse (round-trip tested). **Increment 6 ships the composed-host faithfulness benchmark + Inspect adapter** ([`hosteval/`](../../src/verisim/hosteval/), [`test_hosteval.py`](../../tests/test_hosteval.py)): the host analogue of v0's `eval/` (§1.4's missing metrology) — `score_host_model` grades any `HostModel` through the HC5 composed loop (composed `H_ε`, oracle calls), `host_step_labels`/`grade_host_prediction` are the single-step QA form, and the lazily-imported `host_faithfulness_task` packages it as an `inspect_ai` task so the *whole-machine* benchmark slots into the framework labs already use. The **per-subsystem decode *heads*** shipped in HC7 (EH5-heads, an honest negative); the **Tier-B oracle** (rr/Hermit/gVisor, real-OS dep, deferred under the no-egress posture §15) and the full **technical report** remain future (the host-world results are written up in [`docs/report.md`](../report.md)) |

HC0–HC3 + the HC5 loop are the deterministic core. `M_θ` (HC4) drops into the loop via the
same model-agnostic interface v0 uses. Tier-B, torch, and the LLM client are optional
extras; the deterministic core has no runtime deps.

---

## 14. The autonomous research engine over the host world

SPEC-6 is built, as far as possible, with the human at the boundary — extending SPEC-4's
ratchet, not replacing it.

- **The gate is unfakeable, and now per-subsystem.** You cannot lower bits-to-correct
  (§5.4) or raise privilege-faithfulness (§9.4) without predicting the truth; the
  per-subsystem decomposition gives the engine a *richer* gradient than a single scalar —
  it can search for "improve the subsystem that is leaking" — while staying ground-truth
  (SPEC-4 §9).
- **Search space (knobs the proposer may turn):** composition-depth curriculum, interleaving
  entropy, factored-vs-flat architecture, interaction-graph and memory toggles,
  drift-mitigation strength (noise σ, schedule), `π_c`/`π_w` policies, TTT/stream learning
  rate and replay size, RLVR/GRPO settings.
- **Frozen host eval cells.** A held-out set of host snapshots + workloads + **chaos seeds**
  the proposer never sees or mutates; anomalous jumps re-evaluate on a second held-out set
  (the SPEC-4 §5.3 tripwire).
- **Denylist (the judge is not a knob, DD-AR2).** The proposer cannot edit the oracle, the
  metrics, the goldens, the gate, `docs/host-semantics.md`, or the scheduler semantics.
- **The four irreducibles stay with the human** (SPEC-4 §8), restated: the **objective**
  (faithfulness; bits-to-correct down, the H13 composition understood), the
  **safety/ethics boundary** (defensive-only, sandbox, **no real-internet egress**,
  editable-path denylist, §15), the **kill-switch + resource cap**, and **promotion to
  main** (the engine proposes on a branch; a human/trusted CI merges).

The host world *strengthens* the autoresearch story: the per-subsystem signal is denser
and harder to game than a single scalar, and the chaos seed gives the engine a principled
difficulty dial to search over.

---

## 15. Safety & ethics

This subsystem simulates a running host, which is closer to operational capability than a
filesystem toy. The posture (SPEC.md §13 / SPEC-3 §14 / SPEC-4 §9 / SPEC-5 §15) holds and
tightens:

- **Defensive framing only.** Downstream use is autonomous **defense**, change-safety, SRE,
  incident-response, and capacity planning — predicting the consequences of *your own*
  configuration/process changes in a sandbox before they touch production (the OSWorld-class
  framing, §2.6, §7). Not offense, not exploitation, not third-party targeting. No
  exploit-bearing workload is a goal or deliverable; any environment that meaningfully
  encodes real exploit dynamics is reviewed before release and may be held back (SPEC.md
  §13).
- **No real-internet egress.** Tier-A is a pure simulator. Tier-B runs only under
  hermetic isolation (Hermit/rr) in namespaces with egress disabled and self-contained,
  seeded workloads. The model never touches a host or network it does not own.
- **Reproducibility as a safety property.** Everything replays from `(snapshot, sched_seed,
  rng_cursor)`; no telemetry, no runtime network call (the repo-wide posture).
- **Denylist + kill-switch + resource cap** govern the engine (§14); the golden
  trajectories pin semantics so the engine cannot quietly repurpose them.
- **Dual-use note.** A faithful host simulator is dual-use; the denylist, the defensive task
  framing, and the sandbox-only oracle are the structural mitigations. The privilege-state
  axis (§3.2) is modeled to make the simulator *trustworthy about failures* (a defender's
  need), not to model exploitation.

---

## 16. Open, decentralized, verified contribution (the community protocol)

This section is the concrete, research-framed form of the user's **open, freely-available,
decentralized** intent — and nothing more (no service, no commercial path; SPEC.md §11
out-of-scope holds). The key enabling fact: **the oracle makes contributed data trustless
by construction.**

- **The trust problem decentralized training has, Verisim does not.** Prime Intellect's
  INTELLECT-2 trains across *untrusted* contributors and must *verify* their rollouts with
  TOPLOC's locality-sensitive hashing — a heuristic, probabilistic check (§2.9). Verisim
  replaces the heuristic with **exact re-execution**: any contributed `(state, action,
  next_state)` or `(state, action, delta)` is verified by re-running the deterministic
  oracle and comparing bit-for-bit. A contribution is accepted iff it reproduces. There is
  no trust to establish and no tampering to detect probabilistically — the oracle settles
  it, freely and certainly.
- **What can be contributed.** (a) Oracle-verified host trajectories (expand coverage of
  the transition space — the §2.3 curriculum lever); (b) new `Environment`/`Oracle` pairs
  for subsystems behind the existing protocols (a new symbolic sub-oracle, §2.7; a new
  workload driver); (c) golden trajectories that pin additional semantics (subject to the
  denylist review, §14). Each carries its manifest + content-hash (SPEC-2 §4, §12), so the
  whole contributed corpus is regenerable and integrity-checked.
- **Low-communication merging (designed, not required for v1).** If multiple parties train
  host SLMs on disjoint coverage, the DiLoCo-style low-communication merge (§2.9) is the
  natural way to fuse them — but this is an *optional, deferred* capability, behind the same
  determinism-first, no-runtime-network posture (§15). v1 ships the *verification* protocol
  (the part that is free and certain); distributed training is later and gated.
- **Why this is the defensible community contribution.** The artifact others get is not a
  model to compete with but a **free, exact verification layer** for host-state world-model
  data, and a benchmark/RL-env that anyone can extend and have their extension *certified by
  the oracle*. That is the open, decentralized contribution the deterministic oracle makes
  uniquely possible.

---

## 17. Open questions

The v0/§17 discipline: record them, resolve them in the open.

1. **Syscall semantics boundary.** How much of the process/fd/IPC/signal/scheduling model
   does Tier-A pin before it overfits to a toy or becomes too costly to keep correct? (The
   single most important design call — the host analogue of SPEC-5 §17.1.)
2. **Tier-B determinism for concurrency.** How far Hermit/rr seal scheduler nondeterminism
   in practice, and where chaos-seeded replay must stand in for true determinism — declared
   per figure (§3.3, HW-1). If a regime cannot be pinned, the oracle is *stochastic* there
   and the divergence metric must account for it (SPEC-5 §17.2's problem, harder).
3. **`ε` for composed state.** What "ε-close" means across heterogeneous subsystems — is it
   a single composed threshold, a per-subsystem vector, or weakest-link? Tied to whether
   H13's composition law is multiplicative or min (§9.2).
4. **Composition depth vs tractability.** How many subsystems can be active before training
   cost dominates; where the curriculum stops for v1.
5. **Interaction-graph inference.** Is the per-step interaction graph (§6.1) given by the
   syscall args (cheap, exact) or inferred (the FIOC-WM bet, §2.3)? v1 starts with the
   exact graph and treats inference as an EH4 lever.
6. **Plasticity-loss horizon.** How long the experience stream (§8.5) runs before plasticity
   loss (HW-4) bites at this scale, and which mitigation (replay size, normalization, PEFT)
   pays — measured by the §9.4 probe.
7. **Counterfactual sampling.** Which syscall-flip distribution produces counterfactuals
   that transfer (H16) — random flips, or targeted "near-miss" misconfigurations / privilege
   mistakes (the security-relevant ones)?
8. **Plan-level loop (inherited, SPEC-5 §17.8).** How to define divergence and `H_ε` for the
   LLM-integration case (§7) where the unit is a *plan*, not a syscall.

---

## 18. Definition of done

SPEC-6 is done when:

1. HC0–HC6 ship, tested, with the deterministic core dependency-free, GPU-free.
2. The **composed-host `H_ε(ρ)` curve (EH1)** is plotted once, cleanly, regenerable from
   config + seeds — *whatever it shows* — **and** the **composition-law measurement
   (H13, §9.2)** is reported: is whole-machine faithfulness predictable from its parts? A
   flat interior or an unlawful composition is a reportable result (the honest negative),
   not a failure.
3. The honest write-up (the `docs/report.md` discipline) states, for each hypothesis in §10
   (the operationalized H5/H7 and the new H13–H16), what was found and what the honest
   negative looked like — including the concurrency curve `H_ε(interleaving-entropy)` (H14)
   and the stream-vs-batch result with its plasticity probe (H15/HW-4).
4. The host world is packaged for reuse (Inspect benchmark + `verifiers`-spec host RL env +
   the LLM-callable whole-machine simulator protocol of §7 + the verified-contribution
   protocol of §16), so the community can measure long-horizon faithfulness of a *whole
   running machine* with ground truth, under controllable concurrency — the contribution of
   §1.4.

The science is one curve, again — but for the first time it is the curve of a *whole
machine*, and the question underneath it is whether faithfulness composes. That is the
question that decides whether oracle-grounded world models are a trick that works on toys
or a method that scales to systems.

---

## 19. Provenance and reading order

- **Prereqs:** [SPEC.md](./SPEC.md) (science), [SPEC-2.md](./SPEC-2.md) (filesystem v0),
  [SPEC-5.md](./SPEC-5.md) (network world). SPEC-6 composes the FS and network worlds and
  assumes their builds. Companions: [SPEC-3.md](./SPEC-3.md) (the depth roadmap; W1/W4 and
  HW-1 made concrete here) and [SPEC-4.md](./SPEC-4.md) (the engine; §14 extends it).
- **Lessons grounding this spec** (name + venue + year, per
  [`docs/related-work.md`](./docs/related-work.md)'s no-fabricated-links policy; arXiv IDs
  only where independently verified): NeuralOS (2025, arXiv:2507.08800) as the
  generative-OS anti-pattern; rr / Hermit / gVisor / CRIU for record/replay and the
  concurrency wall; Silver & Sutton, "Welcome to the Era of Experience" (DeepMind, 2025);
  the streaming-RL line ("Streaming Deep RL Finally Works", arXiv:2410.14606; "Experience
  Replay Addresses Loss of Plasticity", arXiv:2503.20018); the compositional/object-centric
  world-model line (COMBO, ICLR 2025; Dyn-O, NeurIPS 2025; factored MDPs; FIOC-WM); OSWorld
  (NeurIPS 2024) and TheAgentCompany for the computer-use payoff; the 2025 causal-world-model
  line (counterfactual calculus generalizing do-calculus, PMLR 2025; Causal-JEPA; CausalVAE);
  formal SQL/transaction semantics for symbolic second-oracles; and Prime Intellect
  INTELLECT-2 / PRIME-RL / TOPLOC / DiLoCo (2025, arXiv:2505.07291) for the
  decentralized-verified-contribution protocol. *(These should be added to
  `docs/related-work.md` when this spec moves from design to build.)*
- **Author:** Clay Good. **License:** MIT. The host oracle runs only inside the sandbox; no
  real-internet egress, no telemetry, defensive framing (§15).
- A living spec: as milestones (§13) land they are marked and their figures linked
  (mirroring SPEC-2 §13 / SPEC-5 §13); as a hypothesis (§10) is tested its result is
  recorded in SPEC.md §9. The spec is the record of what we believed and what we learned.
