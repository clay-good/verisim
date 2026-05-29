# Verisim — Research Specification (SPEC.md)

> **Verisim** *(n., from* verisimilitude *— "the quality of resembling reality")*: a research program and open-source codebase for building **oracle-grounded, neuro-symbolic world models of computer environments**, where a deterministic oracle keeps a learned simulator faithful over long horizons.

**License:** MIT. **Status:** v0, pre-experiment. **Audience:** the author (a security/systems engineer pivoting into ML research), future collaborators, and reviewers. **Companion document:** [SPEC-2.md](./SPEC-2.md) — the concrete v0 engineering and experiment specification. This document is the *science*; SPEC-2 is the *build*.

This file is intended to be read top to bottom by someone who has never seen the project, and to remain the canonical statement of *why the project exists, what it claims, and how we would know if we were wrong*. It is deliberately exhaustive. Sections 1–4 are the argument; 5–7 are the formalism; 8 is positioning; 9–10 are how we test it; 11–16 are scope, risk, and roadmap.

---

## 0. One-paragraph thesis

Generative world models (Genie 3, V-JEPA 2, Cosmos) are the current frontier of AI, and they all hit the same wall: **long-horizon error accumulation** and **faithfulness** — a learned simulator drifts away from the real dynamics it was trained on, and there is no cheap way to detect or correct the drift. The deep reason is that **physical and visual world models have no ground-truth oracle**: nothing can tell the model "the state you just imagined is wrong." Verisim's claim is that **computer environments are an exception**. The "world" that a computer-use agent or a cyber agent acts in — filesystems, processes, networks, APIs — is *digital, deterministic, and fully checkable*: you can run the real system in a sandbox and compare the model's predicted next state against ground truth, bit for bit. This makes computer environments the **one domain where a deterministic oracle can be placed in the loop** to bound a neural world model's drift. Verisim builds that loop, measures the central tradeoff nobody else can measure — **how much oracle consultation buys how much faithful horizon** — and exports the resulting technique (oracle-in-the-loop correction, error-bounded rollouts, a *quantitative* faithfulness metric) back to a field that, in every other domain, can only eyeball its simulators.

---

## 1. The problem: world models drift, and nobody can stop them

A **world model** is a learned model of an environment's dynamics: given a state and an action, predict the next state (and observation, and reward). World models are the substrate of model-based reinforcement learning (Dreamer-style planning), of generative interactive environments (Genie), and increasingly of embodied/robotic planning (V-JEPA 2, Cosmos). The 2025 field split into three branches — prediction (V-JEPA 2), visual synthesis (Cosmos), and interactive environment generation (Genie 3).

All three share the same unsolved failure mode:

1. **Long-horizon error accumulation.** A world model is rolled out *autoregressively*: its own predicted state becomes the input to the next prediction. Small per-step errors compound. Genie 3 holds visual coherence for *minutes* and then drifts; the field's own framing is that "error accumulation over long horizons is a fundamental problem."
2. **Faithfulness / hallucination of state.** The model invents dynamics that never existed — objects appear, constraints are violated, conservation laws break. This is not a low-confidence phenomenon; recent work reframes it as a *structural consistency* failure that occurs independently of the model's confidence.
3. **Causal poverty.** World models are good at "what happens next on the observed distribution" and bad at "what would happen if I did X" — counterfactuals. Causal reasoning is the named open problem.
4. **No verification.** Above all: there is no cheap, principled way to *check* a learned simulator's output. You can hold out trajectories and measure rollout error post hoc, but you cannot, at inference time, ask "is this predicted state real?" and get a trustworthy yes/no.

### 1.1 Why the wall is so high: the missing oracle

Problems 1–3 are downstream of problem 4. If you had a cheap oracle that could tell you the true next state, error accumulation would be trivially correctable (snap back to truth), faithfulness would be measurable (compare to truth), and counterfactuals would be checkable (run the counterfactual through the oracle).

But for **physical and visual worlds, no such oracle exists.** When Genie imagines the next frame of a forest, the only thing that can adjudicate "is that the right frame?" is reality itself — which you cannot rerun — or a human, who cannot scale. This is not a temporary engineering gap; it is structural. The physical world is not resettable, not fully observable, and not cheaply re-executable. So the field is *forced* to treat the simulator as unverifiable and to fight drift indirectly (bigger models, more data, regularization, better architectures), with limited success.

---

## 2. The core insight: computer worlds have a free, perfect oracle

The environments that matter for **agentic AI in security and computer use** — the environments this author has spent a decade inside — are not physical. They are **digital**: operating systems, filesystems, process tables, network stacks, web applications, APIs, key-value stores. These worlds have three properties that physical worlds lack:

- **Deterministic (or determinizable) transitions.** Given a state and an action (a command, a syscall, a packet), the true next state is a deterministic function of the two, modulo a small, controllable set of nondeterminism sources (time, randomness, concurrency) that can be seeded, mocked, or recorded.
- **Full observability and serializability.** The complete state of a constrained computer environment can be captured, serialized, hashed, diffed, and restored. You can know the whole world, exactly.
- **Cheap re-execution and reset.** You can run the real system in a sandbox, step it forward, snapshot it, reset it, and replay it — thousands of times, programmatically.

Together these mean: **for computer worlds, a ground-truth oracle is not just possible, it is cheap.** Run the real thing (or a faithful deterministic reference of it) in a sandbox; it tells you the true next state on demand.

> **This is the asymmetry the whole project rests on.** Physical-world researchers would give anything for a cheap oracle and will never have one. In computer worlds you get it for free. Verisim is the research program that takes that asymmetry seriously and builds the thing it makes possible.

### 2.1 What the oracle is, precisely

An **oracle** `O` is any procedure that, given a true state `s` and an action `a`, returns the true next state `s' = O(s, a)` (and optionally the true observation and any side facts). Two flavors, both used in this project:

- **Reference oracle** — a deterministic interpreter that *implements the semantics* of the environment (e.g., a from-scratch model of POSIX filesystem operations). Perfectly deterministic, perfectly reproducible, very cheap. Caveat: it is itself a symbolic *model* of reality, not reality. Good enough to prove the *mechanism*; not sufficient to prove fidelity to a real OS.
- **System oracle** — the *actual* sandboxed system (a namespaced shell, a container with a pinned image, a real network stack) executed under controlled nondeterminism. True ground truth. More expensive and has nondeterministic edges that must be handled. This is what makes the claim about *reality* rather than about a model of reality.

v0 uses the reference oracle to establish the curves cleanly and reproducibly; a later phase swaps in the system oracle to validate that the mechanism survives contact with a real OS. See SPEC-2 §3 and §14.

---

## 3. The thesis and the research questions

**Thesis.** *In a domain that admits a deterministic oracle, a neural world model coupled to that oracle in a propose–verify–correct loop can be kept faithful over horizons far longer than the neural model alone sustains, at an oracle-consultation cost far lower than running the oracle every step — and the tradeoff between consultation cost and faithful horizon is itself a measurable, optimizable object that no oracle-free domain can expose.*

This is the author's standing thesis — *deterministic verification wrapped around stochastic inference* — applied to the hardest version of the problem: not verifying a single answer, but keeping an entire *simulated world* true over time. The oracle is the verifier; the world model is the stochastic proposer; the loop is the harness.

**Research questions.**

- **RQ1 — Faithful horizon vs. oracle budget.** As a function of the oracle-consultation rate (how often the loop is allowed to consult `O`), how long can the coupled system maintain a faithful simulation before divergence exceeds a threshold? What is the *shape* of this curve, and where are its knees?
- **RQ2 — Consultation policy.** Does *when* you consult the oracle matter more than *how often*? Specifically, do drift-triggered and model-uncertainty-triggered consultation policies dominate fixed-interval consultation at equal budget?
- **RQ3 — Correction operators.** Given a consultation, what is the best way to use it? Hard reset (snap predicted state to truth), residual correction (learn to predict the oracle's correction), or constraint projection (project the predicted state onto the nearest oracle-consistent state)? Which yields the longest faithful horizon per correction?
- **RQ4 — Counterfactual fidelity.** Can an oracle-grounded world model answer interventional/counterfactual queries ("what if the action at step *t* had been *a'* instead of *a*?") more faithfully than an oracle-free one, and can the oracle be used to *train* that counterfactual fidelity rather than only to evaluate it?

A positive, quantitative answer to **RQ1** alone is a publishable contribution, because it is a measurement the rest of the world-model field structurally cannot make. RQ2–RQ4 are the depth.

---

## 4. Why this advances the *general* field, not just security

A skeptic's objection: "You've picked the one easy domain. So what?" The answer is that **the techniques transfer even though the oracle does not.**

- The **faithful-horizon-vs-consultation-budget curve** is a *general* object. In physical domains you cannot plot it because you have no oracle — but the *methodology* for spending a scarce, expensive verifier optimally (RQ2) is exactly what a robotics team needs when their "oracle" is an occasional, costly real-world rollout, or a slow high-fidelity physics sim. Verisim works out the theory of optimal verifier-spending in the domain where the verifier is cheap, then hands the policy to domains where it is expensive.
- **Correction operators** (RQ3) — residual correction, constraint projection — are architecture-level techniques that do not depend on the oracle being cheap, only on it being *occasionally available*. Every world-model domain has *some* occasional ground truth (a held-out rollout, a real episode, a human label). Verisim is the clean testbed for inventing and ranking those operators.
- A **quantitative faithfulness metric** anchored to ground truth lets us calibrate the *proxy* metrics (perceptual, learned, self-consistency) that physical-world researchers are forced to use. If we can show which cheap proxy best predicts true faithful horizon in the domain where we know the truth, that calibration is directly useful to the domain where they don't.

Verisim is, in this framing, the **metrology lab for world-model faithfulness**: the place where you can put a real ruler on a phenomenon everyone else measures by eye.

### 4.1 The second field it advances: autonomous cyber defense

Independently of the world-model contribution, a *faithful, cheap, learned* model of a computer/network environment is a direct contribution to the **autonomous cyber defense (ACD)** line of research, which today is bottlenecked by exactly this. ACD agents (defensive RL policies) are trained in simulators — CybORG, CyGIL, PrimAITE — and the field's central pain is the **simulation-to-emulation gap**: cheap simulators are unfaithful, faithful emulators are too expensive to train against at scale. A world model that is *learned* (hence cheap to roll out) but *oracle-grounded* (hence verifiably faithful) is a candidate resolution to that gap. So Verisim sits at the intersection of two active fields and contributes to both with one artifact.

### 4.2 The third reason: the bottleneck is the author's skillset, not ML theory

The hard part of Verisim is not the neural network. It is building a **faithful, deterministic, resettable, fully-observable oracle of a real computer environment** — sandboxing, deterministic replay, controlled nondeterminism, state serialization, telemetry, attack/defense dynamics. That is systems-and-security engineering of exactly the kind an ML PhD typically cannot do and a senior security/systems engineer does in their sleep. **This is a frontier ML problem gated on systems engineering.** That is the rare problem where a non-credentialed builder has a genuine, defensible edge, and it is the reason this is the right first research bet for this author specifically.

---

## 5. Formalism and definitions

We model a computer environment as a deterministic (or determinized) **state machine**.

- **State** `s ∈ S`: the complete, serializable configuration of the environment. In v0 (SPEC-2 §2): the filesystem tree (paths, types, content hashes, modes), current working directory, and environment variables.
- **Action** `a ∈ A`: an atomic operation the agent/driver can take. In v0: a shell command drawn from a fixed grammar.
- **Observation** `o ∈ Ω`: what is returned to an actor after an action (e.g., stdout, exit code). In v0 the environment is fully observable, so `o` can include the full state; partial observability is introduced deliberately in later phases.
- **True transition** `O: S × A → S`, the **oracle**: the real next state. Deterministic by construction (nondeterminism sources are seeded/mocked/recorded).
- **World model** `M_θ: S × A → Δ̂` (or `→ ŝ'`): the learned model, parameterized by `θ`, predicting either the full next state `ŝ'` or, preferably, a **state delta** `Δ̂` (the set of edits) which applied to `s` yields `ŝ'`. Predicting the delta operationalizes "the *effect* of the action on the state," is far more sample-efficient than regenerating the whole state, and is directly checkable.
- **Rollout**: starting from `s_0`, repeatedly apply `M_θ` autoregressively: `ŝ_{t+1} = apply(ŝ_t, M_θ(ŝ_t, a_t))`. The **ground-truth rollout** applies `O` instead.
- **Divergence** `d(s, ŝ) ≥ 0`: a distance between a predicted and true state (SPEC-2 §7 defines the concrete metric — a normalized symmetric set difference / tree-edit distance over `(path, type, content-hash)` tuples). `d = 0` iff the states are identical.

### 5.1 The central quantities

- **Faithful horizon** `H_ε`: the largest number of steps for which an autoregressive rollout stays within divergence `ε` of the ground-truth rollout: `H_ε = max{ T : d(s_t, ŝ_t) ≤ ε for all t ≤ T }`. This is the headline dependent variable.
- **Oracle-consultation rate** `ρ`: the fraction of steps at which the loop is permitted to call `O`. `ρ = 0` is the pure neural model (no oracle); `ρ = 1` is the oracle every step (perfect, but you've just rerun the simulator and the neural model is doing nothing). The interesting regime is `0 < ρ < 1`.
- **Consultation policy** `π_c`: the rule that decides, at each step, whether to spend a consultation (fixed-interval, drift-triggered, uncertainty-triggered, learned). Subject to budget `ρ`.
- **Correction operator** `C`: how a consultation is used (hard reset, residual correction, constraint projection). See RQ3.

### 5.2 The propose–verify–correct loop (the core mechanism)

```
state s ← s_0
for t in 0..T:
    Δ̂  ← M_θ(s, a_t)              # PROPOSE: neural model predicts the effect
    ŝ' ← apply(s, Δ̂)
    if π_c decides to consult at step t:    # subject to budget ρ
        s'  ← O(s, a_t)            # VERIFY: oracle returns the truth
        record divergence d(s', ŝ')
        s   ← C(ŝ', s')            # CORRECT: combine prediction with truth
    else:
        s   ← ŝ'                   # trust the model
```

The research is in the choices of `π_c` (RQ2), `C` (RQ3), and `M_θ`'s representation/training (SPEC-2 §5–6). Everything Verisim does is an instance of this loop, scaled from a shell sandbox (v0) toward a network (later phases).

---

## 6. The neuro-symbolic architecture

Verisim is **neuro-symbolic** in a specific, literal sense: the *symbolic* half is not a hand-written rule set approximating the world — it is the **real system (or a faithful deterministic reference of it)**, used as the oracle. The *neural* half is the learned transition model that runs cheaply between consultations. This inverts the usual neuro-symbolic pattern (neural perception + symbolic reasoning over an abstracted model): here the symbolic component is the *most* faithful component and the neural component is the *cheap approximation* of it.

Three architectural commitments:

1. **Delta prediction over full-state generation.** `M_θ` predicts structured edits, not a regenerated world. This bounds the surface area for hallucination (the model cannot accidentally delete a file it never mentions), makes verification a localized diff, and is the more learnable target.
2. **Oracle in the loop, not just in training.** The oracle is available at *inference/rollout* time, not only as a training signal. This is the whole point and the thing that separates Verisim from "train a world model on simulator data and hope."
3. **Train against the oracle (RLVR-style), not only supervise on its traces.** Beyond supervised next-state prediction, the oracle is a *verifiable reward*: a rollout that stays faithful for longer is rewarded. This connects directly to RL-with-verifiable-rewards. The oracle is the reward function; faithful horizon is the return. (This is the bridge to the author's verifier-as-reward thesis and to the Prime Intellect verifiers ecosystem — see §8 and SPEC-2 §15.)

---

## 7. Faithfulness theory and metrics

The project's signature contribution is treating **faithfulness as a measured quantity with units**, not a vibe. Concretely:

- **Headline result (RQ1):** the curve `H_ε(ρ)` — faithful horizon as a function of oracle-consultation budget — for fixed `ε`, swept over `ε`. Reported with confidence intervals over seeds and environments. The shape, the knees, and the gap between the `ρ→0` floor and the `ρ→1` ceiling are the findings.
- **Policy result (RQ2):** `H_ε` at equal budget `ρ` for fixed-interval vs. drift-triggered vs. uncertainty-triggered vs. learned `π_c`. The claim to be tested: *smart consultation dominates dumb consultation,* and by how much.
- **Operator result (RQ3):** faithful horizon gained per correction for each `C`. The claim: residual correction and constraint projection beat hard reset because they teach the model rather than merely overwriting it.
- **Counterfactual result (RQ4):** divergence on held-out *interventional* queries, oracle-grounded vs. oracle-free, with the oracle generating the counterfactual ground truth.

Secondary/diagnostic metrics: per-step exact-match accuracy, structural divergence distribution, calibration of the model's own uncertainty against actual divergence (needed for uncertainty-triggered consultation), and compute cost per faithful step (oracle calls are the expensive unit).

A standalone deliverable falls out of this section: **a quantitative world-model faithfulness benchmark** for computer environments, with ground-truth labels — something the visual/physical field cannot construct. This is packaged for the evals ecosystem (SPEC-2 §15).

---

## 8. Positioning and related work

Verisim is *not* claiming to invent world models, neuro-symbolic methods, or cyber simulators. Its novelty is the **specific synthesis**: oracle-in-the-loop, inference-time, in a domain chosen *because* the oracle is cheap, with faithfulness treated as a measured tradeoff against consultation budget. Positioning against the neighbors:

- **Generative/visual world models (Genie 3, V-JEPA 2, NVIDIA Cosmos, World Labs).** Same problem (drift, faithfulness), opposite domain (no oracle). Verisim does not compete on visual fidelity; it competes on *verifiable* fidelity in a domain they cannot reach. Their open problems (long-horizon consistency, causality) are the problems Verisim attacks where they're measurable.
- **Model-based RL / Dreamer-style world models.** These learn a latent world model and plan in it; they generally do *not* keep a deterministic oracle in the loop at rollout time, and they live in domains (Atari, control) where the "oracle" is the env itself but is treated as the thing being modeled away. Verisim keeps the oracle and studies how little of it you can get away with.
- **Neuro-symbolic world models (2025–26 line: symbolic transition structure + neural fill-in; fine-tuning neural models only on trajectories symbolic rules don't cover).** Closest prior art on *method*. Verisim's differences: (a) the symbolic half is the *real system/oracle*, not a hand-built abstraction; (b) the oracle is used at inference, budgeted, and *optimized over* (RQ2); (c) the domain is computer environments specifically, with the explicit faithfulness-vs-budget framing.
- **Neuro-symbolic verifier-gated reasoning (VIRF, EIDOKU, SymCode).** Same instinct (deterministic gate over stochastic generation) at the level of a single reasoning step / answer. Verisim lifts it from one answer to an entire *temporal world rollout*, where errors compound — a strictly harder regime.
- **Autonomous cyber defense simulators (CybORG, CyGIL, PrimAITE; emulation-simulation unification).** The application field. Verisim contributes a learned-but-faithful world model as a candidate fix for their sim-to-emulation gap (§4.1).
- **RLVR / RL with verifiable rewards; Constitutional AI / RLAIF.** Constitutional AI uses an LLM to critique an LLM (probabilistic checking probabilistic). Verisim replaces the critic with a deterministic oracle wherever the domain admits one, and uses it as a *verifiable reward* to train the world model (§6.3). This is the author's standing research thesis instantiated.
- **The author's own prior work** (Proxilion, Invariant, Mantissa, agent-replay, Vaulytica): each is a deterministic verifier over a stochastic process in a security context. Verisim is the generalization — the verifier is no longer guarding a single action; it is grounding a whole simulated world. See the repo's `docs/lineage.md` (to be written) for the explicit mapping.

---

## 9. Hypotheses (falsifiable)

The project lives or dies by these. Each is stated so that a clean experiment can refute it.

- **H1 (the curve exists and is favorable).** There is a regime `0 < ρ* < 1` where the coupled system achieves faithful horizon `H_ε` close to the `ρ=1` ceiling at consultation cost far below it (target: ≥80% of ceiling horizon at ≤20% consultation, on v0). *Refuted if* `H_ε(ρ)` is approximately linear (no free lunch — every faithful step costs a proportional oracle call) across all environments tested.
- **H2 (smart beats dumb).** Drift- or uncertainty-triggered consultation achieves materially higher `H_ε` than fixed-interval consultation at equal `ρ`. *Refuted if* no policy beats fixed-interval within confidence intervals.
- **H3 (correction teaches).** Residual correction / constraint projection yields more faithful horizon per correction than hard reset, and (stretch) reduces future divergence (the model learns from corrections online). *Refuted if* hard reset is statistically indistinguishable or better.
- **H4 (mechanism survives reality).** The curves obtained with the reference oracle qualitatively hold when the system oracle (a real sandboxed shell) replaces it. *Refuted if* real-OS nondeterminism collapses the faithful horizon regardless of consultation.
- **H5 (counterfactual lift).** Oracle-grounding improves interventional-query fidelity over an oracle-free model trained on identical data. *Refuted if* no lift.

---

## 10. Evaluation methodology

- **Environments:** start with v0's deterministic shell/filesystem sandbox (SPEC-2 §2); scale to multi-host network state and syscall traces in later phases. Multiple environment *instances* (different difficulty, branching, state size) to test generality, not a single benchmark.
- **Protocol:** for each (environment, model, `π_c`, `C`, `ρ`, `ε`, seed), run N autoregressive rollouts against ground-truth rollouts; record divergence trajectories and `H_ε`. Aggregate with confidence intervals over seeds. All configs versioned; all runs reproducible from a seed (SPEC-2 §12).
- **Baselines:** (b0) pure neural, `ρ=0`; (b1) oracle every step, `ρ=1` (ceiling/sanity); (b2) symbolic-only (the oracle alone — trivially perfect, included to frame what the neural model is *for*: cheap rollout between consultations); (b3) trivial/frequency predictor (floor). The contribution is the interior and the policy/operator comparisons, not beating b1 (which is unbeatable on fidelity but maximally expensive).
- **Ablations:** representation (delta vs full-state), model size, training objective (supervised vs +RLVR), `ε` sweep, environment size sweep.
- **Reporting:** every claim ties to a hypothesis in §9 and a figure. Negative results are first-class and reported (the field needs honest faithful-horizon numbers more than it needs a hero result).

---

## 11. Scope

**In scope (the research program):** computer/digital environments with a constructible oracle — filesystems, shells, processes, syscalls, constrained networks, simple web/API state machines. The faithfulness-vs-budget science. The neuro-symbolic oracle-in-the-loop architecture. The two field contributions (world-model metrology; ACD sim-to-emulation).

**Explicitly out of scope:** visual/physical world models (no oracle — that's the whole point); building a better inference *engine*; building a product/SaaS (this is a research repo, MIT, no telemetry, no commercial path); modeling environments so large or nondeterministic that no practical oracle exists (that's the *frontier* we approach gradually, not the starting point).

**Non-goals:** beating Genie on video; beating vLLM on throughput; shipping an autonomous offensive cyber agent (see §13 ethics).

---

## 12. Research roadmap

Phases are gated by hypotheses, not calendar. Detailed engineering milestones for Phase 0–1 are in SPEC-2 §13.

- **Phase 0 — v0 shell/filesystem world, reference oracle.** Establish `H_ε(ρ)` (H1), fixed vs. triggered consultation (H2), correction operators (H3). Smallest world with the hard (compounding-state) property. *This is the first paper / artifact.* (SPEC-2 is entirely about this phase.)
- **Phase 1 — system oracle.** Swap the reference interpreter for a real sandboxed shell; test H4 (mechanism survives real-OS nondeterminism). Introduce controlled nondeterminism handling.
- **Phase 2 — partial observability.** Hide part of the state (the realistic case: you can't observe all of an OS). Study faithfulness under partial observation and oracle-assisted state estimation.
- **Phase 3 — counterfactuals & causality.** Interventional/counterfactual queries (H5); oracle-generated counterfactual training data.
- **Phase 4 — network scale & ACD.** Multi-host network state; integrate with / compare against CybORG-class ACD simulators; demonstrate the sim-to-emulation contribution.
- **Phase 5 — agent training.** Train a downstream agent (defensive cyber, or a computer-use agent) *inside* the Verisim world model and measure transfer to the real system — the ultimate test of a world model's usefulness.

---

## 13. Risks, limitations, and ethics

**Threats to validity.**
- *The reference oracle is a model, not reality.* Phase 0 results prove the mechanism, not fidelity to a real OS. Mitigated by Phase 1 (system oracle) and stated plainly in every Phase 0 claim.
- *v0 might be too easy.* If filesystem dynamics are so learnable that the neural model never drifts, there's no curve to study. Mitigation: deliberately include long-range dependencies and branching that make compounding error appear (SPEC-2 §2.4); scale environment difficulty until drift is real.
- *The favorable curve might not exist (H1 refuted).* That is itself a publishable result: "in this domain, faithful simulation costs oracle calls roughly linearly." It would reshape expectations and is worth reporting.
- *Nondeterminism in real systems (Phase 1+)* could dominate. Handled by seeding/mocking/recording and by scoping which nondeterminism sources are in/out.

**Scope/over-reach risk.** The single largest risk to *the author* is letting this balloon into another many-repo saga before producing one falsifiable result. The mitigation is structural: SPEC-2 defines a deliberately tiny v0 whose only job is to plot `H_ε(ρ)` once. Resist building Phase 4 before Phase 0 produces a curve.

**Ethics and dual use.** Computer-environment world models and ACD research are dual-use: a faithful network world model could in principle train offensive agents as well as defensive ones. Commitments: (1) the framing, environments, and downstream agents are **defensive** (cyber *defense*, detection, resilience) and **computer-use** (productivity agents acting safely); (2) no offensive autonomous agent is a goal or deliverable; (3) any environment that meaningfully encodes real exploit dynamics is reviewed before public release and may be held back or released with safeguards; (4) MIT license with a clearly stated intended-use and responsible-disclosure posture in the README. This mirrors the responsible posture across the author's prior security repos.

---

## 14. What "advancing the field" concretely means here

Success is not a product and not a single benchmark number. It is, in order of ambition:

1. **A measurement no one else can make:** the first quantitative `H_ε(ρ)` faithfulness-vs-oracle-budget curves for computer-environment world models, with ground-truth labels. (Refutes or confirms H1.)
2. **A transferable technique:** ranked consultation policies (H2) and correction operators (H3) that generalize the *methodology* of optimal verifier-spending to oracle-scarce domains.
3. **A reusable artifact:** an MIT-licensed faithfulness benchmark + RL environment (packaged for Inspect and the Prime Intellect verifiers ecosystem) that others build on.
4. **A bridge result:** evidence that oracle-grounded learned world models can narrow the ACD sim-to-emulation gap.

Any one of (1)–(3) is a meaningful, citable contribution achievable by a non-PhD builder. All four would be a research identity.

---

## 15. Glossary

- **World model** — a learned model of an environment's dynamics: `(state, action) → next state`.
- **Oracle `O`** — a procedure returning the *true* next state; cheap and exact in computer domains. Reference oracle (deterministic interpreter) vs. system oracle (real sandboxed system).
- **Autoregressive rollout** — feeding the model's own predicted state back as input; the regime where error accumulates.
- **Faithful horizon `H_ε`** — max steps a rollout stays within divergence `ε` of ground truth.
- **Oracle-consultation rate `ρ`** — fraction of steps allowed to call the oracle.
- **Consultation policy `π_c`** — rule deciding *when* to consult (fixed / drift-triggered / uncertainty-triggered / learned).
- **Correction operator `C`** — how a consultation is applied (hard reset / residual / constraint projection).
- **State delta `Δ`** — structured set of edits an action makes to the state (Verisim's prediction target).
- **Divergence `d`** — distance between predicted and true state; `0` iff identical.
- **Propose–verify–correct** — the core loop (§5.2): neural proposes, oracle verifies, operator corrects.
- **RLVR** — RL with verifiable rewards; here the oracle is the reward and faithful horizon is the return.
- **ACD** — autonomous cyber defense; the application field with the sim-to-emulation gap Verisim addresses.

---

## 16. Provenance and reading order

- **Read next:** [SPEC-2.md](./SPEC-2.md) — the concrete v0 environment, oracle, model, metrics, baselines, repo layout, and milestones.
- **Author:** Clay Good. **License:** MIT (see [LICENSE](./LICENSE)). **No telemetry, no commercial path** — this is a research repo.
- **Citations / state of the art** referenced above (Genie 3, V-JEPA 2, Cosmos, neuro-symbolic world models, CybORG/CyGIL/PrimAITE, VIRF/EIDOKU/SymCode, RLVR) are current as of May 2026; a maintained bibliography lives in `docs/related-work.md` (to be written) with links and one-line takes.
- This document will evolve. When a hypothesis in §9 is tested, its result and the experiment that produced it are linked here and the hypothesis is marked confirmed/refuted/open. The spec is the living record of what we believed and what we learned.
