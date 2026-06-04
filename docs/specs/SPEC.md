# Verisim — Research Specification (SPEC.md)

> **Verisim** *(n., from* verisimilitude *— "the quality of resembling reality")*: a research program and open-source codebase for building **oracle-grounded, neuro-symbolic world models of computer environments**, where a deterministic oracle keeps a learned simulator faithful over long horizons.

**License:** MIT. **Status:** v0, pre-experiment. **Audience:** the author (a security/systems engineer pivoting into ML research), future collaborators, and reviewers. **Companion documents:** [SPEC-2.md](./SPEC-2.md) — the concrete v0 engineering and experiment specification (the *build*); [SPEC-3.md](./SPEC-3.md) — the depth expansion (system oracle, networks, partial observability, self-healing, the information-theoretic metric); [SPEC-4.md](./SPEC-4.md) — the Autonomous Research Engine that builds it all with the human out of the loop. This document is the *science*; SPEC-2 is the *v0 build*; SPEC-3 is *what it becomes*; SPEC-4 is *how it builds itself*. Three specs are *cross-cutting methods* every world inherits rather than world verticals: [SPEC-4.md](./SPEC-4.md) (the autonomous research engine), [SPEC-8.md](./SPEC-8.md) (oracle-grounded self-supervision — where the oracle's truth enters training, across all three of LeCun's learning layers), and [SPEC-9.md](./SPEC-9.md) (the free-oracle scaling regime — because the oracle labels for free, world size is a *learner-compute* choice, not a labeling-budget one).

This file is intended to be read top to bottom by someone who has never seen the project, and to remain the canonical statement of *why the project exists, what it claims, and how we would know if we were wrong*. It is deliberately exhaustive. Sections 1–4 are the argument; 5–7 are the formalism; 8 is positioning; 9–10 are how we test it; 11–16 are scope, risk, and roadmap.

---

## 0. One-paragraph thesis

Generative world models (Genie 3, V-JEPA 2, Cosmos) are the current frontier of AI, and they all hit the same wall: **long-horizon error accumulation** and **faithfulness** — a learned simulator drifts away from the real dynamics it was trained on, and there is no cheap way to detect or correct the drift. The deep reason is that **physical and visual world models have no ground-truth oracle**: nothing can tell the model "the state you just imagined is wrong." Verisim's claim is that **computer environments are an exception**. The "world" that a computer-use agent or a cyber agent acts in — filesystems, processes, networks, APIs — is *digital, deterministic, and fully checkable*: you can run the real system in a sandbox and compare the model's predicted next state against ground truth, bit for bit. This makes computer environments the **one domain where a deterministic oracle can be placed in the loop** to bound a neural world model's drift. Verisim builds that loop, measures the central tradeoff nobody else can measure — **how much oracle consultation buys how much faithful horizon** — and exports the resulting technique (oracle-in-the-loop correction, error-bounded rollouts, a *quantitative* faithfulness metric) back to a field that, in every other domain, can only eyeball its simulators. And because the loop treats the world model as a *pluggable proposer* — a transformer, a JEPA-style latent predictor, an RSSM, or an LLM all drop into the same socket (§6) — the deeper claim is about a *method*, not a model: **deterministic verification as a model-agnostic primitive for probabilistic ML**, demonstrated on — but not confined to — computer-environment world models.

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
- **A measured answer to JEPA's open question — "what is safe to discard?"** Every representation-learning world model (JEPA above all, §8) faces the same unverified choice: which features of the world to keep and which to throw away. JEPA decides this *implicitly*, by what its prediction loss happens to find predictable, with nothing to check that it chose well. In a domain with an oracle, "which features are causally load-bearing" is **measurable by intervention** — perturb a feature of the predicted state, run it through the oracle, and see whether the truth diverges (the same machinery as RQ4 counterfactuals). So Verisim can rank, against ground truth, *which* abstraction preserves faithful horizon, and export that ranking as a calibrated rule of thumb to the oracle-free domains that must guess. The discipline that keeps this honest: the learned/abstracted metric is only ever an *internal* signal (a consultation trigger, a training target); the **headline faithfulness stays exact and oracle-grounded** (§7), so the proxy is calibrated *by* the truth, never substituted *for* it.

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

Four architectural commitments:

1. **Delta prediction over full-state generation.** `M_θ` predicts structured edits, not a regenerated world. This bounds the surface area for hallucination (the model cannot accidentally delete a file it never mentions), makes verification a localized diff, and is the more learnable target.
2. **Oracle in the loop, not just in training.** The oracle is available at *inference/rollout* time, not only as a training signal. This is the whole point and the thing that separates Verisim from "train a world model on simulator data and hope."
3. **Train against the oracle across *all three* of LeCun's learning layers, not only the cherry.** The oracle is a *verifiable reward* (RLVR-style): a rollout that stays faithful for longer is rewarded; the oracle is the reward function, faithful horizon is the return. This connects directly to RL-with-verifiable-rewards (the bridge to the author's verifier-as-reward thesis and the Prime Intellect verifiers ecosystem — see §8, SPEC-2 §15). But RLVR is the *cherry* — LeCun's smallest layer. The same free, exact oracle can supervise the *icing* (oracle-labeled next-state deltas — already used in SPEC-2.1 §5/K1) and, with the most leverage, the *bulk* (self-supervised pretraining grounded on oracle truth instead of corpus co-occurrence). That broader placement — *deterministic ground truth in the bulk of the cake, not just the cherry* — is its own cross-cutting method spec: [SPEC-8](./SPEC-8.md) (oracle-grounded self-supervision), tested as H23–H25 (§9).
4. **Model-agnostic by construction — the proposer is a pluggable part.** The loop, the oracle, the divergence metric, and the faithfulness benchmark know nothing about *how* the next state was proposed; they consume a `Model` protocol (SPEC-2 §6.3, M5; the network analogue in SPEC-5 §8). The from-scratch transformer, a symbolic oracle-backed model, a JEPA-style latent predictor, an RSSM, and a frozen LLM prompted as a proposer are all interchangeable behind that one seam. The consequence is the project's most general claim: **the favorable-consultation behavior, if it exists, is a property of the oracle-loop, not of any model class** — verification is the invariant, the proposer is swappable. This is what lets Verisim be *the layer underneath* the world-model race rather than another entrant in it, and it is stated as a falsifiable hypothesis (H22, §9).

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
- **JEPA / V-JEPA 2 specifically — the cleanest oracle-free foil, and a source of components.** LeCun's Joint-Embedding Predictive Architecture is an *energy-based model*: it predicts the *representation* of the future (not the raw input), trained by regression to a learned target encoder, and it plans by minimizing a goal-conditioned *energy* (a learned distance) via MPC. Its defining move is to **discard whatever is not predictable** — the information bottleneck is the whole point. Two observations frame Verisim against it. (i) *The energy is a learned surrogate; there is no oracle.* Nothing tells a JEPA rollout that its predicted latent is the *true* latent, so it can drift in representation space and never be caught — JEPA is the best-in-class instance of exactly the unverifiable regime §1.1 describes. Verisim's divergence `d(s, ŝ)` *is* an energy too, but one **computed against a deterministic ground-truth oracle rather than a learned target encoder** — so Verisim is, in one line, *an energy-based world model whose energy is ground truth.* (ii) *JEPA must guess what to discard; in a checkable domain you can measure it.* What a JEPA encoder throws away is decided implicitly by the prediction loss, unverified. Where an oracle exists, "which features are load-bearing" is itself measurable by intervention (§4, §7), so Verisim can both *borrow* JEPA's machinery (latent belief over the unobserved part, collapse-prevention à la VICReg/EMA — used in the partial-observation arm, SPEC-3 §4.2 / SPEC-5 §6.2) and *calibrate* the proxy JEPA-land is forced to trust blind. The line Verisim does **not** cross: it never latent-ifies the *observable, checkable* part — that would surrender the bit-for-bit verifiability that is the entire asset (DD-3, SPEC-3 §4.2).
- **Model-based RL / Dreamer-style world models.** These learn a latent world model and plan in it; they generally do *not* keep a deterministic oracle in the loop at rollout time, and they live in domains (Atari, control) where the "oracle" is the env itself but is treated as the thing being modeled away. Verisim keeps the oracle and studies how little of it you can get away with.
- **Neuro-symbolic world models (2025–26 line: symbolic transition structure + neural fill-in; fine-tuning neural models only on trajectories symbolic rules don't cover).** Closest prior art on *method*. Verisim's differences: (a) the symbolic half is the *real system/oracle*, not a hand-built abstraction; (b) the oracle is used at inference, budgeted, and *optimized over* (RQ2); (c) the domain is computer environments specifically, with the explicit faithfulness-vs-budget framing.
- **Neuro-symbolic verifier-gated reasoning (VIRF, EIDOKU, SymCode).** Same instinct (deterministic gate over stochastic generation) at the level of a single reasoning step / answer. Verisim lifts it from one answer to an entire *temporal world rollout*, where errors compound — a strictly harder regime.
- **Autonomous cyber defense simulators (CybORG, CyGIL, PrimAITE; emulation-simulation unification).** The application field. Verisim contributes a learned-but-faithful world model as a candidate fix for their sim-to-emulation gap (§4.1).
- **RLVR / RL with verifiable rewards; Constitutional AI / RLAIF.** Constitutional AI uses an LLM to critique an LLM (probabilistic checking probabilistic). Verisim replaces the critic with a deterministic oracle wherever the domain admits one, and uses it as a *verifiable reward* to train the world model (§6.3). This is the author's standing research thesis instantiated.
- **LeCun's "cake" and the supervision taxonomy (NIPS 2016; *self-supervised* per ISSCC 2019).** The cake orders learning signals by *quantity* — self-supervised (bulk) > supervised (icing) > reinforcement (cherry) — but is silent on whether the signal is *true*: self-supervision's target is corpus co-occurrence, supervision's is a human, RL's is a proxy reward; none is at once free, exact, dense, and generative, because no oracle-free domain has such a thing. A deterministic oracle is that fourth source — the *plate* the cake's proxies all stand in for. Verisim today places its oracle only in the cherry (RLVR, §6.3) and partly the icing (SPEC-2.1 §5); the largest, least-anchored layer — the self-supervised bulk — is where the oracle-free field spends the most effort on *proxy* verifiers (JEPA's collapse-prevention machinery; GrndCtrl's geometric rewards; World Action Verifier's cycle-consistency), and where verisim has put none of its truth. Putting ground truth into the bulk is [SPEC-8](./SPEC-8.md) (oracle-grounded self-supervision), with the closest neighbors framed there: Absolute Zero Reasoner (executor-as-verifier, but in the cherry), execution-guided code generation (grounds program I/O, not world transitions), and Snorkel/data-programming (the *noisy* limit of which the oracle is the *exact* case).
- **The author's own prior work** (Proxilion, Invariant, Mantissa, agent-replay, Vaulytica): each is a deterministic verifier over a stochastic process in a security context. Verisim is the generalization — the verifier is no longer guarding a single action; it is grounding a whole simulated world. See the repo's [`docs/lineage.md`](./docs/lineage.md) for the explicit mapping.

---

## 9. Hypotheses (falsifiable)

The project lives or dies by these. Each is stated so that a clean experiment can refute it — and, per the epistemic engine (§10.1), each is pre-registered to a *forward move on both branches*: every hypothesis below names what a refutation (its "honest negative") teaches and which spec it licenses, so no outcome is wasted. Under the oracle a refutation is *bankable* (§10.1, point 1), which is exactly why we can afford to state these as sharply as possible.

- **H1 (the curve exists and is favorable).** There is a regime `0 < ρ* < 1` where the coupled system achieves faithful horizon `H_ε` close to the `ρ=1` ceiling at consultation cost far below it (target: ≥80% of ceiling horizon at ≤20% consultation, on v0). *Refuted if* `H_ε(ρ)` is approximately linear (no free lunch — every faithful step costs a proportional oracle call) across all environments tested.
- **H2 (smart beats dumb).** Drift- or uncertainty-triggered consultation achieves materially higher `H_ε` than fixed-interval consultation at equal `ρ`. *Refuted if* no policy beats fixed-interval within confidence intervals.
- **H3 (correction teaches).** Residual correction / constraint projection yields more faithful horizon per correction than hard reset, and (stretch) reduces future divergence (the model learns from corrections online). *Refuted if* hard reset is statistically indistinguishable or better.
- **H4 (mechanism survives reality).** The curves obtained with the reference oracle qualitatively hold when the system oracle (a real sandboxed shell) replaces it. *Refuted if* real-OS nondeterminism collapses the faithful horizon regardless of consultation.
- **H5 (counterfactual lift).** Oracle-grounding improves interventional-query fidelity over an oracle-free model trained on identical data. *Refuted if* no lift. **Result — objective-dependent (two experiments): the lift exists for the *contrastive representation* (EN9: oracle counterfactual negatives nearly double VICReg's branch-retrieval top-1 at smoke scale) but *not* for plain next-state *supervision* (EN6: counterfactual training examples do not beat a matched-volume trajectory control on held-out intervention delta-exact or change-safety — for supervision a counterfactual is just another labeled transition). So counterfactual structure helps where the objective is interventional/contrastive, not where it is reconstruction. (EN9 itself is scale-fragile, SPEC-9 S2.)**
- **H22 (model-invariance — the benefit belongs to the loop, not the model).** The qualitative shape of `H_ε(ρ)` — floor, knee-or-no-knee, the value of cheap consultation — is governed by the oracle-loop, not by the proposer's architecture: run the *same* loop with materially different proposers (the from-scratch transformer, a JEPA-style latent predictor, an RSSM, a frozen LLM-as-proposer) at matched competence, and the consultation behavior is the same in kind. This is the falsifiable form of §6's commitment 4 and the project's most general claim — that deterministic verification is a *model-agnostic primitive*. *Refuted if* the curve's qualitative shape depends strongly on the proposer class (e.g. a knee appears for one architecture and not another at matched per-step acceptance) — in which case the contribution is narrower (a fact about one model family), not a fact about oracle-grounding. First testable on the existing EN1 machinery (SPEC-5 §12), which is already model-agnostic by construction. **Result — supported in kind (EN7, [SPEC-5 §12](./SPEC-5.md)): the same loop with four proposers (null, flat transformer, graph+RSSM, oracle-backed) gives one qualitative `H_ε(ρ)` shape across the three imperfect ones — floor + cliff, no knee — with the proposer's competence setting the floor height (graph > flat > null) and the loop governing the shape. The EN1/K4 no-knee verdict is not an artifact of the flat arm; it reproduces across architecture. Honest caveat: not matched competence, so the shared *shape* (not magnitude) is the evidence.**

The next three concern *where the oracle's truth enters training* — the supervision-taxonomy claim of §8 and [SPEC-8](./SPEC-8.md). They are claims about the *signal*, where H22 is a claim about the *proposer*; all four are designed to run together on the SPEC-5 apparatus (EN7–EN9).

- **H23 (the collapse tax is a workaround for a missing oracle).** A JEPA-style latent predictor with an **oracle-anchored target** (the true next-state / the exact divergence, not a learned EMA target) matches or exceeds its EMA+VICReg-regularized twin on faithful horizon *and* representation health (embedding rank/variance) with the collapse-prevention terms **ablated** — i.e. the anti-collapse machinery is a proxy for the external referent the oracle supplies directly. *Refuted if* the representation collapses without EMA/VICReg even under the oracle-anchored target (collapse has a cause the referent does not reach — itself a clean result about *why* JEPA needs its crutches). Tested as EN8 (SPEC-5 §12, SPEC-8 §4.1).
- **H24 (residual supervision beats raw-likelihood supervision).** At matched compute, a model trained to minimize **bits-to-correct** (the conditional description length given oracle access, SPEC-3 §7) — masking the gradient on the oracle-decidable bits and concentrating it on the genuinely-uncertain residual — reaches higher faithful horizon per oracle-bit than one trained on full next-state likelihood. *Refuted if* the two are indistinguishable (the decidable part was already cheap to learn; the partition is not load-bearing). Tested as EN8 (SPEC-8 §4.2).
- **H25 (oracle hard-negatives are an exact anti-collapse referent).** A contrastive objective with **oracle-mined one-edit-wrong and counterfactual negatives** matches or beats VICReg-style regularizers at preventing collapse, and the counterfactual negatives additionally lift interventional fidelity (the RQ4 / H5 lift). *Refuted if* exact near-miss negatives add nothing over statistical regularizers. Tested as EN9 (SPEC-8 §4.3).

**Scale-sharpened forms (H23-S / H24-S / H25-S).** H23–H25 are first pre-registered at *smoke* scale (one seed, a 5-host world, a tiny arm — the committed OG3/OG4 figures). Their scale-dependent forms — the prediction that the oracle's advantage *holds or grows* with world/model size and shows **disjoint bootstrap CIs** (the "cannot be dismissed" bar), and the design subtlety that H24 is a *capacity-allocation* claim requiring world size raised at fixed capacity — are specified in [SPEC-8 §7.1–7.3](./SPEC-8.md) and built as milestones OG5 (the local, CPU-proven harness) and OG6 (the scaled runs). No new global hypothesis number: the `-S` suffix marks "the same claim, now with CIs across scale," and either branch (a separated trend, or a CI-bounded null at scale) is bankable per the epistemic engine (§10.1). **Scaled verdicts (committed; the full local surface, 5→200 hosts, [SPEC-9 §4](./SPEC-9.md)).** The smoke
results survive scaling *unevenly*, and the honest mix is itself the finding: **H23-S confirmed but
attenuating** (the collapse gap is disjoint-positive at every world×capacity cell, but shrinks with
scale — the oracle's anti-collapse advantage is real everywhere and diminishing); **H25-S/H5 confirmed at
small scale, reverses at scale with a fixed negative count, then *recovers* when negatives scale** (VICReg
overtakes the oracle at 100–200 hosts/`d128`, but scaling `k_negatives` 8→32 flips the lift back to
disjoint-positive — a confirmed negative-count artifact, fixed modestly by feeding negatives that scale
with the world);
**H24 regime-dependent** (masking the decidable bits in the *loss* helps only narrowly — high capacity,
moderate `R`, small world — and hurts where `R` is tiny, because it removes beneficial multi-task signal;
the *inference-time* partition where the oracle supplies `D` is untouched). The lesson the scaling
made visible: smoke-scale wins can attenuate or reverse, and the oracle is what lets us *see* it.

The last hypothesis scales the *prime directive itself* — the faithful-horizon curve — along model capacity (the standing "is the floor a scale artifact?" objection, [SPEC-10](./SPEC-10.md)).

- **H26 (faithful horizon scales with capacity).** Holding the world fixed and sweeping model capacity across ~100× of parameters (on the free, exact oracle data SPEC-9's regime makes affordable), free-running faithful horizon `H_ε(ρ=0)` grows materially with scale and the **horizon efficiency** `η = H_free / H_indep` (measured against the i.i.d. no-compounding prediction `p/(1-p)`) rises toward 1 — i.e. the floor+cliff that defined v0/EN1/EH1 softens with scale rather than being a fundamental compounding wall. *Refuted if* per-step accuracy `p` rises with capacity but `H_free` does not (η low and flat across the range) — the one-step→horizon gap is governed by compounding, not capacity, and verification is a primitive no reachable model size escapes. Tested as HS1 ([SPEC-10 §3](./SPEC-10.md)). **Result — SUPPORTED with a sharp nuance (HS1, [`horizon_scaling`](../../src/verisim/experiments/horizon_scaling.py)): free-running `H_ε(ρ=0)` lifts ~9× with capacity (1.75 → 15.8 steps over a 32× param range, disjoint bootstrap CIs) on free oracle data, then saturates by mid-capacity (l does not beat m), and the lift transfers to the harder adversarial regime. So the v0/EN1/EH1 floor+cliff was in substantial part an under-resourced-model artifact, not a fundamental compounding wall — and η = `H_free`/`H_indep` stays > 1 throughout (the model free-runs longer than the i.i.d. prediction; no compounding penalty appears at this scale). Honest caveats: this measures the `ρ=0` floor height, not a favorable consultation knee (still open), and the lift saturates early. The scale confound the project always named is here measured and found load-bearing for the headline metric. RESOURCED FRONTIER (HS1.1, [SPEC-10 §4.2](./SPEC-10.md), [`horizon_scaling_xl`](../../configs/horizon_scaling_xl.json)): removing the `l`-undertraining + fixed-data confounds and extending the axis ~400× (4,800-transition coverage, capacity-scaled steps, `xl`/`xxl` to 410k params) reveals the "saturation" is actually a *non-monotone* curve — `H_free` peaks at `l` (17 id / 28 ood) then *declines* (xxl 9.6 id), a compute-optimal frontier for faithfulness; the floor lifts ~4× from resourcing even at fixed tiny capacity (xs 1.75 → 6.83); and decisively, across the top the one-step `p` stays flat/high (0.81–0.90) while `H_free` falls ~45% and ood η crosses below 1 — a per-step-more-accurate model that is *less faithful over the horizon*, a proxy/truth divergence only the exact oracle can measure. The `xl`/`xxl` decline is confounded between capacity-compounding and fixed-data overfitting; the data cross-axis (HS1.2) is queued to separate them.**

---

## 10. Evaluation methodology

- **Environments:** start with v0's deterministic shell/filesystem sandbox (SPEC-2 §2); scale to multi-host network state and syscall traces in later phases. Multiple environment *instances* (different difficulty, branching, state size) to test generality, not a single benchmark.
- **Protocol:** for each (environment, model, `π_c`, `C`, `ρ`, `ε`, seed), run N autoregressive rollouts against ground-truth rollouts; record divergence trajectories and `H_ε`. Aggregate with confidence intervals over seeds. All configs versioned; all runs reproducible from a seed (SPEC-2 §12).
- **Baselines:** (b0) pure neural, `ρ=0`; (b1) oracle every step, `ρ=1` (ceiling/sanity); (b2) symbolic-only (the oracle alone — trivially perfect, included to frame what the neural model is *for*: cheap rollout between consultations); (b3) trivial/frequency predictor (floor). The contribution is the interior and the policy/operator comparisons, not beating b1 (which is unbeatable on fidelity but maximally expensive).
- **Ablations:** representation (delta vs full-state), model size, training objective (supervised vs +RLVR), `ε` sweep, environment size sweep.
- **Reporting:** every claim ties to a hypothesis in §9 and a figure. Negative results are first-class and reported (the field needs honest faithful-horizon numbers more than it needs a hero result).

### 10.1 The epistemic engine — why every result advances the program ("all data is good data")

This is the project's operating stance, and it is not a motivational slogan — it is a property the
oracle *earns* for us, and it is the reason this program can run indefinitely without ever producing
a wasted experiment.

**1. The oracle makes a negative result *trustworthy*.** In an oracle-free domain (vision, language,
robotics) you can never cleanly separate *"the hypothesis is false"* from *"my measurement is broken."*
A flat curve might mean there is no favorable regime, or it might mean the proxy metric is miscalibrated,
the held-out set leaked, or the simulator was subtly wrong. The ambiguity is structural, and it is why
the field accumulates irreproducible "negative" folklore that no one can build on. **A deterministic
oracle removes the ambiguity.** Because ground truth is exact, free, and re-executable, a negative result
is a *fact about the world*, not an artifact of the instrument: when `H_ε(ρ)` is flat under the oracle,
faithful simulation genuinely costs oracle calls roughly linearly *in that world*, and that statement is
as solid as the positive ones. This is the deeper, less-obvious dividend of §2's asymmetry — the oracle
does not only make positive curves measurable; **it makes negative curves *bankable*.** A bankable
negative is the most valuable kind of data, because the whole field can build on it.

**2. Every outcome is pre-registered to an action.** For each hypothesis in §9, and each experiment in
the world/method specs, we state *in advance* what we will conclude and what we will build next under
**both** confirmation and refutation — never only the hoped-for branch. This is pre-registration in the
clinical-trial sense (a "line of retreat" fixed before the data arrives, so no result can be quietly
reinterpreted into a win), and it is what guarantees the property the title claims: an experiment with a
pre-registered implication on *every* branch *cannot* fail to inform. The worked example is already in the
record: v0's E1 returned a null (no knee; the model drifts at step 0), and that null did not stall the
program — it *diagnosed* under-data/under-training and *licensed* SPEC-2.1, which lifted clean faithfulness
from ≈0 to 0.86; SPEC-2.1's own K4 then refuted the knee on the single-filesystem world, and *that*
negative *licensed* the network world (SPEC-5), where drift is gradual and observation is partial. Two
negatives in a row, and the program moved *forward* on both — because each was bankable (point 1) and
pre-registered to a next step.

**3. The program is a ratchet against the test harness of reality.** The roadmap (§12) is gated on
evidence, not calendar: each world graduates only when it produces a committed figure — a knee *or* a
bankable negative that explicitly licenses the next world. The autonomous research engine (SPEC-4) is the
inner ratchet (keep-if-better, with *every* rejected trial logged as data that maps the loss landscape,
SPEC-4 §6); the world-to-world progression (SPEC-2 → 5 → 6 → 7) is the outer ratchet; the system oracle
(SPEC-3 §2) is the move from "faithful to a model of reality" to "faithful to reality itself." No rung is
ever lost: a refuted hypothesis tightens the next hypothesis, a hit wall (§11, and the per-world wall
taxonomies) *names* the next world's central problem, and a metric that saturates motivates the next
metric (bits-to-correct succeeding exact-match, SPEC-3 §7). The intent is dedication to the duty of the
scientific method, iterated against the one judge that cannot be argued with.

**4. Trade-offs are named, not hidden — and each names the next move.** Rigor demands we state the limits
plainly; persistence demands we treat each limit as the next problem, not a stop sign. The standing
trade-offs, each with its forward move: the reference oracle is a *model* of reality, not reality → the
system oracle (SPEC-3 §2) closes it. A world may be too easy to drift → engineer difficulty and move up
the world ladder (SPEC-2 §2.4 → SPEC-5/6/7). The favorable curve may not exist in a given world → that is
a bankable negative that licenses the next (point 1). The oracle is cheap only in computer domains → that
is the *whole point* (the asymmetry, §2), and the *techniques* (consultation policy, correction operators,
oracle-grounded objectives) transfer even where the oracle does not (§4). None of these is a reason to
slow down; each is a reason the next spec exists.

> **The commitment, stated once:** we will report what is true with the same energy whether it confirms or
> refutes, because under the oracle both are real signal — and we will always have already decided, before
> the figure is plotted, how a refutation moves the program forward. That is how a research program becomes
> un-stoppable without becoming dishonest.

---

## 11. Scope

**In scope (the research program):** computer/digital environments with a constructible oracle — filesystems, shells, processes, syscalls, constrained networks, simple web/API state machines. The faithfulness-vs-budget science. The neuro-symbolic oracle-in-the-loop architecture. The two field contributions (world-model metrology; ACD sim-to-emulation).

**Explicitly out of scope:** visual/physical world models (no oracle — that's the whole point); building a better inference *engine*; building a product/SaaS (this is a research repo, MIT, no telemetry, no commercial path); modeling environments so large or nondeterministic that no practical oracle exists (that's the *frontier* we approach gradually, not the starting point).

**Non-goals:** beating Genie on video; beating vLLM on throughput; shipping an autonomous offensive cyber agent (see §13 ethics).

---

## 12. Research roadmap

Phases are gated by **evidence**, not calendar: **no stage graduates from design to build until the prior stage has produced a committed figure showing its knee — `H_ε(ρ)` materially above the floor — or an honest negative that explicitly licenses the next.** Detailed engineering milestones for Phase 0–1 are in SPEC-2 §13; the active, gated build sequence is below.

> **Build order & status (updated 2026-05) — this is the canonical sequence.** After v0's apparatus shipped, the `H_ε(ρ)` curve was an honest null (no knee; the model drifts at step 0 — [docs/report.md](../report.md)). The program therefore **paused its roadmap** to finish Phase 0 first. The single active spec is **[SPEC-2.1 — Earning the Knee](./SPEC-2.1.md)**; every design spec below it is on hold until its predecessor's knee exists.
>
> 0. **✅ v0 apparatus** — deterministic core + propose-verify-correct loop + neural model + E1–E4 experiments (SPEC-2 §13, M0–M8). Built and tested.
> 1. **✅ Earn the knee** ([SPEC-2.1](./SPEC-2.1.md)): **K0** diagnose → **K1** data → **K2** train → **K3** difficulty → **K4** curve. **Done.** The learner is proven and the floor lifted (clean faithfulness ~0 → 0.86), but C-knee is **refuted on the single-FS world** (floor+cliff under every policy; discrete errors make first-exceedance `H_ε` reset-resistant). Per SPEC-2.1 §10 the honest negative **licenses the network world** → Stage 4 is now active.
> 2. **⏸ System oracle** (SPEC-3 §2, milestone **S1**): re-run against a *real* sandboxed shell. Gate: **H4** (mechanism survives reality). *(Orthogonal "faithful-to-reality" upgrade; can run anytime.)*
> 3. **⏸ Package the working simulator** — the agent-callable "what-if" tool + faithfulness benchmark + RL env (the community artifact, SPEC-2 §15).
> 4. **▶ ACTIVE — Network world** ([SPEC-5](./SPEC-5.md)): earn the *network* knee, where drift is gradual and partial observability supplies a calibrated signal (the regime the single-FS world lacked). Gate: **H8**.
> 5. **⏸ Host world** ([SPEC-6](./SPEC-6.md)): earn the *host* knee + composition law. Gate: **H13**.
> 6. **⏸ Distributed world** ([SPEC-7](./SPEC-7.md)): earn the *distributed* knee + the tiered oracle. Gate: **H17**.
>
> The **autonomous research engine** ([SPEC-4](./SPEC-4.md)) is the cross-cutting *tool* used at every stage (its AR0 ratchet already drives the SPEC-2.1 search); it advances its own autonomy levels AR0→AR5 as the science allows, and is not itself a stage.

The original phase list, retained for the science framing (each now gated as above):

- **Phase 0 — v0 shell/filesystem world, reference oracle.** Establish `H_ε(ρ)` (H1), fixed vs. triggered consultation (H2), correction operators (H3). Smallest world with the hard (compounding-state) property. *This is the first paper / artifact — and is **not complete** until [SPEC-2.1](./SPEC-2.1.md) earns the knee.*
- **Phase 1 — system oracle.** Swap the reference interpreter for a real sandboxed shell; test H4 (mechanism survives real-OS nondeterminism). Introduce controlled nondeterminism handling. *(Gated on the Phase-0 knee.)*
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

- **Read next:** [SPEC-2.md](./SPEC-2.md) — the concrete v0 environment, oracle, model, metrics, baselines, repo layout, and milestones. Then [SPEC-3.md](./SPEC-3.md) (the depth: real system oracle, networks, partial observability, self-healing, bits-to-correct) and [SPEC-4.md](./SPEC-4.md) (the autonomous research engine).
- **Author:** Clay Good. **License:** MIT (see [LICENSE](./LICENSE)). **No telemetry, no commercial path** — this is a research repo.
- **Citations / state of the art** referenced above (Genie 3, V-JEPA 2, Cosmos, neuro-symbolic world models, CybORG/CyGIL/PrimAITE, VIRF/EIDOKU/SymCode, RLVR) are current as of May 2026; a maintained bibliography lives in [`docs/related-work.md`](./docs/related-work.md) with one-line takes.
- This document will evolve. When a hypothesis in §9 is tested, its result and the experiment that produced it are linked here and the hypothesis is marked confirmed/refuted/open. The spec is the living record of what we believed and what we learned.
