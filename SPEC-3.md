# Verisim — Depth Expansion Specification (SPEC-3.md)

> The third spec in the set. [SPEC.md](./SPEC.md) is the **science** (why oracle-grounded world models matter and how we would know we are wrong). [SPEC-2.md](./SPEC-2.md) is the **v0 build** (the deterministic shell/filesystem core that plots `H_ε(ρ)` once). **This document is the depth**: how Verisim grows from a reference-interpreter toy into a *real* computer/network world simulator with a *real* feedback loop — the system oracle, partial observability, multi-host network state, online self-healing, counterfactuals, a principled information-theoretic faithfulness metric, and the lessons we import from the 2025–2026 world-model / LLM / RL state of the art.
>
> **Companion:** [SPEC-4.md](./SPEC-4.md) — the **Autonomous Research Engine** that drives all of this with the human out of the loop. SPEC-3 is *what the system becomes*; SPEC-4 is *who builds it* (Verisim improving Verisim).
>
> **Scope discipline.** The long-term vision is a world model of *reality across every signal stream* — biological, chemical, the five senses for robotics. **That vision is stated once, in §16, and then deliberately deferred.** Everything normative in §1–§15 is **computers and networks only**, because that is the one domain where the oracle is *cheap, real, deterministic, and resettable today* (SPEC.md §2). We earn the right to the rest of reality by first being unambiguously correct here.

**License:** MIT. **Status:** design spec, pre-implementation. **Audience:** the author, collaborators, reviewers, and the autonomous research engine (SPEC-4) that will read this file as part of its context.

---

## 0. One-paragraph thesis of the expansion

v0 proved the *mechanism* on a model of POSIX; it did not prove *fidelity to reality*, and it hit a hard floor: the from-scratch model drifts on step one (the H1 negative, SPEC-2 §13/M6). The depth expansion attacks both at once. It replaces the reference interpreter with a **system oracle** — the actual sandboxed OS and network, run under controlled nondeterminism — so "faithful" finally means *faithful to a real computer*, not to our model of one. It deepens the neural half from a tiny under-trained transformer into a properly-trained, curriculum-fed, memory-equipped predictor, and adds an **online self-healing loop** so the model adapts *during* a rollout from the oracle's corrections (test-time training against ground truth). It generalizes the fully-observable single-filesystem state into **partial observability** and **multi-host network state**, the regimes that matter for autonomous cyber defense. And it upgrades the headline metric from a divergence ratio to an **information-theoretic "bits-to-correct"** — the number of bits needed to encode the oracle's correction of the model's prediction, a scale-free, architecture-independent scalar that is exactly the right gate for the autonomous research engine (SPEC-4). The organizing insight, made explicit in §8: **Verisim is speculative execution for world states** — the neural model drafts, the oracle verifies, the loop corrects — and the entire 2025-era speculative-decoding literature is a theory of how to spend the verifier optimally, which is RQ2.

---

## 1. Where v0 ended, and the four walls it hit

v0 is a clean, fully-tested, reproducible artifact (SPEC-2 §13: M0–M8 done; E1–E4, objective, representation, calibration figures; RLVR; the autoresearch ratchet). It is also honest about four walls, and SPEC-3 is organized around knocking each down.

| Wall | v0 status | SPEC-3 response |
|---|---|---|
| **W1 — the oracle is a model, not reality.** | Reference interpreter only (SPEC-2 §3.1). | **§2 System oracle**: real sandboxed shell/container + network, determinized. |
| **W2 — the neural model is too weak.** | Clean per-step accuracy ~0.1; drifts at step 0 (H1 negative). | **§5 Deepened world model** + **§7 metric** + SPEC-4 search: data/curriculum/scale/memory, not just more parameters (E4 ruled out raw size). |
| **W3 — the world is fully observable and tiny.** | One filesystem, full state visible. | **§3 network/multi-host state** + **§4 partial observability**. |
| **W4 — the model is static after training.** | No adaptation at inference. | **§6 self-healing**: online test-time training from oracle corrections (the middle loop, H3 made real). |

These are not independent. W2 is the gating one — without a model that sustains a non-trivial horizon, the `H_ε(ρ)` curve has no interesting interior (SPEC-2 §17.5), the consultation policies have nothing to schedule (H2), and the operators have nothing to correct (H3). So §5–§7 (lift the model) are the critical path; §2–§4 (harder, realer worlds) are what make the eventual result *matter*.

---

## 2. The system oracle: reality as the verifier (Phase 1)

This is the single most important step in the whole program. SPEC.md §2 rests on the claim that computer worlds have a *free, perfect, real* oracle; v0 has only a *model* of one. SPEC-3 cashes the claim.

### 2.1 What "system oracle" means

A `SystemOracle` implements the same `Oracle` protocol v0 already defined for exactly this reason (SPEC-2 §3.2) — `step`, `reset`, `determinism_report` — but its `step(state, action)` executes the action against a **real, sandboxed operating system** and reads back the **real** next state. Nothing else in the codebase learns it switched: the loop, metrics, experiments, and training all consume `Oracle`, so `SystemOracle` is a drop-in for `ReferenceOracle` (SPEC-2 §14, the one forward abstraction v0 paid for).

### 2.2 The hard problem is determinism, not execution

Running a command is easy; making "run a command" a *deterministic, resettable, snapshotable function of `(state, action)`* is the entire engineering challenge — and it is precisely the systems-and-security skillset the project is bet on (SPEC.md §4.2). The nondeterminism sources and how each is sealed:

```
source            mechanism to determinize
----------------  --------------------------------------------------------------
wall-clock time   inject a virtual clock (LD_PRELOAD shim / libfaketime / seccomp
                  trap on clock_gettime); freeze or advance it deterministically
randomness        seed /dev/urandom and getrandom() via a deterministic RNG shim;
                  pin language/runtime PRNG seeds
filesystem state  overlay/copy-on-write rootfs (OverlayFS, btrfs/zfs snapshots);
                  snapshot = the CoW layer hash; reset = drop the upper layer
process/PID/inode normalize/canonicalize volatile identifiers out of the observed
                  state (the canonicalization v0 already does for paths, §2.1)
scheduling/conc.  single-threaded or recorded-schedule execution (rr-style record
                  /replay); pin CPU affinity; disable ASLR
network           emulated, namespaced topology (network namespaces, veth, a
                  containerlab/mininet-class harness); no real internet
I/O nondeterminism record-replay (rr, Hermit-style deterministic execution) for the
                  cases the shims miss
```

The `determinism_report()` is not decoration: it is the **honesty surface**. It enumerates which sources are sealed, which are normalized-away, and which are *recorded* (replayable but not reproducible from scratch). Every figure produced with the system oracle carries this report, so a reader knows exactly how much of "deterministic" is true determinism vs. recorded replay. This mirrors the golden-corpus discipline already in v0 (SPEC-2 §12).

The mature tooling makes this tractable rather than heroic: **rr** (user-space record/replay via `perf`+`ptrace`; replay is bit-deterministic), **Hermit** (hermetic isolation of time, thread interleavings, and RNG via syscall interception, with an optional chaos mode for *controlled* interleaving), **gVisor** (user-space syscall interposition), `CRIU` checkpoint/restore for snapshots, and `mininet`/`containerlab` for resettable network emulation. **Hard wall (HW-1): concurrency / thread-interleaving is the one nondeterminism source that record/replay only tames at cost.** Time, RNG, and I/O are sealed cheaply by shims; scheduling is not. *Decision:* the initial system oracle is scoped to **single-threaded or replay-scheduled** workloads, where bit-exact ground truth is actually attainable, and `determinism_report()` declares it. Concurrency is admitted only later, behind recorded scheduling, and disclosed.

### 2.3 Two implementation tiers (build the cheap one first)

- **Tier A — namespaced shell (the v0-faithful step).** A real `/bin/sh` (or busybox) over a real, snapshotted filesystem in a Linux user namespace, with the time/RNG shims above. This is the *minimum* that makes "faithful" mean "faithful to a real shell." It validates **H4** (mechanism survives real-OS nondeterminism, SPEC.md §9) against the *exact* v0 command grammar, so the v0 figures can be re-run head-to-head: reference oracle vs. system oracle on the same actions. **The first SPEC-3 deliverable.**
- **Tier B — pinned container + record/replay.** A pinned-image container (`gVisor`/`runsc` for a syscall-level interposition point, or a microVM) with `rr`/Hermit-class deterministic record-replay for the residual nondeterminism. This is the production oracle for richer commands (real `coreutils`, package state, services) and the substrate for the network world (§3).

> **Design decision (DD-1): canonicalize, do not capture-everything.** The state is the *canonicalized, salient* projection of the real machine (the §2.1 fields plus what later phases add), not a full memory/disk image. Rationale: the divergence metric and exact-match scoring require a canonical form (SPEC-2 §2.1, §7.1); volatile identifiers (PIDs, timestamps, inode numbers, nondeterministic ordering) would make every step "diverge" for reasons orthogonal to the model's competence. *Alternative considered:* full-image hashing — rejected because it conflates irrelevant nondeterminism with prediction error and is not learnable. The art is choosing the projection that is (a) deterministic, (b) complete enough to be a real test, (c) serializable for the model.

### 2.4 The reference oracle does not retire — it becomes the differential test

v0's `ReferenceOracle` stays, with a new job: **differential oracle testing.** For every `(state, action)` we can run both oracles and diff. Disagreements are either (a) a bug in the reference semantics (fix `docs/semantics.md` + the interpreter, as v0 already does via goldens), or (b) a real-OS behavior the reference does not model (a *finding* about where our symbolic model of POSIX is wrong — itself publishable, and a curriculum signal). This makes the reference oracle a cheap, fast pre-filter and a permanent regression guard, not dead weight.

---

## 3. The network world: multi-host state (Phase 4, designed now)

The payoff domain (SPEC.md §4.1: autonomous cyber defense) is **networks**, not single filesystems. SPEC-3 designs the `Environment` generalization now so it is additive, not a fork (SPEC-2 §14).

### 3.1 State generalization

```
NetworkState = {
  hosts:   map[host_id -> HostState]      # each HostState ⊇ the v0 filesystem State
  links:   map[(host_id, host_id) -> LinkState]   # topology + per-link properties
  flows:   map[flow_id -> FlowState]      # active connections, sockets, sessions
  services: map[(host_id, port) -> ServiceState]  # listeners, daemons
  global:  { time_virtual, routing_table, ... }
}
```

Actions generalize from shell commands to **host-local commands + network events** (open a connection, send a packet/segment of a flow, start/stop a service, a scan). The transition is still `O: S × A → S'`; the oracle is still "run the real (emulated) thing and read back the canonicalized state." The delta-prediction target (SPEC-2 §5.1) generalizes to typed network edits (CreateFlow, ChangeRoute, ServiceUp, …) under the same `apply` discipline.

### 3.2 Why this is the same problem, harder

Networks make every v0 difficulty axis worse in exactly the way that makes the science interesting (SPEC-2 §2.4): **long-range dependencies** (a host compromised at step 5 changes what is reachable at step 50), **branching consequences** (a scan's result depends on the entire accumulated topology), **state size** (the serialized state is now multi-host). This is where the `H_ε(ρ)` curve should finally have an interesting interior, because a competent network world model is genuinely hard and oracle correction is genuinely valuable.

> **Design decision (DD-2): the `Environment` boundary is the only new abstraction.** Per SPEC-2 §14, v0 committed to exactly two forward abstractions: the `Oracle` protocol and the `Environment` boundary. SPEC-3 spends the second one here — `State`/`Action`/`Delta`/serialization/divergence behind one interface — so the network world is a new `Environment` implementation, and the loop/metrics/training/search are unchanged. *No third top-level abstraction is added* (SPEC-2 §10 prime directive).

### 3.3 Bridge to ACD simulators

The network `Environment` is designed to interoperate with / be validated against CybORG/CAGE-class autonomous-cyber-defense simulators (CAGE-4, CybORG++, CyGIL, FARLAND; SPEC.md §4.1). The pitch is the sim-to-emulation gap: a *learned* (cheap to roll out) but *oracle-grounded* (verifiably faithful) network model is a candidate fix for the field's central pain — cheap simulators are unfaithful, faithful emulators are too expensive to train against at scale. Verisim sits exactly in between.

> **Opportunity + caution (the unquantified-transferability gap).** The ACD field *asserts* sim-to-emulation transfer (e.g., CyGIL's claim that an agent trained in the simulated CyGIL-S transfers "directly" to the emulated CyGIL-E with "full transferability") but, as of this writing, publishes **no quantified transfer metric** behind it. That is both the opening and the discipline for Verisim: lead with a **quantified `H_ε(ρ)` curve against a real emulator oracle** — the number nobody else has shown — rather than another unmeasured transfer claim. Measuring the gap *is* the contribution.

---

## 4. Partial observability: the realistic regime (Phase 2)

Real computers and networks are **not fully observable** — you cannot read all of an OS, let alone a whole network. v0 assumed full observability (SPEC-2 §1). SPEC-3 introduces it deliberately.

### 4.1 Formalism

The actor (and the model) see an **observation** `o = obs(s)` that is a partial, possibly noisy projection of the true state `s`. The world model must now maintain a **belief** `b_t` over states (or a latent that summarizes history), and prediction is `M_θ: (b_t, a_t) → (b_{t+1}, ô_{t+1})`. The oracle still returns *true* `s'` on consultation — which is now doubly valuable, because it collapses belief uncertainty, not just corrects a point prediction.

### 4.2 Architecture: recurrent state-space, oracle as belief reset

We adopt the **recurrent-state-space-model (RSSM)** pattern from the Dreamer line: a deterministic recurrent carry plus a stochastic latent, trained to predict observations and reconstruct the (partially) observed state. The Verisim twist: on oracle consultation the belief is **reset to the truth** (hard reset in belief space), and between consultations the RSSM rolls forward. The faithful-horizon question becomes "how long does the *belief* stay faithful," and uncertainty-triggered consultation (RQ2/H2) now has a *principled* uncertainty — the latent posterior entropy — instead of decode entropy (which v0 found uncalibrated, SPEC-2 §17.2).

> **Design decision (DD-3): hybrid explicit+latent state.** Keep the explicit, checkable symbolic state for the *observed* part (so divergence stays exact and the oracle stays a bit-for-bit verifier) and add a latent only for the *unobserved* part. Rationale: a fully-latent world model (Dreamer/JEPA) throws away the one asset that makes computer worlds special — an exact, serializable, checkable state. We keep the symbolic skeleton and use latents only where observability forces us to. *Alternative considered:* go fully latent like the SOTA visual world models — rejected; it surrenders verifiability, the entire point (SPEC.md §6).

---

## 5. Deepening the neural world model (knocking down W2)

E4 showed the H1 floor is **not** a raw-parameter problem (scaling 4× did not move clean accuracy, SPEC-2 §9/§17.5). So the levers are elsewhere. This section is the menu, ordered by expected value, drawing the explicit lessons from the SLM / LLM literature (full citations in §10).

### 5.1 Data and curriculum (the phi/"textbooks" lesson) — highest expected value

The model trains on a few hundred iterations over a tiny driver-generated set (SPEC-2 §5.3). The SLM lesson (Microsoft phi line: *Textbooks Are All You Need*) is that **small models become strong with high-quality, well-ordered data**, not more parameters. Concretely:
- **Curriculum.** Order trajectories easy→hard (short, non-cascading commands first; deep `mv`/`rm -r`/branching later). The §2.4 difficulty dial — depth, breadth, fraction-destructive — finally becomes an explicit knob (it was deferred in v0).
- **Coverage-balanced synthesis.** Generate data to *cover the transition space* (every command × every relevant pre-state shape), not to mimic a single driver. Rare-but-decisive transitions (the failure modes where compounding error bites, SPEC-2 §2.2) are oversampled.
- **Hard-negative mining via the oracle.** Use the oracle to find the `(state, action)` where the *current* model is wrong, and train on those — an active-learning loop the oracle makes free.

### 5.2 Pretraining and distillation

- **Pretrain on the reference oracle, fine-tune on the system oracle.** The reference oracle generates unlimited cheap data to pretrain the structure of FS/shell dynamics; the (more expensive) system oracle fine-tunes the residual where reality differs (the §2.4 differential signal). This is the cheap-teacher → real-student distillation pattern.
- **Distill a large model into the deployed small one** if a larger model clears the floor — standard SLM practice, and the deployed model must stay small (single-GPU rollout, SPEC-2 §11).

### 5.3 Memory and long context

Long-range dependencies (a file created at step 5 referenced at step 50) demand the model *remember* structure across the rollout. The serialized state already carries it, but as state grows (networks, §3) the context window binds. Options, in order: bigger `block_size`; a retrieval/memory over prior states; a structural (graph) encoding of the FS/network tree (the deferred §17.3 alternative). Decision deferred to measurement — adopt the cheapest that lifts long-horizon accuracy.

### 5.4 Representation revisited

E4's representation axis already showed **delta ≫ full-state** (SPEC-2 §9, the representation result): predicting localized edits beats regenerating the world, because the hallucination surface is bounded. SPEC-3 keeps delta as primary and explores **structured/graph deltas** for the network world, where edits are over a graph, not a path tree.

> **Design decision (DD-4): fix the data before scaling the model.** E4 is the evidence: parameters are not the floor. The ordered bet is §5.1 (curriculum/coverage/hard-negatives) → §5.2 (pretraining/distillation) → §5.3 (memory) → scale, and the autonomous research engine (SPEC-4) searches this space automatically, gated on the §7 metric. *Rationale:* spend the next unit of effort where the ablation says the signal is.

---

## 6. The self-healing loop: online adaptation from reality (the middle loop)

This is the most literal expression of the user's vision — *"the model heals itself from reality during a rollout"* — and it makes **H3** ("correction teaches", SPEC.md §9) real. v0's `residual`/`projection` operators were horizon-identical to `hard_reset` (the H3 v0 finding) because nothing *learned* from the correction; SPEC-3 adds the learning.

### 6.1 Where it sits — the three nested loops

```
OUTER  (SPEC-4): propose config/code change → train → score vs oracle → keep if better
                 (timescale: per experiment; the autonomous research engine)
MIDDLE (this §): rollout-time test-time training — each oracle consult is a labelled
                 example; take a gradient step so future divergence drops
                 (timescale: per consultation, within one rollout)
INNER  (built):  RLVR — gradient steps against the oracle faithful-horizon reward
                 (timescale: per training batch; src/verisim/train/rlvr.py)
```

All three are "update from reality." The inner and outer loops already exist; §6 is the middle one.

### 6.2 Mechanism

When the loop consults the oracle at step `t`, it obtains the *true* delta `Δ_t = O(s_t, a_t)`. That is a free, perfectly-labelled training example at exactly the state distribution the model is currently failing on. The self-healer takes one (or a few) gradient step(s) on `(serialize(s_t, a_t) → serialize(Δ_t))`, so the *next* time the model is near this state it predicts better — the rollout literally heals.

```
SelfHealer protocol:
    heal(state, action, oracle_delta) -> None   # one TTT step toward the truth

Runner hook (SPEC-2 §6.3 run_rollout, additive):
    on consult at step t:
        truth = O(s_t, a_t)            # already computed for VERIFY
        s ← C(predicted, truth.state)  # CORRECT (existing)
        if healer: healer.heal(s_t, a_t, truth.delta)   # HEAL (new, opt-in)
```

The runner stays generic and torch-free (SPEC-2 §6.3): `SelfHealer` is an optional protocol; baselines and the static model simply do not implement it (exactly how `UncertaintyModel` is handled today). `NeuralWorldModel` implements it with a lazily-created optimizer.

### 6.3 The stability problem (the TTT literature's hard lesson)

**Hard wall (HW-2): a naive gradient step on every correction invites catastrophic forgetting and instability** — overfitting the latest correction degrades in-distribution accuracy elsewhere. This is the documented failure mode of per-step TTT, and the named 2024–2026 fixes map directly onto our recipe:
- **Small learning rate**, few steps per correction (TTT is a nudge, not a retrain).
- **Replay buffer** of past oracle corrections, so each step mixes the new label with old ones (rehearsal against forgetting; cf. AR-TTA).
- **Parameter-efficient updates** — a low-rank adapter (LoRA-TTT) or selective restoration of most weights (PETAL), so the bulk of learned structure is frozen.
- **EMA / model-reservoir teacher** (cf. ReservoirTTA) and a **trust-region revert**: a correction that worsens a held-out divergence probe is rolled back.

> **Design decision (DD-5b): the keep-if-better ratchet governs the TTT step itself.** The single most important lesson from the TTA literature is that the heal must be *gated*, not blind. So the same accept-if-it-improves logic the autonomous engine uses for config search (SPEC-4) is applied *inside* the rollout: take the candidate gradient step, probe a held-out divergence, and keep the updated weights only if the probe improved — otherwise revert. The oracle that grounds the outer loop also referees the middle loop.

### 6.4 Experiment & hypothesis

**E5 (H3-online).** Fix policy and budget at the E2/E3 winner; compare `static` vs. `self-healing` model on the *same* rollouts. Metrics: (a) `H_ε` lift; (b) the decisive one — **does divergence *after* the k-th correction fall below divergence after the (k−1)-th**, i.e., is the model *learning from corrections within a single rollout*? Determinism: the healer mutates weights, so each rollout deepcopies a fresh model (seeded). **Falsifiable:** H3-online is refuted if self-healing is statistically indistinguishable from `hard_reset` on both metrics (the v0 result would then survive at depth — itself a finding: "online correction does not teach at this scale").

> **Design decision (DD-5): heal opt-in, never on by default.** The headline `H_ε(ρ)` curve (H1) must remain a clean measurement of the *static* model; self-healing is a separate, labelled experimental arm. *Rationale:* conflating them would make the curve un-interpretable. The autonomous engine (SPEC-4) may *search over* healing hyperparameters, but the H1 figure is always healer-off.

---

## 7. Information-theoretic faithfulness: the right scalar (the "val_bpb" of Verisim)

The autonomous research engine (SPEC-4) needs a **single, comparable, scale-free** score — the role `val_bpb` plays in Karpathy's autoresearch. v0 uses *mean clean per-step accuracy* (SPEC-2 auto-search), which is serviceable but blunt (it is 0/1 per step; it cannot tell "almost right" from "catastrophically wrong"). SPEC-3 defines the principled metric.

### 7.1 Bits-to-correct (the description-length view)

Define the faithfulness of a prediction as **the number of bits needed to encode the oracle's correction of the model's predicted delta**:

```
bits_to_correct(s, a) = -log2 P_code(Δ_true | Δ̂)            # ideal: a code for the residual edits
H_model = E_{(s,a)} [ bits_to_correct(s,a) ] / bytes(s')   # normalized: bits per state-byte
```

A **perfect** model needs **0 bits** to correct (its prediction is the truth). A useless model needs as many bits as encoding the whole next state from scratch (the trivial code). This is **minimum-description-length** (Rissanen 1978) / **prequential** evaluation applied to a world model — the same "best model = best compressor" lens that Blier & Ollivier formalized for deep nets (*The Description Length of Deep Learning Models*, NeurIPS 2018), and the same family as the bits-per-byte that Karpathy's autoresearch uses as its scale-free progress scalar. Computed *prequentially* over a rollout — sum the per-step correction code-length as the model self-heals (§6) — it measures, in bits, how fast the model is learning reality. The model's quality *is* how much it compresses reality. It is:
- **Scale-free and architecture-independent** (bits per byte of state), so delta-vs-full-state, tiny-vs-large, symbolic-vs-latent are all directly comparable — exactly the property that lets the search (SPEC-4) roam freely, and exactly why autoresearch chose `val_bpb`.
- **Smooth** (unlike 0/1 accuracy and unlike the near-flat `H_ε` at v0 scale), so a ratchet can climb it.
- **Hack-resistant** (§ SPEC-4 safety): you cannot fake a low bits-to-correct without actually predicting the truth, because the code length is measured against the *oracle's* delta.

### 7.2 Practical estimator (v0-implementable)

The ideal `P_code` is uncomputable, so we use a concrete surrogate code: encode the residual as `(edit-ops to delete the model's wrong edits) + (edit-ops to add the oracle's correct edits)` under a fixed, simple prefix code over the delta grammar (SPEC-2 §5.1). Then `bits_to_correct` is the encoded length of the *symmetric difference between predicted and true deltas* — a direct, cheap, deterministic generalization of the v0 divergence metric (SPEC-2 §7.1) from a ratio to a bit count. The model's own predictive distribution gives a tighter (entropy-coded) estimate when available. Both are reported; the simple code is the committed gate.

> **Design decision (DD-6): keep `H_ε` as the headline science, use bits-to-correct as the optimization gate.** They answer different questions. `H_ε(ρ)` is the *scientific* result (how far does faithfulness reach, RQ1) — it is what the paper reports. Bits-to-correct is the *engineering* signal the autonomous engine optimizes (a dense, comparable scalar). *Rationale:* `H_ε` is the right thing to *report* and the wrong thing to *optimize* (sparse, scale-bound); bits-to-correct is the reverse.

---

## 8. The organizing insight: Verisim is speculative execution for world states

This framing is worth stating loudly because it imports a mature 2023–2025 literature wholesale and reframes RQ2.

**Speculative decoding** (Leviathan et al.; Chen et al.; Medusa; EAGLE) runs a cheap *draft* model to propose several tokens, then a single expensive *verifier* (the true model) checks them in parallel and accepts the longest correct prefix. The speedup is governed by the **acceptance rate** and the **optimal draft length**.

Map it term-for-term:

```
speculative decoding            Verisim
------------------------------  ------------------------------------------------
draft model (cheap, fast)       neural world model M_θ  (cheap rollout)
verifier (expensive, correct)   the oracle O            (expensive ground truth)
token                           one world-state transition
accepted prefix length          faithful horizon H_ε between consultations
acceptance rate                 per-step probability d(ŝ, s') ≤ ε
optimal draft length            optimal consultation interval (RQ2 / π_c)
speculative speedup             compute saved per faithful step (SPEC-2 §7.2 cost)
```

So **RQ2 (consultation policy) is the speculative-decoding scheduling problem** — when to stop trusting the draft and verify — and the literature's results on optimal draft length, adaptive acceptance thresholds, and verifier scheduling are directly importable as consultation policies. v0's `fixed`/`drift`/`uncertainty` policies are the naive baselines; the speculative-decoding-derived adaptive policies are the sophisticated ones SPEC-3 adds. This also gives the project a second, independent framing for reviewers ("error-bounded speculative world simulation") and a principled place to stand for RQ2.

Two concrete imports from the 2024–2026 speculative-decoding results (EAGLE-2, Medusa, and the practitioner consensus):
- **Confidence is calibrated to acceptance, so consult on confidence, not a clock.** EAGLE-2 reports draft confidence ≈0.05 → acceptance ≈0.04 and confidence ≈0.95 → acceptance ≈0.98, and uses this to size the draft tree dynamically. The Verisim analogue: a *confidence/uncertainty-triggered* consultation policy should dominate the fixed-interval baseline once the model's uncertainty is calibrated (the §4.2 posterior entropy, not v0's uncalibrated decode entropy) — this is exactly **H9**.
- **There is a floor on useful ρ (HW-3).** Below ~0.5 acceptance, speculative decoding goes *net-negative* — verifying rejected drafts costs more than it saves. The Verisim corollary: when per-step faithfulness drops below ~0.5, the propose-verify loop saves no compute over just running the oracle (b1), so the interesting regime requires a model already past that acceptance floor. This sharpens *why* the H1 floor (acc ~0.1) has no favorable interior, and predicts the structured filesystem/process state (high acceptance, like code: EAGLE's 4.96–5.41× best case) will reach an interesting interior before chaotic network state does.

> **Design decision (DD-7): adopt speculative-decoding scheduling as the RQ2 hypothesis class, and report `H_ε(ρ)` as a *family indexed by policy*, not a single curve.** Because adaptive (confidence-gated) consultation should beat fixed-rate, the headline object is `H_ε(ρ; π_c)` — one curve per policy — and the contribution is the *gap between adaptive and fixed* at equal budget. The new policies are derived from acceptance-rate estimates (the calibrated uncertainty of §4.2/§6), giving the §17.2 recalibration problem a theory to aim at.

---

## 9. Counterfactuals and causality (Phase 3, designed)

RQ4/H5 (SPEC.md §3, §9): can an oracle-grounded model answer "what if the action at step `t` had been `a'`?" more faithfully than an oracle-free one — and can the oracle *train* that fidelity? The design: the oracle generates **paired interventional rollouts** (identical up to step `t`, divergent action at `t`, oracle-true continuations — the schema reserved in SPEC-2 §4). The model is scored on interventional divergence and trained on oracle-generated counterfactual data. This is where "world model" becomes "causal model," the named open problem of the field (SPEC.md §1). Deferred behind §2–§7 but designed now so the data schema does not need a later migration.

---

## 10. Lessons from the state of the art (2024–2026)

Each lesson ends with its concrete consequence for Verisim. *(Citations are consolidated and kept current in [`docs/related-work.md`](./docs/related-work.md); the autonomous research engine refreshes them.)*

- **Generative/visual world models (Genie 3, V-JEPA 2, NVIDIA Cosmos, Sora-class).** All hit long-horizon drift and have *no oracle* to correct it; their mitigations are indirect (scale, data, architecture, regularization). → *Verisim's edge is the oracle they cannot have; do not compete on visual fidelity, compete on verifiable fidelity (SPEC.md §8).*
- **Model-based RL world models (DreamerV3, TD-MPC2, MuZero/EfficientZero).** Learn latent dynamics and plan in them; the "oracle" is the env, treated as the thing to model *away*, not kept in the loop. RSSM is the partial-observability workhorse. → *Borrow RSSM for §4's unobserved part; invert the philosophy — keep the oracle in the loop (SPEC.md §8).*
- **Speculative decoding (§8).** A theory of optimal cheap-draft + expensive-verify scheduling. → *Directly becomes RQ2's policy class (DD-7).*
- **Test-time training / continual learning (TTT layers; test-time training for ARC-style tasks; sleep-time compute).** Online adaptation works but forgets and destabilizes without small-lr/replay/PEFT/trust-region. → *The §6 self-healing recipe is exactly these mitigations.*
- **RLVR (DeepSeek-R1 and the verifiable-reward line).** Verifiable rewards scale where learned reward models hack; outcome rewards are sparse; verifier design and anti-hacking matter. → *The oracle is the ideal verifiable reward for a world model (SPEC.md §6.3); the §7 metric is hack-resistant by construction.*
- **Small-model data-centrism (phi / "Textbooks Are All You Need"; distillation).** Quality/curriculum/coverage beat parameters at small scale. → *§5.1/§5.2; and E4 already proved size is not the floor here.*
- **Information-theoretic eval (bits-per-byte; MDL; prequential/compression-as-comprehension).** Compression is a principled, scale-free competence metric. → *§7 bits-to-correct.*
- **Autonomous cyber defense (CybORG / CyGIL / PrimAITE; the sim-to-emulation gap).** Cheap sims are unfaithful; faithful emulators are too costly to train against. → *§3's learned-but-grounded network model is a candidate bridge (SPEC.md §4.1).*
- **Deterministic record/replay & sandboxing (rr, Hermit, gVisor, network namespaces / containerlab).** Real systems are *made* deterministic by interposition and record/replay, not by luck. → *§2.2 is built on exactly these tools.*

---

## 11. Evaluation and benchmark expansion

- **Re-run v0 figures on the system oracle** (reference vs. system, same actions): the H4 result.
- **Network `H_ε(ρ)` curves** (§3): the regime where an interesting interior is expected.
- **Partial-observability faithful horizon** (§4): belief-faithfulness vs. budget.
- **Self-healing arm E5** (§6): does correction teach within a rollout?
- **Counterfactual fidelity** (§9): interventional divergence, grounded vs. oracle-free.
- **The benchmark artifact** (SPEC-2 §15) grows from a single-FS suite into a *computer/network faithfulness benchmark* with ground-truth labels — the standalone deliverable for the evals ecosystem, now with a *real* oracle behind the labels.

Every figure remains records-only and reproducible from config+seed (SPEC-2 §7.3, §12); system-oracle figures additionally carry the `determinism_report` (§2.2).

---

## 12. How it all slots into the existing repo (no fork)

```
src/verisim/
  env/         # + NetworkState / generalized Environment boundary (§3, DD-2)
  oracle/      # + SystemOracle (§2): SandboxOracle (Tier A), RecordReplayOracle (Tier B)
               #   ReferenceOracle stays as the differential test (§2.4)
  model/       # + RSSM/latent head for unobserved state (§4); structured/graph delta (§5.4)
  train/       # + curriculum / coverage / hard-negative data (§5.1); distillation (§5.2)
  loop/        # + SelfHealer hook in run_rollout (§6, additive); speculative policies (§8)
  metrics/     # + bits_to_correct (§7) alongside divergence/H_ε
  eval/, rl/   # benchmark + verifiers env grow to network/partial-obs (§11)
  auto/        # the search ratchet — driven by SPEC-4's engine
```

The discipline of SPEC-2 §10 holds: **no new top-level directory**, exactly the two forward abstractions already paid for (`Oracle`, `Environment`), every other addition lives inside an existing module behind a small, optional seam. The autonomous research engine and its proposers (SPEC-4) are the one new top-level concern, justified there.

---

## 13. New falsifiable hypotheses (extending SPEC.md §9)

- **H4 (mechanism survives reality).** *(restated, now testable)* The `H_ε(ρ)` curve obtained with the system oracle (§2) qualitatively matches the reference-oracle curve on the same command grammar. *Refuted if* real-OS nondeterminism collapses the horizon regardless of consultation.
- **H6 (data beats parameters at this scale).** Curriculum + coverage + hard-negative training (§5.1) lifts clean accuracy off the H1 floor more than an equal-compute parameter scale-up. *Refuted if* matched-compute scaling wins. *(E4 is the prior that motivates H6.)*
- **H7 (correction teaches online).** Self-healing (§6) reduces divergence *after* corrections within a single rollout, beating `hard_reset`. *Refuted if* indistinguishable (the v0 H3 null survives at depth).
- **H8 (the interesting interior lives in networks).** The multi-host world (§3) exhibits H1's favorable knee (≥80% of ceiling horizon at ≤20% consultation) where the single-FS world did not. *Refuted if* the network curve is also flat/linear.
- **H9 (speculative scheduling dominates).** Consultation policies derived from acceptance-rate estimates (§8/DD-7) beat fixed-interval at equal budget — H2, finally, with a calibrated signal. *Refuted if* they do not.

Each maps to one experiment and one figure (SPEC-2 §9 discipline). Negatives are first-class (SPEC.md §10).

---

## 14. Risks, determinism, ethics

- **Determinism debt (W1's shadow).** The system oracle's `determinism_report` must be honest; any "recorded, not reproducible-from-scratch" source is disclosed on every figure. The largest threat is silently treating recorded replay as true determinism.
- **Scope creep — the standing risk (SPEC.md §13).** SPEC-3 is broad *by design as a roadmap*, but the build order is strict: §2 Tier-A system oracle and §5/§7 (lift the model) come first; networks/partial-obs/counterfactuals are gated on the model clearing the floor. **Resist building §9 before §5 works.**
- **Ethics / dual use (SPEC.md §13).** The network world (§3) is closer to offensive capability than a filesystem toy. The commitments stand and tighten: defensive framing only; no autonomous offensive agent is a goal; any environment encoding real exploit dynamics is reviewed before release and may be held back; the sandbox has no real-internet egress (§2.2). The autonomous research engine (SPEC-4) operates only inside this sandbox.

---

## 15. Milestones (Phase 1+)

Gated by hypotheses, not calendar (SPEC.md §12). Each has a verify check.

- **S1 — System oracle, Tier A (§2.3).** Namespaced real shell + time/RNG shims behind the `Oracle` protocol. *Verify:* differential test vs. `ReferenceOracle` on the v0 grammar agrees modulo the `determinism_report`; v0 figures re-run on it (H4).
- **S2 — Model lift (§5, §7).** Curriculum/coverage/hard-negative data + the bits-to-correct metric. *Verify:* clean accuracy / bits-to-correct improves materially over the v0 floor at matched compute (H6); the autonomous engine (SPEC-4) drives this search.
- **S3 — Self-healing (§6).** `SelfHealer` + runner hook + E5. *Verify:* the H7 experiment runs; the within-rollout learning curve is measured (refuted-or-not, both reported).
- **S4 — Network world (§3).** `NetworkState` `Environment`; network `H_ε(ρ)`. *Verify:* H8 — is there an interesting interior? Bridge comparison to a CybORG-class sim.
- **S5 — Partial observability (§4).** RSSM hybrid; belief-faithful horizon. *Verify:* oracle-as-belief-reset works; uncertainty-triggered policy uses posterior entropy (H9).
- **S6 — Counterfactuals (§9).** Interventional data + fidelity. *Verify:* H5.

S1–S2 are the critical path (real oracle + a model that clears the floor). Everything after is depth.

---

## 16. The horizon: reality across every signal (deferred, stated once)

The user's true target is larger than computers: a world model of **reality itself** — biological, chemical, the five senses for robotics, every data stream — kept faithful by a feedback loop against the real world. SPEC-3 records that vision and then sets it aside, because the *general theory* is exactly what this repo is building and the *generalization is principled, not hand-wavy*:

> **The unifying principle.** Verisim works in any domain that admits a *constructible, resettable, deterministic-enough oracle*. The propose-verify-correct loop, the faithful-horizon metric, the consultation policy, the correction operators, the self-healing loop, and the autonomous research engine are **domain-agnostic**; only the `Environment` and `Oracle` implementations change. Computers and networks are the **beachhead** because there the oracle is *free, real, and exact today*. A wet-lab assay is an oracle for chemistry; a physics engine or a real robot rig is an oracle for the senses; an instrument is an oracle for biology — each is *more expensive, slower, and noisier* than running a sandboxed shell, which is precisely why the *theory of spending a scarce oracle optimally* (RQ2, the speculative-execution framing of §8) must be worked out *here, where the oracle is cheap*, before it can be exported *there, where it is dear* (SPEC.md §4).

So this section is not scope creep — it is the statement that the computer/network program is **the place we learn the method that the rest of reality will need.** Every other signal stream is a future `Environment`/`Oracle` pair behind the same interfaces (DD-2), gated on the method being unambiguously correct here first. **Nothing in §16 is in scope until S1–S6 are done.**

---

## 17. Reading order and provenance

- **Prereqs:** [SPEC.md](./SPEC.md) (the science) and [SPEC-2.md](./SPEC-2.md) (the v0 build) — read both first; this document assumes them.
- **Companion:** [SPEC-4.md](./SPEC-4.md) — the Autonomous Research Engine that executes this spec with the human out of the loop. SPEC-3 says *what to build*; SPEC-4 says *how it builds itself*.
- **Author:** Clay Good. **License:** MIT. No telemetry, no real-internet egress, defensive framing (SPEC.md §13).
- This is a living design spec. When a milestone (§15) lands it is marked and its figures linked, as in SPEC-2 §13; when a hypothesis (§13) is tested its result is recorded in SPEC.md §9. The spec is the record of what we believed and what we learned.
