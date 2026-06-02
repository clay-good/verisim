# SPEC-8 — Oracle-Grounded Self-Supervision

**Cross-cutting method specification: putting ground truth in the *bulk* of the cake,
not only the cherry. How a deterministic oracle supervises a world model across all
three of LeCun's learning layers — and why the layer with the most leverage (self-supervised)
is the one verisim has not yet used.**

> **▶ METHOD SPEC (design, not yet built) — 2026-05-31.** This is a *cross-cutting* spec,
> a sibling to [SPEC-4](./SPEC-4.md) (the autonomous research engine): it is not a world
> vertical (those are SPEC-2 / SPEC-5 / SPEC-6 / SPEC-7) but a *method* that every world
> inherits. It does not invent a new oracle, world, or metric. It re-reads the apparatus
> verisim already shipped — the reference oracle, the `Model` protocol, bits-to-correct
> (SPEC-3 §7), the partial-observation loop (SPEC-5 §8) — and asks a single question the
> existing specs never framed: **the cake of machine learning is built almost entirely out
> of proxies for truth; verisim has a slab of real truth; where in the cake should it go?**
> The honest current answer is "only the cherry" (RLVR reward, [`verisim.rl`](../../src/verisim/rl/));
> this spec is the case for the bulk. It is gated like everything else — no claim here is
> believed until a committed figure (EN8/EN9, SPEC-5 §12) shows it, and an honest negative is
> first-class.

Read [SPEC.md](./SPEC.md) first for *why* computer environments are the one domain with a
free, exact oracle. Read [SPEC-5](./SPEC-5.md) for the world this method is first built in.
This document is *where the oracle's truth enters training* — a question orthogonal to
*which world* and *which proposer*.

---

## 0. One-paragraph thesis

Yann LeCun's "cake" (NIPS 2016, restated with *self-supervised* in place of *unsupervised*
at ISSCC 2019) orders the sources of a learning signal by how much of it each carries: the
**bulk** is self-supervised, the **icing** is supervised, the **cherry** is reinforcement
learning. The taxonomy is about *quantity of signal*. It is silent on a different axis
entirely — *is the signal true?* Self-supervision's signal is **co-occurrence in a fixed
corpus** (true to the data, not to the world); supervision's is a **human annotator**
(true-ish, unscalable); reinforcement's is a **reward model or environment** (usually a
proxy). None of the three layers contains a signal that is at once **free, exact, dense, and
generative**. That is not an oversight — for vision and language there is no oracle, so the
whole cake is necessarily built from *proxies for truth*, and each layer pays a tax to keep a
proxy from being satisfied without being correct (JEPA's collapse-prevention machinery is the
clearest instance, §2.2). A **deterministic oracle is a fourth source of supervision** that is
free like self-supervision, exact like supervision, verifiable like RLVR — and *dense* (a true
label at every transition, not a sparse scalar). It is not a layer of the cake. It is the
**plate**: the slab of literal ground truth the cake's proxies are all standing in for.
Verisim, uniquely, has a plate. The claim of this spec: **in the one domain that admits a free
exact oracle, ground truth should be pressed into the *bulk* of the cake (self-supervision),
not just the cherry (RL) where verisim currently confines it — and doing so removes the proxy
taxes the oracle-free field cannot remove.**

---

## 1. The cake and its missing plate

### 1.1 The two axes the taxonomy conflates

LeCun's ordering is one-dimensional: bulk > icing > cherry, by *bits of signal per example*.
A second axis is hiding under it — **the trust of the signal**: how close the training target
is to ground truth. The two axes are independent, and the field has been forced to treat them
as the same because, outside computer environments, *more signal* and *more truth* trade off:
the cheap, dense signals (self-supervision) are the least anchored to truth, and the most
truthful signals (a human label, a real-world rollout) are the sparsest and most expensive.

| Layer | Bits of signal | Source of the target | Is the target *true*? | Cost |
|---|---|---|---|---|
| **Self-supervised (bulk)** | very high (every token/patch) | the data predicting itself (co-occurrence) | true to the **corpus**, not the world | free |
| **Supervised (icing)** | medium | a human annotator | usually, modulo noise; **unscalable** | high |
| **Reinforcement (cherry)** | low (scalar, often sparse) | a reward model / environment | a **proxy**, hackable | medium–high |
| **Oracle (the plate)** | **high** (a full true next-state, every step) | **a deterministic interpreter of the world** | **exact, by construction** | **free** (in computer worlds) |

The oracle is the row the taxonomy could never include, because no other domain has it. It
breaks the signal-vs-truth tradeoff: it is *both* dense *and* exactly true *and* free. That is
the entire asymmetry SPEC.md §2 rests on, read through the lens of training signal rather than
inference-time verification.

### 1.2 Where verisim puts the oracle today, and the gap

Verisim already uses the oracle in three places — but two of the three are *downstream* of
training, and the one training use is the smallest layer:

- **Inference-time verification** (the propose–verify–correct loop, SPEC.md §5.2). Not a
  training signal at all — the oracle corrects a *rollout*. This is verisim's signature, and it
  is orthogonal to the cake.
- **RLVR — the cherry** (SPEC.md §6 commitment 3, SPEC-4 §10, [`verisim.rl`](../../src/verisim/rl/)).
  The oracle's faithfulness verdict is a verifiable reward; faithful horizon is the return. This
  is the *one* training placement verisim has built, and it is the cherry — ~10% of the cake by
  LeCun's own ordering.
- **Supervised next-state labels — the icing** (SPEC-2 §5.3; SPEC-2.1 §5 / K1, "the oracle is a
  free, infinite teacher"). Verisim *already does* oracle-supervised learning — it just never
  named it that, or asked whether the same free exact signal belongs in the bulk.

**The gap is the bulk.** The 90% of the cake — self-supervised pretraining — is where verisim
has put *none* of its oracle truth, and it is exactly where the oracle-free field is spending the
most effort on proxies (§2). This spec is the case for closing that gap, and the three concrete
mechanisms (§4) for doing it.

> **Design decision (DD-OG-1): the oracle is the plate, not a fourth cake layer.** We do not
> claim oracle-supervision *replaces* self-/supervised/reinforcement learning. We claim it is the
> *ground truth those three approximate*, and that in a domain with an oracle, each of the three
> can be **re-grounded** on it — self-supervision most of all, because it is the largest and the
> least-anchored. The contribution is a *placement*, not a new algorithm.

---

## 2. Lessons folded in (and the design choice each one forces)

The same discipline as SPEC-5 §2 and SPEC-6 §2: each lesson from the 2023–2026 literature, and
the design choice it forces. These are the foils that prove the lane is open — the field is
reaching for exactly this signal and is forced to substitute a proxy because it has no oracle.

### 2.1 The cake itself (LeCun, NIPS 2016 keynote; ISSCC 2019)
The ordering bulk > icing > cherry, with *self-supervised* the bulk. **Design choice:** treat
the cake as the map of *where a signal can enter training*, and insert the oracle row (§1.1).
We adopt LeCun's framing wholesale and add the one thing his domain cannot supply. (The "70/20/10"
sometimes attributed is apocryphal — the claim is qualitative ordering, not percentages; we state
it that way and do not cite numbers he did not give.)

### 2.2 JEPA, VICReg, and the collapse tax (LeCun "A Path Towards Autonomous Machine Intelligence" 2022; I-JEPA, Assran et al., CVPR 2023; VICReg, Bardes/Ponce/LeCun, ICLR 2022; C-JEPA, NeurIPS 2024; V-JEPA 2, 2025)
JEPA predicts the *representation* of the future produced by a learned EMA target encoder. Because
the target is self-referential, JEPA is prone to representation collapse (entire, dimensional, or
mean-learning), and the literature is explicit that **EMA alone does not prevent it** — you need
**VICReg variance/covariance terms** to force the embeddings apart (C-JEPA, NeurIPS 2024). This is
the **collapse tax**: a large fraction of the engineering of self-supervised world models exists
only to stop a self-referential objective from cheating. **Design choice:** this is the sharpest
falsifiable opportunity (§4.1, H23) — replace the learned, self-referential target with the
oracle's *true* next-state, and test whether the collapse tax (EMA, VICReg) becomes unnecessary.
The line we do **not** cross: we never latent-ify the *observable, checkable* part of the state
(DD-3, SPEC-3 §4.2) — the latent is only ever over the genuinely-unobserved residual.

### 2.3 Grounding world-model SSL with *proxy* verifiers (GrndCtrl, arXiv:2512.01952, 2025; World Action Verifier, arXiv:2604.01985, 2026)
GrndCtrl post-trains a world model with *self-supervised* "verifiable" rewards — pose
cycle-consistency, depth reprojection, temporal coherence — and admits it cannot verify semantic
correctness, only measurable geometry. World Action Verifier lets a model check its own predictions
by cycle-consistency among generated subgoals, inferred actions, and forward rollouts. **Both are
the oracle-shaped hole filled with a proxy**: self-consistency and geometry stand in for a truth
they cannot access. **Design choice:** verisim is the *control* these works need — in the domain
with a real oracle, we can measure how well each proxy (cycle-consistency, perceptual distance)
*predicts true faithful horizon*, and export that calibration to the domains that must trust the
proxy blind (the SPEC.md §4 "calibrate the proxy" contribution, now stated as a training-signal
result, not only an eval one).

### 2.4 Executor-as-verifier in the cherry (Absolute Zero Reasoner, Zhao et al., arXiv:2505.03335, 2025; R1-Code-Interpreter, 2025; RLVR / DeepSeek-R1, 2025; process reward models, Lightman et al. "Let's Verify Step by Step," 2023)
Absolute Zero uses a **code executor** to both validate self-proposed tasks and verify answers,
self-evolving its own curriculum with **zero human data** — the purest existing instance of
"a deterministic checker generates and grades its own training set." It is the nearest neighbor to
verisim's autoresearch (SPEC-4) *and* to this spec's generative-data claim — but it lives in the
**cherry** (RLVR self-play over question-answer reasoning), and its checker grades a *final answer*,
not a *world-state transition* at every step. **Design choice:** adopt AZR's "the verifier generates
its own curriculum" wholesale (it is what the oracle-as-infinite-teacher already is, SPEC-2.1 §5),
and push it *down* from the cherry into the bulk: the oracle does not only *reward* a rollout, it
*supervises a representation* (§4). The distinction is the contribution — same checker, lower in the
cake.

### 2.5 Execution-guided generation (Execution-Guided Line-by-Line Code Generation, 2025; Self-Execution Simulation, 2026; the broader execution-guided-synthesis line)
These condition generation on runtime outcomes — execution as a guide/filter at training or
inference. **Design choice:** distinguish cleanly. (i) They ground *program input→output*; verisim
grounds *state×action→next-state* (a world transition), the harder, compounding object. (ii) They
use execution as a *filter* on candidate code; verisim uses the oracle *generatively* — it
manufactures unbounded, coverage-controlled, perfectly-labeled training transitions and *counterfactuals*
(§4.3), not merely accept/reject signals on samples the model already produced. The oracle is a data
*factory*, not only a *gate*.

### 2.6 Weak supervision is the oracle's noisy limit (Data Programming, Ratner et al., NeurIPS 2016; Snorkel, Ratner et al., VLDB 2017)
Snorkel writes *labeling functions* — programmatic, **noisy, abstaining** heuristics — and learns a
generative model to combine them into probabilistic labels, trading exactness for scale. **Design
choice:** name verisim's oracle as the **strong limit of data programming** — a single labeling
function that is *exact, complete, and never abstains*. Everything weak supervision does to *estimate*
label correctness from disagreement, verisim gets for free because the label *is* correct. This is the
cleanest one-line positioning for an ML reader: *oracle-grounded self-supervision is data programming
where the labeling function is perfect, in the one domain where a perfect one exists.*

### 2.7 SSL-as-compression / MDL (the description-length view; bits-per-byte; verisim's bits-to-correct, SPEC-3 §7)
Self-supervision is compression: predict to minimize description length (`val_bpb`). **Design
choice:** this is the bridge from the metaphor to a concrete objective (§4.2). Verisim already has the
*conditional* description length — **bits-to-correct**, the residual a model still owes *after* the
oracle is consulted. Promoting it from an eval metric (its role today) to a *training objective* is
literally "spend gradient only on what the oracle does not give you for free." The partition principle
(§3) is the MDL statement of "offload deterministic compute to the oracle."

---

## 3. The partition principle (why "not everything is deterministic, stochastic, or probabilistic")

The owner's framing — *not everything can be 100% probabilistic, stochastic, or deterministic* —
is, stated precisely, a **partition of the next-state into two regimes that demand different
treatment**:

- **The decidable part `D(s, a)`** — the bits of `s' = O(s, a)` that the oracle fixes exactly and
  cheaply (in a fully-observed computer world, *all* of them; under partial observation, the bits
  implied by what is observed plus the deterministic semantics). For these, the right operation is
  **verify, do not learn**. Burning network capacity to memorize them in-weights is the waste —
  the model relearns, lossily and expensively, a function the oracle computes perfectly for free.
- **The residual part `R(s, a)`** — the bits left genuinely uncertain given the *observation* (which
  unobserved host changed; a seeded nondeterminism source; a latency the partial view hides). For
  these, the right operation is **learn, because no cheap oracle resolves them** — this is where a
  probabilistic model earns its keep.

> **Design decision (DD-OG-2): respect the partition — the model's job is `R`, the oracle's job is
> `D`.** A well-grounded objective concentrates the model's capacity on the residual `R` and *masks*
> the decidable `D`, because `D` is free from the oracle. "Even nature offloads": evolution does not
> store the laws of chemistry in the genome; it offloads them to physics and encodes only what physics
> leaves underdetermined. The same division of labor is the right training objective for an
> oracle-grounded model — and bits-to-correct (§4.2) is its loss.

This makes the neuro-symbolic split of SPEC.md §6 a statement about *training*, not only
*architecture*: the symbolic half (oracle) owns `D`; the neural half (model) owns `R`; and the
*objective* enforces the split instead of leaving the model to rediscover it from data.

---

## 4. The three mechanisms (the concrete method)

Each mechanism re-grounds one cake layer on the oracle. Each is a falsifiable ablation against the
proxy the oracle-free field is forced to use, and each is implementable in the **network world**
first (SPEC-5), because it already ships the JEPA/RSSM latent arm (§6.2), drift mitigations (§6.3),
and bits-to-correct (§5.4). The proposer stays a pluggable part (the `Model` protocol) throughout —
these are objectives and data sources, not new architectures.

### 4.1 Oracle-anchored predictive targets (re-grounding the *bulk*; removes the collapse tax)
**The proxy:** JEPA regresses the model's latent prediction onto a *learned EMA target encoder* of
the true future, and props the self-referential objective up with VICReg variance/covariance to stop
collapse (§2.2).
**The grounding:** replace (or regularize) the target with the **oracle's true next-state** — or, in
latent form, regress the predicted-state latent so that **latent distance reproduces the oracle's exact
divergence `d(s, ŝ)`** (the metric verisim already computes: set-difference, graph divergence). The
target is no longer the model's own moving encoder; it is ground truth. There is nothing for the
embedding to collapse *toward*, because the geometry is pinned to an external, non-degenerate referent.
**The claim (H23):** with an oracle-anchored target, the collapse-prevention machinery (EMA target,
VICReg variance/covariance) becomes **unnecessary or strictly dominated** — the tricks are a workaround
for a missing external referent, and where the referent exists they cease to pay. *This engages
LeCun's own program on its own terms.*

### 4.2 Residual self-supervision: bits-to-correct as the objective (the *partition*, the *offload*)
**The proxy:** standard SSL maximizes the likelihood of the *entire* next-state (or reconstructs it),
spending capacity on every bit equally — including the bits the oracle fixes for free.
**The grounding:** make the objective the **conditional description length given oracle access** —
i.e., minimize **bits-to-correct** (SPEC-3 §7), the residual the model still owes after a consultation,
rather than the raw next-state likelihood. Mask the gradient on the decidable part `D` (§3); concentrate
it on the residual `R`. The model is trained to be *cheap to correct*, not to *reproduce what is already
free*.
**The claim (H24):** at matched compute, a model trained to minimize bits-to-correct reaches longer
faithful horizon per oracle-bit than one trained on raw next-state likelihood — because its capacity is
spent on the part the oracle cannot supply. This also *aligns the training objective with the
inference-time metric*, which the current supervised objective does not.

### 4.3 Oracle as a hard-negative & counterfactual factory (re-grounding *contrastive* SSL)
**The proxy:** contrastive/JEPA SSL is bottlenecked on **negatives** and on collapse; VICReg's
covariance term is a *statistical* stand-in for "push representations apart."
**The grounding:** the oracle generates the **most informative negatives for free** — the
*one-edit-wrong* next-states (the bits-to-correct neighborhood) and *counterfactual* next-states
(branch the action `a → a'`, run the oracle), each perfectly labeled "not the true successor." These are
exactly the near-miss negatives contrastive learning most wants and can least easily mine. K1 already
builds a weak version for the *supervised* trainer (SPEC-2.1 §5); this lifts it to a contrastive SSL
objective with an **exact** anti-collapse referent.
**The claim (H25):** oracle-mined hard negatives give a contrastive objective an external,
non-degenerate referent that matches or beats VICReg-style regularizers at preventing collapse — and the
*counterfactual* negatives additionally improve interventional fidelity (the SPEC.md RQ4 / H5 lift),
because the model is trained on the very branches it will be asked to predict.

> **Design decision (DD-OG-3): the headline metric stays exact; the oracle-grounded *targets* are
> internal.** As in SPEC.md §4, any learned/latent quantity (the anchored target, the residual
> objective, the contrastive geometry) is an *internal* training signal only. The reported faithfulness
> stays the bit-exact, oracle-grounded `H_ε` and bits-to-correct (§7). The oracle calibrates the proxy;
> it is never substituted *for* the truth. The judge is not a knob (DD-AR2, SPEC-4 §5).

---

## 5. Design decisions (the lines this method holds)

- **DD-OG-1 — the oracle is the plate, not a fourth layer.** (§1.3) A placement, not a new algorithm.
- **DD-OG-2 — respect the partition.** (§3) The model's capacity is for the residual `R`; the oracle
  owns the decidable `D`; the objective enforces the split. **Empirical caveat (§7.2 Result, SPEC-9 S3):
  the *training-objective* half of this — masking `D` in the loss — is refuted at local scale (it removes
  beneficial multi-task signal, not wasted capacity). The decision survives in its *inference-time* form
  (DD-OG-3): the oracle supplies `D` at consultation; the model is simply never *trusted* on it. Keep `D`
  in the loss; let the oracle own `D` at inference.**
- **DD-OG-3 — exact headline, internal grounding.** (§4) Oracle-grounded targets are internal; the
  reported number stays bit-exact and oracle-grounded.
- **DD-OG-4 — never latent-ify the checkable part.** Inherited from DD-3 (SPEC-3 §4.2). The
  oracle-anchored *latent* (§4.1) is only ever over the genuinely-unobserved residual; the observable,
  checkable state is always predicted in the exact, verifiable delta representation. Latent-ifying the
  checkable part would surrender the bit-for-bit verifiability that is the whole asset.
- **DD-OG-5 — additive, not a fork.** This method changes only *objectives and data sources*. It reuses
  the shipped `M_θ`, `Model` protocol, oracle, divergence, bits-to-correct, and `netloop` unchanged
  (§8). If the mechanisms do not pay, they are deleted with no residue.

---

## 6. Hypotheses (H23–H25, non-colliding with H1–H22)

Each is falsifiable and names its honest negative, per SPEC.md §9. All three are first testable on the
existing, model-agnostic SPEC-5 apparatus (EN8/EN9, §7).

- **H23 — the collapse tax is a workaround for a missing oracle.** A JEPA-style latent predictor with an
  **oracle-anchored target** (§4.1) matches or exceeds its EMA+VICReg-regularized twin on faithful
  horizon *and* on representation health (embedding rank/variance), with the collapse-prevention terms
  **ablated**. *Refuted if* removing EMA/VICReg collapses the representation even with the oracle-anchored
  target — i.e. collapse has a cause the external referent does not address, and the tax is intrinsic, not
  a proxy. (Either way it is a clean, reportable result about *why* JEPA needs its crutches.)
- **H24 — residual (bits-to-correct) supervision beats raw-likelihood supervision.** At matched compute, a
  model trained to minimize **bits-to-correct** (§4.2) reaches higher faithful horizon per oracle-bit than
  one trained on full next-state likelihood, because capacity is concentrated on the residual `R`.
  *Refuted if* the two are indistinguishable — the decidable part `D` was already cheap for the model to
  learn, so masking it buys nothing (the partition is not load-bearing at this scale).
- **H25 — oracle hard-negatives are an exact anti-collapse referent.** A contrastive objective with
  **oracle-mined one-edit-wrong and counterfactual negatives** (§4.3) matches or beats VICReg-style
  regularizers at preventing collapse, and the counterfactual negatives additionally lift interventional
  fidelity (the H5 / RQ4 lift). *Refuted if* oracle negatives add nothing over statistical regularizers —
  the near-miss structure of `D` is not what contrastive collapse was about.

These compose with H22 (model-invariance): if the oracle-grounded objective only helps *one* proposer
class, that is itself informative (a fact about that architecture, not about oracle-grounding). H23–H25
are claims about the *signal*, H22 about the *proposer*; EN7 (the model-invariance sweep) and EN8/EN9
(below) are designed to be run together.

---

## 7. Experiments (instantiated in the network world: EN8, EN9)

Per repo convention, cross-cutting methods instantiate as experiments *in the world where they run*;
the active build is the network world, so these extend SPEC-5's EN-series (EN1–EN7 used; EN8/EN9 are
non-colliding). They are added to [SPEC-5 §12](./SPEC-5.md). Host/distributed instantiations
(EH-/ED-series) follow only after the network result, gated as ever.

- **EN8 — objective grounding ablation (H24, H23).** Cross the *training objective* against the
  *collapse-prevention machinery*, on the NW8 graph+RSSM/JEPA arm:
  - objective ∈ { raw next-state likelihood (today's supervised baseline), **bits-to-correct residual**
    (§4.2) };
  - target ∈ { learned EMA target + VICReg (the JEPA baseline), **oracle-anchored target** (§4.1) };
  - collapse terms ∈ { on, **ablated** } — the H23 cell.
  Report faithful horizon `H_ε(ρ)`, bits-to-correct, and representation health (embedding rank/variance,
  the collapse diagnostics). *Does grounding the target remove the collapse tax (H23)? Does residual
  supervision beat likelihood (H24)?* **Honest negative** wired in: if the oracle-grounded cells do not
  beat the proxy cells, report it — it bounds how much the bulk-placement buys, and that bound is the
  result.
- **EN9 — oracle hard-negative / counterfactual contrastive (H25, H5).** Add an oracle-mined contrastive
  loss (one-edit-wrong + counterfactual negatives, §4.3) to the SSL pretraining objective; ablate against
  VICReg-only and against no-contrastive. Measure collapse, faithful horizon, and **interventional
  fidelity** on the branch-replay counterfactuals (the EN6 / H5 set). *Do exact near-miss negatives beat
  statistical regularizers, and do counterfactual negatives lift interventional fidelity?*

Both reuse the EN1 machinery (the `H_ε(ρ)` sweep, bootstrap-CI aggregation, the frozen eval discipline of
SPEC-4 §5.2) and the `figures/reproduce.sh` regenerate-from-seed rule. Negative results are first-class
(the v0 norm).

### 7.1 From smoke to a result that cannot be dismissed (the scale-up, OG5–OG6)

The OG3/OG4 figures are *smoke* instances: one `model_seed`, a 5-host world, a `d_model=48`/3-round arm,
~120 training examples. The verdicts (H23 confirmed, H24 near-tie, H25/H5 lift) are real but **dismissible
on four specific grounds**, in descending severity:

1. **No error bars** — every headline number is `n = 1` (`model_seed = 0`). A reader calls it noise, and at
   one seed they are not wrong. *This is the cheapest and most important to fix.*
2. **Toy world** — 5 hosts / 3 ports, the hardcoded `DEFAULT_NET_CONFIG`. "An artifact of a trivially small
   world."
3. **Tiny model** — `d_model = 48`, 3 message-passing rounds, 2 decoder layers. "The collapse and the lift
   are because the model is undersized."
4. **A single operating point** — no demonstration the effect *survives* scale.

The pre-registered answer to all four is **not one larger run but a scaling curve with disjoint confidence
intervals**: sweep world size and model size across many `model_seed`s, and show the oracle's advantage is
*stable or growing* with **non-overlapping bootstrap CIs**. A point estimate is dismissible; a monotone
trend with separated CIs across a ≥10× world-size sweep is not. The three headline gaps the curves track —
each a *difference* the oracle buys, reported with a bootstrap CI over seeds (reusing
[`metrics/aggregate.bootstrap_ci`](../../src/verisim/metrics/aggregate.py)):

| Gap (the oracle's advantage) | Definition | The undismissable target |
|---|---|---|
| **collapse gap (H23-S)** | `eff_rank(oracle, machinery off) − eff_rank(learned, machinery off)` (and the `emb_std` analogue) | > 0 with disjoint CIs, *stable or growing* with world/model size |
| **residual-objective gap (H24-S)** | `residual_acc(residual obj) − residual_acc(likelihood obj)` on the bits `R` | **Refuted at local scale (§7.2 Result):** no disjoint-positive cell; masking `D` removes beneficial training signal — the *training-objective* partition does not pay, the *inference-time* partition stands |
| **interventional lift (H25-S / H5)** | `top1(oracle) − top1(vicreg)` (and MRR) on held-out counterfactual branches | > 0 with disjoint CIs, and *widening* as more hosts create more distinct branches (chance `1/m` falls) |

These are scale-sharpened forms of the existing H23/H24/H25 (SPEC.md §9) — no new global hypothesis number;
the `-S` suffix marks "the smoke verdict, now with CIs across scale." Either branch is a result: a separated
trend confirms; a CI-bounded null at scale is the bankable negative (§10).

### 7.2 The capacity-binding subtlety (why H24's scale axis is *world size at fixed capacity*)

A measurement made while building this plan: with `observed_fraction = 0.5`, the **residual fraction of the
delta tokens is only ≈ 0.16, and it barely moves as hosts grow** (0.160 at 5 hosts → 0.187 at 20). The
reason is structural — the decidable part `D` already *dominates* the delta (the always-decidable global
clock/result edits plus the structural `<eos>` are ~84% of tokens), and each action is still ≈ one edit
regardless of world size, so scaling hosts does **not** by itself enlarge `R`.

The consequence for H24 is a sharpening, pre-registered here: **H24 is a *capacity-allocation* claim, not a
world-size claim.** Masking `D` (the residual objective) can only beat raw likelihood when (a) the residual
`R` is genuinely *hard* and (b) model capacity is *binding* against it — otherwise both arms fit `R`
trivially and the gap is a tie forever (exactly the OG3 smoke result). So the H24-S sweep must be designed,
not stumbled into:

- **Scale world size to make `R` harder** — more candidate hosts/ports means a residual edit's exact
  host/port identity is one of many, so bit-exact residual prediction stops being free.
- **Hold (or under-provision) model capacity** while the world grows — the residual objective's edge appears
  precisely where the all-token objective wastes a binding budget on the free `D`.
- **Sweep `observed_fraction` down** (e.g. 0.5 → 0.25) to enlarge `R` directly, as a second axis.

If, designed this way, the residual gap still does not open, that is the *strong* form of the H24 negative —
"even with `R` made hard and capacity made binding, masking `D` buys nothing" — and it is bankable. What we
will not do is scale model and world together so generously that both arms saturate and the tie is an
artifact of over-provisioning; that would launder a non-result into a null.

> **Result — the frontier was tested and H24 is REFUTED at local scale, with a mechanism**
> ([`en8_capacity`](../../src/verisim/experiments/en8_capacity.py), [SPEC-9 §4 S3](./SPEC-9.md); 40-host
> world × `d_model` ∈ {16,32,64} × observed-fraction ∈ {0.25,0.5,0.75} × 4 seeds). No cell is
> disjoint-positive; where `D` is large (observed-fraction 0.75, `R` ~11% of tokens) masking it is
> disjoint-*negative* and worse at higher capacity (`d_model=64`: −0.094 [−0.130, −0.057]). The reasoning
> above had it backwards: masking `D` does not *free* capacity for `R`, it *removes training signal* —
> the model is then supervised on only the R-fraction of tokens per step, starving the shared
> encoder/decoder, and learning `D` turns out to be **beneficial multi-task auxiliary signal** for the
> representation `R` also uses. Capacity was never the binding constraint at this scale; supervision was.
> **What this refutes is precisely the *training-objective* form of the partition (§4.2 / DD-OG-2 — mask
> `D` in the loss). The *inference-time* partition (DD-OG-3 — the oracle *supplies* `D`, so the model is
> never trusted on it) is untouched and stands**, and is what the propose–verify–correct loop already
> does. The bankable next variant: keep `D` in the loss (it helps) and let the oracle own `D` only at
> inference. This is the epistemic engine working as designed — a pre-registered negative returning a
> sharp, mechanistic, *trustworthy* result that redirects the program.

### 7.3 Local-first staging and the hardware envelope (the OG discipline, applied to scale)

The repo's invariant — *the deterministic, no-GPU machinery ships and is property-tested before any training
claim* (OG1/OG2 before OG3/OG4) — applies unchanged to scale. The scale-up is therefore two gated stages:

- **OG5 (local, CPU, Mac):** build the scale *harness* — configurable world size, configurable model size,
  multi-seed bootstrap CIs, and the `en8_scale`/`en9_scale` curve runners — and **prove the happy path
  locally**: determinism from seeds, the three gaps computed correctly, the effect direction already visible
  at small multi-seed scale. Gated like any OG milestone; nothing scales until this is green.
- **OG6 (the scaled run):** run the *same* harness at the moderate target (≤ 50 hosts, `d_model ≤ 256`, many
  seeds, longer training) and commit the scaling-curve figures with their CIs. No new code path — only larger
  config values — so there is nothing new to debug under a meter.

**The runtime reality sets the hardware envelope, and it is smaller than it looks.** These models are tiny:
a full JEPA/contrastive training run is **sub-second to a few seconds on this Mac's CPU**, so the entire
*moderate* sweep (≈ 5 world sizes × 3 model sizes × 8 seeds × 3 experiments, with longer training) is an hour
or two **locally** — below ~50 hosts the round-trip to a rented box costs more than it saves. A GPU earns its
keep only when world size gets large, because message passing is the dense `a_link[B, N, N] @ W` product —
**O(N²·d) per round** — so at `N ≈ 100–500` hosts and `d_model ≥ 256` the CPU slows and the GPU wins. The
workload is **many small runs (embarrassingly parallel), not one large model**, so the right rented box is a
single **24 GB GPU (RTX 4090 / A10G) with many vCPUs** — *not* an A100/H100 (no single model uses that VRAM;
you would pay 3–5× for idle silicon). At the chosen moderate scale the expected spend is a few dollars of
GPU time, and possibly none — the Mac may carry it. The harness exposes a `--device {cpu,mps,cuda}` flag so
the *identical* code runs locally and on the rented GPU (seed-level, not bit-level, reproducibility at that
tier). Measured on a 32 GB M4, **CPU is the right local default**: at this model/batch size MPS ran 2–3×
*slower* (Apple's per-kernel launch latency is not amortized by a ~1 M-param model), and CPU is also
bit-deterministic ([SPEC-9 §3](./SPEC-9.md)).

The *full* local envelope — how large the world can be made on one machine before the `O(N^2)` message
passing (not memory, and not the free oracle) binds — and the **model-size axis** the H23-S/H24-S/H25-S
sweeps extend along (the scaling surface, claims S1–S3) are specified in [SPEC-9](./SPEC-9.md), the
free-oracle scaling-regime spec. The moderate-scale answer to "how big, locally" is itself a measured
SPEC-9 result (sweep preset N ≤ 200 hosts, hero preset N ~400–512), not an assumption.

---

## 8. How it slots into the repo (no fork)

This method adds *objectives and data generators*, not architecture. Concretely, against what already
ships:

- **Reuses unchanged:** the reference oracle and `apply == oracle` invariant; the `Model` protocol
  (proposer stays pluggable); divergence `d(s, ŝ)` and **bits-to-correct** (already the metric — §4.2
  only promotes it to a *loss*); the partial-observation `netloop`; the frozen-eval gate (SPEC-4 §5).
- **Adds (network world first):** (i) an *oracle-anchored target* option in the NW8 latent arm's training
  step (§4.1); (ii) a *bits-to-correct training objective* alongside the supervised cross-entropy (§4.2),
  with a `D`-mask derived from the oracle; (iii) an *oracle hard-negative / counterfactual sampler* in
  `netdata` (the K1 hard-negative generator, SPEC-2.1 §5, generalized to contrastive pairs and
  action-branch counterfactuals, §4.3).
- **Touches no judge:** none of the above may edit the oracle, metric, goldens, eval-set generator, or
  gate (DD-AR2). The oracle generates *training data and targets*; it never scores itself.

The autonomous research engine (SPEC-4) is the natural driver: these are three new *objective* knobs in
the search space, and the AZR lesson (§2.4) is precisely that a verifier can search over its own
curriculum. SPEC-4 §10 is extended to note that the search ranges over *training objectives across cake
layers*, not only RL hyperparameters.

---

## 9. Milestones (OG0–OG4)

Non-colliding with `M0–M8 / S1–S6 / AR0–AR5 / NW0–NW8 / HC0–HC8 / DS0–DS8`. Gated like every verisim
milestone: the deterministic data/target machinery is built and tested before any training claim, and no
stage graduates without a committed figure or an honest negative.

| Milestone | What | Verify | Status |
|---|---|---|---|
| **OG0** | The framing as a committed artifact: this spec + the SPEC.md §8 / §9 anchors (H23–H25) + the SPEC-5 §12 EN8/EN9 entries. No code. | the specs cross-reference cleanly; numbering does not collide (H≤25, EN≤9, OG-series new) | ✅ |
| **OG1** | Oracle target + `D`-mask machinery in `netdata`/`netmodel`: emit the true next-state target, the exact divergence target, and the decidable-bit mask for any `(s, a)`. Dependency-free, no GPU. ([`netdata/grounding.py`](../../src/verisim/netdata/grounding.py)) | property test: the mask exactly partitions `D` ∪ `R = s'`; the divergence target equals `netmetrics` `d` by construction | ✅ ([`test_grounding.py`](../../tests/test_grounding.py), 5 cases) |
| **OG2** | Oracle hard-negative & counterfactual sampler: one-edit-wrong successors and action-branch counterfactuals, each labeled against the oracle. ([`netdata/negatives.py`](../../src/verisim/netdata/negatives.py)) | property test: every emitted negative is `≠ O(s,a)` and every counterfactual equals `O(s, a')`; coverage spans the action grammar | ✅ ([`test_negatives.py`](../../tests/test_negatives.py), 5 cases) |
| **OG3** | **EN8** runs (objective × collapse-machinery ablation) on the NW8 arm; committed figure. ([`experiments/en8.py`](../../src/verisim/experiments/en8.py), [`netmodel/grounded_train.py`](../../src/verisim/netmodel/grounded_train.py)) | the `H23`/`H24` cells are populated; regenerates from config+seed with `maxΔ=0` | ◐ smoke shipped ([`test_en8.py`](../../tests/test_en8.py), [`test_grounded_train.py`](../../tests/test_grounded_train.py)): H23 positive, H24 near-tie; CIs/scale-up remain |
| **OG4** | **EN9** runs (oracle-contrastive); committed figure incl. interventional fidelity. ([`experiments/en9.py`](../../src/verisim/experiments/en9.py), [`netmodel/grounded_train.py`](../../src/verisim/netmodel/grounded_train.py) `train_contrastive`) | the `H25`/`H5` cells populated; honest negative reported if the proxy is not beaten | ◐ smoke shipped ([`test_en9.py`](../../tests/test_en9.py)): H25 confirmed, H5 lift ~2× over VICReg; CIs/scale-up remain |
| **OG5** | The **scale harness** (§7.1, §7.3): configurable world size (`scaled_net_config`) + model size (`n_layer`/`n_head` exposed through `build_graph_model`) threaded through EN8/EN9; multi-seed **bootstrap CIs** (reuse [`metrics/aggregate.bootstrap_ci`](../../src/verisim/metrics/aggregate.py)); the curve runners ([`experiments/en8_scale.py`](../../src/verisim/experiments/en8_scale.py), [`experiments/en9_scale.py`](../../src/verisim/experiments/en9_scale.py)) with a `--device {cpu,mps,cuda}` flag. Built and proven **locally on CPU** before any scaled claim. | property tests: the harness is deterministic from seeds; the three gaps (H23-S/H24-S/H25-S, §7.1) + CIs are emitted; the smoke effect direction reproduces at small multi-seed scale | ✅ shipped ([`test_scale_common.py`](../../tests/test_scale_common.py), [`test_en8_scale.py`](../../tests/test_en8_scale.py), [`test_en9_scale.py`](../../tests/test_en9_scale.py)); the local SVD-on-CPU + chunked-eval enablers ([SPEC-9 §3](./SPEC-9.md)) ship with it |
| **OG6** | The **scaled runs**: the *same* harness at the moderate target (≤ 50 hosts, `d_model ≤ 256`, many seeds, longer training, §7.3) → the committed **scaling-curve figures with disjoint CIs** — the "cannot be dismissed" deliverable. | H23-S/H24-S/H25-S verdicts populated *with bootstrap CIs across world/model size*; regenerates from config + seed; CPU-or-GPU identical code path | ◐ first datum shipped (5/10/15 hosts × 4 seeds, [`en8_scale.csv`](../../figures/en8_scale.csv) / [`en9_scale.csv`](../../figures/en9_scale.csv)): **H23-S confirmed** (collapse gap disjoint at every world size), **H25-S/H5 confirmed** (interventional lift disjoint-positive everywhere), **H24-S refuted with a mechanism** ([`en8_capacity`](../../src/verisim/experiments/en8_capacity.py): masking `D` removes beneficial training signal, SPEC-9 S3); the larger world × model surface (SPEC-9 LS2) is the remainder |

**OG0–OG5 have shipped; OG6's first scaled datum is in (the larger surface remains).** OG0–OG2 are the framing + deterministic, property-tested, no-GPU data factory;
**OG3 (EN8)** is the GPU consumer that ablates the *objective × collapse* axes on the NW8 graph+RSSM arm,
and **OG4 (EN9)** the one that ablates the *contrastive* axis (consuming the OG2 hard-negative factory).
Both land *split, honest* smoke verdicts ([report](../report.md)).

**OG3 (EN8):** **H23 confirmed** — with the collapse-prevention machinery ablated, the
naked learned (EMA) target collapses (embedding std 0.557 → 0.276, effective rank 41.8 → 13.4) while the
**oracle-anchored** target holds (std 0.528, rank 25.8 ≈ 2× the collapsed arm), so the external referent
substitutes for EMA+VICReg, exactly as the "collapse tax is a workaround for a missing oracle" claim
predicts; **H24 a near-tie** at this scale (residual-token accuracy 0.426 vs 0.463 raw-likelihood — the
baseline edges it), the honest-negative branch — the decidable part `D` was cheap enough that masking it buys
nothing *yet*, a bound pre-registered to grow with world size (SPEC-6/7).

**OG4 (EN9):** **H25 confirmed, with an honest nuance.** The naked contrastive target collapses (std 0.276);
both the VICReg regularizer (std 0.499) and the **oracle hard-negatives** (std 0.699) prevent it, so on the
collapse axis the exact referent *matches* the statistical stand-in — and VICReg's covariance term even buys
slightly higher effective rank (39.0 vs 31.4). But the **H5 lift is decisive**: only the oracle's
*counterfactual* negatives carry interventional content, so its branch-retrieval fidelity nearly doubles
VICReg's (top-1 **0.519 vs 0.282**, MRR 0.694 vs 0.500) — VICReg keeps the representation full-rank but
interventionally *blind*, while the oracle makes it faithful to the branches the model will be asked to
predict. The honest nuance (VICReg wins on rank, loses on intervention) is the sharper finding: it localizes
*what* the exact negatives buy that a statistical regularizer structurally cannot.

**OG6 (the first scaled datum — the smoke verdicts, now with CIs across a 3× world sweep).** The OG5
harness ran EN8/EN9 at 5/10/15 hosts × 4 seeds ([`en8_scale.csv`](../../figures/en8_scale.csv),
[`en9_scale.csv`](../../figures/en9_scale.csv)):

- **H23-S confirmed.** The collapse gap (oracle-anchored − learned, machinery ablated) is disjoint from
  zero at every world size: +13.4 [12.7, 14.0], +8.4 [7.8, 9.0], +7.7 [6.7, 8.7] effective-rank points,
  with the `emb_std` gap also disjoint everywhere. Honest nuance: the raw rank gap *declines* with world
  size — expected, since effective rank is capped by `d_model=48`, which is exactly why SPEC-9's S1 tracks
  the *normalized* gap and grows `d_model` with the world.
- **H25-S / H5 confirmed.** The interventional lift (oracle − VICReg) is disjoint-positive at every world
  size: top-1 +0.100 [0.059, 0.140], +0.354 [0.266, 0.448], +0.094 [0.055, 0.125]. Honest nuance: it is
  *non-monotone* (peaks at 10 hosts), which SPEC-9's S2 pre-registers as a fixed-capacity undertraining
  artifact to test against the model-size axis.
- **H24-S a CI-bounded near-tie, then refuted with a mechanism.** The residual-objective gap straddles
  zero at all three world sizes (+0.069 [−0.005, 0.130], 0.000 [−0.035, 0.035], +0.006 [−0.009, 0.028]) —
  the smoke near-tie, now with error bars. The dedicated capacity-binding frontier sweep (SPEC-9 S3,
  [`en8_capacity`](../../src/verisim/experiments/en8_capacity.py)) then **refuted** H24's training-objective
  form: no cell is disjoint-positive and masking `D` is disjoint-*negative* where `D` is large, because it
  *removes beneficial training signal* rather than freeing capacity (§7.2 Result). The inference-time
  partition (the oracle supplies `D`) stands; only "mask `D` in the loss" is refuted.

So the two representation-mechanism claims (H23, H25/H5) are now defensible against the "single-seed /
toy-world" dismissal, and H24 remains an honestly-bounded negative. The larger world × model **scaling
surface** (SPEC-9 LS2) extends this up the local envelope. The delta-exact per-step metric the ablations
report on ([`netmetrics/exact.py`](../../src/verisim/netmetrics/exact.py)) shipped as an EN4 column.

---

## 10. Safety, ethics, and the honest-negative posture

Inherited wholesale from SPEC.md §13, SPEC-4 §9, and SPEC-5 §15: defensive-only framing, no real-internet
egress, MIT, no telemetry, the oracle/metric/goldens/gate in the denylist (DD-AR2). One method-specific
note: oracle-grounded objectives are a *stronger* anti-reward-hacking story than RLVR, not a weaker one —
because bits-to-correct is `0` iff the prediction equals the oracle's truth (SPEC-4 §5.2), a degenerate
objective cannot score well without actually being faithful. The dual-use posture is unchanged: this is a
training method for *faithful specialist simulators*, not for any capability the world specs do not
already scope.

The honest-negative posture is load-bearing here specifically. The strong, clean outcome (the collapse
tax falls away, residual supervision wins) would be a real contribution to self-supervised world-model
learning. But the *negative* — "even with a free exact oracle in the bulk, the proxy machinery still
earns its keep" — is equally publishable and arguably more interesting, because it would say something
non-obvious about *why* JEPA needs its crutches (a cause the external referent does not reach). EN8/EN9
are designed so either outcome is a result. The plate is real either way; the open question this spec
opens is only *how much* pressing truth into the bulk actually buys.

Made fully explicit as the pre-registered outcome→implication map (the epistemic engine, SPEC.md §10.1) —
every cell is a forward move, and because the verdict is oracle-grounded, every negative is *bankable*:

| Hypothesis | If confirmed → the contribution | If refuted → the (often deeper) contribution |
|---|---|---|
| **H23** (oracle-anchored target removes the collapse tax) | a *constructive* result for SSL: where an external referent exists, EMA+VICReg are unnecessary → simpler, more stable world-model pretraining | the **more interesting** branch: the representation collapses *even with* the oracle-anchored target → collapse has a cause the referent does not reach → a non-obvious, *bankable* fact about *why* JEPA needs its crutches that the oracle-free field structurally cannot establish |
| **H24** (bits-to-correct residual beats raw-likelihood) | the partition (§3) is load-bearing → "offload the decidable bits, learn only the residual" is a real training principle, and the objective now matches the inference-time metric | the decidable part `D` was already cheap for the model to learn → masking it buys nothing → a clean bound on *when* the partition matters (it will matter more as worlds grow, SPEC-6/7) |
| **H25** (oracle hard-negatives are an exact anti-collapse referent) | exact near-miss/counterfactual negatives match or beat statistical regularizers *and* lift interventional fidelity (the H5 lift) → a second, independent route to grounded SSL | near-miss structure was not the collapse mechanism → narrows precisely *what* anti-collapse fixes → a map of the failure surface contrastive SSL has lacked |
| **H23-S** (the collapse gap holds with CIs across scale) | the collapse gap stays > 0 with **disjoint** bootstrap CIs, stable or growing across world/model size → the smoke result was not an artifact of size; the SSL contribution is real and robust | the gap shrinks or its CIs overlap once seeds and scale are honest → the OG3 smoke "win" was a small-sample mirage → a *bankable* correction that the field needs and only the oracle can certify |
| **H24-S** (residual supervision wins once `R` is hard and capacity binds, §7.2) | the residual gap crosses 0 into positive as the world grows at fixed capacity → the partition (§3) is load-bearing *exactly where the theory says* — when offloading `D` frees a binding budget for a hard `R` | even with `R` made hard and capacity made binding, masking `D` buys nothing (CI-bounded) → the **strong** form of the H24 negative: a sharp, scale-resolved bound on when the partition matters. **[Result: this branch — REFUTED with a mechanism (§7.2, SPEC-9 S3): masking `D` *removes beneficial training signal*, not wasted capacity, so the training-objective partition does not pay; the inference-time partition stands.]** |
| **H25-S** (the interventional lift widens with branch count) | `top1(oracle) − top1(vicreg)` stays > 0 with disjoint CIs and *widens* as more hosts create more distinct branches (chance `1/m` falls) → exact counterfactuals are the scalable route to interventional fidelity | the lift narrows at scale → VICReg's statistical blindness is a small-world artifact → narrows *where* exact negatives are worth their cost |

The pattern across all three is the project's stance in one frame: **the refutation branch is never empty,
and is frequently the branch worth more.** We can ask "what is the collapse-prevention machinery a
workaround for?" and get a *trustworthy* answer — confirm or refute — because we hold the one thing the
self-supervised world-model field lacks: a free, exact, generative oracle. That is why SPEC-8 is worth
building before we know how the figure falls, and why it is worth building with full energy regardless of
which way it falls.

---

## 11. Provenance and reading order

- **Read before:** [SPEC.md](./SPEC.md) (the science — *why* the oracle exists), then
  [SPEC-5](./SPEC-5.md) (the world this is first built in). This spec assumes both.
- **Sibling, not a world:** like [SPEC-4](./SPEC-4.md), this is a *cross-cutting method* every world
  inherits, not a vertical. SPEC-2/5/6/7 are worlds; SPEC-4 is the engine; SPEC-8 is where the oracle's
  truth enters training.
- **Hypotheses:** H23–H25 (§6) extend SPEC.md §9; they are operationalized as EN8/EN9 (§7) in SPEC-5 §12.
- **Status:** OG0–OG2 shipped (the deterministic factory); **OG3 (EN8) and OG4 (EN9) shipped as committed
  smoke figures** ([`figures/en8_grounding.png`](../../figures/en8_grounding.png),
  [`figures/en9_contrastive.png`](../../figures/en9_contrastive.png)) — H23 confirmed / H24 a near-tie (EN8),
  H25 confirmed with the H5 lift ~2× over VICReg (EN9) (§9). CIs and scaled runs are the remaining work;
  nothing beyond the committed figures is believed until it is run.
- **Citations** are name + venue + year (and the arXiv id where verified: AZR 2505.03335; GrndCtrl
  2512.01952; WAV 2604.01985), with no fabricated links, per the repo convention.
- **Author:** Clay Good. **License:** MIT. No telemetry, no commercial path — a research repo.
