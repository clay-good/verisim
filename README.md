# Verisim

> **The claim.** Every other domain trains world models against *proxies* for truth — a fixed
> corpus, a human annotator, a hackable reward. **Computer environments are the one exception:
> filesystems, processes, networks, and APIs are digital, deterministic, and fully checkable, so a
> deterministic *oracle* can return the exact next state for free, at every step.** Verisim is the
> research program built around that single asymmetry — putting an oracle *in the loop* to bound a
> neural world model's drift, and measuring the tradeoff nobody else can measure: **how much oracle
> consultation buys how much faithful horizon.** The world model is a pluggable proposer (transformer,
> JEPA/RSSM, or a frozen LLM), so the real bet is a *method*, not a model: **deterministic verification
> as a model-agnostic primitive for probabilistic ML** — the layer underneath the world-model race, not
> another entrant in it.

## Results at a glance

Six committed, oracle-grounded figures — the smoke-scale bet in one screen — each detailed below, each
with its honest negative. **What survives scaling is the real verdict: see [§8](#8-which-wins-survive-scaling--the-honest-mixed-verdict-spec-9).** Every number regenerates from
config + seeds (`bash figures/reproduce.sh`).

| | |
|---|---|
| [![EN4 graph-vs-flat](figures/en4_graph_vs_flat.png)](figures/en4_graph_vs_flat.png)<br>**Structure helps (EN4 / H11).** The message-passing graph+RSSM world model beats the flat serializer by **+16.5 pts** one-step token accuracy and **+30.6 pts** delta-exact rate — and the gap *widens* on the honest metric. | [![EN8 oracle-grounded SSL](figures/en8_grounding.png)](figures/en8_grounding.png)<br>**The oracle removes the collapse tax (EN8 / H23).** Ablate JEPA's EMA+VICReg crutches and a learned target collapses (eff-rank 41.8→13.4); an **oracle-anchored** target stays healthy (25.8) — the external referent the field lacks. *Scaled to 200 hosts (SPEC-9 S1): the gap stays disjoint-positive everywhere but **attenuates** — real at every scale, diminishing.* |
| [![EN9 oracle hard-negatives](figures/en9_contrastive.png)](figures/en9_contrastive.png)<br>**Exact negatives beat statistical ones where it counts (EN9 / H25 / H5).** VICReg and the oracle both stop contrastive collapse — but only the oracle's *counterfactual* negatives teach which intervention leads where: branch-retrieval top-1 **0.519 vs VICReg's 0.282**. The statistical regularizer is full-rank but interventionally blind. *⚠ Scaled (SPEC-9 S2): this lift **reverses** by 100–200 hosts at higher capacity with a fixed `k=8` (VICReg overtakes) — but the reversal is a **negative-count artifact that recovers**: scaling `k_negatives` 8→32 flips it back to disjoint-positive. Real lift; feed it negatives that scale with the world.* | [![EN1 / K4 the floor](figures/en1_curve.png)](figures/en1_curve.png)<br>**A floor, not a knee (EN1 / K4 / H8).** Faithful horizon vs consultation budget `H_ε(ρ)` is flat-then-cliff on *both* worlds: consultation budget alone does not buy horizon. The honest negative that drove every later design choice. |
| [![EN3 probe efficiency](figures/en3_operators.png)](figures/en3_operators.png)<br>**What you consult beats whether (EN3).** Under partial observation a cheap one-host **probe + belief filter** earns **~2.3×** more faithful horizon per oracle-bit than full consultation — active sensing is a real lever. | [![EN7 model-invariance](figures/en7_invariance.png)](figures/en7_invariance.png)<br>**The shape is the loop's, not the model's (EN7 / H22).** Run the *same* loop with four proposers (null, flat transformer, graph+RSSM, oracle-backed): the three imperfect ones share **one shape — floor + cliff, no knee.** The proposer sets the floor *height* (graph > flat > null); the loop sets the *shape*. The no-knee verdict is model-agnostic. |

## What we've found so far

Every number below is bit-exact and oracle-grounded, regenerates from config + seeds, and is reported
with its honest negative (the v0 norm — [SPEC §9–10](docs/specs/SPEC.md)). The interesting results,
front-loaded:

### 1. Structure helps — and helps *more* on the honest metric (network EN4 / H11)

The message-passing **graph + RSSM** world model beats the **flat** serializer on never-trained eval
seeds by **+16.5 pts** of one-step token accuracy *and* **+30.6 pts** of *delta-exact* rate (did the
model freely decode the **exact** true edit set this step?). The gap *widens* on the stricter metric —
token accuracy understates how much the graph inductive bias buys.

![EN4 graph-vs-flat (H11): token accuracy, delta-exact, and faithful horizon across arms](figures/en4_graph_vs_flat.png)

| arm | one-step token acc | **delta-exact** rate | `H_ε`(ρ=0), ε∈{0,.05,.1} |
|---|---|---|---|
| flat-Markov (NW4) | 0.673 | 0.264 | 0 / 0 / 0 |
| **graph + RSSM (NW8)** | **0.838** | **0.569** | 0 / 0 / 0 |
| graph + RSSM + noise lever (§6.3) | 0.828 | 0.556 | 0 / 0 / 0 |
| graph + RSSM + self-forcing lever (§6.3) | 0.803 | 0.500 | 0 / 0 / 0 |

### 2. …but better one-step prediction does **not** yet buy free-running horizon (the honest negative)

The right-hand column above is the catch: `H_ε(ρ=0) = 0` for **every** arm even at ε=0.1 — each drifts
on the first unaided step. The delta-exact number *quantifies* why: at 0.569 per-step exactness,
whole-delta correctness decays geometrically over unaided steps and first-exceedance is discrete (one
wrong edit spikes the graph divergence past ε in a single step). **Both pre-registered exposure-bias
levers — noise-injection and self-forcing / scheduled sampling — have now run, and both land the *same*
banked negative** (a small one-step dip, no horizon yet). That the random-corruption lever and the
model's-own-drift lever behave identically is itself informative: the wall is **localized to the
one-step→horizon conversion**, not the per-step learner (the arm fits teacher-forced to >0.9). This
routes the remaining budget to scale, the latent-overshooting objective, and *objective grounding*
([SPEC-8](docs/specs/SPEC-8.md)), not to more input-distribution patches.

### 3. Cheap, localized sensing is the efficient way to buy horizon (network EN3)

Under partial observation the oracle has two modes — **full** (the whole next state) and **probe** (one
host's local view). v0's correction operators were indistinguishable on `H_ε`; in the network world a
cheap one-host **probe + belief filter breaks that collapse and earns ~2.3× more faithful horizon per
oracle-bit** than full consultation. *What* you consult, not just *whether*, is a real lever.

### 4. The consultation curve has a floor, not a favorable knee — on *both* worlds (the result that drove the design)

The prime directive is to plot `H_ε(ρ)` — faithful horizon vs. consultation budget — and ask whether a
little consultation buys a lot of horizon. On the single-filesystem world (left) and the network world
(right), the answer so far is a **floor + cliff, not a knee**:

| filesystem (SPEC-2.1 K4) | network (SPEC-5 EN1) |
|---|---|
| ![K4 knee](figures/k4_knee.png) | ![EN1 curve](figures/en1_curve.png) |

This is the **C-knee / H1–H8 honest negative**: discrete per-step errors make first-exceedance `H_ε`
reset-resistant, so consultation budget alone does not lift it. Far from a dead end, this negative is
*precisely* what licensed the network world (gradual drift, partial observability, a calibrated belief
signal) and what makes the NW8 graph arm + drift levers + SPEC-8 objective-grounding load-bearing
rather than speculative. Negatives here are **trustworthy** because the verdict is oracle-grounded, and
a refutation is frequently the deeper result.

### 5. The oracle generates its own training data — perfectly labeled, for free (SPEC-8 OG1/OG2, shipped)

Because the oracle is total, it is a *data factory*, not just a checker. The deterministic machinery is
built and property-tested ahead of the GPU runs: **oracle targets + the decidable/residual partition**
(mask the bits the oracle fixes, learn only the genuine residual), and a **hard-negative / counterfactual
factory** (one-edit-wrong successors and action-branch counterfactuals, each exactly labeled). This is
"oracle-grounded self-supervision" — pressing ground truth into the *bulk* of the cake, not just the RL
cherry.

### 6. …and putting it in the bulk **removes JEPA's collapse tax** (network EN8 / H23, shipped)

The first result from spending that data factory: a JEPA-style latent predictor trained with an
**oracle-anchored target** — a fixed projection of the *true next state*, an external referent with full
variance by construction — keeps its representation healthy *with the EMA-target + VICReg
collapse-prevention machinery ablated*, where the standard learned (EMA) target collapses.

![EN8: oracle-grounded SSL — the collapse axis (H23) and the objective axis (H24)](figures/en8_grounding.png)

| JEPA target | collapse machinery | embedding std | effective rank (d=48) |
|---|---|---|---|
| learned (EMA) | **on** (the baseline) | 0.557 | 41.8 |
| learned | **off** (ablated) | 0.276 | **13.4** (collapsed) |
| **oracle-anchored** | **off** (ablated) | **0.528** | **25.8** (healthy) |

Ablating the machinery collapses the learned target (effective rank 41.8 → 13.4); the oracle-anchored
target holds at 25.8 — roughly twice as healthy, no crutches. This is the strongest possible form of
SPEC-8's thesis: **the collapse-prevention machinery is a workaround for a missing oracle**, and where the
oracle exists the workaround is largely unnecessary — a fact the oracle-free field structurally cannot
establish. The companion **objective axis (H24)** — train only the genuinely-uncertain *residual* bits,
let the oracle supply the decidable ones — is an honest near-tie at this smoke scale (residual-token
accuracy 0.426 vs 0.463 raw-likelihood), the pre-registered negative branch: the decidable part is cheap to
learn until the worlds grow. A split EN8 verdict, like EN4 — and every cell is bankable under the oracle.

### 7. Exact negatives don't just stop collapse — they teach *interventions* (network EN9 / H25 / H5, shipped)

EN8 grounded the *predictive* target on the oracle; **EN9** grounds the *contrastive* one and spends the
OG2 hard-negative factory. A contrastive predictor over the same graph summary, with the only anti-collapse
referent varying across three cells: *none* (naked BYOL), *vicreg* (the field's statistical regularizer), or
*oracle* (InfoNCE against exact one-edit-wrong and counterfactual negatives). Two readouts — representation
health, and **interventional fidelity**: can the representation map each intervention `a'` to its true
successor `O(s, a')` (scored as branch-retrieval top-1 / MRR)?

![EN9: oracle hard-negative contrastive — the collapse axis (H25) and the interventional axis (H5)](figures/en9_contrastive.png)

| anti-collapse referent | embedding std | effective rank (d=48) | intervention top-1 | intervention MRR |
|---|---|---|---|---|
| none (naked BYOL) | 0.276 (collapsed) | 13.4 | 0.214 | 0.426 |
| vicreg (statistical) | 0.499 | **39.0** | 0.282 | 0.500 |
| **oracle (exact)** | **0.699** | 31.4 | **0.519** | **0.694** |

The split is the finding. On **collapse (H25)** the exact referent only *matches* the statistical one — both
hold the representation open, and VICReg's covariance term even buys higher effective rank (39.0 vs 31.4). But
on **intervention (H5)** the oracle wins decisively: its counterfactual negatives nearly double VICReg's
branch-retrieval fidelity (top-1 0.519 vs 0.282). The honest, sharper reading: **VICReg keeps the
representation full-rank but interventionally blind** (0.282 is barely above the naked 0.214), while the
oracle makes it faithful to the very branches the loop will be asked to predict. A statistical regularizer can
prevent collapse; it structurally cannot teach counterfactual structure it has no access to. This is the
H5 / change-safety lift arriving through the *self-supervised* objective rather than the RL cherry — a third
split verdict, every cell bankable under the oracle.

> **⚠ Scaling result (SPEC-9 S2 — the surface run, then the fix).** This lift is **scale-sensitive**, and
> measuring it is the point. On the 25–200-host surface (× `d64`/`d128` × 3 seeds) the oracle's top-1
> advantage is disjoint-positive at the smallest world + smaller capacity (25 hosts/`d64`: +0.106) and
> **reverses** at scale with the fixed `k_negatives=8` — VICReg *overtakes* the oracle at 100 hosts/`d128`
> (−0.086 [−0.113, −0.060]) and 200/`d128` (−0.094 [−0.111, −0.067]). The pre-registered diagnosis — a
> **negative-count artifact** — then proved correct: the LS-S2 sweep
> ([`en9_negatives.png`](figures/en9_negatives.png)) shows that at 100/`d128`, scaling `k_negatives` 8→16→32
> flips `lift_top1` −0.075 → +0.017 → **+0.032 [0.024, 0.044]** (disjoint-positive again). So H5 is
> **confirmed at small scale, reverses at scale with a fixed negative count, and recovers when negatives
> scale** — the magnitude is modest, so the rule is *feed negatives that scale with the world*. The oracle
> let us see a smoke-scale win reverse **and** repair, which is exactly what it is for. (This also refuted my
> own prior that more one-edit negatives wouldn't help — they did, by sharpening the contrastive geometry.)

### 8. Which wins survive scaling — the honest mixed verdict (SPEC-9)

Because the oracle labels for free, world size is a *learner-compute* choice, not a labeling-budget one
([SPEC-9](docs/specs/SPEC-9.md)) — so the smoke-scale EN8/EN9 wins can be carried up an **8× world-size
range on a single 32 GB machine** and stress-tested with bootstrap CIs. The surface (25→200 hosts ×
`d_model` ∈ {64, 128} × seeds) is the most important thing we ran, because it shows the three results
survive *unevenly*, and the unevenness is the finding:

| ![EN8 scaling surface](figures/en8_surface.png) | ![EN9 scaling surface](figures/en9_surface.png) |
|---|---|
| **H23 collapse gap — persists but attenuates (S1).** Disjoint-positive at all 8 cells (the oracle's anti-collapse advantage is real across the whole range and both capacities) but **shrinks** with scale: eff-rank gap 13.4→4.1 over 25→200 hosts at `d128`. Real everywhere, diminishing. | **H25/H5 interventional lift — reverses at fixed `k`, then *recovers* when negatives scale (S2).** Disjoint-positive at 25 hosts/`d64` (+0.106); it **flips negative** at 100/`d128` (−0.086) and 200/`d128` (−0.094) with the fixed `k_negatives=8` — VICReg overtakes. But the reversal is a **negative-count artifact**: scaling `k_negatives` 8→32 at 100/`d128` flips `lift_top1` back to disjoint-positive (+0.032 [0.024, 0.044], [`en9_negatives.png`](figures/en9_negatives.png)). The H5 lift is real; it must be fed negatives that scale with the world. |

The third axis, **H24 (residual objective)**, is **regime-dependent** ([`en8_capacity.png`](figures/en8_capacity.png)):
masking the oracle-decidable bits `D` in the *loss* helps only in a narrow window (high capacity + moderate
residual + small world) and **hurts where `R` is tiny**, because masking removes *beneficial multi-task
training signal* rather than freeing capacity. What is bounded is the *training-objective* partition; the
*inference-time* partition (the oracle simply supplies `D`, the model is never trusted on it) is untouched.

This is the single most valuable thing the scaling bought, and the full arc is the lesson: a headline
(EN9/H5) that looked clean at smoke scale **reversed** under an honest CI sweep at 100–200 hosts — and then,
when the pre-registered lever was tried, **recovered** (scaling `k_negatives` 8→32 flips the lift back to
disjoint-positive). The deterministic oracle is what let us *see* both the reversal and the fix. A win
caught reversing and then honestly repaired is worth far more than one asserted and never stress-tested.
Each verdict carries its next lever (S1: normalize + grow `d_model`; **S2: scale negatives with the world —
now demonstrated**; S3: keep `D` in the loss, oracle-own it at inference).

![EN9 S2-recovery: scaling k_negatives recovers the H5 lift at 100 hosts/d128](figures/en9_negatives.png)

**The local envelope, measured (32 GB M4, CPU — the cost is the learner, not the labeler):**

| hosts `N` | oracle data build | one training run | peak RAM | binding constraint |
|---|---|---|---|---|
| 50 | 0.06 s | ~25 s | 351 MB | — |
| 100 | 0.13 s | ~50 s | 492 MB | — |
| 200 | 0.46 s | ~140 s | 779 MB | wall-clock `O(N²)` message passing |
| 400 | 1.76 s | ~8 min | 1.95 GB | wall-clock (memory has huge headroom) |

The oracle (the labeler) is effectively free at every size; what binds is the learner's `O(N²)` message
passing, not memory and not labels. MPS was *slower* than CPU at this model size (kernel-launch overhead),
so CPU is the bit-deterministic default. Sweep preset `N ≤ 200`; hero preset `N ~400–512`.

### 9. The no-knee shape is the loop's, not the model's (network EN7 / H22)

The project's most general claim is that the *loop*, not the proposer, governs the `H_ε(ρ)` curve — that
deterministic verification is a **model-agnostic primitive**. EN7 tests it by dropping four proposers into
the *same* loop and re-plotting the curve (5 hosts, ε=0.05, T=24, 3 seeds × 2 difficulties, CIs):

![EN7 / H22: the floor+cliff H_ε(ρ) shape is invariant across proposers](figures/en7_invariance.png)

| proposer | ρ=0 | ρ=0.1 | ρ=0.3 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|
| null (empty delta) | 0.0 | 1.2 | 1.2 | 1.3 | 24.0 |
| flat (NW4 transformer) | 0.0 | 1.0 | 1.0 | 1.0 | 24.0 |
| graph (NW8 GNN+RSSM) | 0.0 | 3.2 | 4.3 | 4.7 | 24.0 |
| oracle-backed (perfect) | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |

**H22 supported in kind.** The three imperfect proposers share **one shape — floor + cliff, no knee.** The
proposer's per-step competence sets the *floor height* (graph 3.2–4.7 > flat 1.0 > null), but the loop sets
the *shape* — none shows a favorable knee; all reach the ceiling only at ρ=1. So the EN1/K4 "no-knee"
verdict is **not** an artifact of the flat transformer: it reproduces across materially different
architectures, which is exactly the model-agnostic-primitive claim. The oracle-backed proposer (24
everywhere) is the degenerate ceiling. *Honest caveat:* this is not matched competence (graph is clearly
stronger), so the load-bearing evidence is the *shared shape across differing competence* — what moves with
the proposer is the floor, what stays is the shape.

### 10. Online self-healing (TTT) does not lift the floor — yet (network EN5 / H7, an honest null)

EN7 showed the floor is model-invariant; **EN5** tests the one lever that changes the model *during* the
rollout: when the loop consults the oracle, the revealed `(state, action) → true-delta` is a free labeled
example, so take a small in-rollout gradient step on it (test-time training / self-healing). Does adapting
the weights mid-rollout lift the curve where frozen weights cannot?

![EN5 / H7: online self-healing (TTT) does not lift H_ε(ρ) at this scale](figures/en5_selfheal.png)

| arm | ρ=0 | ρ=0.1 | ρ=0.3 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|
| supervised (frozen) | 0.0 | 3.2 | 4.3 | 4.7 | 24.0 |
| +ttt (self-healing) | 0.0 | 3.2 | 3.5 | 4.7 | 24.0 |

**A null at this scale.** The self-healing arm matches the frozen baseline (marginally *lower* at ρ=0.3 —
single-example updates occasionally perturb without generalizing). This is consistent, not surprising:
EN4 localized the wall to the **one-step→horizon conversion** and EN7 showed the floor is model-invariant,
so a handful of in-rollout steps on single examples can't move the binding per-step competence. It's the
v0 RLVR/H3 null surviving into the network world, and the TTT-stability literature predicted exactly this
for minimal updates. *Pre-registered next lever:* a real self-healing budget — a **replay buffer** of
recent corrections + trust-region revert, not the minimal single-example step. The
[`online_update`](src/verisim/netmodel/graph_train.py) primitive ships for that follow-up.

## The problem, and what we're trying to accomplish

### The wall every world model hits

Generative world models (Genie 3, V-JEPA 2, Cosmos) all hit the same wall: **long-horizon error
accumulation and faithfulness**, with no cheap way to *detect* or *correct* drift, because physical and
visual worlds have **no ground-truth oracle**. You can render a plausible next frame, but you cannot
cheaply ask "is this *exactly* right?" — so error compounds silently and the field spends enormous
effort on proxies that keep a self-referential objective from cheating (JEPA's collapse-prevention
machinery is the clearest instance).

### The one asymmetry computer environments have

| Signal source | Dense? | Exact / true? | Free? | Generative? |
|---|---|---|---|---|
| Self-supervision (corpus co-occurrence) | ✅ | true to the *corpus*, not the world | ✅ | ✅ |
| Human supervision (annotation) | ◐ | usually — but **unscalable** | ❌ | ❌ |
| RL reward / reward model | ❌ (sparse scalar) | a **proxy**, hackable | ◐ | ❌ |
| **A deterministic oracle (computer worlds)** | ✅ | **exact, by construction** | ✅ | ✅ |

No other domain has the last row. A deterministic interpreter of a computer world returns the *entire
true next state* at every step, for free, and can *generate* unbounded perfectly-labeled data and
counterfactuals. Everything in Verisim follows from asking **where to spend that asymmetry**:
inference-time verification, RL reward, and — newest — self-supervised pretraining; and **how much it
actually buys**.

### What we're building: the propose → verify → correct loop

The signature mechanism runs the world model forward and lets the oracle bound its drift under a
consultation budget `ρ`:

```
                 ┌──────────────────────── step t ────────────────────────┐
  state s_t ────▶│  Δ̂ = Mθ.predict_delta(s_t, a_t)      ← neural proposer  │
  action a_t     │       (any model behind the Model protocol)             │
                 │                          │                              │
                 │   consult this step?  ◀── π_c policy, spends budget ρ    │
                 │        │ no                          │ yes               │
                 │        ▼                             ▼                   │
                 │  ŝ_{t+1} = apply(s_t, Δ̂)    O(s_t,a_t)  (oracle: truth)  │
                 │   (free-running prediction)  full | probe → correct ŝ    │
                 └─────────────────────────────────────────┬───────────────┘
                                                            ▼
        divergence d(ŝ_{t+1}, s*_{t+1}) ≤ ε ?   ──▶  faithful horizon
        H_ε(ρ) = first step where d > ε   (how long the model stays bit-exact)
```

- **`apply` is shared by the oracle**, so `apply(s, O(s,a).delta) == O(s,a).state` *by construction* —
  the model and the oracle speak the same delta language (the M1 / NW1 invariant).
- **`ρ`** ranges from 0 (never consult — pure free-running) to 1 (consult every step — always exact).
  The whole research question is the shape of `H_ε(ρ)` between those ends.
- Under partial observation the oracle has **full** and **probe** modes, turning consultation into a
  real bit-budget and opening an active-sensing axis ([SPEC-5 §5.3](docs/specs/SPEC-5.md)).

### The neuro-symbolic split, as a *training* principle

The next-state partitions into two regimes that want opposite treatment — the heart of
[SPEC-8](docs/specs/SPEC-8.md):

```
  s' = O(s, a)
    ├─ D  decidable bits  ── the oracle fixes them exactly & free  ──▶  VERIFY, don't learn  [symbolic]
    └─ R  residual bits   ── genuinely uncertain given what's seen  ──▶  LEARN (the model's job) [neural]
```

Burning network capacity to memorize `D` is waste — the oracle computes it perfectly for free. "Even
nature offloads": evolution does not store chemistry in the genome. SPEC-8 makes this a training
objective (mask `D`, spend gradient on `R`) and ships the deterministic machinery for it (OG1/OG2). *(The
SPEC-9 scaling surface above qualifies this: masking `D` in the **loss** removes beneficial multi-task
signal at small capacity; the partition's load-bearing form is the **inference-time** one — verify `D`,
don't learn-then-mask it.)*

## Architecture & system design

The repo is two parallel **worlds** (filesystem v0, network SPEC-5) over one shared contract — the
propose→verify→correct loop — plus cross-cutting training/packaging. Every box below is dependency-free and
torch-free except `model/`, `netmodel/`, and `train/` (the optional `[model]` extra). The **`Model`
protocol is the seam**: the loop, oracle, metrics, and benchmark never know which proposer they hold, which
is what makes the contribution a *method* rather than a model (the H22 model-invariance claim).

```
                       ACTION a_t
                          │
                          ▼
   ┌────────────────┐  predict_delta   ┌────────────┐   apply(s,Δ̂)   ┌────────────┐
   │  Mθ  proposer  │ ───────────────▶ │  Δ̂  delta  │ ──────────────▶ │  ŝ_{t+1}   │
   │ (Model proto)  │   grammar-       └────────────┘                 └─────┬──────┘
   │ txf | graph+   │   constrained          ▲ same delta grammar           │
   │ RSSM | LLM     │                        │                              │ divergence d(ŝ, s*)
   └────────────────┘                        │                              ▼
   ┌────────────────┐  O(s,a) = (state, Δ*)  │                       ┌──────────────────┐
   │ Oracle (truth) │ ───────────────────────┘   consult on budget ρ │ H_ε(ρ) · bits-to- │
   │ deterministic  │        full | probe ─────────────────────────▶ │ correct · δ-exact │
   └────────────────┘                                                └──────────────────┘
        apply(s, O(s,a).Δ) == O(s,a).state   ← the M1 / NW1 invariant, tested by construction
```

Package map (parallel structure; `net*` mirrors v0 for the graph world):

```
  v0 filesystem (SPEC-2)        network world (SPEC-5)            cross-cutting
  ─────────────────────         ──────────────────────            ────────────────────────────
  env/      state, actions      net/        typed-graph state     train/   supervised + RLVR
  oracle/   O(s,a) truth        netoracle/  Tier-A oracle          eval/    faithfulness benchmark
  delta/    Δ types, apply      netdelta/   graph Δ, apply         rl/      oracle-as-reward env
  metrics/  d, H_ε, bits        netmetrics/ d, reachability,       auto/    autoresearch ratchet
  loop/     runner, π_c, ops                delta-exact, bits      experiments/  E*, EN*, K*,
  model/    Mθ transformer      netmodel/   flat Mθ + graph+RSSM                 en8/9_scale,
  data/     drivers, traj                   + grounded_train (SSL)              en8_capacity,
                                netdata/    drivers + OG1/OG2 factory           en9_negatives
                                netloop/    partial-obs runner, probe, belief filter
```

The deterministic cores (oracle, delta/apply, divergence, the loop, the OG1/OG2 data factory) ship and are
property-tested **before** any training claim — the figure is always gated, never assumed (the NW0–NW3 /
OG1–OG2 discipline). See [SPEC-2 §10](docs/specs/SPEC-2.md) and [SPEC-5 §16](docs/specs/SPEC-5.md) for the
full module-by-module layout.

## Specifications

All specs live under [`docs/specs/`](docs/specs/); the canonical, evidence-gated build order is
[SPEC §12](docs/specs/SPEC.md#12-research-roadmap). The worlds form a ladder (filesystem → network →
host → distributed); three specs are *cross-cutting methods* every world inherits.

| Spec | Role | What it is |
|---|---|---|
| [SPEC.md](docs/specs/SPEC.md) | **the science** | why the project exists, what it claims, how we'd know we were wrong (RQs, H1–H25) |
| [SPEC-2](docs/specs/SPEC-2.md) / [SPEC-2.1](docs/specs/SPEC-2.1.md) | **v0 build** | the shell/filesystem world; the focused effort that earned a competent model and the knee result |
| [SPEC-3](docs/specs/SPEC-3.md) | depth | how the toy grows into a real simulator (system oracle, partial obs, online self-healing, info-theoretic metric) |
| [SPEC-4](docs/specs/SPEC-4.md) | **the engine** | the autonomous research engine — Verisim improving Verisim, human out of the loop |
| [SPEC-5](docs/specs/SPEC-5.md) | **world: network** | the reachability/connectivity world — **the current build front** |
| [SPEC-6](docs/specs/SPEC-6.md) | world: host | the running computer (process tree, memory, scheduler) — design |
| [SPEC-7](docs/specs/SPEC-7.md) | world: distributed | replicated services, transactions, consensus — design |
| [SPEC-8](docs/specs/SPEC-8.md) | **method: oracle-grounded SSL** | put the oracle's truth in the *bulk* of the cake (self-supervised pretraining), not just the cherry (RL) |
| [SPEC-9](docs/specs/SPEC-9.md) | **method: free-oracle scaling** | because the oracle labels for free, world size is a *compute* choice, not a labeling-budget one — how large/deep the world goes on one machine, and what holds as it grows |

Semantics docs ([filesystem](docs/semantics.md), [network](docs/network-semantics.md)) pin the normative
command semantics, paired with the reference oracles, which are the executable truth. The full result
write-up is [docs/report.md](docs/report.md).

## Status

> **Where things stand (2026-06): v0 is done; the network graph arm shipped and split the H11 verdict;
> both §6.3 drift levers, the SPEC-8 data factory, and the SPEC-8 EN8 + EN9 ablations shipped.** Filesystem v0
> (M0–M8) and the focused [SPEC-2.1](docs/specs/SPEC-2.1.md) effort are complete (K0 learner works → K1/K2
> floor ~0 → **0.86** → K3/K4 knee refuted, licensing SPEC-5). The network deterministic core (NW0–NW3),
> flat `M_θ` (NW4), partial-observation loop (NW5), prime-directive EN1 curve (NW6, the H8 negative), and
> EN2/EN3 equal-budget comparisons (NW7, the ~2.3× probe-efficiency result) all ship. **NW8** adds the
> GNN+RSSM graph arm, the EN4 graph-vs-flat comparison (the +16.5/+30.6-pt split verdict), the
> **delta-exact** metric, **both §6.3 exposure-bias levers** (noise-injection + self-forcing), the **SPEC-8
> OG1/OG2** oracle-grounded-SSL data factory, and now both **SPEC-8 EN8 / OG3** and **EN9 / OG4** ablations
> that consume it — two more split verdicts: **H23 confirmed** (the oracle-anchored target removes the
> collapse tax), **H24 a near-tie** (residual masking buys nothing at this scale), **H25 confirmed** (exact
> negatives match VICReg at preventing collapse) with a decisive **H5 lift** (the oracle's counterfactual
> negatives nearly double VICReg's interventional fidelity).
>
> **The [SPEC-9](docs/specs/SPEC-9.md) scaling surface (LS0–LS2) then carried those smoke verdicts up an 8×
> world-size range (25→200 hosts × {d64,d128}) with bootstrap CIs — the honest mixed result of [§8](#8-which-wins-survive-scaling--the-honest-mixed-verdict-spec-9):
> H23 *persists but attenuates*, H24 is *regime-dependent*, and H25/H5 *reverses* at 100–200 hosts with a
> fixed negative count — then **recovers**: the EN9 `k_negatives` S2-recovery diagnostic confirms scaling
> negatives 8→32 flips the lift back to disjoint-positive (the reversal is a negative-count artifact, fixed
> modestly by scaling negatives with the world).** **EN7/H22 model-invariance now ships** ([§9](#9-the-no-knee-shape-is-the-loops-not-the-models-network-en7--h22)):
> the floor+cliff `H_ε(ρ)` shape is the same across null / flat / graph proposers — the loop governs the
> shape, the proposer sets the floor height (H22 supported in kind). **EN5/H7 self-healing also ships**
> ([§10](#10-online-self-healing-ttt-does-not-lift-the-floor--yet-network-en5--h7-an-honest-null)): a
> minimal in-rollout TTT step does *not* lift the floor (an honest null, consistent with EN4/EN7) — the
> pre-registered fix is a real replay-buffer self-healing budget. Next: EN6 (counterfactual two-oracle),
> the SPEC-9 LS3 hero instance (N~400–512), and the replay-buffer EN5 follow-up.

**v0 — shell/filesystem world (`src/verisim/`, SPEC-2 §13): complete.**

| Milestone | What | Status |
|-----------|------|--------|
| **M0–M3** | Env + `ReferenceOracle`, `Delta`/`apply`, drivers/data, divergence + `H_ε` + run-records | ✅ |
| **M4–M5** | Neural `M_θ` (from-scratch transformer, constrained decoder) + propose–verify–correct loop | ✅ |
| **M6–M8** | E1–E4 experiments, smart policies/operators, report, faithfulness benchmark + RL env | ✅ |
| **SPEC-2.1** | K0 (learner works) → K1/K2 (floor ~0 → **0.86**) → K3/K4 (knee refuted on single-FS; licenses SPEC-5) | ✅ |

**Network world (`src/verisim/net*`, SPEC-5 §13): graph arm + EN4 + delta-exact + both §6.3 levers + SPEC-8 factory + EN8/EN9.**

| Milestone | What | Status |
|-----------|------|--------|
| **NW0** | Typed-graph `NetworkState`, action grammar, serialization + **Tier-A reference oracle** + [network semantics](docs/network-semantics.md) + goldens | ✅ |
| **NW1** | Graph `Delta` types, `apply`, serialization; the `apply == oracle` invariant | ✅ |
| **NW2** | Drivers (uniform/weighted/adversarial topology+traffic) + trajectory generation | ✅ |
| **NW3** | Graph divergence, **reachability-faithfulness**, bits-to-correct (`H_ε` + run-records reused from v0) | ✅ |
| **NW4** | Network `M_θ` ([`netmodel/`](src/verisim/netmodel/)): closed vocab, tokenizer, LL(1) graph-delta grammar, constrained decode, supervised training. The **flat** arm (H11 baseline) ships | ◐ flat arm |
| **NW5** | Partial-observation loop ([`netloop/`](src/verisim/netloop/)): two-mode (full / **probe**) oracle, probe policies `π_o`, correction/belief operators, baselines, model-agnostic runner | ✅ |
| **NW6** | **EN1 network `H_ε(ρ)` curve** ([`en1_curve.png`](figures/en1_curve.png)) — the prime directive. Honest H8 negative on the flat arm: near-flat interior | ✅ |
| **NW7** | Equal-budget comparisons. **EN2** (policy `π_c`, H9) + **EN3** (operators, §8.3): EN3 breaks v0's operator-identity collapse — the probe earns **~2.3×** more faithful horizon per oracle-bit | ◐ EN2/EN3 |
| **NW8** | **GNN + RSSM graph arm** ([`graph_model.py`](src/verisim/netmodel/graph_model.py)) + §6.3 **noise + self-forcing** levers + **EN4 graph-vs-flat (H11)** + **delta-exact metric** ([`exact.py`](src/verisim/netmetrics/exact.py)) + **SPEC-8 OG1/OG2 data factory** ([`grounding.py`](src/verisim/netdata/grounding.py), [`negatives.py`](src/verisim/netdata/negatives.py)) + **SPEC-8 EN8/OG3 ablation** ([`en8.py`](src/verisim/experiments/en8.py), [`grounded_train.py`](src/verisim/netmodel/grounded_train.py): H23 collapse-tax removed, H24 near-tie) + **SPEC-8 EN9/OG4 ablation** ([`en9.py`](src/verisim/experiments/en9.py): H25 confirmed, H5 fidelity ~2× over VICReg) + **EN7/H22 model-invariance** ([`en7.py`](src/verisim/experiments/en7.py): the floor+cliff `H_ε(ρ)` shape is invariant across null/flat/graph proposers — H22 supported in kind) + **EN5/H7 self-healing** ([`en5.py`](src/verisim/experiments/en5.py): a minimal online-TTT step is a null at this scale; the `online_update` primitive ships for the replay-buffer follow-up). Then counterfactual (EN6) | ◐ graph arm + EN4 + both levers + OG1/OG2 + EN8/OG3 + EN9/OG4 + EN7/H22 + EN5/H7 |
| **SPEC-9 LS0–LS2** | **Free-oracle scaling** ([`en8_scale.py`](src/verisim/experiments/en8_scale.py), [`en9_scale.py`](src/verisim/experiments/en9_scale.py), [`en8_capacity.py`](src/verisim/experiments/en8_capacity.py), [`scale_common.py`](src/verisim/experiments/scale_common.py)): the measured local envelope + the 8× world-size surface with bootstrap CIs ([§8](#8-which-wins-survive-scaling--the-honest-mixed-verdict-spec-9)). **S1** H23 attenuates, **S2** H25/H5 reverses, **S3** H24 regime-dependent. The [`en9_negatives.py`](src/verisim/experiments/en9_negatives.py) S2-recovery diagnostic **confirms** the lift recovers when negatives scale with the world (k 8→32 flips it back to disjoint-positive) | ✅ LS0–LS2 + S2-recovery + S3 frontier |

The deterministic cores (filesystem and network) have **no runtime dependencies** and need no GPU.
PyTorch is an optional `[model]` extra (see [docs/model-representation.md](docs/model-representation.md)).

## Concepts cheat-sheet

| Term | Meaning | Where |
|---|---|---|
| `O(s, a)` | the **oracle**: deterministic interpreter returning the exact next state + delta | `oracle/`, `netoracle/` |
| `Δ` (delta) | the structured edit set a step makes; `apply(s, Δ)` reconstructs `s'` | `delta/`, `netdelta/` |
| `Mθ` | the **learned proposer** (`predict_delta`); any model behind the `Model` protocol | `model/`, `netmodel/` |
| `d(a, b)` | **divergence**: normalized symmetric set/graph difference, `0` iff identical | `metrics/`, `netmetrics/` |
| `H_ε(ρ)` | **faithful horizon**: first step where `d > ε`, as a function of consultation budget `ρ` | `metrics/horizon.py` |
| `ρ` | **consultation budget** ∈ [0,1]: fraction of steps the oracle is consulted | `loop/policy.py` |
| bits-to-correct | MDL of the oracle's correction of `Δ̂`; `0` iff the prediction is exactly right | `metrics/bits.py` |
| **delta-exact** | per-step: did free decode assemble the exact edit set? (`bits_to_correct = 0`) | `netmetrics/exact.py` |
| full / probe | oracle consultation modes: whole next-state vs one host's local view (cheap) | `netloop/observe.py` |
| `D` / `R` | next-state bits the oracle **decides** vs the genuine **residual** (SPEC-8 partition) | `netdata/grounding.py` |
| oracle-anchored target | a JEPA target pinned to the *true next state* (external referent) instead of a learned EMA | `netmodel/grounded_train.py` |
| collapse readout | embedding std + effective rank — JEPA's collapse diagnostic (→ 0 / → 1 under collapse) | `netmodel/grounded_train.py` |
| noise / self-forcing | §6.3 drift levers: random input corruption vs model's-own-drift rollout, both oracle-relabeled | `netmodel/graph_train.py` |
| reachability-faithfulness | fraction of can-A-reach-service(B) entries that agree | `netmetrics/divergence.py` |

## Design decisions (the load-bearing ones)

| DD | Decision | Why |
|---|---|---|
| **delta prediction** | the model predicts a *structured delta*, not a free-form next state | bounds the hallucination surface; makes `apply == oracle` checkable bit-for-bit |
| **constrained decode** | every prediction is grammar-valid by construction | a model can be wrong but never *malformed*; the parse always succeeds |
| **model-agnostic loop** | the loop never knows which proposer it holds (`Model` protocol) | the contribution is the *method*; H22 asks whether the favorable behavior is the loop's, not a model's |
| **exact headline metric** | reported faithfulness is bit-exact and oracle-grounded; learned signals are *internal* | the oracle calibrates proxies; it is never substituted *for* the truth (DD-3, DD-OG-3) |
| **never latent-ify the checkable part** | latents only ever cover the genuinely-unobserved residual `R` | surrendering verifiability of `D` would give away the whole asset |
| **deterministic core first** | the no-GPU data/metric/loop machinery ships and is property-tested before any training claim | NW0–NW3 / OG1–OG2 discipline; the figure is gated, never assumed |
| **honest negatives are first-class** | every hypothesis pre-registers its refutation branch as a banked result | the oracle makes negatives *trustworthy*; a refutation is often the deeper contribution |

## Verification

The claims above are audited empirically in [docs/verification.md](docs/verification.md): the core
invariants (`apply == oracle`, serialization round-trips, the NW4 tokenizer, metric bounds, exit codes,
in- and cross-process determinism) are proven over **48,000 oracle transitions with zero failures** by
the dependency-free, torch-free [`scripts/verify_invariants.py`](scripts/verify_invariants.py) — and
additionally over the **entire action space** (448,260 state×action pairs) by construction, with
**negative controls** confirming each check detects deliberate corruptions. Every quantitative number in
the report and this README is machine-checked against the committed figure CSVs; the figures regenerate
from config + seeds with `maxΔ = 0`; the NW5 partial-observation loop invariants are tested (ρ=1
full-consult is exact; a one-host probe corrects strictly less than a full consult); and the packaging is
verified end-to-end (the RL-env return equals the faithful horizon, the benchmark separates a perfect
from a trivial model, coverage spans all 13 commands).

## Packaging for reuse

The env + metric are packaged where researchers already look (SPEC-2 §15):

- **Faithfulness benchmark** ([`verisim.eval`](src/verisim/eval/)) — dependency-free; `score_model` /
  `score_suite` grade *any* model implementing the loop `Model` protocol against the oracle's ground
  truth, and `step_labels` + `grade_prediction` expose single-step labels for question-answer frameworks.
  An `inspect_ai` task adapter ships behind the optional `[eval]` extra.
- **Oracle-as-reward RL environment** ([`verisim.rl`](src/verisim/rl/)) — a `verifiers`-spec
  `WorldModelEnv` (with the `load_environment` entrypoint) whose reward is the oracle's faithfulness
  verdict, so the episode return *is* the faithful horizon.

```python
from verisim.eval import score_model, FaithfulnessSample
from verisim.loop import OracleBackedModel
from verisim.oracle import ReferenceOracle

oracle = ReferenceOracle()
score = score_model(OracleBackedModel(oracle), FaithfulnessSample("adversarial", 200, 24), oracle=oracle)
assert score.normalized_horizon == 1.0   # a perfect model is fully faithful, unaided
```

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,model]"   # ".[dev]" alone skips the torch-based M4 tests
pytest            # property tests, semantics goldens, metric/loop/model tests
ruff check .      # lint
mypy              # strict type-check
```

```python
from verisim.env import State, parse_action
from verisim.oracle import ReferenceOracle
from verisim.delta import apply

oracle = ReferenceOracle()
state = State.empty()
for cmd in ["mkdir /a", "write /a/f alpha", "mv /a /b", "cat /b/f"]:
    result = oracle.step(state, parse_action(cmd))
    # apply(state, result.delta) == result.state, by construction (the M1 invariant)
    assert apply(state, result.delta).fs == result.state.fs
    state = result.state
```

Reproduce every figure (E1–E4, calibration, K0/K2/K4, the EN1 curve, EN2/EN3, the EN4 graph-vs-flat
comparison, the EN8 oracle-grounded-SSL ablation, the EN9 oracle-contrastive ablation) from config +
seeds — `figures/reproduce.sh` runs them all:

```bash
bash figures/reproduce.sh
# or the NW8/SPEC-8 smoke figures on their own (each writes CSV + PNG directly):
python -m verisim.experiments.en4_graph --graph-iters 1500 --out figures/en4_graph_vs_flat.csv
python -m verisim.experiments.en8 --out figures/en8_grounding.csv
python -m verisim.experiments.en9 --out figures/en9_contrastive.csv
# the SPEC-9 scaling work (multi-seed, bootstrap CIs; the surface preset is slower):
python -m verisim.experiments.en8_scale --world-sizes 5 10 15 --seeds 0 1 2 3 --out figures/en8_scale.csv
python -m verisim.experiments.en9_scale --world-sizes 5 10 15 --seeds 0 1 2 3 --out figures/en9_scale.csv
python -m verisim.experiments.en8_capacity --out figures/en8_capacity.csv   # the H24/S3 frontier
python -m verisim.experiments.en8_scale --world-sizes 25 50 100 200 --d-models 64 128 --seeds 0 1 2 \
    --out figures/en8_surface.csv   # the §8 surface (likewise en9_scale for en9_surface.png)
```

## Layout

The package map and data flow are in [Architecture & system design](#architecture--system-design) above;
the full module-by-module layout is [SPEC-2 §10](docs/specs/SPEC-2.md) (filesystem) and
[SPEC-5 §16](docs/specs/SPEC-5.md) (network). Everything is under [src/verisim/](src/verisim/). Experiment
configs live in [configs/](configs/); plotting scripts + committed figures (PNG + CSV) in
[figures/](figures/); the run-records they read are git-ignored and regenerable from config + seeds.

## License & posture

MIT (see [LICENSE](./LICENSE)). This is a research repo: **no telemetry, no network calls at runtime, no
commercial path.** The framing and downstream agents are defensive; see
[SPEC.md §13](docs/specs/SPEC.md) for the ethics and dual-use posture.

Author: Clay Good.
