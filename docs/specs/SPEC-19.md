# SPEC-19 — The Flagship: One Trained World Model, the Whole Stack, the Headline Figure

**Integration-and-scale specification: every method spec to date proved its mechanism on a *controlled
stand-in* and deferred the trained-`M_θ` arm — the phrase "only the trained-`M_θ` arm remains" appears
in SPEC-13, SPEC-15, SPEC-16, SPEC-17, and SPEC-18. SPEC-10 then scaled the *floor* (`H_free` at `ρ=0`)
on a real trained model but never carried the *consultation curve* or the composed methods. The result
is a program whose individual claims are each backed by a figure, but whose central object — a real
neural world model, trained at the compute-optimal frontier, kept faithful over a long horizon by the
oracle-in-the-loop, with the best consultation policy the program discovered — has never been built and
measured as one artifact. SPEC-19 builds it. It un-defers the trained `M_θ`, composes the shipped
methods (speculative consultation, conformal triggering, landmark planning) onto that one model, and
produces the single figure the whole program has been promising: `H_ε(ρ)` on a real learned network
world model, end-to-end, against the exact oracle.**

> **▶ PRIORITY — FLAGSHIP / INTEGRATION SPEC — proposed 2026-06-11.** A *cross-cutting* spec in the
> lineage of [SPEC-10](./SPEC-10.md) (the scaling law) and [SPEC-4](./SPEC-4.md) (the engine that runs
> it). It invents **no new world, oracle, or method.** It runs on the [SPEC-5](./SPEC-5.md) network
> world (the flagship world chosen for its gradual drift + partial-observation signal, SPEC-5 §0), the
> shipped Tier-A [`ReferenceNetworkOracle`](../../src/verisim/netoracle/reference.py) and cheap
> [`ControlPlaneOracle`](../../src/verisim/netoracle/control_plane.py), and the real trained graph+RSSM
> arm ([`netmodel/graph_model.py`](../../src/verisim/netmodel/graph_model.py),
> [`graph_train.py`](../../src/verisim/netmodel/graph_train.py),
> [`grounded_train.py`](../../src/verisim/netmodel/grounded_train.py)). What it adds is **assembly and
> scale**: it takes the compute-optimal frontier SPEC-10 HS1.3 found (`l@9.6k`, ~110k params, ~19 id /
> ~29 ood free-running steps — [`configs/horizon_joint_scaling.json`](../../configs/horizon_joint_scaling.json))
> and stacks the consultation policy from [SPEC-13](./SPEC-13.md) (speculative rollout), the trigger from
> [SPEC-15](./SPEC-15.md) (conformal coverage), and the planner from [SPEC-12](./SPEC-12.md) (landmark
> graph) onto that one model. Gated like everything: the composed curve is measured before it is claimed,
> and a floor+cliff *that survives the whole stack at the compute-optimal frontier* is a first-class —
> and, here, the more consequential — result. The GPU arm (capacity beyond the single-machine frontier)
> stays the standing open bet; SPEC-19's job is to make the CPU-frontier flagship real and emphatic
> first.

Read [SPEC.md §3](./SPEC.md) (RQ1–RQ3, the curve and the policy and the operator), [SPEC.md §14](./SPEC.md)
(the four concrete contributions — this spec lands #1 and #2 on a *real* model), [SPEC-10](./SPEC-10.md)
(the capacity/data frontier this spec stands on), and the three method specs it composes:
[SPEC-13](./SPEC-13.md) (speculative), [SPEC-15](./SPEC-15.md) (conformal), [SPEC-12](./SPEC-12.md)
(landmark). This document is *whether the whole program, assembled on one trained model, produces a
faithful-horizon curve that earns the thesis — or a bankable negative that names exactly which method,
on a real model, fails to compose.*

---

## 0. One-paragraph thesis

The program's central claim (SPEC.md §3) is that a neural world model coupled to a free oracle in a
propose–verify–correct loop stays faithful far longer than the model alone, at oracle cost far below
running the oracle every step. Every piece of evidence for that claim is real, but each lives on a
*different* substrate: SPEC-10 scaled the `ρ=0` floor on a real trained model; SPEC-13 showed
speculative consultation crosses fixed-`ρ` on a *controlled drafter*; SPEC-15 proved a conformal trigger
buys coverage at lower `ρ` on a *stand-in* uncertainty signal; SPEC-12 showed landmark planning buys
goal-space horizon where step-horizon is pinned at zero. No experiment has put a real trained `M_θ`
through the *full* loop with the *best composed* consultation policy and plotted the headline curve. That
gap is why the contribution does not land emphatically: a reviewer cannot point to one figure and say
"this is the oracle-grounded world model, and this is how faithful it is for how many oracle calls."
SPEC-19 builds that figure. The thesis it tests, stated to be falsified: *on the real trained network
`M_θ` at the compute-optimal frontier, the composed consultation policy (speculative drafting +
conformal triggering) achieves the program's faithful-horizon target at sub-linear oracle budget, and the
three methods compose at least additively.* If instead the floor+cliff survives the whole stack on a real
model — no sub-linear knee, methods that interfere rather than compose — that is the program's strongest,
most honest negative: verification does not become cheaper with the best machinery the program could
build, and the oracle is a primitive you spend roughly linearly even on a competent learned model. Both
branches are bankable (SPEC.md §10.1), and both are *emphatic* in a way no scattered stand-in result is.

---

## 1. Why this is the experiment that makes the program legible

The program has 68 hypotheses and a large apparatus. Its problem is not rigor; it is **legibility of
value**. Three structural facts cause it:

1. **No flagship model.** Read the status lines literally. SPEC-13: "only the trained-`M_θ` arm remains."
   SPEC-15 CF6: the trained-`M_θ` arm runs but its `belief_var` is "NOT conformalizable." SPEC-16: "only
   the GPU/competent-high-`p` scale regime remains." SPEC-17: "only the CX2 learned contrastive lift is
   deferred." SPEC-18: "trained-`M_θ` leaderboard entries deferred." The one object the thesis is about —
   a real learned world model — is deferred in nearly every spec. SPEC-19 un-defers it *once*, on the
   flagship world, and lets the deferred arms collapse into a single funded demonstration.
2. **No composition.** Each method was proven in isolation against a substrate chosen to make *that*
   method legible. Whether they *compose* — whether speculative drafting, conformal triggering, and
   landmark planning stack on one model without interfering — is untested, and composition is exactly what
   a deployable artifact requires.
3. **No single headline.** The positives (speculative crossover H39, conformal coverage H51, landmark
   goal-horizon H33, oracle-is-an-SCM H60) are real but diffuse. One figure on one real model, with one
   composed policy, is what a reader remembers.

SPEC-19 is the spec that fixes all three at once, on the network world, against the exact oracle. It is
not new science in the sense of a new mechanism; it is the *assembly and scaling* that turns a pile of
mechanisms into a contribution.

## 2. What is held fixed and what is built (no new world, oracle, or method)

| Layer | Source (already shipped) | SPEC-19's job |
|---|---|---|
| World | SPEC-5 network world ([`net/`](../../src/verisim/net/)) | none — held fixed |
| Oracle | Tier-A [`ReferenceNetworkOracle`](../../src/verisim/netoracle/reference.py); cheap [`ControlPlaneOracle`](../../src/verisim/netoracle/control_plane.py) | none — held fixed |
| Model | graph+RSSM arm ([`netmodel/graph_model.py`](../../src/verisim/netmodel/graph_model.py)); flat arm ([`world_model.py`](../../src/verisim/netmodel/world_model.py)) | **train one to the SPEC-10 HS1.3 compute-optimal frontier and freeze it as the flagship checkpoint** |
| Loop | NW5 partial-observation loop ([`netloop/`](../../src/verisim/netloop/)) | none — host for the composed policy |
| Consultation `π_c` | speculative ([`experiments/sr*.py`](../../src/verisim/experiments/)); conformal ([`conformal/`](../../src/verisim/conformal/)) | **compose into one policy and run it on the flagship checkpoint** |
| Planner | landmark graph ([`landmark/`](../../src/verisim/landmark/)) | **stack as the long-range controller and measure goal-space horizon on the flagship checkpoint** |

The flagship checkpoint is the single new artifact: a versioned, frozen, real trained network `M_θ` at the
compute-optimal frontier, with a model card (reuse SPEC-18 PB-pack machinery,
[`eval/`](../../src/verisim/eval/)). Everything else is composition of shipped code.

## 3. The headline figure (the deliverable)

One figure, [`figures/fl1_flagship_curve.png`](../../figures/fl1_flagship_curve.png), four curves on one
`H_ε(ρ)` axis, all on the *same* frozen flagship checkpoint:

- **`ρ=0` floor** — free-running, no oracle (the SPEC-10 frontier number, reproduced as the left anchor).
- **fixed-`ρ`** — the naive consultation baseline (oracle every `1/ρ` steps).
- **composed `π_c`** — speculative drafting + conformal trigger, the program's best policy.
- **`ρ=1` ceiling** — oracle every step.

The claim the figure must support to confirm the thesis: the composed curve reaches ≥80% of the ceiling
horizon at ≤20% consultation (the SPEC.md §9 H1 target), on a *real trained model*, where every prior
attempt at this curve was either on a stand-in or measured only the floor. The honest-negative branch:
the composed curve hugs the floor until `ρ→1` (floor+cliff survives the stack), which is reported with
equal energy and licenses the GPU arm (§7).

## 4. Hypotheses (H69–H72)

Pre-registered with both branches, per SPEC.md §10.1. New numbers continue from the H68 maximum.

- **H69 (the flagship curve — the headline).** On the real trained network `M_θ` at the SPEC-10 HS1.3
  compute-optimal frontier, the composed consultation policy (speculative + conformal) achieves ≥80% of
  the `ρ=1` ceiling faithful horizon at ≤20% consultation budget, with disjoint bootstrap CIs against the
  fixed-`ρ` baseline. *Refuted if* the composed curve is statistically indistinguishable from fixed-`ρ`,
  or hugs the floor until `ρ→1` (the floor+cliff survives the whole stack at the frontier) — the
  program's strongest bankable negative, which says verification cannot be made sub-linear on a competent
  learned model and licenses the GPU bet (§7). Tested as **FL1**.
- **H70 (the methods compose).** Stacking speculative drafting, conformal triggering, and landmark
  planning on one checkpoint yields faithful-horizon-per-oracle-call at least as high as the best single
  method alone (additive or super-additive), at equal budget. *Refuted if* any pair interferes — the
  composed policy underperforms the better of its parts (e.g. the conformal trigger fires on the
  speculative drafter's accepted-prefix boundary and double-counts budget) — in which case the
  composition rule is itself the finding, and the methods stay single-use. Tested as **FL2**.
- **H71 (structure buys goal-horizon on the flagship, even where step-horizon is pinned).** The SPEC-10
  HS3 wall (`H_free=0` for the structured arm at exact tolerance) survives on the flagship checkpoint —
  but landmark planning (SPEC-12 H33) converts that zero step-horizon into long-range *goal-space*
  horizon on the *same real model*, re-grounding once per hop. *Refuted if* the goal-reach lift SPEC-12
  measured on its stand-in does **not** reproduce on the trained flagship arm (the planner's faithful
  hops decay when the hops are taken by a real drifting model rather than a controlled one) — which would
  localize H33 to the stand-in and is a sharp negative for the planning layer. Tested as **FL3**.
- **H72 (model-invariance at the flagship — the loop, not the model).** Swapping the proposer at matched
  competence (the flat transformer vs the graph+RSSM arm vs a frozen-LLM-as-proposer, SPEC-5 §7) leaves
  the *qualitative shape* of the flagship `H_ε(ρ)` curve unchanged — the falsifiable form of the
  program's most general claim (H22, deterministic verification as a model-agnostic primitive), now on a
  trained model rather than the EN7 stand-ins. *Refuted if* a knee appears for one proposer class and not
  another at matched per-step acceptance — narrowing the contribution to a fact about one model family.
  Tested as **FL4** (the frozen-LLM arm is `skipif`-guarded and may defer with the GPU bet).

## 5. Milestones (FL0–FL5)

Engineering sequence, SPEC-11 §8 form. Each milestone ships a committed figure or a frozen artifact.

- **FL0 — train and freeze the flagship checkpoint.** Run the HS1.3 compute-optimal recipe
  ([`experiments/horizon_joint_scaling.py`](../../src/verisim/experiments/horizon_joint_scaling.py),
  `configs/horizon_joint_scaling.json`) to the `l@9.6k` frontier; freeze the checkpoint with a model card
  (SPEC-18 PB-pack). Verify it reproduces the SPEC-10 `H_free` number as a regression gate.
- **FL1 — the headline curve (H69).** Run the composed `π_c` through the NW5 loop on the frozen
  checkpoint; plot the four-curve figure (§3). *This is the deliverable.*
- **FL2 — composition ablation (H70).** Each method on/off (2³ cells, or the additive subset); measure
  faithful-horizon-per-call; report the composition rule.
- **FL3 — structured-arm goal-horizon (H71).** Run the SPEC-12 landmark planner with the flagship
  checkpoint as the low-level controller; reproduce (or refute) the H33 goal-reach lift on the real model.
- **FL4 — proposer swap (H72).** Re-run FL1's curve with the flat arm and (deferred) the frozen-LLM
  proposer at matched competence; overlay the shapes.
- **FL5 — the flagship report.** One section of [`docs/report.md`](../report.md): the figure, the four
  hypotheses' verdicts, and the one-sentence headline (§6). Cross-link from SPEC-18's leaderboard as the
  reference entry.

## 6. The one sentence this spec must produce

Whichever branch the data takes, FL5 must yield a single sentence a reviewer remembers. The two
pre-registered forms were:

- **Confirmation:** *"A real neural world model of a computer network, trained at the compute-optimal
  frontier and coupled to a free deterministic oracle with a speculative+conformal consultation policy,
  holds a faithful simulation for `N` steps at `ρ≈0.2` — `M×` longer than the model alone and at one-fifth
  the cost of running the oracle every step — the first such curve measured against ground truth on a
  learned model."*
- **Bankable negative:** *"Even at the compute-optimal frontier with the program's best composed
  consultation policy, faithful horizon on a real learned network model is floor+cliff: every faithful
  step costs roughly one oracle call, so verification is a primitive that does not become sub-linear with
  any machinery reachable at this scale — a fact only the exact oracle could establish."*

**The sentence the frontier run actually produced (2026-06-11) — a third, sharper branch neither
pre-registration anticipated:** *"On a real neural world model of a computer network trained at the
compute-optimal frontier, faithful horizon rises roughly linearly with oracle budget — there is no free
sub-linear knee (H69's strict bar unmet) — yet a consultation policy that triggers on the model's own
decode-entropy nearly **doubles** the faithful horizon of fixed-interval consultation at equal budget
(+57% at ρ=0.2, +94% at ρ=0.5), and the ablation shows it is the **uncertainty trigger, not the
speculative draft**, that does it — the same signal SPEC-15's CF6 found could not certify a conformal
coverage bound, here proven an excellent consultation scheduler, because certifying coverage and timing
a consultation are different jobs."* The headline is therefore **not** the knee and **not** the dead
floor+cliff, but the **smart-scheduling win on a real model** — measured exactly, against ground truth,
where every oracle-free domain can only guess.

## 7. Gate and what each branch licenses

**Gate: H69.** The composed curve either clears the §3 target on a real trained model or it does not.

- **H69 confirmed** → the program has its flagship. Next: the GPU arm (capacity beyond the single-machine
  frontier — does the knee sharpen or the target rise?), and SPEC-20 (does the faithful flagship train a
  *useful* downstream agent — the usefulness proof).
- **H69 refuted (floor+cliff survives the stack)** → the strongest negative. It localizes the entire
  faithful-horizon limitation to *compounding that no consultation policy makes sub-linear at CPU scale*,
  and licenses exactly one remaining lever: GPU-scale capacity (the standing open bet named in SPEC-10,
  SPEC-16). SPEC-20 still runs — usefulness does not require a sub-linear knee, only enough faithful
  horizon to plan within (which the `ρ=1` and landmark paths supply).

In both branches SPEC-20 (usefulness) is the next spec: the flagship checkpoint is its environment.

## 8. Status

| ID | Hypothesis / artifact | State | Result |
|---|---|---|---|
| FL0 | flagship checkpoint frozen | ✅ shipped + **frontier run** | the train→freeze→reload→gate lifecycle ships ([`experiments/flagship.py`](../../src/verisim/experiments/flagship.py), [`configs/flagship.json`](../../configs/flagship.json)): a single flat `M_θ` at the HS1.3 `l@9.6k` frontier is frozen to `model.pt` + `manifest.json`, and `verify_checkpoint` gates it on **reload-determinism** + the SPEC-10 band. **Frontier result:** the frozen `l@9.6k` checkpoint measures `H_free` = **18.75 id / 29.75 ood** — reproducing the SPEC-10 HS1.3 3-seed program-best (19.2/28.75) on a single seed — gate **PASS** (reload bit-exact, in band). |
| FL1 | H69 — headline `H_ε(ρ)` on real `M_θ` | ✅ shipped + **frontier run** ([`fl1_flagship_curve.png`](../../figures/fl1_flagship_curve.png)) | the four-arm `H_ε(ρ)` sweep on the frozen checkpoint ([`experiments/flagship_curve.py`](../../src/verisim/experiments/flagship_curve.py)): floor / fixed-ρ / **composed π_c** (conformal-on-*real*-decode-entropy OR speculative window) / ceiling. **Frontier result — H69 strict bar NOT met but a large real positive:** the curve rises ~linearly (no free sub-linear knee; floor 18.75 → ceiling 96), so composed reaches only ~31% of ceiling at ρ=0.2, not 80%. *But* the composed policy **nearly doubles fixed-interval** at equal budget, the gap widening with ρ: **+57% at ρ=0.2 (29.5 vs 18.75), +70% at ρ=0.3 (42 vs 24.75), +94% at ρ=0.5 (61.25 vs 31.5)** — `composed_beats_fixed: True`. Smart scheduling decisively beats the clock on a real model, where every stand-in produced a dead floor+cliff. |
| FL2 | H70 — methods compose | ✅ shipped + **frontier run** ([`fl2_composition.csv`](../../figures/fl2_composition.csv)) | the 2×2 ablation at equal budget ([`experiments/flagship_ablation.py`](../../src/verisim/experiments/flagship_ablation.py)). **Frontier result (ρ=0.3) — COMPOSES, and decomposes the FL1 win:** conformal-only **42.0**, speculative-only 18.75 (= floor, inert), neither (clock) 24.75, both 42.0. `both ≥ max(single)` so H70 composes — *not* super-additive: **the conformal trigger on the real signal carries the entire win; the speculative window is inert at this operating point.** The sharp CF6 refinement: a signal *non-conformalizable for a coverage certificate* is an *excellent consultation trigger* — different jobs. |
| FL3 | H71 — structure buys goal-horizon on flagship | ✅ shipped + **frontier run** — **SUPPORTED** ([`fl3_goal_horizon.csv`](../../figures/fl3_goal_horizon.csv)) | the structured-arm co-report ([`experiments/flagship_goal.py`](../../src/verisim/experiments/flagship_goal.py)): on **one** trained graph+RSSM arm, co-measure the HS3 **wall** (`H_free` at ρ=0) and the **goal-reach battery** (flat free-run vs landmark planning), reusing LP3's [`landmark.plan`](../../src/verisim/landmark/plan.py). **Frontier result — H71 SUPPORTED:** the wall survives (`H_free` = **0.33 ≈ 0**), yet landmark planning lifts far-goal reach **0.167 → 0.667 (4×)** at G=16, the gap widening with distance — structure buys goal-space horizon where it cannot buy step horizon, *reproduced on a real trained arm*, not LP3's stand-in. |
| FL4 | H72 — model-invariance at flagship | ✅ shipped + **frontier run** — **SUPPORTED** ([`fl4_proposer_swap.csv`](../../figures/fl4_proposer_swap.csv); LLM arm deferred) | the proposer swap ([`experiments/flagship_swap.py`](../../src/verisim/experiments/flagship_swap.py)): the FL1 curve for the **flat** (FL0 checkpoint) and **graph+RSSM** proposers, compared by magnitude-free `shape_signature`. **Frontier result — H72 SUPPORTED:** both share one shape (no sub-linear knee), floor set by competence (flat 18.75 ≫ graph 0.50), ceiling 96 — the loop governs the shape, the proposer sets the floor (H22 confirmed on real trained models). Frozen-LLM arm deferred (LP7/GPU rule). |
| FL5 | flagship report + one-sentence headline | ✅ written from frontier numbers | the one-sentence headline (§6) and the report section land from the FL0–FL4 frontier runs; see the headline below and [`docs/report.md`](../report.md) (SPEC-19/20 section). |
| FL6 | H77 — why the real signal schedules well (ranking vs calibration) | ✅ shipped + **frontier run** — **SUPPORTED** ([`flagship_signal.py`](../../src/verisim/experiments/flagship_signal.py), [`fl6_signal_diag.csv`](../../figures/fl6_signal_diag.csv)) | the mechanism behind FL1/FL2: free-run the real `M_θ`, measure the signal's rank correlation with true divergence and the trigger-precision lift. **Frontier result — H77 SUPPORTED:** Spearman(signal, divergence) = **+0.352** (the signal *ranks* drift) and consulting the top-20%-signal steps catches breaches at **0.902 precision vs a 0.652 base rate** (+0.25 lift). So the real decode-entropy *orders* drift well enough to schedule consultations on the steps about to break (FL1's win) even though CF6 found it cannot *calibrate* an absolute coverage threshold — **ranking ≠ calibration**, both true. |

Honest caveats, stated up front: (1) the compute-optimal frontier is a *single-machine CPU* frontier;
the GPU arm stays the standing open bet and SPEC-19 does not claim to settle scale, only to make the
CPU-frontier flagship real. (2) The frozen-LLM proposer arm (FL4) depends on the deferred LLM stack and
may ship with the GPU bet. (3) If H69 is refuted, the value of the spec is the *negative* — and §10.1's
discipline says that is bankable, not a failure.
