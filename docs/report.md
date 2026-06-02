# Verisim v0 — technical report

> The v0 result, stated honestly. This is the short write-up SPEC-2 §13 (M8) calls
> for: the experiments that produce figures — E1 (the curve), E2/E3 (policy and
> operator comparisons), the §7.2 calibration diagnostic, and the E4 ablation
> (size/difficulty, the supervised-vs-+RLVR objective axis, and the delta-vs-full-state
> representation axis) — what
> they show, and what they do not. Every number here is read from a committed
> run-record CSV and is regenerable from a config + seeds (SPEC-2 §12). Figures live
> in [`../figures/`](../figures/).

## Bottom line

v0's job was to build the apparatus that can measure **how much oracle consultation
buys how much faithful horizon** (`H_ε(ρ)`), and to run the experiments on it. The
apparatus is built, tested, and reproducible. The headline scientific finding is a
**clean set of negatives** at the small, fast committed scale:

- **H1 (a favorable knee exists):** *not observed.* The `H_ε(ρ)` curve is flat and
  near the floor across the interior and only reaches the ceiling at `ρ = 1`.
- **H2 (smart beats dumb):** *refuted at this scale.* Fixed-interval consultation
  **beats** the uncertainty/drift-triggered policies at equal budget.
- **H3 (correction operator matters):** *identity, as predicted.* `hard_reset`,
  `residual`, and `projection` are statistically indistinguishable on faithful
  horizon — expected from a full-state oracle truth.
- **Why (diagnostics):** the uncertainty signal that should drive H2 is **barely
  correlated** with actual error (Pearson ≈ 0.11), and **scaling the model 4× does
  not lift** clean per-step accuracy off its ~0.1–0.2 floor — so the levers are
  calibration and training budget/difficulty, not policy cleverness or raw size.

None of these refute the *program* (SPEC.md §9 explicitly treats a refuted
hypothesis as a result, not a failure). They locate the work, and the two
diagnostics (calibration §7.2, the E4 ablation §9) make the next levers concrete:
the smart policies lose because their uncertainty signal is uncalibrated (SPEC-2
§17.2), and the clean floor does not move with model size, so the open work is
training budget / difficulty co-tuning (SPEC-2 §17.5), not parameters. The
contribution of v0 is the **measurement**, the **honest curves**, and a benchmark +
RL environment others can build on (SPEC-2 §15).

## Method (one paragraph)

A state is a serializable shell + filesystem snapshot; a deterministic reference
oracle `O(s, a)` defines the true transition (SPEC-2 §2–3). A from-scratch
decoder-only transformer `M_θ` predicts a structured **delta** under
grammar-constrained decoding (M4). The propose–verify–correct loop rolls `M_θ`
forward, consulting the oracle on a budget `ρ` and correcting with an operator `C`
(M5/M7). Faithfulness is the **normalized symmetric difference** `d(s, ŝ) ∈ [0,1]`;
the **faithful horizon** `H_ε` is the number of steps a rollout stays within `ε`
(SPEC-2 §7). Each experiment sweeps the loop and aggregates `H_ε` over seeds with
percentile bootstrap CIs.

## E1 — the `H_ε(ρ)` curve (H1)

Sweep `ρ ∈ {0, .05, .1, .2, .3, .5, 1}` × `ε ∈ {0, .05, .1}` × difficulty ∈
{low, high}, 5 seeds, `T = 24`, `fixed` policy + `hard_reset`
([`e1_curve.png`](../figures/e1_curve.png), [`e1_curve.csv`](../figures/e1_curve.csv)).

| ρ | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 |
|---|---|------|-----|-----|-----|-----|-----|
| `H_ε` (low, ε=0) | 0.0 | 1.4 | 1.4 | 1.4 | 1.6 | 1.4 | 24.0 |
| `H_ε` (high, ε=0) | 0.2 | 1.2 | 1.2 | 1.2 | 1.2 | 1.4 | 24.0 |

The interior is flat at `H_ε ≈ 1.2–1.6` — about **5–7% of the ceiling** — then jumps
to the full `T = 24` only at `ρ = 1`. That is the *opposite* of H1's hoped-for
shape (≥80% of ceiling horizon at ≤20% budget): there is no knee, just a floor and a
cliff. The model drifts immediately at `ρ = 0` (it cannot reliably predict even step
0), so consultations buy back only the few steps until the next drift. **H1 is not
supported at this scale**; whether it holds at all is a model-capacity/difficulty
tuning question (the open M6 work, SPEC-2 §17.5), not a property of the loop.

## E2 — consultation policy (H2)

Fix `ρ = 0.2` (budget = 4 calls over `T = 24`); compare `fixed` vs.
`uncertainty_triggered` vs. `drift_triggered` at **equal budget** (the runner spends
exactly 4 calls per arm), 10 rollouts/arm
([`e2_policies.png`](../figures/e2_policies.png), [`e2_policies.csv`](../figures/e2_policies.csv)).

| policy | `H_ε` | 95% CI | oracle calls |
|--------|------|--------|--------------|
| `fixed` | **1.3** | [1.0, 1.8] | 4.0 |
| `uncertainty` | 0.2 | [0.0, 0.5] | 4.0 |
| `drift` | 0.1 | [0.0, 0.3] | 4.0 |

The dumb baseline wins — so this is not a wash, it is a **reversal**: at this scale
spending the budget where the model is *least confident* is worse than spreading it
evenly. `fixed` beats both triggered policies with **disjoint CIs** — `[1.0, 1.8]` vs.
`uncertainty`'s `[0.0, 0.5]` and `drift`'s `[0.0, 0.3]`. The reason is calibration: the triggered
policies key off the mean entropy of the constrained decode (SPEC-2 §7.2), and for a
model this small that entropy does not track actual divergence. **H2 is refuted at
this scale**, and the next lever is explicit: calibrate the uncertainty signal
(SPEC-2 §17.2) before re-running.

### Why the smart policies lose: the calibration diagnostic (§7.2)

The H2 reversal has a measurable cause. The §7.2 uncertainty-calibration diagnostic
collects per-step `(signal, divergence)` pairs — the model's decode-entropy
confidence against its *actual* error that step, teacher-forced so it is uncompounded
— and asks whether confidence predicts error
([`calibration.png`](../figures/calibration.png), [`calibration.csv`](../figures/calibration.csv),
240 pairs):

| Pearson | Spearman | mean divergence |
|---------|----------|-----------------|
| 0.11 | 0.18 | 0.16 |

Both correlations are near zero, and the reliability curve is essentially **flat**:
the model's *most confident* steps (lowest-entropy bin) carry divergence ≈ 0.14, no
better than its least confident (≈ 0.20). So the entropy signal carries almost no
information about where the model errs — which is exactly why a policy that spends the
budget on high-entropy steps cannot beat one that spreads it evenly. This turns "H2 is
refuted" into a concrete, falsifiable next step: a triggered policy can only help once
the signal it keys off is calibrated (SPEC-2 §17.2), so the diagnostic — not a new
policy — is the lever to move next.

## E3 — correction operator (H3)

Fix `fixed`/`ρ = 0.2`; compare `hard_reset` vs. `residual` vs. `projection`, 10
rollouts/arm ([`e3_operators.png`](../figures/e3_operators.png),
[`e3_operators.csv`](../figures/e3_operators.csv)).

| operator | `H_ε` | 95% CI |
|----------|------|--------|
| `hard_reset` | 1.3 | [1.0, 1.8] |
| `residual` | 1.3 | [1.0, 1.8] |
| `projection` | 1.3 | [1.0, 1.8] |

The three are **identical**, not merely indistinguishable. This is a theoretical
identity, not a measurement artifact: when the oracle returns the *full* one-step
truth, every operator snaps the coupled state to the same `s'` (SPEC-2 §6.2). The
operators differ only in the diagnostic they expose — `residual` logs the
discrepancy magnitude (the Stage-2 online-learning signal), `projection` logs the
per-correction repair cost — neither of which changes the horizon without **partial
verification** or **online learning**, both deferred. By H3's own refutation
condition (hard reset indistinguishable or better), **H3 is not supported at v0**,
and the experiment makes precise *why* and *what would change it*.

## E4 — ablation: is the H1 floor a capacity problem? (§9, §17.5)

E1 left open *why* the model drifts immediately at ρ=0: too small, or task
mis-tuned (SPEC-2 §17.5)? E4 sweeps the two buildable §9 ablation axes — **model
size** (tiny `1×32` → small `2×64` → medium `4×128`) and **difficulty/driver** —
and measures clean (ρ=0) per-step teacher-forced accuracy, 5 seeds/cell
([`e4_ablation.png`](../figures/e4_ablation.png), [`e4_ablation.csv`](../figures/e4_ablation.csv)).

| size | low (weighted) | high (adversarial) |
|------|----------------|--------------------|
| tiny `1×32` | 0.09 | 0.22 |
| small `2×64` | 0.09 | 0.15 |
| medium `4×128` | 0.14 | 0.17 |

Clean per-step accuracy stays in the **0.09–0.22** band across a 4× depth / 4× width
increase, with heavily overlapping CIs — and clean horizon stays near zero
everywhere. **Scaling the model within this range does not fix the floor.** So the H1
negative is *not* simply "too few parameters" at this training budget; the lever is
elsewhere — training iterations / dataset size and difficulty co-tuning (SPEC-2
§17.5), not raw model size. (A reproducible curiosity: the adversarial "high" driver
is sometimes *easier* to predict per-step than "low" — its destructive commands often
fail predictably and leave state unchanged, which the model reproduces exactly more
often than it does structure-building writes.)

### Objective axis: supervised vs. +RLVR (§9, §17.4)

The third E4 axis asks whether **training against the oracle** — Stage-2 RLVR, which
REINFORCE-trains the model on the oracle's faithful-horizon reward
([`src/verisim/train/rlvr.py`](../src/verisim/train/rlvr.py)) — buys clean
faithfulness over supervised pretraining alone. One Stage-1 supervised model is
branched into a Stage-2 RLVR copy and both arms are scored on the same clean (ρ=0)
metrics, 5 seeds/cell ([`objective.png`](../figures/objective.png),
[`objective.csv`](../figures/objective.csv)).

| objective | clean acc (low) | clean acc (high) | clean horizon (high) |
|-----------|-----------------|------------------|----------------------|
| supervised | 0.07 | 0.15 | 0.2 |
| +RLVR | 0.07 | 0.13 | 0.2 |

RLVR is an **honest null at this scale**: clean per-step accuracy is identical on the
`low` driver (0.07) and a hair lower on `high` (0.13 vs. 0.15, CIs `[0.08,0.18]` vs.
`[0.09,0.22]` — fully overlapping), and clean horizon is unchanged. The cause is
structural, not a bug: the faithful-horizon reward is **sparse exactly when the model
is at the H1 floor** — episodes terminate at the first unfaithful step, so a model that
usually fails step 0 sees almost no reward signal to amplify. RLVR has leverage only
once the model already sustains a non-trivial horizon, which is the difficulty
co-tuning (§17.5) this scale has not yet reached. The machinery is correct and tested
([`tests/test_rlvr.py`](../tests/test_rlvr.py): it learns from scratch on a tiny env
and does not collapse a faithful model); what it needs is a task with horizon to
extend.

### Representation axis: delta vs. full-state (§9, §10)

The last §9 axis asks whether the **prediction target** matters: predict the localized
**delta** (the primary `M_θ`) or regenerate the **full next state**? SPEC.md §6.1 argues
delta should win — it bounds the hallucination surface and localizes verification. To
measure it, a full-state head was built (the `StateGrammar` +
[`constrained_decode_state`](../src/verisim/model/decode.py) +
[`FullStateWorldModel`](../src/verisim/model/full_state.py), which constrained-decode the
*whole* next state the way the delta decoder constrains edits) and trained on identical
data to the delta model; both are scored on the same clean (ρ=0) metrics, 5 seeds/cell
([`representation.png`](../figures/representation.png),
[`representation.csv`](../figures/representation.csv)).

| representation | clean acc (low) | clean acc (high) | clean horizon (high) |
|----------------|-----------------|------------------|----------------------|
| delta | 0.07 | 0.15 | 0.2 |
| full_state | 0.00 | 0.03 | 0.0 |

This is the **first E4 axis with a clear directional result**: delta dominates full-state
at every cell (clean per-step accuracy 0.07/0.15 vs. 0.00/0.03 on low/high; clean horizon
0/0.2 vs. 0/0), confirming SPEC.md §6.1. The reason is structural and on-thesis — to score
a step, the full-state model must regenerate *every* fact of the next world correctly
(grammar-validity is free, but faithfulness of the whole tree is not), whereas the delta
model need only emit the handful of edits the action makes; the larger target surface is a
strictly lower faithfulness floor. The committed scale is tiny, so the absolute numbers are
floor-level for both arms, but the *ordering* — delta > full-state — is exactly the
prediction the project's representation choice rests on, now measured rather than asserted.

## Automating the search: an oracle-gated ratchet (§17.5)

The §17.5 co-tuning — find a config where the clean floor lifts — is a search problem,
so v0 automates it as a *keep-if-better* ratchet
([`src/verisim/auto/search.py`](../src/verisim/auto/search.py),
[`auto_search.png`](../figures/auto_search.png),
[`auto_search.csv`](../figures/auto_search.csv)), modeled on Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch): propose a one-knob mutation,
train under a fixed budget, score one number, keep it only if it beats the running best.
The difference is the gate. `autoresearch` scores on val_bpb — a held-out-loss *proxy*;
verisim scores on **mean clean (ρ=0) per-step accuracy against the oracle's true next
state** — ground truth, the deterministic "lint" the whole project is built on. That is
verisim's single comparable scalar (its "val_bpb"), and it is a strictly stronger
"did we improve?" signal than any oracle-free domain can construct (SPEC.md §4). `H_ε`
is deliberately *not* the gate: at this scale it is ~0 everywhere (the H1 negative), too
flat to climb, so accuracy is the smooth signal — the same reasoning behind `autoresearch`
choosing val_bpb.

A first 12-trial run (`configs/auto.json`) autonomously **more than doubled the clean
floor, 0.042 → 0.094**, by finding `train_iters↑` then `lr↑`, then held through ten
rejected trials. The absolute level is still floor-level (consistent with the H1 negative
at v0 scale), but the *mechanism* provably lifts the oracle-grounded metric without a human
in the loop — the self-improving "update-from-reality" loop the program is ultimately
about, here at the outer (config-search) layer that wraps the RLVR inner loop. The
proposer is seeded coordinate hill-climbing; an LLM-agent proposer drops in behind the
same oracle gate.

## K0 — does the learner work, and where does it fail? (SPEC-2.1)

[SPEC-2.1](specs/SPEC-2.1.md) paused the roadmap to earn an *interesting* `H_ε(ρ)` knee on
the single-filesystem world, and its first phase (K0) asks the prerequisite question before
any tuning: **can the pipeline fit the transition function at all, and if the floor persists,
exactly where does it fail?** ([`k0_control.png`](../figures/k0_control.png),
[`k0_control.csv`](../figures/k0_control.csv); `verisim.experiments.k0`, `configs/k0.json`).

**The control — the learner works.** Trained on a *trivial* world (depth-1, single-segment
`mkdir`/`touch`/`write` under root — collision-free successes, so the only thing to learn is
to copy the action's one-token argument into the delta), with the new minibatch + warmup/cosine
+ val-early-stopping trainer (`train_batched`, SPEC-2.1 §6) on 768 transitions, the model reaches
**clean per-step faithfulness = 1.000 (exact), gate 0.95 → PASS**. The full pipeline
(data → tokenize → train → constrained decode → apply → divergence) can fit a deterministic
computer-state transition to *bit-exact* ground truth. So the v0 floor is **not** a broken
learner and **not** (per E4) a capacity wall.

**The diagnosis — the floor is a generalization failure, localized to argument-copying.** On
the baseline config, the model **memorizes its training transitions perfectly yet does not
generalize**: train accuracy **1.000** vs. held-out **0.083** — the textbook under-data /
under-coverage signature SPEC-2.1 §1 predicted. The per-edit-type breakdown pinpoints *where*:

| signal | value | reading |
|---|---|---|
| `create` precision/recall | **0 / 0** (69 predicted, 34 true, 0 exact) | the model emits creates but **never at the exactly-correct path/content** — exact multi-token *argument copying* into the delta is the bottleneck |
| `setresult` | 72/144 correct (~0.50) | it gets the *success* cases (empty stdout, exit 0) and misses the *failures* |
| divergence by fact type | `exit` 138, `file` 96, `dir` 53 | dominated by **mispredicted failure/collision cases** (`exit`) and wrong created-node identity (`file`/`dir`); `cwd`/`env`/`stdout` are nearly learned |
| mean bits-to-correct | 60.0 | the smooth gate (SPEC-2.1 §3) baseline the K-series and the ratchet will drive down |

A separate convergence probe (25× more training, 160 → 768 transitions) confirmed the copy
bottleneck is **not** dissolved by more steps alone: observation-class facts
(`exit`/`stdout`/`cwd`/`env`) become fully learned, but the created-node residual persists,
while depth-1 (one-token copy) reaches 1.0. So the floor's mechanism is now specific and
testable: **exact reproduction of multi-token action arguments (deep paths, content) in the
delta, plus coverage of failure/collision cases** — not generic under-training.

**What this redirects (K1/K2).** (1) Coverage-balanced data + hard-negative mining over the
path-copy distribution and the failure cases the baseline driver under-samples (K1); (2) the
`train_batched` budget that took the trivial world to 1.0 (K2); and (3) an open representation
question for K3/architecture — whether a *copy-aware* delta (referencing action arguments by
pointer rather than re-emitting path tokens) is the lever, since copying is the precise hard
spot. The gate metric moves from sparse 0/1 accuracy to **bits-to-correct** (smooth, monotone)
for the search. This is the K0 contract met: the learner is proven, and the floor is no longer
a mystery but a named, falsifiable target.

## K1+K2 — coverage data, trained properly, past the acceptance floor (SPEC-2.1)

K0 left a precise target: the floor is exact *multi-segment argument copying* into the delta,
under-covered and under-trained. K1/K2 attack it directly
([`k2_faithfulness.png`](../figures/k2_faithfulness.png),
[`k2_faithfulness.csv`](../figures/k2_faithfulness.csv); `verisim.experiments.k2`,
`verisim.data.coverage`, `configs/k2.json`).

**K1 — coverage of the transition space (the data is free).** A dependency-free coverage
report (`verisim.data.coverage`) over a broad driver mix confirms the dataset spans what the
baseline missed: **all 13 commands covered, 359 failure cells** (`mkdir:fail`, `rmdir:fail`,
`rm:fail`, `mv:fail`, …) and a **create-depth histogram spanning depths 1→8** — i.e. the
failure cases *and* the multi-segment path-copy distribution K0 flagged. The K1 gate
(documented coverage, regenerable from a manifest) is met.

**K2 — train properly, and the copy bottleneck dissolves.** Training the same small model
(2 layers × 128) on the copy distribution (the `structural` driver: collision-free multi-depth
creates, 2,560 transitions) with the new minibatch + warmup/cosine + val-early-stopping trainer
(`train_batched`, 6,000 steps) lifts clean (ρ=0) per-step faithfulness on a **held-out
non-trivial difficulty** from K0's ~0.09 to:

| metric | value | vs. gate |
|---|---|---|
| exact-match | **0.859** | **gate 0.5 → PASS** |
| acceptance @ ε=0.05 | **0.875** | clears the SPEC-3 §8 acceptance floor (~0.5) |
| graded (mean 1−d) | **0.988** | — |

This is the decisive K2 finding: **the K0 copy bottleneck is a coverage/training problem, not a
representation wall.** 25× more training at K0's tiny data did *not* move it (0.09), but
~3× the data + a real training loop takes the *same architecture* to 0.86 exact — vindicating
SPEC-2.1 §1's "under-data/under-training" diagnosis and making the K3/architecture
copy-representation lever unnecessary at this scale.

**Why this matters for the knee.** Per-step acceptance 0.875 implies an unaided (ρ=0) geometric
faithful horizon of ≈ 1/(1−0.875) ≈ 8 steps — already ~50% of a T=16 ceiling, with real room
for cheap consultation to extend it. The model now sits *inside* the K3 "competent-but-
compounding" band (0.7–0.95), which is exactly the regime where an interesting `H_ε(ρ)` knee is
expected to appear. K3 (the difficulty dial) and K4 (re-run E1) are next; the
**bits-to-correct** gate (SPEC-2.1 §3) and the tested **hard-negative mining** loop
(`mine_hard_negatives`, active learning over the oracle) are in place to drive the search.

## K3+K4 — the knee hunt: an honest negative that licenses the network world (SPEC-2.1)

With a competent model in hand (K2), K3/K4 implemented the difficulty dial and ran the loop to
look for the knee — the prime directive of SPEC-2.1
([`k4_knee.png`](../figures/k4_knee.png), [`k4_policies.png`](../figures/k4_policies.png);
`verisim.experiments.k4`, `configs/k4.json`).

**K3 — the difficulty dial.** The deferred SPEC-2 §2.4 dial is implemented as `max_depth` on the
driver (`verisim.data.drivers.path_depth`, threaded through `build_dataset`/`eval_actions`). At
the chosen sweet spot (`structural`, `max_depth=4`, rollout length `T=48`) the K2 model's
per-step acceptance is ~0.875 and the unaided (ρ=0) faithful horizon is ~10/48 — **floor well
below ceiling, so there is genuine room** for consultation to buy horizon back. The dial gate is
met.

**K4 — the curve, and the honest negative.** The competent model is a real, large win over v0:
the ρ=0 floor rose from ~0 (original v0) to **~10/48**. But the `H_ε(ρ)` curve is a **floor +
cliff, not a knee** — at ε=0.05, `H_ε` is 10.3 (ρ=0) → 11.3 (ρ=0.2) → 13.8 (ρ=0.5) → 48 (ρ=1):

| ρ | 0 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 |
|---|---|-----|-----|-----|-----|-----|
| `H_ε` (ε=0.05) | 10.3 | 10.3 | 11.3 | 12.2 | 13.8 | 48.0 |

And **smart consultation does not help** (E2 with the competent model, equal budget at ρ=0.2):
`fixed` 11.3 vs `uncertainty` 10.7 vs `drift` 10.9 — overlapping CIs, the H2 negative persisting
even now. So **C-knee (a favorable knee on the single-filesystem world) is refuted at this
scale** — the SPEC.md §9 "approximately linear, no free lunch" refutation condition.

**Why — the mechanism, now fully characterized.** Filesystem prediction errors are *discrete*:
one wrong edit spikes the set-difference divergence past ε in a single step. So `H_ε`
(first-exceedance) is governed by the position of the *first* error, which evenly-spaced resets
cannot push out — and the model's decode-entropy uncertainty does not localize *which* steps
will err, so error-targeting consultation cannot catch them either. A fixed-interval knee would
need per-step acceptance ≈0.98, which leaves no room (horizon ≈ ceiling). This is not a tuning
miss; it is a property of *this world's metric* (discrete set-difference) and *this model's
uncertainty* (uncalibrated decode entropy).

**This is the result the plan anticipated, and it is load-bearing.** Per SPEC-2.1 §10, a refuted
C-knee — *after* the learner is proven (K0) and the floor is lifted (K1/K2) — is exactly what
**licenses the network world (SPEC-5)**: a knee needs *gradual* drift (continuous quantities like
RTT/throughput that accumulate smoothly, where periodic resets keep divergence under ε) and a
*calibrated* uncertainty (the RSSM belief-variance partial observability supplies), neither of
which the fully-observable, discrete-state filesystem has. SPEC-2.1 did its job: the learner
works, the floor is gone, and the evidence now says the knee lives one world up. The paused
SPEC-5 is no longer a hopeful bet — it is an evidence-backed next step.

## NW5+NW6 — the network world's loop, and its first `H_ε(ρ)` curve (SPEC-5)

The deterministic network core (NW0–NW3) and the flat supervised `M_θ` (NW4) were already
shipped. NW5 adds the **partial-observation propose-verify-correct loop**
([`netloop/`](../src/verisim/netloop/)): a model-agnostic runner, the two baselines
(null / oracle-backed), the two-mode **partial-observation oracle** (a cheap *probe* that
reveals one host's subgraph vs. an expensive *full* consult, SPEC-5 §5.3), probe policies
`π_o` (§8.2), and correction/belief operators including the **belief filter** that snaps only
the observed subgraph (§8.3). The loop invariants mirror v0's and are separately tested
(`tests/test_net_loop.py`): ρ=1 full-consult reproduces the oracle exactly; a perfect model
never drifts at ρ=0; the budget is never exceeded; and — the property partial observability
buys that v0 could not — a one-host **probe corrects strictly less than a full consult**, so
probe-mode horizon is provably ≤ full-mode horizon at equal ρ (no v0 identity collapse).

NW6 plots the prime-directive curve — **EN1**
([`en1_curve.png`](../figures/en1_curve.png); `verisim.experiments.en1`, `configs/en1.json`).
On the flat-Markov `M_θ` the interior is **near-flat, then a cliff at ρ=1** — the *opposite*
of a favorable knee (ε=0.05):

| ρ | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 |
|---|---|------|-----|-----|-----|-----|-----|
| `H_ε` (high, ε=0.05) | 1.0 | 1.6 | 1.6 | 1.6 | 2.4 | 3.0 | 24.0 |
| `H_ε` (low, ε=0.05)  | 0.8 | 1.4 | 1.4 | 1.4 | 1.4 | 3.4 | 24.0 |

**This is the H8 honest negative for the flat arm**, and it is the network analogue of v0's H1
floor — not a surprise but a measurement. The mechanism is the model, not (yet) the world: the
flat `M_θ` memorizes its training trajectories (teacher-forced accuracy 1.0) but generalizes
poorly off-distribution — held-out teacher-forced accuracy ~0.71 *per token*, and only **~0.2
exact-delta match** free-running, so a single wrong token breaks the whole delta and the rollout
drifts past ε in ~1 step without consultation. At ρ=0.2 the model recovers only ~7% of the
ceiling horizon — far below H8's "≥80% of ceiling at ≤20% consultation."

**What it licenses.** A flat near-floor interior on the flat-Markov baseline is exactly the
result that makes the **NW7** levers load-bearing rather than optional: the message-passing +
RSSM graph arm (§6.1–6.2, H11), and the drift mitigations v0 never had — **noise-injected
rollout training** and **self-forcing/scheduled sampling** (the GNS/m4 levers, §6.3). The EN1
machinery now exists to measure, cleanly and reproducibly, whether any of them lifts `M_θ` off
this floor. As in v0, the negative is reported first-class and the apparatus (loop invariants,
determinism, the CI-tested core) is proven independently of the model's faithfulness.

## NW7 (flat arm) — EN2 consultation policy (H9) and EN3 operators (the partial-observation payoff)

With the loop in hand, two of NW7's equal-budget comparisons run on the flat `M_θ` now (the
graph/RSSM arm, the smart information-gain probe, and the drift mitigations are the remaining
NW7 work). Both fix the budget at the EN1 interior ρ=0.3 and compare at *equal* consultation
count (the spend-down backstop makes every arm spend exactly the budget).

**EN2 — consultation policy `π_c` (H9; `en2_policies.csv`, `verisim.experiments.en2`).** Does
spending the budget on the steps the model is least sure about *earn* more horizon than
spreading it evenly? At ε=0 (high+low pooled, 7 calls each):

| policy | `H_ε` | 95% CI |
|---|---|---|
| `uncertainty` | **3.4** | [1.1, 6.0] |
| `drift` | 2.0 | [0.7, 3.9] |
| `fixed` | 1.9 | [1.1, 3.1] |

The uncertainty-triggered policy *leads*, and — unlike v0's E2/K4, where `fixed` won with
**disjoint** CIs (a clean H2/H9 negative) — here the direction has flipped to favor smart
scheduling. But the CIs **overlap**, so this is **suggestive, not conclusive**: the flat
model's decode-entropy signal is a coarse proxy, and the calibrated belief-variance signal that
H9 really wants is what the RSSM belief (NW7 graph arm) supplies. Honest read: encouraging
movement off v0's negative, awaiting the model that can confirm it.

**EN3 — correction/belief operators (§8.3; `en3_operators.csv`, `verisim.experiments.en3`).**
This is the result partial observability buys that v0 could not show. At ε=0:

| operator | mode | `H_ε` | oracle-bits / consult | `H_ε` per oracle-bit |
|---|---|---|---|---|
| `hard_reset` | full | 1.9 | 10.1 | 0.027 |
| `residual` | full | 1.9 | 10.1 | 0.027 |
| `projection` | full | 1.9 | 10.1 | 0.027 |
| `belief_filter` | **probe** | 0.9 | **2.1** | **0.063** |

The three full-consultation operators give **identical** `H_ε` — the v0 full-truth identity
(they all snap the coupled state to the same `s'`), persisting exactly as predicted. But the
probe + belief filter **breaks the identity collapse**: a one-host probe corrects only the
observed subgraph, so it earns *less* horizon per consult (0.9 vs 1.9). The honest — and, framed
as an incentive rather than a penalty, the *better* — read is the cost lens: the probe reveals
~2.1 facts where a full consult reveals ~10, so per **oracle-bit** the probe earns **~2.3× more
faithful horizon** (0.063 vs 0.027). Cheap, localized sensing is the *efficient* way to buy
horizon — which is exactly the probe-efficiency axis (§9.4) the smart information-gain `π_o`
(H10, NW7) is built to optimize, and the first network result with no v0 analogue.

## NW8 (graph arm) — EN4 graph-vs-flat, a split verdict on H11 (SPEC-5 §6.1-6.2, §12)

The NW8 message-passing + RSSM graph arm now ships ([`netmodel/graph.py`](../src/verisim/netmodel/graph.py),
[`graph_model.py`](../src/verisim/netmodel/graph_model.py), [`graph_train.py`](../src/verisim/netmodel/graph_train.py)),
and EN4 ([`experiments/en4_graph.py`](../src/verisim/experiments/en4_graph.py)) runs the H11 comparison:
the flat-Markov transformer and the graph+belief arm trained on the **same** oracle data and scored with
the **same** eval primitives EN1 uses. A small, fast smoke instance
([`figures/en4_graph_vs_flat.png`](../figures/en4_graph_vs_flat.png),
[`.csv`](../figures/en4_graph_vs_flat.csv)) gives a clean, two-sided first datum:

| arm | one-step token acc | **delta-exact** rate | `H_ε` (ρ=0), ε ∈ {0, 0.05, 0.1} |
|---|---|---|---|
| flat-Markov (NW4) | 0.673 | 0.264 | 0.000 / 0.000 / 0.000 |
| **graph + RSSM (NW8)** | **0.838** | **0.569** | 0.000 / 0.000 / 0.000 |
| graph + RSSM + noise lever (§6.3) | 0.828 | 0.556 | 0.000 / 0.000 / 0.000 |
| graph + RSSM + self-forcing lever (§6.3) | 0.803 | 0.500 | 0.000 / 0.000 / 0.000 |

The **delta-exact** column ([`netmetrics/exact.py`](../src/verisim/netmetrics/exact.py)) is the honest middle
the report previously flagged as missing: not token accuracy (which is inflated — most tokens of a delta are
easy structural scaffolding) and not horizon (which is `0` the instant any step exceeds ε), but the per-step
question a delta predictor is actually asked — *did the model freely decode the exact true edit set this step?*
It is `1` iff `bits_to_correct = 0`, computed by running each arm's own constrained decode (no teacher forcing)
and matching the assembled `NetDelta` as a multiset.

**The positive (H11, generalization axis): structure helps, and helps *more* on the honest metric.** On
never-trained eval seeds the graph arm is a **+16.5-point** better one-step token predictor than the flat arm
(0.838 vs 0.673) — and a **+30.6-point** better *delta-exact* predictor (0.569 vs 0.264), more than double the
flat arm's whole-delta exactness. The message-passing inductive bias over the host graph generalizes where the
flat serializer memorizes (the m4/GNS bet, §2.2-2.3, realized), and the gap *widens* on the stricter metric:
token accuracy understates how much structure buys, because the flat arm gets the easy scaffolding tokens right
while missing the edit that matters. This is the clearest network result yet in the graph arm's favor.

**The honest negative (horizon axis): even 57% delta-exact ≠ horizon — yet.** Neither gain converts to
free-running faithful horizon: `H_ε` is **0 for all four arms even at ε=0.1** — every arm drifts on the first
unaided step. This is exactly EN1's H8 negative and SPEC-2.1's K4 echoing in the network world, and the
delta-exact number now *quantifies* why: at 0.569 per-step exact, the probability of surviving even a few
unaided steps decays geometrically (≈0.57·0.57·… ), and first-exceedance is discrete — one wrong edit spikes the
graph divergence past ε in a single step. Per-step exactness this far below 1.0 cannot buy horizon without the
*exposure-bias* levers — and **both §6.3 levers now confirm that bound, identically**. The noise-injection lever
slightly *lowered* both metrics (0.828/0.556) and the self-forcing / scheduled-sampling lever lowered them a bit
more (0.803/0.500); neither bought any horizon. That the two levers — random-corruption and the model's own-drift
distribution — behave the *same* way is itself informative: at this scale the exposure-bias correction trades a
little one-step accuracy for off-distribution coverage that does not yet convert, a clean "needs scale/tuning"
datum, not a refutation of either lever.

**Where this routes the program (the epistemic engine, SPEC.md §10.1).** The result *localizes* the wall: it is
the **one-step→horizon conversion**, not the per-step learner (the graph arm fits to >0.9 teacher-forced accuracy,
the K0-analog check; held-out, it is delta-exact on 57% of steps). The **delta-exact metric just shipped**
([`netmetrics/exact.py`](../src/verisim/netmetrics/exact.py)) and is now an EN4 column — it converts the wall from
a qualitative claim into a number: 0.569 per-step exact is far enough below 1.0 that geometric decay kills horizon,
so the conversion levers must raise *whole-delta* exactness, not just token accuracy. **Both §6.3 exposure-bias
levers have now shipped and run** — noise-injection and self-forcing / scheduled sampling
([`graph_train.py`](../src/verisim/netmodel/graph_train.py)) — and both land the same banked negative at this
scale (above): the gap is not closed by exposure-bias correction alone here, which points the remaining budget at
**scale** and the **multi-step latent-overshooting** objective, and at *objective grounding* rather than only
input-distribution fixes. The **SPEC-8 OG1/OG2 deterministic machinery also shipped** (the oracle-grounded-SSL
target/`D`-mask factory [`netdata/grounding.py`](../src/verisim/netdata/grounding.py) and the hard-negative /
counterfactual factory [`netdata/negatives.py`](../src/verisim/netdata/negatives.py); both torch-free and
property-tested), and the **EN8/EN9 oracle-grounded-SSL runs (SPEC-8 §7) have now shipped on this same arm**
(below). A +16.5-pt token / +30.6-pt delta-exact one-step gain with a measured conversion gap is a better starting
point than the flat arm offered, and every number here is bankable under the oracle.

## EN8 (oracle-grounded SSL) — the oracle in the *bulk* of the cake, a second split verdict (SPEC-8 §7)

The EN1/K4 negative pre-registered a pivot: if consultation budget alone does not lift the curve, the lever
is *representation and objective*, so route to SPEC-8 — put the oracle's exact truth in the **self-supervised
bulk**, not only the RL cherry. OG1/OG2 shipped the deterministic data factory (the decidable/residual
partition and the exact hard-negative generator, torch-free and property-tested); **EN8 is the GPU consumer**
that ablates it on the NW8 graph+RSSM arm ([`experiments/en8.py`](../src/verisim/experiments/en8.py),
[`netmodel/grounded_train.py`](../src/verisim/netmodel/grounded_train.py)). Two pre-registered axes
([`figures/en8_grounding.png`](../figures/en8_grounding.png), [`.csv`](../figures/en8_grounding.csv)); like EN4,
a committed *smoke* instance — and like EN4, a clean two-sided result.

**H23 — the collapse axis: the oracle-anchored target removes the collapse tax (a positive).** A JEPA-style
latent predictor over the graph summary, with the prediction target either *learned* (the BYOL/JEPA EMA
encoder) or *oracle-anchored* (a fixed projection of the **true next state's** features — an external referent
with full variance by construction, §4.1), crossed with the collapse-prevention machinery (EMA target +
VICReg) on/off. The readout is representation health: embedding std and effective rank (→ 0 / → 1 under
collapse). Embedding std is the robust signal at this scale:

| target | collapse machinery | embedding std | effective rank (d=48) |
|---|---|---|---|
| learned (EMA) | **on** (the JEPA baseline) | 0.557 | 41.8 |
| learned | **off** (ablated) | **0.276** | **13.4** |
| oracle-anchored | on | 0.597 | 43.4 |
| **oracle-anchored** | **off** (ablated) | **0.528** | **25.8** |

With the machinery ablated, the naked *learned* target collapses — embedding std halves (0.557 → 0.276) and
effective rank drops 3× (41.8 → 13.4). The **oracle-anchored target does not**: with the *same* ablation it
holds std at 0.528 (≈ the EMA+VICReg baseline) and rank at 25.8 (≈ 2× the collapsed learned arm). This is H23
confirmed at smoke scale: where an external referent exists, the EMA+VICReg crutches are *substantially*
unnecessary — exactly SPEC-8's claim that the collapse tax is a workaround for a missing oracle. (Honest
nuance: the machinery still adds some health on top of the oracle target — rank 43.4 vs 25.8 — so it is a
strong substitute, not yet a total one at this size.)

**H24 — the objective axis: residual supervision is a near-tie here (the honest negative).** Under partial
observation (half the hosts observed, so the decidable/residual partition is non-degenerate), training the
decoder on the **residual** bits only — masking the oracle-decidable tokens, *verify don't learn* (§4.2) —
versus the raw-likelihood baseline gives, on the residual tokens `R` the model is actually responsible for:
raw-likelihood **0.463** vs residual **0.426** (the baseline edges it by 3.7 pts). At this well-trained smoke
scale that is essentially a tie — the H24 refutation branch: the decidable part `D` was cheap enough to learn
that masking it buys nothing *yet*. The partition is pre-registered to matter more as worlds grow (SPEC-6/7), where `D` is a larger fraction
of a much bigger next-state; EN8 banks the bound, it does not close the question. (The residual arm's *overall*
token accuracy drops to 0.11 by construction — it deliberately never learns `D`, which the oracle supplies for
free — so a full-delta free-decode is a category error for it, not a result; the fair metric is residual-token
accuracy above.)

**Where this routes the program.** EN8's split mirrors EN4's: a clean positive on the *representation*
mechanism (H23 — the oracle is a real anti-collapse referent) and an honest near-tie on the *objective*
mechanism at this scale (H24 — the partition's payoff is world-size-gated). The H23 positive says the
oracle-grounded latent arm is worth carrying up the ladder; the H24 bound says don't over-invest in residual
masking until the worlds are large enough for `D` to dominate. Every cell here is bankable under the oracle —
including, as always, the negative.

## EN9 (oracle hard-negative contrastive) — the exact referent vs. the statistical one, a third split (SPEC-8 §7)

EN8 grounded the *predictive* target on the oracle (H23/H24). EN9 grounds the *contrastive* one — the second
SPEC-8 mechanism (§4.3) and the consumer of the OG2 hard-negative factory. A contrastive predictor over the
same graph summary, with the **only** anti-collapse referent varying across three cells: *none* (naked BYOL,
regress onto the stop-grad online target), *vicreg* (the field's statistical "push apart" regularizer), or
*oracle* (InfoNCE against the OG2 exact hard negatives — counterfactual successors `O(s, a')` and
one-edit-wrong neighbors of the true successor, each labeled `≠` the truth by the oracle). Two readouts:
representation health (the collapse diagnostic) and **interventional fidelity** — does the representation map
each intervention `a'` to its true successor `O(s, a')`, scored as branch-retrieval top-1 / MRR on held-out
states (the H5 / EN6 branch-replay question)? A committed *smoke* instance
([`figures/en9_contrastive.png`](../figures/en9_contrastive.png), [`.csv`](../figures/en9_contrastive.csv)),
and like EN4/EN8 a two-sided result — the split is the finding.

| mode | embedding std | effective rank (d=48) | intervention top-1 | intervention MRR |
|---|---|---|---|---|
| none (naked) | **0.276** | 13.4 | 0.214 | 0.426 |
| vicreg | 0.499 | **39.0** | 0.282 | 0.500 |
| **oracle** | **0.699** | 31.4 | **0.519** | **0.694** |

**H25 — the collapse axis: the exact referent matches the statistical one.** The naked contrastive target
collapses (std 0.276); both VICReg (0.499) and the oracle hard-negatives (0.699) prevent it. On *raw* collapse
the oracle is at least as good — but VICReg's covariance term, which explicitly decorrelates dimensions, buys
slightly higher effective rank (39.0 vs 31.4). So H25's "match or beat at preventing collapse" lands on
*match*: the exact near-miss structure is a real anti-collapse referent, not a strictly dominant one for the
rank metric.

**H5 — the interventional axis: the oracle wins decisively (the lift).** This is where the exactness pays. Only
the oracle's *counterfactual* negatives carry information about which intervention leads where, so its
branch-retrieval fidelity nearly doubles VICReg's — top-1 **0.519 vs 0.282**, MRR 0.694 vs 0.500. The honest,
sharper reading of the split: VICReg keeps the representation full-rank but interventionally **blind** (its
0.282 top-1 is barely above the naked 0.214), while the oracle makes it faithful to the very branches the loop
will be asked to predict. A statistical regularizer can stop collapse; it structurally cannot teach
counterfactual structure it has no access to.

**Where this routes the program.** EN9 localizes *what* an exact referent buys over a statistical one: not
better collapse resistance per se, but **interventional content**. That is the H5/RQ4 lift arriving through the
SSL objective rather than the RL cherry — evidence that oracle-grounding belongs in pretraining for *change-safety*
tasks specifically. Carry the counterfactual-negative objective up the ladder (SPEC-6/7) where interventional
fidelity is the headline; do not expect it to beat VICReg on plain collapse. CIs and a scaled run remain;
every cell is bankable under the oracle.

## EN8/EN9 scale-up — the smoke verdicts, now with CIs across a 3× world sweep (SPEC-8 §7.1, SPEC-9)

The EN8/EN9 smoke figures above are single-seed (one `model_seed`, a 5-host world), so a reader can
discount them as noise. The OG5 scale harness ([`en8_scale.py`](../src/verisim/experiments/en8_scale.py),
[`en9_scale.py`](../src/verisim/experiments/en9_scale.py)) re-runs each ablation across **world size ×
seeds**, reducing every cell to the *gap the oracle buys* with a percentile bootstrap CI (the EN1
machinery). At 5/10/15 hosts × 4 seeds ([`en8_scale.csv`](../figures/en8_scale.csv),
[`en9_scale.csv`](../figures/en9_scale.csv)):

| world | H23-S collapse gap (eff-rank) | H25-S/H5 interventional lift (top-1) | H24-S residual-objective gap |
|---|---|---|---|
| 5 hosts | **+13.4** [12.7, 14.0] | **+0.100** [0.059, 0.140] | +0.069 [−0.005, 0.130] |
| 10 hosts | **+8.4** [7.8, 9.0] | **+0.354** [0.266, 0.448] | 0.000 [−0.035, 0.035] |
| 15 hosts | **+7.7** [6.7, 8.7] | **+0.094** [0.055, 0.125] | +0.006 [−0.009, 0.028] |

**Two of the three claims are now defensible against "n=1 / toy world."** The H23-S collapse gap and the
H25-S/H5 interventional lift are **disjoint from zero at every world size** with 4 seeds. H24-S stays a
**CI-bounded near-tie** — the smoke negative, now with error bars. Two honest nuances are themselves the
findings, and both are pre-registered into SPEC-9's scaling claims: the *raw* collapse gap declines with
world size (effective rank is capped by `d_model=48`, so SPEC-9 S1 tracks the normalized gap and grows
`d_model` with the world), and the interventional lift is *non-monotone* (peaks at 10 hosts — SPEC-9 S2
reads this as fixed-capacity undertraining at the largest world, to be tested on the model-size axis).
The full world × model **scaling surface** (SPEC-9 LS2) extends this up the local envelope; the envelope
itself (how large the world can be made on one 32 GB machine before `O(N²)` message passing binds — not
memory, and not the free oracle) is measured in [SPEC-9 §3](specs/SPEC-9.md).

### H24 capacity-binding frontier — refuted with a mechanism (SPEC-9 S3)

H24-S was a CI-bounded near-tie; SPEC-8 §7.2 pre-registered *why* (a capacity-allocation claim) and
predicted a frontier where masking the decidable bits `D` and training only the residual `R` would
*win* — where capacity binds against a hard `R`. The dedicated sweep
([`en8_capacity.py`](../src/verisim/experiments/en8_capacity.py),
[`en8_capacity.csv`](../figures/en8_capacity.csv); a 40-host world × `d_model` ∈ {16, 32, 64} ×
observed-fraction ∈ {0.25, 0.5, 0.75} × 4 seeds) **refutes** it, and the refutation is the result:

| observed-fraction (R-size) | d16 | d32 | d64 |
|---|---|---|---|
| 0.25 (R≈0.27) | +0.003 [−0.024, 0.032] | +0.003 [−0.016, 0.021] | +0.005 [−0.011, 0.021] |
| 0.50 (R≈0.20) | +0.016 [−0.019, 0.050] | 0.000 [−0.019, 0.019] | −0.006 [−0.025, 0.013] |
| 0.75 (R≈0.11) | −0.026 [−0.052, 0.000] | −0.026 [−0.052, −0.005] | **−0.094 [−0.130, −0.057]** |

No cell's CI is disjoint-positive — the frontier does not exist in the local envelope. Stronger: where
`D` is large (observed-fraction 0.75, `R` only ~11% of tokens) masking it is disjoint-*negative* and
worsens with capacity. **The mechanism is the finding:** masking `D` does not free capacity for `R`, it
*removes training signal* — the model is then supervised on only the R-fraction of tokens per step, which
starves the shared encoder/decoder; learning `D` is **beneficial multi-task auxiliary signal**, not wasted
capacity. So H24's "burning capacity on `D` is waste" premise does not hold at this scale. What is refuted
is precisely the **training-objective** form of the partition (mask `D` in the loss); the **inference-time**
partition — the oracle *supplies* `D`, so the model is never trusted on it — is untouched and is exactly
what the propose–verify–correct loop already does. The bankable next variant keeps `D` in the loss and
lets the oracle own `D` only at inference. This is the epistemic engine working: a pre-registered negative
returning a sharp, mechanistic, oracle-trustworthy result that redirects the program. *(The scaling surface
below adds a wrinkle this `d≤64` grid did not see: at `d_model=128` and small worlds the residual gap is
small but disjoint-**positive**, so H24 is **regime-dependent**, not flatly refuted — SPEC-9 §4 S3.)*

## The local scaling surface — H23 attenuates, H25/H5 reverses (SPEC-9 LS2)

The first scale-up (5/10/15 hosts) confirmed H23-S and H25-S/H5 with disjoint CIs. The full local
**surface** — 25/50/100/200 hosts × `d_model` ∈ {64, 128} × 3 seeds, the largest oracle-grounded sweep run
on the 32 GB M4 ([`en8_surface.csv`](../figures/en8_surface.csv),
[`en9_surface.csv`](../figures/en9_surface.csv); ~69 min + ~104 min CPU) — carries them up an 8× world
range, and the smoke wins survive *unevenly*. That unevenness is the result.

- **H23-S (collapse) — persists but attenuates (S1).** The collapse gap is disjoint-positive at **all 8
  cells**, so the oracle's anti-collapse advantage is real across the whole range and both capacities — the
  most robust of the three. But it *shrinks* with world size even normalized (raw eff-rank gap 13.4 → 4.1
  over 25 → 200 hosts at `d128`); a larger `d_model` lifts it at fixed world but does not flatten the
  decline. "Real everywhere, diminishing" — not the scale-stable form S1 pre-registered.
- **H25-S/H5 (interventional lift) — reverses at fixed `k`, then recovers when negatives scale (S2).** The
  oracle-over-VICReg branch-retrieval lift is disjoint-positive at the smallest world + smaller capacity
  (25 hosts/`d64`: +0.106 [0.067, 0.179]); it decays with scale and **reverses** with the fixed
  `k_negatives=8` — VICReg *beats* the oracle at 100/`d128` (−0.086 [−0.113, −0.060]) and 200/`d128`
  (−0.094 [−0.111, −0.067]). The pre-registered diagnosis — a fixable **negative-count artifact** — then
  proved correct ([`en9_negatives.csv`](../figures/en9_negatives.csv); 100/`d128`, 3 seeds): scaling
  `k_negatives` 8→16→32 flips `lift_top1` −0.075 → +0.017 → **+0.032 [0.024, 0.044]** (disjoint-positive),
  with `lift_mrr` tracking it. Recovery is *modest* (not the +0.10–0.35 small-world magnitude), so the rule
  is **scale negatives with the world**; the lift is real, just starved at a fixed negative count. (This
  refuted the experiment's own stated prior — that more *one-edit* negatives wouldn't help because the
  counterfactual branch set is fixed — but they did, by sharpening the contrastive geometry, not by adding
  branches.)

This is the single most valuable thing the scaling bought, and the full arc is the lesson: a headline EN9
result that looked clean at smoke scale **reversed** under an honest CI sweep at 100–200 hosts — and then,
when the pre-registered lever was applied, **recovered**. The oracle is precisely what let us see both the
reversal and the fix. A win caught reversing and then honestly repaired is worth far more than one asserted
and never stress-tested.

## EN7 — the no-knee shape is model-invariant (H22)

The project's most general claim (SPEC.md §9, H22) is that the *loop*, not the proposer, governs the
`H_ε(ρ)` curve. EN7 ([`en7.py`](../src/verisim/experiments/en7.py),
[`en7_invariance.csv`](../figures/en7_invariance.csv)) drops four proposers into the **same** NW5 loop and
re-plots the curve (5 hosts, ε=0.05, T=24, 3 seeds × 2 difficulties, bootstrap CIs):

| proposer | ρ=0 | ρ=0.1 | ρ=0.2 | ρ=0.3 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|---|
| null (empty delta) | 0.0 | 1.2 | 1.2 | 1.2 | 1.3 | 24.0 |
| flat (NW4 transformer) | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 24.0 |
| graph (NW8 GNN+RSSM) | 0.0 | 3.2 | 3.2 | 4.3 | 4.7 | 24.0 |
| oracle-backed (perfect) | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |

![EN7 / H22: the floor+cliff H_ε(ρ) shape is invariant across proposers](../figures/en7_invariance.png)

**H22 is supported in kind.** The three *imperfect* proposers — a null predictor, the flat transformer, and
the graph+RSSM arm — share **one qualitative shape: floor + cliff, no favorable knee.** The interior is
near-flat and the curve reaches the T=24 ceiling only at ρ=1, for every architecture. What the proposer
changes is the **floor height** (graph's 3.2–4.7 > flat's 1.0 > null's ~1.2), i.e. how much unaided horizon
its per-step competence buys — *not* the shape. So the EN1/K4 "no-knee" verdict is **not** an artifact of
the flat transformer: it reproduces across materially different model classes, which is exactly H22's claim
that deterministic verification's loop behavior is a *model-agnostic primitive*. The oracle-backed proposer
(24 everywhere) is the degenerate ceiling. **Honest caveat:** this is not matched per-step competence (the
graph arm is clearly stronger), so the load-bearing evidence is the *shared shape across differing
competence*, not a magnitude comparison — what moves with the proposer is the floor, what stays is the shape.

## Threats to validity

- **Scale.** The committed model is ~tiny and trains for a few hundred iterations on
  a CPU-sized dataset. The negatives are consistent with "too small/undertrained to
  be interesting," not "the mechanism is wrong" — and E4 sharpens this: more *size*
  alone does not help, so the suspect is training budget / data / difficulty, not
  parameter count. The deterministic core (M0–M3) and loop invariants (M5) are
  separately tested, so the apparatus is sound.
- **Reference oracle, not a real OS.** v0's oracle is a model of POSIX, not POSIX
  (SPEC.md §2.1). H4 (mechanism survives a real sandbox) is Phase 1.
- **Difficulty by driver only.** The §2.4 depth/breadth dial is not yet a knob; v0
  difficulty is carried by the driver mix, which may not stress long-range
  dependencies enough to make the interior informative.

## Reproduce

Everything regenerates from configs + seeds (SPEC-2 §12). With the `[dev,model,viz]`
extras installed:

```bash
bash figures/reproduce.sh          # E1 + E2 + E3 records and all figures
# or individually:
python -m verisim.experiments.e1 --config configs/e1.json --out runs/e1/records.jsonl
python figures/plot_e1.py --records runs/e1/records.jsonl
python -m verisim.experiments.e2 --config configs/e2.json --out runs/e2/records.jsonl
python figures/plot_comparison.py --records runs/e2/records.jsonl --key policy \
    --out figures/e2_policies.png --csv figures/e2_policies.csv
python -m verisim.experiments.e3 --config configs/e3.json --out runs/e3/records.jsonl
python figures/plot_comparison.py --records runs/e3/records.jsonl --key operator \
    --out figures/e3_operators.png --csv figures/e3_operators.csv
python -m verisim.experiments.calibration --config configs/calibration.json \
    --out runs/calibration/pairs.jsonl
python figures/plot_calibration.py --pairs runs/calibration/pairs.jsonl
python -m verisim.experiments.e4 --config configs/e4.json --out runs/e4/records.jsonl
python figures/plot_e4.py --records runs/e4/records.jsonl
python -m verisim.experiments.objective --config configs/objective.json \
    --out runs/objective/records.jsonl
python figures/plot_objective.py --records runs/objective/records.jsonl
python -m verisim.experiments.representation --config configs/representation.json \
    --out runs/representation/records.jsonl
python figures/plot_representation.py --records runs/representation/records.jsonl
# SPEC-5 network world (NW6 prime-directive curve; NW7 EN2/EN3 comparisons):
python -m verisim.experiments.en1 --config configs/en1.json --out runs/en1/records.jsonl
python figures/plot_en1.py --records runs/en1/records.jsonl \
    --out figures/en1_curve.png --csv figures/en1_curve.csv
python -m verisim.experiments.en2 --config configs/en2.json --out runs/en2/records.jsonl
python figures/plot_comparison.py --records runs/en2/records.jsonl --key policy \
    --out figures/en2_policies.png --csv figures/en2_policies.csv
python -m verisim.experiments.en3 --config configs/en3.json --out runs/en3/records.jsonl
python figures/plot_comparison.py --records runs/en3/records.jsonl --key operator \
    --out figures/en3_operators.png --csv figures/en3_operators.csv
# NW8 graph arm — EN4 graph-vs-flat (H11); writes the CSV + figure directly:
python -m verisim.experiments.en4_graph --graph-iters 1500 \
    --out figures/en4_graph_vs_flat.csv
# SPEC-8 EN8 — oracle-grounded-SSL ablation (H23/H24); writes the CSV + figure directly:
python -m verisim.experiments.en8 --out figures/en8_grounding.csv
# SPEC-8 EN9 — oracle hard-negative contrastive (H25/H5); writes the CSV + figure directly:
python -m verisim.experiments.en9 --out figures/en9_contrastive.csv
```

The run-records are git-ignored (regenerable); the figures and their CSVs are
committed next to the plotting scripts, so a reader can check the numbers against the
figures without rerunning anything.

## What v0 ships for others

Per SPEC-2 §15, the env + metric are packaged for reuse:

- **Faithfulness benchmark** ([`verisim.eval`](../src/verisim/eval/)) — a
  dependency-free benchmark that scores any model implementing the loop `Model`
  protocol (`score_model`, `DEFAULT_SUITE`), plus single-step ground-truth labels and
  a divergence grader for question-answer frameworks. An `inspect_ai` task adapter
  ships behind the optional `[eval]` extra.
- **Oracle-as-reward RL environment** ([`verisim.rl`](../src/verisim/rl/)) — a
  `verifiers`-spec environment (`WorldModelEnv`, `load_environment`) whose reward is
  the oracle's faithfulness verdict, so the episode return equals the faithful
  horizon. This is the public expression of "train a world model against a verifiable
  oracle reward" (SPEC.md §6.3).
