# SPEC-2.1 — Earning the Knee

**A focused v0 engineering spec with one job: lift the clean (`ρ=0`) model above the
faithfulness floor so the `H_ε(ρ)` curve finally shows an interesting knee — or prove,
with evidence, that the single-filesystem world cannot produce one.**

This is not a new world. It is the spec that goes *back* and finishes the open M6 work
([SPEC-2.md §13, §17.5](./SPEC-2.md)) before anything else is built. It supersedes the
"tuning ongoing" status of M6 and **pauses the roadmap specs
([SPEC-3](./SPEC-3.md)/[SPEC-4](./SPEC-4.md)/[SPEC-5](./SPEC-5.md)/[SPEC-6](./SPEC-6.md)/[SPEC-7](./SPEC-7.md))
until the knee exists** — they are designs we have earned the right to *think* about but
not to *build*, because the v0 result is currently a null (the model drifts at step 0,
[docs/report.md](../report.md)). One real knee is worth more than four unbuilt worlds.

> **Reading order.** Prereq: [SPEC.md](./SPEC.md) (the science — what `H_ε(ρ)` is and why
> a knee matters) and [SPEC-2.md](./SPEC-2.md) (the v0 build this completes). The roadmap
> specs are explicitly *out of scope here* and on hold.

**License:** MIT. **Status:** active build spec — the *only* active spec. **Audience:** the
author, and the autonomous ratchet ([`src/verisim/auto/`](../../src/verisim/auto/)) that
will do most of the search.

---

## 0. Prime directive

> **SPEC-2.1's only job is to produce one `H_ε(ρ)` curve, on the existing single-filesystem
> world, that shows a *knee*: faithful horizon materially above the floor at a small
> consultation budget. The interim target is ≥50% of the `ρ=1` ceiling horizon at `ρ≤0.2`
> with non-overlapping CIs against the `ρ=0` floor; the stretch is SPEC.md H1's ≥80%.
> If no `(data, training, difficulty)` configuration achieves it after the search, that is
> the result, and it is what licenses the harder worlds.**

Everything below serves that one figure. No new hypotheses, no new world, no new
abstraction. This spec *removes a confound* (the under-trained model) so that H1/H2/H3 can
finally be tested honestly, and it tunes one dial (difficulty) that v0 deferred.

---

## The build order — concrete action items, in sequence

Do these in order; each gate must pass before the next. This is the whole of the near-term
build process. The cross-spec gating (what comes after the knee) lives in
[SPEC §12](./SPEC.md#12-research-roadmap).

- [x] **K0 — Diagnose** (§4). ✅ **Done.** The trivial-world control reaches **exact 1.0**
  (gate 0.95 → PASS): the pipeline provably fits a deterministic transition. Diagnostics
  localize the floor to a **generalization failure** (baseline train acc 1.0 vs val 0.083)
  whose mechanism is **exact multi-token argument (path) copying into the delta** (`create`
  precision/recall 0) plus mispredicted **failure/collision** cases (`exit` dominates
  divergence) — not capacity, not a broken learner. See [docs/report.md](../report.md) §K0,
  `verisim.experiments.k0`, `configs/k0.json`, `figures/k0_control.*`. *Bits-to-correct
  (SPEC-2.1 §3) shipped as the smooth gate.*
- [x] **K1 — Data at coverage** (§5). ✅ **Done.** `verisim.data.coverage` reports coverage of
  the transition space; the committed run spans **all 13 commands, 273 failure cells, and
  create-depths 1→8** (the failure cases *and* the multi-segment path-copy distribution K0
  flagged). Hard-negative mining (`mine_hard_negatives`, active learning over bits-to-correct)
  shipped and tested. *Gate met.*
- [x] **K2 — Train properly** (§6). ✅ **Done — and the copy bottleneck dissolved.** Training the
  same 2×128 model on the copy distribution (2,560 transitions) with `train_batched` (6,000
  steps) lifts clean held-out faithfulness on the non-trivial `structural` world from K0's
  ~0.09 to **exact 0.859 / acceptance@0.05 0.875 / graded 0.988 — gate 0.5 → PASS**. The K0
  bottleneck was coverage/training, *not* representation (the K3 copy-rep lever is unneeded at
  this scale). See [report §K1+K2](../report.md), `configs/k2.json`, `figures/k2_faithfulness.*`.
- [x] **K3 — Difficulty sweet-spot** (§7). ✅ **Done.** The SPEC-2 §2.4 dial is implemented as
  `max_depth` on the driver (`path_depth`, threaded through `build_dataset`/`eval_actions`). At
  the sweet spot (`structural`, `max_depth=4`, `T=48`) the K2 model's acceptance is ~0.875 and
  the ρ=0 horizon is ~10/48 — room below the ceiling. *Gate met.*
- [x] **K4 — Earn the knee** (§8). ✅ **Done — honest negative (C-knee refuted on single-FS).**
  The competent model lifts the ρ=0 floor from ~0 to **~10/48**, but `H_ε(ρ)` is **floor+cliff,
  not a knee** (ε=0.05: 10.3 → 11.3 at ρ=0.2 → 48 at ρ=1), and smart consultation does **not**
  beat fixed (E2: fixed 11.3 vs uncertainty 10.7 vs drift 10.9, overlapping CIs — H2 negative
  persists). *Mechanism:* filesystem errors are **discrete** (one wrong edit spikes the
  set-difference past ε; first-exceedance `H_ε` can't be reset-extended) and decode-entropy
  **doesn't localize errors**. See [report §K3+K4](../report.md), `figures/k4_knee.*`,
  `figures/k4_policies.*`. Per §10 this **licenses SPEC-5** (the knee needs gradual drift +
  calibrated belief, which networks have and the discrete fully-observable filesystem does not).
- [x] **Engine** (§9). bits-to-correct (SPEC-2.1 §3) shipped as the smooth gate and used by the
  K1/K2 hard-negative mining; the autoresearch ratchet space extension is available behind the
  same gate.

**SPEC-2.1 is complete.** The learner is proven (K0), the floor is gone (K1/K2: ~0 → competent),
and the knee hunt returned a clean, mechanistic negative (K3/K4) that — exactly as §10 planned —
**licenses the network world**. The next stage is **SPEC-5** (the network `H_ε(ρ)` curve), where
drift is gradual and partial observability supplies a calibrated signal — *not* more single-FS
tuning. (The system oracle, SPEC-3 S1, remains the orthogonal "faithful-to-reality" upgrade.)

---

## 1. The diagnosis: why v0 sits on the floor

The v0 report ([docs/report.md](../report.md)) is honest about the symptom — clean per-step
accuracy ~0.1–0.2, `H_ε≈0` at `ρ=0`, a flat interior. The code makes the *cause* concrete,
and it is not what the roadmap specs assumed.

| Suspected cause | Verdict | Evidence |
|---|---|---|
| Model too small (capacity) | **Ruled out** | E4: scaling `1×32 → 4×128` left accuracy in 0.09–0.22 ([report §E4](../report.md)). |
| World too easy (no drift to study) | **Not the binding cause** | The model can't predict step 0 — it never gets far enough for "too easy" to matter. |
| **Too little data + too little training** | **The smoking gun** | [`configs/e1.json`](../../configs/e1.json): ~**160 transitions** (`train_seeds:[0,1,2,3] × 40 steps`), `train_iters:500`, **full-batch GD** ([`train/supervised.py`](../../src/verisim/train/supervised.py)), one driver, no minibatch / schedule / early-stop / coverage. The M4 test shows this same pipeline overfits *tiny* data to teacher-forced accuracy **1.0** — so it memorizes train and fails to generalize. That is a data/optimization gap, not a capacity wall. |
| Difficulty dial never implemented | **True, and now needed** | [SPEC-2 §2.4](./SPEC-2.md)'s depth/breadth/destructive-fraction knob was deferred (M2 note); v0 difficulty is "driver mix only," so we cannot tune the world to the regime where a competent model still drifts. |

**The step-0 puzzle, resolved.** It looks paradoxical that the world is "too easy" *and* the
model "drifts at step 0." It is not: a deterministic filesystem transition is easy *to learn*
given coverage, and the model has been given almost none. Teacher-forced **token** accuracy
of 0.15 on a deterministic mapping is the signature of a model that has not seen enough of the
transition space to generalize — not of a hard task. Fix the data and training, and step-0
prediction should approach near-perfect on easy difficulties. *That* is the bet, and it is far
more grounded than "make the world harder."

---

## 2. The target and the theory: cross the acceptance floor

SPEC-3 §8 imported the decisive result from speculative decoding (HW-3): **below ~0.5
per-step acceptance, propose-verify is net-negative — there is no favorable interior.** v0
sits at ~0.1–0.2 acceptance, so a flat interior is *exactly what the theory predicts*. The
knee is not missing because the loop is wrong; it is missing because the draft model is below
the floor the loop needs.

So the North Star is a single number:

> **Clean (`ρ=0`) per-step faithfulness `a = E[1 − d(s_t, ŝ_t)]` (teacher-forced) must cross
> ~0.5, and ideally land in the 0.7–0.95 "competent-but-compounding" band.**

Why the band, not just "as high as possible":
- At per-step acceptance `a`, the unaided faithful horizon is roughly geometric, `H ≈ 1/(1−a)`:
  `a=0.5 → ~2 steps`, `a=0.9 → ~10`, `a=0.95 → ~20`.
- For an *interesting* knee you want the `ρ=0` floor well below the `ρ=1` ceiling (room to
  buy back) **and** acceptance high enough that a cheap consultation, by resetting drift,
  extends the horizon a lot. With `T=24`, `a≈0.85–0.95` is the sweet spot: floor ≈ 7–20 steps,
  ceiling 24, and even-spaced consultations should lift the interior sharply.
- If the well-trained model overshoots to `a≈0.99` the rollout barely drifts and there is no
  curve to draw — so **difficulty is tuned down or up to land `a` in the band** (Phase K3).

This is the SPEC-2 §17.5 difficulty co-tuning, but done for the first time *with a model that
can actually learn the world*. E4 tuned difficulty with an under-trained model and so could
never find the band; that is why E4 read as "size doesn't help" rather than "data does."

---

## 3. Metric upgrade: bits-to-correct as the smooth gate

The search needs a smooth, comparable scalar to climb. Clean per-step accuracy is 0/1 per
token and `H_ε` is flat at this scale — both are poor optimization signals (the report says as
much). Pull **bits-to-correct** forward from [SPEC-3 §7](./SPEC-3.md) into v0 now:

- `bits_to_correct(s,a)` = MDL of the oracle's correction of the predicted delta, under the
  simple prefix code over the delta grammar (the symmetric difference of predicted vs. true
  delta, in bits). **0 iff the prediction equals the truth**, smooth otherwise, scale-free.
- It is a direct, cheap generalization of the existing divergence metric
  ([`metrics/divergence.py`](../../src/verisim/metrics/divergence.py)) from a ratio to a bit
  count — a small addition to `metrics/`, not a new subsystem.
- **`H_ε` stays the headline science** (the knee is reported in `H_ε(ρ)`); bits-to-correct is
  the *optimization gate* the ratchet (§9) climbs (SPEC-3 DD-6). The autoresearch gate upgrades
  from "mean clean accuracy" to "mean clean bits-to-correct↓."

---

## 4. Phase K0 — Diagnose before tuning (do not skip)

Science before search: *measure where the model fails* so the fix is targeted, not a blind
sweep. Instrument the existing eval to emit, on held-out trajectories:

- **Per-command accuracy** — which of the §2.2 commands does the model get right/wrong?
  (Hypothesis: `mv`/`rm -r`/branching failures dominate; structure-building writes are easier.)
- **Per-edit-type precision/recall** on the predicted delta (the metric already exists as a
  §7.1 diagnostic) — is it missing edits, inventing them, or mis-typing them?
- **Accuracy vs. trajectory position** and **vs. state size** — does competence decay with
  depth (a memory/context problem) or with breadth (a coverage problem)?
- **Train vs. val gap** — confirm the memorize-train/fail-val signature directly.

**The decisive K0 control experiment.** Take a *trivially easy* difficulty (1–2 commands, no
cascading, shallow tree), generate **plenty** of data, train to convergence, and measure clean
val faithfulness.

> **K0 gate:** on the trivial world, clean per-step faithfulness ≥ **0.95**. *This proves the
> pipeline can learn the transition function at all.* If it cannot — if even a trivial
> deterministic world with abundant data stays on the floor — the cause is representation /
> tokenization / optimization (e.g., the serialized-DSL makes exact deltas too brittle), and
> *that* bug is fixed first, before any difficulty tuning. This gate protects against spending
> the whole search on a pipeline that structurally cannot learn.

**Result (2026-05) — gate PASS, with a sharp finding.** The control reaches **exact 1.0** on
the depth-1 trivial world (`verisim.experiments.k0`, `configs/k0.json`,
[`figures/k0_control.*`](../../figures/), [report §K0](../report.md)). The pipeline works. The
diagnostics localize the v0 floor precisely: the baseline **memorizes train (1.0) but fails val
(0.083)** — an under-coverage generalization gap, not capacity (E4) or a broken learner — and
the residual is **exact multi-token argument (path) copying into the delta** (`create`
precision/recall = 0) plus **mispredicted failure/collision cases** (`exit` dominates
divergence). A convergence probe showed 25× more training does *not* dissolve the copy residual
(observation facts become fully learned; the created-node identity does not), while one-token
copies reach 1.0 — so the bottleneck is *copying*, not *steps*. This is the diagnosis K1/K2 now
target.

---

## 5. Phase K1 — Data at coverage (the oracle is a free, infinite teacher)

The single highest-leverage fix (SPEC-3 §5.1, pulled forward). The reference oracle generates
unlimited, perfectly-labeled transitions at zero cost — there is no reason to train on 160.
**K0 made the targets concrete:** oversample the two things the baseline fails — the
**path-copy distribution** (deep multi-segment paths the model must reproduce exactly in a
`create`/`modify`) and **failure/collision cases** (`mkdir`/`rmdir`/`rm` on existing/missing/
non-empty targets, where the model wrongly predicts success). If coverage alone does not lift
exact-match, the K0 finding flags a representation lever for K3 — a *copy-aware* delta that
references action arguments by pointer rather than re-emitting path tokens.

- **Scale up.** Generate thousands–tens of thousands of transitions (hundreds of trajectories ×
  longer rollouts), held-out splits by trajectory (the discipline already exists in
  [`data/generate.py`](../../src/verisim/data/generate.py)).
- **Coverage-balanced synthesis.** Generate to *cover the transition space* — every command ×
  the relevant pre-state shapes (empty dir, non-empty dir, deep path, missing path, permission
  failure) — not to mimic one driver's marginal distribution. Rare-but-decisive transitions
  (the failure modes where compounding error bites — `rmdir` on non-empty, `mv` of a subtree,
  `rm -r` cascade) are *oversampled*, because those are exactly where the model drifts.
- **Hard-negative mining (the active-learning loop the oracle makes free).** Periodically find
  the `(state, action)` where the *current* model is wrong (bits-to-correct high) and add those
  to the training set. The oracle labels them for free; this targets data where it is needed.

> **K1 gate:** the training set covers every command × pre-state-shape class with a documented
> minimum count, and is regenerable from a manifest (the §12 discipline). No metric gate yet —
> K1 is the input to K2.

**Result (2026-05) — gate met.** `verisim.data.coverage` produces a records-only coverage report
(command × {ok,fail} cells + a create-depth histogram). The committed K2 run spans **all 13
commands, 273 failure cells, and create-depths 1→8** — the two regions K0 flagged (failures +
the multi-segment copy distribution). **Hard-negative mining** (`mine_hard_negatives`) is
implemented and tested: the active-learning loop the oracle makes free (rank transitions by
bits-to-correct, retrain on the worst). It is available behind a config knob; the K2 gate is met
without it, so the committed run leaves it off.

---

## 6. Phase K2 — Train properly

The current full-batch-160-for-500-steps loop ([`train/supervised.py`](../../src/verisim/train/supervised.py))
is a placeholder that the M4 verify ("overfit a tiny set") was written around. Make it a real
training loop — a small, surgical change to `train/`, no new module:

- **Minibatch SGD** over the K1 dataset (not full-batch).
- **LR warmup + cosine decay**; tune `lr`, batch size, and total steps as first-class knobs.
- **Validation-based early stopping / best-checkpoint selection** on held-out bits-to-correct —
  so we report the *generalizing* model, not the most-overfit one.
- **Enough steps to converge** the val metric (hundreds → tens of thousands of steps, as the
  data demands; E4 already showed `train_iters↑` was the ratchet's first win).
- Keep determinism (single-thread, seeded — the existing `train_model` discipline).

> **K2 gate:** clean (`ρ=0`) per-step faithfulness on **held-out** trajectories crosses
> **0.5** on at least one non-trivial difficulty — i.e. the model clears the acceptance floor
> (§2). This is the gate that, per speculative-decoding theory, makes a knee *possible*.

**Result (2026-05) — gate PASS, and the K0 bottleneck dissolved.** The same 2×128 model trained
on the copy distribution (the `structural` driver: collision-free multi-depth creates, 2,560
transitions) with `train_batched` (minibatch + warmup/cosine + val-early-stopping, 6,000 steps)
reaches, on the **held-out non-trivial `structural` world**: **exact 0.859, acceptance@ε=0.05
0.875, graded 0.988** — gate 0.5 → **PASS** (`verisim.experiments.k2`, `configs/k2.json`,
[`figures/k2_faithfulness.*`](../../figures/), [report §K1+K2](../report.md)). K0's copy floor
(~0.09 at 768 transitions / 4,000 steps, unmoved by more steps alone) was a **coverage/training**
problem, not a representation wall — so the K3 copy-representation lever is unneeded at this
scale. Decisively, **acceptance 0.875 puts the model inside the K3 "competent-but-compounding"
band (0.7–0.95)**: the unaided geometric horizon is ≈ 8 steps (~50% of a T=16 ceiling), exactly
the regime where the `H_ε(ρ)` knee is expected. K3/K4 are next.

---

## 7. Phase K3 — Find the difficulty sweet spot

Implement the deferred [SPEC-2 §2.4](./SPEC-2.md) **difficulty dial** as an explicit
`EnvConfig` knob (the one genuinely new piece of env work, justified because the knee needs it):
max tree depth, breadth, and fraction of destructive/cascading commands, plus rollout length
`T`. Then sweep difficulty with the K2-trained model to land per-step acceptance in the
**0.7–0.95 band** (§2):

- Too easy → `a≈0.99`, `ρ=0` horizon ≈ ceiling, no room for a knee → make it harder.
- Too hard → `a<0.5`, back on the floor → make it easier (or feed K1/K2 more coverage of that
  difficulty).

> **K3 gate:** a difficulty exists where the K2 model has per-step acceptance in `[0.7, 0.95]`
> **and** the `ρ=0` faithful horizon is materially below the `ρ=1` ceiling (room to buy back).
> This is the regime the knee lives in.

**Result (2026-05) — gate met.** The dial is `max_depth` on the driver
(`verisim.data.drivers.path_depth`, threaded through `build_dataset`/`eval_actions`,
`verisim.experiments.k4`). The committed sweet spot is `structural`, `max_depth=4`, `T=48`:
acceptance ~0.875 (in band) and ρ=0 horizon ~10/48 (room below ceiling). *Note:* the band itself
turned out to be **necessary but not sufficient** for a knee on this world — see §8.

---

## 8. Phase K4 — Earn the knee (re-run E1, then E2/E3 honestly)

With a competent model on a sweet-spot difficulty, re-run the existing experiments unchanged in
structure (`experiments/e1.py`, `e2.py`, `e3.py` already exist):

- **E1 (H1) — the curve.** Sweep `ρ × ε`, fixed policy, `hard_reset`. *Look for the knee.*
  > **K4 gate / the prime directive:** `H_ε(ρ)` shows a knee meeting the §0 interim target
  > (≥50% ceiling at `ρ≤0.2`, CIs disjoint from the `ρ=0` floor) — **or** an honest negative
  > with the §10 diagnosis.
- **E2 (H2) — policy.** Now the model is competent, its decode-entropy uncertainty should
  finally carry signal; re-run the calibration diagnostic (`experiments/calibration.py`) first
  and *only* expect triggered policies to win if Pearson has moved off ~0.11. H2 is tested
  honestly for the first time.
- **E3 (H3) — operator.** Still expected to be an identity under a *full*-state oracle (the v0
  theoretical result stands); the operators only diverge under **partial verification**, which
  is SPEC-3/SPEC-5 work and stays paused. Report E3 as the confirmed identity, not as a failure.

**Result (2026-05) — C-knee refuted on the single-FS world (the honest negative §10 anticipated).**
*E1:* the competent model lifts the ρ=0 floor from ~0 to **~10/48**, but `H_ε(ρ)` is **floor+cliff**
(ε=0.05: 10.3 → 11.3 at ρ=0.2 → 13.8 at ρ=0.5 → 48 at ρ=1) — the §0 knee target is **not** met.
*E2:* smart consultation does **not** beat fixed at equal budget (fixed 11.3 vs uncertainty 10.7
vs drift 10.9, overlapping CIs) — the H2 negative survives even for the competent model.
**Mechanism:** filesystem errors are *discrete* — one wrong edit spikes the set-difference past ε
in a single step, so first-exceedance `H_ε` is set by the *first* error's position and cannot be
reset-extended; and decode-entropy uncertainty does not localize which steps will err, so
error-targeting consultation cannot catch them. A fixed-interval knee would need acceptance ≈0.98,
which leaves no room. This is a property of *this world* (discrete, fully-observable, set-difference
metric), not a tuning miss — and per §10 it **licenses SPEC-5**.

---

## 9. The ratchet does the co-tuning (humans out of the loop, as intended)

K1–K3 are a search over `(data coverage, training budget, difficulty)` — exactly what the
shipped autoresearch ratchet ([`src/verisim/auto/search.py`](../../src/verisim/auto/search.py),
[`configs/auto.json`](../../configs/auto.json)) automates. Three changes, all small:

- **Upgrade the gate** from mean clean accuracy to mean clean **bits-to-correct↓** (§3) — a
  smoother climb.
- **Expand the search space** to the levers this spec adds: dataset size / coverage knobs,
  minibatch + schedule hyperparameters, and the K3 **difficulty dial** — not just `n_layer/lr`.
- **Keep the keep-if-better ratchet** and the oracle gate exactly as they are (the safety
  property: the judge is not a knob, SPEC-4 DD-AR2).

The first ratchet run already doubled the floor (0.042 → 0.094) on the *old* tiny space; with
data/training/difficulty in the space and bits-to-correct as the gate, this is the engine that
finds the knee overnight. This is the v0-scale realization of "the self-improving loop, gated
by reality" the program is ultimately about — and it keeps the human at the boundary (objective,
budget, safety), per SPEC-4 §8.

---

## 10. The falsifiable claim, and the honest negative

> **Working claim (C-knee, a sharpened test of H1):** there exists a `(data, training,
> difficulty)` configuration of the single-filesystem world at which clean per-step
> faithfulness exceeds the ~0.5 acceptance floor and `H_ε(ρ)` exhibits a knee (§0 target).

> **Outcome (2026-05): C-knee is REFUTED on the single-filesystem world (§8).** The first half
> held — clean per-step faithfulness reached 0.875 ≫ 0.5 (K2) — but no consultation policy
> (fixed *or* smart) produced a knee, because the world's *discrete* errors and *uncalibrated*
> uncertainty make first-exceedance `H_ε` reset-resistant. This is the §10 honest-negative path,
> and it **licenses SPEC-5** (see below).

**If C-knee holds:** H1 is supported at v0 scale, the apparatus is validated end-to-end, and
the harder worlds (SPEC-5+) inherit a *working* method rather than a hopeful one.

**If C-knee is refuted** — after the K0 gate passes (the pipeline *can* learn) and the ratchet
has searched data/training/difficulty thoroughly, no configuration yields `a>0.5` with room for
a knee — then the honest finding is: *in the single-filesystem world, faithful simulation costs
oracle calls roughly linearly; the world is too low-entropy for a learned draft to clear the
acceptance floor while still leaving room to buy horizon.* That is a publishable result, and it
is precisely what **licenses** moving to the network/host/distributed worlds (SPEC-5/6/7) —
which were designed for exactly this regime (combinatorial reachability, partial observability,
intractable oracle). The roadmap specs stop being a hopeful bet and become an *evidence-backed*
next step. Either way, v0.1 produces the result that should gate the rest of the program.

**This is the canonical worked example of the epistemic engine (SPEC.md §10.1).** K4's refutation is
exactly the kind of result the project is built to metabolize, and it is worth being explicit about *why
it was progress, not a setback*: (i) it is **bankable** — under the deterministic oracle, "no knee here"
is a fact about the single-filesystem world (its errors are *discrete*, so one wrong edit spikes the
set-difference past ε and first-exceedance `H_ε` is reset-resistant), not a suspicion about a broken
metric (SPEC.md §10.1 point 1); (ii) it was **pre-registered** — the §10 "if refuted" branch was written
*before* K4 ran, so the conclusion could not be reinterpreted after the fact; (iii) it **routed the
program forward** — it named the precise property the next world must have (gradual, continuous drift with
a *calibrated* uncertainty signal), which is the exact design center of SPEC-5's network world. A negative
that is bankable, pre-registered, and forward-routing is not a dead end; it is the most informative single
figure v0 produced, and it is *why* SPEC-5 exists as an evidence-backed step rather than a hope. The duty
is to keep iterating against the oracle — every curve, knee or cliff, tightens the next question. We do not
quit on a negative; we *spend* it.

---

## 11. After the knee: the honest direction to a world-simulation AI

The larger intent — a real, useful world-simulation AI for the community — is reachable, but
not as "a model that rivals LLMs." The realistic, valuable form is a **verifiable specialist
simulator that complements LLM agents**: the cheap, faithful "what-if" an agent consults before
it acts on a real computer. The staged path, *each step gated on the last and on evidence*, with
the roadmap specs staying paused until their predecessor pays off:

1. **The knee (this spec).** Prove a learned draft + budgeted oracle keeps a computer world
   faithful, cheaply, measurably.
2. **Faithful to *reality*, not to a model of it** — the system oracle (SPEC-3 §2, milestone S1):
   swap the reference interpreter for a real sandboxed shell, re-run the knee against a real OS.
   This is where "world simulation" stops being a model of POSIX and starts being POSIX. *This is
   the next spec to activate after the knee — not a new world.*
3. **The agent-callable simulator** (SPEC-5 §7 / SPEC-6 §7 packaging, but only for the FS/host
   world that works): expose `M_θ` + the tiered oracle as a tool an LLM agent calls — "predict
   the consequence of this command before running it" — plus the Inspect benchmark and
   `verifiers`-spec RL env already scaffolded ([`eval/`](../../src/verisim/eval/),
   [`rl/`](../../src/verisim/rl/)). The community contribution is the **ground-truth faithfulness
   benchmark and the cheap verified simulator**, not a model to compete with.
4. **Then, and only then,** the harder worlds (network → host → distributed), each earning its
   knee before the next is built.

This is the same ambition, made survivable: the world-simulation AI is real, it helps the
community as a measuring instrument and a safe agent substrate, and it complements LLMs by being
the one thing they cannot be — *verifiable*. The dream is intact; the path is one knee at a time.

And the complement is *symmetric in the model*: because the loop treats the proposer as a
pluggable part (the `Model` protocol, SPEC-2 §14), the verifiable specialist is not tied to one
architecture — a transformer, a JEPA-style latent predictor, an RSSM, or an LLM itself can sit in
the proposer slot, and the oracle grounds whichever one does. That is the broader contribution this
knee is a down payment on: **deterministic verification as a model-agnostic primitive** (SPEC.md §6
commitment 4, H22), demonstrated here on the smallest world before it is claimed on any larger one.

One thing K1 did without naming it: **it was oracle-supervised learning.** The "free, infinite teacher"
of §5 — the oracle emitting perfectly-labeled, coverage-balanced deltas and hard negatives on demand — is
the oracle entering training as the *icing* layer of LeCun's cake (supervised), distinct from the loop's
inference-time correction and from the RLVR *cherry* ([`verisim.rl`](../../src/verisim/rl/)). Naming it
that opens the obvious next question: the same free exact signal could ground the *bulk* (self-supervised
pretraining), the largest layer, where verisim has put none of its truth. That question is its own
cross-cutting method spec — [SPEC-8](./SPEC-8.md) (oracle-grounded self-supervision) — first tested on the
network world's latent arm (EN8/EN9, H23–H25). K1 is the existence proof that the oracle is a usable
training teacher; SPEC-8 asks how far up the cake that teacher reaches.

---

## 12. Definition of done

SPEC-2.1 is done when:

1. **K0** passes: the pipeline reaches ≥0.95 clean faithfulness on a trivial world (the learner
   works), with the diagnostic battery (§4) committed as figures.
2. **K2** passes: clean per-step faithfulness clears 0.5 on a non-trivial difficulty (the
   acceptance floor is crossed), regenerable from config + seeds.
3. **K4** produces the `H_ε(ρ)` curve and it either **shows the knee** (§0 target) or reports the
   **honest negative** (§10) with its diagnosis — whatever it shows, regenerable, records-only,
   in [docs/report.md](../report.md).
4. The autoresearch ratchet (§9) is upgraded (bits-to-correct gate + expanded space) and its
   search curve is committed.
5. SPEC-2.md's M6 status is updated to the result (knee or negative), and this spec records which
   roadmap spec activates next (S1, the system oracle) — *one* next step, not five.

The science is one curve. This time we earn it before we draw anything else.
