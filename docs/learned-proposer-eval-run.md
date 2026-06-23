# SPEC-27 — does learning actually help find soundness holes? (the honest evaluation)

> **Status: PRE-REGISTERED, sweep pending.** This section is written *before* the multi-seed sweep is
> run, per `plans/SPEC-27-honest-evaluation.md`. The predictions below are locked so the result cannot
> be retrofitted to whatever the data shows. Honesty disclosure: the predictions are informed by two
> single-seed peeks already in the commit history (step 1 found blind's first-bug at 6–23 calls and the
> bandit's at 10; the step-3 sizing run reproduced RA24's 17.8× raw-count ratio and timed neural at
> ~28× blind's wall-clock). The full CI'd aggregate across ≥20 seeds × 3 targets has **not** been seen.

## The claim under audit

RA24 ([neural-proposer-run.md](neural-proposer-run.md)): the neural compositional adversary finds the
printf-format-escape silent miss "**~18× more often per oracle call** than blind search" (1316 vs 74
true silent misses at 1600 oracle calls). RA23 made the analogous 2.3× claim. Both were measured
against `blind uniform` only, single seed, no CIs, on the **raw silent-miss count** metric.

Three reasons (SPEC-27 §1) the 18× may not mean what it sounds like:
1. **Weak baseline.** Only `blind uniform`, never a competent adaptive search (the bandit, SPEC-27 §3).
2. **Region-gameable metric.** 1316 vs 74 are *compositions of one bug* (the printf-fold). Distinct
   *bug classes* may be 1 for every arm that finds printf at all.
3. **Free-learning fiction.** "Per oracle call" ignores that a neural call costs ~28× a blind call in
   wall-clock. On a real time budget the accounting may invert.

## Pre-registered predictions (locked)

For each primary metric, the predicted direction and the kill-criterion-relevant CI call:

| metric | prediction | overlaps best non-learned baseline? |
|---|---|---|
| **raw silent-miss count / oracle call** (the RA24 metric) | neural ≫ blind, ~15–20×; reproduces RA24 | **no** — neural wins (but this is the demoted metric) |
| **distinct bug CLASSES at budget** (primary B) | neural ≈ blind ≈ bandit; ~1–2 classes all round | **yes — overlap** (advantage collapses) |
| **time-to-first-bug** (primary A, oracle calls) | neural ≈ blind ≈ bandit; ~5–15 calls all round | **yes — overlap** (advantage collapses) |
| **wall-clock-normalized silent count** (axis ii) | blind/bandit/enumerate **beat** neural (neural ~28× slower/call) | **yes/inverts** — neural loses on equal wall-clock |

**Predicted kill-criterion call (SPEC-27 §5): NULL — reframe the 18×.** The 18× is predicted *real but
metric-bound*: it exists only on raw-count-per-oracle-call. On the metric that matters (distinct bug
classes) and on the axis a practitioner actually has (equal wall-clock), the learned proposer is
predicted to show **no advantage** over a competent or even a blind baseline — a Revisiting-Neural-
Program-Smoothing result for guardrail auditing. Honest hedge kept: if distinct-classes or
time-to-first-bug separate with non-overlapping CIs in neural's favor on equal wall-clock, the claim
stands and is re-reported. We publish whichever lands.

## What the sweep will measure (committed before running)

- **Arms:** `blind`, `enumerate`, `bandit` (Thompson, the strong baseline), `neural` (REINFORCE, the
  claim) — all on the RA24 apparatus (`judge()` reward, `_Recorder`) at equal oracle-call budget, so
  this is RA24's own measurement plus the metrics it omitted.
- **Targets:** ≥3 retargeted protected paths (the printf bug is target-agnostic; this tests it isn't a
  single-path artifact).
- **Seeds:** ≥20 per (arm × target). Bootstrap 95% CIs on every number; no bare point estimates.
- **Both compute axes:** (i) equal oracle calls, (ii) equal wall-clock including proposer cost.
- **Metrics:** raw silent count, distinct compositions, **distinct classes**, **first-bug call**,
  reward/call, wall-clock.

## Results (30 seeds × 3 targets, budget 1600 oracle calls/arm, printf hole OPEN)

Run: `python -m verisim.experiments.spec27_proposer_eval --seeds 30` → [`runs/spec27/sweep.json`](../runs/spec27/sweep.json).
Every number is a bootstrap 95% CI of the mean over 90 runs (30 seeds × 3 retargeted targets).

| arm | raw silent count | distinct **classes** | **time-to-first-bug** | wall-clock (s) |
|---|---|---|---|---|
| `blind` | 58.3 [55.7, 61.0] | 1.00 [1.00, 1.00] | 23.0 [18.6, 27.9] | 0.05 |
| `enumerate` | 6.0 [6.0, 6.0] | 1.00 [1.00, 1.00] | **10.0 [10.0, 10.0]** | 0.05 |
| `bandit` | **1562.8 [1557, 1568]** | **2.00 [2.00, 2.00]** | 25.2 [20.1, 30.7] | 0.17 |
| `neural` | 1205.0 [1167, 1241] | 1.79 [1.70, 1.87] | 28.1 [21.9, 35.5] | 0.89 |

### Verdict: NULL — the 18× does not survive prudent evaluation. (Pre-registered prediction held.)

The kill criterion is read on the axis **most favorable to neural** — equal oracle calls, which hides
neural's wall-clock cost. Even there, the learned proposer does not beat the best non-learned baseline
on a single honest metric; it *loses* on all three, every CI non-overlapping:

1. **Raw count (the RA24 metric): the bandit beats neural** — 1563 vs 1205. The "18×" reproduces only
   against `blind` (1205 vs 58 ≈ 20×, byte-consistent with RA24's 1316/74). Add the competent adaptive
   baseline RA24 omitted and the advantage **inverts**: the cheap Thompson bandit finds *more*. The 18×
   was an artifact of comparing against uniform random, exactly the Klees/SoK weak-baseline failure.
2. **Distinct bug classes (the metric that matters): the bandit beats neural** — 2.00 vs 1.79. The
   1205–1563 "silent misses" are *compositions of one bug* (the printf-fold); there are only 2 classes
   on this target (`printf_fmt`, `mixed`) and the bandit finds both in every run while neural sometimes
   finds one. The raw-count gap buys **zero** additional distinct bugs.
3. **Time-to-first-bug: enumerate beats neural** — 10 vs 28 calls. Deterministic systematic search hits
   the printf witness fastest; the neural policy is *slowest* (it spends early proposals exploring
   before the policy sharpens).
4. **Wall-clock: neural costs ~16× blind** (0.89s vs 0.05s) and ~5× the bandit. So axis (ii) only
   widens the gap: in neural's wall-clock, blind makes ~16× the oracle calls.

### The honest, precise statement

This is not "learning never helps." Adaptivity *does* help raw count — the bandit's 1563 ≫ blind's 58.
The finding is sharper and more useful: **the cheap adaptivity (a Thompson bandit) captures all of the
benefit; the expensive adaptivity (a neural REINFORCE policy) adds 16× wall-clock cost and buys
nothing** — not more distinct bugs, not a faster first bug, not even more raw compositions. And the raw
count advantage that *does* exist is on a metric that rewards re-spelling one bug, which is not what a
soundness auditor wants. This is a Revisiting-Neural-Program-Smoothing result for guardrail auditing:
the learned method's headline win dissolves under a strong baseline and the right metric.

### Honest boundaries (what this does and does not show)

- **At RA24's own hyperparameters and scope.** One bug family (the printf-fold on a protected path), a
  2-layer GPT, the 12-mechanism grammar, 1600-call budget. A *harder* hole-finding task — many distinct
  bug classes, a sparse reward where blind/bandit stall — could still favor a neural policy. This result
  is that *on the task RA24 used to claim 18×*, the claim does not hold up. It does not prove learning
  is useless on richer auditing problems; that is a separate, open question.
- **The bandit is itself adaptive**, so the contribution is a *baseline-strength* and *metric-choice*
  result, not "deterministic beats learned." Enumerate (no learning at all) wins time-to-first-bug;
  bandit (cheap learning) wins raw count and classes; neural (expensive learning) wins nothing.
- **The certify-don't-assert loop is unaffected.** SPEC-27 audits the *proposer-efficiency* claim, not
  the soundness result. The printf bug is real, the oracle-confirmed certificate is real, and any of the
  four proposers finds the bug. What is retracted is only "the *neural* proposer finds it dramatically
  faster" — the discovery does not need the neural net.

Per SPEC-27 §5 the null lands, so the RA23 "2.3×" and RA24 "18×" claims are corrected in
[`docs/learned-proposer-run.md`](learned-proposer-run.md), [`docs/neural-proposer-run.md`](neural-proposer-run.md),
[`docs/paper.md`](paper.md), and the README ledger to "no measured advantage over a competent baseline
under prudent evaluation."
