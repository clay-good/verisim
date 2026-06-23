# SPEC-27 — Does learning actually help find soundness holes? An honest, pre-registered evaluation

**A HUNT TURNED ON OURSELVES. This is not another layer of auditor machinery and not a new mechanism —
the diligence (three web passes, June 2026) found that every mechanism in this repo's neighborhood is
owned: learned adversaries against a checker (BanditFuzz, FM 2021; CovRL, ISSTA 2024), neural
counterexample-guided synthesis (Code2Inv; RL-CEGIS, 2107.09766), certify-don't-assert for verifiers
(SoundnessBench, 2412.03154; Arguzz, 2025), learned grammar bypass of a checker (WAF-A-MoLE; RAT), and
even the good-Turing residual certificate (Böhme, STADS, 2018). What is NOT crowded is the thing this
repo has repeatedly proven it can do and the learned-bug-finding subfield publicly admits it lacks:
an honest, pre-registered, strong-baseline, CI'd evaluation that reports the null. SPEC-27 turns that
discipline on the program's own most-quoted positive claim.**

> **▶ PROPOSED — EVALUATION / SELF-AUDIT — 2026-06-23.** No new mechanism. The only new code is one
> *baseline* (a Thompson-sampling bandit proposer, the BanditFuzz analogue) plus a multi-seed,
> multi-target harness around the existing `audit()` loop and the existing proposers. If the learned
> proposer's advantage does not survive the protocol below (see §5 kill criterion), the honest finding
> is a **retraction/reframing** of the 18× claim — a "Revisiting Neural Program Smoothing" for agent-
> guardrail auditing — and that null IS the contribution. If it survives, it is a real positive in a
> field full of weak-baseline ones. Either way we publish what is true.

---

## 1. The claim under test (and why it is suspect)

The program's most-quoted positive result is that a **learned** adversarial proposer finds soundness
holes in a guardrail more efficiently than a non-learned one:

- **RA23** ([learned-proposer-run.md](../learned-proposer-run.md)): "the learned adversary finds
  **2.3× more holes** per oracle call than blind search," 600 oracle calls, **one seeded run**,
  baseline = `blind uniform`.
- **RA24** ([paper.md](../paper.md) line 25): the neural escalation, "about **18× faster** than blind
  search," carried into the abstract as the headline that the method "behaves as a method, not a slogan."

Three reasons, from [SoK: Prudent Evaluation Practices for Fuzzing](https://arxiv.org/pdf/2405.10220)
and [Revisiting Neural Program Smoothing](https://arxiv.org/pdf/2309.16618), to distrust these numbers
*as written*:

1. **Weak baseline.** `blind uniform` is the weakest possible comparator (Klees et al.'s canonical
   complaint). A learned proposer beating uniform random is nearly uninformative; the question is
   whether it beats a *competent* non-learned search (systematic enumeration, or a construct-level
   bandit) on equal compute.
2. **N = 1, no CIs.** Both numbers are single seeded runs. The field standard is ≥ 20 repetitions with
   confidence intervals, because learned-fuzzer speedups routinely collapse into the noise band once
   seed variance is shown.
3. **Reward-gameable metric.** "Holes per oracle call" rewards a proposer for camping in a hole-rich
   region of the grammar. The thing that actually matters for a certificate is **distinct genuine
   soundness bugs** (silent misses, `realizes ∧ ¬target` *inside the monitor's contract*) and
   **time-to-first** such bug — metrics a region-camper does not win.

## 2. The hypothesis (falsifiable, pre-registered)

**H-eval:** the learned proposer's hole-finding advantage is a property of the method, not of the weak
baseline / single seed / gameable metric, and therefore survives a prudent-evaluation protocol.

Operationally, H-eval predicts: on equal compute, with ≥ 20 seeds and 95% CIs, across ≥ 3 distinct
target monitors, the learned (neural) proposer **strictly dominates the strongest non-learned baseline**
on the primary metrics (§4), with **non-overlapping** CIs.

The null **¬H-eval** (which the diligence makes the live possibility) is: once the baseline is strong,
the seeds are many, and the metric is bug-count not hole-count, the advantage's CI **overlaps** the
best non-learned baseline — i.e. learning does not help find soundness holes here, and the 18×/2.3×
were artifacts of the comparison, not the method.

## 3. Method — arms (all driven through the existing `audit(monitor, oracle, proposer, budget)` loop)

The audit loop, oracle, and monitors are unchanged. Only the **proposer** varies. Reward / verdict is
always the free exact oracle's `realizes ∧ ¬target`, never a learned reward model.

| arm | proposer | role | code |
|---|---|---|---|
| `neural` | `NeuralGrammarProposer` (REINFORCE) | **the claim under test** | `audit/proposers.py` (exists) |
| `blind` | `GrammarProposer(mode="blind")` (uniform) | weak baseline, kept for continuity with RA23/24 | exists |
| `enumerate` | `GrammarProposer(mode="enumerate")` (systematic) | **strong baseline #1** — competent deterministic search | exists |
| `bandit` | Thompson-sampling / UCB over grammar constructs | **strong baseline #2** — the BanditFuzz analogue, the comparator the literature *requires* | **NEW (the only new code; a baseline, not a mechanism)** |

The `bandit` arm is the load-bearing addition: it is a non-learned-but-adaptive search that keys on
construct-level reward, exactly the thing RA23/24 never compared against. If `neural` cannot beat
`bandit`, the "learning helps" claim does not hold.

## 4. Metrics (pre-registered; primary vs secondary stated up front)

- **Primary A — time-to-first soundness bug.** Oracle calls until the first genuine silent miss
  (a realizing action off-surface *within contract*) is surfaced, per target, swept over budget B.
- **Primary B — distinct soundness-bug classes found at budget B.** Count of *distinct* realizing
  residual classes (the printf-format-escape fold, the quote-splice form, etc.), not raw holes.
- **Secondary — holes per oracle call.** The RA23 metric, reported but **demoted** and explicitly
  flagged as region-gameable, so the reader can see the gap between the old metric and the right one.
- **Compute parity.** Report results at **equal total compute** two ways: (i) equal oracle calls
  (RA23's axis), and (ii) equal wall-clock including proposer cost (the neural arm pays for its
  forward/backward passes; a fair race counts that). A speedup that exists on (i) but vanishes on (ii)
  is reported as exactly that.
- **Statistics.** ≥ 20 seeds per (arm × target). Bootstrap 95% CIs on every headline number; effect
  size (e.g. ratio of medians) with its own CI. No bare point estimates in the write-up.

Targets: ≥ 3 distinct protected-path monitors (the existing retargetable `GrammarProposer` /
`CorpusProposer` are parametric in target, so this is a sweep, not new monitors).

## 5. Kill criterion (the honest core — committed before any run)

Decided in advance so the result cannot be rationalized after the fact:

- **Reframe / retract (the null, ¬H-eval).** If, on equal compute (axis ii) with ≥ 20 seeds, the
  `neural` arm's 95% CI on **both** primary metrics **overlaps** that of the best non-learned baseline
  (`enumerate` or `bandit`), then the 18× and 2.3× claims are **corrected in paper.md, report.md, the
  README ledger, and learned-proposer-run.md** to "no measured advantage over a competent baseline
  under prudent evaluation." The contribution becomes the honest negative: a Revisiting-Neural-Program-
  Smoothing result for adversarial guardrail auditing.
- **Confirm (H-eval holds).** If `neural` dominates the best non-learned baseline with non-overlapping
  CIs on equal compute (axis ii), the claim stands and is re-reported **with** its CIs, baseline, and
  compute axis stated — an honestly-earned positive, which is itself rare in this subfield.
- **Publish either way.** Pre-registration means the write-up is committed regardless of direction. The
  value is the rigor, not the sign.

## 6. Threats to validity (controlled, per SoK)

- **Baseline strength** — addressed by `enumerate` + `bandit`, not just `blind`.
- **Seed variance** — ≥ 20 seeds, CIs on everything.
- **Metric gaming** — primary metric is distinct-bug-count / time-to-first-bug, not hole-count.
- **Compute parity** — both equal-oracle-calls and equal-wall-clock axes reported.
- **Single-target overfit** — ≥ 3 targets.
- **Our own prior numbers** — RA23's 2.3× and RA24's 18× are **re-run under this exact protocol first**,
  as the inciting measurement. We audit ourselves before anyone else has to.

## 7. Scope and honest boundaries

- This evaluates **one harm family** (file corruption on a protected path) over the existing
  compositional shell-encoding grammar — the same scope as RA18/RA22/RA24. It does not claim a result
  about all guardrails or all grammars; it claims a clean answer about *this* learned proposer vs
  competent baselines on *this* well-characterized task. That narrowness is what makes the CIs mean
  something.
- It is an **evaluation**, not a new method. Its novelty is rigor in a subfield with a documented
  evaluation-honesty deficit, not a new mechanism (there isn't one left to claim here, and SPEC-27
  exists partly to stop pretending otherwise).
- The deliverable is `docs/specs/SPEC-27.md` (this file), the runbook
  [`plans/SPEC-27-honest-evaluation.md`](../../plans/SPEC-27-honest-evaluation.md), the new `bandit`
  baseline + harness, a results doc `docs/learned-proposer-eval-run.md`, and the committed corrections
  to the prior claims if the null lands.

## 8. Reproduce (target interface; harness is the build step)

```
python -m verisim.experiments.spec27_proposer_eval --seeds 20 --targets 3 --budget-sweep
python figures/plot_spec27.py            # -> figures/spec27_proposer_eval.png (CIs, both compute axes)
python -m pytest tests/test_spec27_proposer_eval.py -q
```

---

*Lineage: this is the continuation of the program's defining move — kill your own favorite hypothesis
with honest measurement ([docs/review.md](../review.md), the RA arc, SPEC-26). SPEC-27 points that move
at the one positive number the program still leans on. References that motivate it:
[SoK: Prudent Evaluation Practices for Fuzzing](https://arxiv.org/pdf/2405.10220),
[Revisiting Neural Program Smoothing for Fuzzing](https://arxiv.org/pdf/2309.16618),
[BanditFuzz](https://uwspace.uwaterloo.ca/bitstreams/798ce0c9-aa29-416f-9ed4-0d968578cb02/download),
[SoundnessBench](https://arxiv.org/abs/2412.03154).*
