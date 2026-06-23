# SPEC-27 runbook — the honest proposer evaluation

> Why this file exists: so the *context* behind SPEC-27 survives a cold restart. The spec says what to
> run; this says **why we landed here**, what was already ruled out, and the exact build order.

## How we got here (2026-06-23 session, so we don't relitigate it)

We asked: is the program done, and if not, where is the genuinely-useful, hard, not-already-owned work?
Four web-diligence passes (recorded below) answered it:

1. **Auditor pivot (audit a third party's guardrail)** — occupied: "Who Tests the Testers?"
   (arXiv 2603.18245), SafeAudit/BenchGuard (2604.24955), Microsoft AI Red Teaming Agent.
2. **Confidentiality / covert-channel leak-bandwidth bound** — occupied: "Verifier-Bound Communication
   for LLM Agents: Certified Bounds on Covert Signaling" (2603.00381) does exactly the oracle-grounded
   bits-per-action bound; plus Microsoft IFC (2505.23643), Ghost-in-the-Agent (2604.23374). Our own
   RA15 already judged confidentiality "out" (no sparse surface). **Do not chase this.**
3. **CEGIS / learned-adversary mechanism** — occupied: BanditFuzz (RL fuzzing SMT solvers, 2021), CovRL
   (ISSTA 2024), Code2Inv, RL-CEGIS (2107.09766), SoundnessBench (2412.03154), Arguzz (2025),
   WAF-A-MoLE/RAT, and STADS (good-Turing residual, 2018). Every ingredient of our auditor is published.
4. **Stochastic-LLM-judge certification** — partly occupied: "Noisy but Valid" (2601.20913), conformal
   LLM-as-judge (2509.18658).

**Conclusion:** there is no unowned *mechanism* left in this neighborhood. The one thing that is
under-supplied — and that this repo has demonstrably done better than most (`review.md`, the RA arc) —
is **honest, pre-registered, strong-baseline evaluation that reports the null.** The learned-bug-finding
subfield publicly admits it lacks exactly this (SoK 2405.10220; Revisiting Neural Program Smoothing
2309.16618). So we turn that discipline on our own most-quoted positive number: the 18× learned-proposer
speedup. This is true research, it is hard (rigor is the hard part), it is gated by temperament not
credentials, and it is the dignified continuation of what the program already is.

## The thing under test

`docs/learned-proposer-run.md` (RA23): "2.3× more holes per oracle call than blind search," N=1 seed,
baseline = blind uniform. `docs/paper.md:25` (RA24): "about 18× faster than blind search," in the
abstract. Both: weak baseline, single seed, no CIs, hole-count metric. All three are the documented
failure modes of ML-fuzzing evaluation.

## Build order (each step has a verify check)

1. **Lift the existing arms behind one harness.** A single entry point that runs `audit()` with a
   pluggable proposer over a pluggable target, logging per-call: proposer, target, seed, oracle calls
   so far, holes found, distinct-bug-classes found, wall-clock, first-silent-miss step.
   → verify: reproduces the RA23 `neural` vs `blind` 600-call numbers at the original seed (sanity:
   our refactor didn't change the measured quantity).
2. **Write the `bandit` baseline** (Thompson sampling / UCB over grammar constructs — ATOMS × MECHANISMS
   × VERBS × REDIRECTS). This is the only new code and it is a *baseline*. Torch-free, seeded.
   → verify: on a planted-hole toy target it converges to the hole-bearing constructs; unit test green.
3. **Multi-seed, multi-target sweep.** ≥ 20 seeds × ≥ 3 retargeted monitors × {neural, blind,
   enumerate, bandit}, budget B swept. Record both compute axes (oracle calls; wall-clock incl.
   proposer cost).
   → verify: every headline number has a bootstrap 95% CI; no bare point estimates.
4. **Decide against the pre-registered kill criterion (SPEC-27 §5).** CIs overlap on equal-wall-clock →
   null → corrections land in paper.md / report.md / README / learned-proposer-run.md. CIs separate →
   confirmed → re-report with CIs + baseline + axis.
   → verify: the result doc `docs/learned-proposer-eval-run.md` states the direction, the CIs, and the
   exact edits made (or the exact claim re-stated), with a reproduce block.

## Pre-registration lock (write the prediction down before running step 3)

Before the sweep runs, record in `docs/learned-proposer-eval-run.md` a one-line **prediction** for each
primary metric (will the neural CI clear the best non-learned baseline on equal wall-clock, yes/no, and
a rough guess). This is so we cannot, after the fact, claim we expected whatever happened. Honest
forecasting is part of the deliverable.

## Anti-goals (don't let the work drift back into the treadmill)

- No new harm family, no new monitor mechanism, no new certificate type. SPEC-27 is an evaluation.
- Do not improve the neural proposer to "win." If it loses honestly, that is the result. Tuning the
  thing under test to beat its baseline is the exact malpractice this spec exists to expose.
- Do not drop the `bandit` baseline if it turns out strong. Its strength is the finding.

## Definition of done

`docs/learned-proposer-eval-run.md` exists with: the pre-registered prediction, the CI'd multi-seed
multi-target results on both compute axes, the kill-criterion decision, and either (a) the committed
claim corrections (null) or (b) the re-stated claim with its CIs (confirmed). SPEC-27 §5 satisfied
either way.
