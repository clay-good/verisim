# SPEC-13 — Speculative World-Model Rollout: Draft-Verify as the Consultation Policy

**Method specification: the program's propose–verify–correct loop is, when read at the right altitude,
*speculative decoding lifted from tokens to world-STATES*. The cheap learned `M_θ` drafts `k` steps
forward; the oracle verifies the draft and **accepts the longest correct prefix**, rejecting and
correcting at the first divergence — structurally the same accept-longest-correct-prefix rule that
makes LLM speculative decoding exact-by-construction and fast. This spec asks whether that rule, used
*as the consultation policy `π_c`*, is the one shape that beats the fixed/uniform policies which
produced the program's central negative — the floor + cliff with no favorable knee — and pre-registers
exactly where it should and should not win.**

> **▶ METHOD SPEC — 2026-06 — SHIPPED (✅ SR1–SR6 on the controlled-drafter core; trained-`M_θ` arm
> deferred). See §11 for the status table and results.** A *method* spec in the lineage of
> [SPEC-3](./SPEC-3.md) (the consultation-policy program) and [SPEC-12](./SPEC-12.md) (planning
> altitude above the loop): it invents **no new world** and **no new oracle**. It runs on the SPEC-5
> network world (`ReferenceNetworkOracle` + `ControlPlaneOracle`), the SPEC-6 host world
> (`hostsim`'s shipped `imagine`/`verify`), and the SPEC-2.1 single-filesystem world — all already
> built. What it adds is a *consultation policy*: replace the fixed-`ρ` and uncertainty-triggered
> schedules ([`verisim.loop.policy`](../../src/verisim/loop/policy.py)) with a **speculative
> draft-verify schedule** — draft `k`, verify, accept the longest faithful prefix, re-anchor at the
> first divergence — and measure whether *accepted faithful steps per oracle call* (a "speculative
> speedup") beats the no-knee floor. Inspiration is acknowledged plainly: the mechanism is
> speculative decoding (Leviathan et al., Google, ICML 2023; Chen et al., DeepMind, 2023) and its
> tree/multi-draft and calibration descendants (SpecInfer, Medusa, EAGLE / EAGLE-2 / EAGLE-3,
> lookahead decoding). The verisim contribution is the one move that line cannot self-validate: the
> verifier is an **exact deterministic oracle**, not a second neural net, so "accept the longest
> correct prefix" is correct against ground truth rather than against a larger model's distribution.

Read [SPEC.md §5](./SPEC.md) (the formalism: `H_ε`, `ρ`, `π_c`, the propose–verify–correct loop),
[SPEC.md §6 commitment 4](./SPEC.md) (the model-agnostic pluggable proposer this spec exploits),
[SPEC-2.1 §8](./SPEC-2.1.md) (K4 — the *reset-resistant cliff*, the open problem this spec must
survive), the consultation policies in
[`loop/policy.py`](../../src/verisim/loop/policy.py) (`FixedInterval`, `UncertaintyTriggered`,
`DriftTriggered` — the policies SPEC-13 adds a speculative sibling to), the shipped net loop
[`netloop/runner.py`](../../src/verisim/netloop/runner.py) (`run_net_rollout` — the PROPOSE/VERIFY/
CORRECT loop, already structured exactly as a speculative step), and the plan-level draft-verify in
[`hostsim/simulator.py`](../../src/verisim/hostsim/simulator.py) (`imagine`/`verify`/`PlanReport` —
*literally* a draft, a verify, and a `plan_faithful_horizon` that is already the accepted-prefix
length).

---

## 0. One-paragraph thesis

LLM speculative decoding speeds up an expensive autoregressive model by interposing a cheap drafter: the
small model proposes `k` tokens, the large model verifies all `k` in one parallel pass, and the system
**accepts the longest prefix the large model agrees with**, paying one expensive call to advance by
`1 + (accepted)` tokens (Leviathan et al., ICML 2023; Chen et al., 2023). Verisim's loop has the same
three moves — PROPOSE (cheap `M_θ`), VERIFY (expensive oracle), CORRECT at the first disagreement
([`netloop/runner.py`](../../src/verisim/netloop/runner.py), [`hostsim/simulator.py`](../../src/verisim/hostsim/simulator.py)) —
but to date the program has only run the *fixed* and *uncertainty-triggered* schedules
([`loop/policy.py`](../../src/verisim/loop/policy.py)), which produced the headline negative: a **floor
+ cliff, no favorable knee** in `H_ε(ρ)` across every world and proposer, with the shape attributed to
the *loop*, not the model (EN7/H22). The speculative-decoding lens says those schedules were the wrong
shape: fixed-`ρ` consults on a clock, ignoring where the draft actually broke; the accept-longest-
faithful-prefix rule consults *exactly at* the break and nowhere else, which is **optimal by
construction** in the sense the fixed policy is not — every oracle call advances the trusted state by
the *whole* faithful prefix, and the expected accepted steps per call is precisely a speculative
speedup `E[accepted] = (1−α^{k+1})/(1−α)` for per-step acceptance `α`. The claim, stated to be
falsified: **a speculative draft-verify schedule lifts faithful steps-per-oracle-call above the fixed-`ρ`
floor on worlds where drift is *gradual* (network, host), because there the faithful prefix is long; and
it does *not* beat fixed on the SPEC-2.1 single-filesystem world, because K4 proved the cliff there is
*reset-resistant* — discrete errors place the first divergence early, so the longest faithful prefix is
short no matter how the schedule is chosen.** Both branches are banked (SPEC §10.1): a win names the
first consultation policy that escapes the no-knee floor and explains *why* (the floor was a scheduling
artifact, not a model ceiling); a loss on the gradual worlds proves the floor is the loop's regardless of
policy (the deepest possible negative, closing the speculative line); and the *predicted* split between
gradual and discrete worlds is itself the result — it localizes the floor to per-step **error placement**,
not consultation strategy.

---

## 1. Why now: the loop already *is* a speculative step

The faithful-horizon program has produced one shape across every world and proposer — a **floor + cliff,
no favorable knee** (EN1, EH1, ED-loop) — and attributed it to the loop, not the model (EN7/H22). SPEC-3
opened the consultation-policy program (which questions to ask the oracle, when); SPEC-2.1's K4 closed the
"earn the knee" hunt on the single-filesystem world with a *mechanistic* negative: a discrete filesystem
edit is a set-difference that lands past `ε` in a single step, so **first-exceedance `H_ε` is set by the
first error's position and cannot be reset-extended** — the cliff is *reset-resistant*, and a smarter
schedule did not beat fixed there. The relevant code is already shaped like a speculative step. The net
loop [`run_net_rollout`](../../src/verisim/netloop/runner.py) is a per-step PROPOSE (`_predict`) → decide-
to-VERIFY (`policy.should_consult`) → CORRECT (`correct(predicted, truth)`). The host simulator
[`HostSimulator.verify`](../../src/verisim/hostsim/simulator.py) is *exactly* draft-then-verify: `imagine`
rolls `M_θ` forward with no oracle (the draft), `verify` checks step by step and returns
`plan_faithful_horizon` — **the length of the longest prefix the model kept within `ε`, i.e. the accepted
prefix.** The one thing missing is a policy that *uses* the accepted-prefix length to schedule the next
consult instead of a fixed clock.

This is the precise condition under which the LLM-inference field abandoned fixed-stride verification.
Speculative decoding's whole point is that verification placement should be *adaptive to where the draft
breaks*, not periodic — and the accept-longest-correct-prefix rule is what makes the accelerated output
**identical** to the slow model's (exact-by-construction), the same property verisim wants for its
rollout. Two facts make verisim the right place to test the lens, one unique:

- **The verifier is an exact oracle, not a bigger model.** LLM speculative decoding accepts a draft token
  iff a *larger neural model* would have produced it — correctness is relative to a model, and a longer
  draft is trusted via a *learned* confidence proxy (EAGLE-2's confidence↔acceptance calibration). Verisim
  accepts a drafted state iff the *oracle* agrees within `ε` — correctness is relative to ground truth.
  The accepted prefix is *faithful*, not merely *self-consistent*.
- **The acceptance rate `α` is a measurable, per-region property already in hand.** The RSSM exposes
  `belief_var` ([`GraphRSSMNet.encode`](../../src/verisim/netmodel/graph_model.py)) and the grammar decode
  exposes per-step entropy — the two signals EAGLE-2 uses to choose draft length. Verisim can *calibrate*
  draft length `k` to predicted acceptance and then *check the calibration against the oracle*, which the
  LLM line can only approximate.

---

## 2. The lineage folded in (and the design choice each one forces)

### 2.1 Speculative decoding (the mechanism)

- **Speculative decoding / speculative sampling** (Leviathan et al., Google, ICML 2023; Chen et al.,
  DeepMind, 2023). A cheap drafter proposes `γ` tokens; the target verifies all in one parallel pass and
  accepts the longest matching prefix, falling back to one fresh token on the first rejection — output
  distribution provably unchanged. → **Design choice, almost one-to-one:** a `SpeculativeConsult` policy
  (the loop's `π_c`) that lets `M_θ` free-run `k` steps (the draft), then issues **one** oracle
  verification of the draft window and accepts the longest within-`ε` prefix, correcting and re-anchoring
  at the first exceedance (§3, SR1). "Accepted faithful steps per oracle call" *is* the speculative speedup.
- **Optimal draft length / acceptance theory.** Expected accepted tokens per cycle for i.i.d. per-step
  acceptance `α` over a draft of `k` is `E = (1−α^{k+1})/(1−α)`; the optimum trades `α` against `k`, and
  the i.i.d. assumption is known to *break* because acceptance falls with draft position. → **Design
  choice:** SR2 fits this curve directly — measure per-step faithfulness `α(t)` against the oracle, report
  the *empirical* accepted-prefix distribution vs the i.i.d. prediction, and read off the position-
  dependence (the network/host drift profile vs the K4 discrete step).
- **EAGLE / EAGLE-2 / EAGLE-3** (Li et al.; EAGLE arXiv:2401.15077, ICML 2024; EAGLE-2 arXiv:2406.16858,
  EMNLP 2024; EAGLE-3 arXiv:2503.01840, NeurIPS 2025). EAGLE drafts at the *feature* level and identifies
  feature uncertainty as the limit; EAGLE-2 uses the draft model's **confidence to approximate acceptance
  rate** and grows the draft only where confidence is high; the net-negative regime sits near the
  ≈0.5-acceptance floor (already cited in [related-work.md](../related-work.md), SPEC-3 era). → **Design
  choice:** the *calibrated-`k`* policy (SR4) — choose draft length from `belief_var` /
  decode-entropy ([`GraphRSSMNet`](../../src/verisim/netmodel/graph_model.py)), draft longer where the
  model is confident, shorter where it is not — and check the confidence↔acceptance link against the
  oracle, the EAGLE-2 theory turned into a measurement.

### 2.2 Tree / multi-draft verification (verify many rollouts in one oracle pass)

- **SpecInfer** (Miao et al., ASPLOS 2024, arXiv:2305.09781) and **Medusa** (Cai et al., ICML 2024,
  arXiv:2401.10774). SpecInfer verifies a whole *token tree* of candidate continuations against the target
  in a single pass (tree attention); Medusa adds parallel draft heads producing multiple candidates. →
  **Design choice:** *multi-draft* verification (SR3) — sample several candidate rollouts from `M_θ`
  (temperature / dropout / top-k over the grammar decode) and verify them against **one** oracle trajectory,
  accepting the best (longest-faithful) branch. The oracle step is the same cost whether one or many drafts
  are checked against it, so a tree of drafts is a near-free way to raise the accepted-prefix length — the
  SpecInfer move, with the oracle as the single verifying pass.

### 2.3 Self-speculative drafting (a tiny model pre-filters for a bigger one)

- **Self-speculative / staged drafting** (the self-speculative-decoding line; layer-skipping drafters such
  as the KnapSpec/self-spec family). A subset of the model drafts cheaply and the full model verifies,
  removing the separate-draft-model maintenance cost. → **Design choice:** a *two-tier* speculative cascade
  (SR5) — a tiny/cheap `M_θ` (the SPEC-5 flat baseline or a low-capacity graph arm) drafts, a larger `M_θ`
  *re-drafts only the rejected suffix* as a cheaper pre-filter, and the oracle is consulted last. The
  question is whether the cheap pre-filter reduces *oracle* calls per faithful step (the only cost that
  matters here, since the oracle is the expensive verifier, not a GPU).

### 2.4 What the program already built that this sits on

- **The loop is a speculative step.** `run_net_rollout`'s PROPOSE/VERIFY/CORRECT and `HostSimulator`'s
  `imagine`/`verify` are draft/verify; `PlanReport.plan_faithful_horizon` and the
  [`faithful_horizon`](../../src/verisim/metrics/horizon.py) metric are the accepted-prefix length. SPEC-13
  adds a policy that schedules on the accepted prefix; it adds almost no new primitive.
- **The acceptance signal exists.** `belief_var` and grammar decode-entropy are the EAGLE-2 confidence
  proxy; the equal-`ρ` spend-down backstop in `run_net_rollout` already enforces a true equal-budget
  comparison, so a speculative policy can be measured against fixed at *identical* oracle spend.
- **The model-agnostic proposer seam** (SPEC §6 commitment 4; the `NetModel`/`NetUncertaintyModel`
  protocols in [`netloop/model.py`](../../src/verisim/netloop/model.py)) lets a tiny drafter and a larger
  re-drafter be the *same interface* — the self-speculative cascade is a clean composition, not a rewrite.

---

## 3. The architecture: speculative rollout as a consultation policy

```
   step t:  s_t (trusted, oracle-anchored)
              │
              ▼  DRAFT k steps, NO oracle  (imagine / free-run M_θ)
   ŝ_{t+1} ─▶ ŝ_{t+2} ─▶ … ─▶ ŝ_{t+k}            ← optionally a TREE of drafts (§2.2, SR3)
              │
              ▼  VERIFY the draft window against ONE oracle trajectory   (the single expensive pass)
   accept the longest prefix with divergence ≤ ε  ───────────►  accepted length a = H_ε(window)
              │                                                   (= PlanReport.plan_faithful_horizon)
              ▼  CORRECT + RE-ANCHOR at the first exceedance
   s_{t+a+1} (trusted)  ──▶  choose next draft length k' from belief_var / entropy   (§2.1, SR4)
```

Four commitments, each tied to a measured verisim fact or a pre-registered branch:

1. **Consult at the break, not on a clock.** The policy issues an oracle verification of a *draft window*
   and re-anchors at the first divergence — the accept-longest-correct-prefix rule. This is the shape the
   fixed-`ρ` and uncertainty-triggered policies are not, and the candidate escape from the no-knee floor.
2. **Accepted steps per call is the speedup.** The figure of merit is *faithful steps advanced per oracle
   call* at equal total budget `ρ` — the lifted speculative speedup `E[a] = (1−α^{k+1})/(1−α)`. SPEC-13
   measures it against fixed-`ρ` at identical spend (the `run_net_rollout` backstop guarantees parity).
3. **Calibrate `k` to acceptance, then check the calibration.** Draft length is chosen from `belief_var` /
   decode-entropy (the EAGLE-2 confidence↔acceptance link) and the link is *verified* against the oracle —
   a measurement the LLM line can only approximate (§2.1, SR4).
4. **The oracle pass is shared across drafts.** A tree of candidate rollouts is verified against one oracle
   trajectory; checking many drafts costs the same one expensive pass (§2.2, SR3) — the SpecInfer move with
   ground truth as the single verifier.

---

## 4. The load-bearing assumption (and the K4 honest negative pre-registered)

The whole speculative win rests on **the faithful prefix being long enough that consulting at the break
beats consulting on a clock.** That is true exactly when drift is *gradual* — per-step faithfulness `α(t)`
stays high for several steps before the divergence — and false when the first error lands past `ε`
*immediately*. K4 ([SPEC-2.1 §8](./SPEC-2.1.md)) proved the second case holds on the single-filesystem
world: a discrete edit is a set-difference that exceeds `ε` in one step, so first-exceedance `H_ε` is set
by the first error and is *reset-resistant*. The honest pre-registration, banked in advance:

- **If drift is gradual (network, host worlds), the accepted prefix is long** and speculative rollout lifts
  faithful-steps-per-call above fixed-`ρ`: the policy spends its oracle budget on the steps that actually
  break, and free-runs the rest. This is where the lens *should* win (SR1 on net/host).
- **If drift is discrete-and-early (the SPEC-2.1 filesystem world, K4), the accepted prefix is short** no
  matter the schedule — `a ≈ H_ε ≈ small` — so accept-longest-prefix *inherits the same floor + cliff* and
  ties or loses to fixed. **This branch is predicted, not feared.** Its lesson is sharp: the no-knee floor
  is about **per-step error placement** (a property of the world's discreteness and the divergence metric),
  *not* about consultation policy — which retires the "smarter schedule earns the knee" hope for discrete
  worlds permanently and points the next move at the *metric/representation* (a continuous or
  edit-distance-graded `ε` that does not saturate on the first discrete error), not the policy.

This is the program's epistemic engine at design time: the assumption that would sink a naive "speculative
decoding fixes the knee" claim is, here, a *pre-registered split* between world types, with the failing
branch turned into a localization result. The single experiment that decides it (SR2) measures `α(t)` per
world *before* the policy is run, so the split is observed, not assumed.

---

## 5. Hypotheses (pre-registered, continuing the global H-ID space past SPEC-12's H38)

- **H39 — speculative rollout lifts faithful-steps-per-oracle-call on gradual worlds (THE headline).** On
  the network and host worlds, the accept-longest-faithful-prefix policy advances more faithful steps per
  oracle call than fixed-`ρ` at *identical* total budget — i.e. it shows the favorable knee no fixed/uniform
  policy showed (EN7/H22). *Refuted if* speculative ties or loses to fixed-`ρ` at equal budget on the
  gradual worlds — in which case the no-knee floor is the loop's *regardless of consultation policy*, the
  deepest negative, and it closes the speculative line on these worlds. Tested as **SR1**.

- **H40 — the accepted-prefix distribution follows the speculative-speedup law where drift is gradual and
  the K4 floor where it is discrete.** Measured per-step faithfulness `α(t)` against the oracle yields an
  accepted-prefix distribution that tracks `E[a] = (1−α^{k+1})/(1−α)` on net/host (long prefixes, position-
  dependence mild) and collapses to `a ≈ H_ε ≈ small` on the SPEC-2.1 filesystem world (K4: the first error
  is past `ε`). *Refuted if* the network/host prefixes are *also* short (drift is discrete-and-early
  everywhere — speculative cannot help any world, a strong unifying negative) or if the filesystem prefix is
  long (K4 reset-resistance does not bind here — a surprising positive re-opening the knee hunt on discrete
  worlds). Tested as **SR2** (the gate that predicts SR1's per-world outcome).

- **H41 — calibrated draft length beats fixed draft length (the EAGLE-2 link, against the oracle).** Choosing
  `k` from `belief_var`/decode-entropy (draft longer where the model is confident) advances more faithful
  steps per oracle call than a fixed `k`, *and* `belief_var`/entropy correlates with measured per-step
  acceptance `α` (the EAGLE-2 confidence↔acceptance calibration, checked against ground truth). *Refuted if*
  calibrated `k` ties fixed `k` **or** the confidence signal is uncorrelated with acceptance — the latter is
  the EH2-style negative (uncalibrated belief, as on the flat arm) lifted to draft-length choice, banking
  that the structured arm's uncertainty is not acceptance-predictive. Tested as **SR4**.

- **H42 — multi-draft (tree) verification raises the accepted prefix at near-zero extra oracle cost.**
  Verifying several candidate rollouts against one oracle trajectory and keeping the longest-faithful branch
  advances more faithful steps per oracle call than single-draft, since the oracle pass is shared (the
  SpecInfer move). *Refuted if* the best-of-`m` prefix is no longer than the single-draft prefix (the model's
  draft variance does not span the faithful direction — its errors are systematic, not stochastic) — banking
  that `M_θ`'s divergence is bias, not variance, which itself licenses a debias/correction step over a wider
  tree. Tested as **SR3**.

- **H43 — a cheap drafter pre-filtering for a larger one reduces oracle calls per faithful step.** A two-tier
  cascade (tiny `M_θ` drafts; larger `M_θ` re-drafts only the rejected suffix; oracle verifies last) reaches
  a target faithful horizon with fewer *oracle* calls than the single-model speculative policy at equal
  faithfulness. *Refuted if* the cascade needs as many or more oracle calls (the cheap drafter's rejections
  are not recoverable by the larger model without the oracle — the two models fail in the same places),
  banking that model-vs-model speculation buys nothing when only the *oracle* can adjudicate, a clean result
  about where the cheapness lives (the oracle, not the GPU). Tested as **SR5**.

- **H44 — the speculative-vs-fixed gap is governed by world discreteness, not by world identity (the unifying
  fork).** Across all three worlds the *size* of speculative's win over fixed is predicted by the measured
  `α(t)` drift profile (H40), monotone in how gradual the drift is — so a single curve (win vs drift-
  gradualness) collapses network, host, and filesystem onto one law. *Refuted if* the per-world wins do not
  collapse onto the `α(t)` profile (some world wins/loses for a reason orthogonal to drift gradualness),
  which would mean error *placement* is not the sole governor and re-opens what else sets the floor. Tested
  as **SR6** (deferred fork; runs after SR1–SR2 land on all three worlds).

---

## 6. Experiments (prefix **SR** — speculative rollout; network + host + SPEC-2.1 filesystem worlds)

Each follows the house template: a `Config` dataclass with `from_json_file`, a CLI entry point, a JSONL
record stream, a `plot_*.py` emitting a committed `.png` + `.csv`, regenerable from `reproduce.sh`,
deterministic and seeded (SPEC-2 §12). All reuse shipped apparatus (the loop, the oracles, the
`imagine`/`verify` draft-verify, `faithful_horizon`, `belief_var`) — SPEC-13 adds one new consultation
policy plus these harnesses.

- **SR2 — the accepted-prefix law per world (H40).** *Runs first; gates SR1's predicted outcome.* For each
  world, free-run `M_θ` from many anchored states and verify step by step against the oracle; record per-step
  faithfulness `α(t)`, the accepted-prefix length distribution, and the fit/residual to
  `E[a] = (1−α^{k+1})/(1−α)`. Output: the per-world `α(t)` drift profile + accepted-prefix histogram, and the
  network/host-vs-filesystem split (the K4 prediction made visible). `experiments/sr2.py`, `configs/sr2.json`.

- **SR1 — speculative vs fixed-`ρ` at equal budget (H39, the headline).** Add a `SpeculativeConsult` policy
  to [`loop/policy.py`](../../src/verisim/loop/policy.py) (draft `k`, one oracle verify of the window, accept
  longest within-`ε` prefix, re-anchor at first exceedance). Run it through `run_net_rollout` /
  `HostSimulator.verify` against `FixedInterval` at *identical* total budget (the spend-down backstop
  guarantees parity). Report faithful-steps-per-oracle-call and `H_ε(ρ)` for both, per world. **The figure
  that shows (or refutes) the favorable knee.** `experiments/sr1.py`, `configs/sr1.json`.

- **SR3 — multi-draft (tree) verification (H42).** Sample `m` candidate rollouts per anchor
  (temperature/top-k over the grammar decode; dropout on the RSSM), verify all against one oracle trajectory,
  accept the longest-faithful branch. Sweep `m`; report best-of-`m` accepted prefix vs single-draft and the
  oracle-cost parity (one pass regardless of `m`). `experiments/sr3.py`, `configs/sr3.json`.

- **SR4 — calibrated draft length (H41, the EAGLE-2 link).** Choose `k` per step from `belief_var` /
  decode-entropy; compare to fixed `k` at equal budget; and, as a standalone panel, the calibration curve —
  measured per-step acceptance `α` vs `belief_var`/entropy bin, with bootstrap CIs (the confidence↔acceptance
  link checked against the oracle). `experiments/sr4.py`, `configs/sr4.json`.

- **SR5 — two-tier self-speculative cascade (H43).** Tiny `M_θ` drafts (the SPEC-5 flat baseline / a
  low-capacity graph arm via the `NetModel` seam); larger `M_θ` re-drafts only the rejected suffix; oracle
  verifies last. Report oracle calls per faithful step vs the single-model speculative policy at equal
  faithfulness. `experiments/sr5.py`, `configs/sr5.json`.

- **SR6 — the discreteness law (H44, deferred fork).** Collapse SR1's per-world speculative-vs-fixed win
  against SR2's `α(t)` drift-gradualness onto one curve. Runs only after SR1/SR2 land on all three worlds.
  `experiments/sr6.py`, `configs/sr6.json`.

---

## 7. What is confidently buildable now vs gated on a result

The user's standing rule — *build what is confidently positive, experiment on what is not* — maps onto the
dependency order:

- **Confidently buildable now (machinery exists, result near-certain):**
  - The **`SpeculativeConsult` policy** itself. It is a drop-in `ConsultationPolicy`
    ([`loop/policy.py`](../../src/verisim/loop/policy.py)): draft `k`, verify the window, accept the longest
    within-`ε` prefix (which the loop *already computes* as `plan_faithful_horizon` /
    [`faithful_horizon`](../../src/verisim/metrics/horizon.py)), re-anchor. This is wiring an existing
    quantity into a scheduling decision, not a new primitive.
  - The **SR2 accepted-prefix measurement.** Free-run + step-verify is exactly what `HostSimulator.verify`
    and `run_net_rollout` do; SR2 just records `α(t)` and the prefix distribution. Deterministic, GPU-free
    on the symbolic baselines.
- **Gated on SR2 (the §4 branch point):** *which worlds SR1 should win on.* SR2's `α(t)` profile predicts
  the gradual-vs-discrete split; it is cheap and runs first.
- **The genuine bets (must be measured):** SR1 (does consulting-at-the-break beat the clock at equal budget —
  the headline), SR3 (does the model's draft variance span the faithful direction), SR4 (is `belief_var`
  acceptance-predictive — the EAGLE-2 link), SR5 (does model-vs-model speculation save *oracle* calls). SR1
  on the *gradual* worlds is high-confidence positive (the accepted prefix is long there, by SR2), but "the
  knee appears under a real equal-budget comparison" is exactly the claim the program measures before
  asserting — and on the filesystem world SR1 is a high-confidence *negative* (K4), banked as a localization
  result.

**Recommended build order:** SR2 (the per-world `α(t)` law; predicts everything) → SR1 (the headline knee,
net+host expected win, filesystem expected tie) → SR4 (calibrated `k`) → SR3 (tree drafts) → SR5 (cascade) →
SR6 (the discreteness collapse). Each rung graduates on a committed figure or a banked negative (SPEC §10.1,
§12).

---

## 8. Scope, non-goals, honest caveats

- **This is a policy, not a new world or model.** SPEC-13 runs on the SPEC-5 network, SPEC-6 host, and
  SPEC-2.1 filesystem worlds and reuses every oracle. Over-claiming a new dynamics result is the failure
  mode this line forbids; the result is about *scheduling the oracle*, measured against the existing loop.
- **The headline is pre-split by world type.** The filesystem-world tie is *predicted* (K4 reset-resistance),
  not a surprise; reporting it as a speculative "failure" would be dishonest — it is the localization result.
  The genuine open question is whether the network/host win *survives a real equal-budget comparison*.
- **"Optimal by construction" is a within-policy-class claim, not a global one.** Accept-longest-correct-
  prefix is optimal *given a draft* (it wastes no faithful step and no oracle call on a clock), but it does
  not make a *bad* drafter good — if `α(t)` is low, the speculative speedup is near 1 and there is no knee.
  SPEC-13 measures the drafter's `α(t)` (SR2) precisely so this is not assumed away.
- **The oracle, not the GPU, is the expensive resource here** — the inverse of LLM speculative decoding,
  where the *target model* is expensive and the drafter is the cheap GPU. So self-speculative cascading (SR5)
  is only a win if the cheap model *reduces oracle calls*; saving GPU drafting time is not a verisim cost.
  This inversion is stated so the self-speculative result is read correctly.
- **Equal-budget parity is load-bearing and already shipped.** The `run_net_rollout` spend-down backstop
  forces every policy to spend *exactly* `budget`, so speculative-vs-fixed is a true equal-`ρ` comparison;
  any reported knee is not a budget artifact. The host loop needs the same backstop added for SR1.
- **Acceptance is position-dependent.** The i.i.d. `α` speedup law is a baseline; SR2 reports the *empirical*
  prefix distribution against it, so the known position-dependence of acceptance is measured, not assumed.

---

## 9. Build, reproduce, CI

### 9.1 Module layout (additive only)

```
src/verisim/loop/
  policy.py            # EDIT (additive): add SpeculativeConsult (draft k, verify window, accept prefix)
src/verisim/loop/
  speculative.py       # NEW — draft-window construction, tree-draft sampling, prefix acceptance
src/verisim/experiments/
  sr1.py … sr6.py      # NEW — the SR experiments
figures/
  plot_sr1.py … plot_sr6.py   # NEW — committed-figure generators
configs/
  sr1.json … sr6.json         # NEW — committed sweep configs
```

The policy consumes the shipped `NetModel`/`HostModel` proposer seam (SPEC §6 commitment 4), the loops'
PROPOSE/VERIFY/CORRECT structure, `faithful_horizon`, and `belief_var` **unchanged** — the one edit is the
additive `SpeculativeConsult` policy alongside the existing `FixedInterval`/`UncertaintyTriggered`/
`DriftTriggered`. Nothing in the deterministic core, the oracles, the metrics, or any existing experiment is
modified.

### 9.2 `reproduce.sh` (new SR block, in dependency order)

```bash
echo "== SR2: the accepted-prefix law per world (H40) — predicts the gradual/discrete split =="
python -m verisim.experiments.sr2 --config configs/sr2.json --out runs/sr2/records.jsonl --plot figures/sr2_accept_law.png
echo "== SR1: speculative vs fixed-rho at equal budget (H39) — THE HEADLINE KNEE =="
python -m verisim.experiments.sr1 --config configs/sr1.json --out runs/sr1/records.jsonl --plot figures/sr1_knee.png
echo "== SR4: calibrated draft length + the EAGLE-2 confidence<->acceptance link (H41) =="
python -m verisim.experiments.sr4 --config configs/sr4.json --out runs/sr4/records.jsonl --plot figures/sr4_calibration.png
# SR3 (tree drafts), SR5 (cascade) follow; SR6 (discreteness collapse) gated on SR1+SR2 across worlds.
```

The non-cascade SR block runs on CPU (the symbolic baselines and the deterministic oracles); the trained
`M_θ` arms are the only GPU dependency and are `skipif`-guarded with disclosure (never counted as a result
when skipped). CI (`ubuntu-latest`) stays the free Linux confirmation; SR tests assert *structural*
invariants — the accepted prefix equals `faithful_horizon` of the draft window; speculative never spends more
than `budget`; on the SPEC-2.1 filesystem world the accepted prefix is short (the K4 invariant) — not exact
magnitudes, so the same tests pass on the macOS primary host and Linux CI (the macOS-first principle).

---

## 10. Provenance & reading order

SPEC-13 is the consultation-policy program (SPEC-3) read through the speculative-decoding lens: the loop the
whole program is built on — PROPOSE (`M_θ`) / VERIFY (oracle) / CORRECT — *is* a speculative step, and its
`faithful_horizon` *is* the accepted prefix. SPEC-2.1's K4 (the reset-resistant cliff) supplies the honest
negative this spec must survive and turns into a localization result; SPEC-12 supplies the altitude lesson
(schedule the oracle, do not just roll forward). The mechanism is speculative decoding (Leviathan et al.,
ICML 2023; Chen et al., 2023) and its tree/calibration descendants (SpecInfer, Medusa, EAGLE/EAGLE-2/
EAGLE-3, lookahead decoding); the unique contribution is the **exact-oracle verifier** — accept-longest-
*faithful*-prefix instead of accept-longest-*self-consistent*-prefix — which only computer worlds can offer
(SPEC §2). It is a *method* spec: it advances how the program *uses* its oracle budget, not what the model is.

Reading order for a newcomer: [SPEC.md §5](./SPEC.md) (the `π_c`/`ρ`/`H_ε` formalism) →
[SPEC-2.1 §8](./SPEC-2.1.md) (K4 — the reset-resistant cliff this spec is pre-split around) →
[`loop/policy.py`](../../src/verisim/loop/policy.py) and
[`hostsim/simulator.py`](../../src/verisim/hostsim/simulator.py) (the loop that already *is* draft-verify) →
this document (§1 motivation, §3 architecture, §4 the K4 pre-registration, §5 the hypotheses) →
`src/verisim/loop/speculative.py` and `src/verisim/experiments/sr2.py` (the concrete build, once shipped).

---

## 11. Status (2026-06-08) — SR1–SR6 SHIPPED (committed CPU core; trained-`M_θ` arm deferred)

The full SR experimental program is built and committed. As pre-registered (§7, §9), the committed core
uses a **controlled stand-in drafter** — a proposer that predicts the oracle's true next state with
per-step probability `α` and otherwise *stalls* (predicts no change), with `α` held identical across
worlds so the figures isolate the *world's* contribution, not the proposer's (the LP7 discipline: the
trained-`M_θ` `belief_var`/decode-entropy arm is `skipif`-guarded and never counted). The primitive is
[`loop/speculative.py`](../../src/verisim/loop/speculative.py) (`speculative_rollout`,
`fixed_interval_rollout`, `accepted_prefix_law`, `free_run_divergences`); the policy descriptor is
`SpeculativeConsult` in [`loop/policy.py`](../../src/verisim/loop/policy.py); the world-generic bundles
and the two controlled drafters (`StallDrafter`, `VaryingDrafter`) are in
[`experiments/sr_common.py`](../../src/verisim/experiments/sr_common.py). All six run on CPU,
deterministic and seeded, across the SPEC-5 network, SPEC-6 host, and SPEC-2.1 filesystem worlds.

**The headline reframing (made before any policy was run, SR2/§4).** The pre-registered *world-identity*
split (network/host gradual, filesystem discrete) is not the controlling variable. The actual governor
is the **dimensionless ratio `g = ε/δ`**, where `δ` is the world's *single-edit divergence granularity*
(the median divergence one missed edit produces). When `g ≥ ~2` several edits fit under `ε` before the
prefix breaks (gradual); when `g < 1` the first missed edit already exceeds `ε` (the K4 cliff). The SR
figures sweep `g` and show the worlds collapse onto one law in `g` — the result is governed by the
metric's granularity, not the world's identity. This *sharpens* K4 from a per-world claim into a
per-metric one, and points the next move at the metric/representation (an edit-distance-graded `ε`),
exactly as §4 anticipated.

- ✅ **SR2 — the accepted-prefix law (H40 SUPPORTED).** Per world, the empirical mean accepted prefix
  grows with `g` and tracks the i.i.d. law `E[a] = α(1−α^k)/(1−α)` fed the *measured* per-step
  acceptance `α̂` (residual = position-dependence). Discrete-regime prefix ~3.7 vs gradual ~11.7.
  [`sr2`](../../src/verisim/experiments/sr2.py), [`figures/sr2_accept_law.png`](../../figures/sr2_accept_law.png).
- ✅ **SR1 — speculative vs fixed-ρ at equal budget (H39 SUPPORTED above ρ\*, REFUTED below — the
  headline).** At *equal expensive budget* (matched oracle corrections) there is a **budget crossover
  ρ\***: above it speculative reaches **full faithfulness** (consult-at-break wastes no clock tick on a
  still-faithful step), below it fixed's *uniform spread* wins because accept-longest-prefix is
  **budget-greedy** — it spends corrections early and free-runs the tail. `ρ\* ≈ 0.10` (network) /
  `0.13` (host) / `0.20` (filesystem). The no-knee floor (EN7/H22) is *escaped* at sufficient budget; the
  scarce-budget loss is banked as a real property of reactive scheduling.
  [`sr1`](../../src/verisim/experiments/sr1.py), [`figures/sr1_knee.png`](../../figures/sr1_knee.png).
- ✅ **SR3 — multi-draft (tree) verification (H42 SUPPORTED).** best-of-`m` lifts the accepted prefix
  ~2.3× under **variance** (independent stalls) and is **flat under bias** (systematic stalls) — a draft
  tree helps iff the drafter's divergence is stochastic; systematic error needs debiasing, not more
  drafts. [`sr3`](../../src/verisim/experiments/sr3.py), [`figures/sr3_tree.png`](../../figures/sr3_tree.png).
- ✅ **SR4 — calibrated draft length & the EAGLE-2 link (H41 SPLIT: link transfers, policy does not).**
  The confidence↔acceptance link *transfers* (calibration slope ~+0.22 with a calibrated signal vs ~0
  with the null, the ED2-smart/EH2 case), but **calibrated-`k` does not beat draft-long-everywhere** —
  the oracle-cost inversion (§8): the verify *stops at the first divergence*, so a long draft that
  rejects early costs no more than a short one, and calibrating `k` down only adds oracle calls. A
  bankable negative whose premise holds and whose mechanism is absent.
  [`sr4`](../../src/verisim/experiments/sr4.py), [`figures/sr4_calibration.png`](../../figures/sr4_calibration.png).
- ✅ **SR5 — the two-tier self-speculative cascade (H43 REFUTED — banked negative).** A cheap drafter
  pre-filtering for a larger one does **not** cut *oracle* calls per faithful step versus running the
  larger drafter directly — only the oracle adjudicates faithfulness, and the cheap tier adds a verify
  round without removing one. The cheapness the self-speculative line exploits lives on the GPU (free
  here), not in the oracle. [`sr5`](../../src/verisim/experiments/sr5.py), [`figures/sr5_cascade.png`](../../figures/sr5_cascade.png).
- ✅ **SR6 — the discreteness law / g-collapse (H44 PARTIALLY SUPPORTED — deferred fork).** The
  speculative-vs-fixed win is **hump-shaped in `g`** (small at the K4 cliff, small once free-run is
  already faithful, peaking in the transition band); the worlds share the shape but not exactly the peak
  (the network saturates at lower `g`), so `g` governs the *shape* across worlds but the collapse is
  approximate. [`sr6`](../../src/verisim/experiments/sr6.py), [`figures/sr6_discreteness.png`](../../figures/sr6_discreteness.png).

Remaining: only the **trained-`M_θ` arm** (the one external/GPU dependency) — the rest of the SPEC-13
program (SR1–SR6, network + host + filesystem) is shipped on the controlled-drafter core, each rung
graduating on a committed figure or a banked negative.
