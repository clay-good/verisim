# SPEC-15 — Oracle-Calibrated Conformal Consultation: Coverage-Guaranteed Verification

**Method specification: every consultation policy `π_c` the program has measured so far is a *heuristic* —
"consult when the signal is high," with the threshold hand-set and the payoff (does it beat fixed?) decided
empirically per arm. The result is a split nobody has a *theory* for: the host's calibrated RSSM
belief-variance trigger beat fixed-interval ~2.2× (EH2/H9), the flat arm's decode-entropy trigger lost to
fixed and lost *badly* (ED2-smart, lift 0.08–0.12×). This spec turns the trigger into a *guarantee*. The
oracle is a free, exact, unlimited calibration set — the textbook precondition for CONFORMAL PREDICTION —
so we can set the consultation threshold to GUARANTEE `P(undetected divergence > ε) ≤ α` distribution-free,
finite-sample, and predict *which arm's trigger will work* from whether its signal is conformalizable.**

> **◐ DESIGN — METHOD SPEC, 2026-06.** A *method* spec in the lineage of [SPEC-12](./SPEC-12.md): it
> invents **no new world** (it runs on the SPEC-5 network world and its shipped flat + graph arms) and
> **no new oracle** (it reuses the data-plane `ReferenceNetworkOracle` for exact divergence and the
> `ControlPlaneOracle` for the cheap reachability projection). What it adds is a *calibration altitude*
> on top of the propose–verify–correct loop: a conformal calibrator that consumes the oracle's free
> ground truth and emits a threshold on the model's existing uncertainty signal
> (`NetUncertaintyModel.predict_delta_with_uncertainty` → `belief_var` for the graph arm, decode-entropy
> for the flat arm) with a *coverage guarantee* the heuristic triggers never had. Inspiration is
> acknowledged plainly: split/inductive conformal prediction (Vovk–Gammerman–Shafer), the modern
> distribution-free toolkit (Angelopoulos–Bates), conformal **risk** control (Bates et al.; Angelopoulos
> et al.), and — load-bearing here — the **online/adaptive** conformal line (Gibbs–Candès ACI; conformal
> PID) that exists precisely because autoregressive rollout states are **not exchangeable**, which is the
> one assumption vanilla conformal needs and the one this setting violates. The verisim contribution is
> the one thing oracle-free domains structurally cannot do: **a calibration set that is exact and free,
> so the coverage guarantee is real rather than asymptotic-and-hoped — and a measurement of when the
> guarantee *holds* under drift, which is the honest core of the program.**

Read [SPEC.md §5](./SPEC.md) (the loop and `π_c`), [SPEC.md §7](./SPEC.md) (faithfulness metrics; the
calibration-of-uncertainty-vs-divergence diagnostic this spec upgrades from *measured* to *guaranteed*),
[SPEC-7 §9.4](./SPEC-7.md) (belief calibration and the ED2-smart null), and the shipped policy seam
[`loop/policy.py`](../../src/verisim/loop/policy.py) (`ConsultationPolicy`, `StepContext`,
`UncertaintyTriggered` — the exact interface a conformal threshold drops into).

---

## 0. One-paragraph thesis

RQ2 asks whether a *smart* consultation policy beats a *fixed-interval* one at equal oracle budget `ρ`. The
program's answer is a contradiction it cannot yet explain: the host's calibrated belief-variance trigger
**won** (EH2/H9, ~2.2× over fixed), but the flat arm's decode-entropy trigger **lost, hard** (ED2-smart,
lift 0.08–0.12×; the network/distributed flat `M_θ` repeated it). The repo's own reading is that
decode-entropy is "a decode-time artifact, not a calibrated belief" — correct, but stated as a *property of
the signal*, with no instrument to decide it before running the whole sweep. This spec supplies the
instrument and an upgrade in kind. The oracle gives, for free, the exact next state and therefore the exact
per-step divergence — a perfect, unlimited *calibration set*. Feed it to **split conformal prediction**:
choose the threshold `τ` on the uncertainty signal as the `(1−α)`-quantile of the signal over the
calibration steps where divergence stayed within `ε`, and the resulting trigger carries a distribution-free,
finite-sample guarantee — `P(undetected divergence > ε) ≤ α` — that **no oracle-free domain can make**,
because none has an exact calibration truth. The claim, stated to be falsified: **a conformally-calibrated
consultation trigger hits its target coverage `α` and reaches that coverage at *lower* `ρ` than
fixed-interval — but only for a signal that is conformalizable, and `belief_var` is while flat
decode-entropy is not, which is precisely the EH2-yes / ED2-smart-no split given a mechanism.** Every branch
is bankable (SPEC §10.1): if conformal triggers do not beat fixed at equal `ρ`, the limit is the *signal*
(conformal inherits whether the base uncertainty is informative — exactly the ED2-smart reading), not the
calibration method, and that is a clean finding. And the deepest pre-registered risk is structural:
autoregressive rollout states are **not exchangeable**, so vanilla split-conformal coverage may *fail to
hold* under drift — itself the cleanest possible motivation for the online/adaptive conformal arm (ACI),
and a result the program would publish either way.

---

## 1. Why now: a split with no theory, and the calibration set that gives it one

The faithful-horizon program treats *calibration* — does the model's stated uncertainty predict its actual
divergence? — as a **measured** quantity (SPEC §7; SPEC-7 §9.4: "does RSSM belief variance predict
error?"). It has never had a **guarantee** on it. That gap is exactly why RQ2 is a contradiction rather than
a law:

- **EH2/H9 (host, structured arm): smart won.** The factored arm's RSSM belief-variance trigger beat
  fixed-interval ~2.2× at equal `ρ`. The signal is "calibrated by construction" (it is a posterior variance,
  not a decode statistic).
- **ED2-smart (distributed, flat arm): smart lost, strictly.** Entropy-gated consultation was *worse* than
  fixed (lift 0.08–0.12× at every budget), because faithful horizon is a *prefix* property and `fixed`
  consults at step 0 to protect the prefix while the entropy trigger spends late and lets the model derail.
  The network and distributed flat arms repeated the null.

The repo's diagnosis — "entropy is a decode-time artifact, not a calibrated belief" — is right but
*untooled*: there is no way to decide, before paying for the full sweep, whether a given signal is
trigger-worthy. Conformal prediction is that tool, and the oracle is what makes it apply here without
compromise. Two facts make verisim the right and perhaps the *only* clean place to do it:

- **The calibration set is free, exact, and unlimited.** Split conformal needs a held-out calibration set
  with ground-truth labels. Every other conformal application pays for labels and rations them; verisim's
  oracle emits the exact next state, hence exact per-step divergence, for free at any volume
  ([`ReferenceNetworkOracle`](../../src/verisim/netoracle/base.py)). The finite-sample guarantee is *real*,
  not an asymptotic stand-in.
- **The signal already exists at the seam the trigger plugs into.**
  `NetUncertaintyModel.predict_delta_with_uncertainty` ([`netloop/model.py`](../../src/verisim/netloop/model.py))
  returns `(delta, belief_var)`; the graph arm's `belief_var` is the RSSM posterior variance
  ([`netmodel/graph_model.py`](../../src/verisim/netmodel/graph_model.py), `encode` → `belief_var`), the
  flat arm's is decode-entropy. The conformal threshold is a `ConsultationPolicy`
  ([`loop/policy.py`](../../src/verisim/loop/policy.py)) — a one-line generalization of the shipped
  `UncertaintyTriggered(τ)` where `τ` is *calibrated* rather than hand-set.

The one subtlety, surfaced now rather than buried: split conformal's guarantee assumes **exchangeability**,
and an autoregressive rollout is the canonical violation — step `t+1`'s state is the model's own (drifting)
output, not an i.i.d. draw. So the headline guarantee is split into a *static* claim (conformal on i.i.d.
teacher-forced steps, where exchangeability holds and the guarantee is clean) and a *rollout* claim (where
it may break and the online/adaptive arm is needed). The program's instinct — measure the assumption, bank
the failing branch — is the spec's spine.

---

## 2. The lineage folded in (and the design choice each one forces)

### 2.1 Split / inductive conformal prediction (the calibrator)

- **Algorithmic Learning in a Random World** (Vovk, Gammerman, Shafer, 2005; 2nd ed. 2022) and the
  inductive/split formulation. A nonconformity score `s(x)`; the threshold is the `⌈(1−α)(n+1)⌉/n`-quantile
  of calibration scores; any new point's set has marginal coverage `≥ 1−α`, distribution-free, for
  *exchangeable* data. → **Design choice:** the nonconformity score *is* the model's uncertainty signal
  (`belief_var` / decode-entropy); the calibration set is oracle-labeled teacher-forced steps; the
  threshold `τ_α` is the conformal quantile. One score, one quantile, a guarantee — the whole CF1 apparatus.
- **A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification**
  (Angelopoulos & Bates, 2021, arXiv:2107.07511). The modern recipe + the marginal-vs-conditional caveat
  (coverage holds *on average*, not per-input). → **Design choice:** report **conditional** coverage slices
  (by rollout depth, by region) not just marginal — the marginal guarantee can hide exactly the
  prefix-vs-tail failure ED2-smart exposed.

### 2.2 Conformalized quantile regression (the adaptive-width score)

- **Conformalized Quantile Regression** (Romano, Patterson, Candès, NeurIPS 2019, arXiv:1905.03222). Wrap a
  quantile regressor in conformal so interval *width* adapts to local difficulty while keeping coverage. →
  **Design choice:** the divergence-aware variant of the score — calibrate a *region-conditional* threshold
  (heteroscedastic `τ(region)`) rather than one global `τ`, since `H_ε` is a region property (EN7/H22). CF1's
  ablation: global `τ` vs region-conditional `τ`.

### 2.3 Risk control: from coverage to "undetected breach rate" (the metric that matters)

- **Distribution-Free, Risk-Controlling Prediction Sets** (Bates, Angelopoulos, Lei, Malik, Jordan,
  arXiv:2101.02703) and **Conformal Risk Control** (Angelopoulos, Bates, Fisch, Lei, Schuster,
  arXiv:2208.02814), which control the expectation of any *monotone* loss, not just miscoverage. → **Design
  choice:** the risk verisim actually cares about is not "the prediction set covers the next delta" but
  "the rollout's faithful horizon is not silently breached." That is a monotone loss in `τ` (raise the
  threshold → consult less → more undetected breaches), so conformal risk control sets `τ` to guarantee
  `E[undetected-breach] ≤ α` directly. CF3 is this experiment.

### 2.4 Online / adaptive conformal (the non-exchangeability fix — load-bearing)

- **Adaptive Conformal Inference Under Distribution Shift** (Gibbs & Candès, NeurIPS 2021,
  arXiv:2106.00170). When data are not exchangeable / the distribution drifts, update the level `α_t` online
  from realized coverage (`α_{t+1} = α_t + γ(α − err_t)`); achieves the long-run coverage frequency
  *regardless of the data-generating process*. → **Design choice:** the rollout calibrator. An
  autoregressive rollout drifts by construction, so the static split-conformal `τ` may lose coverage as the
  state leaves the calibration distribution; ACI re-tunes `τ_t` from each step's oracle-revealed
  hit/miss. CF2 measures static-vs-ACI coverage along the rollout.
- **Conformal PID Control for Time Series Prediction** (Angelopoulos, Candès, Tibshirani, NeurIPS 2023,
  arXiv:2307.16895). Treats online conformal as a control problem; P+I terms track and integrate coverage
  error (the D/scorecaster term forfeits the guarantee). → **Design choice:** the CF2 stretch — a P+I
  threshold controller as a stronger drift-tracker than vanilla ACI, with the guarantee-preserving terms
  only. *(Stretch; ACI is the committed baseline.)*

### 2.5 What the program already built that this sits on

- **The policy seam exists and is exactly the right shape.** `ConsultationPolicy.should_consult(ctx)` with
  `StepContext(step, signal, cumulative_signal)` ([`loop/policy.py`](../../src/verisim/loop/policy.py)) is
  the drop-in point. A conformal trigger is `UncertaintyTriggered` with a *calibrated* `τ`; an ACI trigger
  is the same with `τ` updated per step. The runner already enforces the budget on top of any policy's
  proposals (SPEC-2 §16 invariant), so CF comparisons are at truly equal `ρ` for free.
- **The uncertainty signal exists for both arms**, behind one protocol
  ([`netloop/model.py`](../../src/verisim/netloop/model.py)) — so "is *this* signal conformalizable?" is
  asked identically of `belief_var` and decode-entropy, which is what makes CF4 a controlled explanation of
  the EH2/ED2-smart split rather than two unrelated runs.
- **Both oracles ship** for the score's truth: data-plane for exact divergence, control-plane for the cheap
  reachability projection ([`netoracle/control_plane.py`](../../src/verisim/netoracle/control_plane.py)) —
  so the score can be defined on bytes *or* on reachability, the EN10 strong projection.

---

## 3. The method: a calibrated threshold with a coverage guarantee (three layers)

```
   Calibration (free, exact)   teacher-force M_θ over held-out steps; for each, record
                               (uncertainty signal s_i, oracle divergence d_i)
                                       │  score s_i, label [d_i ≤ ε]
                                       ▼
   Conformal calibrator        τ_α = (1−α)-quantile of { s_i : d_i ≤ ε }  (split conformal, §2.1)
                               region-conditional τ_α(r) (CQR-style, §2.2)  OR
                               risk-controlled τ_α for E[undetected breach] ≤ α (§2.3)
                                       │  threshold τ_α  →  ConsultationPolicy
                                       ▼
   Consultation (the loop)     UncertaintyTriggered(τ_α): consult iff signal s_t > τ_α
                               ROLLOUT: ACI updates τ_t from oracle hit/miss (§2.4, handles drift)
                                       │  guarantee: P(undetected divergence > ε) ≤ α
                                       ▼
   Oracle (ground truth)       data-plane ReferenceNetworkOracle (exact)  +  ControlPlaneOracle (reach)
```

Four commitments, each tied to a measured verisim fact:

1. **The threshold is calibrated, not chosen.** `τ_α` is a conformal quantile of the oracle-labeled score —
   the upgrade of the shipped `UncertaintyTriggered(τ)`'s hand-set `τ`. The guarantee is the deliverable.
2. **The signal is the model's existing one.** No new head, no new training: `belief_var` (graph arm) and
   decode-entropy (flat arm) are conformalized *as-is*, so the result is a property of the *signal*, which
   is what explains the EH2/ED2-smart split.
3. **Exchangeability is tested, not assumed.** The static guarantee is clean only on i.i.d. teacher-forced
   steps. On the autoregressive rollout it may break; ACI is the banked fix and CF2 measures the gap.
4. **Coverage is reported conditionally.** Marginal coverage can pass while the *prefix* silently fails
   (the ED2-smart mechanism). CF1/CF2 slice coverage by rollout depth and region.

---

## 4. The load-bearing assumption (and its free fallback)

Split conformal's guarantee rests on **exchangeability** of the calibration and test scores. On i.i.d.
teacher-forced steps that holds and the guarantee is exact. On an autoregressive rollout it does *not*: the
test point at step `t` is the model's own drifting output, distributed unlike the calibration draws. The
honest pre-registration:

- **If exchangeability approximately holds along the rollout** (early steps, slow drift), static
  split-conformal `τ_α` keeps coverage and the headline (CF1) transfers to rollout directly.
- **If it breaks** (the expected case at depth), verisim has a fallback no static method has: **ACI
  re-tunes `τ_t` online from the oracle's free, exact per-step hit/miss** (§2.4). The oracle that breaks
  exchangeability (by being the thing the loop consults) is the same oracle that *repairs* coverage online.
  Either branch yields a coverage statement; the branch point only decides whether the *static* calibrator
  suffices, which is itself a clean result about how fast the rollout leaves its calibration distribution —
  a conformal restatement of the faithful-horizon decay.

This is the program's epistemic engine at design time: the assumption that quietly invalidates naive
conformal-on-rollouts is, here, a *measurement* (CF2) with a banked alternative (ACI) on the failing branch.

---

## 5. Hypotheses (pre-registered, continuing the global H-ID space past SPEC-14's H49)

- **H50 — the oracle-calibrated trigger hits target coverage on exchangeable steps.** On i.i.d.
  teacher-forced held-out steps, the split-conformal threshold `τ_α` on `belief_var` achieves empirical
  undetected-divergence rate `≤ α` (within finite-sample slack `1/(n+1)`) across `α ∈ {0.01,0.05,0.10,0.20}`.
  *Refuted if* empirical miscoverage materially exceeds `α` on exchangeable data — which would indict the
  conformal implementation itself (a bug, not a finding) and must be fixed before any rollout claim. Tested
  as **CF1**.

- **H51 — calibrated coverage at lower `ρ` than fixed (THE headline, RQ2 with a guarantee).** At a matched
  target undetected-breach rate `α`, the conformal `belief_var` trigger reaches that coverage at *lower*
  oracle budget `ρ` than fixed-interval — i.e. a *guaranteed* version of the EH2/H9 win, now stated as
  "fewer consults to certify the same safety" rather than "more horizon at equal `ρ`." *Refuted if* the
  conformal trigger needs `≥` the fixed budget to hit `α` — in which case the calibration method buys a
  guarantee but no efficiency, and the limit is the signal's informativeness (the banked ED2-smart reading,
  §5/H53). Tested as **CF1** (the `ρ`-vs-coverage frontier).

- **H52 — vanilla conformal loses coverage under autoregressive drift; ACI restores it.** Static
  split-conformal `τ_α`, applied unchanged along the *rollout*, sees its empirical undetected-breach rate
  drift *above* `α` with rollout depth (the exchangeability violation, §4), while ACI's online `τ_t`
  recovers the long-run target rate. *Refuted if* static conformal holds coverage along the full rollout
  (drift is slow enough that exchangeability is effectively preserved — a strong, clean positive that makes
  the online machinery unnecessary on this world and is itself worth reporting), *or if* ACI also fails to
  recover (the drift is too violent for any online method here — a deep negative bounding the approach).
  Tested as **CF2**.

- **H53 — conformalizability explains the EH2-yes / ED2-smart-no split (the mechanism).** Run the identical
  conformal calibration on `belief_var` (graph arm) and on flat decode-entropy. The `belief_var` calibration
  yields a threshold whose *test* coverage matches its *target* (the score is calibrated: high score ⇒ high
  divergence), while decode-entropy's calibration is *miscoverage-prone or efficiency-null* — its score does
  not separate within-ε from breach steps, so any threshold is no better than fixed. This is the measured,
  mechanistic statement of "entropy is a decode artifact, not a calibrated belief." *Refuted if*
  decode-entropy conformalizes as cleanly as `belief_var` (then ED2-smart's null was a tuning artifact, not
  a signal property — a surprising positive that re-opens the flat-arm trigger) *or if* `belief_var` fails to
  conformalize (then EH2/H9's win does not survive the guarantee framing — the headline's deepest negative).
  Tested as **CF4**.

- **H54 — risk control on the right loss: bounded undetected faithful-horizon breach.** Conformal *risk*
  control (§2.3) sets `τ_α` to bound the expectation of the program's actual loss — undetected faithful-
  horizon breach (a monotone function of `τ`) — and the realized breach expectation is `≤ α` at a budget
  competitive with the coverage-only trigger (H51). *Refuted if* the risk-controlled threshold is no
  tighter than the coverage-only one (the loss and the miscoverage event coincide here, so risk control adds
  nothing beyond split conformal — a clean simplification, banked) *or if* it cannot hit `α` at any
  affordable `ρ` (the breach loss is too heavy-tailed for a single threshold — motivating per-region risk
  control, §2.2). Tested as **CF3**.

*(Cross-world fork, deferred:* the calibrated trigger transfers to the host arm (where EH2/H9 was *measured*,
the natural confirmation) and the distributed arm (where ED2-smart was measured, the natural challenge) —
re-run CF1/CF4 there once the network headline lands. Numbered on its own increment, not as a global H, to
keep H50–H54 the load-bearing set.)*

---

## 6. Experiments (prefix **CF** — conformal consultation; network world unless noted)

Each follows the house template: a `Config` dataclass with `from_json_file`, a CLI entry point, a JSONL
record stream, a `plot_*.py` emitting a committed `.png` + `.csv`, regenerable from `reproduce.sh`,
deterministic and seeded (SPEC-2 §12). All reuse the shipped network apparatus (both arms, both oracles, the
`ConsultationPolicy` seam, the runner's budget enforcement) — SPEC-15 adds the conformal calibrator and a
thin `ConformalTriggered` / `AdaptiveConformalTriggered` policy.

- **CF1 — split-conformal trigger: coverage + the `ρ`-vs-coverage frontier (H50, H51).** Teacher-force the
  graph arm over held-out network steps; record `(belief_var, oracle divergence)` per step; compute `τ_α`
  for `α ∈ {0.01,0.05,0.10,0.20}`; verify empirical undetected-breach rate `≤ α` on a disjoint exchangeable
  test split (H50). Then run the loop with `ConformalTriggered(τ_α)` and with `FixedInterval(k)` swept, and
  plot **oracle budget `ρ` needed to reach each target `α`** for both — the guaranteed RQ2 figure (H51).
  Report **marginal and depth-conditional** coverage (§2.1 caveat). `experiments/cf1.py`, `configs/cf1.json`,
  `figures/plot_cf1.py`. **The headline.**

- **CF2 — exchangeability under rollout: static conformal vs ACI (H52).** Apply the CF1 `τ_α` *unchanged*
  along the autoregressive rollout and record empirical breach rate vs rollout depth (expected to climb
  above `α` — the exchangeability violation). Then run `AdaptiveConformalTriggered` (Gibbs–Candès ACI
  update, step size `γ` swept; the conformal-PID P+I controller as the stretch arm) and record its long-run
  breach rate. Plot **breach rate vs depth** for {static, ACI(γ)}, with the target `α` line.
  `experiments/cf2.py`, `figures/plot_cf2.py`.

- **CF3 — conformal risk control on undetected faithful-horizon breach (H54).** Define the monotone loss
  (undetected breach as a function of `τ`); apply the conformal-risk-control calibration (Angelopoulos et
  al.) to bound its expectation at `α`; compare realized breach expectation and `ρ` against CF1's
  coverage-only trigger. Adds the region-conditional `τ_α(r)` ablation (CQR-style, §2.2) as the heavy-tail
  fallback. `experiments/cf3.py`, `figures/plot_cf3.py`.

- **CF4 — the conformalizability of the signal: `belief_var` vs decode-entropy (H53, the mechanism).** The
  *same* split-conformal calibration applied to the graph arm's `belief_var` and the flat arm's
  decode-entropy. Report the score-conditional divergence curve (does a higher score mean higher
  divergence?), the calibration reliability diagram, the achieved-vs-target coverage gap, and the `ρ`-saving
  over fixed for each signal. **The figure that explains EH2-yes / ED2-smart-no with one controlled
  comparison.** `experiments/cf4.py`, `figures/plot_cf4.py`.

- **CF5 — cross-world fork (deferred).** Re-run CF1/CF4 on the host arm (EH2/H9 confirmation) and the
  distributed arm (ED2-smart challenge). Runs only after CF1 lands, per the evidence gate.
  `experiments/cf5_host.py`, `experiments/cf5_dist.py`.

---

## 7. What is confidently buildable now vs gated on a result

The user's instruction — *build what is confidently positive, experiment on what is not* — maps onto the
dependency order:

- **Confidently buildable now (machinery exists, result near-certain):**
  - The **split-conformal calibrator** (`τ_α` from oracle-labeled scores). It is a quantile computation over
    free, exact labels — deterministic, GPU-free, a build not a bet. The finite-sample coverage on
    *exchangeable* held-out steps (H50/CF1) is a near-certain positive *if the implementation is correct*,
    which is exactly why H50 is framed as an implementation gate, not a discovery.
  - The **`ConformalTriggered` policy**: `UncertaintyTriggered` with a calibrated `τ` — a one-line
    generalization of shipped code at the `ConsultationPolicy` seam.
- **Gated on CF1 (the headline):** whether the guarantee comes at lower `ρ` than fixed (H51). High-confidence
  positive for `belief_var` (EH2/H9 already won heuristically; conformal should win *and* certify) but
  "guaranteed coverage at sub-fixed budget" is a measurement the program will not assert unmeasured.
- **The genuine bets (must be measured):** CF2 (does exchangeability survive the rollout, and does ACI save
  it — the real scientific question), CF4 (does the split mechanism reproduce), CF3 (does risk control buy
  anything over coverage).

**Recommended build order:** CF1 (calibrator + H50 gate + the headline frontier) → CF4 (the mechanism, cheap
and decisive) → CF2 (the exchangeability/ACI question, the deepest result) → CF3 → CF5 fork. Each rung
graduates on a committed figure or a banked negative (SPEC §10.1, §12).

---

## 8. Build, reproduce, CI

### 8.1 Module layout (additive only)

```
src/verisim/conformal/                # NEW — the calibration layer (no new world, no new oracle)
  calibrate.py        # split-conformal τ_α from (score, oracle-divergence) pairs; region-conditional τ_α(r)
  policy.py           # ConformalTriggered, AdaptiveConformalTriggered (ACI) — ConsultationPolicy impls
  risk.py             # conformal risk control for the undetected-breach loss (CF3)
src/verisim/experiments/
  cf1.py … cf5_*.py   # NEW — the CF experiments
figures/
  plot_cf1.py … plot_cf4.py  # NEW — committed-figure generators
configs/
  cf1.json … cf4.json        # NEW — committed sweep configs
```

The calibrator consumes the shipped `NetUncertaintyModel` seam, both network oracles, and the
`ConsultationPolicy`/runner loop **unchanged** — nothing in the deterministic core, either model arm, the
metrics, or any existing experiment is edited. SPEC-15 is a layer *beside* the loop (it computes a
threshold the loop already accepts), not a change *to* it.

### 8.2 `reproduce.sh` (new CF block, in dependency order)

```bash
echo "== CF1: split-conformal trigger — coverage gate (H50) + the ρ-vs-coverage headline (H51) =="
python -m verisim.experiments.cf1 --config configs/cf1.json --out runs/cf1/records.jsonl --plot figures/cf1_coverage_frontier.png
echo "== CF4: conformalizability — belief_var vs decode-entropy (H53), the EH2/ED2-smart mechanism =="
python -m verisim.experiments.cf4 --config configs/cf4.json --out runs/cf4/records.jsonl --plot figures/cf4_signal_split.png
echo "== CF2: exchangeability under rollout — static conformal vs ACI (H52) =="
python -m verisim.experiments.cf2 --config configs/cf2.json --out runs/cf2/records.jsonl --plot figures/cf2_drift_aci.png
# CF3 (risk control) follows; CF5 (cross-world fork) gated on CF1.
```

The CF block runs on CPU once a trained `M_θ` checkpoint exists (the calibrator itself is torch-free — it
operates on recorded `(score, divergence)` pairs); CF tests assert *structural* invariants (calibration on
exchangeable data hits coverage within `1/(n+1)`; static conformal's rollout breach rate is monotone
non-decreasing in depth; ACI's long-run rate brackets `α`; `belief_var` separates within-ε from breach
steps more than decode-entropy by a sign test), not exact magnitudes — so the same tests pass on the macOS
primary host and the Linux CI confirmation (the §macOS-first principle).

---

## 9. Scope, non-goals, honest caveats

- **This is a calibrator, not a new world or a new signal.** It runs on the SPEC-5 network world, reuses
  both oracles, and conformalizes the *existing* `belief_var` / decode-entropy. Over-claiming a new
  uncertainty *model* from SPEC-15 is the failure mode this line forbids; the result is about *guaranteeing*
  and *explaining* triggers the program already has.
- **The guarantee is marginal unless sliced.** Conformal's `≥ 1−α` is *average* coverage (Angelopoulos &
  Bates, §2.1). Faithful horizon is a *prefix* property (the ED2-smart mechanism), so marginal coverage can
  pass while the prefix fails. CF1/CF2 report depth-conditional coverage precisely to catch this; a
  marginal-only pass is explicitly *not* claimed as a horizon win.
- **Exchangeability is the real risk, and it is pre-registered as such.** Vanilla conformal may *fail* under
  rollout drift (H52). That is not a bug to hide but the spec's central scientific finding-either-way:
  failure motivates ACI; success retires the online machinery on this world. The deepest negative — ACI
  *also* fails — bounds the whole approach and is bankable.
- **Conformal inherits the signal.** If `belief_var` is uninformative, no calibration rescues it (the
  guarantee holds *trivially* by consulting often, buying no `ρ`). The pre-registered limit on the headline
  is the *signal*, not the method — the exact ED2-smart reading, now with a mechanism (CF4).
- **A trained checkpoint is the one dependency.** The calibrator needs `M_θ`'s scores; everything downstream
  of the recorded `(score, divergence)` JSONL is torch-free. The spec's spine (the calibrator, the policy,
  the coverage math, the tests) runs without a GPU; only score *generation* needs the model.
- **`ε` and `α` are the user's dials, surfaced not hidden.** The guarantee is *conditional on `ε`* (the
  divergence tolerance defining a "breach") and the *target `α`*. Both are config fields swept in CF1, so
  the spec reports a guarantee *family*, not a single magic number.

---

## 10. The defensive payoff (why the guarantee matters beyond the metric)

A coverage guarantee on consultation is the operational form of a defensive promise. A verisim world model
used to answer "did this change open a path?" or to draft a rollout for an analyst is only trustworthy if
its *silent* errors are bounded. SPEC-15 turns "the model is usually right" into **"the probability of an
undetected faithful-horizon breach is ≤ α, distribution-free, certified against the exact oracle"** — a
statement an operator can act on. This is the SPEC §7 calibration diagnostic promoted from a *measured*
property to a *guaranteed* one, and it is a guarantee **no oracle-free world model can make**, because none
has the exact, free calibration truth computer environments do (SPEC §2). It is explicitly defensive and
human-out-of-loop: the calibrator bounds the model's silent error rate; it does not act on the world.

---

## 11. Provenance & reading order

SPEC-15 is the calibration altitude the program's RQ2 thread has been pointing at: SPEC.md §5 posed the
"when to consult" question; EH2/H9 and ED2-smart gave it a *contradictory* empirical answer; SPEC §7 made
calibration a measured quantity without a guarantee. SPEC-15 supplies the guarantee (split conformal on the
free oracle calibration set), the mechanism for the contradiction (CF4: `belief_var` conformalizes, flat
entropy does not), and the honest subtlety (CF2: exchangeability breaks under rollout; ACI is the fix). It
is a *method* spec: it advances how the program *decides to spend the oracle*, not what the model is.

Reading order for a newcomer: [SPEC.md §2](./SPEC.md) (the oracle asymmetry — why the calibration set is
free and exact) → [SPEC.md §5, §7](./SPEC.md) (the loop, `π_c`, and the calibration diagnostic) →
[SPEC-7 §9.4](./SPEC-7.md) (the ED2-smart null this explains) → this document (§1 the split with no theory,
§3 the method, §4 the exchangeability assumption, §5 the hypotheses) →
[`loop/policy.py`](../../src/verisim/loop/policy.py) and `src/verisim/conformal/calibrate.py` (the concrete
build, once shipped).
