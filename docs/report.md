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
| `fixed` | **1.4** | [1.1, 1.8] | 4.0 |
| `uncertainty` | 0.6 | [0.2, 1.0] | 4.0 |
| `drift` | 0.1 | [0.0, 0.3] | 4.0 |

The dumb baseline wins — so this is not a wash, it is a **reversal**: at this scale
spending the budget where the model is *least confident* is worse than spreading it
evenly. `fixed` beats `drift` unambiguously (disjoint CIs) and beats `uncertainty`
at `ε = 0` (`[1.1, 1.8]` vs. `[0.2, 1.0]`). The reason is calibration: the triggered
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
