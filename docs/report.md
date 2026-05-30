# Verisim v0 — technical report

> The v0 result, stated honestly. This is the short write-up SPEC-2 §13 (M8) calls
> for: the three experiments that produce figures (E1/E2/E3), what they show, and
> what they do not. Every number here is read from a committed run-record CSV and
> is regenerable from a config + seeds (SPEC-2 §12). Figures live in
> [`../figures/`](../figures/).

## Bottom line

v0's job was to build the apparatus that can measure **how much oracle consultation
buys how much faithful horizon** (`H_ε(ρ)`), and to run the first three experiments
on it. The apparatus is built, tested, and reproducible. The headline scientific
finding is a **clean set of negatives** at the small, fast committed scale:

- **H1 (a favorable knee exists):** *not observed.* The `H_ε(ρ)` curve is flat and
  near the floor across the interior and only reaches the ceiling at `ρ = 1`.
- **H2 (smart beats dumb):** *refuted at this scale.* Fixed-interval consultation
  **beats** the uncertainty/drift-triggered policies at equal budget.
- **H3 (correction operator matters):** *identity, as predicted.* `hard_reset`,
  `residual`, and `projection` are statistically indistinguishable on faithful
  horizon — expected from a full-state oracle truth.

None of these refute the *program* (SPEC.md §9 explicitly treats a refuted
hypothesis as a result, not a failure). They locate the work: the curve is
uninformative because the committed model is too small and the difficulty dial is
not yet tuned (SPEC-2 §17.5), and the smart policies lose because their uncertainty
signal is not yet calibrated (SPEC-2 §17.2). The contribution of v0 is the
**measurement**, the **honest curves**, and a benchmark + RL environment others can
build on (SPEC-2 §15).

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

## Threats to validity

- **Scale.** The committed model is ~tiny (1–2 layers, 64-dim) and trains for a few
  hundred iterations on a CPU-sized dataset. Every negative above is consistent with
  "too small to be interesting," not "the mechanism is wrong." The deterministic core
  (M0–M3) and loop invariants (M5) are separately tested, so the apparatus is sound.
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
