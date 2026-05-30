# Verisim — v0 Engineering & Experiment Specification (SPEC-2.md)

> The buildable companion to [SPEC.md](./SPEC.md). SPEC.md is the science (why, what we claim, how we'd know we're wrong). **This document is the build**: the exact v0 environment, oracle, model, metrics, baselines, repo layout, tech stack, reproducibility regime, and milestone plan. It is written so that someone (the author, a collaborator, or a coding agent) can start implementing immediately and so that every experiment that produces a figure in the paper is fully specified here first.

**License:** MIT. **Scope of this doc:** Phase 0 (v0) only — see SPEC.md §12 for later phases. **Prime directive:** *the only job of v0 is to plot `H_ε(ρ)` once, cleanly, and test H1–H3.* Everything below serves that and refuses scope creep.

---

## 1. v0 scope: the smallest world with the hard property

The "hard property" of world models is **compounding state**: a mistake early in a rollout poisons everything after it, because state created (or destroyed) at step *t* must remain consistent at step *t+50*. The smallest environment that exhibits this — while being fully observable, deterministic, serializable, resettable, and unmistakably a *computer* world — is a **constrained shell over an in-memory filesystem**.

v0 is exactly that and nothing more. No networking, no real processes, no partial observability, no concurrency. Those are later phases and must not leak into v0.

**v0 deliverable:** the `H_ε(ρ)` curve (H1), the consultation-policy comparison (H2), the correction-operator comparison (H3), reproducibly, with figures, on this one environment family. Optionally a short technical report. That's the whole of v0.

---

## 2. The environment: a deterministic shell + filesystem world

### 2.1 State `S`

A state is a fully serializable snapshot:

```
State = {
  fs:   map[path -> Node]          # the filesystem tree
  cwd:  path                       # current working directory
  env:  map[str -> str]            # environment variables (small, fixed keyset)
  last: { exit_code: int, stdout_hash: hash }   # result of the last action (observation)
}
Node = File{ mode: int, content_hash: hash, size: int }
     | Dir { mode: int }
```

- Paths are POSIX-style absolute paths. Content is stored content-addressably (a hash → bytes store) so states are cheap to diff and compare; equality of two files is equality of `content_hash`.
- The state is **canonicalizable**: there is one and only one serialized form (sorted paths, normalized modes), so two semantically equal states serialize identically. This is essential for the divergence metric and for exact-match scoring.

### 2.2 Action `A` (the command grammar)

A fixed, small, unambiguous grammar of shell commands. v0 keyset (extend only with cause):

```
mkdir <path>            rm <path>              rmdir <path>
touch <path>            rm -r <path>           mv <src> <dst>
write <path> <token>    cp <src> <dst>         cp -r <src> <dst>
append <path> <token>   chmod <mode> <path>    cd <path>
cat <path>              ls <path>              export <KEY>=<token>
```

- `<token>` is drawn from a small fixed vocabulary of content tokens (so content space is finite and learnable; v0 is about *structure and consequence*, not arbitrary byte modeling).
- Every command has fully specified semantics, including failure modes (e.g., `rmdir` on a non-empty dir fails with a defined exit code and leaves state unchanged; `mv` of a dir moves the whole subtree; `rm -r` cascades). The failure cases are where compounding error bites and must be modeled exactly.
- Actions are emitted by a **driver** (a stochastic policy over the grammar; see §4) — *not* by a learned agent in v0. v0 studies the world model, not an agent.

### 2.3 Transition semantics

The semantics are defined once, in the **reference oracle** (§3), as a deterministic function `O(state, action) -> state'`. The semantics doc (`docs/semantics.md`, to be written alongside the implementation) is the normative English description; the oracle code is the executable truth. Any disagreement is a bug in one of them and is resolved by test (§16).

### 2.4 Making drift real (anti-triviality)

If the world is too simple, the neural model never drifts and there is no curve (SPEC.md §13 risk). v0 deliberately includes drift-inducing structure, and the environment has a **difficulty dial**:

- **Long-range dependencies:** files created early must persist and be referenced late; `mv`/`rm -r` on deep subtrees force the model to track structure it saw many steps ago.
- **Branching consequences:** the same command has different effects depending on accumulated state (e.g., `rmdir` succeeds or fails depending on whether earlier steps emptied the dir).
- **Difficulty parameters** (config): max tree depth, breadth, fraction of destructive/cascading commands, rollout length. Sweep these until the pure-neural model (`ρ=0`) drifts within the horizons we test — that's the regime where the science lives.

---

## 3. The oracle `O`

### 3.1 v0 oracle: deterministic reference interpreter

v0 uses a **reference oracle**: a from-scratch, deterministic Python implementation of the §2.3 semantics. Properties it must guarantee:

- **Determinism:** `O(s, a)` depends only on `(s, a)`. No clock, no RNG, no environment leakage. (Any pseudo-randomness in driver/policy lives in the driver, not the oracle.)
- **Totality:** defined for every `(s, a)`, including all failures; failures return a well-defined `(state unchanged or partially changed per real semantics, exit_code, stdout)`.
- **Purity / snapshotability:** `O` does not mutate its input state; it returns a new state. States are cheap to snapshot, hash, and restore.
- **Speed:** fast enough to generate large trajectory datasets and to serve as the in-loop verifier at `ρ=1` for the ceiling baseline.

The reference oracle *is* the symbolic half of the neuro-symbolic system (SPEC.md §6). It is honest about being a model of POSIX, not POSIX.

### 3.2 Interface (so the system oracle can drop in later)

Define one interface now so Phase 1's real-sandbox oracle is a drop-in (this is the only forward-looking abstraction v0 builds — justified because it costs one trait and saves a rewrite):

```python
class Oracle(Protocol):
    def step(self, state: State, action: Action) -> StepResult: ...   # true next state + observation
    def reset(self, state: State) -> None: ...                        # restore to a snapshot
    def determinism_report(self) -> DeterminismReport: ...            # which nondeterminism sources are sealed
```

v0 ships `ReferenceOracle`. Phase 1 ships `SandboxOracle` (namespaced shell / pinned container) behind the same interface. Nothing else in the codebase knows which oracle it's talking to.

---

## 4. Data: trajectory generation

- **Driver policies** (over the §2.2 grammar) generate `(state, action, next_state)` triples by rolling the oracle forward:
  - `uniform` — uniform over valid commands;
  - `weighted` — tilted toward structure-building then structure-mutating (produces realistic long-range dependencies);
  - `adversarial` — biased toward the drift-inducing/cascading commands (§2.4), used to stress-test.
- **Trajectory record (JSONL, one rollout per file):**
  ```
  { "env_config_hash": ..., "seed": ..., "driver": "...",
    "steps": [ { "state": <canonical>, "action": <str>, "next_state": <canonical>,
                 "delta": <structured edits>, "observation": {...} }, ... ] }
  ```
- **Splits:** train / val / test by *trajectory* (never leak a trajectory across splits). A separate **interventional/counterfactual** test set (Phase 3, but schema reserved now): pairs of rollouts identical up to step *t*, differing in action at *t*, with oracle-generated divergent continuations.
- **Dataset versioning:** every dataset carries the env-config hash, oracle version, driver, and seed, so any dataset is regenerable from its manifest (mirrors the author's existing content-hash/manifest discipline in `enklayve`/`vaulytica`).

---

## 5. The neural world model `M_θ`

### 5.1 Prediction target: structured state delta

`M_θ` predicts a **structured delta** `Δ̂` — the set of edits the action makes — *not* a regenerated full state. Rationale in SPEC.md §6.1 (bounds hallucination surface, localizes verification, more learnable). A delta is a sequence of typed edit ops:

```
Delta = [ Edit ]
Edit  = Create(path, node) | Delete(path) | Modify(path, new_content_token)
      | Move(src, dst) | Chmod(path, mode) | SetCwd(path) | SetEnv(key, token)
      | SetResult(exit_code, stdout_token)
```

`apply(state, Δ̂) -> state'` is a pure function (shared with the oracle's own apply step, so prediction and truth use identical application semantics — only the *deltas* are compared/produced differently).

### 5.2 Representation and model

- **Serialization:** state and action are serialized to a canonical token sequence (a compact DSL over paths, node types, modes, content tokens). Paths use a subword/segment tokenizer over the path alphabet; content uses the fixed content-token vocabulary (§2.2).
- **Model:** a small **decoder-only transformer** trained **from scratch** to map `serialize(state) ++ serialize(action) -> serialize(Δ̂)`. From-scratch is deliberate: it is small enough to train on the author's single local GPU, and building it doubles as the author's transformer-internals learning goal (nanoGPT lineage). A pretrained small code model is an allowed alternative baseline, not the primary.
- **Sizes:** start tiny (a few M params) and scale only as needed to make the *clean* (`ρ=0`) curve interesting; model size is an ablation axis, not a goal.
- **Output decoding:** constrained decoding to the `Delta` grammar (the model cannot emit a syntactically invalid edit). Note: grammar-validity ≠ semantic faithfulness — a syntactically valid delta can still be *wrong*; catching wrongness is the oracle's job, not the decoder's. This distinction is the whole point and must be kept crisp in code and writing.

### 5.3 Training objectives

- **Stage 1 — supervised next-delta prediction** on oracle-generated trajectories (teacher-forced). Loss over the serialized delta tokens.
- **Stage 2 (ablation / RQ via H-tests) — RLVR fine-tuning:** roll the model out autoregressively, reward = faithful horizon under the divergence metric (oracle as the verifiable reward; SPEC.md §6.3). This is where "train against the oracle" is realized. Kept as a clearly-scoped second stage so v0's first result doesn't depend on getting RL working.

---

## 6. The propose–verify–correct loop (implementation)

Direct implementation of SPEC.md §5.2. The three pluggable pieces:

### 6.1 Consultation policy `π_c` (RQ2 / H2)

Decides, per step, whether to spend an oracle call, subject to budget `ρ`:

- `fixed(k)` — consult every *k* steps (baseline / dumb).
- `drift_triggered(τ)` — consult when an estimate of accumulated drift exceeds τ. (Estimate via cheap self-consistency or a learned drift predictor.)
- `uncertainty_triggered(τ)` — consult when the model's own predictive uncertainty over the delta exceeds τ (requires calibrated uncertainty; calibration is a diagnostic metric in §7).
- `learned` — a small policy trained to spend the budget where it most extends faithful horizon (stretch).

All policies are budget-normalized so comparisons in §10 are at *equal* `ρ`.

### 6.2 Correction operator `C` (RQ3 / H3)

Given a consultation returning truth `s'` and prediction `ŝ'`:

- `hard_reset` — `s ← s'` (overwrite; baseline).
- `residual` — model predicts the *correction* `s' − ŝ'`; the discrepancy is logged and (Stage 2) used as an online learning signal so the model improves from corrections.
- `projection` — project `ŝ'` onto the nearest oracle-consistent state (repair the specific edits the oracle says are wrong, keep the rest). Cheapest-faithful-horizon-per-correction candidate.

### 6.3 Loop driver

A single `Rollout` runner takes `(M_θ, O, π_c, C, ρ, ε, seed, env_config)` and produces a divergence trajectory `[d(s_t, ŝ_t)]`, the consultation schedule actually used, and `H_ε`. Everything in §10 is sweeps over this runner.

---

## 7. Metrics and instrumentation

### 7.1 Divergence `d(s, ŝ)`

Primary: **normalized symmetric difference** over the canonical state's tuple-set. Represent a state as the set `T(s) = { (path, type, content_hash, mode) }` plus scalar facts (cwd, env, last-result). Then

```
d(s, ŝ) = ( |T(s) △ T(ŝ)| + scalar_mismatches ) / ( |T(s)| + |T(ŝ)| + n_scalars )
```

`d ∈ [0,1]`, `d = 0` iff identical. Secondary/diagnostic: tree-edit distance (more structure-aware, more expensive), and per-edit precision/recall on the predicted delta.

### 7.2 Headline and supporting metrics

- **`H_ε`** — faithful horizon (SPEC.md §5.1), the headline. Swept over `ε`.
- **`H_ε(ρ)` curve** — headline result (H1). Plus the `ρ=0` floor and `ρ=1` ceiling.
- **Per-step exact-match accuracy**, **divergence-over-time curves**, **edit-level P/R** — diagnostics.
- **Uncertainty calibration** — reliability of the model's confidence vs. actual per-step divergence (needed for `uncertainty_triggered`).
- **Cost** — oracle calls per faithful step (the real currency; the point of the project is *spending oracle calls well*).

### 7.3 Instrumentation

Every rollout emits a structured run-record (JSONL) with full config, seed, the divergence trajectory, the consultation schedule, and `H_ε`. Figures are generated *only* from these records (no hand-massaged numbers), so every figure is reproducible from a run-record + a plotting script.

---

## 8. Baselines (frames the result; we are not trying to "win")

- **b0 — pure neural (`ρ=0`).** The floor: how far the learned model drifts unaided. The thing oracle-grounding is supposed to fix.
- **b1 — oracle every step (`ρ=1`).** The ceiling: perfect faithfulness, maximal cost. Defines what "close to ceiling at low budget" (H1) means. Beating b1 on fidelity is impossible by construction; that is *not* the goal.
- **b2 — symbolic-only (the oracle alone).** Included to make explicit what the neural model is *for*: cheap rollout between consultations. b2 is perfectly faithful but is just rerunning the simulator — it has no learned model and no cost savings. The neural model's value is doing the steps b2 would have to pay full price for.
- **b3 — trivial predictor** (e.g., "state unchanged" / most-frequent-delta). The absolute floor; sanity that the task is nontrivial.

The contribution is the **interior** (`0<ρ<1`) and the **policy/operator** comparisons (H2/H3), not a leaderboard win.

---

## 9. Experiment protocol

For each cell in the sweep, run `N` seeds; aggregate with bootstrap confidence intervals.

- **E1 (H1) — the curve.** Sweep `ρ ∈ {0, .05, .1, .2, .3, .5, 1}` × `ε ∈ {0, small, medium}` × difficulty ∈ {low, med, high}, policy = `fixed`. Output: `H_ε(ρ)` curves. *Decision:* does H1's "≥80% of ceiling horizon at ≤20% consultation" hold anywhere?
- **E2 (H2) — consultation policy.** Fix `ρ` at the interesting knee from E1; compare `fixed` vs `drift_triggered` vs `uncertainty_triggered` (vs `learned`, stretch) at equal `ρ`. Output: `H_ε` by policy.
- **E3 (H3) — correction operator.** Fix policy at E2's winner; compare `hard_reset` vs `residual` vs `projection`. Output: faithful-horizon-per-correction by operator; check whether `residual` reduces *future* divergence (online learning effect).
- **E4 (ablations).** Representation (delta vs full-state), model size, objective (supervised vs +RLVR), difficulty sweep, driver policy.
- **(Reserved) E5/E6** — Phase 1 system-oracle (H4) and Phase 3 counterfactuals (H5). Schemas reserved; out of v0 scope.

Each experiment maps to exactly one hypothesis and one figure. If E1 refutes H1 (linear curve), that is reported as the finding, not buried.

---

## 10. Repo layout

```
verisim/
├── SPEC.md                  # research spec (the science)
├── SPEC-2.md                # this file (the v0 build)
├── LICENSE                  # MIT
├── README.md                # short: what/why + pointers to specs (to be written)
├── pyproject.toml           # package + deps + tooling
├── docs/
│   ├── semantics.md         # normative shell/fs semantics (paired with reference oracle)
│   ├── related-work.md      # maintained bibliography, one-line takes
│   └── lineage.md           # mapping from author's prior verifier repos -> Verisim
├── src/verisim/
│   ├── env/                 # State, Action grammar, canonical serialization, difficulty config
│   ├── oracle/              # Oracle protocol; ReferenceOracle (v0); SandboxOracle (Phase 1 stub)
│   ├── delta/               # Delta types, apply(), delta<->serialization
│   ├── model/               # tokenizer, transformer (from scratch), constrained decoder
│   ├── train/               # supervised stage 1; RLVR stage 2
│   ├── loop/                # propose-verify-correct runner; policies (π_c); operators (C)
│   ├── metrics/             # divergence, faithful-horizon, calibration, cost
│   ├── data/                # trajectory generation, schema, versioned manifests
│   └── experiments/         # E1..E4 entry points, config-driven
├── configs/                 # versioned experiment configs (one per cell/sweep)
├── runs/                    # run-records (JSONL); git-ignored; regenerable from configs+seeds
├── figures/                 # plotting scripts -> figures from run-records only
└── tests/                   # see §16
```

Keep it this size. New top-level dirs require justification against the prime directive.

---

## 11. Tech stack

- **Python 3.11+.** PyTorch for the model. Pydantic (or dataclasses) for State/Delta schemas. `pytest` for tests. `ruff` + `mypy` (strict) for hygiene. Hydra/OmegaConf (or plain dataclass configs) for the config sweeps.
- **No heavyweight framework** for the transformer in v0 — a small from-scratch implementation (nanoGPT-class) is the point and keeps it on the local GPU. HF `transformers`/`TRL` are allowed for the *optional* pretrained-baseline and for Stage-2 RLVR if hand-rolling RL proves distracting.
- **Determinism first:** seeded everything; the reference oracle is pure; torch determinism flags set for reproducible training where feasible (documented where exact reproducibility isn't achievable on GPU).
- **No telemetry, no network calls at runtime** (consistent with the author's posture across repos). Experiments run fully local or on rented GPUs the author controls.

---

## 12. Reproducibility regime

- **Everything is a function of (config + seed).** Datasets, training runs, and rollouts all regenerate from their config hash and seed. Run-records embed the full resolved config and the git commit.
- **Golden trajectories:** a small committed set of `(state, action, next_state)` goldens pins the reference-oracle semantics; CI fails on any drift (mirrors the golden-corpus discipline in the author's `enklayve` tax engine and `vaulytica`).
- **Manifest + content-hash** for datasets (regenerable, integrity-checked).
- **Figures from records only:** no figure is produced by hand; the plotting script + run-record IDs are committed next to each figure.

---

## 13. Milestones (Phase 0 / v0)

Each milestone has a concrete verify check. Gate, don't rush.

- **M0 — Env + reference oracle. ✅ Done.** Implement `State`, the §2.2 grammar, canonical serialization, and `ReferenceOracle` with full §2.3 semantics. *Verify:* golden trajectories pass; round-trip serialize/deserialize is identity; `O` is pure (property test: same input → same output, input unmutated). *Implemented in* `src/verisim/env/`, `src/verisim/oracle/`; semantics in `docs/semantics.md`; verified by `tests/test_goldens.py`, `tests/test_oracle_properties.py`, `tests/test_serialize.py`, `tests/test_action.py`.
- **M1 — Delta + apply. ✅ Done.** `Delta` types, `apply(state, Δ)`, delta↔serialization. *Verify:* for random `(s,a)`, `apply(s, oracle_delta(s,a)) == O(s,a)` (the oracle and the delta-apply agree exactly). *Implemented in* `src/verisim/delta/`; the oracle produces deltas and derives every next state via the shared `apply`, so the invariant holds by construction and is property-tested in `tests/test_oracle_properties.py`.
- **M2 — Data generation. ✅ Done.** Drivers, trajectory JSONL, versioned splits/manifests. *Verify:* dataset regenerates identically from its manifest; no trajectory leaks across splits. *Implemented in* `src/verisim/data/`; verified by `tests/test_data.py`. (The full §2.4 difficulty dial — explicit depth/breadth knobs — is deferred to M6 where it is tuned empirically; v0 difficulty is carried by the driver weighting.)
- **M3 — Divergence + faithful-horizon metrics. ✅ Done.** `d(s,ŝ)`, `H_ε`, run-record schema. *Verify:* `d=0` iff identical (property test); `H_ε` correct on hand-built diverging/non-diverging rollouts. *Implemented in* `src/verisim/metrics/`; verified by `tests/test_metrics.py`.
- **M4 — Model + supervised training (Stage 1). ✅ Done.** Tokenizer, from-scratch transformer, constrained decoder, training loop. *Verify:* model fits a tiny env (overfits a small dataset to ~0 train loss); produces only grammar-valid deltas. *Implemented in* `src/verisim/model/` (closed `Vocab`, the `tokenizer` serialization DSL, the LL(1) `DeltaGrammar` for constrained decoding, a nanoGPT-class `GPT`, the `constrained_decode`, and `NeuralWorldModel` — which implements the M5 loop's `Model` protocol so the learned model drops straight into the loop) and `src/verisim/train/` (Stage-1 supervised next-delta training). Representation decisions recorded in `docs/model-representation.md` (resolving §17 open questions #1 and #3). *Verified by* `tests/test_tokenizer.py` (encode/parse round-trip, grammar acceptance) and `tests/test_model.py` (overfits to loss < 0.05 with teacher-forced accuracy 1.0 and reproduces training deltas under constrained decode; an untrained model still decodes only grammar-valid deltas). PyTorch is an optional `[model]` extra (CPU-only in CI); Stage 2 (RLVR) is deferred.
- **M5 — Propose–verify–correct loop. ✅ Done (infrastructure).** Rollout runner + `fixed` policy + `hard_reset` operator. *Verify:* `ρ=1` reproduces the oracle exactly (`H_ε = T` always); `ρ=0` matches pure-neural rollout; intermediate `ρ` runs end to end. *Implemented in* `src/verisim/loop/` (a model-agnostic `Model` protocol, the `fixed`/`never` policies §6.1, the `hard_reset` operator §6.2, and the `run_rollout` runner §6.3 that records the divergence trajectory and consultation schedule against an independent ground-truth rollout). The runner is generic over any `Model`, so the learned `M_θ` (M4) drops in unchanged. *Verified by* `tests/test_loop.py` and `tests/test_experiments.py`. The §8 baselines **b2** (`OracleBackedModel`, perfect) and **b3** (`NullModel`, trivial) are shipped now and stand in for "pure-neural" until M4 lands; a runnable sweep is in `src/verisim/experiments/baselines.py`. *Note:* the smart policies (`drift_triggered`, `uncertainty_triggered`) and the `residual`/`projection` operators are M7, not M5.
- **M6 — E1 (H1). ◐ Harness done; curve plotted; tuning ongoing.** Run the curve sweep; produce `H_ε(ρ)` figures with CIs. *Verify:* the **headline curve exists and is plotted.** ✅ The reproducible pipeline is in place: `src/verisim/experiments/e1.py` (config-driven sweep over ρ×ε×difficulty×seed through the M5 loop with the M4 neural model), `src/verisim/metrics/aggregate.py` (bootstrap-CI curve aggregation), `configs/e1.json` (the committed config), and `figures/plot_e1.py` → `figures/e1_curve.png` + `figures/e1_curve.csv`. *Verified by* `tests/test_aggregate.py`, `tests/test_e1.py`, `tests/test_plot.py`. **Finding so far (honest, not tuned):** with the small/fast committed config the curve has `H_ε≈0` at ρ=0, `H_ε=T` at ρ=1, and an interior near the floor (≈7% of ceiling at ρ=0.2) — i.e. this scale does **not** yet exhibit H1's favorable knee. Per SPEC.md §9 that is a reportable result, not a failure; getting an *interesting* interior is a model-capacity/training/difficulty tuning problem (see §17.5) and is the continuing M6 work. (A bug found by this experiment — `apply` was not total over arbitrary grammar-valid deltas — was fixed; see `tests/test_apply.py`.)
- **M7 — Policies + operators (E2/E3, H2/H3). ✅ Done.** Added `drift_triggered`, `uncertainty_triggered` (§6.1) and `residual`, `projection` (§6.2); ran E2/E3. *Verify:* equal-budget comparisons produced with CIs. ✅ *Implemented in* `src/verisim/loop/policy.py` (the `StepContext` a policy sees, plus `UncertaintyTriggered`/`DriftTriggered` thresholding the model's instantaneous / accumulated uncertainty), `src/verisim/loop/operator.py` (`Residual`/`Projection`), `src/verisim/loop/runner.py` (threads the per-step uncertainty signal and adds a **spend-down backstop** so *every* policy spends exactly `floor(ρ·T)` calls — true equal-`ρ`, §16), `src/verisim/model/decode.py` + `world_model.py` (the neural `M_θ` exposes mean decode entropy as its uncertainty signal, making it a `loop.UncertaintyModel`), and `src/verisim/experiments/e2.py` / `e3.py` (config-driven comparisons; `configs/e2.json`, `configs/e3.json`; `verisim.metrics.aggregate.aggregate_comparison` for the equal-budget CIs; `figures/plot_comparison.py` → `figures/e2_policies.*`, `figures/e3_operators.*`). *Verified by* `tests/test_loop.py` (triggered-policy firing, operator identity + diagnostics, spend-down, equal-budget), `tests/test_aggregate.py`, `tests/test_e2.py`, `tests/test_e3.py`, `tests/test_plot.py`. **Findings (honest, not tuned):** *E2 (H2)* — at equal budget the dumb `fixed` policy (`H_ε≈1.1`) *beats* both triggered policies (`H_ε≈0.2`); the small model's decode-entropy uncertainty is not yet calibrated enough to beat even-spacing, so calibration (§17.2) is the next lever. *E3 (H3)* — all three operators give an **identical** `H_ε` with identical CIs, the expected v0 identity: with a full-state one-step oracle truth every operator corrects to the same `s'`, so they differ only in the diagnostic they expose (residual's online-learning signal magnitude; projection's per-correction repair cost). H3's horizon differences need partial verification or Stage-2 online learning (deferred).
- **M8 — Write-up.** Short technical report (the four figures + honest negative results); package the env+metric as an Inspect eval and a Prime Intellect environment (§15). *Verify:* a stranger can reproduce a figure from configs+seeds.

M0–M6 is the minimum that constitutes a result. M7–M8 is the depth and the distribution.

---

## 14. Forward interfaces (the only future-proofing v0 builds)

To make later phases additive rather than rewrites, v0 commits to exactly two abstractions and no more:

- **`Oracle` protocol (§3.2)** — so `SandboxOracle` (Phase 1, real shell) and future network/syscall oracles drop in unchanged.
- **`Environment` boundary** — `State`/`Action`/`Delta`/serialization behind a small interface, so Phase 4's network state is a new `Environment` implementation, not a fork.

Everything else (partial observability, counterfactual harness, agent training) is explicitly deferred and must not add abstraction weight to v0.

---

## 15. Packaging for the research ecosystems

When M6 produces a curve, the artifact is published where researchers already look (SPEC.md §14):

- **arXiv / technical report** — the `H_ε(ρ)` result, policy/operator comparisons, honest negatives.
- **Inspect eval** — package the faithfulness benchmark (env + divergence + ground-truth labels) as an `inspect_evals`-compatible task, so it slots into the framework labs already use for evaluations.
- **Prime Intellect environment** — wrap the env + oracle-as-reward as a `verifiers`-spec RL environment on the Environments Hub, making "train a world model against a verifiable oracle reward" a reusable community artifact. This is also the cleanest public expression of the author's verifier-as-reward thesis.
- **MIT license + responsible-use README** — per SPEC.md §13 ethics (defensive framing; review before releasing anything encoding real exploit dynamics).

---

## 16. Testing strategy

- **Semantics goldens** — committed `(s, a, s')` cases pin the reference oracle; CI fails on drift (§12).
- **Property tests** — oracle purity/determinism; `d=0 ⇔ identical`; `apply∘oracle_delta == oracle`; serialize round-trip identity; grammar-validity of all decoded deltas.
- **Loop invariants** — `ρ=1 ⇒ H_ε=T`; `ρ=0 ⇒` matches unaided neural rollout; consultation budget never exceeded.
- **Metric tests** — `H_ε` on hand-built rollouts with known horizons.
- **Reproducibility test** — a run reproduces bit-for-bit from (config, seed) where determinism is claimed; documented exceptions where GPU nondeterminism prevents it.

Tests are not optional polish here — they are the thing that makes the eventual claims *trustworthy*, which for a project whose entire thesis is "deterministic verification" is the point.

---

## 17. Open engineering questions (to resolve during M0–M4)

These are known unknowns; record decisions in `docs/` as they're made.

1. **Content modeling. ✅ Resolved (M4).** Fixed content-token vocab (chosen for v0) vs. eventually real byte content. v0 fixes it: content is a string over a **prefix-free** content vocabulary, so it decomposes uniquely by greedy longest-match. Boundary to real byte content noted. See `docs/model-representation.md`.
2. **Uncertainty estimation. ◐ Decided (M7); calibration open.** The `uncertainty_triggered` / `drift_triggered` policies are driven by the **mean Shannon entropy of the constrained decode** — the per-step spread of probability over the grammar-valid next tokens — exposed by `NeuralWorldModel.predict_delta_with_uncertainty` (`uncertainty` thresholds the instantaneous value; `drift` thresholds its running sum since the last consult, a cheap oracle-free proxy for compounding drift). This was chosen over ensemble disagreement (too expensive for a single local model) and a learned drift head (extra training surface) as the cheapest signal that needs no second model. **Open:** E2 shows it is not yet *calibrated* — at the small committed scale it does not beat even-spacing — so calibrating decode entropy against actual per-step divergence (the §7.2 reliability diagnostic), and revisiting ensemble/learned heads if it stays uninformative at larger scale, is the remaining work.
3. **Tokenization of paths/trees. ✅ Resolved (M4).** Segment tokenizer vs. structural/graph encoding of the FS tree. v0 uses the **serialized-DSL + segment tokenizer** (absolute paths as `<p>`-delimited name segments); structural/graph encoding revisited only if structure proves hard to learn. See `docs/model-representation.md`.
4. **RLVR stability (Stage 2).** Hand-rolled vs. TRL; reward shaping for faithful-horizon. Deferred until Stage 1 + E1 are done.
5. **Difficulty calibration. ◐ In progress (M6).** What difficulty setting makes `ρ=0` drift within tested horizons *without* being pathological? Tuned empirically at M6 (the curve must have room to be interesting). The E1 harness exists; with the current small config the model is too weak to predict even step 0 reliably (`ρ=0` drifts immediately, interior near the floor), so model capacity/training budget and difficulty must be co-tuned until the interior `0<ρ<1` becomes informative. This is the open empirical work that turns "the curve is plotted" into "H1 is tested."

---

*This document is the v0 contract. When a milestone completes, mark it and link its run-records and figures here. When an open question (§17) is resolved, record the decision in `docs/` and reference it. Keep v0 small; the science is one curve. Build the rest only after the curve exists.*
