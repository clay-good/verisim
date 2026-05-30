# Verisim

**Oracle-grounded, neuro-symbolic world models of computer environments.**

Generative world models (Genie 3, V-JEPA 2, Cosmos) all hit the same wall:
long-horizon error accumulation and faithfulness, with no cheap way to detect or
correct drift, because physical and visual worlds have no ground-truth oracle.
Verisim's claim is that **computer environments are the exception** — filesystems,
processes, networks, and APIs are digital, deterministic, and fully checkable, so
a deterministic oracle can be placed in the loop to bound a neural world model's
drift. Verisim builds that loop and measures the central tradeoff nobody else can
measure: **how much oracle consultation buys how much faithful horizon.**

- **The science:** [SPEC.md](./SPEC.md) — why the project exists, what it claims,
  and how we would know if we were wrong.
- **The build:** [SPEC-2.md](./SPEC-2.md) — the concrete v0 environment, oracle,
  model, metrics, baselines, repo layout, and milestones.
- **Semantics:** [docs/semantics.md](./docs/semantics.md) — the normative
  description of the v0 shell/filesystem command semantics (paired with the
  reference oracle, which is the executable truth).

## Status

Pre-experiment (v0). The deterministic foundation — milestones **M0–M3** of
SPEC-2 §13 — is implemented and tested:

| Milestone | What | Status |
|-----------|------|--------|
| **M0** | Env (state, command grammar, canonical serialization) + `ReferenceOracle` | ✅ |
| **M1** | `Delta` types, `apply(state, delta)`, delta↔serialization | ✅ |
| **M2** | Drivers, trajectory JSONL, versioned manifests/splits | ✅ |
| **M3** | Divergence `d(s,ŝ)`, faithful horizon `H_ε`, run-record schema | ✅ |
| **M5** | Propose–verify–correct loop: `fixed` policy + `hard_reset` operator + baselines (b2/b3) | ✅ |
| **M4** | Neural world model `M_θ`: tokenizer, from-scratch transformer, constrained decoder, supervised training | ✅ |
| **M6** | E1 — the `H_ε(ρ)` curve: sweep harness + bootstrap-CI aggregation + figure (curve plotted; tuning ongoing) | ◐ |
| **M7** | Smart policies (`drift`/`uncertainty`) + operators (`residual`/`projection`); E2/E3 equal-budget comparisons with CIs | ✅ |
| **M8** | Write-up ([`docs/report.md`](./docs/report.md)) + packaging: faithfulness benchmark (Inspect eval) + oracle-as-reward RL environment (verifiers-spec) | ◐ |

M0–M3 plus the M5 loop are the deterministic core; they have **no runtime
dependencies** and need no GPU. The propose–verify–correct loop is built
model-agnostically, so the learned model `M_θ` (M4) — a from-scratch decoder-only
transformer that predicts structured deltas under grammar-constrained decoding —
drops straight into the loop via `NeuralWorldModel`. PyTorch is an optional
`[model]` extra (see [docs/model-representation.md](./docs/model-representation.md)
for the tokenization/representation decisions).

### The headline result (E1)

The whole point of v0 is to plot `H_ε(ρ)` — faithful horizon vs. oracle-consultation
budget — once, cleanly (SPEC-2 §9). The reproducible pipeline is in place:

```bash
python -m verisim.experiments.e1 --config configs/e1.json --out runs/e1/records.jsonl
python figures/plot_e1.py --records runs/e1/records.jsonl   # -> figures/e1_curve.{png,csv}
```

![E1 faithful-horizon vs consultation-budget curve](figures/e1_curve.png)

The figure and its CSV are generated *only* from run-records (regenerable from the
config + seeds). **Honest status:** with the small, fast committed config the curve
shows `H_ε≈0` at ρ=0 and `H_ε=T` at ρ=1 with an interior near the floor — i.e. it
does **not** yet show H1's favorable knee. That is a reportable result, not a
failure (SPEC.md §9); making the interior informative is a model-capacity /
difficulty tuning problem ([SPEC-2 §17.5](./SPEC-2.md)) and is the continuing M6
work.

### Policy and operator comparisons (E2 / E3)

E2 fixes the budget at a knee `ρ` and compares the §6.1 consultation policies
(`fixed` vs. `uncertainty_triggered` vs. `drift_triggered`); E3 fixes the policy
and compares the §6.2 correction operators (`hard_reset` vs. `residual` vs.
`projection`). Both are *equal-budget* by construction — the runner spends exactly
`floor(ρ·T)` oracle calls per arm — so the comparison isolates **where** the budget
is spent (E2) and **how** corrections are applied (E3).

```bash
python -m verisim.experiments.e2 --config configs/e2.json --out runs/e2/records.jsonl
python figures/plot_comparison.py --records runs/e2/records.jsonl --key policy \
    --out figures/e2_policies.png --csv figures/e2_policies.csv
python -m verisim.experiments.e3 --config configs/e3.json --out runs/e3/records.jsonl
python figures/plot_comparison.py --records runs/e3/records.jsonl --key operator \
    --out figures/e3_operators.png --csv figures/e3_operators.csv
```

**Honest findings** (small committed config, not a tuned run):

- **E2 (H2):** at equal budget `fixed` (`H_ε≈1.4`) *beats* both triggered policies
  (`uncertainty≈0.6`, `drift≈0.1`). The model's decode-entropy uncertainty signal
  ([SPEC-2 §7.2](./SPEC-2.md)) is not yet calibrated enough at this scale to beat
  naive even-spacing. The **calibration diagnostic** ([`figures/calibration.png`](figures/calibration.png),
  Pearson ≈ 0.11) measures exactly this: confidence barely predicts error, so the
  signal — not a new policy — is the lever to move next.
- **E3 (H3):** all three operators give an **identical** `H_ε≈1.3` with identical
  CIs. This is the expected v0 *identity*: with a full-state one-step oracle truth,
  every operator snaps the coupled state to the same `s'`. `residual` and
  `projection` differ only in the diagnostic they expose (the online-learning signal
  magnitude / per-correction repair cost), which is where H3 will bite once partial
  verification or Stage-2 online learning lands.

The full write-up — all three figures, the honest H1/H2/H3 negatives, threats to
validity, and exact reproduction — is in [docs/report.md](./docs/report.md). Every
figure regenerates from its config + seeds with `bash figures/reproduce.sh`.

## Packaging for reuse

The env + metric are packaged where researchers already look (SPEC-2 §15):

- **Faithfulness benchmark** ([`verisim.eval`](src/verisim/eval/)) — dependency-free;
  `score_model` / `score_suite` grade *any* model implementing the loop `Model`
  protocol against the oracle's ground truth, and `step_labels` + `grade_prediction`
  expose single-step labels for question-answer frameworks. An `inspect_ai` task
  adapter ships behind the optional `[eval]` extra.
- **Oracle-as-reward RL environment** ([`verisim.rl`](src/verisim/rl/)) — a
  `verifiers`-spec `WorldModelEnv` (with the `load_environment` entrypoint) whose
  reward is the oracle's faithfulness verdict, so the episode return *is* the faithful
  horizon. The public form of "train a world model against a verifiable oracle reward."

```python
from verisim.eval import score_model, FaithfulnessSample
from verisim.loop import OracleBackedModel
from verisim.oracle import ReferenceOracle

oracle = ReferenceOracle()
score = score_model(OracleBackedModel(oracle), FaithfulnessSample("adversarial", 200, 24), oracle=oracle)
assert score.normalized_horizon == 1.0   # a perfect model is fully faithful, unaided
```

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,model]"   # ".[dev]" alone skips the torch-based M4 tests
pytest            # property tests, semantics goldens, metric/loop/model tests
ruff check .      # lint
mypy              # strict type-check
```

```python
from verisim.env import State, parse_action
from verisim.oracle import ReferenceOracle
from verisim.delta import apply

oracle = ReferenceOracle()
state = State.empty()
for cmd in ["mkdir /a", "write /a/f alpha", "mv /a /b", "cat /b/f"]:
    result = oracle.step(state, parse_action(cmd))
    # apply(state, result.delta) == result.state, by construction (the M1 invariant)
    assert apply(state, result.delta).fs == result.state.fs
    state = result.state
```

## Layout

See [SPEC-2.md §10](./SPEC-2.md). The implemented packages live under
[src/verisim/](src/verisim/): `env/`, `oracle/`, `delta/`, `data/`, `metrics/`,
`loop/` (the propose–verify–correct runner, policies, operators, baseline
models), `experiments/` (the baseline sweep and E1/E2/E3), `model/` (the learned model `M_θ`:
vocab, tokenizer, grammar, transformer, constrained decoder), `train/`
(Stage-1 supervised training), and the M8 packaging `eval/` (faithfulness benchmark
+ Inspect adapter) and `rl/` (oracle-as-reward environment). The experiment configs
live in [configs/](configs/) and the plotting scripts + figures in [figures/](figures/).

## License & posture

MIT (see [LICENSE](./LICENSE)). This is a research repo: **no telemetry, no
network calls at runtime, no commercial path.** The framing and downstream agents
are defensive; see [SPEC.md §13](./SPEC.md) for the ethics and dual-use posture.

Author: Clay Good.
