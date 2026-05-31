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

The world model is a **pluggable proposer** — a from-scratch transformer, a JEPA-style
latent predictor, an RSSM, or a frozen LLM all drop into the same loop (the model-agnostic
`Model` protocol). So the deeper bet is about a *method*, not a model: **deterministic
verification as a model-agnostic primitive for probabilistic ML** — the layer underneath the
world-model race, not another entrant in it. That the favorable-consultation behavior belongs
to the oracle-loop and not to any model class is itself a falsifiable hypothesis
([SPEC §9, H22](docs/specs/SPEC.md#9-hypotheses-falsifiable)).

All specs live under [`docs/specs/`](docs/specs/), with the canonical, evidence-gated
build order in [SPEC §12](docs/specs/SPEC.md#12-research-roadmap):

- **The science:** [SPEC.md](docs/specs/SPEC.md) — why the project exists, what it
  claims, and how we would know if we were wrong.
- **The v0 build:** [SPEC-2.md](docs/specs/SPEC-2.md) (the shell/filesystem world) and
  [SPEC-2.1.md](docs/specs/SPEC-2.1.md) (the focused effort that earned a competent
  model and a clean knee result).
- **The network world:** [SPEC-5.md](docs/specs/SPEC-5.md) — the current build front.
- **Semantics:** [docs/semantics.md](docs/semantics.md) (filesystem) and
  [docs/network-semantics.md](docs/network-semantics.md) (network) — the normative command
  semantics, paired with the reference oracles, which are the executable truth.
- **The full result write-up:** [docs/report.md](docs/report.md).

## Status

> **Where things stand (2026-05): v0 is done; the network world is plotting its first curve.**
> The filesystem v0 (M0–M8) and the focused **[SPEC-2.1](docs/specs/SPEC-2.1.md)** effort are
> complete: K0 proved the learner works; K1/K2 lifted clean per-step faithfulness from ~0 to
> **0.86** (dissolving the diagnosed copy bottleneck); K3/K4 then found that **no consultation
> policy yields a favorable `H_ε(ρ)` knee on the single-filesystem world** — its *discrete* errors
> spike the set-difference past ε in one step, so first-exceedance `H_ε` is reset-resistant (the
> honest negative, [report §K3+K4](docs/report.md)). Per SPEC-2.1 §10 that **licenses
> [SPEC-5 (the network world)](docs/specs/SPEC-5.md)**, where drift is gradual and observation is
> partial. The network **deterministic core (NW0–NW3)** ships (typed-graph reachability world,
> free Tier-A oracle, graph deltas, drivers, graph/reachability metrics — all dependency-free,
> no GPU), and so does **NW4's** grammar-constrained supervised network `M_θ`. **NW5 now ships
> the partial-observation propose-verify-correct loop** (full / probe consultation modes, probe
> policies, belief operators, baselines, [`netloop/`](src/verisim/netloop/)), and **NW6 plots the
> prime-directive EN1 `H_ε(ρ)` curve**: on the flat-Markov `M_θ` the interior is **near-flat —
> the H8 honest negative**, the network analogue of v0's H1 floor, which licenses the NW7
> graph-arm + drift-mitigation work ([report §NW5+NW6](docs/report.md)). Canonical build order:
> [SPEC §12](docs/specs/SPEC.md#12-research-roadmap).

**v0 — shell/filesystem world (`src/verisim/`, SPEC-2 §13): complete.**

| Milestone | What | Status |
|-----------|------|--------|
| **M0–M3** | Env + `ReferenceOracle`, `Delta`/`apply`, drivers/data, divergence + `H_ε` + run-records | ✅ |
| **M4–M5** | Neural `M_θ` (from-scratch transformer, constrained decoder) + propose–verify–correct loop | ✅ |
| **M6–M8** | E1–E4 experiments, smart policies/operators, report, faithfulness benchmark + RL env | ✅ |
| **SPEC-2.1** | K0 (learner works) → K1/K2 (floor ~0 → **0.86**) → K3/K4 (knee refuted on single-FS; licenses SPEC-5) | ✅ |

**Network world (`src/verisim/net*`, SPEC-5 §13): partial-observation loop + first `H_ε(ρ)` curve.**

| Milestone | What | Status |
|-----------|------|--------|
| **NW0** | Typed-graph `NetworkState`, action grammar, canonical serialization + **Tier-A reference oracle** + [network semantics](docs/network-semantics.md) + golden trajectories | ✅ |
| **NW1** | Graph `Delta` types, `apply`, serialization; the `apply == oracle` invariant | ✅ |
| **NW2** | Drivers (uniform/weighted/adversarial topology+traffic) + trajectory generation | ✅ |
| **NW3** | Graph divergence, **reachability-faithfulness**, bits-to-correct (`H_ε` + run-records reused from v0) | ✅ |
| **NW4** | Network `M_θ` ([`netmodel/`](src/verisim/netmodel/)): closed vocab, tokenizer, LL(1) graph-delta grammar, constrained decode, supervised training. The **flat** arm (the H11 flat-Markov baseline, reusing v0's transformer + trainer) ships and is tested | ◐ flat arm |
| **NW5** | Partial-observation propose-verify-correct loop ([`netloop/`](src/verisim/netloop/)): two-mode (full / **probe**) oracle, probe policies `π_o`, correction/belief operators, baselines, model-agnostic runner — loop invariants tested | ✅ |
| **NW6** | **EN1 network `H_ε(ρ)` curve** ([`en1_curve.png`](figures/en1_curve.png)) — the prime directive (H8). Honest negative on the flat arm: near-flat interior | ✅ |
| **NW7** | Equal-budget comparisons. **EN2** (consultation policy `π_c`, H9) + **EN3** (correction/belief operators, §8.3) ship on the flat arm: EN3 breaks v0's operator identity collapse — the probe earns **~2.3× more faithful horizon per oracle-bit**. Graph/RSSM arm (H11), smart info-gain `π_o` (H10), drift mitigations, EN4 remain | ◐ EN2/EN3 |
| **NW8** | **Message-passing + RSSM graph arm** ([`netmodel/graph_model.py`](src/verisim/netmodel/graph_model.py)) + the §6.3 noise lever + **EN4 graph-vs-flat (H11)** — then RLVR/online-TTT (EN5), counterfactual (EN6), EN7/H22, EN8/EN9 (SPEC-8 SSL), LLM-callable simulator, packaging | ◐ graph arm + EN4 |

The deterministic cores (filesystem and network) have **no runtime dependencies** and need
no GPU. The propose–verify–correct loop is model-agnostic, so the learned model `M_θ` — a
from-scratch decoder-only transformer predicting structured deltas under grammar-constrained
decoding — drops in via `NeuralWorldModel` (filesystem) or `NeuralNetworkWorldModel`
(network). PyTorch is an optional `[model]` extra (see
[docs/model-representation.md](docs/model-representation.md)).

### The v0 result, honestly (E1 → SPEC-2.1)

The point of v0 is to plot `H_ε(ρ)` — faithful horizon vs. oracle-consultation budget — and
ask whether cheap consultation buys a *favorable knee*. The original tiny config sat on the
floor (`H_ε≈0` at ρ=0). The focused **SPEC-2.1** effort then diagnosed and fixed *why*:

1. **K0** — a depth-1 control reaches exact-match **1.0**, proving the pipeline can fit a
   deterministic transition; the floor is a generalization gap, localized to exact
   *multi-token path copying* in the delta (not capacity, not a broken learner).
2. **K1/K2** — coverage-balanced data + a proper minibatch/schedule/early-stopping trainer take
   clean per-step faithfulness from ~0.09 to **0.86** on a non-trivial world — the copy
   bottleneck was *coverage/training*, not representation.
3. **K3/K4** — with that competent model the ρ=0 floor rises to ~10/48, but `H_ε(ρ)` is a
   **floor + cliff, not a knee**, under *every* consultation policy (fixed and smart alike):

   ![K4 H_ε(ρ) on the structural world](figures/k4_knee.png)

   Filesystem errors are **discrete** — one wrong edit spikes the set-difference past ε in a
   single step — so first-exceedance `H_ε` is governed by the first error and resets can't push
   it out. **C-knee / H1 is refuted on the single-filesystem world**, and per SPEC-2.1 §10 that
   honest negative is exactly what **licenses the network world** (gradual drift, partial
   observability with a calibrated signal).

The full write-up — every figure (E1–E4, calibration, K0/K2/K4), the honest negatives, the
mechanism, threats to validity, and exact reproduction — is in [docs/report.md](docs/report.md).
Every figure regenerates from its config + seeds with `bash figures/reproduce.sh`.

### The network world's first curve (NW6 / EN1)

The same loop now runs on the partially-observable network world. EN1 — SPEC-5's prime
directive — sweeps the consultation budget on the flat-Markov network `M_θ`:

![EN1 H_ε(ρ) on the network world](figures/en1_curve.png)

The interior is **near-flat, then a cliff at ρ=1** — the opposite of a favorable knee. This is
the **H8 honest negative for the flat arm**, the network analogue of v0's H1 floor: the model
memorizes its training trajectories (teacher-forced accuracy 1.0) but generalizes poorly
off-distribution (~0.2 exact-delta match free-running), so it drifts past ε in ~1 step without
consultation. That near-floor result is exactly what makes the **NW7** levers load-bearing — the
message-passing + RSSM graph arm (H11) and the drift mitigations v0 never had (noise-injected
rollout training, self-forcing; the GNS/m4 levers). The EN1 machinery now exists to measure,
reproducibly, whether any of them lifts `M_θ` off the floor ([report §NW5+NW6](docs/report.md)).

The first NW7 comparisons already run on that machinery. **EN3** (correction/belief operators)
shows the payoff partial observability buys that v0 could not: the full-consult operators
coincide on `H_ε` (v0's identity), but a cheap one-host **probe + belief filter breaks that
collapse** and earns **~2.3× more faithful horizon per oracle-bit** — cheap, localized sensing
is the efficient way to buy horizon. **EN2** (consultation policy) finds the uncertainty-triggered
policy leading `fixed` (3.4 vs 1.9 `H_ε`), reversing v0's clean negative — suggestive but with
overlapping CIs, awaiting the calibrated belief-variance signal the RSSM arm supplies
([report §NW7](docs/report.md)).

### Verification

The claims above are audited empirically in [docs/verification.md](docs/verification.md):
the core invariants (`apply == oracle`, serialization round-trips, the NW4 tokenizer,
metric bounds, exit codes, in- and cross-process determinism) are proven over **48,000
oracle transitions with zero failures** by the dependency-free, torch-free
[`scripts/verify_invariants.py`](scripts/verify_invariants.py) — and additionally over the
**entire action space** (448,260 state×action pairs), **by construction**, and with
**negative controls** confirming each check detects deliberate corruptions. Every
quantitative number in the report *and* this README is machine-checked against the committed
figure CSVs; all 15 CSVs (including the NW6 network `H_ε(ρ)` curve and the NW7 EN2/EN3
comparisons) regenerate from config + seeds with `maxΔ = 0`; the NW5 partial-observation loop
invariants are tested (ρ=1 full-consult is exact; a one-host probe corrects strictly less than
a full consult); and the packaging is
verified end-to-end (the RL-env return equals the faithful horizon, the benchmark separates a
perfect from a trivial model, coverage spans all 13 commands). The audit found and fixed two
stale-documentation drifts (an E2 table and the K1 coverage diagnostic), with no invariant,
reproducibility, or packaging claim refuted.

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

See [SPEC-2 §10](docs/specs/SPEC-2.md) (filesystem) and [SPEC-5 §16](docs/specs/SPEC-5.md)
(network). Packages under [src/verisim/](src/verisim/):

- **v0 filesystem world:** `env/`, `oracle/`, `delta/`, `data/`, `metrics/`,
  `loop/` (propose–verify–correct runner, policies, operators, baselines),
  `model/` (`M_θ`: vocab, tokenizer, grammar, transformer, constrained decoder),
  `train/` (supervised + minibatch + RLVR), `experiments/` (baselines, E1–E4, K0/K2/K4,
  diagnostics, the autoresearch ratchet `auto/`), and the packaging `eval/` (faithfulness
  benchmark + Inspect adapter) and `rl/` (oracle-as-reward environment).
- **Network world (SPEC-5):** `net/` (typed-graph `NetworkState`, action grammar,
  serialization), `netoracle/` (Tier-A reference oracle), `netdelta/` (graph deltas + `apply`),
  `netdata/` (drivers + generation), `netmetrics/` (graph divergence, reachability-faithfulness,
  bits-to-correct), `netmodel/` (the NW4 network `M_θ`: vocab, tokenizer, LL(1) graph-delta
  grammar, constrained decode, supervised dataset — reusing v0's transformer + trainer), and
  `netloop/` (the NW5 partial-observation propose-verify-correct runner, two-mode oracle, probe
  policies, correction/belief operators, baselines). The EN1 curve is `experiments/en1.py`.

Experiment configs live in [configs/](configs/); plotting scripts + figures in
[figures/](figures/); the run-records they read are git-ignored and regenerable from config + seeds.

## License & posture

MIT (see [LICENSE](./LICENSE)). This is a research repo: **no telemetry, no
network calls at runtime, no commercial path.** The framing and downstream agents
are defensive; see [SPEC.md §13](docs/specs/SPEC.md) for the ethics and dual-use posture.

Author: Clay Good.
