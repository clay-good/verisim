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

## The one asymmetry everything rests on

| Source of a learning/verification signal | Dense? | Exact / true? | Free? | Generative? |
|---|---|---|---|---|
| Self-supervision (co-occurrence in a corpus) | ✅ | true to the *corpus*, not the world | ✅ | ✅ |
| Human supervision (annotation) | ◐ | usually, but **unscalable** | ❌ | ❌ |
| RL reward / reward model | ❌ (sparse scalar) | a **proxy**, hackable | ◐ | ❌ |
| **A deterministic oracle (computer worlds)** | ✅ | **exact, by construction** | ✅ | ✅ |

No other domain has the last row. A deterministic interpreter of a computer world returns the
*entire true next state* at every step, for free, and can *generate* unbounded perfectly-labeled
data and counterfactuals. Verisim is the research program built around that single asymmetry:
where to spend it (inference-time verification, RL reward, **and** — newest — self-supervised
pretraining, [SPEC-8](docs/specs/SPEC-8.md)), and how much it actually buys.

## How the loop works

The signature mechanism is **propose → verify → correct**, run under a consultation budget `ρ`:

```
                 ┌──────────────────────── step t ────────────────────────┐
  state s_t ────▶│  Δ̂ = Mθ.predict_delta(s_t, a_t)      ← neural proposer  │
  action a_t     │       (any model behind the Model protocol)             │
                 │                          │                              │
                 │   consult this step?  ◀── π_c policy, spends budget ρ    │
                 │        │ no                          │ yes               │
                 │        ▼                             ▼                   │
                 │  ŝ_{t+1} = apply(s_t, Δ̂)    O(s_t,a_t)  (oracle: truth)  │
                 │   (free-running prediction)  full | probe → correct ŝ    │
                 └─────────────────────────────────────────┬───────────────┘
                                                            ▼
        divergence d(ŝ_{t+1}, s*_{t+1}) ≤ ε ?   ──▶  faithful horizon
        H_ε(ρ) = first step where d > ε   (how long the model stays bit-exact)
```

- **`apply` is shared by the oracle**, so `apply(s, O(s,a).delta) == O(s,a).state` *by
  construction* — the model and the oracle speak the same delta language (the M1 / NW1 invariant).
- **`ρ`** ranges from 0 (never consult — pure free-running) to 1 (consult every step — always
  exact). The whole research question is the shape of `H_ε(ρ)` between those ends: is there a
  *favorable knee* where a little consultation buys a lot of horizon?
- Under partial observation the oracle has two modes — **full** (the whole next state) and
  **probe** (one host's local view, cheap) — which turns consultation cost into a real
  bit-budget and opens an active-sensing axis ([SPEC-5 §5.3](docs/specs/SPEC-5.md)).

### The neuro-symbolic split, as a *training* principle (SPEC-8)

The next-state partitions into two regimes that want opposite treatment:

```
  s' = O(s, a)
    ├─ D  decidable bits  ── the oracle fixes them exactly & free  ──▶  VERIFY, don't learn  [symbolic]
    └─ R  residual bits   ── genuinely uncertain given what's seen  ──▶  LEARN (the model's job) [neural]
```

Burning network capacity to memorize `D` is waste — the oracle computes it perfectly for free.
"Even nature offloads": evolution does not store chemistry in the genome. [SPEC-8](docs/specs/SPEC-8.md)
makes this a *training objective* (mask `D`, spend gradient on `R`) and ships the deterministic
machinery for it (OG1/OG2, below).

## Specifications

All specs live under [`docs/specs/`](docs/specs/); the canonical, evidence-gated build order is
[SPEC §12](docs/specs/SPEC.md#12-research-roadmap). The worlds form a ladder (filesystem →
network → host → distributed); two specs are *cross-cutting methods* every world inherits.

| Spec | Role | What it is |
|---|---|---|
| [SPEC.md](docs/specs/SPEC.md) | **the science** | why the project exists, what it claims, how we would know we were wrong (RQs, H1–H25) |
| [SPEC-2](docs/specs/SPEC-2.md) / [SPEC-2.1](docs/specs/SPEC-2.1.md) | **v0 build** | the shell/filesystem world; the focused effort that earned a competent model and the clean knee result |
| [SPEC-3](docs/specs/SPEC-3.md) | depth | how the toy grows into a real simulator (system oracle, partial obs, online self-healing, info-theoretic metric) |
| [SPEC-4](docs/specs/SPEC-4.md) | **the engine** | the autonomous research engine — Verisim improving Verisim, human out of the loop |
| [SPEC-5](docs/specs/SPEC-5.md) | **world: network** | the reachability/connectivity world — **the current build front** |
| [SPEC-6](docs/specs/SPEC-6.md) | world: host | the running computer (process tree, memory, scheduler) — design |
| [SPEC-7](docs/specs/SPEC-7.md) | world: distributed | replicated services, transactions, consensus — design |
| [SPEC-8](docs/specs/SPEC-8.md) | **method: oracle-grounded SSL** | put the oracle's truth in the *bulk* of the cake (self-supervised pretraining), not just the cherry (RL) |

Semantics docs ([filesystem](docs/semantics.md), [network](docs/network-semantics.md)) pin the
normative command semantics, paired with the reference oracles, which are the executable truth.
The full result write-up is [docs/report.md](docs/report.md).

## Status

> **Where things stand (2026-06): v0 is done; the network graph arm has shipped and split the H11 verdict.**
> The filesystem v0 (M0–M8) and the focused **[SPEC-2.1](docs/specs/SPEC-2.1.md)** effort are complete:
> K0 proved the learner works; K1/K2 lifted clean per-step faithfulness from ~0 to **0.86**; K3/K4 then
> found that **no consultation policy yields a favorable `H_ε(ρ)` knee on the single-filesystem world** —
> discrete errors spike the set-difference past ε in one step, so first-exceedance `H_ε` is reset-resistant
> (the honest negative). Per SPEC-2.1 §10 that **licenses [SPEC-5 (the network world)](docs/specs/SPEC-5.md)**,
> where drift is gradual and observation is partial. The network **deterministic core (NW0–NW3)** ships, as
> do **NW4** (flat supervised `M_θ`), **NW5** (the partial-observation propose-verify-correct loop), **NW6**
> (the prime-directive EN1 `H_ε(ρ)` curve — flat-arm **H8 honest negative**), and **NW7** (EN2/EN3 equal-budget
> comparisons — the probe earns ~2.3× more faithful horizon per oracle-bit). **NW8 now ships the
> message-passing + RSSM graph arm and the EN4 graph-vs-flat comparison**: a *split* H11 verdict — structure is
> a **+16.5-pt** better one-step token predictor and a **+30.6-pt** better *delta-exact* predictor, but neither
> arm yet converts that to free-running horizon. The **delta-exact per-step metric** and the **SPEC-8 OG1/OG2**
> oracle-grounded-SSL data factory (torch-free, property-tested) have also shipped, readying the EN8/EN9 runs.

**v0 — shell/filesystem world (`src/verisim/`, SPEC-2 §13): complete.**

| Milestone | What | Status |
|-----------|------|--------|
| **M0–M3** | Env + `ReferenceOracle`, `Delta`/`apply`, drivers/data, divergence + `H_ε` + run-records | ✅ |
| **M4–M5** | Neural `M_θ` (from-scratch transformer, constrained decoder) + propose–verify–correct loop | ✅ |
| **M6–M8** | E1–E4 experiments, smart policies/operators, report, faithfulness benchmark + RL env | ✅ |
| **SPEC-2.1** | K0 (learner works) → K1/K2 (floor ~0 → **0.86**) → K3/K4 (knee refuted on single-FS; licenses SPEC-5) | ✅ |

**Network world (`src/verisim/net*`, SPEC-5 §13): graph arm + EN4 + delta-exact + SPEC-8 data factory.**

| Milestone | What | Status |
|-----------|------|--------|
| **NW0** | Typed-graph `NetworkState`, action grammar, serialization + **Tier-A reference oracle** + [network semantics](docs/network-semantics.md) + goldens | ✅ |
| **NW1** | Graph `Delta` types, `apply`, serialization; the `apply == oracle` invariant | ✅ |
| **NW2** | Drivers (uniform/weighted/adversarial topology+traffic) + trajectory generation | ✅ |
| **NW3** | Graph divergence, **reachability-faithfulness**, bits-to-correct (`H_ε` + run-records reused from v0) | ✅ |
| **NW4** | Network `M_θ` ([`netmodel/`](src/verisim/netmodel/)): closed vocab, tokenizer, LL(1) graph-delta grammar, constrained decode, supervised training. The **flat** arm (H11 baseline) ships | ◐ flat arm |
| **NW5** | Partial-observation loop ([`netloop/`](src/verisim/netloop/)): two-mode (full / **probe**) oracle, probe policies `π_o`, correction/belief operators, baselines, model-agnostic runner | ✅ |
| **NW6** | **EN1 network `H_ε(ρ)` curve** ([`en1_curve.png`](figures/en1_curve.png)) — the prime directive. Honest H8 negative on the flat arm: near-flat interior | ✅ |
| **NW7** | Equal-budget comparisons. **EN2** (policy `π_c`, H9) + **EN3** (operators, §8.3): EN3 breaks v0's operator-identity collapse — the probe earns **~2.3×** more faithful horizon per oracle-bit | ◐ EN2/EN3 |
| **NW8** | **GNN + RSSM graph arm** ([`graph_model.py`](src/verisim/netmodel/graph_model.py)) + §6.3 noise lever + **EN4 graph-vs-flat (H11)** + **delta-exact metric** ([`exact.py`](src/verisim/netmetrics/exact.py)) + **SPEC-8 OG1/OG2 data factory** ([`grounding.py`](src/verisim/netdata/grounding.py), [`negatives.py`](src/verisim/netdata/negatives.py)). Then RLVR/TTT (EN5), counterfactual (EN6), EN7/H22, EN8/EN9 | ◐ graph arm + EN4 + delta-exact + OG1/OG2 |

The deterministic cores (filesystem and network) have **no runtime dependencies** and need no GPU.
PyTorch is an optional `[model]` extra (see [docs/model-representation.md](docs/model-representation.md)).

## Results, honestly

### v0 — the filesystem knee (E1 → SPEC-2.1)

The point of v0 is to plot `H_ε(ρ)` and ask whether cheap consultation buys a *favorable knee*.
The original tiny config sat on the floor (`H_ε≈0` at ρ=0). SPEC-2.1 diagnosed and fixed *why*:

1. **K0** — a depth-1 control reaches exact-match **1.0**: the pipeline can fit a deterministic
   transition, so the floor is a *generalization* gap localized to multi-token path copying.
2. **K1/K2** — coverage-balanced data + a proper trainer take clean per-step faithfulness from
   ~0.09 to **0.86** on a non-trivial world — the bottleneck was *coverage/training*, not capacity.
3. **K3/K4** — with that competent model, `H_ε(ρ)` is a **floor + cliff, not a knee**, under
   *every* policy:

   ![K4 H_ε(ρ) on the structural world](figures/k4_knee.png)

   Filesystem errors are **discrete** — one wrong edit spikes the set-difference past ε in a single
   step — so first-exceedance `H_ε` is governed by the first error and resets can't push it out.
   **C-knee / H1 is refuted on the single-filesystem world**, and per SPEC-2.1 §10 that honest
   negative is exactly what **licenses the network world** (gradual drift, partial observability).

### Network — the EN1 curve and the EN4 split verdict (NW6–NW8)

EN1 sweeps the consultation budget on the flat-Markov network `M_θ`. The interior is **near-flat,
then a cliff at ρ=1** — the **H8 honest negative**, the network analogue of v0's H1 floor:

![EN1 H_ε(ρ) on the network world](figures/en1_curve.png)

That near-floor is exactly what makes the NW8 levers load-bearing. **EN4** then trains the
message-passing + RSSM **graph arm** against the flat arm on identical oracle data and scores both
with the same eval primitives — a *split* verdict on H11 (does structure beat the flat serializer?):

![EN4 graph-vs-flat (H11)](figures/en4_graph_vs_flat.png)

| arm | one-step token acc | **delta-exact** rate | `H_ε`(ρ=0), ε∈{0,.05,.1} |
|---|---|---|---|
| flat-Markov (NW4) | 0.673 | 0.264 | 0 / 0 / 0 |
| **graph + RSSM (NW8)** | **0.838** | **0.569** | 0 / 0 / 0 |
| graph + RSSM + noise lever | 0.828 | 0.556 | 0 / 0 / 0 |

- **Positive (generalization): structure helps, and helps *more* on the honest metric.** The graph
  arm is +16.5 pts on token accuracy and **+30.6 pts on delta-exact** (0.569 vs 0.264 — more than
  double). **Delta-exact** ([`netmetrics/exact.py`](src/verisim/netmetrics/exact.py)) is the honest
  middle metric: not token accuracy (inflated by easy scaffolding tokens) and not horizon (`0` the
  instant any step exceeds ε), but *did the model freely decode the exact true edit set this step?*
  (`1` iff `bits_to_correct = 0`). The gap *widens* on the stricter metric — token accuracy
  understates how much structure buys.
- **Honest negative (horizon): even 57% delta-exact ≠ horizon — yet.** `H_ε(ρ=0)=0` for all arms;
  at 0.569 per-step exactness, whole-delta correctness decays geometrically over unaided steps and
  first-exceedance is discrete. The wall is *localized* to the one-step→horizon conversion (the arm
  fits teacher-forced to >0.9), routing to the §6.3 exposure-bias levers (self-forcing, latent
  overshooting) + scale, and to the SPEC-8 oracle-grounded objectives.

### SPEC-8 — the oracle-grounded-SSL data factory (OG1/OG2, shipped)

[SPEC-8](docs/specs/SPEC-8.md)'s deterministic, no-GPU machinery is built and property-tested ahead
of the GPU runs (the same discipline that put the oracle/metric/loop in place before any model):

- **OG1 — oracle targets + the decidable/residual partition** ([`netdata/grounding.py`](src/verisim/netdata/grounding.py)):
  for any `(s, a)`, emit the true next-state target, the exact divergence target (equal to
  `netmetrics.divergence` by construction), and the `D`/`R` mask tied to the partial observation.
  Full observation ⇒ `D = s'`, `R = ∅`; an empty observation leaves only the global clock/exit facts
  decidable. The training objective masks `D` and spends gradient on `R` (§4.2).
- **OG2 — the hard-negative / counterfactual factory** ([`netdata/negatives.py`](src/verisim/netdata/negatives.py)):
  *one-edit-wrong* successors (the bits-to-correct neighborhood — exact near-misses contrastive SSL
  most wants and can least easily mine) and *action-branch counterfactuals* `O(s, a')` spanning the
  whole command grammar (the interventional-fidelity set). Every negative is `≠ O(s,a)`; every
  counterfactual is the oracle's exact truth, by construction.

These ship as **data and targets only** — they never edit the oracle, metric, or gate (DD-AR2) —
so the EN8/EN9 runs (collapse-tax ablation, residual vs raw-likelihood supervision, oracle vs VICReg
negatives) plug into a ready apparatus on the same NW8 arm.

## Concepts cheat-sheet

| Term | Meaning | Where |
|---|---|---|
| `O(s, a)` | the **oracle**: deterministic interpreter returning the exact next state + delta | `oracle/`, `netoracle/` |
| `Δ` (delta) | the structured edit set a step makes; `apply(s, Δ)` reconstructs `s'` | `delta/`, `netdelta/` |
| `Mθ` | the **learned proposer** (`predict_delta`); any model behind the `Model` protocol | `model/`, `netmodel/` |
| `d(a, b)` | **divergence**: normalized symmetric set/graph difference, `0` iff identical | `metrics/`, `netmetrics/` |
| `H_ε(ρ)` | **faithful horizon**: first step where `d > ε`, as a function of consultation budget `ρ` | `metrics/horizon.py` |
| `ρ` | **consultation budget** ∈ [0,1]: fraction of steps the oracle is consulted | `loop/policy.py` |
| bits-to-correct | MDL of the oracle's correction of `Δ̂`; `0` iff the prediction is exactly right | `metrics/bits.py` |
| **delta-exact** | per-step: did free decode assemble the exact edit set? (`bits_to_correct = 0`) | `netmetrics/exact.py` |
| full / probe | oracle consultation modes: whole next-state vs one host's local view (cheap) | `netloop/observe.py` |
| `D` / `R` | next-state bits the oracle **decides** vs the genuine **residual** (SPEC-8 partition) | `netdata/grounding.py` |
| reachability-faithfulness | fraction of can-A-reach-service(B) entries that agree | `netmetrics/divergence.py` |

## Design decisions (the load-bearing ones)

| DD | Decision | Why |
|---|---|---|
| **delta prediction** | the model predicts a *structured delta*, not a free-form next state | bounds the hallucination surface; makes `apply == oracle` checkable bit-for-bit |
| **constrained decode** | every prediction is grammar-valid by construction | a model can be wrong but never *malformed*; the parse always succeeds |
| **model-agnostic loop** | the loop never knows which proposer it holds (`Model` protocol) | the contribution is the *method*; H22 asks whether the favorable behavior is the loop's, not a model's |
| **exact headline metric** | reported faithfulness is bit-exact and oracle-grounded; learned signals are *internal* | the oracle calibrates proxies; it is never substituted *for* the truth (DD-3, DD-OG-3) |
| **never latent-ify the checkable part** | latents only ever cover the genuinely-unobserved residual `R` | surrendering verifiability of `D` would give away the whole asset |
| **deterministic core first** | the no-GPU data/metric/loop machinery ships and is property-tested before any training claim | NW0–NW3 / OG1–OG2 discipline; the figure is gated, never assumed |
| **honest negatives are first-class** | every hypothesis pre-registers its refutation branch as a banked result | the oracle makes negatives *trustworthy*; a refutation is often the deeper contribution |

## Verification

The claims above are audited empirically in [docs/verification.md](docs/verification.md): the core
invariants (`apply == oracle`, serialization round-trips, the NW4 tokenizer, metric bounds, exit
codes, in- and cross-process determinism) are proven over **48,000 oracle transitions with zero
failures** by the dependency-free, torch-free [`scripts/verify_invariants.py`](scripts/verify_invariants.py)
— and additionally over the **entire action space** (448,260 state×action pairs) by construction,
with **negative controls** confirming each check detects deliberate corruptions. Every quantitative
number in the report and this README is machine-checked against the committed figure CSVs; the
figures regenerate from config + seeds with `maxΔ = 0`; the NW5 partial-observation loop invariants
are tested (ρ=1 full-consult is exact; a one-host probe corrects strictly less than a full consult);
and the packaging is verified end-to-end (the RL-env return equals the faithful horizon, the
benchmark separates a perfect from a trivial model, coverage spans all 13 commands).

## Packaging for reuse

The env + metric are packaged where researchers already look (SPEC-2 §15):

- **Faithfulness benchmark** ([`verisim.eval`](src/verisim/eval/)) — dependency-free; `score_model` /
  `score_suite` grade *any* model implementing the loop `Model` protocol against the oracle's ground
  truth, and `step_labels` + `grade_prediction` expose single-step labels for question-answer
  frameworks. An `inspect_ai` task adapter ships behind the optional `[eval]` extra.
- **Oracle-as-reward RL environment** ([`verisim.rl`](src/verisim/rl/)) — a `verifiers`-spec
  `WorldModelEnv` (with the `load_environment` entrypoint) whose reward is the oracle's faithfulness
  verdict, so the episode return *is* the faithful horizon.

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

Reproduce every figure (E1–E4, calibration, K0/K2/K4, the EN1 curve, EN2/EN3, the EN4 graph-vs-flat
comparison) from config + seeds:

```bash
bash figures/reproduce.sh
python -m verisim.experiments.en4_graph --graph-iters 1500 --out figures/en4_graph_vs_flat.csv
```

## Layout

See [SPEC-2 §10](docs/specs/SPEC-2.md) (filesystem) and [SPEC-5 §16](docs/specs/SPEC-5.md) (network).
Packages under [src/verisim/](src/verisim/):

- **v0 filesystem world:** `env/`, `oracle/`, `delta/`, `data/`, `metrics/`, `loop/` (runner,
  policies, operators, baselines), `model/` (`M_θ`: vocab, tokenizer, grammar, transformer,
  constrained decoder), `train/` (supervised + minibatch + RLVR), `experiments/` (baselines, E1–E4,
  K0/K2/K4, diagnostics, the autoresearch ratchet `auto/`), and packaging `eval/` + `rl/`.
- **Network world (SPEC-5):** `net/` (typed-graph state, grammar, serialization), `netoracle/`
  (Tier-A oracle), `netdelta/` (graph deltas + `apply`), `netdata/` (drivers + generation + the
  **SPEC-8 grounding/negatives factory**), `netmetrics/` (graph divergence, reachability-faithfulness,
  bits-to-correct, **delta-exact**), `netmodel/` (the flat NW4 `M_θ` + the **NW8 GNN+RSSM graph arm**),
  and `netloop/` (the NW5 partial-observation runner, two-mode oracle, probe policies, operators).
  Experiments: `experiments/en1.py` … `en4_graph.py`.

Experiment configs live in [configs/](configs/); plotting scripts + figures in [figures/](figures/);
the run-records they read are git-ignored and regenerable from config + seeds.

## License & posture

MIT (see [LICENSE](./LICENSE)). This is a research repo: **no telemetry, no network calls at runtime,
no commercial path.** The framing and downstream agents are defensive; see
[SPEC.md §13](docs/specs/SPEC.md) for the ethics and dual-use posture.

Author: Clay Good.
