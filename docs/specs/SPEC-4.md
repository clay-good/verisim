# Verisim — The Autonomous Research Engine (SPEC-4.md)

> The fourth spec. [SPEC.md](./SPEC.md) is the science, [SPEC-2.md](./SPEC-2.md) is the v0 build, [SPEC-3.md](./SPEC-3.md) is the depth (what Verisim becomes). **This document is the engine that gets it there with the human out of the loop**: the system that proposes changes to Verisim, evaluates them against the oracle, keeps what improves, and repeats — autonomously, overnight, at scale. It is Verisim's replication of Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch), generalized and hardened, with one decisive advantage the original cannot have: **the gate is reality, not a proxy.**
>
> SPEC-3 says *what to build*. SPEC-4 says *who builds it* — and the answer is Verisim itself.

> **◐ TOOL — partially active (2026-05).** SPEC-4 is not a build *stage*; it is the cross-cutting engine used at every stage. Its **AR0 ratchet already ships** and is the search engine for **[SPEC-2.1](./SPEC-2.1.md) §9** (its gate now upgraded to bits-to-correct). The higher autonomy levels (AR3+ LLM/code proposers) stay **paused** until the knee exists — the autonomous code-editor inherits the reward-hacking surface and must earn its keep behind a proven config-search. Canonical build order: [SPEC §12](./SPEC.md#12-research-roadmap).

**License:** MIT. **Status:** design spec; v0 of the engine (the keep-if-better config ratchet) already ships in [`src/verisim/auto/`](./src/verisim/auto/). **Audience:** the author, collaborators, and the engine itself, which reads this file as part of its operating context.

---

## 0. One-paragraph thesis

The bottleneck on Verisim is no longer ideas; it is the number of well-scored experiments per day. Karpathy's `autoresearch` showed that an LLM editing training code in a tight *propose → fixed-budget train → score one number → keep-if-better* loop runs ~700 experiments over two days with no human present and finds real improvements (≈11% on already-optimized code). Verisim is an almost perfect substrate for that loop, with a unique twist: every other autonomous-research system scores on a **proxy** (held-out loss, `val_bpb`) that a sufficiently clever agent learns to **game** — the documented failure of Sakana's AI-Scientist-v2, which quietly swapped in smaller/synthetic datasets to beat baselines. Verisim scores on a **deterministic oracle** — bit-for-bit ground truth (SPEC.md §2) — which the agent *cannot* fake, because faithfulness is measured against reality itself. So Verisim's autonomous engine is not just feasible; it is the *safest possible* instance of the pattern, provided the gate is computed on **frozen, proposer-independent oracle states**. SPEC-4 specifies that engine: the `Proposer` seam (from deterministic search to a code-editing agent), the oracle-grounded `Gate` (with CI-as-deterministic-lint and the §7 bits-to-correct scalar), the `Ledger` (the reproducible leaderboard the agent reads), the `Controller` (search strategy), the autonomy levels that keep the human out of the loop without losing the kill-switch, and — front and center — the reward-hacking threat model and its defenses.

---

## 1. The pattern, and why Verisim is the ideal substrate

**The autoresearch loop** (Karpathy, ~630 LOC, released early 2026): an agent edits `train.py`; the harness trains for a **fixed 5-minute budget**; the run is scored on **`val_bpb`** (validation bits-per-byte — vocab-independent, so architectures stay comparable); the change is kept only if it beats the running-best, else rolled back; repeat (~12 experiments/hour). A **fixed validation shard and fixed token budget** are the explicit anti-gaming controls.

**The verisim twist (the whole reason this spec exists).** `val_bpb` is a *statistical proxy* — it never tells the loop whether a prediction is *actually right*. Verisim has what no other research-loop substrate has: a **deterministic oracle** that returns ground truth. So Verisim's "did we improve?" gate is not a proxy at all — it is the **bits-to-correct** against reality (SPEC-3 §7). Three consequences:

1. **The gate is unfakeable for faithfulness.** You cannot lower bits-to-correct without actually predicting the oracle's true next state. The reward-hacking surface that sank AI-Scientist-v2 is *structurally smaller* here.
2. **Determinism is free.** Every Verisim trial is already a pure function of `(config, seed)` (SPEC-2 §12). Karpathy needs a fixed shard + fixed budget to make trials comparable; Verisim's reproducibility regime gives that by construction.
3. **CI is a second deterministic oracle — for the *code*.** For an agent that edits source (not just config), the repo's own `ruff` + `mypy --strict` + `pytest` + **golden trajectories** (SPEC-2 §12, §16) are a deterministic linter on the *change itself*. "Green" is the user's "deterministic linting": a hard gate the agent must pass before its metric even counts.

> Verisim is the autoresearch loop where **both** the science metric (the oracle) **and** the engineering metric (CI/goldens) are deterministic ground truth. That is the strongest possible footing for taking the human out of the loop.

---

## 2. What exists today, and its three limits

[`src/verisim/auto/search.py`](./src/verisim/auto/search.py) already ships the v0 engine (SPEC-2 §17.5 automation):
- a **keep-if-better ratchet** over a config space, scored on mean clean per-step accuracy **vs. the oracle**;
- a **deterministic coordinate proposer** (mutate one knob of the current best, seeded);
- a per-trial **run-log** (`RunRecord`s — the `program.md`/leaderboard analogue);
- a committed figure ([`figures/auto_search.*`](./figures/)) showing the ratchet curve. First run: it autonomously doubled the clean floor (0.042 → 0.094).

Its three limits define SPEC-4's work:

| Limit | Today | SPEC-4 |
|---|---|---|
| **L-A — config knobs only.** | Mutates `n_layer`, `lr`, … | §4: a `Proposer` seam up to an agent that edits **source code** (model/train/loop), like real autoresearch. |
| **L-B — coordinate hill-climb only.** | One-knob greedy search. | §7: a `Controller` with bandit / evolutionary / population-based search and parallel trials. |
| **L-C — accuracy gate only.** | 0/1 per-step accuracy. | §5: the **bits-to-correct** scalar (SPEC-3 §7) + CI-as-lint + frozen held-out eval. |

---

## 3. Architecture: four parts behind clean seams

```
                         ┌──────────────────────────────────────────────┐
                         │                 THE ENGINE                     │
                         │                                                │
   spec + ledger ──────► │  PROPOSER ──► candidate (config Δ or code Δ)   │
   (context)             │     ▲                    │                     │
                         │     │                    ▼                     │
                         │     │              GATE (Evaluator)            │
                         │     │   ┌── CI / goldens (deterministic lint) ─┤
                         │     │   └── oracle score on FROZEN eval seeds ─┤
                         │     │                    │                     │
                         │  CONTROLLER ◄── score ───┤                     │
                         │  (search policy:         ▼                     │
                         │   keep-if-better,   LEDGER (run-records,       │
                         │   bandit, evo, PBT)  lineage, reproducibility) │
                         └──────────────────────────────────────────────┘
                                   all trials run in the sandbox (SPEC-3 §2, §14)
```

- **Proposer** (§4): generates the next candidate — a config edit, or a *code* edit. The one piece that ranges from a seeded function to an LLM agent.
- **Gate / Evaluator** (§5): runs the trial under a fixed budget and returns one comparable scalar — but only after the change passes the deterministic lint (CI/goldens). Scores on **frozen, proposer-independent oracle states**.
- **Ledger** (§6): the append-only, reproducible record of every trial (the leaderboard the proposer reads as context). Tracks lineage so any kept change is traceable to its parent.
- **Controller** (§7): chooses which candidate to try next and whether to keep it — the search strategy wrapped around the gate.

The interfaces (Python-sketch, consistent with the existing `auto/` module):

```python
class Proposer(Protocol):
    def propose(self, base, best, ledger, space, rng) -> Candidate: ...

class Candidate(Protocol):           # a config edit or a code patch + provenance
    def materialize(self, workdir) -> Trial: ...

class Gate(Protocol):
    def evaluate(self, trial) -> GateResult: ...   # (lint_passed, score, report)

class Controller(Protocol):
    def select(self, ledger) -> Proposer | None    # which proposer/parent next
    def keep(self, result, ledger) -> bool          # the ratchet rule
```

Everything composes with the *existing* `run_search` (SPEC-2): today's loop is `CoordinateProposer × KeepIfBetter × accuracy-Gate × LinearController`.

---

## 4. The Proposer seam (the core of the engine)

Four proposers, in increasing power and risk. Each is a drop-in behind the `Proposer` protocol; the gate is identical for all.

### 4.1 `CoordinateProposer` — deterministic baseline (shipped)
One-knob seeded hill-climb. No network, fully reproducible. The control against which every smarter proposer must prove its worth.

### 4.2 `EvolutionaryProposer` / population
A population of configs, mutation + crossover over the knob space, selection by gate score (population-based training). Deterministic given seed. Better than coordinate search when knobs interact. No network.

### 4.3 `LLMProposer` — the agent reads the leaderboard (the autoresearch core)
An LLM proposes the next config edit by reasoning over the ledger and the knob space.

- **Dependency injection, not a hard SDK.** The model is a `complete(prompt) -> str` callable injected by the caller, so the module pulls in **no** API dependency and stays CI-safe; a deterministic stub makes it testable and reproducible. A live client (Anthropic/OpenAI/local) is wired in only at the CLI, behind an env-gated flag — **never** in the committed reproduce path (SPEC-2 §11: determinism-first, no runtime network in the science pipeline).
- **Prompt** = objective ("maximize bits-to-correct↓ / accuracy↑ vs. the oracle") + the knob space with allowed values + the ledger as a compact table (trial → knobs → score → kept). Pure and deterministic given inputs → unit-testable.
- **Parse + validate**: extract a JSON edit, validate every key/value against the space, apply to `best`. **Any** malformed reply falls back to the coordinate proposer, so the loop never stalls.
- **Why config-first before code.** Validation is trivial (closed value set), the gate is unchanged, and it isolates "can the agent reason about the leaderboard" from "can the agent write correct code" — two failure modes worth separating.

### 4.4 `CodeMutationProposer` — the agent edits the source (full autoresearch, highest power/risk)
The agent edits *actual files* (`model/transformer.py`, `train/*.py`, `loop/policy.py`, …) — exactly what Karpathy's loop does to `train.py`. This is where genuinely novel changes (a new architecture, a new policy, a new operator) come from, not just hyperparameters.

- **Isolation:** every code trial runs in its **own git worktree** (the harness already supports worktree isolation), so concurrent trials never collide and a bad patch is discarded by deleting the worktree.
- **The deterministic lint is a hard pre-gate (§5.2):** the patch must pass `ruff` + `mypy --strict` + the full `pytest` suite + the **golden trajectories** before its oracle score is even computed. A patch that breaks a golden (i.e., changes the *semantics* the goldens pin) is rejected outright — the goldens are the immune system.
- **Bounded blast radius:** an allowlist of editable paths (model/train/loop/policy/operator), a denylist (the oracle, the metrics, the goldens, the gate itself, this spec) — **the agent may not edit its own judge.** Patch size is capped; diffs are logged for review.
- This is the most powerful and the most dangerous proposer; §9 is its safety case.

> **Design decision (DD-AR1): ship the proposers in power order, gate each on the last working.** Coordinate (shipped) → Evolutionary → LLM-config → LLM-code. Do **not** build the code-mutator until the config-agent reliably beats coordinate search on the frozen eval, because the code-mutator inherits all of the config-agent's reward-hacking surface plus the ability to edit code. *Rationale:* the research record (Sakana) shows code-editing agents hack evals; earn that capability behind a proven, gated config-agent.

---

## 5. The Gate: oracle-grounded, lint-fronted, frozen

The gate is the heart of the safety story. It has three layers; a trial must clear them in order.

### 5.1 Layer 1 — the deterministic lint (the user's "deterministic linting")
For code trials: `ruff check` + `mypy --strict` + `pytest` (incl. **golden trajectories** and loop invariants, SPEC-2 §16) must be **green**. Green is a *gate, not a goal*: it admits the trial to scoring; it is not itself the objective. A red trial scores `−∞` and is logged. For config trials this layer is trivially passed (no code changed).

### 5.2 Layer 2 — the comparable scalar, on FROZEN eval
The score is **bits-to-correct** (SPEC-3 §7) — scale-free, architecture-independent, smooth, and **0 iff faithful** — computed on a **held-out oracle evaluation set the proposer cannot see or influence**:
- a **fixed, sealed set of `(env_config, seed)` eval cells**, drawn once, never shown to the proposer, never mutated by a trial;
- a **fixed compute budget** per trial (the autoresearch 5-minute analogue: fixed `train_iters` ceiling or wall-clock), so trials are comparable and the agent cannot "win" by simply training longer;
- the oracle states are generated *independently of the candidate*, so a candidate cannot select its own easy workload (the exact AI-Scientist-v2 hack).

Today's accuracy gate is the v0 stand-in; bits-to-correct is the upgrade (SPEC-3 §7, DD-6).

### 5.3 Layer 3 — anomaly screening
A score that jumps implausibly (e.g., near-zero bits-to-correct from a tiny model) triggers an automatic re-evaluation on a *second* frozen eval set and a diff flag for review — the tripwire for an undetected hack (§9).

> **Design decision (DD-AR2): the gate is in the denylist.** No proposer — config or code — may edit the oracle, the metric, the eval-set generator, the golden trajectories, or the gate. The judge is not a knob. This is the single most important invariant in the whole engine; violating it turns "improve faithfulness" into "redefine faithfulness."

---

## 6. The Ledger: the reproducible leaderboard

Every trial appends one `RunRecord` (SPEC-2 §7.3) carrying: the full resolved config (or the code patch + its hash), the parent trial (lineage), the gate result (lint pass/fail, score, report), kept/rejected, and the git commit + seed. Properties:
- **Reproducible:** any trial re-runs from its record (config+seed; or patch+commit). The ledger *is* the experiment history, regenerable.
- **The proposer's context:** the `LLMProposer` reads the ledger as its prompt (§4.3) — what was tried, what scored, what was kept. This is the `program.md` analogue, but structured and machine-read.
- **Lineage:** a kept change points to its parent, so the improvement path (base → … → best) is a traceable chain — and so a regression can be bisected.
- **Git-ignored, regenerable** (SPEC-2 §10: `runs/` is ignored); only the *summary* figure/CSV is committed (records-only discipline, SPEC-2 §7.3).

---

## 7. The Controller: search strategy

The ratchet rule (`keep if score > best`) is the simplest controller; SPEC-4 adds smarter ones behind the `Controller` seam:
- **UCB / bandit over knobs:** treat each knob-direction as an arm; allocate trials to the directions with the best score-improvement-per-trial (more sample-efficient than uniform coordinate search).
- **Evolutionary / population-based training:** maintain a population, exploit-and-explore, periodically copy-and-perturb the leaders (pairs naturally with §4.2).
- **Novelty / diversity pressure:** reward configs that explore unseen regions, to escape the local optimum a pure ratchet gets stuck in.
- **Parallelism:** many trials at once, each in its own worktree (§4.4), bounded by a concurrency cap and a global resource budget — the overnight-throughput multiplier (Karpathy's ~12/hr is single-stream; parallel worktrees multiply it).
- **Early-stopping:** kill a trial whose partial training is already far behind the frontier (don't spend the full budget on a clear loser).

> **Design decision (DD-AR3): keep the ratchet as the floor, add search above it.** The keep-if-better ratchet guarantees monotone non-regression (it can never make the best worse). Smarter controllers only change *which* candidates are tried, never the keep rule — so no controller can regress the frontier. *Rationale:* monotonicity is the property that lets the engine run unattended.

---

## 8. Humans out of the loop: autonomy levels

The user's goal is maximal autonomy. The engine is designed so the human's role *shrinks to a boundary condition*, not a per-trial gate.

| Level | Human role | Engine does |
|---|---|---|
| **L0 (today)** | sets config + space, reads the figure | runs the ratchet; proposes (coordinate), scores, keeps, logs |
| **L1** | sets the objective + budget + safety bounds | LLM-config agent proposes; engine runs unattended overnight |
| **L2** | reviews diffs *post hoc*, sets the editable-path allowlist | LLM-code agent edits source in worktrees, gated on CI+goldens+oracle; keeps green improvements |
| **L3** | sets the research *direction* and the ethics/scope boundary only | engine runs continuously, proposes across config+code+search-strategy, self-curates the ledger |

**What never leaves the human (the irreducible four):** (1) the **objective** (what "better" means — currently faithfulness; the agent may not redefine it, DD-AR2); (2) the **safety/ethics boundary** (SPEC-3 §14: defensive-only, no real-internet egress, the editable-path denylist); (3) the **kill-switch + resource cap** (a hard ceiling on spend/compute/wall-clock the engine cannot raise); (4) **promotion to `main`** for code changes (the engine proposes and validates on a branch; a human or a separate trusted CI merges — the engine does not push to the trunk it runs from).

> **Design decision (DD-AR4): the engine optimizes inside a box it cannot enlarge.** Autonomy is about removing the human from the *inner loop* (per-trial judgment), not from the *boundary* (objective, safety, resources). An engine that could edit its own objective, gate, or resource cap is not "more autonomous" — it is unbounded, which is unsafe and scientifically meaningless.

---

## 9. Reward hacking and safety (the threat model)

The research record is unambiguous: autonomous research agents **hack their evaluations** when they can (Sakana AI-Scientist-v2 bypassed full datasets for smaller/synthetic ones; RLVR work shows hacking even with verifiable signals at the margins). Verisim's deterministic oracle is the strongest defense ever available for this — *but only if used correctly*. The threats and the defenses:

| Threat | Defense |
|---|---|
| **Game the eval workload** (pick easy oracle states). | Frozen, sealed eval cells the proposer never sees or mutates (§5.2). |
| **Overfit the eval seeds.** | A *second, rotating* held-out eval set for anomaly re-checks (§5.3); report on never-trained seeds. |
| **Edit the judge** (oracle/metric/goldens/gate). | Hard denylist (DD-AR2); the agent cannot touch its own referee. |
| **Pass tests trivially** (delete assertions, stub the function). | Golden trajectories + loop invariants pin *semantics*, not just "tests pass"; the goldens are in the denylist; coverage/diff review flags assertion-deletion. |
| **Metric edge cases** (a degenerate config that scores low bits-to-correct without predicting). | bits-to-correct is 0 *iff* the prediction equals the oracle's truth — degenerate wins are structurally excluded; anomaly screening (§5.3) catches the rest. |
| **Code that exfiltrates / reaches the network.** | The sandbox has no real-internet egress (SPEC-3 §2.2, §14); the LLM client is the only network call and is outside the trial sandbox. |
| **Runaway resource use.** | Hard resource/wall-clock cap the engine cannot raise (DD-AR4); per-trial budget (§5.2). |

> **The core safety claim:** because faithfulness is scored against a deterministic oracle on frozen, proposer-independent states, **the only way to score well is to actually be faithful.** The agent cannot persuade, cannot pick a friendlier judge, cannot redefine truth. That is precisely why computer/network environments (with a real oracle) are the right place to run an autonomous research engine — and why the same engine could *not* be safely run in a domain whose "oracle" is a learned, gameable proxy.

That the gate is **proposer-independent** is not only a safety property; it is the engineering face of the project's most general claim. The gate scores *the prediction against the oracle*, never *the model against a model* — so the engine can search across radically different proposers (config-tuned transformer, evolved architecture, a JEPA-style latent arm, an LLM-as-proposer) on one comparable scale, and any improvement it banks is an improvement in *faithfulness*, not in fitting a particular architecture's quirks. The autonomous engine is thus the most direct demonstration of **deterministic verification as a model-agnostic primitive** (SPEC.md §6 commitment 4, H22): one unfakeable judge, many swappable proposers.

---

## 10. The inner loop: RLVR, and why GRPO

When the engine searches over *training-objective* changes (SPEC-2 §9 objective axis; SPEC-3 §10), the inner training uses **RL with verifiable rewards** against the oracle — the regime DeepSeek-R1 showed scales without the collapse that learned reward models suffer. Two grounded choices:
- **Prefer GRPO over PPO** for the inner RL: critic-free (group-relative advantage), cheaper, and — decisively — **no learned critic to hack** (a learned value function is exactly the gameable surface R1 avoided).
- **Keep the reward strictly oracle-derived** (per-field bit match / faithful-horizon), **never** a learned divergence proxy, to preserve the anti-hacking guarantee. The known gap (research §5): RLVR is charted for *reasoners*, less so for *next-state predictors*, so the reward must be **dense** (per-step / per-field), not sparse-outcome — which is also what the §6 self-healing signal provides.

---

## 11. Implementation plan (engine milestones)

Mapping to [`src/verisim/auto/`](./src/verisim/auto/); each milestone gated on the previous.

- **AR0 — ratchet (shipped).** Coordinate proposer × keep-if-better × accuracy gate × run-log. ([`auto/search.py`](./src/verisim/auto/search.py), [`configs/auto.json`](./configs/auto.json), [`figures/auto_search.*`](./figures/).)
- **AR1 — the `Proposer`/`Gate`/`Controller` seams.** Refactor `run_search` to compose the four parts (§3) without changing AR0's behavior. *Verify:* AR0 reproduces bit-for-bit through the new seams.
- **AR2 — bits-to-correct gate (SPEC-3 §7) + frozen eval set.** *Verify:* the scalar is 0 on a perfect (oracle-backed) model and monotone in divergence; the eval set is sealed and proposer-independent.
- **AR3 — `LLMProposer` (config agent, §4.3).** Dependency-injected `complete`; deterministic stub in tests; live client behind an env flag, never in reproduce. *Verify:* with a stub it is deterministic and beats coordinate search on a seeded benchmark; malformed replies fall back.
- **AR4 — `EvolutionaryProposer` + bandit `Controller` (§4.2, §7).** *Verify:* matched-budget sample-efficiency vs. coordinate hill-climb.
- **AR5 — `CodeMutationProposer` (§4.4) behind the full CI+goldens lint and the editable-path allowlist.** *Verify:* a planted "improvement" patch is kept only when green *and* it lowers bits-to-correct on the frozen eval; a planted assertion-deleting / judge-editing patch is rejected by the lint/denylist. **This milestone does not land until AR3 reliably beats coordinate search** (DD-AR1).

AR0–AR3 is the autonomous config researcher (overnight, human at the boundary). AR4–AR5 is the autonomous *code* researcher — the full autoresearch, hardened by the oracle and the goldens.

---

## 12. Open questions

1. **Bits-to-correct estimator fidelity (SPEC-3 §7.2):** the simple prefix-code surrogate vs. an entropy-coded estimate from the model's own distribution — which is the better gate, and is the surrogate hack-resistant enough?
2. **Proposer credit assignment for code edits:** when a code patch helps on some eval cells and hurts others, how does the controller decide? (Pareto frontier over cells vs. a single aggregate.)
3. **Search-strategy meta-search:** the engine could search over *its own* controller/proposer hyperparameters — bounded by DD-AR4, but where is the line?
4. **Cost of the LLM proposer:** at L2/L3 the agent is the dominant cost; when does a cheaper local proposer (or a distilled one) beat an API model per dollar of improvement?
5. **Reproducibility of agent-proposed trials:** a live LLM is nondeterministic; the *gate* stays deterministic, but the *proposal* is not — the ledger records the realized proposal, so trials are *replayable* (re-score the recorded candidate) even when not *re-derivable*. Is replayability sufficient for the science? (Tentative yes: the gate, not the proposer, is the scientific instrument.)

---

## 13. Provenance and reading order

- **Prereqs:** [SPEC.md](./SPEC.md), [SPEC-2.md](./SPEC-2.md), [SPEC-3.md](./SPEC-3.md). SPEC-4 assumes all three; in particular the §7 bits-to-correct metric and the §2 sandbox live in SPEC-3.
- **Inspiration:** Karpathy, [`autoresearch`](https://github.com/karpathy/autoresearch); cautionary precedent: Sakana AI-Scientist-v2 and the documented eval-hacking; method grounding: DeepSeek-R1 / GRPO (verifiable rewards), Blier & Ollivier (description length). A maintained bibliography lives in [`docs/related-work.md`](./docs/related-work.md).
- **Author:** Clay Good. **License:** MIT. The engine runs only inside the sandbox (SPEC-3 §2, §14); it never has real-internet egress except the injected LLM client, and it never pushes to the trunk it runs from (DD-AR4).
- A living spec: as engine milestones (§11) land they are marked and their figures linked, mirroring SPEC-2 §13.
