# SPEC-18 — The Product: A Ground-Truth Faithfulness Benchmark and Sim-to-Emulation ACD Environment

**Packaging-and-validation specification: every spec to date *produced* a result; this one *ships* one.
The program has accumulated a unique asset — a world-model faithfulness benchmark with **ground-truth
labels** (the oracle), and an oracle-grounded learned network/host world model — and that asset is still
scattered across `eval/`, `rl/`, `hostrl/`, `disteval/`, and a dozen `figures/`. SPEC-18 hardens it into
two versioned, reproducible, community-usable artifacts: (1) the **faithfulness benchmark** — the one
world-model benchmark with a real ruler, an `H_ε(ρ)` leaderboard whose labels are exact; and (2) the
**ACD environment** — a learned-but-faithful network/host simulator, cheap to roll out like a sim and
oracle-verifiable like an emulator, that closes the sim-to-emulation gap §4.1 names. The science it adds
is two falsifiable claims the field has never measured: does the benchmark *discriminate* proposers
(stable rankings), and is the sim-to-emulation transfer *measured* against the SPEC-11 real-OS oracle —
the quantified version of what CybORG-class fields assert and leave open.**

> **▶ PRODUCT / VALIDATION SPEC — 2026-06.** A *packaging-and-validation* spec, sibling to
> [SPEC-11](./SPEC-11.md) (the system oracle — the real-OS anchor this spec's transfer claim is measured
> against). It invents **no new world, oracle, or model.** It consumes what already ships: the
> Inspect faithfulness task ([`eval/inspect_task.py`](../../src/verisim/eval/inspect_task.py)), the
> framework-agnostic scorer ([`eval/faithfulness.py`](../../src/verisim/eval/faithfulness.py)), the
> `verifiers`-spec RL env ([`rl/environment.py`](../../src/verisim/rl/environment.py) with
> `load_environment`), and the same trio for the host and distributed worlds
> ([`hostrl/`](../../src/verisim/hostrl/), [`hosteval/`](../../src/verisim/hosteval/),
> [`distrl/`](../../src/verisim/distrl/), [`disteval/`](../../src/verisim/disteval/)). What it adds is the
> *product layer* above them: a frozen versioned eval battery, a leaderboard with rank-stability
> statistics, Croissant/datasheet/model-card metadata, Gymnasium+verifiers conformance tests, and a
> transfer measurement against the SPEC-11 `SandboxOracle`. Two distinct epistemic registers live here
> and are kept apart on purpose: the *science* is pre-registered as falsifiable hypotheses (H65–H68); the
> *packaging* is a milestone table (PB0–PB6, the SPEC-11 §8 form). **Adoption is not a hypothesis** (§9).

Read [SPEC.md §14](./SPEC.md) (the four concrete artifacts: the measurement, the technique, the *reusable
artifact*, the bridge result — this spec ships #3 and measures #4), [SPEC.md §4.1](./SPEC.md) (the ACD
sim-to-emulation gap), [SPEC.md §10.1](./SPEC.md) (why a bankable negative is a result), and
[SPEC-11](./SPEC-11.md) (the `SandboxOracle` — the real `/bin/sh`/real-kernel substrate the transfer is
measured against). This document is *whether the asset discriminates, whether the transfer is real and by
how much, and how the whole thing is frozen, versioned, and regenerated.*

---

## 0. One-paragraph thesis

The program's structural asset is the one thing the oracle-free world-model field cannot have: a
faithfulness benchmark with **exact ground-truth labels**. Generative world models (Genie, Dreamer,
V-JEPA) are evaluated by eyeball, FID, or held-out rollout error — never against a checkable next-state
oracle, because their domains have none (SPEC.md §1.1). Verisim's `H_ε(ρ)` is that missing ruler, and the
`load_environment`/Inspect surfaces to expose it already ship. SPEC-18's thesis, stated to be falsified,
is two-pronged: **(1) the faithfulness benchmark is *discriminative* — over a frozen world/seed battery it
ranks proposers (flat transformer, GNN+RSSM, trivial/frequency, oracle-ceiling) in an order that is
*stable across seeds*, so the leaderboard measures the model and not the draw; and (2) the
sim-to-emulation transfer is *measurable and quantified* — a policy/world-model trained against the
reference oracle, evaluated against the SPEC-11 system oracle (real `/bin/sh`, real kernel), exhibits a
transfer gap we report as a number with a CI, turning the field's *asserted* "full transferability"
(CyGIL) into a measured `Δ`.** Both branches are bankable in the program's sense (SPEC.md §10.1): if the
rankings are seed-unstable the benchmark lacks discriminative validity and we have located a fixable
measurement defect (more seeds / a tighter `ε` / a harder battery); if the transfer gap is large, *that
gap is the result the ACD field needs* — the first quantification of a transfer everyone asserts and no
one measures. The packaging that surrounds these two claims — frozen battery, Croissant metadata,
datasheet/model-card, conformance tests, semantic versioning, `reproduce.sh` coverage — is engineering
graded against milestones (PB0–PB6), and the program's anti-gaming discipline (SPEC-4: frozen,
proposer-independent eval) is what keeps the leaderboard honest as the community pushes on it.

---

## 1. Why now: the asset exists, scattered; the field has a hole shaped like it

Three facts make this the right time to package rather than to discover.

**The artifact already ships in pieces.** `eval/inspect_task.py` exposes the single-step faithfulness
task as an `inspect_ai` `Task` with a divergence scorer; `eval/faithfulness.py` scores *any* `loop.Model`
(learned or symbolic) for its rollout faithful horizon `H_ε` with no torch dependency; `rl/environment.py`
is a `verifiers`-spec env with a `load_environment` entrypoint whose **return equals `H_ε`** (the headline
metric *is* the RL return); and the same three surfaces exist for the host (`hostrl`/`hosteval`) and
distributed (`distrl`/`disteval`) worlds. SPEC-18 does not build these — it *freezes, versions, documents,
and validates* them.

**The eval ecosystems have converged on exactly the two sockets verisim already fills.** Inspect /
`inspect_evals` (UK AISI, with Arcadia Impact and the Vector Institute) is the framework labs and safety
institutes use, with a published submission standard (unit tests for custom scorers, verification tests
for dynamic resources). Prime Intellect's `verifiers` spec + the Environments Hub standardize an RL env
behind a `load_environment` entrypoint, distributed as a wheel with `pyproject.toml`-declared deps. Both
are the *distribution targets* SPEC.md §14 #3 names, and verisim's adapters already conform to them. The
remaining work is product hardening, not invention.

**The benchmark-quality field has documented precisely the gap a verisim product can fill.** BetterBench
(Reuel et al., NeurIPS 2024) assessed 24 widely-used benchmarks across 46 lifecycle best practices and
found the *implementation* and *maintenance* stages weakest: 17 of 24 ship no easy replication script, and
standardized metadata is "fulfilled by almost no benchmark." The program's standing discipline — every
figure regenerable from `reproduce.sh`, deterministic and seeded — is *already* the practice BetterBench
finds missing; SPEC-18 adds the metadata layer (Croissant; datasheet/model-card for the benchmark) that
closes the rest. The world-model angle is the differentiator: there is no ground-truth faithfulness
metric in the Genie/Dreamer/JEPA evaluation literature, because those domains have no oracle (SPEC.md
§1.1). Verisim's leaderboard is the one that can.

And the ACD field has an *unquantified* claim sitting exactly where verisim can measure it. CyGIL (Li et
al., 2023) reports a simulated `CyGIL-S` whose agent is "transferrable directly" to the emulated
`CyGIL-E`, claiming "full transferability"; CybORG/CybORG++ (Emerson et al., 2024) and PrimAITE/FARLAND
live with the same sim-vs-emulation tension (cheap-but-unfaithful vs faithful-but-expensive, SPEC.md
§4.1). None of them report a *number* for the gap, because none has a free exact oracle on both sides.
Verisim does: the reference oracle on the cheap side, the SPEC-11 `SandboxOracle` (real `/bin/sh`, real
kernel) on the faithful side. The transfer gap is, here, a measurement — not an assertion.

---

## 2. The two artifacts (and what each one *is*, precisely)

### 2.1 Artifact A — the faithfulness benchmark (`verisim-bench`)

A **frozen, versioned battery** of `(world, driver, seed)` rollouts with exact oracle ground-truth labels,
plus two scoring granularities and a leaderboard:

- **Rollout faithfulness** — `eval/faithfulness.score_model` runs a proposer through the
  propose–verify–correct loop on each battery rollout and reports `H_ε`, final divergence, and oracle
  calls. The headline leaderboard column is `H_ε(ρ)` at a small grid of budgets `ρ ∈ {0, ρ*, 1}` (free-run
  floor, a sweet-spot, oracle-ceiling), per world (FS / host / network / distributed).
- **Single-step labels** — `eval/faithfulness.step_labels` + the Inspect task: the per-step
  `(serialize(s,a) → s')` QA pairs and the `divergence`-based grader, the natural shape for Inspect's
  `dataset → Task → Solver → Scorer` pipeline. Already an `inspect_ai` `Task`; SPEC-18 pins the suite,
  adds the BetterBench-style metadata, and submits to `inspect_evals`' standard (scorer unit tests; a
  dynamic-resource verification test for the frozen battery).
- **The leaderboard** — a committed table over the **reference proposers** (flat transformer; GNN+RSSM;
  trivial/frequency floor b3; oracle-ceiling b1) with `H_ε(ρ)` and bits-to-correct per world, plus the
  **rank-stability statistic** (§5, H65) that says whether the ordering is real. The leaderboard is the
  product's face; the rank-stability number is what licenses calling it a *benchmark* rather than a table.

What makes it the one world-model benchmark with a real ruler: the `target` in every Inspect `Sample` is
the **oracle's** canonical next state, not a human label or a held-out frame. The grade is exact
divergence from ground truth. No oracle-free world-model benchmark can produce that column.

### 2.2 Artifact B — the ACD environment (`verisim-acd`)

The learned-but-faithful network/host world model exposed as **both** a Gymnasium-conformant env and a
`verifiers`-spec env (`load_environment`), for defensive ACD research:

- **As a fast sim:** the learned `M_θ` rolls forward without an oracle consult — cheap, like CybORG's
  simulation tier, so a defensive policy can train over many episodes.
- **As a verifiable emulator:** the oracle is in the loop on a budget — every claimed transition can be
  adjudicated against ground truth, the property an emulator has and a sim lacks. This is the
  sim-to-emulation gap (SPEC.md §4.1) closed by construction: one env that is cheap to roll out *and*
  oracle-checkable.
- **The reward is verifiable.** Per `rl/environment.py`, with `terminate_on_divergence` the return equals
  the faithful horizon `H_ε` — no learned reward model is in the loop (the verifier-as-reward thesis,
  SPEC.md §8). A defensive task reward (detection/containment) plugs in on top via the same socket.

**Ethics gate (SPEC.md §13), load-bearing here.** `verisim-acd` is a *defensive* substrate: it surfaces
reachability, detection, and resilience dynamics for blue-team policies and is shipped with the SPEC-11
hermeticity contract (no egress, dropped caps) as its execution boundary. It is **not** a packaged
offensive agent and ships none; any environment encoding real exploit dynamics is reviewed before release
(SPEC.md §13 commitment 3). The README states intended use and a responsible-disclosure posture.

---

## 3. The scientific spine (what is *measured*, not asserted)

The packaging is engineering; these two questions are the science, and the product is only worth shipping
if both are answered honestly.

1. **Does the benchmark discriminate?** A benchmark that cannot separate a good proposer from a bad one,
   *stably across seeds*, is a table, not a ruler. SPEC-18 measures rank stability directly (H65) — the
   BetterBench "statistical significance" criterion, made first-class. This is the discriminative-validity
   question every benchmark *should* answer and most (per BetterBench) do not.
2. **Is the sim-to-emulation transfer real, and how big is the gap?** The field asserts transferability;
   verisim measures it. A proposer/policy fit against the **reference** oracle is scored against the
   SPEC-11 **system** oracle, and the drop in `H_ε` (and in any downstream policy metric) is the transfer
   gap (H66). A second-order question — does *oracle-in-the-loop correction during transfer* shrink the
   gap (H67) — tests whether the program's own loop is the bridge, not just the measuring stick.

These are exactly the contributions SPEC.md §14 lists as #1 (a measurement no one else can make) and #4
(the bridge result), now packaged behind a frozen, contamination-resistant battery (#3).

---

## 4. Anti-gaming: the frozen, proposer-independent eval (the SPEC-4 discipline, applied)

A public leaderboard invites optimization-against-the-test. The program already has the antidote: the
oracle is the anti-hacking asset *only on a frozen, proposer-independent eval* (SPEC-4 §5, §9; the
AI-Scientist eval-hacking cautionary tale, related-work.md). SPEC-18 makes that concrete for the product:

- **The battery is frozen and content-addressed.** `(world, driver, seed)` tuples and the resulting label
  set are pinned by a manifest hash committed to the repo; a submission is scored against *that* hash. A
  changed battery is a new benchmark *version* (§7), never a silent edit.
- **The labels are oracle-derived, not model-derived.** The ground truth is the reference oracle's exact
  next state — there is no learned grader to game (DeepSeek-R1's reason for refusing neural reward models,
  related-work.md). The only way to score well is to predict the real next state.
- **A held-out shard is reserved.** Following Karpathy-autoresearch's fixed-shard/budget anti-gaming, a
  fraction of the battery is held out of the public manifest and used to detect overfit submissions
  (public-score ≫ held-out-score ⇒ flagged). Contamination resistance is a measured property (H68), not a
  promise.

---

## 5. Hypotheses (pre-registered, continuing the global H-ID space)

> **H-ID provenance.** The program's hypotheses run contiguous through **H54** (SPEC-15, CF). SPEC-18 is
> assigned the band **H65–H68** (SPEC-16/SPEC-17 reserve H55–H64). Each hypothesis below states the form
> that would falsify it and banks the honest negative in advance (the program norm, SPEC.md §9–10). Only
> *measurable scientific claims* are hypotheses; the packaging is milestones (§8) and adoption is
> explicitly not a hypothesis (§9).

- **H65 — the faithfulness benchmark is discriminative (stable rankings).** Over the frozen battery, the
  reference proposers' ordering by `H_ε(ρ)` is **stable across seeds**: the rank-correlation of per-seed
  leaderboards (Kendall's τ between disjoint seed splits, with a bootstrap CI) is high (target: τ ≥ 0.8,
  CI excluding 0), and the top proposer's `H_ε` advantage over the floor exceeds the across-seed CI width
  at every world. *Refuted if* rankings are seed-unstable (τ not distinguishable from 0, or the
  proposer-gap is inside the seed-noise band) — in which case the benchmark lacks discriminative validity
  **as currently configured**, and the banked negative is a *fixable measurement finding*: increase seeds,
  tighten `ε`, or harden the battery (longer rollouts / adversarial driver) until the signal clears the
  noise, and report the configuration at which it does. Tested as **PB-bench**.

- **H66 — the sim-to-emulation transfer gap is measurable and quantified (THE bridge result).** A proposer
  fit against the **reference** oracle, re-scored against the SPEC-11 **system** oracle (real `/bin/sh`,
  real kernel) on the structure-building grammar, has a transfer gap `ΔH = H_ε^{ref} − H_ε^{sys}` that is
  a finite number with a bootstrap CI. *Refuted if* the gap is not estimable — i.e. the two oracles
  disagree so pervasively (or the system oracle is unavailable so often) that `ΔH` has no meaningful CI.
  **Crucially, a *large* `ΔH` does not refute H66 — it confirms it:** the hypothesis is that the gap is
  *measurable*, and the program's contribution is the number, whichever way it lands. SPEC-11 already
  found bit-exact agreement on the structure grammar (its H27, residual 0), so the *expected* `ΔH` on that
  grammar is near zero and the headline reads "transfer is essentially lossless on the validated grammar,
  measured — the first such number in the ACD field"; outside that grammar, `ΔH` is the quantified version
  of CyGIL's asserted "full transferability." Either way the field gets the measurement it lacks. Tested
  as **PB-transfer**.

- **H67 — oracle-in-the-loop correction shrinks the transfer gap (the loop *is* the bridge).** When the
  proposer is run against the system oracle *with* a non-zero consultation budget `ρ > 0` (the
  propose–verify–correct loop, not free-running), the residual transfer gap is smaller than at `ρ = 0`,
  and shrinks monotonically with `ρ` — i.e. spending a few real-OS consults buys back the faithfulness
  lost in transfer. *Refuted if* correction does not reduce `ΔH` (the gap is in a regime the loop cannot
  reach — e.g. the model's transfer errors are subtle in exactly the way SPEC-7's H17 found for the
  distributed `M_θ`, where only bit-exact verification is efficient) — a bankable negative that says the
  bridge is the *measurement*, not the *fix*, on this grammar, and points at where a richer correction
  operator is needed. Tested as **PB-transfer** (the `ρ`-sweep arm).

- **H68 — the frozen eval is contamination-resistant.** A proposer deliberately overfit to the *public*
  battery manifest (memorizing its labels) scores conspicuously worse on the **held-out shard** than an
  honestly-fit proposer of equal capacity — i.e. the public-minus-heldout score gap is a usable
  overfit detector (target: the gap for the memorizer exceeds the honest proposer's gap by a margin
  outside the seed CI). *Refuted if* memorization transfers to the held-out shard as well as honest
  fitting (the shards are not independent enough — the held-out shard leaks the public one's structure),
  which would itself be a finding about battery construction and license a stronger holdout (different
  worlds/seeds, not just different draws). Tested as **PB-pack** (the contamination-control arm).

A note on the asymmetry, in the program's style. **H66 is the rare hypothesis whose every outcome is a
win:** near-zero `ΔH` makes "the learned world model transfers losslessly to a real OS" a committed CSV
(the strongest possible form of the §4.1 bridge claim); a large `ΔH` *is* the quantified sim-to-emulation
gap the ACD field has never reported. The only adverse outcome is H66 *unmeasurable* — and SPEC-11's H27
result (the system oracle works, bit-exact on the structure grammar) already rules that out for the
headline grammar.

---

## 6. Experiments / build (prefix **PB** — product benchmark)

Each follows the house template: a `Config` dataclass with `from_json_file`, a CLI entry point, a JSONL
record stream, a `plot_*.py` emitting a committed `.png` + `.csv`, regenerable from `reproduce.sh`,
deterministic and seeded (SPEC-2 §12). All reuse the shipped eval/RL surfaces; SPEC-18 adds the freezing,
the metadata, the conformance harness, and the transfer driver.

- **PB-bench — the discriminative-validity leaderboard (H65).** Score the reference proposers (flat
  transformer, GNN+RSSM, trivial/frequency b3, oracle-ceiling b1) on the frozen battery across `N` seeds,
  per world; compute per-seed leaderboards, the Kendall-τ rank-stability across disjoint seed splits with
  a bootstrap CI, and the proposer-gap-vs-seed-noise comparison. Output: the committed leaderboard table +
  the rank-stability figure. `experiments/pb_bench.py`, `configs/pb_bench.json`,
  `figures/plot_pb_bench.py` → `pb_bench_leaderboard.{png,csv}`.

- **PB-transfer — the sim-to-emulation gap (H66, H67) — the bridge result.** For each reference proposer,
  score `H_ε(ρ)` against the reference oracle and against the SPEC-11 `SandboxOracle`, on the
  structure-building grammar (where SY1/H27 proved the system oracle exact), sweeping `ρ ∈ {0, …, ρ*}`.
  Report `ΔH(ρ) = H_ε^{ref}(ρ) − H_ε^{sys}(ρ)` with bootstrap CIs (the H66 number) and its trend in `ρ`
  (the H67 shrink-with-correction test). `skipif`-guarded and disclosed when no real shell is present (a
  skip is never counted as a transfer result — the SPEC-11 §2.5 rule). Output:
  `figures/pb_transfer_gap.{png,csv}` — *the figure that quantifies what the ACD field asserts.*
  `experiments/pb_transfer.py`, `configs/pb_transfer.json`.

- **PB-pack — packaging + contamination control (H68, and the PB milestones).** (a) Emit the **Croissant**
  metadata for the frozen battery and the **datasheet/model-card** for the benchmark (provenance,
  intended use, the SPEC.md §13 ethics statement, known limits). (b) Run the **conformance** suite: assert
  the RL env satisfies the Gymnasium reset/step contract *and* the `verifiers` `load_environment` contract
  (FS/host/network/distributed); assert the Inspect task builds and its scorer has unit tests (the
  `inspect_evals` standard). (c) The **contamination control**: fit a deliberate public-manifest memorizer
  and an honest proposer of equal capacity, score both on public vs held-out shards, report the
  overfit-gap (H68). Output: `figures/pb_pack_contamination.{png,csv}` + the committed metadata files +
  a green conformance table. `experiments/pb_pack.py`, `configs/pb_pack.json`.

**Build order (gated, SPEC.md §12 discipline):** PB-bench first (a leaderboard that does not discriminate
is not worth packaging — H65 gates the rest); then PB-transfer (the bridge result, the second scientific
claim); then PB-pack (metadata, conformance, contamination — the engineering hardening that turns two
validated claims into a versioned release). Each rung graduates on a committed figure or a banked negative
that licenses the next.

---

## 7. Versioning, metadata, and release (the product surface)

- **Semantic versioning of the benchmark.** `verisim-bench@MAJOR.MINOR.PATCH`: MAJOR = battery manifest
  hash change (a different benchmark — old scores are not comparable); MINOR = additive worlds/proposers
  (old scores still valid, new columns); PATCH = metadata/doc/figure regeneration only. The manifest hash
  is committed and printed on every leaderboard row, so a score is always attributable to a battery
  version. The ACD env versions independently (`verisim-acd@…`) as a `verifiers` wheel.
- **Croissant metadata** (MLCommons' ML-ready dataset format) for the frozen battery — the standardized,
  machine-readable descriptor BetterBench found almost no benchmark ships. Discoverability + portability
  for the Inspect/Hub ecosystems.
- **Datasheet + model-card for the benchmark** (the Gebru-et-al. datasheets / model-cards practice applied
  to a benchmark): provenance (oracle-generated), composition (worlds/drivers/seeds), intended use
  (defensive ACD + world-model faithfulness research), the SPEC.md §13 ethics/disclosure posture, and the
  honest limits (the validated grammar is v0's, not all of POSIX — SPEC-11 §9).
- **Distribution.** The Inspect task ships to `inspect_evals` (UK AISI submission standard); the RL envs
  ship to the Prime Intellect Environments Hub as `verifiers` wheels via their `load_environment`
  entrypoints (already present). Both are MIT, no telemetry, no commercial path — the program's posture.

---

## 8. Milestones (the packaging, graded as engineering — SPEC-11 §8 form)

These are *deliverables*, not hypotheses: they are done or not done, and they carry no falsification
branch. They are design-stage (◐) targets, gated behind the science (PB-bench's H65 first).

| ID | Deliverable | Done-when (◐ design-stage targets) |
|---|---|---|
| **PB0** | Frozen, content-addressed battery + manifest hash, committed, reproducible from `reproduce.sh` | manifest hash stable across two clean regenerations; held-out shard reserved |
| **PB1** | Leaderboard surface (rollout `H_ε(ρ)` + bits-to-correct, per world) over the reference proposers | committed `pb_bench_leaderboard.csv`; every row stamped with battery version |
| **PB2** | Gymnasium + `verifiers` conformance for all four worlds (FS/host/network/distributed) | conformance suite green; `isinstance`/contract assertions pass on each `load_environment` |
| **PB3** | Inspect task submitted to the `inspect_evals` standard (scorer unit tests, dynamic-resource test) | task builds; scorer unit-tested; `skipif`-disclosed when `[eval]` extra absent |
| **PB4** | Croissant metadata + datasheet + model-card for the benchmark | committed metadata files validate against the Croissant schema |
| **PB5** | SPEC-11 `SandboxOracle` wired as the transfer-evaluation oracle (PB-transfer substrate) | PB-transfer runs against a real shell on the macOS host; `skipif`-disclosed elsewhere |
| **PB6** | Semantic-versioned release of `verisim-bench` + `verisim-acd` (MIT wheels, README intended-use/ethics) | both wheels build; README states defensive scope + responsible disclosure (SPEC.md §13) |

Build order is strict and gated: **PB0 → PB1 → PB2/PB3 → PB4 → PB5 → PB6**, with PB-bench (H65) proving
discriminative validity before PB1's leaderboard is published, and PB-transfer (H66) landing before PB6's
release claims a bridge result.

---

## 9. Adoption is not a hypothesis (stated plainly)

The program is honest about what is and is not a falsifiable scientific claim. **Whether the community
adopts `verisim-bench` or `verisim-acd` — stars, downloads, citations, leaderboard submissions — is not a
hypothesis and is reported nowhere as a result.** Adoption depends on timing, marketing, luck, and the
field's attention, none of which the oracle can adjudicate; treating it as a measured outcome would be
exactly the category error the program's epistemic engine (SPEC.md §10.1) exists to avoid — a number that
is not a fact about the world. Adoption belongs in a **community section** of the README with honest
framing ("here is the artifact, here is why it is unique, here is how to submit; whether anyone does is
out of our hands"), never in §5's hypotheses or §8's milestones. The scientific claims (H65–H68) and the
deliverables (PB0–PB6) stand on their own whether or not a single external user ever appears: the
benchmark either discriminates or it does not; the transfer gap either has a CI or it does not.

---

## 10. Build, reproduce, CI

### 10.1 Module layout (additive only)

```
src/verisim/product/                  # NEW — the product layer (no new world, oracle, or model)
  battery.py          # the frozen (world, driver, seed) battery + content-addressed manifest hash
  leaderboard.py      # score reference proposers → leaderboard rows + Kendall-τ rank stability
  transfer.py         # reference-oracle vs SPEC-11 SandboxOracle scoring; ΔH(ρ) with CIs
  metadata.py         # Croissant emitter + datasheet/model-card generator
  conformance.py      # Gymnasium + verifiers load_environment + Inspect-task contract assertions
src/verisim/experiments/
  pb_bench.py  pb_transfer.py  pb_pack.py     # NEW — the PB experiments
figures/
  plot_pb_bench.py  plot_pb_transfer.py  plot_pb_pack.py   # NEW — committed-figure generators
configs/
  pb_bench.json  pb_transfer.json  pb_pack.json            # NEW — committed configs
```

The product layer consumes the shipped `eval`/`rl`/`hostrl`/`hosteval`/`distrl`/`disteval` surfaces and
the SPEC-11 `SandboxOracle` **unchanged** — nothing in the deterministic core, the loop, the models, the
oracles, or any existing experiment is edited. SPEC-18 is a layer *above* the benchmark surfaces, not a
change *to* them.

### 10.2 `reproduce.sh` (new PB block, in gate order)

```bash
echo "== PB-bench: discriminative-validity leaderboard (H65) — gates the product =="
python -m verisim.experiments.pb_bench --config configs/pb_bench.json --out runs/pb_bench/records.jsonl \
    --csv figures/pb_bench_leaderboard.csv --plot figures/pb_bench_leaderboard.png
echo "== PB-transfer: sim-to-emulation gap vs the SPEC-11 system oracle (H66, H67) — THE BRIDGE =="
python -m verisim.experiments.pb_transfer --config configs/pb_transfer.json --out runs/pb_transfer/records.jsonl \
    --csv figures/pb_transfer_gap.csv --plot figures/pb_transfer_gap.png
echo "== PB-pack: Croissant/datasheet/model-card + conformance + contamination control (H68) =="
python -m verisim.experiments.pb_pack --config configs/pb_pack.json --out runs/pb_pack/records.jsonl \
    --csv figures/pb_pack_contamination.csv --plot figures/pb_pack_contamination.png
```

The PB-bench and PB-pack blocks run on CPU (the leaderboard scores the trivial/symbolic proposers
torch-free; the learned proposers reuse the SPEC-10 checkpoints). PB-transfer's system-oracle arm
`skipif`-guards on `/bin/sh` availability with disclosure (the SPEC-11 §2.5 cross-POSIX rule); a skip is
never counted as a transfer number.

### 10.3 CI

The existing CI (`ubuntu-latest`) is the free Linux confirmation. PB conformance tests assert
*structural* invariants — the env satisfies the Gymnasium and `verifiers` contracts; the Inspect task
builds; the battery manifest hash is stable; the leaderboard is monotone in the obvious sanity ordering
(oracle-ceiling ≥ learned ≥ trivial floor) — not exact magnitudes, so the same tests pass on the macOS
primary host and on Linux CI (the macOS-first principle). PB-transfer's bit-exact-grammar transfer claim
is platform-independent (it inherits SPEC-11 §2.5); the committed numbers are the macOS host's,
platform-stamped. The Croissant metadata is validated against its schema in CI as a documentation gate.

---

## 11. Scope, non-goals, honest caveats

- **This packages and validates; it does not discover a new dynamics result.** The two scientific claims
  (H65 discriminative validity, H66 transfer gap) are *measurements about the existing asset*, not new
  world dynamics. Over-claiming a model result from SPEC-18 is the failure mode this section forbids.
- **The validated grammar is v0's, not all of POSIX.** PB-transfer's transfer claim is exact only on the
  structure-building grammar SPEC-11 (H27) validated; outside it, `ΔH` carries the SPEC-11 §4.1 named
  modeling boundaries (root protection, overwrite policy, permission enforcement, self-subtree). The
  datasheet states this limit; over-claiming "faithful to a real OS" beyond the validated grammar is
  barred.
- **Discriminative validity is configuration-dependent.** H65 is a claim about *this* battery at *this*
  `ε` and seed count. If it refutes, the negative is fixable (more seeds / tighter `ε` / harder battery)
  and the program reports the configuration at which the signal clears the noise — a measurement finding,
  not a dead end.
- **Transfer is measured on the network/host worlds the SandboxOracle reaches.** The distributed world's
  bit-exact global oracle is intractable (SPEC-7), so PB-transfer's headline is the FS/host/network
  worlds; a distributed transfer number would use the tiered oracle and is deferred (a named fork, not a
  claim).
- **Adoption is out of scope as a result (§9).** Reiterated because it is the most tempting category error
  for a "product" spec: the artifact's worth is its discriminative validity and its measured transfer gap,
  not its popularity.
- **The defensive scope is a hard constraint, not a disclaimer.** `verisim-acd` ships no offensive agent;
  exploit-encoding environments are reviewed before release (SPEC.md §13). The SPEC-11 hermeticity
  contract is the env's execution boundary.

---

## 12. Provenance & reading order

SPEC-18 is the program's §14 #3 artifact (the reusable benchmark + RL env) hardened into a versioned
product, and its §14 #1/#4 contributions (the measurement no one else can make; the bridge result)
re-validated behind a frozen, contamination-resistant battery. It invents no world, oracle, or model: it
consumes the `eval`/`rl`/`hostrl`/`hosteval`/`distrl`/`disteval` surfaces and the SPEC-11 `SandboxOracle`
that already ship, and adds the freezing, the metadata, the conformance harness, and the transfer
measurement. Its science is two falsifiable claims (does the benchmark discriminate; is the transfer
measured) and its packaging is a milestone table; adoption is deliberately neither.

Reading order for a newcomer: [SPEC.md §14](./SPEC.md) (the four concrete artifacts this spec ships and
measures) → [SPEC.md §4.1](./SPEC.md) (the ACD sim-to-emulation gap) → [SPEC-11](./SPEC-11.md) (the
system oracle, the real-OS anchor for the transfer claim) → this document (§2 the two artifacts, §3 the
scientific spine, §5 the hypotheses, §8 the milestones, §9 why adoption is not a hypothesis) →
`src/verisim/product/` and `src/verisim/experiments/pb_bench.py` (the concrete build, once shipped).
