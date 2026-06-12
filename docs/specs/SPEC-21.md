# SPEC-21 — Scaling the Boundary: The Faithfulness-for-Control Scale Law on a Verifiable Computer-Use Environment

**Scale-and-artifact specification: SPEC-20 drew a sharp, cross-world boundary — oracle-grounded
world-model faithfulness is load-bearing for control *exactly when* the task's optimal policy depends
on the dynamics the model gets *wrong* (content), not the dynamics it learns *faithfully* (structure)
— and showed the gap is bought back cheaply by oracle-in-the-loop at a sub-linear budget (the useful
knee, ρ≈0.2). But that boundary was measured on *one tiny model per world* (110k params). The
standing objection is the one every program of this kind faces: **does the boundary survive scale, or
is it a small-model vignette?** SPEC-21 answers it. The boundary is not a fixed line — it is a
*function of the model's faithful/drifting frontier*, which moves with capacity. So the scale-robust
result is not "the boundary holds" but **how the boundary *moves***: as the model scales, structural
dynamics it already learns stay drift-free, content dynamics fall to it in order of learnability, and
the load-bearing frontier recedes — toward, but (the program's thesis predicts) not *to*, zero. The
residue that does *not* recede is the irreducible, effectively-unlearnable content where verification
is a *permanent* primitive, at any scale. SPEC-21 measures that trajectory, on a **verifiable
computer-use environment** (the host world positioned and hardened as a sandboxed shell / filesystem /
process world an agent acts in, anchored to a real `/bin/sh` via the SPEC-11 system oracle), and ships
it as the two things a non-credentialed researcher most needs: a scale law a frontier lab would read,
and an adopted artifact (the computer-use faithfulness environment + the load-bearing-frontier
benchmark).**

> **▶ PROPOSED — SCALE / ARTIFACT SPEC — 2026-06-11.** A *cross-cutting* spec, sibling to
> [SPEC-10](./SPEC-10.md) (the faithful-horizon scaling law) and [SPEC-20](./SPEC-20.md) (the
> usefulness boundary it scales). It invents **no new world or oracle**: it runs on the
> [SPEC-6](./SPEC-6.md) host world (re-positioned as the verifiable computer-use environment), the
> shipped [`ReferenceHostOracle`](../../src/verisim/hostoracle/reference.py) and the SPEC-11
> [`SandboxOracle`](../../src/verisim/oracle/sandbox.py) reality anchor, and the SPEC-20 content-task
> machinery ([`acd/host_integrity.py`](../../src/verisim/acd/host_integrity.py),
> [`experiments/ua_host_grounded.py`](../../src/verisim/experiments/ua_host_grounded.py)). What it adds
> is **the capacity axis** (SPEC-10's `ModelScale` ladder, run *through* the SPEC-20 measurements) and
> the **GPU-readiness contract** (§5): every mechanism is implemented and proven on CPU at smoke scale
> first, so the only thing that changes on a rented GPU is the dial. The committed scale law comes from
> the GPU run; the CPU core is what makes that run a one-command config swap, not a rewrite.

Read [SPEC-20 §7](./SPEC-20.md) (the boundary law + the useful knee, the result this spec scales),
[SPEC-10](./SPEC-10.md) (the capacity ladder and the free-oracle scaling envelope, the axis this spec
reuses), [SPEC-11](./SPEC-11.md) (the `SandboxOracle` — the real-shell reality anchor for the
computer-use claim), and the host content-task apparatus already shipped
([`experiments/host_drift.py`](../../src/verisim/experiments/host_drift.py),
[`experiments/ua_host_integrity.py`](../../src/verisim/experiments/ua_host_integrity.py),
[`experiments/host_flagship.py`](../../src/verisim/experiments/host_flagship.py)). This document is
*whether the SPEC-20 boundary moves predictably with scale, whether an irreducible residue remains, and
how the whole thing is packaged as a verifiable computer-use environment others use.*

---

## 0. One-paragraph thesis

SPEC-20 established, exactly and twice, that world-model faithfulness is load-bearing for control iff
the task keys on the dynamics the model drifts on — *content* (file writes, network flows), not
*structure* (process trees, reachability) — and that the oracle-in-the-loop buys the content gap back
at a sub-linear budget. The thesis of SPEC-21: **that boundary is a moving function of model scale, and
its trajectory is the scale-robust result.** Per-dimension model accuracy rises with capacity
(SPEC-10), so a dimension that is "content the model drifts on" at small scale (host `write` 0.36)
becomes "structure the model learns faithfully" at larger scale — and the set of tasks for which
faithfulness is load-bearing *shrinks*, predictably, structural-first. Stated to be falsified: *the
load-bearing frontier recedes monotonically and predictably with capacity (H87), the per-dimension
drift profile forecasts which tasks remain load-bearing at each scale (H89), and the frontier recedes
toward but not to zero — an irreducible high-entropy content residue stays load-bearing at every
reachable scale (H88), so verification is a primitive you cannot scale away, exactly where it matters.*
The opposite branches are each first-class: a frontier that does *not* move with scale would say the
boundary is a fixed property of the world (a different, also-publishable result); a frontier that
recedes *to* zero would say faithfulness is a small-model crutch the field can scale past (the deepest
negative, and the one that would most change the program's claims). The measurement runs on a verifiable
computer-use environment so the answer is about the domain a frontier lab cares about, and ships as a
benchmark so the answer is something others can extend.

---

## 1. Why scaling the boundary is the experiment that makes the contribution legible

The SPEC-20 boundary is a genuinely sharp, cross-world, dual-direction result — six structural-control
nulls and two content-control positives, each measured exactly against ground truth, with the gap
bought back cheaply. Its one weakness is the one every result of its kind has: it lives on 110k-param
models in small worlds, so a frontier reader's first question — *does it survive scale?* — has no
answer. A boundary measured only on miniatures is a vignette; a boundary whose *motion with scale is
itself measured and predicted* is a law. SPEC-21 converts the former into the latter by sweeping the one
axis SPEC-10 already built (capacity) *through* the SPEC-20 measurements, and asking not "does the line
hold" but "where does the line go." That reframing is what makes the result robust to the scale
objection instead of vulnerable to it: the prediction (the frontier recedes structural-first, leaving an
irreducible residue) is *more* interesting if it holds at scale and *still interesting* (a clean
negative) if it doesn't, because the oracle makes both branches exact. And it is the experiment that
connects the program's two halves — SPEC-10's "can you scale away the compounding wall?" and SPEC-20's
"when does faithfulness matter for control?" — into one statement: *scale closes the structural gap and
exposes the irreducible-content gap, which is exactly where the oracle-in-the-loop is permanently
load-bearing.*

## 2. The verifiable computer-use environment (the target and the artifact)

The host world (SPEC-6) — processes, file descriptors, a filesystem, commands, with a deterministic
reference oracle and a real-`/bin/sh` system oracle (SPEC-11) — *is* a computer-use environment: the
CLI/shell/filesystem/process slice of computer use, which is precisely the slice that admits a
ground-truth oracle (unlike GUI/browser computer use, which has none — the whole §2 asymmetry). SPEC-21
positions and hardens it as **`verisim-cue`**, a verifiable computer-use environment with three
properties no other computer-use environment has at once:

- **Ground-truth next-state**, for free, at every step (the reference oracle), *and* a real-kernel
  anchor (the SPEC-11 `SandboxOracle`) so the claim is about reality, not a model of it (SPEC.md §2.1).
- **A spectrum of computer-use tasks** spanning structure→content (§3), each scored by an exact
  faithful-vs-free predictor (the SPEC-20 apparatus), so the environment measures not just *whether* an
  agent succeeds but *whether faithfulness was load-bearing* for its success.
- **A scale axis** built in: the same environment, the same tasks, swept across model capacity, so the
  environment is the substrate of a *scale law*, not a single number.

This is the artifact half of the spec. The honest scope: it is *shell/file/process* computer use, not
GUI — stated plainly, and that is the point (it is the oracle-grounded slice). It connects directly to
the SPEC-18 `verisim-bench` line as the computer-use vertical of the faithfulness benchmark, and to the
SPEC.md §4 metrology claim (the place where computer-use world models can be measured against a real
ruler).

## 3. The task suite spanning structure → content (the measurement substrate)

To measure *where* the load-bearing frontier sits at each scale, we need a battery of computer-use tasks
ordered along the structure→content spectrum — the dimension whose model-accuracy the capacity axis
moves. Each task is a SPEC-20-style predictive-defense task: a faithful predictor (oracle rollout) vs a
free predictor (`M_θ` rollout), scored on an exact metric, with the *gap* the load-bearing signal and
the ρ-grounded knee (UA9/H81) the cheap-purchase signal. The suite (reusing the shipped task machinery,
adding only the ordering and the missing rungs):

| task | keyed dimension | structure↔content | shipped as |
|---|---|---|---|
| process containment | process tree (`fork`/`kill`) | structure (host model 0% drift) | SPEC-20 UA0/UA7 (host analogue) |
| reachability / fd control | fds, links | near-structure (~4–26% drift) | new rung (CP2) |
| **file integrity** | `write` / fs (which files written) | content (~25–36% drift) | SPEC-20 UA8/UA9 ([`host_integrity`](../../src/verisim/acd/host_integrity.py)) |
| data-dependency / content-value | file *content tokens* | deep content (highest-entropy) | new rung (CP3) — the irreducible-residue probe |

The two new rungs (CP2, CP3) extend the existing predictive-defense pattern to a structural lever
(fd/reachability) and to the deepest content (the actual written *content*, not just *which* file) — the
latter is the candidate for the irreducible residue (H88): content that depends on input tokens the
model has no way to learn, so its accuracy should plateau below threshold at *every* scale. The suite is
deliberately small and ordered; the science is in sweeping each task across capacity, not in the count.

## 4. The load-bearing frontier and the scale ladder (the headline measurement)

The headline object is the **load-bearing frontier**: at each model scale `S`, the boundary in
task-space separating tasks where the faithful predictor still materially beats the free one (oracle
load-bearing) from tasks where the free predictor has caught up (oracle not needed). Concretely, for the
ordered suite (§3) and the capacity ladder, the measurement at each scale is:

1. **Train + freeze** a host `M_θ` at scale `S` (the HFL0 lifecycle,
   [`host_flagship.py`](../../src/verisim/experiments/host_flagship.py), across the ladder).
2. **The drift profile** at `S` ([`host_drift.py`](../../src/verisim/experiments/host_drift.py)): the
   per-dimension free-running accuracy — the cheap predictor of the frontier (H89).
3. **The per-task gap** at `S`: the faithful-vs-free predictive-defense gap for each task in the suite
   (the SPEC-20 UA8/UA10 apparatus), plus the ρ-grounded knee for each load-bearing task.

The headline figure, [`figures/cs1_loadbearing_frontier.png`](../../figures/cs1_loadbearing_frontier.png):
the per-task gap vs model scale, one line per task, ordered structure→content — the frontier is the
contour where the gap crosses the load-bearing threshold, and the claim (H87) is that it **recedes
structural-first** as scale grows, with the deep-content task (CP3) staying above threshold at every
rung (H88). A second panel overlays the drift profile (the per-dimension accuracy vs scale) to show the
frontier tracks it (H89).

**The scale ladder** is SPEC-10's, extended at the top: `xs`(1k) → `s` → `m` → `l`(110k) → `xl`(410k) →
`xxl`(1.6M) → `xxxl`(GPU-only, ≥10M). The CPU core proves the apparatus on `xs…l` (minutes–hours, the
shipped SPEC-10 envelope); the GPU run extends to `xl…xxxl`, where the frontier's *motion* becomes
measurable across a wide enough capacity range to fit a trajectory rather than two points.

## 5. The CPU-proven / GPU-ready contract (the engineering discipline this spec is built on)

This is the section the spec exists to enforce: **everything runs on CPU at smoke scale, identically,
before any GPU is rented, so the GPU run is a config swap, not a rewrite.** The contract, concretely:

- **One pipeline, one dial.** A single `run_scale_law(ladder_config)` drives the whole measurement
  (train → drift profile → per-task gap → knee → frontier) over a list of `ModelScale` rungs. The
  *only* difference between the CI smoke run and the GPU run is the `ladder_config`: the smoke config
  lists `xs,s`; the committed config lists `xs…xxxl`. No code path differs.
- **Device-agnostic, deterministic.** Training and rollout are seeded and run bit-deterministically at
  `num_threads=1` (the repo default); the GPU rung sets `device=cuda` and a larger batch, and the
  apparatus (the metrics, the frontier, the verdicts) is identical and device-independent. Every
  committed number is reproducible from `(seed, config)`.
- **CPU proves correctness; GPU provides range.** The CPU core's job is *not* to produce the headline —
  it cannot, the capacity range is too narrow — its job is to prove every component is correct,
  deterministic, and composes, so the GPU run's only new variable is scale. CI runs the smoke ladder and
  asserts the apparatus (frontier well-formed, monotone where it must be, verdicts computed); the
  committed scale law is the GPU run of the *same* pipeline.
- **One command, costed.** The GPU entry point is a single command
  (`python -m verisim.experiments.scale_law --config configs/scale_law_gpu.json --device cuda`), with a
  pre-registered cost estimate (rungs × train-steps × params → GPU-hours) in the config header, so the
  rent decision is informed before the card is booked. A single mid-range GPU (one 24–48 GB card) is
  the target; the spec does *not* assume a cluster (SPEC-9's single-machine-envelope discipline,
  extended one tier).
- **Checkpoints are artifacts, not state.** Each rung's frozen checkpoint (the HFL0 manifest + weights)
  is gitignored and regenerable; the committed outputs are the CSVs, the frontier figure, and the
  benchmark metadata — so a reader checks the law against the figure without the GPU.

The CPU core is *done* when: the full pipeline runs green on the smoke ladder in CI, every task/knee/
drift component has a deterministic test, and a dry-run of the GPU config validates (shapes, device
switch, cost estimate) without training — i.e., the rented GPU runs a *proven* program.

## 6. Hypotheses (H87–H90)

Pre-registered with both branches, per SPEC.md §10.1.

- **H87 (the load-bearing frontier recedes with scale).** As model capacity grows across the ladder,
  the per-task faithful-vs-free gap shrinks in order of the task's keyed-dimension learnability —
  structural tasks lose their gap first (already near-zero at `l`), content tasks later — so the
  load-bearing frontier recedes monotonically and predictably toward the deepest content. *Refuted if*
  the frontier does not move with scale (the boundary is a fixed world-property, independent of the
  model — a different, also-publishable result) or moves non-monotonically/unpredictably. Tested as
  **CS1** (the scale-law sweep).
- **H88 (the irreducible residue — verification as a permanent primitive).** The frontier recedes
  toward but *not to* zero: the deepest-content task (CP3, keyed on file *content* that depends on
  effectively-unlearnable input) keeps a material faithful-vs-free gap at *every* reachable scale,
  because its keyed-dimension accuracy plateaus below threshold — so the oracle-in-the-loop is
  *permanently* load-bearing there, and verification is a primitive scale cannot remove. *Refuted if*
  every dimension's accuracy crosses threshold with scale (the residue vanishes; faithfulness is a
  small-model crutch the field can scale past) — the deepest negative, and itself a headline result
  that would reshape the program's central claim. Tested as **CS1** (the deep-content rung across the
  ladder).
- **H89 (the drift profile forecasts the frontier — cheap predicts expensive).** At each scale `S`, the
  per-task load-bearing verdict is predicted by the cheap per-dimension drift profile at `S` (a task is
  load-bearing iff its keyed dimension's accuracy is below the threshold the gap crosses) — so the
  expensive per-task ablation is forecastable from the cheap profile, and the frontier can be *predicted*
  for a new task without running it. *Refuted if* the per-task gaps do not track the per-dimension
  accuracies (the frontier has structure the profile misses). Tested as **CS2** (the forecast check).
- **H90 (the scale law survives the system oracle — real computer-use).** The frontier and its motion,
  measured against the reference oracle, hold when the SPEC-11 `SandboxOracle` (real `/bin/sh`, real
  kernel) replaces it as the reality anchor — so the scale law is about real computer-use dynamics, not
  a model of them. *Refuted if* real-kernel nondeterminism or semantics move the frontier qualitatively.
  Tested as **CS3** (the system-oracle anchor; gated on SPEC-11 maturity, may defer with the top GPU
  rungs and that deferral is stated, not hidden).

## 7. Milestones (CP0–CP5 CPU core, then the single GPU run)

The CPU core (CP0–CP4) is fully buildable now and is the deliverable that makes the GPU run a config
swap. Each ships a deterministic test and a smoke-ladder result.

- **CP0 — the scale-law harness.** `run_scale_law(ladder)` composing the shipped pieces
  (HFL0 train/freeze per rung → `host_drift` → per-task predictive-defense gap → ρ-knee) into one
  device-agnostic, seeded pipeline; the smoke ladder (`xs,s`) green in CI.
- **CP1 — the task suite, ordered.** Wrap the existing process-containment + file-integrity tasks and
  add the structure→content ordering + the suite runner (the per-task gap at one scale).
- **CP2 — the structural rung (fd/reachability control).** The missing near-structure task, to populate
  the middle of the spectrum.
- **CP3 — the deep-content rung (content-value).** The file-*content* predictive task — the
  irreducible-residue probe (H88). Includes the per-dimension content-accuracy diagnostic.
- **CP4 — the frontier + the forecast.** The load-bearing-frontier reducer + figure (CS1) and the
  drift-profile-forecasts-frontier check (CS2/H89), on the smoke ladder.
- **CP5 — the GPU-readiness gate.** The `scale_law_gpu.json` config (full ladder + `device=cuda` +
  cost estimate), a `--dry-run` that validates shapes/device/cost without training, and the one-command
  entry point. *When this is green, the rented GPU runs a proven program.*
- **GPU — the committed scale law.** One run of the same pipeline on the full ladder
  (`xs…xxxl`) on a rented GPU → the CS1 frontier figure, the CS2 forecast, the CS3 system-oracle anchor.
  This is the only step that needs the card; everything above proves it will work.

## 8. The three deliverables

1. **The scale law (the result a lab reads).** The load-bearing-frontier figure: the faithfulness-for-
   control boundary as a *function of scale*, with the structural-first recession (H87), the irreducible
   residue (H88), and the cheap-forecast (H89) — the scale-robust form of the SPEC-20 boundary, on a
   computer-use environment, against ground truth.
2. **The artifact (the thing others use).** `verisim-cue`: the verifiable computer-use environment +
   the load-bearing-frontier benchmark — the one computer-use world-model benchmark with ground-truth
   labels *and* a faithfulness-load-bearing verdict per task, packaged on the SPEC-18 `verisim-bench`
   line (Croissant + datasheet + model-card, a frozen eval battery). Adoption is *not* a hypothesis
   (SPEC-18 §9); the artifact is shipped whether or not it is adopted.
3. **The writeup (the thing that circulates).** A single essay — the verifier-as-primitive axis, the
   structure/content boundary, the bankable-negative methodology, and now the *scale law* — drafted via
   the `essay` skill, the form that reaches the right readers for a non-credentialed researcher.

## 9. Gate and what each branch licenses

**Gate: H87** (the frontier moves with scale, measurably).

- **H87 confirmed + H88 confirmed** (recedes structural-first, irreducible residue remains) → the
  program's strongest, most lab-legible result: *scale closes the structural gap and exposes the
  irreducible-content gap, where verification is permanently load-bearing* — the scale-robust form of
  the whole thesis. Licenses the writeup and the artifact release.
- **H87 confirmed + H88 refuted** (recedes *to* zero) → the deepest negative: faithfulness is a
  small-model crutch the field can scale past. This *changes the program's central claim* and is itself
  a major, honest, publishable result — and the oracle is what makes it trustworthy (SPEC.md §10.1).
- **H87 refuted** (the frontier is scale-invariant) → the boundary is a fixed property of the world, not
  the model — a different result that re-frames SPEC-20 as world-metrology rather than scale-dependent,
  and is reported as such.

In every branch the artifact and the writeup ship; only their headline sentence changes.

## 10. Honest caveats, stated up front

- **Computer use here is shell/file/process, not GUI.** That is the oracle-grounded slice and the point
  (§2); GUI computer use has no oracle and is out of scope (SPEC.md §11).
- **The ladder is single-GPU, not frontier-scale.** `xxxl` (≥10M params) is far below a real LLM; the
  claim is about the *trajectory* of the frontier across the reachable range, not about LLM-scale
  models. The honest statement is "the frontier recedes predictably across the measured range," with the
  extrapolation flagged, not asserted.
- **The irreducible residue (H88) is a hypothesis, not a guarantee.** It may be that the deep-content
  task's keyed dimension *also* becomes learnable at a scale beyond the ladder; the spec measures the
  plateau across the reachable range and states the extrapolation risk.
- **The system-oracle anchor (H90/CS3) is gated on SPEC-11 maturity** and may defer with the top GPU
  rungs; that deferral is stated, not hidden (SPEC.md §13).

## 11. Status

| ID | Hypothesis / artifact | State | Result |
|---|---|---|---|
| CP1 | the ordered task suite | ✅ shipped (CPU core) | the `verisim-cue` task suite ([`cue/tasks.py`](../../src/verisim/cue/tasks.py)): four computer-use predictive-defense tasks ordered structure→content (`process-control` → `fd-control` → `file-integrity` → `content-value`), each a SPEC-20 faithful-vs-free gap over a generic **keyed-set extractor** (procs / fds / written-files / (path,content)), reusing the shipped UA8 workload+predictor machinery. The structure→content gap gradient appears already at smoke scale (process **+0.00** / fd **+0.06–0.12** / file **+0.31–0.38** / content **+0.50**). |
| CP2 | the structural (fd/reachability) rung | ✅ shipped (CPU core) | the `fd-control` task (keyed on the open-fd table) populates the near-structure middle of the spectrum — a structural lever that drifts moderately (gap ~0.06–0.19 at smoke scale). |
| CP3 | the deep-content rung (the residue probe) | ✅ shipped (CPU core) | the `content-value` task (keyed on the actual *(path, content)* pairs, not just which file) is the highest-entropy rung and the irreducible-residue probe (H88); the per-task `keyed_drift` is the CP3 content-accuracy diagnostic (a single free-run measurement). |
| CP0 | the scale-law harness (one pipeline, one dial) | ✅ shipped (CPU core) | `run_scale_law(config)` ([`experiments/scale_law.py`](../../src/verisim/experiments/scale_law.py)): per rung, train+freeze a host `M_θ` (the HFL0 lifecycle, scale swapped via `dataclasses.replace`) → the drift profile → the per-task gap + cheap keyed drift + the ρ-grounded knee for each load-bearing task. Device-agnostic, seeded; the smoke ladder runs green in CI. |
| CP4 | the load-bearing frontier + the forecast (CS1/CS2) | ✅ shipped (CPU core) | the reducers: `load_bearing_frontier` (the contour where the gap crosses threshold), `frontier_verdict` (H87 recession + H88 residue), `forecast_check` (H89). **Committed 4-rung CPU run** ([`cs1_loadbearing_frontier.csv`](../../figures/cs1_loadbearing_frontier.csv), [`.png`](../../figures/cs1_loadbearing_frontier.png)): the structure→content gap gradient holds at every rung (process ≤0.16 → fd 0.13–0.25 → file 0.56–0.88 → content **0.81–0.94**); process-control falls below the load-bearing threshold after `xs` (the structural-first recession *beginning*), content-value stays load-bearing at every rung (H88 directionally confirmed), and the cheap keyed drift forecasts the gap at **Spearman +0.965** (H89). The CPU range is too narrow to fit the full recession — that is the GPU run's job, stated plainly. |
| CP5 | the GPU-readiness gate (config + dry-run + cost) | ✅ shipped (CPU core) | [`configs/scale_law_gpu.json`](../../configs/scale_law_gpu.json) (full ladder `xs…xxxl` + `device=cuda`) + `--dry-run` (validates shapes/device/cost-estimate **without training**) + the one-command entry point `python -m verisim.experiments.scale_law --config … --device cuda`. The dry-run on the committed config is green — a rented GPU runs a proven program. |
| CS1 | H87/H88 — the frontier recedes; the residue remains | ▶ proposed (GPU run) | the committed scale law (apparatus CPU-proven; the headline needs the wide GPU ladder) |
| CS2 | H89 — the drift profile forecasts the frontier | ◐ apparatus shipped + smoke-confirmed | `forecast_check` Spearman +0.90 at smoke scale; the committed number is the GPU run |
| CS3 | H90 — survives the system oracle (real computer-use) | ▶ proposed (system-oracle-gated) | — |
| — | `verisim-cue` artifact (packaging + eval surface) | ✅ shipped (CPU) | the artifact half hardened the SPEC-18 way ([`cue/pack.py`](../../src/verisim/cue/pack.py), [`cue/scorecard.py`](../../src/verisim/cue/scorecard.py), [`cue/conformance.py`](../../src/verisim/cue/conformance.py), [`experiments/cue_pack.py`](../../src/verisim/experiments/cue_pack.py)): a frozen, hashed, versioned `CueManifest` (`verisim-cue@0.1.0+<hash>`) over the ordered task suite, emitting the full metadata triple §8.2 names — **Croissant** + **datasheet** + **model-card** — plus a **task-card** carrying the **per-task load-bearing verdict** (process-control *not* load-bearing +0.03 / fd +0.16 / file +0.56 / content +0.84 at the top CPU rung). The **eval surface** is `score_model(model)` — run any host world-model through the suite and get its per-task scorecard (catch rate + *whether the oracle was load-bearing for that model*); the model-card shows the reference scorecards (one per scale-law rung). Conformance (**ground-truth labels exact** — the faithful predictor scores 1.000 on every task; ordered spectrum; recognized dimensions) all green. Committed under [`cue/`](../../cue/). Adoption is not a hypothesis (SPEC-18 §9); the artifact ships regardless. |
| — | the SPEC-21 essay | ▶ proposed | — |

The discipline of §5 is the load-bearing commitment of this spec, and it is now **met**: the CPU core
(CP0–CP5) is shipped — the full pipeline runs green on the smoke ladder in CI, every task/knee/drift
component has a deterministic test, and the GPU config's `--dry-run` validates without training. The
headline scale law (CS1/CS2/CS3) is now one command — `python -m verisim.experiments.scale_law --config
configs/scale_law_gpu.json --device cuda` — away.
