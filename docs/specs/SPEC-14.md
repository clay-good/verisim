# SPEC-14 — Neural Algorithmic Reasoning: Breaking the Structured-Arm Compounding Wall

**Method specification: SPEC-10's HS3 found the program's one genuine wall — the structured GNN+RSSM
arm's free-running faithful horizon is pinned at *zero* at exact tolerance across capacity, data, and
world size, with η < 1 (it free-runs *shorter* than its own i.i.d. prediction). Yet the same arm beats
the flat transformer ~6.6× on one-step delta-exact (EN4/H11). Great one-step, zero horizon: a
proxy/truth split. This spec attacks the wall with the literature built for exactly this failure —
Neural Algorithmic Reasoning (NAR) — and the one asset that literature never had: the oracle emits the
algorithm's intermediate computation states for free.**

> **▶ METHOD SPEC — 2026-06 — NA0 + NA5 SHIPPED (✅ the diagnosis gate + the decode-side
> confirmation; H45 REFUTED → the wall is the decoder/rollout, not the processor — confirmed at the
> rollout level; NA1–NA4 re-scoped, see §11).** A *method* spec in the
> lineage of [SPEC-8](./SPEC-8.md)
> (oracle-grounded self-supervision) and [SPEC-12](./SPEC-12.md) (planning over the same model). It
> invents **no new world** (the SPEC-5 network world), **no new oracle** (the data-plane
> `ReferenceNetworkOracle` + the control-plane `ControlPlaneOracle`), and **no new arm** — it changes
> *how the shipped graph arm is trained and run*. SPEC-12's move was to route *around* the wall (plan
> landmark-to-landmark, never roll forward). SPEC-14's move is to attack the wall *head-on*: the
> network oracle's reachability/FIB computation **is** a deterministic graph algorithm (BFS /
> Bellman–Ford-style propagation), the graph arm's `mp_rounds` message passing **is** the natural
> processor for it, and NAR's central finding — that step-wise **hint** supervision on the algorithm's
> intermediate states is what makes a GNN execute multi-step and generalize OOD — maps onto an asset no
> NAR paper has: those hints are **free and exact** here, because the oracle is the algorithm.

Read [SPEC.md §0/§5/§6](./SPEC.md) (the oracle, `H_ε(ρ)`, the two arms),
[SPEC-10 §4.6–4.11](./SPEC-10.md) (HS3 — the wall this spec attacks, and HS3-T's trainer caveat),
[SPEC-5 §6.1–6.2](./SPEC-5.md) (the graph arm and the two-plane oracle), and the shipped graph arm
[`graph_model.py`](../../src/verisim/netmodel/graph_model.py) (`GraphRSSMNet`: `_message_pass` over
`mp_rounds`, `encode`/`embed`/`decode_logits`, the per-round `mp_link`/`mp_flow_fwd`/`mp_flow_rev`/
`mp_update`/`mp_norm` stack) and [`graph.py`](../../src/verisim/netmodel/graph.py)
(`build_graph`, `link_edges`/`flow_edges`, `feature_dims`).

---

## 0. One-paragraph thesis

The structured arm's failure is *precisely* the failure NAR was invented to fix. NAR (Veličković et
al.) showed that a GNN trained end-to-end on input→output pairs of a graph algorithm does *not*
generalize multi-step and does *not* extrapolate out of distribution, but the *same* GNN trained to
**imitate every intermediate step** of the algorithm — with step-wise "hint" supervision — does. The
verisim graph arm is, today, an end-to-end model: `train_graph_model` supervises only the next-step
delta, and HS3 reports the textbook end-to-end-NAR symptom — high one-step accuracy that **does not
compose** (η < 1, `H_free` = 0). The network oracle's hidden machinery is a reachability /
forwarding-table computation that is itself a multi-round graph propagation (which hosts can reach
which, given links + firewall denies + flows), and the graph arm's `mp_rounds` message passing
(`_message_pass`, depth ≈ network diameter) is the processor that *should* execute it. The thesis,
stated to be falsified: **supervising the graph arm on the oracle's free intermediate
reachability-propagation states — plus aligning `mp_rounds`/the update to the actual algorithm and
letting the processor "think longer" at inference — lifts the structured arm's exact free-running
horizon off zero, transferring its already-superior one-step skill (EN4) into *horizon* and closing the
proxy/truth split.** The crucial honest negative is pre-registered and *more* valuable than the
positive: **if free hints + alignment + iterate-to-convergence still leave `H_free` = 0, the wall is
genuinely *compounding* (error accumulation in the autoregressive delta rollout), not
*under-alignment* of the representation — a deep, bankable result that sharpens HS3 from "this trainer
plateaus" to "the structured ceiling is fundamental to rolling a delta-predictor forward."** And
because HS3's own caveat is that the committed trainer plateaus at `p` ≈ 0.66 < 0.82 (HS3-T showed a
schedule fix did not help), every experiment here is built to **separate trainer/representation from
compounding** — the confound the whole HS3 line flags but cannot resolve from inside the end-to-end
recipe.

---

## 1. Why now: the proxy/truth split is the canonical NAR failure mode

SPEC-10 produced two verdicts that, read together, name this spec. (i) The flat transformer's
floor+cliff **dissolved into a resourcing story** (HS1→HS1.3): capacity × data × training scaled
together buy ~19 id / ~29 ood free-running steps; its rollouts *self-stabilize* (η > 1). (ii) The
structured GNN+RSSM arm's floor is a **genuine wall** (HS3 §4.6–4.8): `H_free` ≈ 0 at exact tolerance
across a 108× capacity range, a 10× data range, and an 8× world-size range, with **η < 1** — it
free-runs *shorter* than `p/(1-p)`, the compounding penalty the flat arm never paid. And the killer
contrast (EN4/H11): the graph arm beats the flat arm **~6.6×** on one-step delta-exact. So the
structured arm is the *better one-step predictor* and the *worse free-runner*. That is not a paradox in
the NAR literature; it is the headline finding.

- **End-to-end GNNs do not generalize multi-step; hint supervision is the fix.** *Neural Execution of
  Graph Algorithms* (Veličković et al., ICLR 2020) showed a GNN trained only on an algorithm's
  input→output fails to execute it, while supervising each intermediate step ("learn the trajectory,
  not the answer") yields multi-step execution and size extrapolation. *The CLRS-30 benchmark*
  (Veličković et al., 2022) made step-wise hint supervision the standard, exposing the full trajectory
  of "hint vectors" (intermediate algorithmic states) as supervision precisely to *prevent the model
  from latching onto non-generalizable shortcuts of one train set* — which is exactly the HS3 reading:
  the graph arm has learned a one-step shortcut (great `p`) that does not compose (`H_free` = 0).
- **Algorithmic alignment predicts which architectures *can* multi-step.** *What Can Neural Networks
  Reason About?* (Xu et al., ICLR 2020) formalizes "algorithmic alignment": a network learns a
  reasoning task with low sample complexity when its computation structure mirrors the task's
  algorithm, and GNNs align with dynamic-programming / shortest-path style computation. The oracle's
  reachability is exactly such a computation; the graph arm's per-round update is the alignment knob.
- **Verisim has the one asset NAR assumes is expensive: free, exact hints.** Every NAR paper either
  pays to generate intermediate-state labels or, recently, tries to *avoid* them (*NAR Without
  Intermediate Supervision*, Rodionov & Prokhorenkova, 2023, builds a self-supervised proxy precisely
  *because* hints are costly and the algorithm-trajectory is usually unavailable at scale). In verisim
  the oracle **is** the algorithm: it can emit, for free and bit-exact, the intermediate
  reachability-propagation state at each message-passing round. This inverts the field's hardest
  constraint into a free supervisory signal — the same inversion SPEC-8 made for representation
  learning.

This is the precondition under which NAR is the right tool, not landmark planning (SPEC-12's route
*around* the wall): the wall is "a structured proposer that predicts one step well but does not execute
the multi-step computation," and NAR is the literature on making a structured proposer execute the
multi-step computation.

---

## 2. The lineage folded in (and the design choice each one forces)

### 2.1 Hint / step-wise supervision (the load-bearing idea)

- **Neural Execution of Graph Algorithms** (Veličković et al., ICLR 2020, arXiv:1910.10593).
  Encode-process-decode; supervise every step of BFS/Bellman–Ford via intermediate "hint" tensors;
  multi-algorithm training; size extrapolation. → **Design choice:** add an auxiliary head on the
  graph arm's per-round node embeddings (`_message_pass`'s `h` after each round `r`) that predicts the
  oracle's reachability-propagation state at round `r`, teacher-forced from the oracle (NA1).
- **The CLRS-30 Algorithmic Reasoning Benchmark** (Veličković et al., 2022, arXiv:2205.15659). Full
  hint trajectories; strict step-wise evaluation; the explicit anti-shortcut rationale. → **Design
  choice:** the supervision target is a *trajectory* (one label per `mp_round`), and the eval reports
  per-round hint accuracy alongside `p`/`H_free`, so the spec can say *where* the propagation breaks.
- **A Generalist Neural Algorithmic Learner / Triplet-GMPNN** (Ibarz et al., LoG 2022,
  arXiv:2209.11142). Architectural choices (gating, triplet messages) that make hint-supervised
  processors OOD-robust. → **Design choice (NA2's alignment knob):** the graph arm's
  `mp_update` (a `Linear(4d→d)` over `[h, m_link, m_fwd, m_rev]`) is the natural place to test a
  *propagation-aligned* update (e.g. max/min aggregation matching Bellman–Ford relaxation rather than
  the current mean-aggregation `_row_normalize`).
- **NAR Without Intermediate Supervision** (Rodionov & Prokhorenkova, NeurIPS 2023, arXiv:2306.13411).
  Hints are costly and usually unavailable, so they regularize intermediate computation *without* the
  trajectory. → **Design choice (the contrast that makes verisim's asset legible):** this paper exists
  because hints are expensive; verisim's hints are *free and exact*, so NA1 is the experiment that
  field could not cheaply run. We pre-register it as the baseline-beater and bank the comparison.

### 2.2 Iterative / "think-longer" processors (test-time depth)

- **Deep Equilibrium Models** (Bai, Kolter & Koltun, NeurIPS 2019, arXiv:1909.01377) and **The Deep
  Equilibrium Algorithmic Reasoner** (Veličković et al. line, 2024). A propagation whose solution is a
  *fixed point* can be solved to convergence rather than at a fixed unrolled depth — and the
  reachability transitive closure *is* a fixed point. → **Design choice (NA3):** at inference, iterate
  `_message_pass`'s update to convergence (or for `k > mp_rounds` rounds) instead of the fixed
  `mp_rounds = 3`, so the processor "thinks longer" on harder states — the ACT/DEQ recipe applied to
  the shipped arm.
- **Adaptive Computation Time / logical extrapolation** (Graves 2016; Schwarzschild et al. on
  recurrent/implicit nets extrapolating to harder instances by iterating more at test time). →
  **Design choice:** NA3 sweeps inference rounds as a free knob (weights frozen) and measures whether
  *more rounds at the same weights* lifts `H_free` — the cheapest possible intervention, run first
  after NA1.

### 2.3 What the program already built that this sits on

- **The arm is already an encode-process-decode processor.** `GraphRSSMNet._message_pass` is the
  processor (link + forward-flow + reverse-flow messages, `mp_update`, `mp_norm`, `mp_rounds` deep);
  `embed` exposes the bare summary; `encode` adds the RSSM belief; `decode_logits` is the grammared
  decoder. NA1's hint head attaches to the per-round `h`; NA2 swaps the aggregation/update; NA3
  changes only the inference loop count. **No new module is required for NA1–NA3** beyond an auxiliary
  head and a training-loss term.
- **The oracle is the hint source.** The data-plane `ReferenceNetworkOracle` computes exact
  reachability; the `ControlPlaneOracle` computes it symbolically and cheaply. Either can emit the
  per-round propagation frontier (round-`r` = hosts reachable in ≤ `r` hops under current
  links/denies/flows) as the free hint trajectory.

---

## 3. The architecture: hint-supervised, propagation-aligned, iterate-to-converge

```
   build_graph(state, action)  ──▶  node[B,N,Dn], gfeat[B,Dg], a_link, a_flow      (graph.py, unchanged)
                                            │
                ┌───────────────────────────┴───────────────────────────┐
                ▼                                                         │
   _message_pass round r = 0 … K:                                        │  K = mp_rounds (NA1/NA2)
     m_link/m_fwd/m_rev  ──▶  mp_update  ──▶  mp_norm  ──▶  h_r            │  or iterate-to-fixed-point (NA3)
                │                                                         │
                ├──▶  hint_head(h_r)  ──▶  predicted reachability frontier at hop r   ◀── NA1 (NEW head)
                │            ▲                                                          │
                │            └── oracle round-r reachability frontier (FREE, EXACT)  ──┘  hint loss
                ▼
   pooled = mean(h_K)  ──▶  encode (RSSM)  ──▶  cond  ──▶  decode_logits  ──▶  NetDelta   (grammared, unchanged)
                │                                                                │
                ▼                                                                ▼
   total loss = delta_CE  +  λ · Σ_r hint_loss_r              one-step delta (p)  +  free-run rollout (H_free)
```

Three commitments, each tied to a measured HS3 fact:

1. **Supervise the computation, not just the answer (NA1).** HS3 says the arm learns a one-step
   shortcut that does not compose. Hint supervision forces the *intermediate* propagation to be
   correct at every hop — the NAR fix for exactly this. The hints are free; the only cost is the
   auxiliary head and `λ`.
2. **Align the processor to the algorithm (NA2).** The oracle's reachability is a min-hop /
   relaxation computation; the current `mp_update` uses mean aggregation (`_row_normalize`). Algorithmic
   alignment predicts a relaxation-shaped update (max/min over neighbors) is more sample-efficient and
   OOD-robust for this computation. NA2 is the alignment ablation.
3. **Let the processor think longer (NA3).** Reachability closure is a fixed point; a fixed
   `mp_rounds = 3` truncates it on diameters > 3 (and HS3 incr 3 showed `p` *degrades* as the world —
   and its diameter — grows). NA3 iterates at inference, weights frozen, the cheapest lever.

---

## 4. The load-bearing claim (and its banked negative)

NAR rests on one claim: **hint supervision converts a non-composing one-step model into a multi-step
executor.** Verisim can *test* it under an exact oracle, and the branch point is the whole spec:

- **If free hints (± alignment ± iterate) lift `H_free` off zero**, the HS3 wall was
  *under-alignment*: the structured arm had the capacity to execute the computation but was never
  supervised to, and the proxy/truth split (EN4's 6.6× one-step win) converts into horizon. That is the
  program's first crossing of its one genuine wall, and it generalizes (NA-fork to host/dist).
- **If free hints + alignment + iterate still leave `H_free` = 0**, the wall is **compounding, not
  representation**: even a processor supervised to execute the exact intermediate computation cannot
  survive its own delta-rollout's error accumulation at exact tolerance. This is the deepest possible
  HS3 result — it removes the HS3-T trainer confound (we will have driven hint accuracy and `p` up
  independently) and promotes "this trainer plateaus" to "the structured ceiling is fundamental to
  rolling a delta-predictor forward." It is fully bankable and, given the program's priors (η < 1
  thrice-measured), arguably the likelier branch.

The confound HS3 names and cannot resolve — *is `H_free` = 0 the trainer (p ≈ 0.66 < 0.82) or the
architecture?* — is resolved by construction here: NA1 reports hint accuracy and `p` separately, so a
flat `H_free` at *high* hint accuracy and *lifted* `p` is unambiguously compounding, while a flat
`H_free` at still-low hint accuracy is trainer/representation (and NA2/NA3 then test whether alignment
or depth lifts the hint accuracy). Either way the negative is *localized*, not confounded.

---

## 5. Hypotheses (pre-registered, continuing the global H-ID space; SPEC-13 ended at H44)

- **H45 — the graph arm fails to execute the oracle's multi-step reachability propagation (the NAR
  diagnosis).** With no hint supervision, the trained graph arm's per-round node embeddings do **not**
  linearly decode the oracle's intermediate reachability frontier at hops > 1 (probe accuracy at hop
  `r ≥ 2` is materially below hop 1), even while one-step delta `p` ≈ 0.66 (EN4 level). *Refuted if*
  the per-round embeddings already encode the full propagation trajectory (the arm executes the
  algorithm internally and the `H_free` = 0 failure is purely the decoder/rollout, not the processor —
  a strong, surprising result that would redirect the spec to the decode side). Tested as **NA0**
  (the diagnostic probe; gates the rest).

- **H46 — free hint supervision lifts the structured arm's exact free-running horizon off zero (THE
  headline).** Training the graph arm with the oracle's free, exact per-round reachability hints
  (`total = delta_CE + λ·Σ hint_loss`) yields `H_free > 0` at exact tolerance (ε ≤ 0.1) and η ≥ 1,
  beating the HS3 hint-free arm (`H_free` ≈ 0, η < 1) with disjoint CIs — i.e. the network's free
  intermediate supervision converts the 6.6× one-step advantage (EN4) into horizon. *Refuted if*
  hint-supervised `H_free` stays at 0 with η < 1 despite hint accuracy rising — **the bankable
  compounding result** (§4): the wall is error accumulation, not under-alignment, and supervising the
  exact computation does not save the autoregressive delta-rollout. Tested as **NA1**.

- **H47 — algorithmic alignment of the update is necessary, not just hints (the Xu/Ibarz axis).** A
  relaxation-aligned aggregation/update (max/min over neighbors, matching the oracle's min-hop
  computation) in `mp_update` lifts hint accuracy and `H_free` above the mean-aggregation
  (`_row_normalize`) processor *at equal hint supervision* — i.e. the inductive bias of the processor,
  not only the supervision, is load-bearing. *Refuted if* the aligned update ties the mean-aggregation
  update under hints (alignment is not the bottleneck once hints are present — banks that hint
  supervision alone carries the OOD-execution work, the simpler positive). Tested as **NA2**.

- **H48 — iterate-to-convergence at inference buys horizon for free (the DEQ/ACT axis).** Running
  `_message_pass` for `k > mp_rounds` rounds (to fixed point) at inference, **weights frozen**, lifts
  `H_free` and the OOD horizon on larger-diameter worlds versus the fixed `mp_rounds = 3` — the model
  "thinks longer" where the computation is deeper (the HS3 incr-3 finding that `p` falls with world
  size predicts this should help most there). *Refuted if* extra inference rounds do not lift `H_free`
  (or destabilize it) — banking that the truncation depth was not the binding constraint and that
  test-time depth cannot substitute for hint-trained intermediate states. Tested as **NA3**.

- **H49 — NAR-style training closes the proxy/truth split and is a cross-world method (the
  generalization claim + fork).** The hint-supervised, aligned arm (best of NA1–NA3) (a) closes the
  EN4/H11 proxy/truth split — its horizon advantage over the flat arm now matches the *direction* of
  its one-step advantage, not contradicts it; and (b) the recipe transfers to the **host** world
  (`hostsim`, where HS2 measured the floor) and the **distributed** world (`distsim`, where
  partition/reachability is the hidden state ED12 measured), lifting `H_free` there too. *Refuted if*
  the structured arm still under-runs the flat arm on horizon despite winning one-step (the split is
  not closed — the proxy stays misleading, the strongest restatement of HS3's central puzzle), or if
  the lift is network-specific (banks *which* worlds admit NAR repair). Tested as **NA4** (cross-world
  fork; runs only after the NA1 headline lands).

---

## 6. Experiments (prefix **NA** — neural algorithmic reasoning; network world unless noted)

Each follows the house template: a `Config` dataclass with `from_json_file`, a CLI entry point
(`python -m verisim.experiments.naN`), a JSONL record stream, a `plot_*.py` emitting a committed
`.png` + `.csv`, regenerable from `reproduce.sh`, deterministic and seeded (SPEC-2 §12). All reuse the
shipped graph arm, the two oracles, and the `H_free`/`p`/`η` grid from HS3 verbatim (same
seed-reduction, same CSV schema) so every NA number is read on the *same axis* as SPEC-10 §4.6–4.8.

- **NA0 — the NAR diagnosis: does the processor execute the propagation? (H45).** Take the committed
  HS3-trained graph arm; for held-out states, extract the per-round node embeddings `h_r` from
  `_message_pass` and fit a linear probe to the oracle's round-`r` reachability frontier. Report probe
  accuracy vs hop `r`, against one-step `p`. **The gate that licenses NA1** (if the processor already
  executes the algorithm, the fix is the decoder, not hints). `experiments/na0.py`, `configs/na0.json`,
  `figures/plot_na0.py` → `na0_hint_probe.{png,csv}`.

- **NA5 — the decode-side rollout diagnostic: confirm the wall is the decoder, not the processor
  (post-NA0, pure measurement).** Added after NA0's refutation to test the redirection *directly*.
  Free-run the NA0 arm; at each rollout depth apply NA0's **frozen** reachability probe to the embedding
  of the model's *own drifted* state, and disambiguate the in-distribution probe's off-distribution
  decay with a **refit** control (a fresh probe fit on the drifted states, whose oracle frontier is
  free, evaluated held-out). If the refit probe recovers what the frozen one loses, the reachability is
  still in the embedding and the wall is the decoder/rollout, not the processor. Reuses the NA0 arm —
  no trained-arm bet, so it ships now alongside NA0. `experiments/na5.py`, `configs/na5.json`
  → `na5_decode_rollout.{png,csv}`.

- **NA1 — free hint supervision vs the HS3 hint-free arm (H46, the headline).** Add a hint head on
  `h_r` and the loss term `λ·Σ_r hint_loss_r` (oracle round-`r` frontier, teacher-forced); train at
  fixed `m` capacity on the HS3 coverage set. Head-to-head against the committed HS3 graph arm on the
  identical `H_free`/`p`/`η` grid (id + ood, 3 seeds, ε = 0). Report hint accuracy, `p`, and `H_free`
  with CIs, and a `λ` sweep. **The figure that shows free hints lifting the structured floor — or the
  banked compounding negative.** `experiments/na1.py`, `configs/na1.json`,
  `figures/plot_na1.py` → `na1_hint_horizon.{png,csv}`.

- **NA2 — algorithmic-alignment ablation: aggregation/update (H47).** The NA1 trainer, processor built
  two ways — the shipped mean-aggregation `mp_update`/`_row_normalize` vs a relaxation-aligned
  (max/min) aggregation — head to head on the same grid, both with hints. Isolates the processor's
  inductive bias from the supervision. `experiments/na2.py`, `configs/na2.json`,
  `figures/plot_na2.py` → `na2_alignment.{png,csv}`.

- **NA3 — iterate-to-convergence at inference (H48).** Best NA1/NA2 weights, **frozen**; sweep
  inference message-passing rounds `k ∈ {3, 6, 12, to-convergence}`; measure `H_free` (id + ood) and
  per-round hint accuracy, on the 5-host world *and* the 40-host (large-diameter) world where HS3 incr
  3 saw `p` degrade. The cheapest lever; run right after NA1. `experiments/na3.py`, `configs/na3.json`,
  `figures/plot_na3.py` → `na3_think_longer.{png,csv}`.

- **NA4 — closing the proxy/truth split + cross-world fork (H49, deferred).** (a) Plot the EN4/H11
  one-step advantage against the horizon advantage for {flat, HS3 graph, NA1 graph} — does NAR make the
  two agree? (b) Re-run NA1 on `hostsim` and `distsim`. Runs only after NA1 lands, per the evidence
  gate. `experiments/na4_split.py`, `experiments/na4_host.py`, `experiments/na4_dist.py`.

---

## 7. What is confidently buildable now vs gated on a result

The user's standing instruction — *build what is confidently positive, experiment on what is not* —
maps onto the dependency order:

- **Confidently buildable now (machinery exists, result near-certain):**
  - **The free hint emitter.** The oracle already computes reachability; emitting the per-round
    frontier (round-`r` = reachable-in-≤-`r`-hops) is a deterministic read of existing oracle state,
    not a new computation. Building and *committing the hint trajectories* is a build, not a bet.
  - **The hint head + loss term.** A `Linear(d → frontier_dim)` on each round's `h_r` plus a CE/BCE
    term is ~20 lines on top of `_message_pass`; additive, the rest of the arm unchanged.
  - **The NA0 probe.** Reading per-round embeddings and fitting a linear probe is pure diagnostics on
    a shipped, trained arm — deterministic, free, and it *decides whether NA1 is even the right fix*.
- **Gated on NA0 (the diagnosis branch):** if the processor already executes the propagation (H45
  refuted), NA1's hint head is redundant and the spec pivots to the decoder/rollout side — cheap to
  learn, so it runs first.
- **The genuine bets (must be measured):** NA1 (do free hints lift `H_free`? — the headline and the
  banked compounding negative), NA2 (is alignment necessary?), NA3 (does thinking longer help?), NA4
  (split + cross-world). NA1 is the one the whole spec turns on, and its negative is as valuable as its
  positive.

**Recommended build order:** NA0 (diagnose; gate) → NA1 (the headline — free hints vs HS3 hint-free) →
NA3 (free inference-depth lever) → NA2 (alignment ablation) → NA4 (split + cross-world fork). Each rung
graduates on a committed figure or a banked negative that licenses the next (SPEC §10.1, §12).

---

## 8. Build, reproduce, CI

### 8.1 Module layout (additive only)

```
src/verisim/netmodel/
  graph_model.py        # EDIT (additive): optional hint_head on per-round h_r; encode returns hints
  graph_hints.py        # NEW — oracle → per-round reachability-frontier hint trajectories (free, exact)
  graph_train_nar.py    # NEW — train_graph_model + λ·Σ hint_loss (NA1); aligned-update variant (NA2)
src/verisim/experiments/
  na0.py … na4_*.py      # NEW — the NA experiments
figures/
  plot_na0.py … plot_na4.py   # NEW — committed-figure generators
configs/
  na0.json … na4.json         # NEW — committed sweep configs
```

The hint head is **off by default** (`hint_dim: int | None = None` on `GraphRSSMConfig`), so the
shipped `GraphRSSMNet`/`GraphRSSMWorldModel` path, every existing test, and all of SPEC-10's committed
numbers are **unchanged** — SPEC-14 adds a training mode and a diagnostic, it does not alter the arm's
default behavior or the deterministic core.

### 8.2 `reproduce.sh` (new NA block, in dependency order)

```bash
echo "== NA0: does the processor execute the propagation? (H45) — gates NA1 =="
python -m verisim.experiments.na0 --config configs/na0.json --out runs/na0/records.jsonl --plot figures/na0_hint_probe.png
echo "== NA1: free hint supervision vs HS3 hint-free arm (H46) — THE HEADLINE =="
python -m verisim.experiments.na1 --config configs/na1.json --out runs/na1/records.jsonl --plot figures/na1_hint_horizon.png
echo "== NA3: iterate-to-convergence at inference (H48) — free depth lever =="
python -m verisim.experiments.na3 --config configs/na3.json --out runs/na3/records.jsonl --plot figures/na3_think_longer.png
echo "== NA2: algorithmic-alignment ablation (H47) =="
python -m verisim.experiments.na2 --config configs/na2.json --out runs/na2/records.jsonl --plot figures/na2_alignment.png
# NA4 (split + cross-world fork) gated on NA1.
```

The NA block runs on CPU (the deterministic gate); training the hint-supervised `m` arm is the same
order of cost as the committed HS3 graph trainer. CI (`ubuntu-latest`) stays the free Linux
confirmation; NA tests assert structural invariants (the hint emitter's round-`r` frontier is monotone
non-decreasing in `r` and matches the oracle's closure at `r = diameter`; the hint head is a no-op when
`hint_dim is None`; NA1's `H_free` is read on the same grid as HS3), not exact magnitudes — so the same
tests pass on the macOS primary host and Linux CI (the macOS-first principle).

---

## 9. Scope, non-goals, honest caveats

- **This repairs the arm; it does not invent a new arm.** SPEC-14 changes how the shipped graph arm is
  trained (hints), shaped (alignment), and run (iterate). Over-claiming a new architecture is the
  failure mode this line forbids; the result is about *supervising the existing processor on the
  oracle's free computation*.
- **The headline is conditional on NA0's branch.** If the processor already executes the propagation
  (H45 refuted), hints are redundant and the fix is the decoder/rollout — a clean, banked redirection,
  not a failure.
- **NA1's negative is first-class and likely.** η < 1 was measured *three times* in HS3 (capacity,
  data, world size). The honest prior is that the wall is compounding; if NA1's `H_free` stays 0 at
  *high* hint accuracy and *lifted* `p`, that is the deepest HS3 result, not a disappointment — it
  removes the trainer confound HS3-T could not.
- **The trainer/representation confound is addressed, not assumed away.** By reporting hint accuracy
  and `p` separately from `H_free`, NA1 can attribute a flat horizon to compounding (high hint acc,
  flat `H_free`) vs trainer (low hint acc) — the disambiguation HS3 explicitly flags it cannot make
  from inside the end-to-end recipe.
- **Hints are free here *because* the oracle is the algorithm.** This is the inversion of the NAR
  field's hardest constraint (hints are usually costly/unavailable — the whole motivation of
  arXiv:2306.13411). The claim is narrow and exact: in computer worlds with an oracle, intermediate
  algorithmic supervision is free; whether that suffices to break compounding is the measured question.
- **Iterate-to-convergence assumes a fixed-point computation.** Reachability closure is one; the *full*
  delta (services, firewall edits, flows) is not purely a fixed point, so NA3's benefit is expected on
  the reachability-shaped part of the prediction, and the spec measures it there (large-diameter world)
  rather than assuming it globally.

---

## 10. Provenance & reading order

SPEC-14 is the head-on attack on the wall SPEC-10's HS3 found and SPEC-12 routed around. SPEC-5 gave
the graph arm and the two oracles; SPEC-8 put the oracle in the representation (the same free-oracle
inversion SPEC-14 applies to intermediate computation); SPEC-10 measured the structured arm's genuine
compounding ceiling (η < 1, `H_free` = 0 across all three resource axes) and the EN4 proxy/truth split
(6.6× one-step win, zero horizon). The method is Neural Algorithmic Reasoning — hint/step-wise
supervision (Veličković et al., ICLR 2020; CLRS-30), algorithmic alignment (Xu et al., ICLR 2020),
generalist processors (Ibarz et al., 2022), and iterate-to-converge processors (DEQ; the DEQ
algorithmic reasoner) — and the unique contribution is the one NAR never had: **the oracle emits the
algorithm's exact intermediate states for free**, so the supervision the field pays for (or tries to
avoid, arXiv:2306.13411) is, here, a free read of the ground truth.

Reading order for a newcomer: [SPEC.md §0/§5](./SPEC.md) (the oracle, `H_ε(ρ)`, the two arms) →
[SPEC-10 §4.6–4.11](./SPEC-10.md) (HS3 — the wall, and the EN4 proxy/truth split this attacks) →
[`graph_model.py`](../../src/verisim/netmodel/graph_model.py) (the processor the hints attach to) →
this document (§1 the motivation, §3 the architecture, §4 the load-bearing claim, §5 the hypotheses) →
`src/verisim/netmodel/graph_hints.py` and `src/verisim/experiments/na0.py` (the concrete build, once
shipped).

---

## 11. Status (2026-06-09) — NA0 + NA5 SHIPPED; H45 REFUTED, the decode-side redirection confirmed

The diagnosis gate is built, committed, and **decided the branch §7 said it would**. NA0 trains three
HS3-level graph arms (8 hosts, `d_model=64`, `mp_rounds=3`, 600 steps; one-step next-state-exact
`p = 0.475`, the EN4 regime), then on held-out states extracts the per-round node embeddings via the
new additive [`GraphRSSMNet.message_pass_trace`](../../src/verisim/netmodel/graph_model.py) and fits a
closed-form ridge probe from each round-`r` embedding `h_r` to the oracle's free, exact `≤ r`-hop
reachability frontier `F_r` ([`reach_frontiers`](../../src/verisim/experiments/na0.py), hop-bounded
BFS over up-links between up-hosts). The load-bearing comparison is the **processed** probe `h_r → F_r`
against a **pre-propagation control** `h_0 → F_r` (the input projection, which carries the node
features but *no link adjacency*) on the identical target — so any margin is exactly what the message
passing *adds*. CPU, deterministic, seeded; multi-seed with bootstrap CIs.

- ✅ **NA0 — the NAR diagnosis (H45 REFUTED, the gate).** The processor's per-round embeddings linearly
  decode the multi-hop reachability frontier, and *increasingly so with depth*: the lift over the
  marginal-rate baseline is `0.119 / 0.237 / 0.283` at hops `r = 1 / 2 / 3`, while the pre-propagation
  control (`h_0 → F_r`) reaches only `0.037 / 0.090 / 0.131` — the processed lift is **~2–3× the
  control at every deep hop, with non-overlapping CIs**. Message passing demonstrably injects the
  multi-hop reachability a linear readout extracts and the input embedding lacks. **H45 is refuted**:
  the `mp_rounds` processor *does* execute the propagation (in the NAR linear-decodability sense). This
  is precisely the §5 "*Refuted if* the per-round embeddings already encode the full propagation
  trajectory … a strong, surprising result that would redirect the spec to the decode side."
  [`na0`](../../src/verisim/experiments/na0.py), [`configs/na0.json`](../../configs/na0.json),
  [`figures/na0_hint_probe.png`](../../figures/na0_hint_probe.png).
- ✅ **NA5 — the decode-side rollout diagnostic (the redirection confirmed at the rollout level).** NA0
  read the embedding on *teacher-forced* states; NA5 tests the redirection *directly* by **free-running**
  the same arm and asking, at each rollout depth, whether the processor still encodes the reachability of
  its *own drifted* state. The trap is that NA0's probe, fit in-distribution, degrades off-distribution
  for *either* reason (representation drift *or* probe-transfer failure); the control that resolves it is
  to **refit a fresh probe on the drifted states** (their oracle frontier is free) and evaluate it
  held-out. Result: the frozen in-distribution probe falls with depth (`0.87 → 0.71`) but a refit probe
  **recovers most of it** (`0.87 → 0.83`, **+0.12 over frozen at the deepest bucket**), so the reachability
  is *still linearly in the embedding* of the drifted state — and `tracks-truth` (probe vs the *true*
  state's frontier) falls **~4× more than the refit probe** as the state divergence climbs. **The wall is
  predominantly the decoder/rollout**: the processor stays faithful to whatever state it is in, the
  autoregressive decoder emits wrong deltas that compound. A small residual (~0.04–0.06) of genuine
  off-distribution representation drift is reported, not hidden. Pure measurement on the NA0 arm + a frozen
  probe — no trained-arm bet. [`na5`](../../src/verisim/experiments/na5.py),
  [`configs/na5.json`](../../configs/na5.json),
  [`figures/na5_decode_rollout.png`](../../figures/na5_decode_rollout.png).

**What this means for the spec (the principled redirection, §7's gate firing).** SPEC-14 was built to
attack the HS3 `H_free = 0` wall by supervising the processor's intermediate computation (NA1's hint
head on `h_r`). NA0 shows that supervision would be **redundant**: the per-round embeddings already
carry the round-`r` frontier linearly, so the processor is not where the propagation fails. By the §7
dependency gate — *"if the processor already executes the propagation (H45 refuted), NA1's hint head is
redundant and the spec pivots to the decoder/rollout side"* — the genuine open work moves **downstream
of the processor**: the autoregressive **delta decoder + free-running rollout**, where the EN4/H11
proxy/truth split and the η < 1 compounding actually live. The bankable conclusion: the HS3 wall is a
**decode/compounding** failure, not an under-aligned **processor** — the message passing computes the
reachability; the autoregressive head fails to turn it into a faithful multi-step rollout.

- **NA1–NA4, re-scoped (genuine bets, deferred — the LP7 discipline).** The hint-on-`h_r` head (NA1 as
  originally cast) is deprioritized by the NA0 verdict. The redirected open work is decoder-side and
  remains a trained-arm bet: supervise the **decoder/rollout** against the oracle (free-run relabel /
  pushforward loss — the SPEC-16 RS-family lever, now on the structured arm), and test whether
  iterate-to-convergence (NA3) on the *reachability-shaped* part of the prediction buys horizon. NA2
  (aggregation alignment) and NA4 (cross-world fork) follow only if a decoder-side lift lands. As in
  SPEC-13/15/16/17, the controlled diagnostic ships now; the learned lift is measured, not assumed.

**Honest caveats.** (i) "Executes the propagation" is the NAR operational sense — *linear
decodability* of `F_r` from `h_r`, not a proof the network's forward computation is BFS. (ii) The
control lift also rises with `r` (the probe is expressive and denser frontiers correlate with up-host
status); the claim rests on the **processed-minus-control margin**, which is decisive (2–3×,
CI-separated), not on the processed lift alone. (iii) The world is 8 hosts (small diameter); the
qualitative gate (processor computes multi-hop reachability) is robust, but the quantitative margins
are this-scale numbers, read on the same axis as HS3 per §6.
