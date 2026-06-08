# SPEC-16 — Rollout-Stability Training: Free-Oracle DAgger and the Exposure-Bias Cure

**Method specification: every horizon number to date trained the proposer *teacher-forced* (on the
oracle's true states) and then rolled it out *free-running* (on its own predicted states). That is the
textbook exposure-bias / covariate-shift gap — and it is exactly the gap [SPEC-10](./SPEC-10.md) HS1.1
caught red-handed (a bigger, per-step-*more*-accurate model that is *less* faithful over the horizon).
The classic cure for that gap — query an expert on the LEARNER'S OWN drifted state distribution
(DAgger) — has one bottleneck: the expert is expensive. In verisim the expert is the oracle: free,
exact, callable at any visited state. So verisim can run the cure the simulator-learning field can only
approximate. This spec measures whether the cure lifts `H_ε`, or whether the gap survives it — in which
case the wall is fundamental compounding, not a train/deploy mismatch.**

> **▶ TRAINING-METHOD SPEC — 2026-06 — DESIGN-STAGE (◐).** A *method* spec, sibling to
> [SPEC-8](./SPEC-8.md) (oracle-grounded SSL) and [SPEC-12](./SPEC-12.md) (planning over the model):
> it invents **no new world** (runs on the SPEC-5 network world and the SPEC-6 host world) and **no
> new oracle** (reuses [`ReferenceNetworkOracle`](../../src/verisim/netoracle/reference.py) for the
> exact relabel and [`ControlPlaneOracle`](../../src/verisim/netoracle/control_plane.py) for the cheap
> reachability projection). What it changes is the **training loop** of `M_θ`, not the model and not
> the inference loop. Inspiration is acknowledged plainly: the cure is DAgger (Ross, Gordon, Bagnell,
> AISTATS 2011) and the learned-simulator "pushforward"/noise-injection lineage (GNS, MeshGraphNets,
> Stachenfeld, Brandstetter, PDE-Refiner); the contribution verisim alone can make is that DAgger's
> expensive expert query is here **free and exact at every drifted state**, so the aggregation step
> that oracle-free simulator learning skips can be run to convergence and measured against an exact
> long-horizon ground truth.

Read [SPEC.md §2](./SPEC.md) (why the oracle is free and exact), [SPEC.md §6](./SPEC.md) (the
delta-prediction commitment and oracle-as-reward), [SPEC-10 §4.2](./SPEC-10.md) (HS1.1 — the
exposure-bias signature this spec attacks) and [SPEC-10 §4.6–4.8](./SPEC-10.md) (HS3 — the structured
arm's `H_free=0` wall), and the **already-shipped drift-mitigation levers** in
[`netmodel/graph_train.py`](../../src/verisim/netmodel/graph_train.py)
([`build_self_forced_examples`](../../src/verisim/netmodel/graph_train.py),
[`train_graph_model_self_forced`](../../src/verisim/netmodel/graph_train.py), and the noise-injection
branch of [`train_graph_model`](../../src/verisim/netmodel/graph_train.py)) — SPEC-16 is the rigorous
faithful-horizon study those levers were built for but never received.

---

## 0. One-paragraph thesis

The program's sharpest finding is a *divergence*: as the flat arm grows past its compute-optimal peak,
one-step accuracy `p` stays flat and high (0.81–0.90) while the exact free-running horizon `H_free`
**falls ~45%** and ood horizon-efficiency `η` crosses below 1 (HS1.1, [SPEC-10 §4.2](./SPEC-10.md)); and
the structured arm sits at `H_free≈0` with `η<1` across capacity, data, and world size (HS3,
[SPEC-10 §4.6–4.8](./SPEC-10.md)) — it free-runs *shorter* than its own i.i.d. prediction `p/(1-p)`.
Both are the canonical signature of **compounding error from exposure bias**: a model trained on the
oracle's true states is never shown the off-distribution states its own drift produces, so at deploy it
has no idea how to recover. The simulator-learning field's standard fixes — noise injection (GNS,
MeshGraphNets, Stachenfeld), the pushforward trick (Brandstetter), multi-step refinement (PDE-Refiner),
scheduled sampling / professor forcing (Bengio, Lamb) — all approximate the one thing that *exactly*
removes covariate shift: DAgger's dataset aggregation, retraining on the learner's own visited state
distribution with the expert's correct label there. DAgger is rarely run to its limit because the expert
is costly; **in verisim the expert is the oracle, free and exact at any visited state**, so we can run
**free-oracle DAgger** — roll `M_θ` free-running, query the oracle for the exact delta at each drifted
state it lands in, aggregate, retrain, repeat — and measure, against exact long-horizon ground truth,
whether it lifts `H_free` where teacher-forced supervision plateaus. The claim, stated to be falsified:
**rollout-aware training closes the HS1.1 exposure-bias gap — `H_free` rises toward `H_indep` (`η→1`)
and the structured arm leaves its `H_free=0` floor — and it does so by paying the bias-stability price
the literature documents (one-step `p` may dip while horizon climbs).** Every branch is bankable
(SPEC §10.1): if rollout-aware training does *not* lift `H_free`, the HS3/HS1.1 gap is **fundamental
compounding**, not exposure bias — a deeper, sharper result that retires the "just train it on its own
states" hope; and if it *does* lift horizon but only by trading away accuracy, the honest figure is the
**net faithful-horizon-per-compute**, not either axis alone.

---

## 1. Why now: the gap SPEC-10 named and the cure only the oracle can run

Two facts from SPEC-10 set this spec up, and one asset makes it uniquely verisim's:

- **HS1.1 is an exposure-bias signature, exactly.** [SPEC-10 §4.2](./SPEC-10.md): across the top of the
  capacity axis the one-step proxy `p` is flat-and-high while `H_free` falls and ood `η` crosses below 1.
  A model that is per-step *more* accurate but horizon-*less* faithful is precisely what train-on-true /
  deploy-on-own-predictions produces — the model overfits the on-distribution one-step map and has never
  seen the drifted states free-running visits. The free exact oracle is the only instrument that can
  *see* this divergence (the proxy `p` is blind to it); this spec asks whether it can also *close* it.
- **HS3 is `η<1`: compounding, localized.** [SPEC-10 §4.7](./SPEC-10.md): the structured arm free-runs
  *shorter* than its i.i.d. prediction (`η<1`) across every resource axis — its small-magnitude errors
  are ubiquitous and *compound*. Whether that compounding is curable by rollout-aware training, or is a
  representational wall no training schedule crosses, is the single most consequential open question the
  program has, because it decides whether HS3 retires the structured arm or merely its *trainer*.
- **The DAgger bottleneck vanishes here.** DAgger (Ross et al. 2011) proves that aggregating data from
  the learner's own state distribution, labeled by the expert there, reduces the compounding imitation
  error from `O(εT²)` to `O(εT)`. The catch every applied paper hits: the expert is a human, a slow
  planner, or an expensive solver, so only a few DAgger iterations are affordable and the learner's
  distribution is under-covered. **Verisim's expert is `O(s,a)` — free, exact, deterministic, and
  callable at *any* state `M_θ` drifts into.** The aggregation DAgger's theory wants is, here, the cheap
  step; the metric DAgger never had (exact long-horizon faithfulness) is, here, free. This is the
  asymmetry SPEC.md §2 promises, cashed out at training time.

The levers themselves are not new to the repo — [SPEC-5 §6.3](./SPEC-5.md) already ships noise injection
and scheduled-sampling self-forcing, and EN4 ran them once at small scale, banking a thin negative (a
small one-step dip, no horizon lift yet, [SPEC-5 report §NW8](../report.md)). What is new is the
**rigorous faithful-horizon study at HS1.1/HS3 scale**: the levers run as a pre-registered horizon
experiment with the independence baseline, the net-per-compute accounting, and the free-oracle DAgger
formalization the one-shot EN4 lever was never given.

---

## 2. The lineage folded in (and the design choice each one forces)

### 2.1 Imitation / sequence-modeling exposure bias (the diagnosis)

- **DAgger** (Ross, Gordon, Bagnell, AISTATS 2011, arXiv:1011.0686). Aggregate trajectories from the
  *learner's* state distribution, label with the expert there, retrain; `O(εT)` not `O(εT²)`. →
  **Design choice:** RS1 *is* DAgger with the oracle as expert — roll `M_θ` free-running, query
  [`ReferenceNetworkOracle.step`](../../src/verisim/netoracle/reference.py) for the exact delta at each
  visited state, aggregate into the [`netmodel/dataset.py`](../../src/verisim/netmodel/dataset.py)
  example pool, retrain `M_θ`, iterate. The expert query is the free step.
- **Scheduled Sampling** (Bengio, Vinyals, Jaitly, Shazeer, NeurIPS 2015, arXiv:1506.03099) and
  **Professor Forcing** (Lamb, Goyal, Zhang, Zhang, Courville, Bengio, NeurIPS 2016, arXiv:1610.09038).
  Anneal from teacher-forced toward self-generated inputs; match train/sample dynamics. → **Design
  choice:** RS2 reuses the *shipped*
  [`train_graph_model_self_forced`](../../src/verisim/netmodel/graph_train.py) (ramping `sample_prob`
  0→`max_sample_prob`) and adds the missing measurement — the one-step-`p`-vs-`H_free` tradeoff curve.

### 2.2 Learned-simulator rollout stability (the cure, oracle-free)

- **GNS — Learning to Simulate Complex Physics with Graph Networks** (Sanchez-Gonzalez et al., ICML
  2020, arXiv:2002.09405) and **MeshGraphNets** (Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia, ICLR
  2021, arXiv:2010.03409). The decisive long-rollout trick is **training-time noise injection** so the
  model learns to correct its own drift. → **Design choice:** RS3 reuses the shipped oracle-relabeled
  noise branch of [`train_graph_model`](../../src/verisim/netmodel/graph_train.py) (corrupt the input
  with [`corrupt_state`](../../src/verisim/netmodel/graph_train.py), relabel the target with
  `O(s̃,a)` — the GNS lever *made exact* by the free total oracle) and sweeps `noise_prob` against
  `H_free`. The point GNS could only approximate (what label is correct at the noisy state?) verisim
  answers exactly.
- **Learned Coarse Models for Efficient Turbulence** (Stachenfeld et al., ICLR 2022, arXiv:2112.15275).
  Learned simulators are generically unstable; **tuning training noise and temporal downsampling** is
  what buys stable rollouts. → **Design choice:** the noise magnitude is a *measured* knob (RS3), not a
  fixed heuristic, and the result is read on `η` (the compounding penalty), which the oracle makes exact.
- **Message Passing Neural PDE Solvers / the pushforward trick** (Brandstetter, Worrall, Welling, ICLR
  2022, arXiv:2202.03376). Pose stability as domain adaptation: unroll one extra model step *without
  gradient* and train the next prediction to be correct *from the model's own pushed-forward state*. →
  **Design choice:** RS4's **multi-step unrolled loss** is the pushforward made exact — unroll `M_θ` for
  `k` steps and supervise *every* unrolled step against the oracle's exact delta for the *visited*
  (drifted) state, not the true state. The oracle removes the pushforward's load-bearing approximation
  (no need to stop-gradient to dodge an unknown target; the target is known exactly at the drifted
  state).
- **PDE-Refiner** (Lippe, Veeling, Perdikaris, Turner, Brandstetter, NeurIPS 2023, arXiv:2308.05732).
  Long-rollout accuracy is gated by neglected non-dominant frequencies; a multi-step diffusion-style
  refinement recovers them. → **Design choice (deferred, RS-future §7):** if RS1–RS4 lift horizon but
  leave a residual, an oracle-supervised refinement pass over the predicted delta is the banked next
  lever — out of scope for the design-stage headline.

### 2.3 What the program already built that this sits on

- **The exposure-bias levers ship** ([SPEC-5 §6.3](./SPEC-5.md)): noise injection and self-forcing are
  in [`graph_train.py`](../../src/verisim/netmodel/graph_train.py) and ran once in EN4. SPEC-16 does not
  re-invent them; it gives them the faithful-horizon axis, the independence baseline, and the
  net-per-compute accounting — and adds RS1 (free-oracle DAgger as a *loop*, not a one-shot relabel) and
  RS4 (the multi-step unrolled loss).
- **The metric and loop ship verbatim.** `H_free = H_ε(ρ=0)` comes from
  [`run_net_rollout`](../../src/verisim/netloop/runner.py) (the NW5 loop) at `budget=0`; per-step `p`
  from [`delta_exact_rate`](../../src/verisim/netmetrics/exact.py); the independence prediction
  `H_indep = p/(1-p)` and `η = H_free/H_indep` from the HS1 grid; seed-aggregated CIs from
  [`bootstrap_ci`](../../src/verisim/metrics/aggregate.py). SPEC-16 changes only how `M_θ` was *trained*
  before it enters that unchanged measurement.

---

## 3. The method: four rollout-stability trainers, one measurement

```
   TEACHER-FORCED (today's baseline)        FREE-ORACLE DAGGER (RS1)
   train on O's true states  ───────▶       roll M_θ free-running ─▶ visit drifted s̃_t
   deploy on M_θ's own states (drift)        query O(s̃_t, a_t) ─▶ exact delta  (FREE expert)
   ⇒ exposure bias, η<1 / H_free↓            aggregate {(s̃_t, a_t) → O-delta} into the pool
   (HS1.1, HS3)                              retrain M_θ on the union; iterate N rounds
                                            ⇒ M_θ learns to recover from its OWN errors
        │                                                    │
        ▼                                                    ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  THE UNCHANGED MEASUREMENT (HS1 grid, reused verbatim)                    │
   │  p = delta_exact_rate · H_free = run_net_rollout(budget=0) · η = H_free/H_indep │
   │  read each trainer on the SAME axis: does H_free rise? does η→1? net/compute?  │
   └─────────────────────────────────────────────────────────────────────────┘
        ▲                         ▲                         ▲
   SCHEDULED SAMPLING (RS2)   NOISE INJECTION (RS3)   MULTI-STEP UNROLLED (RS4)
   anneal teacher→self-forced  corrupt input, O-relabel  unroll k steps, O-supervise
   (Bengio/Lamb; shipped)      (GNS/Stachenfeld; shipped) every drifted step (Brandstetter)
```

Four commitments, each tied to a measured verisim fact:

1. **Train on the deploy distribution.** The defining move of all four trainers is to supervise `M_θ` on
   states it actually visits at `ρ=0` (its own drift), labeled by the exact oracle there — directly
   attacking the HS1.1 train/deploy mismatch.
2. **The label is always exact and free.** Every drifted state's target is `O(s̃,a)` — the SPEC-2.1
   "free infinite teacher" at the off-distribution state. This is the one step DAgger and the pushforward
   trick must approximate; here it is a deterministic oracle call.
3. **Read the result on the exact horizon, not the proxy.** HS1.1 proved `p` is blind to the gain that
   matters. Every RS experiment's headline is `H_free` and `η`, with `p` reported alongside *to expose
   the tradeoff*, never as the success metric.
4. **Account for the bias-stability tradeoff honestly.** Noise/self-forcing/unrolling are documented to
   *lower* one-step accuracy while *raising* horizon — the HS1.1 signature run in reverse, deliberately.
   The load-bearing figure is **net faithful-horizon-per-training-compute** (§4 H58), not horizon alone.

---

## 4. Hypotheses (pre-registered, continuing the global H-ID space past SPEC-12's H38)

- **H55 — free-oracle DAgger lifts the free-running horizon where teacher forcing plateaus (THE
  headline).** Aggregating the learner's own drifted states with exact oracle labels and retraining lifts
  `H_free` materially above the teacher-forced baseline at equal capacity and equal *total* training
  data, with `η` rising toward 1 (the compounding penalty shrinking). *Refuted if* DAgger rounds do not
  move `H_free` beyond teacher-forced CIs — in which case **the gap is fundamental compounding, not
  exposure bias**: training on the deploy distribution does not help, the model simply cannot represent
  the recovery map, and the HS1.1/HS3 wall is deeper than a train/deploy mismatch (the deepest, most
  valuable negative — it sharpens HS3 from "wrong trainer" to "wrong model class"). Tested as **RS1**.

- **H56 — the structured arm leaves its `H_free=0` floor under rollout-aware training, or the wall is
  confirmed representational.** Applying RS1–RS4 to the GNN+RSSM graph arm (the HS3 subject, `H_free≈0`,
  `η<1` across capacity/data/world) lifts its exact free-running horizon off zero. *Refuted if* the graph
  arm stays pinned at `H_free≈0` under every rollout-aware trainer — directly upgrading HS3's verdict
  from "the committed trainer plateaus" (its standing caveat, [SPEC-10 §4.6](./SPEC-10.md)) to "the
  representation cannot self-stabilize even when trained on its own drift," the strongest possible form
  of the structured-arm wall. Tested as **RS1/RS3/RS4 on the graph arm (RS5)**.

- **H57 — the bias-stability tradeoff is real and signed (the HS1.1 signature, induced on purpose).**
  Rollout-aware training *lowers* one-step `p` while *raising* `H_free` — the exact HS1.1 divergence run
  in reverse, deliberately induced and measured: a per-step-*less*-accurate model that is *more* faithful
  over the horizon. *Refuted if* `p` and `H_free` move together (no tradeoff — rollout-aware training is
  a free lunch on both axes here, a clean positive that would say exposure bias was pure waste with no
  accuracy cost). Tested as **RS2/RS3 sweeps** (the `sample_prob` / `noise_prob` knobs are the x-axis).

- **H58 — net faithful-horizon-per-compute is the honest verdict, and it may be flat.** Charged for the
  extra training compute (DAgger's rollout+relabel rounds, unrolling's `k×` forward cost) *and* for any
  one-step accuracy lost (H57), the **net** faithful horizon per unit training compute may not beat the
  teacher-forced baseline even when raw `H_free` rises. *Refuted if* the net-per-compute frontier of a
  rollout-aware trainer dominates teacher forcing — rollout-aware training is worth its cost. *The
  banked null:* the tradeoff nets to zero (horizon gained ≈ accuracy/compute lost), so exposure-bias
  training is a *reshaping* of the error, not a reduction — itself a clean, citable result about where
  the faithful-horizon budget actually goes. Tested as **RS6** (the cross-trainer Pareto figure).

- **H59 — the cure transfers across worlds and proposers (fork).** The rollout-aware trainer that wins
  on the network flat arm also lifts `H_free` on the host world (SPEC-6, where HS2 showed the floor is
  re-lowered and the headroom re-opened) and on the structured arm — i.e. the cure is a property of the
  *oracle loop*, not of one world×proposer cell. *Refuted if* the winning trainer is
  world-or-proposer-specific (e.g. it helps the flat network arm but not the host arm) — itself a result
  about which (world, proposer) pairs admit an exposure-bias cure, mirroring HS2/HS3's
  proposer-dependence verdict. Tested as **RS7** (deferred fork; runs after the network headline RS1).

---

## 5. Experiments (prefix **RS** — rollout stability; network world, flat arm, unless noted)

Each follows the house template: a `Config` dataclass with `from_json_file`, a CLI entry point, a JSONL
record stream, a `plot_*.py` emitting a committed `.png` + `.csv`, regenerable from `reproduce.sh`,
deterministic and seeded (SPEC-2 §12; `torch.set_num_threads(1)`). All reuse the shipped HS1 measurement
grid (`p`, `H_free`, `H_indep`, `η` from [`run_net_rollout`](../../src/verisim/netloop/runner.py) /
[`delta_exact_rate`](../../src/verisim/netmetrics/exact.py) /
[`bootstrap_ci`](../../src/verisim/metrics/aggregate.py)) — SPEC-16 adds only the *trainers* and these
harnesses.

- **RS1 — free-oracle DAgger vs teacher forcing (H55, the headline).** Fix capacity at the HS1.1
  compute-optimal `l` cell (110k params) and a fixed *total* example budget (so DAgger and teacher
  forcing see equal data — the only fair comparison). Compare **(a)** teacher-forced baseline
  (`train_batched` on `n` oracle examples) against **(b)** `N` DAgger rounds: train on `n/N` examples,
  roll `M_θ` free-running, relabel each visited drifted state with
  [`ReferenceNetworkOracle.step`](../../src/verisim/netoracle/reference.py), aggregate, retrain. Report
  `H_free`/`η` vs DAgger round, and a same-budget bar against teacher forcing.
  `experiments/rs1_dagger.py`, `configs/rs1_dagger.json`. **The figure that shows the cure (or banks the
  fundamental-compounding negative).**

- **RS2 — scheduled sampling: the `sample_prob` tradeoff curve (H57).** Sweep
  [`train_graph_model_self_forced`](../../src/verisim/netmodel/graph_train.py)'s `max_sample_prob`
  ∈ {0, 0.25, 0.5, 0.75, 1.0} (0 = teacher forcing); the shipped lever, finally on the horizon axis.
  Report `p` and `H_free` on the *same* x-axis — the two-curve crossing *is* the HS1.1 signature reversed.
  `experiments/rs2_scheduled.py`.

- **RS3 — noise injection: the `noise_prob`×magnitude grid (H57).** Sweep the oracle-relabeled noise
  branch of [`train_graph_model`](../../src/verisim/netmodel/graph_train.py) over `noise_prob` and
  [`corrupt_state`](../../src/verisim/netmodel/graph_train.py) magnitude; report the `H_free` response
  surface and the `p`-cost contour (the GNS/Stachenfeld lever, made exact, measured on horizon).
  `experiments/rs3_noise.py`.

- **RS4 — multi-step unrolled loss (the pushforward, made exact) (H55/H57).** New trainer
  `train_unrolled` ([`netmodel/graph_train.py`](../../src/verisim/netmodel/graph_train.py), additive):
  unroll `M_θ` for `k` ∈ {1,2,4,8} steps on its own predictions and supervise *every* unrolled step
  against the oracle's exact delta for the *visited* state. Report `H_free`/`η` vs `k` and the per-step
  forward-cost multiplier (the H58 denominator). `experiments/rs4_unrolled.py`.

- **RS5 — rollout-aware training on the structured arm (H56).** RS1/RS3/RS4 re-run with the GNN+RSSM
  graph proposer (the HS3 subject). The single most decisive cell: does *any* rollout-aware trainer move
  the structured `H_free` off 0? `experiments/rs5_graph.py`. *(High-stakes: a lift refutes HS3's
  representational reading; a pin confirms it at maximal strength.)*

- **RS6 — net faithful-horizon-per-compute Pareto (H58).** All trainers on one figure: x = total
  training compute (FLOP-proxy: forward passes × params, charging DAgger rounds and unroll depth),
  y = `H_free`; the teacher-forced baseline is the reference frontier. The honest verdict on whether the
  cure pays. `experiments/rs6_pareto.py`.

- **RS7 — cross-world / cross-proposer fork (H59, deferred).** Re-run the RS1 winner on the host world
  ([`horizon_host_scaling`](../../src/verisim/experiments/horizon_host_scaling.py)'s apparatus) and the
  structured arm. Runs only after the network headline (RS1) lands, per the evidence gate.
  `experiments/rs7_host.py`, `experiments/rs7_graph.py`.

---

## 6. What is confidently buildable now vs gated on a result

The user's discipline — *build what is confidently positive, experiment on what is not* — maps onto the
dependency order:

- **Confidently buildable now (the machinery exists, the build is near-certain):**
  - The **free-oracle DAgger loop** (RS1's apparatus). The oracle relabel is
    [`ReferenceNetworkOracle.step`](../../src/verisim/netoracle/reference.py); the free-run rollout that
    produces drifted states is [`run_net_rollout`](../../src/verisim/netloop/runner.py) at `budget=0`;
    the example pool is [`netmodel/dataset.py`](../../src/verisim/netmodel/dataset.py). DAgger is a
    *loop over shipped parts*, not a new primitive.
  - **RS2/RS3 harnesses**: the trainers
    ([`train_graph_model_self_forced`](../../src/verisim/netmodel/graph_train.py), the noise branch)
    already ship; only the horizon-axis sweep + plot are new. These run almost immediately.
- **The one genuinely new trainer:** `train_unrolled` (RS4) — additive, ~50 lines, the pushforward
  made exact. Buildable now; it is the only non-trivial new code.
- **The genuine bets (must be measured):** RS1 (does DAgger actually lift horizon, or is the gap
  fundamental — H55), RS5 (does *anything* move the structured floor — H56), RS6 (does the cure pay net
  — H58). RS1 is the gate: a lift licenses RS4–RS7; a null is itself the headline (fundamental
  compounding) and reshapes the rest of the program.

**Recommended build order:** RS2/RS3 (cheap, shipped levers on the horizon axis — fast tradeoff curves)
→ RS1 (the DAgger headline; gates everything) → RS4 (unrolled) → RS5 (the structured-arm decider) →
RS6 (the net-per-compute verdict) → RS7 fork. Each rung graduates on a committed figure or a banked
negative that licenses the next (SPEC §10.1, §12).

---

## 7. Scope, non-goals, honest caveats

- **This is a trainer, not a new world, model, or metric.** It changes how `M_θ` is fit before the
  *unchanged* SPEC-10 measurement runs. Over-claiming a new dynamics or representation result from
  SPEC-16 is the failure mode this line forbids; the result is strictly about *how training-distribution
  choice moves the exact faithful horizon*.
- **The headline negative is first-class and arguably the bigger result.** If RS1 does not lift `H_free`
  (H55 refuted), the HS1.1/HS3 gap is **fundamental compounding** — the model class cannot represent the
  recovery map even when shown its own drift with exact labels. That refutes the "wrong trainer" reading
  of HS3 and is the deepest negative the program can bank; it is *not* feared, it is pre-registered.
- **The tradeoff may net to zero (H58).** Noise/self-forcing/unrolling are documented to cost one-step
  accuracy. If horizon gained ≈ accuracy/compute lost, rollout-aware training *reshapes* the error
  budget rather than reducing it — a clean, citable null, not a failure to report.
- **Equal-data and equal-compute accounting is load-bearing and easy to get wrong.** DAgger sees the
  same drifted state-region repeatedly; the comparison is only fair at *equal total examples* (RS1) and
  *equal total compute* (RS6). The harness fixes both explicitly; a sloppy comparison would manufacture a
  spurious win.
- **`H_indep` / `η` caveats carry over from HS1.** `η>1` partly reflects measuring `p` on a more diverse
  held-out set than the rollout visits ([SPEC-10 §4.1](./SPEC-10.md)); the load-bearing number is
  `H_free` itself, and the *change* in `η` under a trainer (not its absolute value) is the compounding
  read.
- **Single-machine CPU caps the scale.** Like SPEC-10, the committed sweeps run on the M4 (the
  macOS-first gate); CI (`ubuntu-latest`) runs only smoke instances asserting structural invariants
  (a DAgger round strictly grows the example pool; the relabel target equals `O(s̃,a)` exactly;
  `sample_prob=0` reproduces teacher forcing byte-for-byte), not exact magnitudes — so the same tests
  pass on the primary host and the free Linux confirmation.

---

## 8. Build, reproduce, CI

### 8.1 Module layout (additive only)

```
src/verisim/netmodel/graph_train.py   # EXTEND — add train_unrolled (RS4); DAgger loop helper (RS1)
                                      #   (build_self_forced_examples / train_graph_model_self_forced
                                      #    / the noise branch already ship — SPEC-5 §6.3)
src/verisim/experiments/
  rs1_dagger.py … rs7_*.py            # NEW — the RS experiments (Config/from_json_file, CLI, JSONL)
figures/
  plot_rs1.py … plot_rs6.py          # NEW — committed-figure generators (png + csv)
configs/
  rs1_dagger.json … rs6_pareto.json  # NEW — committed sweep configs
```

The trainers consume the shipped `NetModel`/`NetUncertaintyModel` seam, the two network oracles, and the
NW5 loop **unchanged** — nothing in the deterministic core, the model, the metrics, or the inference loop
is edited. SPEC-16 is a change to *how `M_θ` is fit*, read on the *existing* `H_ε(ρ=0)` measurement.

### 8.2 `reproduce.sh` (new RS block, in dependency order)

```bash
echo "== RS2/RS3: shipped exposure-bias levers on the horizon axis (H57) =="
python -m verisim.experiments.rs2_scheduled --config configs/rs2_scheduled.json --out runs/rs2/records.jsonl --plot figures/rs2_sample_prob_tradeoff.png
python -m verisim.experiments.rs3_noise     --config configs/rs3_noise.json     --out runs/rs3/records.jsonl --plot figures/rs3_noise_surface.png
echo "== RS1: free-oracle DAgger vs teacher forcing (H55) — THE HEADLINE / the gate =="
python -m verisim.experiments.rs1_dagger    --config configs/rs1_dagger.json    --out runs/rs1/records.jsonl --plot figures/rs1_dagger_horizon.png
echo "== RS4: multi-step unrolled loss — the pushforward made exact (H55/H57) =="
python -m verisim.experiments.rs4_unrolled  --config configs/rs4_unrolled.json  --out runs/rs4/records.jsonl --plot figures/rs4_unroll_depth.png
echo "== RS5: rollout-aware training on the structured arm (H56) — the HS3 decider =="
python -m verisim.experiments.rs5_graph     --config configs/rs5_graph.json     --out runs/rs5/records.jsonl --plot figures/rs5_graph_floor.png
echo "== RS6: net faithful-horizon-per-compute Pareto (H58) — the honest verdict =="
python -m verisim.experiments.rs6_pareto    --config configs/rs6_pareto.json    --out runs/rs6/records.jsonl --plot figures/rs6_net_pareto.png
# RS7 (cross-world / cross-proposer fork) gated on RS1.
```

The RS block runs on CPU (the deterministic gate); CI runs the smoke instances only. Determinism is the
same regime as every other figure: a record stream is a deterministic function of (config, seeds).

---

## 9. Provenance & reading order

SPEC-16 is the training-time answer to the divergence SPEC-10 found: SPEC-5 §6.3 shipped the
exposure-bias *levers* (noise injection, self-forcing) and EN4 ran them once; SPEC-10 HS1.1 caught the
exposure-bias *signature* (per-step accuracy up, horizon down) and HS3 localized the compounding wall to
the structured arm; SPEC-16 runs the *cure* — free-oracle DAgger and the rollout-aware trainer family —
as a rigorous faithful-horizon study, and either lifts `H_free` (exposure bias was the gap) or banks the
fundamental-compounding negative (the wall is the model class). The architecture is DAgger and the
learned-simulator stability lineage (GNS, MeshGraphNets, Stachenfeld, Brandstetter, PDE-Refiner) and the
sequence-modeling exposure-bias line (scheduled sampling, professor forcing); the unique contribution is
the one thing that line structurally cannot do — **the expert is the oracle, free and exact at every
drifted state**, so DAgger's bottleneck vanishes and the cure can be run to convergence and measured
against an exact long-horizon ground truth (SPEC §2). It is a *method* spec: it advances how the program
*trains* its world model, not what the model is.

Reading order for a newcomer: [SPEC.md §2](./SPEC.md) (the oracle asymmetry) →
[SPEC.md §6](./SPEC.md) (delta-prediction + oracle-as-reward) →
[SPEC-10 §4.2](./SPEC-10.md) (HS1.1 — the exposure-bias signature) →
[SPEC-10 §4.6–4.8](./SPEC-10.md) (HS3 — the structured wall) → this document (§1 motivation, §3 the four
trainers, §4 the hypotheses) → [`netmodel/graph_train.py`](../../src/verisim/netmodel/graph_train.py)
(`build_self_forced_examples`, the shipped levers) and `src/verisim/experiments/rs1_dagger.py` (the
concrete build, once shipped).
