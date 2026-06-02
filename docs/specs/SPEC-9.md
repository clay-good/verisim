# SPEC-9 — The Free-Oracle Scaling Regime

**Cross-cutting scaling specification: because the oracle makes labels free and exact, world
size is a *learner-compute* choice, not a *labeling-budget* choice. How large and how deep can
an oracle-grounded world be made on a single commodity machine — and what holds as it grows?**

> **▶ SCALING SPEC — 2026-06-01.** A *cross-cutting* spec, sibling to [SPEC-4](./SPEC-4.md) (the
> engine) and [SPEC-8](./SPEC-8.md) (oracle-grounded SSL): not a new world, oracle, metric, or
> model, but a *regime* every world inherits. It does not invent a world — it instantiates the
> **existing Tier-A network oracle** (SPEC-5) as large and as deep as one machine allows, and
> measures the scaling behavior of the SPEC-8 results on it. Gated like everything: the envelope
> is *measured* before it is claimed, and an honest negative (a result that does **not** scale) is
> first-class.

Read [SPEC.md](./SPEC.md) for *why* the oracle is free and exact, [SPEC-5](./SPEC-5.md) for the
world being scaled, and [SPEC-8](./SPEC-8.md) for the H23–H25 results whose scaling this spec maps
(H23-S/H24-S/H25-S, SPEC-8 §7.1). This document is *how big the lab can get, locally, and what the
oracle's advantage does as the world grows*.

---

## 0. One-paragraph thesis

Every other domain pays for scale twice: a bigger world needs more *world* **and** more *labels for
it* — more annotated frames, more engineered reward, more human judgment — so the cost of studying a
larger world is bounded by the **labeling budget**. The deterministic oracle removes the second cost
entirely: a true, exact next-state label at every transition, for free, at any world size (SPEC.md
§2). The consequence, stated as this spec's thesis: **in the one domain with a free exact oracle, the
size and depth of the world a researcher can study is bounded only by the *learner's* compute, never
by the labeler's budget.** That makes the network world a controlled laboratory no oracle-free SSL
program can build — one where world size can be dialed across one to two orders of magnitude at zero
labeling cost, and the scaling behavior of oracle-grounded self-supervision (the SPEC-8 H23–H25 gaps)
measured directly. This spec maps that lab's local ceiling on a single 32 GB Apple-silicon machine,
and pre-registers what the oracle's advantage is expected to do as the world grows toward it.

---

## 1. The two scaling regimes

| | Oracle-free SSL (vision/language/RL) | **Oracle-grounded SSL (computer worlds)** |
|---|---|---|
| Labels for a bigger world | more annotation / reward engineering — **paid, per example, at every scale** | the oracle, **free and exact at any size** (data build is `O(seconds)` even at 400 hosts, §3) |
| Binding resource as the world grows | the **labeling budget** (and label *noise* compounding) | the **learner's compute** only (`O(N^2)` message passing, §3) |
| What sets the largest studyable world | how much labeling you can afford | how much compute the *one machine* has |
| Can you sweep world size at fixed cost? | no — each larger world re-pays labeling | **yes** — re-instantiate the oracle; labels stay free |

The right column is the regime this spec names. It is not a claim that the *model* is cheap (it is
not — §3 shows the `O(N^2)` wall); it is a claim that the *truth* is, and that this moves the
binding constraint from a budget you must raise to a machine you already own. That is why a scaling
study that would be a major labeling program in vision is, here, an afternoon on a laptop.

---

## 2. The scaling axes (grounded in SPEC-5's real dials — no new world)

Every axis below is an existing dial of the real network world; this spec only pushes them. Nothing
here is fabricated — the oracle, grammar, and semantics (SPEC-5 §3, [network-semantics](../network-semantics.md))
are unchanged, just instantiated larger.

- **Size — hosts, ports, services.** `NetConfig(hosts, ports)` (SPEC-5 §3.1), now generated at
  arbitrary count by [`scaled_net_config`](../../src/verisim/net/config.py). More hosts ⇒ a larger
  reachability graph and a harder *residual* (an edit's exact host/port identity is one of many,
  SPEC-8 §7.2).
- **Scale — trajectories, seeds, data volume.** Driver rollout length and seed count (SPEC-5 §3.2,
  NW2). Because the oracle is free, data volume is not a budget line; it is bounded only by the
  memory of the eval readout, which [`_embed_all`](../../src/verisim/netmodel/grounded_train.py)
  chunks so deep datasets at large host counts do not exhaust RAM.
- **Depth — topology diameter, flow-port richness, dependency chains.** The message-passing depth
  `mp_rounds ≈ network diameter` (SPEC-5 §6.1): a deeper topology (more tiers/hops between a change
  and where it is observed) needs more rounds to be representable, so *depth* is a genuine model+world
  dial, not just a size one. The v1 graph arm's documented gaps — flow edges featurized without their
  port, belief computed per-step rather than carried across a partial rollout (SPEC-5 §6.1–6.2 "v1
  scope") — are the **depth frontier**: the places where a deeper world is currently under-modeled,
  and therefore where a measured drop in the oracle's advantage at depth is informative, not noise.
- **Capacity — model size.** `d_model`/`mp_rounds`/`n_layer`/`n_head` via
  [`build_graph_model`](../../src/verisim/netmodel/graph_model.py). The axis SPEC-8 §7.2 holds fixed
  to test H24 and the one §4 below predicts must grow *with* world size to keep the H23 rank-gap and
  the H25 lift from compressing.

---

## 3. The measured local envelope (32 GB M4, CPU)

Measured directly (graph arm `d_model=128`, `mp_rounds=3`, 120-example train set; jepa 50-iter
proxy, peak RSS), so the presets below are grounded, not guessed:

| hosts `N` | vocab | data build | jepa 50-iter | ≈ jepa 400-iter | peak RSS |
|---|---|---|---|---|---|
| 50 | 97 | 0.06 s | 3.2 s | ~25 s | 351 MB |
| 100 | 147 | 0.13 s | 6.3 s | ~50 s | 492 MB |
| 200 | 247 | 0.46 s | 17.5 s | ~140 s | 779 MB |
| 400 | 447 | 1.76 s | 59.8 s | ~480 s | 1.95 GB |

Three findings that set the regime:

1. **Time is the binding resource, not memory.** Peak RSS is < 2 GB even at 400 hosts — the 32 GB
   machine is nowhere near a memory wall (and [`_embed_all`](../../src/verisim/netmodel/grounded_train.py)
   chunking keeps it that way for deep datasets). What binds is wall-clock: the dense `[B, N, N]`
   message passing is **`O(N^2)`** (the table's per-iter time roughly quadruples per host-doubling),
   so a single training run runs ~25 s at 50 hosts and ~8 min at 400.
2. **The oracle (the labeler) is effectively free at every size** — data build is under 2 s even at
   400 hosts, two-to-three orders of magnitude below the training time. This is §0's thesis made
   literal: the cost is entirely the learner.
3. **MPS does not help at this model/batch size** — it ran 2–3× *slower* than CPU (200 hosts: 17.5 s
   CPU vs 51 s MPS; 400 hosts: 60 s vs 128 s), because the model (~1 M params) and batch (32) are too
   small to amortize Apple's per-kernel launch latency across many tiny message-passing matmuls. CPU
   is both faster and bit-deterministic here, so it stays the default; the `--device` flag remains for
   completeness and for any future larger model where the trade flips.

**Local-maximal presets (the honest ceiling on this machine):**

- **Scaling-sweep preset** (many runs: world × model × seeds): `N ≤ 200` keeps each run ≤ ~2.5 min, so
  a full multi-seed surface is tens of minutes to ~2 hours — entirely local.
- **Hero-instance preset** (a single large committed point): `N` up to ~400–512 is feasible at
  single-digit minutes per training run; memory allows well beyond, but `O(N^2)` time makes it the
  practical wall without a GPU.
- **Depth** (trajectory length, seeds) is nearly *free in time* — `train_*` run a fixed step count, so
  a deeper/larger dataset costs only the chunked eval pass, not training — so depth is the cheap axis
  to push and the one to lean on for coverage.

---

## 4. Scaling-trend claims (sharpening H23-S/H24-S/H25-S; SPEC-8 §7.1)

The SPEC-8 §7.1 scale-up already produced disjoint-CI verdicts at 5/10/15 hosts; this spec carries
them up the envelope and pre-registers what the *trend* should be. Each is falsifiable and names its
honest negative.

- **S1 — the collapse gap is scale-stable once normalized (refines H23-S).** The raw eff-rank collapse
  gap shrinks with world size (+13.4 → +7.7 over 5→15 hosts) because effective rank is capped by
  `d_model`. The claim: the gap *normalized by `d_model`* (or the `emb_std` gap, which is already
  scale-free) stays positive and roughly stable as `N` grows, and the raw gap is recovered by scaling
  `d_model` with `N`. *Refuted if* even the normalized gap decays to zero with scale — the oracle's
  anti-collapse advantage would then be a small-world effect, a bankable correction.

  > **Result — H23-S survives scaling; S1's *stable* form does NOT** ([`en8_surface.csv`](../../figures/en8_surface.csv);
  > 25/50/100/200 hosts × `d_model` ∈ {64,128} × 3 seeds). The collapse gap is **disjoint-positive at
  > every one of the 8 cells** — so H23-S (the oracle removes the collapse tax) holds robustly across the
  > whole 8× world range and both capacities; this is the strongest of the three results. But it
  > **attenuates** with world size: raw eff-rank gap 13.4→13.2→6.9→4.1 at `d128` (and 7.5→6.1→5.3→3.2 at
  > `d64`), and the *normalized* gap declines too (≈0.105→0.032 at `d128`). Scaling `d_model` lifts the raw
  > gap at fixed world (`d128 > d64` everywhere) but does **not** flatten the decline. So S1's specific
  > "scale-stable once normalized" prediction is refuted: the honest claim is **persistent but
  > attenuating** — the oracle's anti-collapse advantage is real at every scale tested, and shrinking.
- **S2 — the interventional lift's non-monotonicity is a capacity artifact (refines H25-S).** The
  oracle-over-VICReg lift was positive and disjoint at 5/10/15 hosts but *non-monotone* (peaked at 10).
  The claim: holding training adequate and scaling `d_model` with `N` restores monotone (or at least
  non-decreasing) lift; the dip is the larger world undertraining at fixed small capacity, not a real
  ceiling on what counterfactual negatives buy. *Refuted if* the lift stays non-monotone or narrows
  with capacity — then exact negatives have a genuine large-world limit worth mapping.

  > **Result — S2 REFUTED, and the lift *reverses* at scale (the most important negative so far)**
  > ([`en9_surface.csv`](../../figures/en9_surface.csv); 25/50/100/200 hosts × `d_model` ∈ {64,128} × 3
  > seeds). The oracle-over-VICReg interventional lift (top-1) is disjoint-positive **only** at the
  > smallest world and smaller capacity (25 hosts/`d64`: +0.106 [0.067, 0.179]); it **decays with world
  > size and reverses** — at higher capacity it goes disjoint-**negative**, so VICReg *beats* the oracle
  > on branch retrieval (100 hosts/`d128`: −0.086 [−0.113, −0.060]; 200/`d128`: −0.094 [−0.111, −0.067]).
  > Capacity does the *opposite* of what S2 predicted: more `d_model` makes the lift worse at scale, not
  > monotone. **So the headline EN9/H5 result (a ~2× lift at 5 hosts) is scale-fragile — it does not
  > survive to 100–200 hosts at `d128`.** The most likely mechanism is a design artifact, not a death of
  > the idea: `k_negatives` is fixed at 8 while the *counterfactual branch space grows with hosts*, so the
  > oracle's InfoNCE negatives become an ever-sparser sample of the branches it must separate, while
  > VICReg (which needs no negatives) is unaffected. The pre-registered next test: **scale `k_negatives`
  > with the branch space** and re-measure. Until then, H25-S/H5 is **confirmed at small scale, refuted at
  > large scale + high capacity** — a sharp, bankable bound on a result the README/report had presented as
  > clean, and exactly why scaling with CIs was worth doing.
- **S3 — the H24 capacity-binding frontier exists and is locatable (sharpens H24-S).** Residual-objective
  supervision was a CI-bounded tie at full capacity. The claim: there is a locatable frontier in the
  (world size, model capacity, observed-fraction) space where the residual gap turns *positive with a
  disjoint CI* — specifically where capacity binds against a hard residual `R` (SPEC-8 §7.2). *Refuted
  if* no such frontier appears anywhere in the local envelope — the strong form of the H24 negative:
  masking the decidable bits never pays at any scale one machine can reach, a sharp, scale-resolved bound.

  > **Result — S3 is REGIME-DEPENDENT, with a mechanism.** Two complementary sweeps, read together:
  > the capacity frontier ([`en8_capacity.csv`](../../figures/en8_capacity.csv); 40-host world × `d_model`
  > ∈ {16,32,64} × observed-fraction ∈ {0.25,0.5,0.75} × 4 seeds) finds **no** disjoint-positive cell and
  > a disjoint-**negative** gap where the decidable part `D` is large (observed-fraction 0.75, `R` ~11% of
  > tokens; `d64`: −0.094 [−0.130, −0.057]). But the scaling surface
  > ([`en8_surface.csv`](../../figures/en8_surface.csv)), which the frontier sweep did not cover at
  > `d_model=128`, finds a small but **disjoint-positive** gap at higher capacity + smaller world +
  > moderate `R` (25 hosts/`d128`: +0.038 [0.013, 0.064]; 50/`d128`: +0.027 [0.023, 0.035]), vanishing by
  > 100–200 hosts. So masking `D` is **not uniformly useless: it helps narrowly (high capacity, moderate
  > `R`, small world) and hurts where `R` is tiny.** The mechanism is the tension: masking `D` *removes
  > training signal* (the model is supervised on only the R-fraction of tokens/step, starving the shared
  > encoder/decoder — learning `D` is **beneficial multi-task auxiliary signal**), and this cost dominates
  > the capacity-allocation benefit except in the narrow regime where capacity is ample and `R` is large
  > enough to matter. The earlier reading ("refuted") was too strong — it generalized the `d≤64` frontier
  > past the `d128` cells it never tested. **What is genuinely bounded is the *training-objective* form of
  > the partition (mask `D` in the loss); the *inference-time* partition — the oracle *supplies* `D` so the
  > model is never trusted on it (DD-OG-3) — is untouched.** Next variant: keep `D` in the loss (the
  > auxiliary signal helps) and let the oracle own `D` only at inference, as the loop already does.

These reuse the SPEC-8 §7.1 harness ([`en8_scale`](../../src/verisim/experiments/en8_scale.py),
[`en9_scale`](../../src/verisim/experiments/en9_scale.py)) extended along the **model-size axis** (S1/S2),
plus the dedicated [`en8_capacity`](../../src/verisim/experiments/en8_capacity.py) frontier sweep (S3),
with the same bootstrap-CI discipline and the regenerate-from-seed rule. Negatives are first-class.

---

## 5. Milestones (LS0–LS3)

Non-colliding with `M*/S*/AR*/NW*/HC*/DS*/OG*`. Gated as ever: measure the envelope before claiming it.

| Milestone | What | Verify | Status |
|---|---|---|---|
| **LS0** | The envelope, measured: time/memory vs host count, CPU vs MPS, the `O(N^2)` law and the memory-not-binding finding (§3). | the numbers regenerate from a probe script on the target machine | ✅ (§3 table) |
| **LS1** | The enablers: configurable world size + model size threaded through EN8/EN9 ([`scaled_net_config`](../../src/verisim/net/config.py), the [`build_graph_model`](../../src/verisim/netmodel/graph_model.py) knobs), the SVD-on-CPU MPS fix, and the chunked [`_embed_all`](../../src/verisim/netmodel/grounded_train.py) eval (deep datasets at large `N` without OOM). | ruff/mypy clean; harness tests + full suite green; deterministic on CPU | ✅ |
| **LS2** | The local-maximal **scaling surface**: EN8/EN9 over world size × model size × seeds up to the §3 sweep preset, committed CSV + scaling-curve figures with CIs, testing S1/S2. | the surface regenerates from config + seeds; S1/S2 verdicts populated with bootstrap CIs | ✅ shipped (25–200 hosts × {d64,d128} × 3 seeds, [`en8_surface.csv`](../../figures/en8_surface.csv) / [`en9_surface.csv`](../../figures/en9_surface.csv); ~69 min + ~104 min CPU): **S1 — H23-S persists but attenuates** (collapse gap disjoint-positive at all 8 cells, declining with scale); **S2 — refuted and reversed** (the H25-S/H5 lift is scale-fragile: disjoint-negative at 100–200 hosts/`d128`) (§4) |
| **LS-S3** | The **H24 capacity-binding frontier** ([`en8_capacity`](../../src/verisim/experiments/en8_capacity.py)): residual gap over `d_model` × observed-fraction at a fixed hard world, with CIs — the dedicated S3 test. | the frontier regenerates from config + seeds; S3 verdict populated | ✅ shipped ([`test_en8_capacity.py`](../../tests/test_en8_capacity.py), [`en8_capacity.csv`](../../figures/en8_capacity.csv)): **S3 refuted with a mechanism** — masking `D` removes beneficial training signal; the *training-objective* partition does not pay, the *inference-time* partition stands (§4) |
| **LS3** | A **hero instance** at the §3 hero preset (`N` ~400–512), single large committed point, as the "largest oracle-grounded world proven on one machine" datum. | regenerates from config + seed; reported with its honest caveats (single point, capacity-limited) | ☐ planned |

---

## 6. Safety, ethics, and the honest-negative posture

Inherited wholesale from SPEC.md §13, SPEC-5 §15, and SPEC-8 §10: defensive-only framing, no real-internet
egress, MIT, no telemetry, the oracle/metric/goldens/gate in the denylist (DD-AR2). Scaling changes only
*world instantiation size and model capacity* — it adds no capability the SPEC-5 world does not already
scope, and it never edits the judge.

The honest-negative posture is the whole point of pre-registering S1–S3 before the surface is run. A
clean positive (the oracle's advantage holds or grows up the envelope) strengthens the SPEC-8 contribution
from "true at smoke scale" to "true and scale-robust on the largest world one machine can hold." But the
negative — "the advantage is a small-world effect that decays with scale" — is equally bankable and
arguably sharper, because the oracle makes the verdict *trustworthy* at every size: we are not guessing
whether a result generalizes, we are measuring it against exact truth across a 10–20× world-size range at
zero labeling cost. That measurement is something the oracle-free SSL field structurally cannot make, and
it is the reason this regime is worth mapping regardless of which way the curves fall.

---

## 7. Provenance and reading order

- **Read before:** [SPEC.md](./SPEC.md) (why the oracle is free and exact), [SPEC-5](./SPEC-5.md) (the
  world scaled here), [SPEC-8](./SPEC-8.md) (the H23–H25 / H23-S–H25-S results whose scaling this maps).
- **Sibling, not a world:** like SPEC-4 and SPEC-8, a cross-cutting regime every world inherits.
- **Claims:** S1–S3 (§4) sharpen SPEC-8's H23-S/H24-S/H25-S along the model-size axis; they coin no new
  global `H` number.
- **Status:** LS0–LS1 shipped (the envelope is measured, the enablers are in and green); LS2–LS3
  (the scaling surface and the hero instance) are pre-registered here and run next. Nothing beyond the
  committed figures is believed until it is run.
- **Author:** Clay Good. **License:** MIT. No telemetry, no commercial path — a research repo.
