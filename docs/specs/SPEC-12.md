# SPEC-12 — The Oracle-Grounded Landmark Graph: Planning Over a Faithful World Model

**World-method specification: every figure to date measures how *faithful* a learned world model stays
under oracle budget. None of them *plans*. This spec adds the planning layer the program has been
building toward — a sparse graph of *landmarks* (waypoint states) whose edges are *reachability*,
scattered over the network world, with the one property no other landmark-planner can have: **every
edge is verified by the oracle.** It is the direct architectural answer to the wall
[SPEC-10](./SPEC-10.md) found — the structured arm's free-running horizon is pinned at zero (HS3) — by
the move the goal-conditioned-RL literature invented for exactly that failure: don't roll the dynamics
forward step by step (it diverges); plan landmark-to-landmark and execute only short, trustable hops.**

> **▶ PLANNING SPEC — 2026-06.** A *world-method* spec in the lineage of [SPEC-8](./SPEC-8.md)
> (oracle-grounded self-supervision) and [SPEC-10](./SPEC-10.md) (the faithful-horizon scaling law):
> it invents **no new world** — it runs on the SPEC-5 network world and its shipped graph arm — and
> **no new oracle** — it reuses the data-plane `ReferenceNetworkOracle` and the Batfish-style
> `ControlPlaneOracle` already built for EN10. What it adds is a *planning altitude* above the
> propose–verify–correct loop: the loop keeps a single rollout faithful; SPEC-12 composes many short
> faithful rollouts into a long-range plan over a verified graph. Inspiration is acknowledged plainly:
> the architecture is L3P (*World Model as a Graph: Learning Latent Landmarks for Planning*, Zhang et
> al., ICML 2021) and its goal-conditioned-RL lineage (SoRB, SPTM, SFL), and a conversation with
> practitioners building security world models who frame network traversal as a robotics-style planning
> problem (high-level plan over a sparse map, not pixel-by-pixel rollout). The verisim contribution is
> the one thing that line structurally cannot do: **make the graph's edges *exact* instead of
> *distilled-and-hoped*, because computer worlds have an oracle and continuous-control worlds do not.**

Read [SPEC.md §2](./SPEC.md) (why the oracle is free and exact), [SPEC-5 §5.1](./SPEC-5.md) (the
two-plane oracle and the control-plane reachability oracle this spec plans over),
[SPEC-10 §4.6–4.10](./SPEC-10.md) (HS3 — the structured-arm compounding wall this spec is the escape
from), and the shipped plan-level loop in [`hostsim/simulator.py`](../../src/verisim/hostsim/simulator.py)
and [`distsim/simulator.py`](../../src/verisim/distsim/simulator.py) (`imagine`/`verify`/`PlanReport` —
the low-level controller SPEC-12 plans on top of).

---

## 0. One-paragraph thesis

The program's headline finding from SPEC-10 is a wall: the *structured* (GNN+RSSM) proposer's
free-running faithful horizon `H_ε(ρ=0)` is **pinned at zero** at exact tolerance across capacity,
data, world-size, and their joint product (HS3, [SPEC-10 §4.6–4.10](./SPEC-10.md)) — a genuine
compounding ceiling, not a resourcing artifact. Read naively that retires the structured arm. Read
correctly it *names the next move*: a model that cannot be rolled forward many steps must instead be
asked only the questions it answers well, over short spans it can sustain. Verisim has independently
measured *which* question it answers well — **reachability**: the network model predicts the
reachability *effect* of an action far better than the exact delta (finding 11/12, EN6/EN10), and a
reachability consult costs ~35% of a full one and is decision-sufficient ~30% of the time the full
delta is wrong (H12). That is *exactly* the edge metric L3P uses to wire a landmark graph. So SPEC-12
builds the **oracle-grounded landmark graph**: waypoint states as nodes; edges = reachability, the
signal the model is strong on; **every edge verified by the oracle** (the data-plane oracle for exact,
the control-plane oracle for cheap), the verification no continuous-control landmark-planner can
perform; and a two-level planner that hops landmark-to-landmark (high level) while the existing
`imagine`/`verify` loop executes each hop within its faithful horizon (low level). The claim, stated to
be falsified: **a verified landmark graph converts the structured arm's zero free-running horizon into
long-range goal reach under a bounded oracle budget — structure buys *goal-space* horizon where it
could not buy *step* horizon — and it does so because the graph plans on the reachability projection
the model is faithful on, not the exact-state rollout it is pinned at.** Every branch is bankable
(SPEC §10.1): if the graph fails to plan, the cliff is the loop's and not the rollout strategy's; if it
succeeds, the program has its first long-horizon planner *and* a defensive artifact — an oracle-verified
attack/reachability graph with zero false paths, the standing weakness of every attack-graph tool.

---

## 1. Why now: the wall SPEC-10 found and the escape L3P names

The faithful-horizon program has produced one shape across every world and proposer: a **floor + cliff,
no favorable knee** (EN1, EH1, ED-loop; the shape is the loop's, not the model's — EN7/H22). SPEC-10
then asked whether the floor is a scale artifact and found a *proposer-dependent* answer: the flat
transformer's floor dissolved into a resourcing story (HS1→HS1.3, best free-running horizon ~19 id /
~29 ood steps at a compute-optimal sweet spot), but the **structured GNN+RSSM arm's floor is a genuine
wall** — `H_free ≈ 0` at exact tolerance across capacity (HS3 incr 1), data (incr 2), world size
(incr 3), and the joint compute-optimal ladder (incr 4), with `η < 1` (it free-runs *shorter* than its
own i.i.d. prediction — the compounding penalty the flat arm never paid).

This is the precise condition under which the goal-conditioned-RL field abandoned free-running rollout.
The L3P framing (Zhang et al., ICML 2021): *step-by-step model rollouts diverge over long horizons, so
plan sparse multi-step transitions between learned waypoints instead.* SoRB (Eysenbach et al., 2019)
and SPTM (Savinov et al., 2018) made the same move earlier — waypoints from the replay buffer / from
observed states, edges from a learned reachability/value estimate, graph search for the route, a
goal-conditioned policy for each leg. **The structured arm's `H_free = 0` is not a dead end; it is the
exact precondition that makes the landmark layer necessary rather than optional.** A model that can be
trusted for *k* steps but not *k+1* is precisely a low-level controller whose competence radius is *k* —
and a landmark graph is how you compose competence radii into a long plan.

Two facts make verisim the right place to build it, and one of them is unique to this program:

- **The model's strong signal is already the right edge metric.** L3P distills edges from a Q/value
  function as reachability estimates. Verisim *predicts reachability directly and well* (EN6/EN10), and
  has a free symbolic reachability ground truth ([`ControlPlaneOracle`](../../src/verisim/netoracle/control_plane.py)).
  The edge an L3P author must *learn and hope* is, here, *predicted accurately and checked exactly.*
- **The edges can be verified — the move no oracle-free planner can make.** Every landmark-graph paper
  lives with edge error: a hallucinated edge sends the planner down an unreachable route, and there is
  no ground truth to prune it. Verisim's oracle prunes it for free. The deliverable is a **faithful
  landmark graph** — the rigorous version of the artifact, and the one the visual/continuous-control
  field is structurally barred from producing.

---

## 2. The lineage folded in (and the design choice each one forces)

### 2.1 Landmark-graph planning (the architecture)

- **L3P — World Model as a Graph** (Zhang et al., ICML 2021, arXiv:2011.12491). Latent landmarks via
  clustering in a learned latent space; edges = reachability distilled from Q-learning; latent distance
  ≈ reachability; hierarchical plan (high-level landmark hops + low-level goal-conditioned controller);
  re-plan only when reachability says the subgoal is reached/failed, not every step. → **Design
  choices, almost one-to-one:** landmarks = sampled network *states* clustered in the EN8 `embed()`
  latent ([`GraphRSSMNet.embed`](../../src/verisim/netmodel/graph_model.py)); edges = reachability,
  *verified by the oracle* rather than distilled from a value function; controller = the shipped
  `imagine`/`verify` loop; replanning = reachability-triggered (§6, LP6).
- **SoRB — Search on the Replay Buffer** (Eysenbach et al., NeurIPS 2019). Waypoints = buffer states;
  edge weight = the value function's predicted distance; Dijkstra for the route. → **Design choice:**
  the cheap baseline graph — landmarks drawn from the EN eval drivers' visited states, edges weighted by
  predicted reachability, classical shortest-path on top.
- **SPTM** (Savinov et al., ICLR 2018) and **SFL — Successor Feature Landmarks** (Hoang et al., NeurIPS
  2021). Non-parametric graph + a learned reachability/similarity metric deciding which states become
  nodes. → **Design choice:** the *landmark admission* rule (which sampled states to keep) is itself a
  measured knob (§6, LP5), not a fixed heuristic.

### 2.2 LLMs are bad at querying graphs (the division of labor)

- **NLGraph** (Wang et al., NeurIPS 2023) and **Talk like a Graph** (Fatemi et al., Google, ICLR 2024).
  LLM graph-reasoning accuracy degrades with traversal depth and with per-node degree; advanced
  prompting helps less as problems deepen; a positive-answer bias makes models assert paths that do not
  exist. → **Design choice:** the LLM is confined to the *leaves* — natural-language intent → a single
  hop's action plan, and exploitation/explanation at a target — exactly as both the practitioner notes
  and verisim's own §20 (LLM = NL intent → syscall plan) already place it. The *traversal* is done by
  graph search over the verified graph, server-side, never by the LLM hop-by-hop. SPEC-12 measures this
  boundary rather than asserting it (§6, LP7): the depth-vs-accuracy curve of an LLM walking the raw
  graph, against a planner-fed LLM that only ever sees one leaf.

### 2.3 Attack graphs and network reachability (the defensive application)

- **MulVAL** (Ou et al., USENIX Security 2005) and the attack-graph / lateral-movement-as-graph-walk
  line. Reachability + exploit preconditions compose into multi-host attack paths; the field's standing
  pain is *unsound* graphs (paths that the real network does not actually permit), because the graph is
  derived from a static config model, not a verified one. → **Design choice:** the oracle-grounded
  landmark graph *is* an attack/reachability graph whose every edge the data-plane oracle confirms — so
  it has **zero false paths by construction**, and high-value targets / chokepoints are the landmarks the
  placement policy surfaces (§6, LP5). This is the SPEC §4.1 ACD contribution made concrete and
  defensive (the program's stance: defensive, oracle-grounded, human-out-of-loop — never a shipped
  offensive agent, SPEC §13).

### 2.4 What the program already built that this sits on

- **The plan-level loop exists.** [`hostsim`](../../src/verisim/hostsim/simulator.py) /
  [`distsim`](../../src/verisim/distsim/simulator.py) already expose `imagine(state, plan)` (roll `M_θ`
  forward, no oracle) and `verify(state, plan)` (check against the oracle on a budget, returning a
  `PlanReport` with the **plan-faithful horizon**). SPEC-12 lifts this from a *given* plan to a
  *graph-search-produced* plan: the high-level planner emits the plan, the existing loop executes and
  verifies each hop.
- **The latent and the reachability metric exist.** `embed()` is the EN8 representation (the candidate
  landmark space); `reachability_matrix` / `ControlPlaneOracle` are the edge metric and its free ground
  truth. SPEC-12 adds the graph and the search; it builds almost no new primitive.

---

## 3. The architecture: the oracle-grounded landmark graph (four layers)

```
                      ┌─────────────────────────────────────────────────────────────┐
   LLM (leaves only)  │  NL intent ─▶ goal state g ;  per-hop: leaf subgoal ─▶ action │  §2.2, §20
                      └─────────────────────────────────────────────────────────────┘
                                            │ goal g                  ▲ one leaf at a time
                                            ▼                         │
   High-level planner   graph search (Dijkstra/A*) over the VERIFIED landmark graph
                        nodes = landmarks ℓ_i (waypoint states, clustered in embed() latent)
                        edges = reachability(ℓ_i → ℓ_j), ADMITTED iff oracle-verified AND
                                hop length ≤ local faithful horizon H_ε
                                            │ subgoal sequence ℓ_a → ℓ_b → … → g
                                            ▼
   Low-level controller  imagine(ℓ_i, hop_plan)  ‖  verify against oracle on budget ρ   (the shipped loop)
                         each hop bounded by H_ε; re-ground at hop boundaries (reachability-triggered)
                                            │
                                            ▼
   Oracle (ground truth)  data-plane ReferenceNetworkOracle (exact)  +  ControlPlaneOracle (cheap reach)
```

Four commitments, each tied to a measured verisim fact:

1. **Edges are reachability, not exact state.** The graph is wired on the projection the model is
   faithful on (EN10), not the exact-state rollout it is pinned at (HS3). This is the load-bearing
   design inversion: build the planner on the model's *strength*.
2. **Edges are oracle-verified.** A model-proposed edge is admitted only if the oracle (data-plane for
   exact, control-plane for cheap) confirms it. The verified graph is the deliverable; the unverified
   one is the baseline we measure the gap against (§6, LP2).
3. **Edge length is bounded by the faithful horizon.** An edge whose hop exceeds the local `H_ε` is
   rejected — the planner only commits to legs the low-level loop can actually sustain. `H_ε` is a
   *region* property (EN7/H22: the shape is the loop's), so the radius is measured per region, not
   assumed constant.
4. **The LLM never traverses.** Graph search does the multi-hop reasoning; the LLM sees one leaf
   (§2.2). This is the practitioner's founding insight and the NLGraph result, made a measured boundary.

---

## 4. The load-bearing assumption (and its free fallback)

L3P scatters landmarks in a learned latent space and trusts that **latent distance ≈ reachability**.
That assumption is the single thing the whole architecture rests on, and verisim can *test* it rather
than assume it — which is the first experiment (LP1). The honest pre-registration:

- **If `embed()` latent distance tracks oracle reachability**, the L3P recipe transfers directly:
  cluster in the latent, scatter landmarks, weight edges by latent distance, verify with the oracle.
- **If it does not**, verisim has a fallback no continuous-control planner has: **the control-plane
  oracle gives a verified reachability metric *for free*** (`reachability_matrix`). The landmark graph
  is then built directly in reachability space — skip the latent entirely, scatter landmarks to cover
  the reachability matrix, edges are exact by construction. Either branch yields a graph; the branch
  point only decides whether the *latent* is load-bearing, which is itself a clean result about the EN8
  representation (does it encode the planning-relevant geometry, or only the one-step-prediction
  geometry?).

This is the program's epistemic engine applied at design time: the assumption that sinks every
oracle-free landmark planner is, here, a measurement with a banked alternative on the failing branch.

---

## 5. Hypotheses (pre-registered, continuing the global H-ID space past SPEC-11's H30)

- **H31 — the latent encodes planning geometry.** In the EN8 `embed()` latent, distance between two
  network states correlates with their true oracle transition-distance (BFS over the action graph) and
  with control-plane reachability, strongly enough to seed landmark clustering (target: Spearman ρ ≥ 0.6
  on held-out state pairs). *Refuted if* latent distance is uncorrelated with oracle distance — in which
  case the EN8 representation captures one-step-prediction geometry but not planning geometry, and
  SPEC-12 proceeds in reachability space directly (§4 fallback). Tested as **LP1**. **Result —
  REFUTED, the §4 fallback branch taken ([`lp1`](../../src/verisim/experiments/lp1.py),
  [`landmark/geometry.py`](../../src/verisim/landmark/geometry.py), 24 anchors, 5-host world):
  `embed()` latent distance correlates only weakly with the exact action-graph BFS geodesic
  (Spearman ρ = 0.27, bootstrap CI [0.24, 0.32]) and barely at all with control-plane reachability
  distance (ρ = 0.10, CI [0.05, 0.16]) — both far below the ρ ≥ 0.6 bar. The EN8 representation
  encodes *one-step-prediction* geometry, not *planning* geometry: a clean, bankable negative that
  decides the architecture. Notably even the two free ground truths agree only weakly with each
  other (geodesic↔reachability ρ = 0.23) — a transition can move far in action-steps while changing
  little reachability, and vice versa — which is itself why the graph must be wired on the
  reachability projection the model is strong on, not the latent. SPEC-12 therefore builds the
  landmark graph **directly in reachability space** (the control-plane metric, free and exact), and
  the *latent-landmark* framing weakens to a *reachability-landmark* one (the honest §10 caveat
  realized). The oracle is what let us measure — rather than assume — the load-bearing L3P
  assumption, and find it false in this world.**

- **H32 — the model's edges are unreliable but the oracle makes the graph exact cheaply.** Edges
  proposed by the model (latent distance / predicted reachability) have materially imperfect precision
  against the oracle (we expect false edges — the HS3 reading), *and* verifying every candidate edge
  with the control-plane oracle yields an **exact** graph at a consult cost far below verifying with the
  full data-plane oracle (the H12 ~3.6× ratio, lifted to edges). *Refuted if* model edges are already
  near-exact (the verification is redundant — a strong, surprising positive about the structured arm) or
  if control-plane verification misses real edges the data-plane catches (reachability is not a
  sufficient edge oracle here). Tested as **LP2**. **Result — SUPPORTED, decisively on the first
  clause ([`lp2`](../../src/verisim/experiments/lp2.py),
  [`landmark/build.py`](../../src/verisim/landmark/build.py),
  [`landmark/verify.py`](../../src/verisim/landmark/verify.py); landmarks = reachability-distinct
  states, an edge = a reachability-changing hop; 8 difficulty×seed cells). The structured arm's
  *hoped* graph is overwhelmingly wrong: edge precision **0.11** (CI [0.03, 0.21]), edge recall
  **0.08**, and a **false-edge rate of 0.77** (CI [0.53, 0.95]) — of the edges the model proposes
  confidently onto a known landmark, three-quarters point at the *wrong* one. This is the HS3 wall
  read at edge altitude: a structured arm that cannot free-run an exact rollout cannot propose an
  exact reachability edge either. The oracle then makes the graph exact: control-plane verification
  prunes **every** false edge — **verified residual false-edge rate = 0.000** (the SPEC-12 §8
  zero-false-paths guarantee, now a measured fact) — at a consult cost of **0.62×** the full
  data-plane consult (cheaper and *sufficient*, since an edge is a reachability claim; the ~3.6×
  ratio of H12's larger world attenuates to ~1.6× here, the honest world-dependent number). The
  reading licenses the rest of SPEC-12: the model's graph is unusable raw (you must verify), and the
  verified graph is the faithful-graph artifact — the MulVAL-unsoundness fix (§2.3) made a number.**

- **H33 — the verified landmark graph converts zero free-running horizon into long-range goal reach
  (THE headline).** To a distant goal, landmark-hop planning with per-hop `imagine`/`verify`
  re-grounding reaches the goal at a goal-space distance far beyond the structured arm's `H_free ≈ 0`
  cliff (HS3), under a bounded oracle budget — i.e. *structure buys goal-space horizon where SPEC-10
  proved it cannot buy step horizon.* *Refuted if* landmark planning does not beat flat free-running at
  equal oracle budget — in which case the cliff is a property of the loop irrespective of rollout
  strategy (a deep, bankable negative that would close the planning line on this world). Tested as
  **LP3**. **Result — SUPPORTED ([`lp3`](../../src/verisim/experiments/lp3.py),
  [`landmark/plan.py`](../../src/verisim/landmark/plan.py),
  [`figures/lp3_goal_reach.png`](../../figures/lp3_goal_reach.png); 6 difficulty×seed cells, the
  64-d structured arm). Two arms run on the *same* model and the *same* trajectory, differing only in
  re-grounding. As the goal-space distance `G` grows, flat free-running's goal reach **decays** —
  0.50 → 0.33 → 0.17 from `G = 4` to `G = 20` (the HS3 cliff at reachability altitude: error
  compounds) — while landmark planning, re-grounding once per `L = 4`-step hop (`ρ ≈ 0.2`),
  **sustains and rises**: 0.50 → 0.67 → **0.83**. At the far goal the gap is **5×** (0.83 vs 0.17),
  and it *widens* with distance — the signature of compounding-vs-re-grounding, not a constant offset.
  The budget sweep confirms the mechanism: at fixed `G = 16`, goal reach climbs monotonically with the
  re-grounding budget — flat (`ρ = 0`) 0.17, then 0.33 (`ρ = 1/8`), 0.67 (`ρ = 1/4`), 0.83 (`ρ = 1/2`)
  — so the verified graph buys goal-space horizon *in proportion to* the cheap oracle budget it spends.
  At `G = L` (a single hop, no interior boundary) landmark *equals* flat at `ρ = 0` — the honest
  control showing the win comes from re-grounding, nothing else. The model's own free-run reachability
  horizon stays ~3–7 steps (`reach_horizon`), which is exactly why hops must re-ground. So the
  structured arm whose *step* horizon SPEC-10 pinned at zero acquires a long *goal-space* horizon once
  short, faithful hops are composed over the verified graph — the SPEC-12 thesis, measured. The
  magnitudes are at the small fast-CI scale; the separation and its distance-widening shape are the
  result.**

- **H34 — reachability edges beat exact-state edges (the EN10 finding, lifted to planning).** A graph
  whose edges are the *reachability* projection plans successfully where a graph whose edges are the
  *exact-delta* prediction fails, because the model is faithful on the former (EN10) and pinned on the
  latter (HS3). *Refuted if* exact-state edges plan as well — meaning the structured arm's exact-rollout
  failure does not propagate to edge construction. Tested as **LP4**. **Result — SUPPORTED
  ([`lp4`](../../src/verisim/experiments/lp4.py),
  [`figures/lp4_edge_metric.png`](../../figures/lp4_edge_metric.png); the same LP3 planner and the
  same re-grounding boundaries, graded under two edge projections). As goal-space distance `G` grows,
  the **reachability-edge** arm sustains and rises (goal reach 0.50 → 0.83) while the
  **exact-state-edge** arm collapses (≈0, far-goal 0.33) — a 5× far-goal gap. The mechanism is the
  per-step horizon: the model's *exact-state* free-run horizon is **pinned at 0** at every distance
  (the HS3 wall at exact tolerance — it never lands even step 0's bit-for-bit state), while its
  *reachability* horizon climbs to ~7 steps (EN10). So the structured arm's exact-rollout failure
  *does* propagate to edge construction — wiring edges on the exact-delta prediction yields a graph
  that cannot plan — and the verisim move (wire on the reachability projection the model is faithful
  on) is what makes the planner work. The EN10-over-HS3 inversion, confirmed at planning altitude.**

- **H35 — chokepoint + uncertainty placement beats random (the "interesting points").** Landmarks
  placed at high reachability-betweenness (chokepoints) and high RSSM belief-variance (where the model
  is least sure) give higher goal-reach per landmark than random placement — the measured form of the
  "interesting points" an analyst or a security world model flags. *Refuted if* random placement is
  within CI of the informed policy at this scale (placement is not load-bearing here — the floor-type
  negative, banked, and a signal to revisit at larger worlds). Tested as **LP5**. **Result — a
  SPLIT verdict, the honest mix the finding ([`lp5`](../../src/verisim/experiments/lp5.py),
  [`landmark/placement.py`](../../src/verisim/landmark/placement.py),
  [`figures/lp5_placement.png`](../../figures/lp5_placement.png); LP5 refines LP3 — the candidate
  re-grounds are the uniform hop boundaries, and the budget keeps the top-`k` by placement score, so
  full budget recovers LP3's landmark arm and the question is purely *which* boundaries to keep).
  **The two informed signals diverge.** *Belief-variance* (uncertainty — where the model is least
  sure) is the load-bearing one: it reaches the LP3 goal-reach ceiling with fewer landmarks than
  random (mean advantage **+0.10**, beats random 2/5 budgets, loses 0/5). *Betweenness* (chokepoints)
  **under**performs random here (adv **−0.20**, loses 3/5) — at this small world the chokepoint signal
  is not where the model drifts. So placement *is* load-bearing, but it is the model's own uncertainty,
  not graph centrality, that carries it — a sharper claim than H35 pre-registered. Honest caveat: the
  bootstrap CIs overlap at this fast-CI scale, so this is *leaning supported for uncertainty*, banked
  and flagged to revisit at larger worlds (the pre-registered floor-type branch, with the surprise that
  the uncertainty signal, not betweenness, is the survivor). It pairs with LP6's mirror result.**

- **H36 — reachability-triggered replanning beats fixed-interval (the RQ2 axis, at plan altitude).**
  Re-planning when the reachability signal says a subgoal is reached or has become unreachable beats
  fixed-interval replanning at equal oracle budget — the EH2 lesson (calibrated belief beat fixed where
  flat entropy could not) carried to the planner. *Refuted if* fixed-interval ties or wins (the
  ED2-smart negative, lifted) — banking that the trigger signal is not calibrated at plan altitude.
  Tested as **LP6**. **Result — SUPPORTED for the reachability trigger, and the mirror of LP5
  ([`lp6`](../../src/verisim/experiments/lp6.py),
  [`figures/lp6_replanning.png`](../../figures/lp6_replanning.png); three triggers spending the same
  budget `B`, read from a single undisturbed free-run so the budget is equal by construction). The
  **reachability-triggered** policy — re-ground at the steps where the model's *predicted reachability*
  changes most (the online "subgoal reached/became unreachable" signal) — beats fixed-interval
  (mean advantage **+0.13**, beats fixed 2/5, loses 0/5) and pinpoints the single critical re-ground
  at `B = 1`, reaching the ceiling fixed-interval needs `B = 3` for. The **belief-variance** trigger,
  by contrast, is *miscalibrated* at plan altitude (adv **−0.20**, loses 4/5) — the exact mirror of
  LP5, where belief-variance was the *good* static-placement signal and betweenness the bad one. So
  the calibrated signal is signal-dependent: *uncertainty* for **where** to keep a landmark (LP5),
  *reachability-change* for **when** to re-plan (LP6). The EH2 "smart beats dumb" lift carries to the
  planner, on the right trigger. Honest caveat: CIs overlap at this scale — leaning supported, banked.**

- **H37 — the planner is why you keep the LLM at the leaves.** An LLM asked to traverse the raw graph
  (find a path A→B) degrades with hop count and node degree (the NLGraph/Talk-like-a-Graph result),
  while an LLM fed only the planner's leaf subgoals stays accurate independent of total path length —
  quantifying, under the oracle, *why* the traversal belongs to graph search and the LLM to the leaves.
  *Refuted if* the LLM traverses the verified graph as well as graph search out to long paths — in which
  case the division of labor is an efficiency choice, not a correctness one. Tested as **LP7**.
  **Result — the dependency-free CORE SUPPORTED; the LLM arm DEFERRED (honest split,
  [`lp7`](../../src/verisim/experiments/lp7.py),
  [`figures/lp7_traversal.png`](../../figures/lp7_traversal.png)). LP7 is the one SPEC-12 experiment
  needing an external model, so it is split exactly as §10 pre-registers. The committed core measures,
  over the verified graph, **graph search** (the planner's traversal — exact and complete, every path
  it returns is real by LP2's zero-false-edges) against a **myopic greedy walk** — the deterministic
  structural class of an LLM deciding the next node from local information, never presented *as* an
  LLM. Search stays at **validity 1.0 and optimality 1.0 at every path length and every node degree**;
  the myopic walk decays monotonically with depth (path-validity **1.00 → 0.39** over 1→8 hops, CIs
  disjoint from search by depth ≥ 3) and with degree (**1.00 → 0.68**) — the NLGraph depth-and-degree
  shape, derived mechanistically rather than asserted. That is the *correctness* argument for
  delegating traversal to search: local choices cannot recover the global route on a branching graph.
  The real LLM-traverser-vs-leaf-executor number is wired behind the `NetModel` seam
  (`llm_traverse_available`) and **never counted while absent** (§9), so the H37 LLM claim itself
  remains open pending a frozen proposer — but the structural boundary the spec rests on is measured.**

- **H38 — the landmark graph is a cross-world method (fork).** The verified-landmark-graph planner
  transfers to the host world (`hostsim`, where `imagine`/`verify` already ships) and to the distributed
  world (`distsim`, where partition/reachability is the hidden state ED12 measured), giving long-range
  plans the same way it does on the network. *Refuted if* it is network-specific (the reachability edge
  metric does not generalize) — itself a result about which worlds admit landmark planning. Tested as
  **LP8** (deferred fork; runs after the network headline).

---

## 6. Experiments (prefix **LP** — landmark planning; network world unless noted)

Each follows the house template: a `Config` dataclass with `from_json_file`, a CLI entry point, a JSONL
record stream, a `plot_*.py` emitting a committed `.png` + `.csv`, regenerable from `reproduce.sh`,
deterministic and seeded (SPEC-2 §12). All reuse the shipped network apparatus (the graph arm, the two
oracles, the `imagine`/`verify` loop) — SPEC-12 adds the graph builder, the search, and these harnesses.

- ✅ **LP1 — does the latent encode planning geometry? (H31).** Sample held-out network state pairs;
  compute `embed()` latent distance, oracle BFS transition-distance, and control-plane reachability
  distance; report Spearman/Pearson with bootstrap CIs. **The gate that decides §4's branch.**
  `experiments/lp1.py`, `configs/lp1.json`. **Shipped — H31 refuted (ρ = 0.27), reachability-space
  branch taken** (the per-anchor BFS geodesic is exact within a capped action-graph ball,
  `landmark/geometry.py`; the planned 2-D scatter is deferred with the rest of the plotting polish).

- ✅ **LP2 — the faithful landmark graph + the verified-vs-hoped gap (H32).** Build the landmark graph;
  for every candidate edge record model proposal (predicted reachability) vs the oracle truth; report
  edge precision/recall of the *unverified* graph, the residual after control-plane verification, and
  the consult-cost ratio (control-plane vs data-plane, the H12 ~3.6× lifted to edges). Output: the
  committed faithful graph + the gap figure. `experiments/lp2.py`, `configs/lp2.json`,
  `src/verisim/landmark/{graph,build,verify}.py`. **Shipped — H32 supported: hoped graph 77% false
  edges, verified residual 0.000, consult cost 0.62×.**

- ✅ **LP3 — goal reach: landmark planning vs flat free-running (H33, the headline).** Fix a battery of
  (start, distant-goal) pairs at increasing goal-space distance. Compare **(a)** flat/structured
  free-running rollout toward the goal (the HS3 `ρ=0` reading) and **(b)** landmark-hop planning over
  the verified graph with per-hop `imagine`/`verify` re-grounding. Report goal-reach success vs oracle
  budget `ρ`, and *planning horizon* (goal-space distance reached) vs *faithful horizon* (steps). **The
  figure that shows structure buying goal-space horizon.** `experiments/lp3.py`, `configs/lp3.json`,
  `src/verisim/landmark/plan.py` (graph search → subgoal sequence + the torch-free re-grounding hop
  executor). **Shipped — H33 supported: flat free-running decays with distance (0.50 → 0.17), landmark
  planning sustains (0.50 → 0.83) at `ρ ≈ 0.2`; a 5× far-goal gap that widens with distance, and goal
  reach monotone in the re-grounding budget.**

- ✅ **LP4 — edge-metric ablation: reachability vs exact-state edges (H34).** The LP3 planner, graph
  built two ways — edges from exact-delta prediction vs edges from reachability prediction — head to head
  on the same goal battery. `experiments/lp4.py`, `configs/lp4.json`. **Shipped — H34 supported:
  reachability edges sustain goal reach (0.50 → 0.83 with distance) while exact-state edges collapse
  (≈0, far-goal 0.33); the model's exact-state free-run horizon is pinned at 0 (HS3) while its
  reachability horizon climbs to ~7 (EN10) — the EN10-over-HS3 inversion at planning altitude.**

- ✅ **LP5 — landmark placement policy (H35).** Goal-reach per landmark for placement ∈ {random,
  reachability-betweenness (chokepoints), RSSM belief-variance (uncertainty), combined}, swept over
  landmark budget; LP5 refines LP3 (the budget keeps the top-`k` uniform boundaries by score).
  Surfaces the "interesting points." `experiments/lp5.py`, `configs/lp5.json`,
  `src/verisim/landmark/placement.py`. **Shipped — H35 split: belief-variance (uncertainty) is the
  load-bearing placement signal (adv +0.10 over random), betweenness underperforms random (adv −0.20);
  CIs overlap → leaning supported for uncertainty, banked.**

- ✅ **LP6 — replanning policy (H36).** Goal-reach at equal oracle budget for replanning trigger ∈
  {fixed-interval, reachability-triggered, belief-variance-triggered}. The RQ2 axis at plan altitude.
  `experiments/lp6.py`, `configs/lp6.json`. **Shipped — H36 supported for the reachability trigger
  (adv +0.13 over fixed-interval, pinpoints the critical re-ground at `B=1`); belief-variance trigger
  miscalibrated (adv −0.20) — the exact mirror of LP5.**

- ✅ **LP7 — the LLM-at-the-leaves boundary (H37).** Validity/optimality vs path length / node degree
  for graph search vs a myopic walk (the LLM-walk structural class) on the verified graph; the real
  LLM-as-traverser vs LLM-as-leaf-executor is wired behind the model-agnostic `NetModel` seam (SPEC §6
  commitment 4). `experiments/lp7.py`, `configs/lp7.json`. **Shipped (core) — H37 core supported:
  search stays exact (validity & optimality 1.0) at every depth and degree, the myopic walk decays
  with depth (1.00 → 0.39) and degree (1.00 → 0.68). LLM arm deferred (`llm_traverse_available`),
  never counted while absent (§9).** *(Honest scope: an LLM proposer is the one component needing an
  external model; the harness is built so this is a clean swap and the rest of SPEC-12 runs with no
  LLM.)*

- **LP8 — cross-world fork (H38, deferred).** Re-run LP2/LP3 on `hostsim` and `distsim`. Runs only after
  the network headline (LP3) lands, per the evidence gate. `experiments/lp8_host.py`,
  `experiments/lp8_dist.py`.

---

## 7. What is confidently buildable now vs gated on a result

The user's instruction — *build what is confidently positive, experiment on what is not* — maps cleanly
onto the dependency order:

- **Confidently buildable now (the machinery exists, the result is near-certain):**
  - The **faithful graph builder + control-plane verification** (LP2's apparatus). The two oracles and
    `reachability_matrix` ship; verifying edges is deterministic and free. The *graph itself* is a build,
    not a bet.
  - The **plan executor**: graph search → subgoal sequence → existing `imagine`/`verify` per hop. The
    low-level loop already ships; this is wiring.
- **Gated on LP1 (the §4 branch point):** *where* landmarks live — the EN8 latent (if H31 holds) or
  reachability space directly (if it does not). Cheap to run, decides the builder's metric space, so it
  runs first.
- **The genuine bets (must be measured):** LP3 (does the graph actually compose into long-range reach —
  the headline), LP4 (reachability vs exact edges), LP5/LP6 (placement, replanning), LP7 (the LLM
  boundary). LP3 is *high-confidence positive* — HS3 makes free-running hopeless for the structured arm
  and EN10 makes reachability strong, so any working hop-composition is a win — but "the graph composes
  cleanly under a real budget" is exactly the kind of claim the program insists on measuring before
  asserting.

**Recommended build order:** LP1 (decide the metric space) → LP2 (build + verify the graph; ship the
faithful-graph artifact) → LP3 (the headline planning win) → LP4 → LP5/LP6 → LP7 → LP8 fork. Each rung
graduates on a committed figure or a banked negative that licenses the next (SPEC §10.1, §12).

**Status (2026-06-08).** The §7 *confidently-buildable* tranche shipped, the headline bet (LP3)
landed and supported H33, and **the full network LP series LP1–LP7 has now shipped** (only the LP8
cross-world fork remains, plus LP7's deferred LLM arm) — each rung graduating on a committed figure:

- ✅ **LP1** ([`lp1`](../../src/verisim/experiments/lp1.py),
  [`figures/lp1_latent_geometry.png`](../../figures/lp1_latent_geometry.png)) — the gate ran and
  **refuted H31**: the latent does not encode planning geometry (ρ = 0.27 < 0.6), taking the §4
  fallback. The metric space is decided: **reachability**, not the latent.
- ✅ **LP2** ([`lp2`](../../src/verisim/experiments/lp2.py),
  [`figures/lp2_faithful_graph.png`](../../figures/lp2_faithful_graph.png),
  [`src/verisim/landmark/`](../../src/verisim/landmark/)) — the faithful graph builder + control-plane
  verification, **supporting H32**: the hoped graph is 77% false edges; verification prunes all of
  them (residual 0.000) at 0.62× the data-plane cost. The faithful-graph artifact ships.
- ✅ **LP3** ([`lp3`](../../src/verisim/experiments/lp3.py),
  [`landmark/plan.py`](../../src/verisim/landmark/plan.py),
  [`figures/lp3_goal_reach.png`](../../figures/lp3_goal_reach.png)) — the headline bet, **supporting
  H33**: long-range goal reach by composing *multi-step* faithful hops between landmarks (verify only
  at landmark boundaries, so ρ < 1 — the favorable-budget regime), against flat free-running over the
  same model. Flat decays with goal-space distance (0.50 → 0.17, the HS3 cliff at reachability
  altitude); landmark planning re-grounding once per `L = 4`-step hop (`ρ ≈ 0.2`) sustains and rises
  (0.50 → 0.83) — a **5× far-goal gap that widens with distance**, and goal reach monotone in the
  re-grounding budget (0.17 → 0.33 → 0.67 → 0.83 as `ρ: 0 → 1/8 → 1/4 → 1/2`). The graph search
  (`shortest_landmark_path`) produces the subgoal sequence; the shipped loop executes each hop; the
  oracle re-grounds at the boundary. **The structured arm whose *step* horizon SPEC-10 pinned at zero
  acquires a long *goal-space* horizon — measured, not asserted.**
- ✅ **LP4** ([`lp4`](../../src/verisim/experiments/lp4.py),
  [`figures/lp4_edge_metric.png`](../../figures/lp4_edge_metric.png)) — the edge-metric ablation,
  **supporting H34**: graded under two edge projections, the reachability arm sustains goal reach
  (0.50 → 0.83 with distance) while the exact-state arm collapses (≈0, far-goal 0.33). The cause is
  the per-step horizon — exact-state pinned at 0 (HS3) vs reachability climbing to ~7 (EN10) — so the
  structured arm's exact-rollout failure propagates to edge construction, and wiring on the
  reachability projection is what makes the graph plannable.
- ✅ **LP5/LP6** ([`lp5`](../../src/verisim/experiments/lp5.py),
  [`figures/lp5_placement.png`](../../figures/lp5_placement.png);
  [`lp6`](../../src/verisim/experiments/lp6.py),
  [`figures/lp6_replanning.png`](../../figures/lp6_replanning.png)) — the placement (H35) and
  replanning (H36) axes, a **mirrored pair**: *belief-variance (uncertainty)* is the load-bearing
  signal for **where** to keep a landmark (LP5: adv +0.10 over random; betweenness underperforms),
  while *reachability-change* is the load-bearing trigger for **when** to re-plan (LP6: adv +0.13 over
  fixed-interval; belief-variance miscalibrated). Both lean supported with overlapping CIs at this
  scale — banked, to revisit at larger worlds.
- ✅ **LP7** ([`lp7`](../../src/verisim/experiments/lp7.py),
  [`figures/lp7_traversal.png`](../../figures/lp7_traversal.png)) — the LLM-at-the-leaves boundary,
  **core supporting H37**: graph search is exact and complete at every depth/degree (validity &
  optimality 1.0) while the myopic walk (the LLM-walk structural class) decays with depth (1.00 →
  0.39) and degree (1.00 → 0.68) — the NLGraph shape, the correctness argument for delegating
  traversal to search. The real LLM-traverser arm is wired and deferred (never counted, §9).

Remaining: the **LP8 cross-world fork** (re-run LP2/LP3 on `hostsim`/`distsim`, H38) and **LP7's
deferred LLM arm** (the one external-model dependency).

---

## 8. The defensive payoff (why the artifact matters beyond the metric)

The oracle-grounded landmark graph *is* an oracle-verified reachability/attack graph. Against the
attack-graph field's standing weakness — unsound graphs full of paths the real network does not permit
(§2.3) — every edge here is confirmed by the data-plane oracle, so the graph has **zero false paths by
construction**. The defensive uses fall straight out of the planner:

- **Chokepoint and high-value-target surfacing** = LP5's placement policy (betweenness + the model's own
  uncertainty about a region).
- **"Can an attacker reach the database from the DMZ, and by what path?"** = a goal-conditioned graph
  search whose answer the oracle has already verified is real — the question an analyst asks, answered
  soundly and cheaply.
- **"Did this firewall change open a new path?"** = a diff of the verified graph before/after the
  intervention — the EN6/H5 change-safety reading lifted to whole-graph reachability.

This is the SPEC §4.1 ACD contribution (a learned-but-faithful network model closing the
sim-to-emulation gap) delivered as a concrete, defensive, human-out-of-loop artifact — consistent with
the program's direction (network-first, oracle-grounded, defensive, open-source). It is explicitly *not*
a shipped offensive agent (SPEC §13): the planner finds and verifies paths; it does not exploit them.

---

## 9. Build, reproduce, CI

### 9.1 Module layout (additive only)

```
src/verisim/landmark/                 # NEW — the planning layer (no new world, no new oracle)
  graph.py            # LandmarkGraph: nodes (waypoint states), edges (verified reachability), search
  build.py            # sample landmarks, cluster in embed() latent OR reachability space (§4 branch)
  verify.py           # oracle edge-verification (data-plane exact + control-plane cheap)
  plan.py             # high-level graph search → subgoal sequence; per-hop H_ε bound
  placement.py        # landmark placement policies (random / betweenness / belief-variance / combined)
src/verisim/experiments/
  lp1.py … lp8_*.py   # NEW — the LP experiments
figures/
  plot_lp1.py … plot_lp7.py  # NEW — committed-figure generators
configs/
  lp1.json … lp7.json        # NEW — committed sweep configs
```

The planner consumes the shipped `NetModel`/`NetUncertaintyModel` seam, the two network oracles, and the
`imagine`/`verify` loop **unchanged** — nothing in the deterministic core, the graph arm, the metrics, or
any existing experiment is edited. SPEC-12 is a layer *above* the loop, not a change *to* it.

### 9.2 `reproduce.sh` (new LP block, in dependency order)

```bash
echo "== LP1: does the latent encode planning geometry? (H31) — gates the metric space =="
python -m verisim.experiments.lp1 --config configs/lp1.json --out runs/lp1/records.jsonl --plot figures/lp1_latent_geometry.png
echo "== LP2: the faithful landmark graph + verified-vs-hoped gap (H32) =="
python -m verisim.experiments.lp2 --config configs/lp2.json --out runs/lp2/records.jsonl --plot figures/lp2_faithful_graph.png
echo "== LP3: goal reach — landmark planning vs flat free-running (H33) — THE HEADLINE =="
python -m verisim.experiments.lp3 --config configs/lp3.json --out runs/lp3/records.jsonl --plot figures/lp3_goal_reach.png
# LP4–LP7 follow; LP8 (cross-world fork) gated on LP3.
```

The non-LLM LP block runs on CPU (the deterministic gate); LP7's LLM proposer is the one optional
external dependency and is `skipif`-guarded with disclosure (never counted as a result when skipped).
CI (`ubuntu-latest`) stays the free Linux confirmation; LP tests assert structural invariants (a verified
graph has zero false edges; planning reaches goals the free-running floor cannot; LP1's correlation sign),
not exact magnitudes, so the same tests pass on the macOS primary host and Linux CI (the §macOS-first
principle).

---

## 10. Scope, non-goals, honest caveats

- **This is a planner, not a new world.** It runs on the SPEC-5 network world and reuses both oracles.
  Over-claiming a new dynamics result from SPEC-12 is the failure mode this line forbids; the result is
  about *planning over* the existing model, measured against the existing oracle.
- **The headline is conditional on LP1's branch.** If the EN8 latent does not encode planning geometry
  (H31 refuted), the graph is built in reachability space and the *latent-landmark* framing weakens to a
  *reachability-landmark* one — still a verified graph and a working planner, but a narrower claim about
  representation. That branch is banked, not feared.
- **LP3 is high-confidence but unproven.** HS3 + EN10 make it the expected positive, but "the graph
  composes under a real budget" is a measurement the program will not skip. A refutation (the cliff is
  the loop's regardless of rollout strategy) is the deepest possible negative and closes the line
  honestly.
- **The LLM piece needs an external model.** LP7 is the one experiment requiring a frozen LLM proposer;
  everything else is dependency-free. The harness isolates it behind the `NetModel` seam so the spec's
  spine runs without it.
- **Edge verification cost is real.** Verifying every candidate edge is `O(landmarks²)` consults in the
  worst case; the control-plane oracle's ~3.6× cheaper consult (H12) and landmark sparsity are what keep
  it affordable, and LP2 measures the actual cost rather than assuming it.
- **Faithful-horizon radius is region-dependent.** `H_ε` is the loop's property (EN7/H22), so edge-length
  admission is measured per region, not set globally — more work than L3P's single learned distance, and
  the reason the graph is *correct* rather than *hoped*.

---

## 11. Provenance & reading order

SPEC-12 is the planning altitude the program has been building toward: SPEC-5 gave the network world and
the two oracles; SPEC-8 put the oracle in the representation (the `embed()` latent this spec scatters
landmarks in); SPEC-10 found the structured-arm compounding wall (HS3) that makes a landmark layer
*necessary*; and the shipped `imagine`/`verify` loop is the low-level controller it plans on top of. The
architecture is L3P and its goal-conditioned-RL lineage (SoRB, SPTM, SFL); the unique contribution is the
oracle-verified edge — the faithful landmark graph — which is only possible because computer worlds have
the oracle continuous-control worlds lack (SPEC §2). It is a *method* spec: it advances how the program
*uses* its world model, not what the model is.

Reading order for a newcomer: [SPEC.md §2](./SPEC.md) (the oracle asymmetry) →
[SPEC-5 §5.1](./SPEC-5.md) (the two-plane oracle this plans over) →
[SPEC-10 §4.6–4.10](./SPEC-10.md) (HS3 — the wall this is the escape from) → this document
(§1 the motivation, §3 the architecture, §4 the load-bearing assumption, §5 the hypotheses) →
`src/verisim/landmark/` and `src/verisim/experiments/lp1.py` (the concrete build, once shipped).
