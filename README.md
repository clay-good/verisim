# Verisim

> **The claim.** Every other domain trains world models against *proxies* for truth — a fixed
> corpus, a human annotator, a hackable reward. **Computer environments are the one exception:
> filesystems, processes, networks, and APIs are digital, deterministic, and fully checkable, so a
> deterministic *oracle* can return the exact next state for free, at every step.** Verisim is the
> research program built around that single asymmetry — putting an oracle *in the loop* to bound a
> neural world model's drift, and measuring the tradeoff nobody else can measure: **how much oracle
> consultation buys how much faithful horizon.** The world model is a pluggable proposer (transformer,
> JEPA/RSSM, or a frozen LLM), so the real bet is a *method*, not a model: **deterministic verification
> as a model-agnostic primitive for probabilistic ML** — the layer underneath the world-model race, not
> another entrant in it.

## Why this isn't RLVR — and why it isn't redundant

The obvious objection: *"deterministic verifiers grounding probabilistic models — isn't that just
RLVR (verifiable rewards)?"* No. RLVR and verisim use the **same kind of verifier at opposite ends
of one axis: how much of the oracle's signal you actually use.** A verifier always holds the *exact*
truth — it must, to grade (`expected 5, got 7`). RLVR reads **one bit** off it (right/wrong), as a
**terminal** reward, at **training** time, and throws the rest away.

| | RLVR (verifiable rewards) | Verisim |
|---|---|---|
| signal used | a **scalar** reward (right/wrong) | the **exact next state**, bit-for-bit |
| when consulted | terminal — the final output | **in the loop**, every step, on a budget `ρ` |
| what the oracle does | updates weights | **corrects drift at runtime** + labels training data + branches counterfactuals |
| consulted at | training time only | training **and** inference |

RLVR collapses to a scalar because in its native domains (math, code-as-text) the *intermediate*
states aren't oracle-defined — there is no unique "correct next reasoning token," so only the endpoint
is checkable. **Computer-state worlds remove exactly that limit:** the state after `mkdir /a` is one
thing, exactly checkable, and so is every step after it. That single asymmetry is what lets verisim do
what RLVR structurally cannot — teacher-force the exact next state, correct mid-rollout, partition the
decidable bits from the genuine residual, and re-run actions *not taken* for free. **RLVR is the
minimum-bandwidth point on an axis whose entire interior is unexplored and measurable.** Verisim draws
the curve.

## The gradient it predicts

If faithfulness scales with oracle bandwidth, then how reliably headless AI works in a domain should
track *how good a deterministic oracle that domain admits* — and it does:

| oracle character | domain | headless AI today |
|---|---|---|
| total · exact · free · every step | computer state (fs / proc / net), SQL, types, config, IaC | **works** — this is enterprise headless SDLC |
| endpoint-only · exact | math answer, "the tests pass" | works *at the endpoint* — RLVR's home |
| approximate · partial | physics, robotics, video dynamics | struggles |
| absent | image aesthetics, audio timbre, open prose | "looks plausible" is the ceiling |

Computer worlds are the **one** place you can turn the oracle up and down *continuously* (that is what
`ρ` and the tiered oracle are) and watch faithfulness move with everything else held fixed — the
dose-response no other domain can measure. So verisim is best read as an **instrument**: it measures
the dose-response of verification in the one laboratory clean enough to measure it, and reports what
the proxy metrics everyone else uses cannot see (a bigger, per-step-*more*-accurate model that is
*less* faithful over the horizon — [§2](#2-but-better-one-step-prediction-does-not-yet-buy-free-running-horizon-the-honest-negative),
[HS1.1](#results-at-a-glance)). The figures below are that instrument's readings.

## Results at a glance

Twenty-five committed, oracle-grounded figures — the smoke-scale bet in one screen — each detailed below, each
with its honest negative, now joined by the **SPEC-11 system-oracle validation** ([§35](#35-the-oracle-is-faithful-to-a-real-computer--the-one-structural-bet-measured-spec-11--sy1--h27)) — *the figure that retires W1*: the program's load-bearing assumption (a free, exact oracle exists for computer worlds) measured against a **real `/bin/sh` on a real kernel**, bit-for-bit. **Two one-figure theses frame the rest: the cross-*world* synthesis ([§22](#22-the-thesis-in-one-figure-the-floorcliff-is-the-same-in-every-world-cross-world-synthesis)) — the floor+cliff `H_ε(ρ)` is the *same shape in all four worlds* — and the cross-*proposer* synthesis ([§34](#34-the-spec-10-capstone-the-floor-is-proposer-dependent-cross-proposer-synthesis)) — whether that floor is a *resourcing artifact* depends on the proposer.** **What survives scaling is the real verdict: see [§8](#8-which-wins-survive-scaling--the-honest-mixed-verdict-spec-9).** Every number regenerates from
config + seeds (`bash figures/reproduce.sh`).

| | |
|---|---|
| [![SY1 the system oracle — the reference oracle agrees with a real /bin/sh bit-for-bit](figures/sy1_agreement.png)](figures/sy1_agreement.png)<br>**The one structural bet, measured against reality (SPEC-11 / SY1 / H27).** Every figure here proves the loop against a *from-scratch reference model* of POSIX — the program's central exposure (W1: "the oracle is a model, not reality"). SY1 closes it: run the exact v0 grammar against a **real `/bin/sh` on a real kernel** in a hermetic sandbox and compare bit-for-bit. On the **structure-building grammar** the reference oracle's predicted next state equals the real kernel's at **agreement = 1.000** (left, the green bars; degenerate CI), and the head-to-head `H_ε(ρ)` curve is **oracle-invariant** (right, max gap = 0.000) — *the oracle substitution is transparent where v0 claims fidelity.* The disagreement branch is a gift (H28): the harness **caught a real reference bug** (subtree-move, now fixed) and every remaining divergence is one of **four named, intentional modeling boundaries** with **residual = 0**. Built macOS-first; the headline is pure POSIX, so it reproduces on Linux CI. "Computer worlds have a free, perfect oracle" stops being an argument and becomes a committed CSV. | [![SY3 the hermeticity contract — no action can occur](figures/sy3_hermeticity.png)](figures/sy3_hermeticity.png)<br>**Proven safe before any number is claimed (SPEC-11 / SY3 / H29).** The agreement figure is only meaningful if the sandbox is a *closed system*. SY3 is the negative-control battery with teeth: every prohibited action — filesystem escape, network egress, privilege gain, cross-step persistence — is **denied**, and the benign positive control is allowed-then-discarded. "No action can occur" made into a green table. With SY4 (the system oracle is bit-reproducible under its determinism seal), this is the gate *on* the agreement claim — prove the checks have teeth before reporting what they catch. |
| [![HS1.1 the faithful-horizon scaling law — non-monotone, and the proxy goes blind](figures/horizon_scaling_xl.png)](figures/horizon_scaling_xl.png)<br>**Faithful horizon is *non-monotone* in capacity — and the one-step metric can't see it (SPEC-10 / HS1.1 / H26).** Hold the world fixed, sweep model capacity ~400× with an adequate coverage set, and measure free-running faithful horizon `H_ε(ρ=0)` *exactly* against the oracle. It **rises to a compute-optimal peak at `l` (17 id / 28 ood steps)** then **declines** (xxl 9.6) — a Chinchilla-style frontier, but for long-horizon *faithfulness*, not test loss. The headline: across the whole top end the per-step accuracy `p` any normal world-model paper reports stays **flat and high** (0.81–0.90), while the *exact* horizon **falls ~45%** and ood efficiency `η` **crosses below 1** — **a bigger, per-step-more-accurate model that is *less faithful over the horizon*, a divergence only the free exact oracle reveals.** (And the floor itself lifts ~4× from *resourcing alone*: `xs` 1.75 → 6.83.) The data cross-axis (HS1.2) then shows that decline is **data starvation, not a wall** — at fixed `xl`, feeding 2× the data recovers `H_free` 13.9 → 16.2 and ood η from 0.97 back to 1.90, the Chinchilla prescription made exact: once capacity is adequate, the lever is data, and only the oracle could diagnose it (`p` stays flat while the horizon recovers). The joint push (HS1.3) then scales **both** levers together: it lifts the peak to a **program-best `l@9.6k` = 19.2 id / 28.75 ood** (tight, disjoint CIs) but returns vanish past `l` — so the floor+cliff resolves into a **resourcing story with a measurable compute-optimal frontier**, not a fundamental compounding wall, all measured exactly against the oracle. | [![EN1 / K4 the floor](figures/en1_curve.png)](figures/en1_curve.png)<br>**The floor HS1 revises (EN1 / K4 / H8).** Faithful horizon vs consultation budget `H_ε(ρ)` was flat-then-cliff on every world at the committed (tiny) scale: the honest negative that drove every later design choice. HS1/HS1.1 show the *height* of that `ρ=0` floor is not fixed — it scales with capacity+data, then *declines* once capacity outruns the data — which sharpens what this curve does and does not show: a property of the *consultation* axis at fixed model, not a ceiling on what a larger model can free-run. |
| [![HS2 the scaling law re-run on the host world — the lift is universal, the floor re-lowered](figures/horizon_host_scaling.png)](figures/horizon_host_scaling.png)<br>**The capacity lift is *universal* — and a harder world re-lowers the floor (SPEC-10 / HS2 / H26).** Re-run the *identical* HS1 capacity axis on the harder **host** world (SPEC-6: the composed process/fd/filesystem/exit bundle) and the verdict survives the world swap: free-running `H_free` scales **monotonically** with capacity (id **1.00 → 5.08** over 108× params, **disjoint CIs** xs vs l). So "capacity buys horizon" is a property of the oracle *loop*, not the easy network world. But the richer host dynamics do exactly what HS1 predicted: they **re-lower the floor ~3–5×** (host `l` 5.08 vs network `l` 15.7) and the curve **has not saturated by `l`** (the network saturated by `m`) — a harder world both lowers the floor and re-opens the headroom. The host one-step `p` is far lower (0.11 → 0.49 vs network's 0.47 → 0.79) — a genuinely harder per-step target — and `η`, though > 1 throughout, **declines toward 1 with capacity** here (the mirror of the network's rising `η`). **World difficulty sets the floor height and the peak location; the capacity-buys-horizon verdict is cross-world.** | [![EH1 composed-host floor](figures/eh1_curve.png)](figures/eh1_curve.png)<br>**The host floor HS2 scales (EH1 / HC6).** The composed-host `H_ε(ρ)` is the same flat-then-cliff as the filesystem and network worlds — `ρ=0` drifts in <1 step at the committed tiny scale. HS2 shows the *height* of that host `ρ=0` floor lifts with capacity (xs 1.0 → l 5.1 free-running steps), just as HS1 showed for the network — but stays well below the network's, because the composed bundle is the harder world. |
| [![HS3 the scaling law with the structured graph arm — the lift is proposer-dependent](figures/horizon_graph_scaling.png)](figures/horizon_graph_scaling.png)<br>**The capacity lift is *proposer-dependent* — it does NOT reproduce for the structured arm (SPEC-10 / HS3 / H26).** Re-run the *identical* HS1 axis with the **GNN+RSSM graph arm** — the proposer that *beats* the flat arm ~6.6× on one-step delta-exact (EN4/H11) — and the verdict flips: for the graph arm capacity buys **neither** per-step accuracy **nor** horizon. `p` is **flat** (0.64 → 0.66, vs the flat arm's 0.47 → 0.82 climb) and `H_free` is **≈ 0 at every capacity** (η ≈ 0) — the floor+cliff in its purest, capacity-invariant form. So **HS1's lift was the flat arm's specific p-vs-capacity climb crossing the self-stabilization threshold, not a universal loop property.** The graph arm makes *near-but-not-exact* predictions (an ε-sweep gives `H_free`=0 up to ε=0.1, only 4–6 steps at ε=0.2) — small but ubiquitous errors the flat arm's self-stabilizing rollout avoided. Honest caveat: the graph trainer plateaus at p≈0.66 < the flat arm's 0.82, and the arm's *flat* p says it is **data-limited, not capacity-limited** (its lever is data, the HS1.2 reading). Consistent with EN7/H22: the loop governs the shape, the proposer's competence sets whether it escapes the floor. | [![EN4 graph beats flat per-step](figures/en4_graph_vs_flat.png)](figures/en4_graph_vs_flat.png)<br>**Why HS3 is the sharp foil (EN4 / H11).** The graph arm *wins* on the one-step metric a normal paper reports — +16.5 pts token accuracy, +30.6 pts delta-exact over the flat arm. HS3 is the catch the oracle exposes: that per-step win **does not** convert to free-running horizon (`H_free`≈0), and capacity doesn't fix it. The metric that wins (delta-exact) and the metric that matters (exact free-running horizon) come apart — exactly the proxy/truth divergence the whole program is built to measure. |
| [![HS3 incr 2 the graph data cross-axis — the structured floor is not data starvation](figures/horizon_graph_data_scaling.png)](figures/horizon_graph_data_scaling.png)<br>**The structured floor is *not* data starvation — a genuine ceiling (SPEC-10 / HS3 incr 2 / H26).** HS3 left a confound: the graph arm's flat `p` looks *data*-limited, so — exactly as HS1.2 was for the flat arm — hold graph capacity fixed and sweep the coverage set 960 → 9,600 transitions. A **10× data increase does NOT lift `H_free`** (≈0 throughout) or `p` (flat ~0.6) — the *opposite* of the flat arm, whose HS1.2 floor *recovered* with data (7.7 → 16.2). And **η stays below 1** (0.00–0.52): unlike the flat arm (η>1, self-stabilizing), the graph arm free-runs **shorter** than its i.i.d. prediction — its near-but-not-exact errors **compound**, the genuine compounding wall H26 pre-registered. **Net across HS3: the structured floor moves with *neither* capacity nor data — the first floor in the program that does not dissolve into resourcing.** So "the floor+cliff is a resourcing story" is itself proposer-dependent (under-resourcing for the flat arm; a genuine ceiling for the structured one). Honest caveat: the graph trainer plateaus at p≈0.6 and p doesn't climb with data, so the binding constraint is plausibly the trainer/representation, not data per se. | [![HS1.2 flat-arm data recovery](figures/horizon_data_scaling.png)](figures/horizon_data_scaling.png)<br>**The contrast that makes HS3 incr 2 sharp (HS1.2 / H26).** For the **flat** arm, the same data cross-axis was a *recovery*: at fixed `xl` capacity, feeding 2× the data lifted `H_free` 13.9 → 16.2 and ood η from 0.97 back to 1.90 — the floor *was* data starvation (Chinchilla). The structured arm (left) shows the opposite on the identical axis: data buys nothing. Same experiment, two proposers, opposite verdicts — which is exactly why "is the floor a resourcing artifact?" has no single answer: it depends on the proposer. |
| [![HS3 incr 3 the world-size cross-axis — the structured ceiling is world-size-invariant](figures/horizon_graph_world_scaling.png)](figures/horizon_graph_world_scaling.png)<br>**The structured ceiling is *world-size-invariant* — it survives the last axis too (SPEC-10 / HS3 incr 3 / H26).** The graph arm exists *for* its inductive bias over network structure, which has **more** to exploit as the world grows — so a bigger world is where it could finally pull off the floor. Hold graph capacity fixed and sweep the world over SPEC-9's `O(N²)` host-count axis (5 → 40 hosts): `H_free` stays **0 at every world size** (8× range, tight zero CIs), η = 0 throughout, and `p` actually **degrades** (0.66 → 0.59) as the world gets harder per step. **This completes the HS3 sweep: the structured arm's exact free-running floor is pinned at 0 across *all three* axes — capacity, data, AND world size** — a genuine compounding ceiling, not an artifact of any single axis. Where the flat arm's floor dissolved into a resourcing story on every axis (HS1/HS1.2/HS2), the structured arm's moves on **none** of them. Honest caveat: the graph trainer plateaus at p≈0.6 and p falls with world size, so the binding constraint is plausibly the trainer/representation; this is the strict ε≤0.1 tolerance. | [![EN7 model-invariance](figures/en7_invariance.png)](figures/en7_invariance.png)<br>**Why the two verdicts coexist (EN7 / H22).** The loop governs the *shape* of `H_ε(ρ)` (floor + cliff, no knee) across proposers; what differs is the *floor height* and whether a given proposer ever escapes it. The flat arm's per-step accuracy climbs high enough (with resourcing) that its rollout self-stabilizes off the floor; the graph arm's plateaus at ~0.6 and never does — so HS1's resourcing story and HS3's genuine ceiling are two faces of the same H22 fact: verification is the invariant, the proposer's competence sets the floor it lives on. |
| [![SPEC-10 capstone — the floor is proposer-dependent](figures/horizon_synthesis.png)](figures/horizon_synthesis.png)<br>**The SPEC-10 capstone — the floor is *proposer-dependent* (cross-proposer synthesis).** One figure for the whole scaling arc, read straight off the two committed capacity sweeps (figures-from-records — it re-runs nothing). On the **same** capacity axis the **flat** transformer's free-running horizon **lifts ~9×** (1.75 → 15.8 steps) and its floor dissolves into a resourcing story across capacity, data, and world size (HS1/HS1.2/HS2); the **structured** GNN+RSSM graph arm — which *beats* the flat arm on one-step delta-exact (EN4) — stays **pinned at ≈ 0** across all three axes (HS3). So the program's standing question, *"is the floor+cliff a resourcing artifact?"*, has **no single answer: it depends on the proposer** — under-resourcing for one, a genuine compounding ceiling for the other. A per-step *winner* that is the long-horizon *loser*, and vice versa — the proxy/truth divergence only the exact free oracle can expose. | [![cross-world synthesis](figures/synthesis_floor_cliff.png)](figures/synthesis_floor_cliff.png)<br>**Its sibling thesis (cross-world synthesis, [§22](#22-the-thesis-in-one-figure-the-floorcliff-is-the-same-in-every-world-cross-world-synthesis)).** The other one-figure claim: normalize each world's `H_ε(ρ)` by its ceiling and the filesystem, network, host, and distributed worlds trace **the same floor+cliff** — even the one (distributed) whose oracle is intractable. Read together, the two syntheses bracket the program's generality: the curve's *shape* is world- and model-invariant (the loop's), while whether its `ρ=0` *floor height* is a resourcing artifact is proposer-dependent (the model's). Verification is the invariant; the proposer sets the floor it lives on. |

| | |
|---|---|
| [![cross-world synthesis](figures/synthesis_floor_cliff.png)](figures/synthesis_floor_cliff.png)<br>**The thesis in one figure (cross-world synthesis).** Normalize each world's faithful horizon by its own ceiling and overlay: the filesystem (a tree), the network (a graph), the host (a coupled bundle), and the distributed cluster (replicated services under faults) trace **the same floor+cliff** — a near-zero floor across the `ρ` interior, then a cliff to full horizon only at `ρ=1`. Four different state types, oracles, and models; one curve — and the fourth's oracle is only *tiered*, not exact, so the shape isn't an artifact of a cheap exact oracle. "A little consultation doesn't buy a lot of horizon; you pay near-linearly for faithfulness" is a property of the *oracle-loop method*, not any one world. | [![ED1 the distributed prime directive + the tiered-oracle H17 measurement](figures/ed1_dist.png)](figures/ed1_dist.png)<br>**The distributed world debuts — and the tiered oracle is a *conditional* lever (SPEC-7 / ED1 / H17).** The fourth world (replicated services across machines) is the first where the bit-exact global oracle is **intractable**, so it can't be spent every step — the payload is a *tiered* oracle (metamorphic ↔ cycle ↔ symbolic ↔ bit-exact) and the loop chooses *which tier* (`π_w`) to pay. **Left:** the distributed `H_ε(ρ)` is the *same floor→cliff* as every prior world (0.2 → 40). **Right (H17):** **oracle-$ per faithful step** by tier × error class — and the answer is sharper than "cheap wins": it **depends where the model's errors fall**. For **gross** (out-of-vocab) errors the cheap metamorphic tier buys faithful horizon at **$9.4/step vs bit-exact's $16**; for **subtle** (in-flight) errors the cheap tiers miss the drift entirely (H≈0, **$848/step**) and only full bit-exact truth is efficient. A real, conditional lever — the measurement the oracle-free distributed-systems field cannot make. (Apparatus on a controlled-noise proposer; the learned `M_θ` now supplies the real distribution — [ed1_learned.png](figures/ed1_learned.png) — where the constrained decoder removes gross errors, so the cheap tiers catch nothing and bit-exact is the efficient choice: the honest inverse.) |
| [![ED2 equal-dollar-budget — does a cheap tier buy more horizon per $?](figures/ed2.png)](figures/ed2.png)<br>**H17 in its sharpest form: the equal-*dollar*-budget frontier (SPEC-7 / ED2 / H17 + H18).** ED1 asked "which tier is cheaper per faithful step?"; ED2 asks the question the hypothesis is really about — *at an equal oracle-dollar budget, does a cheap or cheapest-refutation (`escalate`) tier policy buy more faithful horizon than spending the same dollars on bit-exact truth?* It sweeps `ρ` and plots, per tier policy, the **faithful-horizon-vs-oracle-dollar frontier** (the Pareto front), comparing policies at a matched budget by interpolating each one's horizon along its envelope — a true equal-*dollar*, not equal-`ρ`, comparison. At the sub-linear **quarter budget** `B/4`: for **gross** (cheaply-catchable) errors the metamorphic tier reaches **H=14.2 vs bit-exact's 4.2** (tiering wins per dollar — H17 holds, a **0.36** competitive ratio of the full-truth ceiling at ¼ the cost, the **H18** readout); for **subtle** (bit-exact-only) errors the cheap tiers sit flat at the floor (1.5) and even `escalate` *loses* to single-tier bit-exact — H17's honest negative, in the form the spec poses it. (The learned `M_θ` now supplies the real distribution — [ed2_learned.png](figures/ed2_learned.png) — where the constrained decoder's subtle-only errors put a real model entirely in the right-hand regime: at `B/4` only bit-exact buys horizon and `escalate` loses, the H18 ratio just 0.06 — the budget-form honest inverse.) | [![ED4 H21 fault-injected training beats fault-free](figures/ed4_fault.png)](figures/ed4_fault.png)<br>**The DST/BUGGIFY data-factory lesson, made measurable (SPEC-7 / ED4 / H21).** Train two equal-volume `M_θ` — one fault-free (`fault_prob=0`), one fault-injected — then sweep the eval workload's fault-intensity **free-running** (`ρ=0`, so it exposes the *model*, not the loop). **H21 confirmed with the sharpest control:** at zero eval-fault the two coincide, but as faults intensify the fault-injected model holds **~3× more** free-run horizon (0.375 vs 0.125 steps) — *even though the fault-free model is the **better** clean predictor* (teacher-forced accuracy 0.60 vs 0.49). The fault-free model never saw a partition/crash/heal, so under fault it derails immediately. Fault injection buys robustness factual data cannot — DST as a *data factory*, not just a test harness; and a bonus proxy/truth-divergence instance (higher clean accuracy, lower compounding horizon). |
| [![ED5 consistency-vs-bit horizon + the competitive ratio](figures/ed5.png)](figures/ed5.png)<br>**Consistency-faithful horizon *outlasts* bit-faithful, and the loop is learning-augmented (SPEC-7 / ED5 / H19 + H18).** Two findings on the dependency-free synthetic proposer. **Left (H19):** free-running, the **consistency-faithful** horizon (does the model predict each object's converged/split state, §9.1?) **outlasts the bit-faithful one for `subtle` (in-flight) errors — H=13.1 vs 1.5, gap +11.6 (disjoint CI)** — because a corrupted in-flight replication message is immediately *bit*-visible but **consistency-invisible** until `advance` delivers it and writes a replica; for `gross` (durable-replica) errors the two coincide (the control). So W7's "no global state" *does* buy the model forgiveness — but only where the error hides in the consistency-invisible medium. **Right (H18):** the **competitive ratio** `H_ε(ρ)/ceiling` fit across `ρ × prediction error` — the loop is "algorithms with predictions" (model = predictor, bit-exact oracle = worst-case-safe fallback). The fan of lines is the learning-augmented signature: the ratio **degrades gracefully with prediction error** (quarter-budget ratio monotone 1.00 → 0.05, recovering the trivial bound for a perfect model) — *confirmed*. But the cliff to the ceiling only appears at ρ→1: at the sub-linear `B/4` budget a noisy model's ratio sits near the floor — the same **floor→cliff / no-knee** negative as every prior world, now in competitive-ratio form. Learning-augmented in the *error* axis; no free lunch in the *budget* axis. | [![ED3 the distributed world breaks v0's operator identity](figures/ed3.png)](figures/ed3.png)<br>**The distributed world breaks v0's correction-operator identity (SPEC-7 / ED3 / RQ3).** v0 proved an *identity*: a full-truth consult makes `hard_reset`/`residual`/`projection` behaviorally identical on `H_ε` (they all snap to the same truth). ED3 shows the distributed world breaks it, **mode-dependently**, with a *partial* operator: `ReplicasOnlyCorrection` snaps the durable replicas to truth but **trusts the model's predicted in-flight** — the stale-read medium, the `subtle` error class. For **gross** (replica-write) errors all four operators recover the same horizon (H=7.2, identity holds); for **subtle** (in-flight) errors the three full-correction operators hold (H=6.2) but `ReplicasOnlyCorrection` **collapses to H=1.8** (gap 4.5) — the in-flight medium is the distributed world's hidden state a partial correction cannot see, the same gross/subtle structure H17/H19 turn on. |
| [![ED4 consistency level — the H19 gap tracks the in-flight medium](figures/ed4_consistency.png)](figures/ed4_consistency.png)<br>**The H19 gap is a *weak-consistency* phenomenon — it tracks the in-flight medium (SPEC-7 / ED4 / H20).** The `CONSISTENCY_MODELS` curriculum dial (§3.4) gains its first strong end: **`linearizable`** — synchronous all-replica writes, CP write-rejection under partition (CAP, HW-5), so no replica is ever stale and there is **no in-flight medium**. Sweeping the declared model resolves H20 *through* H19. The **subtle** (in-flight) panel is the headline: under **`eventual`** (in-flight rate **3.2/step**) the consistency-faithful horizon outlasts the bit-faithful one by **+10.5 steps** (the H19 gap — errors hide in flight); under **`linearizable`** (in-flight rate **0**) the gap is **exactly 0** — the consistency-invisible medium is gone, so the subtle error class is structurally empty and there is nothing for consistency-faithfulness to forgive. The **gross** (durable-replica) panel is the control: a durable error is consistency-visible at once, so bit and consistency horizons coincide (gap 0) at both levels. Strong consistency buys the model no forgiveness because there is no hidden state to forgive — the H20 mechanism made concrete and exact (dependency-free). | [![ED1 learned-model curve + the real-model H17](figures/ed1_learned.png)](figures/ed1_learned.png)<br>**The real learned `M_θ` makes the tiered oracle's value model-dependent (SPEC-7 / ED1-learned / H17).** The flat DS4 `M_θ` trained on seeded rollouts, then run through the *same* tiered loop, so the curve and H17 are measured on a **real** error distribution. **Left:** the same **floor→cliff** (0.2 → 32). **Right:** the **honest inverse** of the synthetic H17 — the LL(1)-constrained decoder removes the *gross* (out-of-vocab) error class *by construction*, so a real model's residual errors are all *subtle*: the cheap **metamorphic** tier catches none (H=0.2, **$624/faithful-step**), **symbolic** few ($411), and only **bit_exact** is efficient (H=32, **$16**); cheapest-refutation **escalate** reaches full horizon but pays *more* ($21.6) because a real model's errors need the bit-exact correction anyway. A cheap tier helps exactly when a model makes cheaply-catchable errors — and a grammar-constrained learned model, by design, does not. |
| [![ED6 distributed counterfactual grounding — H5 finally pays](figures/ed6.png)](figures/ed6.png)<br>**The distributed world is where counterfactual replay finally pays (SPEC-7 / ED6 / H5).** The deterministic DES is *total*, so from any visited cluster state it returns the true next state of an **alternative fault** the trajectory never took — a free counterfactual branch (re-run from `(seed, t)` with one fault flipped). Three matched-count arms train the same flat `M_θ`: `trajectory` (base light-fault on-policy), `trajectory-more` (5× more on-policy data — the **volume control**), and `+counterfactual` (base + free oracle **fault**-flip branches — the near-miss partitions/crashes). Scored on **held-out fault interventions**: `+counterfactual` beats **both** the base **and** the volume control on **both** readouts with disjoint CIs — **intervention-exact 0.51 vs 0.25 vs 0.06**, and **medium-recall 0.56 vs 0.22 vs 0.05** (does it predict the partition/crash split-brain?). This is the **honest inverse** of the network (EN6) and host (EH6/H16) supervision *null*, where counterfactual data did **not** beat volume. The mechanism: the distributed **medium** (partition/crash/in-flight) is a hidden state the light-fault on-policy distribution structurally underrepresents — so on-policy *volume* buys little (0.06→0.25) while off-policy oracle **fault branches** buy a lot (0.25→0.51). The held-out-intervention analogue of the H21 data-factory result. *(Honest caveat: the branches are fault-heavier than the on-policy control, so the lift conflates counterfactual branching with the fault coverage it carries — but EN6/EH6 found null under the identical design, so the distributed positive is the result; the disentanglement is future work.)* | [![ED1 the distributed prime directive — H_ε(ρ) + the tiered oracle H17](figures/ed1_dist.png)](figures/ed1_dist.png)<br>**The distributed prime directive: the `H_ε(ρ)` curve + the first tiered-oracle verdict (SPEC-7 / ED1 / H8 + H17).** The first world where the bit-exact global oracle is *intractable*, so the payload is the **tiered oracle** (metamorphic ¢1 → cycle ¢2 → symbolic ¢4 → bit-exact ¢16). **Left:** the same **floor→cliff** `H_ε(ρ)` (floor 0.2 free-running → ceiling 40 fully-consulted, bootstrap-CI over seeds) — the distributed knee is the no-knee negative again (H8). **Right (H17):** oracle-**dollar** *per faithful step* per tier × error class — *whether a cheap tier buys more horizon per dollar depends where the model's errors fall*: for **gross** (out-of-vocab) errors the metamorphic tier is cheaper per faithful step ($9.4 vs bit-exact's $16); for **subtle** (in-flight) errors the cheap tiers miss the drift entirely and bit-exact is the only efficient choice. The central SPEC-7 tradeoff, made exact on a controlled error distribution. |
| [![ED6 two-oracle — the consistency oracle is redundant but decision-sufficient and cheaper](figures/ed6_two_oracle.png)](figures/ed6_two_oracle.png)<br>**A cheap consistency oracle is redundant but decision-sufficient and cheaper (SPEC-7 / ED6 / H12).** The distributed analogue of the host's privilege second-oracle (EH6) and network's control-plane oracle (EN10). The full **bit-exact** oracle verifies the predicted cluster bit-for-bit; the cheap **consistency oracle** answers only the one operationally-decisive question an SRE asks under partition — *is each object converged or split (a split-brain)?* (the §9.1 metric). Teacher-forced over the fault-heavy `adversarial` workload: **non-redundant rate 0** by construction (the consistency view is a pure function of the replicas, so a bit-exact-correct prediction is always consistency-correct — it catches *nothing* the full oracle misses); but **consistency-sufficient** — of the steps where the model's *full* prediction is wrong, the split-brain verdict is still right **1.00 for `subtle` (in-flight) errors vs 0.00 for `gross` (durable-replica) errors** (the per-step form of ED5's H19 horizon gap, tracking the in-flight medium) — at a **consult-fact ratio of 0.28** (~3.6× cheaper, the gap widening under fault because the medium inflates the full state but never the consistency view). *Redundant for verification, but a cheaper, decision-sufficient consult for the question that matters* — the tiered-oracle premise made concrete, dependency-free. | [![ED2 smart-when — entropy-gated consultation is worse than fixed](figures/ed2_smart.png)](figures/ed2_smart.png)<br>**The flat model's decode-entropy is not a calibrated consult trigger (SPEC-7 / ED2-smart / H9).** The missing *when* axis of ED2: at a fixed interior budget `ρ`, does spending the consults on the steps the flat `M_θ` is least sure about (its constrained-decode entropy) beat spreading them evenly? **No — it is strictly *worse* than `fixed` (lift 0.08–0.12× at every budget).** Faithful horizon is a *prefix* property: `fixed` consults at step 0 to protect the prefix, while the entropy signal spends late and lets the model derail early. The flat decode-entropy is a decode-time artifact, not a calibrated belief — the standing H2/H9 negative carried into the distributed world, sharper than a tie. This localizes the smart-`π_c` lever to the (deferred) structured `M_θ`'s RSSM belief variance — the EH2 lesson, where the host's factored belief-variance *did* beat fixed ~2.2× where flat entropy could not. |
| [![EH8 privilege denial recall](figures/eh8_privilege.png)](figures/eh8_privilege.png)<br>**Aggregate faithfulness hides a security-critical denial gap (host EH8).** On a denial-heavy workload, overall privilege-faithfulness looks high (0.91 flat, 0.94 factored) — but that is the easy successes. The defensively-critical number is **denied recall** (does the model predict failures when truth fails?): the flat arm scores **0.000** (it *never* predicts an EPERM/EBADF — it would tell a defender every blocked action succeeded), the factored arm only 0.286. Structure helps, but predicting *denials* is the open security gap. | [![EH6 two-oracle H12](figures/eh6_two_oracle.png)](figures/eh6_two_oracle.png)<br>**A cheap security oracle is redundant but decision-sufficient (host EH6 / H12).** A symbolic privilege invariant ("no non-root process holds `/passwd`") vs the full state oracle: it catches **nothing** the full oracle misses (non-redundant rate **0**, by construction) — but in **95%** of the steps where the model's *full* prediction is wrong it still gets the **security verdict right**, at ~**3×** lower consult cost. Redundant for verification; cheaper and decision-sufficient for the question a defender asks. The host H12. |
| [![EH9 denial-weighted objective](figures/eh9_denial_weighted.png)](figures/eh9_denial_weighted.png)<br>**The free oracle closes the EH8 denial gap — data balance, not architecture (host EH9).** EH8's blindness to failures is largely a *data-balance artifact the oracle can label away for free*. Just adding the denial-carrying driver to training lifts flat denied recall **0.000→0.333** and factored **0.286→0.952**; then **oversampling** denials lifts flat further to **0.762** at 4× — at **no specificity cost** (it never cries wolf, ~0.98–1.0 throughout). But too much backfires (flat falls to 0.524 at 16×): the flat arm needs a *tuned* factor, while the structured arm saturates to perfect recall and is robust. | [![EH7 host model-invariance](figures/eh7_invariance.png)](figures/eh7_invariance.png)<br>**The floor+cliff is the loop's, not the model's — even in the hardest world (host EH7 / H22).** Swap the proposer (untrained, biased, trained) under the composed host oracle and the `H_ε(ρ)` shape is invariant: a near-zero floor across the `ρ` interior, the cliff only at `ρ=1`. The curve is a property of the verify-correct method, not any one `M_θ`. |
| [![EH-stream experience stream vs batch](figures/eh_stream.png)](figures/eh_stream.png)<br>**The stream doesn't beat the batch — but replay is what saves it, and the plasticity probe says why (host EH-stream / H15 / HW-4).** At *equal compute*, the Era-of-Experience stream loses to the offline batch (one-step exact 0.47 vs 0.54, free `H_ε` 1.7 vs 4.0) — the manifesto's promise does not survive contact with the oracle at this scale (H15 negative). But **experience replay is decisively load-bearing**: it rescues the stream from collapse (0.47 vs the no-replay 0.10), and the **plasticity probe localizes why** — the no-replay stream *loses plasticity* (0.77 vs 0.95), replay keeps it high. The §2.5 "replay fixes forgetting *and* plasticity loss" lesson, grounded. | [![EH6 counterfactual H16](figures/eh6_counterfactual.png)](figures/eh6_counterfactual.png)<br>**Counterfactual replay is just more data for plain supervision — the same null the network found (host EH6 / H16).** The total oracle re-runs any process tree with one syscall changed, for free. Training on those counterfactual branches *does* beat the base trajectory on held-out intervention-exactness (0.46 vs 0.34) — but it **loses to a matched-volume control** of plain trajectory data (0.59). So the lift is **data volume, not counterfactual structure**: for next-state supervision a counterfactual is just another labeled transition. H16 is a null beyond volume — now confirmed *world-agnostic* (it matched the network's EN6/H5). |

| | |
|---|---|
| [![EN4 graph-vs-flat](figures/en4_graph_vs_flat.png)](figures/en4_graph_vs_flat.png)<br>**Structure helps (EN4 / H11).** The message-passing graph+RSSM world model beats the flat serializer by **+16.5 pts** one-step token accuracy and **+30.6 pts** delta-exact rate — and the gap *widens* on the honest metric. | [![EN8 oracle-grounded SSL](figures/en8_grounding.png)](figures/en8_grounding.png)<br>**The oracle removes the collapse tax (EN8 / H23).** Ablate JEPA's EMA+VICReg crutches and a learned target collapses (eff-rank 41.8→13.4); an **oracle-anchored** target stays healthy (25.8) — the external referent the field lacks. *Scaled to 200 hosts (SPEC-9 S1): the gap stays disjoint-positive everywhere but **attenuates** — real at every scale, diminishing.* |
| [![EN9 oracle hard-negatives](figures/en9_contrastive.png)](figures/en9_contrastive.png)<br>**Exact negatives beat statistical ones where it counts (EN9 / H25 / H5).** VICReg and the oracle both stop contrastive collapse — but only the oracle's *counterfactual* negatives teach which intervention leads where: branch-retrieval top-1 **0.519 vs VICReg's 0.282**. The statistical regularizer is full-rank but interventionally blind. *⚠ Scaled (SPEC-9 S2): this lift **reverses** by 100–200 hosts at higher capacity with a fixed `k=8` (VICReg overtakes) — but the reversal is a **negative-count artifact that recovers**: scaling `k_negatives` 8→32 flips it back to disjoint-positive. Real lift; feed it negatives that scale with the world.* | [![EN1 / K4 the floor](figures/en1_curve.png)](figures/en1_curve.png)<br>**A floor, not a knee (EN1 / K4 / H8).** Faithful horizon vs consultation budget `H_ε(ρ)` is flat-then-cliff on *both* worlds: consultation budget alone does not buy horizon. The honest negative that drove every later design choice. |
| [![EN3 probe efficiency](figures/en3_operators.png)](figures/en3_operators.png)<br>**What you consult beats whether (EN3).** Under partial observation a cheap one-host **probe + belief filter** earns **~2.3×** more faithful horizon per oracle-bit than full consultation — active sensing is a real lever. | [![EN7 model-invariance](figures/en7_invariance.png)](figures/en7_invariance.png)<br>**The shape is the loop's, not the model's (EN7 / H22).** Run the *same* loop with four proposers (null, flat transformer, graph+RSSM, oracle-backed): the three imperfect ones share **one shape — floor + cliff, no knee.** The proposer sets the floor *height* (graph > flat > null); the loop sets the *shape*. The no-knee verdict is model-agnostic. |
| [![EH1 composed-host curve](figures/eh1_curve.png)](figures/eh1_curve.png)<br>**The floor+cliff holds in the *composed* world (host EH1 / HC6).** First result in the host world (process table + fd tables + embedded fs): the composed `H_ε(ρ)` is the same flat-then-cliff shape as the filesystem and network worlds — `ρ=0` drifts in <1 step, the cliff only at `ρ=1`. The no-knee verdict generalizes across all three worlds. | [![EH1 composition law H13](figures/eh1_composition.png)](figures/eh1_composition.png)<br>**Whole-machine faithfulness is *coupled*, not independent (host EH1 / H13).** The headline-new measurement: composed per-step acceptance (orange) sits *below* the multiplicative/independence prediction `∏ aᵢ` (blue) — the flat baseline's subsystem failures are **anti-correlated**. Modeling subsystems independently is the wrong bet; coupling is load-bearing. The honest negative that licenses the factored interaction-graph arm. |
| [![EH4 factored vs flat](figures/eh4_factored_vs_flat.png)](figures/eh4_factored_vs_flat.png)<br>**Structure helps in the bundle world too — but the coupling is real (host EH4 / H11 / H13).** The factored interaction-graph arm (GNN+RSSM over the process spine's lineage + shared-file edges) beats the flat serializer **~6.6× on delta-exact (0.058→0.388)** and **~5.3× on composed acceptance** — the host echo of EN4. But *both* stay **coupled** (composed still below the independence floor): modeling the references explicitly buys a lot of faithfulness, yet does **not** uncouple the composition — so H13's coupling is genuine, not a flat-arm artifact. | [![EH3 per-subsystem efficiency](figures/eh3_operators.png)](figures/eh3_operators.png)<br>**Which subsystem you verify is a lever — the cheapest beats the weakest (host EH3).** At equal budget the full-consult operators coincide on `H_ε`, but a per-subsystem probe earns **~3.7× more faithful horizon per oracle-bit**. The twist: the *cheapest* subsystem (`fd`) wins, not the H13-*weakest* (`proc`) — so efficient `π_w` is a cost-vs-consequence tradeoff, not "verify the worst." |
| [![EH2 smart consultation policy](figures/eh2_policies.png)](figures/eh2_policies.png)<br>**A calibrated signal finally makes *smart* consultation pay (host EH2 / H9).** The program's first smart-`π_c` **positive**. At equal budget the **flat** arm reproduces the standing negative (uncertainty/drift *worse* than fixed — decode entropy mis-localizes error), but the **factored** arm's **RSSM belief variance** makes uncertainty-triggered consultation earn **~2.2× more faithful horizon than fixed** (5.8 vs 2.6 steps). *When* to consult finally beats spreading the budget evenly — because the calibrated signal knows where the model is wrong. | [![EH-H14 concurrency dial](figures/eh_h14_interleaving.png)](figures/eh_h14_interleaving.png)<br>**Concurrency is a measurable dial, not a binary wall (host EH-H14 / H14).** The host world's defining result — the thing the filesystem and network worlds *cannot* study. A chaos-seeded scheduler interleaves a multi-thread workload; free-running faithful horizon **degrades monotonically with interleaving entropy** (~8×, from `H_ε`≈12.5 at the recorded/sequential regime to ≈1.5 under chaos), and the recorded end recovers it. The **first quantification of HW-1's cost** — thread interleaving, the record/replay literature's named-unsolved nondeterminism source, made a continuous knob the chaos seed sweeps. |

## What we've found so far

Every number below is bit-exact and oracle-grounded, regenerates from config + seeds, and is reported
with its honest negative (the v0 norm — [SPEC §9–10](docs/specs/SPEC.md)). The interesting results,
front-loaded:

### 1. Structure helps — and helps *more* on the honest metric (network EN4 / H11)

The message-passing **graph + RSSM** world model beats the **flat** serializer on never-trained eval
seeds by **+16.5 pts** of one-step token accuracy *and* **+30.6 pts** of *delta-exact* rate (did the
model freely decode the **exact** true edit set this step?). The gap *widens* on the stricter metric —
token accuracy understates how much the graph inductive bias buys.

![EN4 graph-vs-flat (H11): token accuracy, delta-exact, and faithful horizon across arms](figures/en4_graph_vs_flat.png)

| arm | one-step token acc | **delta-exact** rate | `H_ε`(ρ=0), ε∈{0,.05,.1} |
|---|---|---|---|
| flat-Markov (NW4) | 0.673 | 0.264 | 0 / 0 / 0 |
| **graph + RSSM (NW8)** | **0.838** | **0.569** | 0 / 0 / 0 |
| graph + RSSM + noise lever (§6.3) | 0.828 | 0.556 | 0 / 0 / 0 |
| graph + RSSM + self-forcing lever (§6.3) | 0.803 | 0.500 | 0 / 0 / 0 |

### 2. …but better one-step prediction does **not** yet buy free-running horizon (the honest negative)

The right-hand column above is the catch: `H_ε(ρ=0) = 0` for **every** arm even at ε=0.1 — each drifts
on the first unaided step. The delta-exact number *quantifies* why: at 0.569 per-step exactness,
whole-delta correctness decays geometrically over unaided steps and first-exceedance is discrete (one
wrong edit spikes the graph divergence past ε in a single step). **Both pre-registered exposure-bias
levers — noise-injection and self-forcing / scheduled sampling — have now run, and both land the *same*
banked negative** (a small one-step dip, no horizon yet). That the random-corruption lever and the
model's-own-drift lever behave identically is itself informative: the wall is **localized to the
one-step→horizon conversion**, not the per-step learner (the arm fits teacher-forced to >0.9). This
routes the remaining budget to scale, the latent-overshooting objective, and *objective grounding*
([SPEC-8](docs/specs/SPEC-8.md)), not to more input-distribution patches. **Scale partly resolves this
(SPEC-10 / [§30](#30-capacity-buys-free-running-horizon--and-the-verdict-is-cross-world-spec-10--hs1hs2--h26)):** the `ρ=0` floor measured here at tiny scale *lifts* with capacity+data — to ~16–19 steps on
the network and ~5 on the harder host world — so this negative is in substantial part an
under-resourcing artifact of the committed (tiny) arm, not a fundamental compounding wall. What it does
*not* yet show is a favorable *consultation* knee, which stays open.

### 3. Cheap, localized sensing is the efficient way to buy horizon (network EN3)

Under partial observation the oracle has two modes — **full** (the whole next state) and **probe** (one
host's local view). v0's correction operators were indistinguishable on `H_ε`; in the network world a
cheap one-host **probe + belief filter breaks that collapse and earns ~2.3× more faithful horizon per
oracle-bit** than full consultation. *What* you consult, not just *whether*, is a real lever.

### 4. The consultation curve has a floor, not a favorable knee — on *both* worlds (the result that drove the design)

The prime directive is to plot `H_ε(ρ)` — faithful horizon vs. consultation budget — and ask whether a
little consultation buys a lot of horizon. On the single-filesystem world (left) and the network world
(right), the answer so far is a **floor + cliff, not a knee**:

| filesystem (SPEC-2.1 K4) | network (SPEC-5 EN1) |
|---|---|
| ![K4 knee](figures/k4_knee.png) | ![EN1 curve](figures/en1_curve.png) |

This is the **C-knee / H1–H8 honest negative**: discrete per-step errors make first-exceedance `H_ε`
reset-resistant, so consultation budget alone does not lift it. Far from a dead end, this negative is
*precisely* what licensed the network world (gradual drift, partial observability, a calibrated belief
signal) and what makes the NW8 graph arm + drift levers + SPEC-8 objective-grounding load-bearing
rather than speculative. Negatives here are **trustworthy** because the verdict is oracle-grounded, and
a refutation is frequently the deeper result.

### 5. The oracle generates its own training data — perfectly labeled, for free (SPEC-8 OG1/OG2, shipped)

Because the oracle is total, it is a *data factory*, not just a checker. The deterministic machinery is
built and property-tested ahead of the GPU runs: **oracle targets + the decidable/residual partition**
(mask the bits the oracle fixes, learn only the genuine residual), and a **hard-negative / counterfactual
factory** (one-edit-wrong successors and action-branch counterfactuals, each exactly labeled). This is
"oracle-grounded self-supervision" — pressing ground truth into the *bulk* of the cake, not just the RL
cherry.

### 6. …and putting it in the bulk **removes JEPA's collapse tax** (network EN8 / H23, shipped)

The first result from spending that data factory: a JEPA-style latent predictor trained with an
**oracle-anchored target** — a fixed projection of the *true next state*, an external referent with full
variance by construction — keeps its representation healthy *with the EMA-target + VICReg
collapse-prevention machinery ablated*, where the standard learned (EMA) target collapses.

![EN8: oracle-grounded SSL — the collapse axis (H23) and the objective axis (H24)](figures/en8_grounding.png)

| JEPA target | collapse machinery | embedding std | effective rank (d=48) |
|---|---|---|---|
| learned (EMA) | **on** (the baseline) | 0.557 | 41.8 |
| learned | **off** (ablated) | 0.276 | **13.4** (collapsed) |
| **oracle-anchored** | **off** (ablated) | **0.528** | **25.8** (healthy) |

Ablating the machinery collapses the learned target (effective rank 41.8 → 13.4); the oracle-anchored
target holds at 25.8 — roughly twice as healthy, no crutches. This is the strongest possible form of
SPEC-8's thesis: **the collapse-prevention machinery is a workaround for a missing oracle**, and where the
oracle exists the workaround is largely unnecessary — a fact the oracle-free field structurally cannot
establish. The companion **objective axis (H24)** — train only the genuinely-uncertain *residual* bits,
let the oracle supply the decidable ones — is an honest near-tie at this smoke scale (residual-token
accuracy 0.426 vs 0.463 raw-likelihood), the pre-registered negative branch: the decidable part is cheap to
learn until the worlds grow. A split EN8 verdict, like EN4 — and every cell is bankable under the oracle.

### 7. Exact negatives don't just stop collapse — they teach *interventions* (network EN9 / H25 / H5, shipped)

EN8 grounded the *predictive* target on the oracle; **EN9** grounds the *contrastive* one and spends the
OG2 hard-negative factory. A contrastive predictor over the same graph summary, with the only anti-collapse
referent varying across three cells: *none* (naked BYOL), *vicreg* (the field's statistical regularizer), or
*oracle* (InfoNCE against exact one-edit-wrong and counterfactual negatives). Two readouts — representation
health, and **interventional fidelity**: can the representation map each intervention `a'` to its true
successor `O(s, a')` (scored as branch-retrieval top-1 / MRR)?

![EN9: oracle hard-negative contrastive — the collapse axis (H25) and the interventional axis (H5)](figures/en9_contrastive.png)

| anti-collapse referent | embedding std | effective rank (d=48) | intervention top-1 | intervention MRR |
|---|---|---|---|---|
| none (naked BYOL) | 0.276 (collapsed) | 13.4 | 0.214 | 0.426 |
| vicreg (statistical) | 0.499 | **39.0** | 0.282 | 0.500 |
| **oracle (exact)** | **0.699** | 31.4 | **0.519** | **0.694** |

The split is the finding. On **collapse (H25)** the exact referent only *matches* the statistical one — both
hold the representation open, and VICReg's covariance term even buys higher effective rank (39.0 vs 31.4). But
on **intervention (H5)** the oracle wins decisively: its counterfactual negatives nearly double VICReg's
branch-retrieval fidelity (top-1 0.519 vs 0.282). The honest, sharper reading: **VICReg keeps the
representation full-rank but interventionally blind** (0.282 is barely above the naked 0.214), while the
oracle makes it faithful to the very branches the loop will be asked to predict. A statistical regularizer can
prevent collapse; it structurally cannot teach counterfactual structure it has no access to. This is the
H5 / change-safety lift arriving through the *self-supervised* objective rather than the RL cherry — a third
split verdict, every cell bankable under the oracle.

> **⚠ Scaling result (SPEC-9 S2 — the surface run, then the fix).** This lift is **scale-sensitive**, and
> measuring it is the point. On the 25–200-host surface (× `d64`/`d128` × 3 seeds) the oracle's top-1
> advantage is disjoint-positive at the smallest world + smaller capacity (25 hosts/`d64`: +0.106) and
> **reverses** at scale with the fixed `k_negatives=8` — VICReg *overtakes* the oracle at 100 hosts/`d128`
> (−0.086 [−0.113, −0.060]) and 200/`d128` (−0.094 [−0.111, −0.067]). The pre-registered diagnosis — a
> **negative-count artifact** — then proved correct: the LS-S2 sweep
> ([`en9_negatives.png`](figures/en9_negatives.png)) shows that at 100/`d128`, scaling `k_negatives` 8→16→32
> flips `lift_top1` −0.075 → +0.017 → **+0.032 [0.024, 0.044]** (disjoint-positive again). So H5 is
> **confirmed at small scale, reverses at scale with a fixed negative count, and recovers when negatives
> scale** — the magnitude is modest, so the rule is *feed negatives that scale with the world*. The oracle
> let us see a smoke-scale win reverse **and** repair, which is exactly what it is for. (This also refuted my
> own prior that more one-edit negatives wouldn't help — they did, by sharpening the contrastive geometry.)

### 8. Which wins survive scaling — the honest mixed verdict (SPEC-9)

Because the oracle labels for free, world size is a *learner-compute* choice, not a labeling-budget one
([SPEC-9](docs/specs/SPEC-9.md)) — so the smoke-scale EN8/EN9 wins can be carried up an **8× world-size
range on a single 32 GB machine** and stress-tested with bootstrap CIs. The surface (25→200 hosts ×
`d_model` ∈ {64, 128} × seeds) is the most important thing we ran, because it shows the three results
survive *unevenly*, and the unevenness is the finding:

| ![EN8 scaling surface](figures/en8_surface.png) | ![EN9 scaling surface](figures/en9_surface.png) |
|---|---|
| **H23 collapse gap — persists but attenuates (S1).** Disjoint-positive at all 8 cells (the oracle's anti-collapse advantage is real across the whole range and both capacities) but **shrinks** with scale: eff-rank gap 13.4→6.9→4.1→**2.2** over 25→100→200→**300** hosts at `d128` (the last is the **LS3 hero instance** — the largest oracle-grounded world proven on one machine; still disjoint-positive at 300 hosts, [`en8_ls3_hero.csv`](figures/en8_ls3_hero.csv)). Real everywhere, diminishing. | **H25/H5 interventional lift — reverses at fixed `k`, then *recovers* when negatives scale (S2).** Disjoint-positive at 25 hosts/`d64` (+0.106); it **flips negative** at 100/`d128` (−0.086) and 200/`d128` (−0.094) with the fixed `k_negatives=8` — VICReg overtakes. But the reversal is a **negative-count artifact**: scaling `k_negatives` 8→32 at 100/`d128` flips `lift_top1` back to disjoint-positive (+0.032 [0.024, 0.044], [`en9_negatives.png`](figures/en9_negatives.png)). The H5 lift is real; it must be fed negatives that scale with the world. |

The third axis, **H24 (residual objective)**, is **regime-dependent** ([`en8_capacity.png`](figures/en8_capacity.png)):
masking the oracle-decidable bits `D` in the *loss* helps only in a narrow window (high capacity + moderate
residual + small world) and **hurts where `R` is tiny**, because masking removes *beneficial multi-task
training signal* rather than freeing capacity. What is bounded is the *training-objective* partition; the
*inference-time* partition (the oracle simply supplies `D`, the model is never trusted on it) is untouched.

This is the single most valuable thing the scaling bought, and the full arc is the lesson: a headline
(EN9/H5) that looked clean at smoke scale **reversed** under an honest CI sweep at 100–200 hosts — and then,
when the pre-registered lever was tried, **recovered** (scaling `k_negatives` 8→32 flips the lift back to
disjoint-positive). The deterministic oracle is what let us *see* both the reversal and the fix. A win
caught reversing and then honestly repaired is worth far more than one asserted and never stress-tested.
Each verdict carries its next lever (S1: normalize + grow `d_model`; **S2: scale negatives with the world —
now demonstrated**; S3: keep `D` in the loss, oracle-own it at inference).

![EN9 S2-recovery: scaling k_negatives recovers the H5 lift at 100 hosts/d128](figures/en9_negatives.png)

**The local envelope, measured (32 GB M4, CPU — the cost is the learner, not the labeler):**

| hosts `N` | oracle data build | one training run | peak RAM | binding constraint |
|---|---|---|---|---|
| 50 | 0.06 s | ~25 s | 351 MB | — |
| 100 | 0.13 s | ~50 s | 492 MB | — |
| 200 | 0.46 s | ~140 s | 779 MB | wall-clock `O(N²)` message passing |
| 400 | 1.76 s | ~8 min | 1.95 GB | wall-clock (memory has huge headroom) |

The oracle (the labeler) is effectively free at every size; what binds is the learner's `O(N²)` message
passing, not memory and not labels. MPS was *slower* than CPU at this model size (kernel-launch overhead),
so CPU is the bit-deterministic default. Sweep preset `N ≤ 200`; hero preset `N ~400–512`.

### 9. The no-knee shape is the loop's, not the model's (network EN7 / H22)

The project's most general claim is that the *loop*, not the proposer, governs the `H_ε(ρ)` curve — that
deterministic verification is a **model-agnostic primitive**. EN7 tests it by dropping four proposers into
the *same* loop and re-plotting the curve (5 hosts, ε=0.05, T=24, 3 seeds × 2 difficulties, CIs):

![EN7 / H22: the floor+cliff H_ε(ρ) shape is invariant across proposers](figures/en7_invariance.png)

| proposer | ρ=0 | ρ=0.1 | ρ=0.3 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|
| null (empty delta) | 0.0 | 1.2 | 1.2 | 1.3 | 24.0 |
| flat (NW4 transformer) | 0.0 | 1.0 | 1.0 | 1.0 | 24.0 |
| graph (NW8 GNN+RSSM) | 0.0 | 3.2 | 4.3 | 4.7 | 24.0 |
| oracle-backed (perfect) | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 |

**H22 supported in kind.** The three imperfect proposers share **one shape — floor + cliff, no knee.** The
proposer's per-step competence sets the *floor height* (graph 3.2–4.7 > flat 1.0 > null), but the loop sets
the *shape* — none shows a favorable knee; all reach the ceiling only at ρ=1. So the EN1/K4 "no-knee"
verdict is **not** an artifact of the flat transformer: it reproduces across materially different
architectures, which is exactly the model-agnostic-primitive claim. The oracle-backed proposer (24
everywhere) is the degenerate ceiling. *Honest caveat:* this is not matched competence (graph is clearly
stronger), so the load-bearing evidence is the *shared shape across differing competence* — what moves with
the proposer is the floor, what stays is the shape.

### 10. Online self-healing (TTT) does not lift the floor — yet (network EN5 / H7, an honest null)

EN7 showed the floor is model-invariant; **EN5** tests the one lever that changes the model *during* the
rollout: when the loop consults the oracle, the revealed `(state, action) → true-delta` is a free labeled
example, so take a small in-rollout gradient step on it (test-time training / self-healing). Does adapting
the weights mid-rollout lift the curve where frozen weights cannot?

![EN5 / H7: online self-healing (TTT) does not lift H_ε(ρ) at this scale](figures/en5_selfheal.png)

| arm | ρ=0 | ρ=0.1 | ρ=0.3 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|
| supervised (frozen) | 0.0 | 3.2 | 4.3 | 4.7 | 24.0 |
| +ttt (single-example) | 0.0 | 3.2 | 3.5 | 4.7 | 24.0 |
| +ttt-replay (replay buffer) | 0.0 | 3.2 | 3.5 | 4.7 | 24.0 |

**A robust null — and the pre-registered lever was run, not just promised.** *Both* self-healing arms —
the minimal single-example update **and** the replay-buffer budget (a growing buffer of corrections, 5
minibatch updates per consult) — match the frozen baseline; neither changes *where* the first drift
happens, so `H_ε` is unmoved. The richer budget does not rescue H7. This is consistent, not surprising:
EN4 localized the wall to the **one-step→horizon conversion** and EN7 showed the floor is model-invariant,
so online adaptation — in either form — can't move the binding per-step competence. **Where this routes
the floor:** self-healing-as-floor-lifter is closed at this scale; the floor's real levers are **scale
([SPEC-9](docs/specs/SPEC-9.md)) and objective grounding ([SPEC-8](docs/specs/SPEC-8.md))**, not
adaptation. The [`online_update`](src/verisim/netmodel/graph_train.py) primitive ships for the
host/distributed worlds where horizons are longer.

### 11. Counterfactual grounding helps the contrastive objective, not supervision (network EN6 / H5)

The oracle generates **counterfactual branches for free** — the exact next state `O(s, a')` of actions
not taken. EN6 asks whether *training* the delta predictor on them improves prediction of **interventions**
(the change-safety question a network defender asks). A rigorous **3-arm, matched-example-count** design
separates the counterfactual signal from raw volume:

![EN6 / H5: counterfactual grounding vs a matched-volume control](figures/en6_counterfactual.png)

| arm | intervention delta-exact | change-safety (reachability) |
|---|---|---|
| trajectory | 0.551 | 0.924 |
| trajectory-more (volume control) | **0.604** | 0.933 |
| +counterfactual | 0.588 | 0.935 |

**H5 is a null for the predictive model — beyond volume.** `+counterfactual` (0.588) does *not* beat the
volume control `trajectory-more` (0.604) — marginally lower, CIs overlapping; change-safety (~0.93) is
indistinguishable. So the lift over the base is **data volume, not counterfactual structure** — for plain
next-state supervision, a counterfactual is just another labeled transition. The control arm is what makes
this honest. **The coherent contrast with EN9:** counterfactual *negatives* **did** lift the *contrastive*
representation (structure matters there) — but counterfactual *examples* don't lift plain *supervision*. So
H5 is objective-dependent. **Mild standalone positive:** change-safety (~0.93) ≫ delta-exact (~0.58) across
all arms — the model predicts the *reachability effect* of interventions far better than the exact delta,
which is the metric the defense use case cares about. *(The two-oracle axis H12 is measured in §12.)*

### 12. The control-plane oracle is redundant for verification but cheaper + decision-sufficient (network EN10 / H12)

The two-oracle axis: alongside the data-plane oracle (exact next state), a Batfish-style
[**control-plane oracle**](src/verisim/netoracle/control_plane.py) returns only the **reachability**
truth. H12 asks whether it's a *non-redundant* signal — does it catch reachability errors a full-state
consult misses? On held-out transitions of the trained graph arm:

![EN10 / H12: the control-plane oracle is redundant for verification but cheaper + decision-sufficient](figures/en10_two_oracle.png)

| metric | mean | 95% CI |
|---|---|---|
| data-plane bits-to-correct (full delta) | 14.4 | [11.8, 17.2] |
| control-plane bits-to-correct (reachability) | 0.4 | [0.20, 0.54] |
| **non-redundant rate** | **0.000** | [0.000, 0.000] |
| control-plane-sufficient rate | 0.30 | [0.22, 0.36] |
| consult-bits ratio (control / data) | 0.35 | [0.20, 0.49] |

**H12 ("non-redundant") is refuted, provably.** Non-redundant rate is **exactly 0** — the control-plane
oracle never catches a reachability error the full-state oracle misses, because reachability is a
deterministic function of the state. **But the experiment reframes its value:** it's ~38× cheaper to
satisfy (0.4 vs 14.4 bits-to-correct), a consult costs ~35% of a full one, and the model gets reachability
*exactly right in ~30% of the steps where its full delta is wrong*. So the control-plane oracle is
redundant as a *verification signal* but a **cheaper, decision-relevant consultation** for the
change-safety question — the tiered-oracle premise [SPEC-7](docs/specs/SPEC-7.md) builds on. The oracle
ships as a property-tested deterministic component (the NW0/OG1 "core-first" discipline).

### 13. The third world, and a new question: whole-machine faithfulness is *coupled* (host EH1 / H13, HC6)

The **host world** (SPEC-6) is the first world whose state is not one tree (filesystem) or one graph
(network) but a **bundle of coupled subsystems** — a process table, per-process fd tables, and the
embedded v0 filesystem, sharing references. Its oracle *composes* the v0 FS sub-oracle verbatim, so a
`write`'s bundle delta literally embeds the FS sub-oracle's own delta. With the deterministic core
(HC0–HC3), the flat `M_θ` (HC4), and the composed loop (HC5) in place, the prime-directive experiment
(HC6) runs and asks two questions.

**The composed `H_ε(ρ)` curve is the same floor+cliff (the no-knee verdict generalizes).** Train the
flat host `M_θ`, sweep the composed loop over `ρ × ε × difficulty × seed`: at `ρ=0` the model drifts in
under one step (the honest floor), the interior is near-flat, and the cliff to `H_ε=T` appears only at
`ρ=1`. The same shape as the filesystem (K4) and network (EN1) worlds — now in a coupled, composed
world. Consultation budget alone does not buy horizon; this is **model- *and* world-invariant** (the
host analogue of EN7/H22).

![EH1 composed-host H_ε(ρ): the floor+cliff shape, reproduced in the bundle world](figures/eh1_curve.png)

**The headline-new measurement — the composition law (H13) — reads `coupled`.** This is the question
only the bundle world can ask: *is whole-machine faithfulness predictable from the faithfulness of its
parts?* For each subsystem `i` measure the per-step (teacher-forced) acceptance `aᵢ` — the fraction of
one-step predictions that keep subsystem `i` faithful — and compare the **composed** acceptance `a`
(every subsystem faithful at once) against two candidate laws: multiplicative `a ≈ ∏ aᵢ` (failures
independent) and weakest-link `a ≈ min aᵢ` (failures coincide).

![EH1 composition law (H13): composed acceptance sits below the independence floor — coupled](figures/eh1_composition.png)

| difficulty | composed `a` | `∏ aᵢ` (independent) | `min aᵢ` (weakest-link) | verdict |
|---|---|---|---|---|
| low (`forky`) | 0.083 | 0.248 | 0.483 | **coupled** |
| high (`adversarial`) | 0.067 | 0.196 | 0.417 | **coupled** |

Composed acceptance sits *below* the multiplicative floor — the flat baseline's subsystem failures are
**anti-correlated** (it fails *different* subsystems on *different* steps, so the whole machine is
faithful even less often than independence predicts). **Modeling the subsystems independently is the
wrong bet; the coupling is load-bearing.** That is the honest negative HC6 was built to surface, and it
is exactly what licenses the next step: the **factored interaction-graph arm** (HC4 incr-2, the DD-H1
alternative the flat baseline is the floor for), which is built to model the cross-subsystem references
the flat serializer flattens away. (The per-step acceptance is measured teacher-forced, not on the
free-running rollout: a compounding subsystem that drifts once would otherwise read as permanently
unfaithful, making `aᵢ` bimodal rather than a rate — SPEC-6 §9.2.)

### 14. *Which subsystem* you verify is a real efficiency lever — but the cheapest, not the weakest (host EH3 / HC7)

The composition being coupled (§13) raises the operational question: given a consultation budget, *which
subsystem's truth should you buy?* (the host's new `π_w` axis, §8.2). EH3 fixes the policy and budget `ρ`
and compares correction operators at **equal `ρ`** (the host analogue of network EN3). The three
full-consult operators (`hard_reset`/`residual`/`projection`) all snap the whole bundle to truth, so
their `H_ε` is **identical** — the v0 full-truth identity. A per-subsystem `SubsystemFilter` corrects
only one subsystem, so it corrects *strictly less* (lower `H_ε`) but spends *far fewer* oracle-bits — and
the cost lens is the real verdict:

![EH3 host operators at equal budget: full operators coincide; per-subsystem correct less](figures/eh3_operators.png)

| operator | `H_ε` | oracle-bits / consult | **`H_ε` per oracle-bit** | vs full |
|---|---|---|---|---|
| `hard_reset` = `residual` = `projection` (full) | 1.10 | 75.3 | 0.0146 | 1.0× |
| `subsystem_proc` (target the H13-weakest link) | 0.40 | 20.8 | 0.0192 | 1.3× |
| `subsystem_rr` (round-robin) | 0.40 | 20.2 | 0.0198 | 1.4× |
| **`subsystem_fd` (target the cheapest)** | 0.90 | 16.8 | **0.0536** | **~3.7×** |

**Per-subsystem consultation earns up to ~3.7× more faithful horizon per oracle-bit than full** — *what*
you verify is a real lever, the host echo of EN3's ~2.3×. But the honest twist is the punchline: the
winner is the **cheapest** subsystem (`fd`, fewest facts), **not** the **weakest-link** subsystem (`proc`,
the one H13 said dominates the coupling). Targeting the weakest by the static heuristic barely beats full.
So efficient `π_w` is a genuine **cost-vs-consequence tradeoff** — exactly the optimization a *smart* `π_w`
(the remaining HC7 work) exists to solve, and a clean negative for "just verify the worst subsystem."

### 15. Modeling the composition explicitly helps a lot — but the coupling is real, not an artifact (host EH4 / H11 / H13)

H13 said the flat baseline's whole-machine faithfulness is *coupled*. Two readings were possible: either
the coupling is a real property of the host dynamics, or it is an artifact of the flat serializer
flattening the cross-subsystem references away. EH4 settles it by building the **factored interaction-graph
arm** (DD-H1) — the structured alternative the flat arm is the floor for — and comparing the two on
*identical* data. The factored arm featurizes the bundle as a **process-interaction graph** (process-
indexed nodes; two edge sets — the **lineage** fork-tree and the **shared-file** coupling, which fold the
fd/fs subsystems onto the process spine), message-passes over it with an RSSM belief, and decodes the
bundle delta under the *same* grammar as the flat arm — so the only thing that changes is whether the
composition is modeled or flattened.

![EH4 factored interaction-graph vs flat M_θ: delta-exact and the composition law](figures/eh4_factored_vs_flat.png)

| arm | delta-exact (free decode) | composed `a` | `∏ aᵢ` | `min aᵢ` | verdict |
|---|---|---|---|---|---|
| flat serializer (HC4 incr-1) | 0.058 | 0.075 | 0.223 | 0.450 | coupled |
| **factored graph (HC4 incr-2)** | **0.388** | **0.396** | 0.470 | 0.750 | coupled |

**Structure helps, decisively — the host echo of EN4/H11:** the factored arm is **~6.6×** more
delta-exact and **~5.3×** higher on composed acceptance. Modeling the references the flat arm flattens is
a large lever, exactly as the network graph arm was. **But the coupling survives:** the factored arm's
composed acceptance (0.396) is *still below* its own independence floor `∏ aᵢ` (0.470), so the verdict
stays **coupled** for both arms. The composition being coupled is therefore a **genuine property of the
host dynamics**, not an artifact of flattening — which sharpens H13 from "the flat model couples" to "the
world couples, and even the structured model only attenuates it." That residual coupling is the standing
target for the smart `π_w` (§17) and the per-subsystem decode heads (§17, EH5-heads — a negative).

### 16. A calibrated uncertainty signal finally makes *smart* consultation pay (host EH2 / H9)

Across v0 and the network world, *when* to consult was a negative: uncertainty- and drift-triggered
policies — spending the oracle budget on the steps the model flags as uncertain — did **not** beat a
dumb fixed interval, because the only available signal was the flat model's **decode entropy**, which
mis-localizes where the model is actually wrong (the standing "H2-negative"). SPEC-6 §8.1 conjectured
the factored arm's **RSSM belief variance** — an uncertainty *calibrated by construction* (§6.2), not a
decode-time artifact — would be the signal that finally makes smart `π_c` pay. EH2 tests it: both arms,
all three policies, at equal budget.

![EH2 consultation-policy comparison: flat decode-entropy vs factored belief-variance, three policies](figures/eh2_policies.png)

| arm (uncertainty signal) | `fixed` | `uncertainty` | `drift` |
|---|---|---|---|
| flat (decode entropy) | 1.1 | 0.5 | 0.4 |
| **factored (RSSM belief variance)** | 2.6 | **5.8** | 3.1 |

**This is the program's first smart-`π_c` positive.** The flat arm reproduces the negative exactly —
uncertainty (0.5) and drift (0.4) are *worse* than fixed (1.1), because high decode entropy does not
mark the steps that actually drift. But on the factored arm, **uncertainty-triggered consultation earns
~2.2× more faithful horizon than fixed** (5.8 vs 2.6 steps) at the same 7-consultation budget. The
difference is entirely the signal: the belief variance knows where the model is wrong, so spending the
budget there beats spreading it evenly. *When* to consult is a real lever — once you have an honest
estimate of *when you are about to be wrong*. (The full-truth identity still holds per consult; this is
about allocating a fixed budget across steps, the §8.1 H9 question.)

### 17. The other consultation axis: a smart *which-subsystem* policy gives a modest edge (host EH5 / H10)

EH2 settled *when* to consult; EH5 takes the host's second, novel axis — *which subsystem's truth to
buy* (`π_w`, §8.2). The factored arm now exposes **per-subsystem decode entropy** (each decoded token's
uncertainty bucketed into the subsystem of the op it belongs to), and an `UncertaintySubsystem` policy
spends each per-subsystem consult on the subsystem the model is **least certain** about. At equal budget,
against the static baselines:

| `π_w` policy | `H_ε` | oracle-bits | `H_ε` per bit |
|---|---|---|---|
| `fixed_fd` (cheapest subsystem) | 2.3 | 10.2 | **0.226** |
| `fixed_proc` (the H13-weakest) | 2.6 | 21.5 | 0.121 |
| `round_robin` (uniform) | 2.3 | 19.0 | 0.121 |
| **`uncertainty` (smart, information-gain)** | 2.6 | 20.1 | 0.129 |

**The honest read is a modest, mixed positive.** Targeting the uncertain subsystem ties the best raw
horizon (`fixed_proc`) while beating `round_robin` on *both* horizon and per-bit — so adaptive `π_w` is
a real, if small, lever (raw-horizon CIs overlap at this smoke scale). But the *cheapest* fixed
subsystem (`fd`) still wins pure bit-efficiency, exactly the cost-vs-consequence tension EH3 flagged:
the smart policy chases consequence (where the model is wrong) but ignores cost (how many bits the
subsystem costs to verify). The ideal `π_w` weights both — the standing target. What ships here is the
*apparatus* (per-subsystem uncertainty + the information-gain policy + the equal-budget harness); the
smoke-scale edge is reported as-is, not oversold.

**EH5-heads — the trained per-subsystem decode heads lose to the entropy bucket (an honest negative).**
The signal above is *post-hoc* (it reads the ambiguity of a constrained decode) and *sparse* (a
subsystem whose ops do not appear this step gets entropy 0, invisible to `π_w` even if the model is
quietly wrong about it). The named open HC7 lever was the calibrated alternative: a **trained
per-subsystem head** (opt-in `per_subsystem_heads`) that predicts *which subsystem the decoder will get
wrong* directly, regressed against the decoder's own per-subsystem error — the free oracle supplying the
target. EH5-heads ([`eh5_heads.py`](src/verisim/experiments/eh5_heads.py)) trains a *single*
heads-enabled arm exposing **both** signals on the **identical** proposer, so the comparison is
confound-free, and asks the §9.4 question: does each signal predict held-out per-subsystem error?

![EH5-heads: the trained per-subsystem head spends the most bits for the least horizon](figures/eh5_heads.png)

| `π_w` signal | Spearman(signal, per-subsystem error) | verdict |
|---|---|---|
| bucketed decode entropy | **+0.57** | well-calibrated |
| trained per-subsystem head | −0.02 | **uncalibrated** |

The head is essentially uncorrelated with held-out error (robust across noise levels), so the
head-driven `π_w` arm spends the **most** bits for the **least** horizon. The mechanism is clean: the
head's training target — the decoder's per-subsystem cross-entropy — collapses to ~0 on the overfit
training distribution, so it learns nothing about the deploy-time divergence that the entropy, measured
*on the actual decode*, tracks directly. This is the **per-subsystem echo of v0's H2 negative** (a
learned uncertainty proxy underperforms a decode-coupled one), and it **closes the open HC7 item with a
reproducible negative** rather than vague future work — the next lever is a head trained on the
deploy-time (drift) divergence, or scale, not this head.

### 18. The drift levers don't buy horizon here either — the same banked negative (host EH4-drift / §6.3)

The factored arm is far more one-step-accurate than flat (EH4), yet it still drifts at `ρ=0` — good
per-step prediction does not buy free-running horizon. That one-step→horizon gap is the program's
standing wall, and the §6.3 drift levers are its standard attack: **noise injection** (oracle-relabeled
state-noise augmentation) and **self-forcing** (re-roll on the model's own predictions, oracle-relabel),
both exploiting the oracle being a total, free teacher. EH4-drift trains the factored arm three ways on
identical seeds and asks whether either lever converts accuracy into horizon:

| arm | delta-exact | free-running `H_ε` (ρ=0), ε∈{0, .05, .1} |
|---|---|---|
| clean | 0.388 | 2.3 / 2.3 / 2.6 |
| +noise | 0.325 | 2.0 / 2.0 / 2.8 |
| +self-forcing | 0.379 | 2.0 / 2.0 / 2.2 |

**The same banked negative the network found.** Neither lever buys free-running horizon (all arms sit
at ~2–3 faithful steps, within noise), and noise injection slightly *lowers* one-step exactness. The
exposure-bias patches that should close the train/deploy gap don't, here — the wall is in the
one-step→horizon *conversion*, not the input distribution, so the remaining budget routes to scale and
objective grounding rather than more input patches (exactly the network's EN4 conclusion, now
replicated in the composed host world). A required ablation, run, and reported as the honest null it is.

### 19. Concurrency is a measurable dial, not a binary wall — the host world's defining result (host EH-H14 / H14)

The filesystem and network worlds run one action at a time; the host world's whole reason to exist is
that it has **concurrency** — multiple processes interleave, and the scheduler is a genuine
nondeterminism source. The record/replay literature (rr, Hermit) calls thread interleaving *the* one
unsolved determinism source (HW-1). SPEC-6 doesn't claim to solve it; it claims to make it a **measured
dial** (H14): a chaos-seeded scheduler interleaves a multi-thread workload (threads sharing files, so
the final content is last-writer-wins and order-sensitive; forks interleaved, so the pid a thread is
allocated depends on the global order), with an `interleave` knob from sequential (the *recorded*
regime) to fully random (*chaos*). The factored arm is trained on the recorded regime, then evaluated
free-running across the dial.

![EH-H14: free-running faithful horizon collapses as interleaving entropy rises](figures/eh_h14_interleaving.png)

| interleaving entropy (thread context-switch rate) | free-running `H_ε` (ρ=0, ε=0) |
|---|---|
| 0.17 — recorded / near-sequential | **12.5** |
| 0.28 | 5.2 |
| 0.42 | 2.0 |
| 0.59 | 1.5 |
| 0.71 — chaos | **1.5** |

**H14 confirmed.** Faithful horizon degrades **monotonically** with interleaving entropy — an ~8×
collapse from the recorded regime (≈12.5 steps) to chaos (≈1.5), and the low-entropy/recorded end
**recovers** it. This is the first quantification of HW-1's cost: concurrency is not a binary
"deterministic or not" but a continuum the chaos seed sweeps, and `H_ε(interleaving-entropy)` is its
curve. The honest alternative (a flat line — the model learns schedule-invariant effects, concurrency a
non-issue) would have been a *surprising* simplification of every downstream world; the monotone
collapse is the expected-but-now-quantified result, and it is the one experiment in the whole program
that only the host world could run. The scheduler ships dependency-free (the deterministic-core
discipline) and emits concrete, replayable schedules, so every point regenerates from `(workload,
interleave, chaos_seed)`.

### 20. The payoff: a verified whole-machine simulator an LLM agent calls (host §7 / HC8)

Every result above exists to build one thing: the cheap, faithful, verifiable machine an LLM agent
*reasons over*. A computer-use or cyber-defense agent acts on a whole host — "kill this process,
write this config, drop this privilege" — so the host world *is* the simulator it needs. `HostSimulator`
([`hostsim/`](src/verisim/hostsim/)) packages the loop's `M_θ` + the oracle into the object the agent
calls. It is not a competitor to the LLM; it is the layer the LLM is bad at (simulating host dynamics)
made fast and honest, leaving the LLM only what it is good at (natural-language intent → a syscall plan):

```
  LLM agent ── "kill the rogue proc, scrub /tmp" ──▶ syscall PLAN  (NL intent → plan; the LLM's job)
                                                         │
                          ┌──────────────────────────────┴───────────────────────────┐
            imagine(plan) │  Mθ rolls the plan in imagination — fast, NO oracle        │ the cheap draft
                          ▼                                                            │
                    predicted final state ── agent explores many plans cheaply ◀───────┘
                          │
            verify(plan)  │  Mθ imagination  ‖  oracle truth, step by step (on a budget ρ)
                          ▼
              PlanReport: plan-faithful-horizon H_ε  (how many steps to trust the draft, §17.8)
                          + task-oracle: does the plan hit the GOAL?  predicted ?= true  (the 3rd oracle)
```

Two calls, the propose-verify-correct loop lifted from the *syscall* level to the *plan* level:
`imagine(state, plan)` rolls `M_θ` forward with **no oracle** — Dreamer's "plan in imagination", the
draft an agent explores by the hundred; `verify(state, plan)` runs that imagination against the oracle
step by step and returns a `PlanReport` — the predicted-vs-true final state, the **plan-level faithful
horizon** (the §17.8 `H_ε`-for-a-plan: how many leading steps the agent can trust the draft before
re-grounding), the oracle cost paid, and — composing the **task oracle** (a `Goal` predicate, the §7
"third oracle") — whether the plan *achieves the goal* and whether the model **agrees with the oracle**
that it did. On a trained model this is exactly as honest as the rest of the program: for a plan the
model drifts on, `verify` reports a short plan-faithful-horizon and flags that the model and oracle
*disagree* on task success — so the agent knows precisely when to stop trusting the draft and spend an
oracle consultation. This is the SLM/LLM-complementarity thesis made executable: ground truth at scale,
on a budget, for the thing OSWorld-class agents actually do.

### 21. The deepest claim holds in the hardest world: the shape is the loop's, not the model's (host EH7 / H22)

The program's deepest claim is not about a model but a *method*: **deterministic verification is a
model-agnostic primitive** — the qualitative shape of the faithful-horizon-vs-consultation curve is a
property of the oracle-loop, not the proposer's architecture (the network EN7 established this; finding
[§11](#9-the-no-knee-shape-is-the-loops-not-the-models-network-en7--h22)). EH7 asks the sterner
question: does it survive the **hardest** world — the coupled, concurrent host bundle? It drops four
materially different proposers into the *same* HC5 loop and sweeps `H_ε(ρ)` for each.

![EH7: composed-host H_ε(ρ) is the same floor+cliff shape across all four proposers](figures/eh7_invariance.png)

| proposer | `H_ε` at ρ=0 | ρ interior (0.1–0.5) | ρ=1 |
|---|---|---|---|
| null (empty delta) | 0.0 | ~1.0, flat | 24 (=T) |
| flat transformer | 0.4 | ~1.1, flat | 24 |
| **factored graph+RSSM** | **2.3** | 2.3 → 4.2, flat-ish | 24 |
| oracle-backed (ceiling) | 24 | 24 | 24 |

**H22 holds in the composed world.** The three imperfect proposers — despite *materially different*
per-step competence (factored ≫ flat ≫ null, the EH4 ordering) — share **one shape**: a low floor
across the ρ interior, then the cliff to `H_ε=T` only at ρ=1. The proposer sets the floor *height*
(better models float higher); the loop sets the *shape* (flat-then-cliff, no favorable knee). That a
shared shape survives across proposers of such different competence, in the world with the most
coupling (H13) and the only one with concurrency (H14), is the strongest evidence the program has that
the oracle-in-the-loop method is **model-agnostic** — exactly what makes the contribution a *method*,
not a model. The same floor+cliff now appears in all four worlds (filesystem K4, network EN1, host
EH1, distributed ED1) and across every proposer in each — the claim's most general statement.

### 22. The thesis in one figure: the floor+cliff is the same in every world (cross-world synthesis)

If the claim is that the oracle-in-the-loop tradeoff is a property of the *method*, the cleanest test
is to put all four worlds on one axis. Each world's `H_ε(ρ)` curve is normalized by its own horizon
ceiling `T` (so a tree, a graph, a coupled bundle, and a replicated cluster with different rollout
lengths are comparable), difficulty-averaged, and overlaid:

![Cross-world synthesis: normalized H_ε/T vs ρ for filesystem, network, host, and distributed — one shape](figures/synthesis_floor_cliff.png)

| world | state type | floor `H_ε/T` at ρ=0 | ρ=1 |
|---|---|---|---|
| filesystem (E1) | a tree | 0.00 | 1.0 |
| network (EN1) | a typed graph | 0.04 | 1.0 |
| host (EH1) | a coupled bundle | 0.02 | 1.0 |
| distributed (ED1) | a replicated cluster under faults | 0.01 | 1.0 |

**Four worlds, one curve.** Despite entirely different state representations, oracles, grammars, and
models, the normalized faithful horizon traces the same **floor + cliff**: a near-zero floor across the
whole `ρ` interior, then a steep climb to full horizon only as `ρ→1`. This is the program's thesis made
visual — *a little consultation does not buy a lot of horizon; faithfulness is paid for near-linearly
in oracle calls* — and it is a property of the **oracle-loop method**, not of any one world or model.
The fourth world (SPEC-7, the distributed cluster) makes it the *strongest* version of the claim: it is
the one world where bit-exact global truth is **intractable** (SPEC-7 §5, NP-complete consistency
checking), so its curve is measured against a **tiered, cost-bounded** oracle (ED1, the `panel == curve`
rows at the bit-exact tier) rather than a cheap exact one — the floor+cliff is therefore not an artifact
of having an exact oracle to spend. Combined with the model-invariance result (§21, the shape is
constant across proposers *within* each world), the floor+cliff is now both **model-agnostic and
world-agnostic** — the strongest statement the smoke-scale evidence supports. (Honest scope: same small
models; the *shape* is the robust claim, not the floor's exact height — and "what survives scaling"
remains the open question, [§8](#8-which-wins-survive-scaling--the-honest-mixed-verdict-spec-9).)

### 23. Concurrency's cost scales with concurrency's width (host EH-H14-scale)

H14 (§19) showed concurrency is a measurable dial at one workload width (5 threads). The scaling
question — the host analogue of the free-oracle scaling work ([§8](#8-which-wins-survive-scaling--the-honest-mixed-verdict-spec-9)) — is whether the *cost* of
concurrency grows with the *amount* of it. EH-H14-scale reruns the dial at 2→8 threads, each with its
own factored arm trained on its own recorded (sequential) regime:

![EH-H14-scale: the recorded→chaos H_ε collapse steepens with more threads](figures/eh_h14_scale.png)

| threads | `H_ε` recorded (low entropy) | `H_ε` chaos (high entropy) | collapse |
|---|---|---|---|
| 2 | 7.6 | 3.1 | ~2.5× |
| 4 | 13.8 | 1.8 | ~7.7× |
| 6 | 18.2 | 1.5 | ~12× |
| 8 | 14.3 | 1.2 | ~12× |

**The collapse steepens with width.** At 2 threads chaos costs only ~2.5× of faithful horizon; by 6–8
threads it costs ~12×, dropping the model to barely one faithful step under chaos. More concurrent
threads mean more shared-file contention and more interleaved forks, so the schedule space the chaos
seed explores is both larger and more damaging — **concurrency width is a genuine difficulty axis, and
its cost is super-linear in the early regime before saturating.** (Honest wrinkle: the *recorded*-regime
horizon rises with width to 6 threads then dips at 8 — the longer 40-syscall schedule begins to strain
the tiny model's capacity, a smoke-scale artifact, not a property of the dynamics; the collapse *ratio*
is the robust signal.) This is the first quantification of *how* HW-1's cost grows with load.

### 24. Aggregate faithfulness hides a security-critical denial gap (host EH8)

The host world exists for cyber-defense, and a defender's trust hinges not on the model getting
*successes* right but *failures*: does it predict that a `setuid` by a non-root process is **denied**
(EPERM), that a write to a closed fd is **EBADF**? EH8 measures privilege-faithfulness (denied/allowed
agreement) for the flat and factored arms on a denial-heavy workload, reading the exit code off each
predicted bundle delta:

![EH8: overall vs setuid vs denied-recall privilege-faithfulness, flat vs factored](figures/eh8_privilege.png)

| arm | overall priv-faithfulness | setuid only | **denied recall** |
|---|---|---|---|
| flat | 0.912 | 0.344 | **0.000** |
| factored | 0.938 | 0.531 | **0.286** |

**Overall faithfulness is a comforting lie.** At 0.91–0.94 it looks like the model nails privilege —
but that number is dominated by the common case (operations that *succeed*). The security-critical
metric is **denied recall**: of the transitions truth says *failed*, what fraction does the model also
predict failed? The flat arm scores **zero** — it never predicts a denial, so it would assure a
defender that *every* blocked, unprivileged action succeeded. The factored arm is materially better
(0.286 recall, 0.531 on `setuid`) — structure helps here too — but still misses most denials. This is
the sharpest defensively-relevant negative in the program: **a host simulator can look 94% faithful
and still be blind to the failures that matter most**, and it makes "denied recall," not aggregate
faithfulness, the metric a security use-case must gate on. (Honest scope: smoke-scale; the *gap* and
the flat-vs-factored ordering are the robust signal, not the exact recall.)

### 25. A cheap symbolic second-oracle is redundant but decision-sufficient (host EH6 / H12)

The full state oracle answers "is the predicted bundle faithful, bit for bit?"; a defender often needs
only one cheap, formally-checkable property. EH6 adds a **symbolic privilege second-oracle** — *no
non-root process holds an fd to a protected path (`/passwd`)* — and asks the network EN10/H12 questions
of it against the factored arm, teacher-forced:

![EH6: the privilege invariant — non-redundant rate, decision-sufficiency, consult cost](figures/eh6_two_oracle.png)

| metric | value | reading |
|---|---|---|
| non-redundant rate | **0.000** | catches nothing the full oracle misses (it is a pure function of the state) — *redundant for verification* |
| invariant-sufficient rate | **0.952** | of the steps where the *full* prediction is wrong, the model still gets the security verdict right |
| consult-bits ratio | **0.31** | the security answer (procs × protected) costs ~⅓ of the full state |

**Redundant, but cheaper and decision-sufficient — the host H12.** The privilege oracle never flags an
error a bit-exact prediction would miss (non-redundant rate is exactly 0, by construction), so as a
*verification* signal it adds nothing. But that is the wrong lens: in **95%** of the steps where the
model's full bundle prediction is *wrong*, the model still gets the **privilege-safety verdict** right,
and answering that verdict costs ~3× fewer bits than the full state. So an agent can ask the cheap
security question and trust the answer far more often than it can trust the whole prediction — the
tiered-oracle premise ([SPEC-7](docs/specs/SPEC-7.md)) the network's EN10 first showed, now replicated in the host with a
formally-checkable security invariant.

### 26. Concurrency manufactures the composition coupling (host EH-H13-scale / H13 × H14)

Two host-unique findings meet here: the composition is **coupled** (H13 — composed acceptance below the
independence floor `∏ aᵢ`), and concurrency is a **dial** (H14). EH-H13-scale asks whether the first is
*caused* by the second: does running more concurrent threads (more shared-file contention, more
interleaved forks coupling the subsystems through the schedule) deepen the coupling? It measures the
composition law teacher-forced on chaos-scheduled workloads at 2→8 threads:

![EH-H13-scale: composed vs independence floor, and the independence gap, vs thread count](figures/eh_h13_scale.png)

| threads | composed `a` | `∏ aᵢ` | **independence gap** | verdict |
|---|---|---|---|---|
| 2 | 0.392 | 0.468 | 0.076 | coupled |
| 4 | 0.175 | 0.364 | 0.189 | coupled |
| 6 | 0.233 | 0.382 | 0.148 | coupled |
| 8 | 0.165 | 0.332 | 0.167 | coupled |

**Concurrency manufactures coupling — then saturates.** The composition is `coupled` at *every* width
(composed always below the independence floor), but the gap is smallest at 2 threads (0.076 — failures
nearly independent) and roughly **doubles–triples by 4 threads (~0.19)**, then plateaus at ~0.15–0.19
through 8. So the anti-correlated subsystem failures H13 found are, in part, *made by the schedule*: a
little concurrency couples the subsystems sharply, and beyond a handful of threads the coupling is
saturated rather than ever-deepening. This ties the host's two signature results together — H13's
coupling is not a fixed property of the bundle but one **concurrency (H14) drives** — and says a faithful
host model must reason about the schedule, not just the per-subsystem dynamics.

### 27. The free oracle closes the EH8 denial gap — it was data balance, not architecture (host EH9)

EH8 banked the program's sharpest defensive negative: a host model looks ~94% privilege-faithful yet
has **denied recall ≈ 0** — it almost never predicts the EPERM/EBADF *failures* a defender most needs,
because denials are rare and the cross-entropy is dominated by the common success case. EH9 turns that
negative into an *intervention* using the program's signature lever — **the oracle labels every outcome
for free, so reweight the data**: oversample the training transitions whose oracle outcome is a denial by
a factor `k` (the cheapest, trainer-agnostic fix — it touches the data, not the loss), and read whether
recall lifts and at what specificity cost. EH9 differs from EH8 in one other deliberate way — it trains
on **both** the `forky` *and* the denial-carrying `adversarial` driver (EH8 trained `forky`-only), so its
`k=1` row already isolates a second, even cheaper intervention: *merely exposing the model to denials in
training at all*.

![EH9: denied recall and allowed specificity vs denial-oversample factor, flat vs factored](figures/eh9_denial_weighted.png)

| arm | `k=1` (driver-only) | `k=4` | `k=16` | allowed specificity |
|---|---|---|---|---|
| flat | 0.333 | **0.762** | 0.524 | 0.98 → 0.99 → 0.99 |
| factored | 0.952 | 1.000 | 1.000 | 0.99 → 1.00 → 1.00 |

**The gap was a data-balance artifact the free oracle can fix — at no specificity cost.** Two stacked,
nearly-free interventions close most of EH8's blindness. First, just *including the denial-carrying
driver in training* (the `k=1` rows vs EH8's `forky`-only 0.000 / 0.286) lifts flat denied recall to
**0.333** and factored to **0.952** — exposure alone nearly closes the structured arm's gap. Second,
**oversampling** denials on top lifts the flat arm further to **0.762** at 4×, and the cost the
worry-line predicted — crying wolf on successes — **never materializes**: allowed specificity holds at
0.98–1.00 across every factor. The free oracle relabels the imbalance away without trading off the
common case.

But the lever is not monotone for the weaker arm: flat recall *falls* to **0.524** at 16× — too
aggressive a reweight distorts the flat model — so the flat arm needs a **tuned** factor (4× is its
sweet spot here), while the **factored arm saturates to perfect recall and is robust to the factor**.
That is the honest shape of the result: the EH8 negative is *largely* an artifact of data balance that
costs nothing to fix, structure both starts far ahead and stays well-behaved under the fix, and the
unstructured arm can be lifted a long way but needs the dial set with care. (Honest scope: smoke-scale;
the recall-lift trend, the zero specificity cost, the flat non-monotonicity, and the flat-vs-factored
ordering are the robust signals — not the exact recall values.) This closes the EH8 loop the way the
program is meant to: a banked negative becomes a measured intervention, and the intervention is the
oracle doing the one thing only an oracle can — labeling the rare, security-critical event for free.

### 28. The experience stream doesn't beat the batch — but replay saves it, and the plasticity probe says why (host EH-stream / H15 / HW-4)

SPEC-6's §8.5 puts the host world's defining promise on the table: the propose-verify-correct loop need
never stop. Because the oracle labels every transition for free, you can run a **continuous stream** of
sandboxed host activity from which the model predicts, the oracle verifies, and the model heals from a
replay buffer — forever (the "Era of Experience" loop). **H15** asks whether that stream *beats the
batch* at **equal total compute**; **HW-4** (loss of plasticity) asks whether a model that learns
forever keeps learning or **ossifies**. EH-stream makes both falsifiable on the factored arm: three
arms see the **same** ordered, oracle-labeled stream and take the **same number of gradient steps**,
differing only in *how* they consume it — `batch` (shuffle the whole stream, the offline baseline),
`stream+replay` (walk it in order, train on a minibatch sampled from a growing replay buffer per
arrival), and `stream-no-replay` (walk it in order but train only on the most recent window — the
forgetting-prone control that isolates *replay* as the lever).

![EH-stream: one-step exact, free-running H_ε, and the HW-4 plasticity probe, by arm](figures/eh_stream.png)

| arm | one-step exact | free-running `H_ε` (T=24) | plasticity (HW-4) |
|---|---|---|---|
| batch | **0.542** | **4.0** | 0.950 |
| stream+replay | 0.471 | 1.7 | **0.966** |
| stream-no-replay | 0.096 | 0.5 | 0.766 |

**H15 is a negative at this scale — and that is the honest, pre-registered result.** At equal compute the
experience stream (`stream+replay`) does *not* beat the offline `batch`: it trails on both one-step
exactness (0.47 vs 0.54) and free-running horizon (1.7 vs 4.0). The Era-of-Experience promise does not
survive contact with the oracle here; the i.i.d. shuffle of the batch is simply a better diet than the
correlated order of the stream when the compute is the same. The negative is the §10.3 H15-refuted
branch, and the program banks it.

**But the experiment's real payoff is the mechanism, which only the controlled arms reveal.** Replay is
**decisively load-bearing**: without it the stream *collapses* (one-step 0.10, horizon 0.5) — correlated
sequential updates catastrophically overwrite earlier learning — and replay rescues it back to 0.47, a
~5× recovery. And the **plasticity probe localizes the failure precisely**: after training, we clone
each arm and measure how much loss it can still shed on a frozen, never-seen probe batch. The no-replay
stream's plasticity has **decayed to 0.77** while the batch and the replay stream sit at **0.95–0.97** —
the forgetting-prone regime has measurably begun to **ossify**, exactly the HW-4 wall §2.5 warned of,
and replay is its documented fix, now *grounded* rather than anecdotal. (Honest scope: smoke-scale; the
robust signals are the orderings — batch > stream+replay ≫ stream-no-replay on competence, and
no-replay's plasticity sitting well below the others — not the exact values.) The honest shape: the
stream loses to the batch at this scale, but the oracle-grounded probe turns "continual learning is
hard" into a number and names *which* lever (replay) holds the line on *which* wall (plasticity).

### 29. Counterfactual replay is just more data for plain supervision — the same null in the host world (host EH6 / H16)

The host oracle is **total**, so from any visited bundle state it returns the exact next state of an
**alternative** syscall the trajectory never took — a free counterfactual branch, the capability
physical-domain causal world models structurally lack (you can literally re-run the process tree from
step `t` with one syscall changed and read back the *true* alternative state). **H16** asks whether
*training* the host delta predictor on these free counterfactual branches lifts its prediction of
**interventions** — the change-safety question a defender actually asks ("what if this process had
called `setuid` instead?"). The counterfactual driver is `adversarial` on purpose: the near-miss
**privilege mistakes** (§17.7) whose oracle outcome is an EPERM/EBADF denial. To separate the
counterfactual *signal* from raw volume, three arms train the same factored arm for the same steps,
differing only in the training set's composition (matched example count): `trajectory` (base),
`trajectory-more` (extra trajectory seeds to the same count — the volume control), and
`+counterfactual` (base plus oracle counterfactual branches).

![EH6 / H16: counterfactual grounding vs a matched-volume control, held-out interventions](figures/eh6_counterfactual.png)

| arm | intervention exact | intervention denied recall |
|---|---|---|
| trajectory | 0.338 | 0.62 |
| trajectory-more | **0.588** | 0.60 |
| +counterfactual | 0.457 | 0.60 |

**H16 is a null beyond volume — and it matches the network's EN6/H5 exactly.** On the robust metric
(held-out intervention exactness), `+counterfactual` (0.457) *does* beat the base `trajectory` (0.338) —
but it **loses to the matched-volume control** `trajectory-more` (0.588). So the lift over the base is
**data volume, not counterfactual structure**: for plain next-state supervision a counterfactual is just
another labeled transition, and plain extra trajectory data of equal volume generalizes at least as
well. The control arm is what makes this honest — without it the counterfactual arm's beat over the base
would have looked like a win. (The denied-recall column is directionally flat across arms but too sparse
at smoke scale — denials are rare in held-out — to separate them; the §17.7 near-miss regime needs a
denser denial workload to read, a scope note that licenses future work.) The coherent payoff is the
**cross-world echo**: the network world found the *identical* result (EN6/H5 — counterfactual *examples*
don't lift plain *supervision*, though counterfactual *negatives* did lift the *contrastive*
representation, EN9). The H16 null is therefore not a host quirk but a **world-agnostic property of the
oracle-grounded method**, the same way the floor+cliff is — and it *bounds* how much counterfactual
augmentation buys for supervised world models: nothing beyond the labeled transitions it adds.

### 30. Capacity buys free-running horizon — and the verdict is *cross-world* (SPEC-10 / HS1–HS2 / H26)

Every floor+cliff above was at the *committed* (tiny) scale, and the report's own *Threats to validity*
named the confound: the models are small, so the floor *might* be a capacity artifact. SPEC-10 measures
that confound directly for the headline metric — it holds the world fixed and sweeps **model capacity**,
measuring two numbers exactly against the oracle on held-out rollouts: one-step acceptance `p` (per-step
accuracy) and free-running faithful horizon `H_free = H_ε(ρ=0)` (the headline). The sharp object is the
**independence baseline** `H_indep = p/(1−p)` — the horizon you'd get if per-step failures were i.i.d.
(no compounding) — and its ratio `η = H_free/H_indep`, the scale-free compounding penalty.

**HS1 (network world):** free-running horizon **lifts ~9× with capacity** (1.75 → 15.8 steps over 32×
params, disjoint CIs) then saturates; HS1.1 sharpens this into a **non-monotone, compute-optimal
frontier** (peak at `l`, then a data-starvation decline the one-step proxy `p` cannot see); HS1.2 shows
that decline is **data starvation, not a wall** (feeding `xl` more data recovers it, the Chinchilla
prescription); HS1.3 confirms the recipe lifts the peak to a **program-best `l@9.6k` = 19.2 id / 28.75
ood**. The floor+cliff dissolves into a **resourcing story with a measurable compute-optimal frontier**
([§ Results at a glance](#results-at-a-glance) top tile; full numbers in [docs/report.md](docs/report.md)).

**HS2 (the universality test):** the whole HS1 arc ran in the network world, so is the lift a fact about
the oracle loop or about one easy world? HS2 re-runs the **identical** capacity axis
([`horizon_host_scaling.py`](src/verisim/experiments/horizon_host_scaling.py), the HS1 harness reused
verbatim) on the harder **host** world (SPEC-6: the composed process/fd/filesystem/exit bundle).

![HS2: free-running faithful horizon scales monotonically with capacity on the host world too, but the floor is re-lowered ~3-5× vs the network](figures/horizon_host_scaling.png)

| scale | params | `p` (id / ood) | **`H_free`** id [95% CI] | `H_free` ood | η (id) | η (ood) |
|---|---|---|---|---|---|---|
| xs | 1,024 | 0.11 / 0.11 | **1.00** [0.75, 1.25] | 0.75 | 8.56 | 6.33 |
| s | 8,192 | 0.21 / 0.26 | **2.42** [1.25, 3.25] | 1.50 | 8.65 | 4.41 |
| m | 32,768 | 0.30 / 0.44 | **2.92** [1.25, 4.00] | 1.33 | 6.37 | 1.75 |
| **l** | 110,592 | 0.49 / 0.52 | **5.08** [3.50, 8.25] | 2.92 | 5.30 | 2.70 |

**The verdict survives the world swap.** (1) **The lift is universal:** `H_free` scales *monotonically*
with capacity on the host world too (id 1.00 → 5.08, ~5× over 108× params, **disjoint CIs** xs [0.75,
1.25] vs l [3.50, 8.25]) — so "capacity buys horizon" is a property of the loop, not the easy world. (2)
**The harder world re-lowers the floor ~3–5× and re-opens the headroom** — exactly HS1's prediction:
every host `H_free` sits far below its network twin (host `l` 5.08 vs network `l` 15.7), and the host
curve **has not saturated by `l`** (the network saturated by `m`), so the harder dynamics push the
compute-optimal peak rightward. (3) **The per-step problem is genuinely harder, and `η` mirrors the
network:** the host `p` runs 0.11 → 0.49 (vs network's 0.47 → 0.79); `η` stays > 1 throughout (the
rollout self-stabilizes on the easy manifold) but **declines toward 1 with capacity** here — the mirror
of the network's rising `η` — because `p` climbs steeply from its low base. Honest caveats: seed variance
is high (the load-bearing facts are the monotone trend and the disjoint xs-vs-l gap); this is the `ρ=0`
floor on the *capacity* axis only (the host data/joint cross-axes are the open follow-up); and `η > 1` is
partly the held-out-`p` artifact HS1 flagged, so `H_free` is the unambiguous number. Net: **capacity buys
horizon on a second, harder world**, and **world difficulty — not a fixed compounding wall — sets the
floor height.** That is exactly the measurement the oracle-free world-model field structurally cannot
make: long-horizon fidelity, scored against exact ground truth, across a 100× capacity range and two
worlds.

### 31. …but the capacity lift is *proposer-dependent* — it does not reproduce for the structured arm (SPEC-10 / HS3 / H26)

HS1–HS2 swept the **flat** transformer. Is "capacity buys horizon" a property of the oracle loop, or of
that one proposer? HS3 ([`horizon_graph_scaling.py`](src/verisim/experiments/horizon_graph_scaling.py))
re-runs the **identical** axis with the **GNN+RSSM graph arm** — the NW8 proposer that *beats* the flat
arm ~6.6× on one-step delta-exact (EN4/H11) — as the proposer.

![HS3: for the structured graph arm, capacity buys neither per-step accuracy nor free-running horizon — both flat, the floor+cliff in its purest form](figures/horizon_graph_scaling.png)

| scale | params | `p` (id / ood) | **`H_free`** id | `H_free` ood | `H_indep` id | η (id) |
|---|---|---|---|---|---|---|
| xs | 1,024 | 0.64 / 0.64 | **0.00** | 0.00 | 1.75 | 0.00 |
| s | 8,192 | 0.66 / 0.67 | 0.67 [0, 2] | 0.89 | 1.93 | 0.35 |
| m | 32,768 | 0.67 / 0.67 | **0.00** | 0.00 | 2.01 | 0.00 |
| l | 110,592 | 0.66 / 0.64 | **0.00** | 0.00 | 1.92 | 0.00 |

**The lift does not reproduce for the structured arm.** (1) For the graph arm, capacity buys **neither**
per-step accuracy **nor** horizon: `p` is **flat** (0.64 → 0.66, vs the flat arm's 0.47 → 0.82 climb;
the `s` `H_free`=0.67 is a single-seed blip, CI [0, 2]) and `H_free` is **≈ 0 at every capacity** (η ≈ 0)
— the floor+cliff in its purest, capacity-invariant form. **So HS1's lift was the *flat arm's specific
p-vs-capacity climb* crossing the self-stabilization threshold, not a universal loop property.** (2) The
graph arm makes **near-but-not-exact** predictions — an ε-sweep on the trained `m` arm gives `H_free`=0
up to ε=0.1 and only **4–6 steps at ε=0.2**: small-magnitude but *ubiquitous* errors that exceed ε≤0.1
within one step, where the flat arm's rollout self-stabilized *exactly*. The oracle exposes this
same-loop, opposite-behavior split that one-step delta-exact (where the graph arm wins) cannot see.

Honest caveats — a confounded negative, stated plainly: the committed graph trainer plateaus at `p` ≈ 0.66
(below the flat arm's 0.82 under `train_batched`; more iters 1.5k → 5k barely move it), so part of the
`H_free`=0 gap is the graph arm not reaching the flat arm's per-step operating point — an
architecture×optimizer interaction, not a proven architectural ceiling. And the graph arm's **flat** `p`
(it ceilings already at `xs`, *above* the flat `xs`'s 0.47) says it is **data-limited, not
capacity-limited**: the inductive bias is data-efficient but early-saturating, so its lever is *data* (the
HS1.2 reading), the graph data cross-axis being the HS3-increment-2 follow-up. The load-bearing fact:
across 108× capacity the structured arm's exact free-running horizon **never leaves the floor** — so
capacity-buys-horizon is **not** automatic across model classes, which sharpens HS1 and is consistent with
EN7/H22 (the loop governs the shape; the proposer's competence sets whether it escapes the floor).

### 32. …and the structured floor is *not* data starvation either — a genuine ceiling (SPEC-10 / HS3 incr 2 / H26)

HS3 left its own confound: the graph arm's **flat `p`** is the signature of a *data*-limited model, so —
exactly as HS1.2 was the clean test for the flat arm — HS3 increment 2
([`horizon_graph_data_scaling.py`](src/verisim/experiments/horizon_graph_data_scaling.py)) holds graph
capacity fixed at `m` and sweeps the coverage set 960 → 9,600 transitions.

![HS3 incr 2: feeding the structured graph arm 10× more data does not lift its free-running horizon off the floor](figures/horizon_graph_data_scaling.png)

| n_train | `p` (id / ood) | **`H_free`** id [95% CI] | `H_free` ood | `H_indep` id | η (id) |
|---|---|---|---|---|---|
| 960 | 0.65 / 0.64 | **0.00** [0, 0] | 0.00 | 1.87 | 0.00 |
| 2,400 | 0.65 / 0.72 | 1.00 [0, 3] | 1.11 | 1.87 | 0.52 |
| 4,800 | 0.59 / 0.69 | **0.00** [0, 0] | 0.00 | 1.46 | 0.00 |
| 9,600 | 0.60 / 0.70 | 0.11 [0, 0.33] | 0.22 | 1.49 | 0.07 |

**The structured floor is not data starvation.** (1) A **10× data increase does not lift `H_free`** (≈ 0
throughout; the 2,400 cell's 1.00 is a single-seed blip, CI [0, 3]) — the *opposite* of the flat arm, whose
HS1.2 floor *recovered* with data (7.7 → 16.2). (2) **`p` does not rise with data either** (flat ~0.60–0.72,
even dipping id), so the bottleneck is not the coverage set — the §31 capacity-flatness and this
data-flatness are the same phenomenon on two axes. (3) **η < 1 throughout (0.00–0.52)** — the tell that
splits the proposers: the flat arm's η stayed **> 1** (its rollout self-stabilizes, free-running *longer*
than its i.i.d. prediction), but the graph arm free-runs **shorter** than `p/(1-p)` — its near-but-not-exact
errors **compound**, the genuine compounding wall H26's honest-negative branch predicted, which the flat arm
escaped on this same world.

**Net across HS3: the structured arm's exact free-running floor moves with *neither* capacity nor data**,
while the flat arm's moved with both — so "the floor+cliff is a resourcing story" is itself
**proposer-dependent**: under-resourcing for the flat arm, a genuine compounding ceiling for the structured
one. Honest caveat: the committed graph trainer plateaus at `p` ≈ 0.6 and `p` does not climb with data, so
the binding constraint is plausibly the trainer/representation on this world, not data per se — "neither lever
lifts it" is shown for this committed graph recipe, not proven for every possible graph optimizer.

### 33. …and it survives the world-size axis too — the structured ceiling is world-size-invariant (SPEC-10 / HS3 incr 3 / H26)

Increments 1–2 swept capacity and data at the 5-host world. The last axis is the one the graph arm
exists **for**: world size — its inductive bias over network structure has *more* to exploit as the world
grows, so a bigger world is where the structured arm could finally pull off the floor. HS3 incr 3
([`horizon_graph_world_scaling.py`](src/verisim/experiments/horizon_graph_world_scaling.py)) holds graph
capacity fixed at `m` and sweeps `n_hosts` over SPEC-9's `O(N²)` axis.

![HS3 incr 3: the structured graph arm's free-running horizon stays pinned at 0 across an 8× world-size range](figures/horizon_graph_world_scaling.png)

| n_hosts | `p` (id / ood) | **`H_free`** id [95% CI] | `H_free` ood | `H_indep` id | η (id) |
|---|---|---|---|---|---|
| 5 | 0.66 / 0.67 | **0.00** [0, 0] | 0.00 | 1.91 | 0.00 |
| 10 | 0.63 / 0.67 | **0.00** [0, 0] | 0.00 | 1.67 | 0.00 |
| 20 | 0.58 / 0.67 | **0.00** [0, 0] | 0.00 | 1.40 | 0.00 |
| 40 | 0.59 / 0.67 | **0.00** [0, 0] | 0.00 | 1.43 | 0.00 |

**The structured ceiling is world-size-invariant.** Across an **8× world-size range** (5 → 40 hosts)
`H_free` is **0 at every world size** (tight zero CIs, 3 seeds), η = 0 throughout — and the graph arm's
per-step `p` **degrades** as the world grows (id 0.66 → 0.59; the bigger world is harder per step, faster
than the inductive bias compensates). The structural bias the graph arm exists for does **not** rescue its
floor at scale. **This completes the HS3 sweep: the structured arm's exact free-running floor is pinned at
0 across *all three* axes — capacity ([§31](#31-but-the-capacity-lift-is-proposer-dependent--it-does-not-reproduce-for-the-structured-arm-spec-10--hs3--h26)),
data ([§32](#32-and-the-structured-floor-is-not-data-starvation-either--a-genuine-ceiling-spec-10--hs3-incr-2--h26)),
and world size** — a genuine compounding ceiling, not an artifact of any single axis. Where the flat arm's
floor dissolved into a resourcing story on every axis (HS1/HS1.2/HS2), the structured arm's moves on **none**
of them: **"is the floor+cliff a resourcing artifact?" is decisively proposer-dependent.** Honest caveat:
the committed graph trainer plateaus at `p` ≈ 0.6 and `p` falls with world size, so the binding constraint is
plausibly the trainer/representation — shown for this committed graph recipe, at the strict tolerance ε ≤ 0.1.

**The joint push closes it (HS3 incr 4).** One marginal at a time isn't the whole story: HS1.3 showed
the *flat* arm's horizon lifts *above* either marginal when capacity and data scale **together** (a
compute-optimal ladder). So the structured joint ladder — a bigger graph arm in a bigger world, capacity
*and* world size scaled together
([`horizon_graph_joint_scaling.py`](src/verisim/experiments/horizon_graph_joint_scaling.py), s@5h → xl@40h)
— is the pre-registered final test. `H_free` is **0 at every rung** (η = 0; `p` flat ~0.6), while HS1.3's
flat joint ladder reached the **program-best** 19.2/28.75 steps. So the structured ceiling survives even
the joint scaling that lifted the flat arm to its peak: **across capacity, data, world size, *and* their
product, the structured floor is pinned at 0** — a genuine wall, the strongest form of the verdict, which
closes the SPEC-10 milestone table.

**And the under-training caveat is now refuted (HS3-T).** Every result above carried one qualifier — *the
graph trainer plateaus at `p` ≈ 0.66, below the flat arm's 0.82, so maybe the floor is just
under-training.* That's concrete and testable: the flat arm reached 0.82 with `train_batched`'s
**warmup+cosine** schedule while the graph trainer used a **flat LR**. HS3-T
([`horizon_graph_schedule.py`](src/verisim/experiments/horizon_graph_schedule.py)) gives the graph arm the
flat arm's own schedule (an opt-in `warmup_frac`, default-off so every committed result is byte-identical,
regression-pinned) and finds it lifts the graph arm's `p` only **0.66 → 0.68** (nowhere near 0.82) with
`H_free` still **0 for both arms**. So the plateau is the graph arm's **representation on this world, not
the flat LR** — the structured ceiling survives the trainer fix, and the load-bearing under-training
caveat is refuted against the flat arm's own winning recipe.

![HS3-T: the schedule that lifted the flat arm to 0.82 barely moves the graph arm (0.66→0.68) and the horizon stays at 0 — the plateau is the representation, not the flat LR](figures/horizon_graph_schedule.png)

### 34. The SPEC-10 capstone: the floor is proposer-dependent (cross-proposer synthesis)

The whole SPEC-10 arc reduces to one contrast, and
[`horizon_synthesis.py`](src/verisim/experiments/horizon_synthesis.py) draws it in one figure — a
*figures-from-records* overlay (like the cross-world [§22](#22-the-thesis-in-one-figure-the-floorcliff-is-the-same-in-every-world-cross-world-synthesis))
that re-reads the two committed capacity-sweep CSVs (the flat `horizon_scaling`, the structured
`horizon_graph_scaling`) and **re-runs nothing**.

![SPEC-10 synthesis: the flat arm's free-running horizon lifts ~9× with capacity while the structured graph arm stays pinned at the floor — the floor is proposer-dependent](figures/horizon_synthesis.png)

Sweeping the **same** capacity axis, the **flat** transformer's free-running horizon **lifts ~9×**
(1.75 → 15.8 steps) and its floor dissolves into a resourcing story across capacity, data, and world
size (HS1/HS1.2/HS2 — [§30](#30-capacity-buys-free-running-horizon--and-the-verdict-is-cross-world-spec-10--hs1hs2--h26)).
The **structured** GNN+RSSM graph arm — the proposer that *beats* the flat arm on one-step delta-exact
(EN4/H11) — shows the **opposite**: its `H_free` is **pinned at ≈ 0** and moves with *neither* capacity
([§31](#31-but-the-capacity-lift-is-proposer-dependent--it-does-not-reproduce-for-the-structured-arm-spec-10--hs3--h26))
*nor* data ([§32](#32-and-the-structured-floor-is-not-data-starvation-either--a-genuine-ceiling-spec-10--hs3-incr-2--h26))
*nor* world size ([§33](#33-and-it-survives-the-world-size-axis-too--the-structured-ceiling-is-world-size-invariant-spec-10--hs3-incr-3--h26)).

**So the program's standing question — "is the floor+cliff a resourcing artifact?" — has no single
answer: it depends on the proposer.** For the flat arm it is under-resourcing (scale lifts it); for the
structured arm it is a genuine compounding ceiling at this world's exact tolerance (nothing lifts it). A
per-step *winner* (the graph arm, on delta-exact) that is the long-horizon *loser* (η < 1), and a
per-step *loser-that-catches-up* (the flat arm) that is the long-horizon winner — exactly the proxy/truth
divergence the whole program exists to expose, and invisible without the exact, free oracle. Together
with the cross-*world* synthesis ([§22](#22-the-thesis-in-one-figure-the-floorcliff-is-the-same-in-every-world-cross-world-synthesis)),
the two capstones bracket the generality: the curve's **shape** is world- and model-invariant (the
loop's), while whether its `ρ=0` **floor height** is a resourcing artifact is proposer-dependent (the
model's). Verification is the invariant; the proposer sets the floor it lives on.

### 35. The oracle is faithful to a *real computer* — the one structural bet, measured (SPEC-11 / SY1 / H27)

Every result above rests on one sentence: *for computer worlds, a deterministic ground-truth oracle is
free, exact, and resettable.* But every figure to date proves the loop against a **from-scratch reference
model** of POSIX — a symbolic interpreter, not a real OS. That is the program's central exposure (W1: "the
oracle is a model, not reality") and the sharpest reviewer attack. SPEC-11 closes it by running the **exact
v0 grammar against a real `/bin/sh` on a real kernel**, inside a hermetic, no-side-effect sandbox, and
measuring agreement bit-for-bit.

![SY1: reference vs. real /bin/sh — per-driver agreement (left) is bit-exact 1.000 on the structure-building grammar; the H_ε(ρ) curve overlay (right) is oracle-invariant](figures/sy1_agreement.png)

The result is the strongest honest form the claim can take. On the **structure-building grammar** — the
regime v0 is *designed* to model — the reference oracle's predicted next state equals the real kernel's
**bit-for-bit, agreement = 1.000** (degenerate CI), over both an exhaustive action battery and driver
trajectories. The head-to-head `H_ε(ρ)` curve (right) seals it: swap the `SandboxOracle` in for the
`ReferenceOracle` and the prime-directive curve is **indistinguishable** (max `|H_ref − H_sys| = 0.000` at
every ρ) — the oracle substitution is *transparent* where v0 claims fidelity. **"Computer worlds have a
free, perfect oracle" stops being an argument and becomes a committed CSV.**

And the disagreement branch is a *gift*, exactly as H28 pre-registered: the differential harness is a
**debugger for the reference model**. It immediately caught a **genuine reference-oracle bug** — `mv`/`cp`
of a directory into its own subtree (`mv /e /e/e`) produced an *invalid* state with an orphaned subtree —
now fixed at the source and re-run to agreement. Beyond that one bug, **every** remaining divergence on the
destructive grammar falls into one of **four named, documented modeling boundaries** with a measured
**residual of zero**: root protection, overwrite policy, permission enforcement (v0 models `mode` as data,
not access control), and self-subtree (a GNU-vs-BSD coreutils choice, platform-stamped). The weakness
becomes a maintenance instrument.

This is built **macOS-first** (the dev host is an Apple Silicon Mac mini; the cross-POSIX
[`SandboxOracle`](src/verisim/oracle/sandbox.py) runs on BSD and GNU coreutils alike) — "reality" does not
mean "Linux," and the headline is pure POSIX, so it reproduces on Linux CI for free. The sandbox is proven
hermetic *before* any agreement number is claimed: [SY3](src/verisim/experiments/sy3.py)'s negative-control
battery shows every prohibited action (filesystem escape, egress, privilege gain, cross-step persistence)
is **denied**, and [SY4](src/verisim/experiments/sy4.py) shows the system oracle is **bit-reproducible**
under its determinism seal. *Prove the checks have teeth before reporting what they catch* —
[SY3](src/verisim/experiments/sy3.py) → [SY4](src/verisim/experiments/sy4.py) →
[SY1](src/verisim/experiments/sy1.py) → [SY2](src/verisim/experiments/sy2.py), the gate order.

## The problem, and what we're trying to accomplish

### The wall every world model hits

Generative world models (Genie 3, V-JEPA 2, Cosmos) all hit the same wall: **long-horizon error
accumulation and faithfulness**, with no cheap way to *detect* or *correct* drift, because physical and
visual worlds have **no ground-truth oracle**. You can render a plausible next frame, but you cannot
cheaply ask "is this *exactly* right?" — so error compounds silently and the field spends enormous
effort on proxies that keep a self-referential objective from cheating (JEPA's collapse-prevention
machinery is the clearest instance).

### The one asymmetry computer environments have

| Signal source | Dense? | Exact / true? | Free? | Generative? |
|---|---|---|---|---|
| Self-supervision (corpus co-occurrence) | ✅ | true to the *corpus*, not the world | ✅ | ✅ |
| Human supervision (annotation) | ◐ | usually — but **unscalable** | ❌ | ❌ |
| RL reward / reward model | ❌ (sparse scalar) | a **proxy**, hackable | ◐ | ❌ |
| **A deterministic oracle (computer worlds)** | ✅ | **exact, by construction** | ✅ | ✅ |

No other domain has the last row. A deterministic interpreter of a computer world returns the *entire
true next state* at every step, for free, and can *generate* unbounded perfectly-labeled data and
counterfactuals. Everything in Verisim follows from asking **where to spend that asymmetry**:
inference-time verification, RL reward, and — newest — self-supervised pretraining; and **how much it
actually buys**.

### What we're building: the propose → verify → correct loop

The signature mechanism runs the world model forward and lets the oracle bound its drift under a
consultation budget `ρ`:

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

- **`apply` is shared by the oracle**, so `apply(s, O(s,a).delta) == O(s,a).state` *by construction* —
  the model and the oracle speak the same delta language (the M1 / NW1 invariant).
- **`ρ`** ranges from 0 (never consult — pure free-running) to 1 (consult every step — always exact).
  The whole research question is the shape of `H_ε(ρ)` between those ends.
- Under partial observation the oracle has **full** and **probe** modes, turning consultation into a
  real bit-budget and opening an active-sensing axis ([SPEC-5 §5.3](docs/specs/SPEC-5.md)).

### The neuro-symbolic split, as a *training* principle

The next-state partitions into two regimes that want opposite treatment — the heart of
[SPEC-8](docs/specs/SPEC-8.md):

```
  s' = O(s, a)
    ├─ D  decidable bits  ── the oracle fixes them exactly & free  ──▶  VERIFY, don't learn  [symbolic]
    └─ R  residual bits   ── genuinely uncertain given what's seen  ──▶  LEARN (the model's job) [neural]
```

Burning network capacity to memorize `D` is waste — the oracle computes it perfectly for free. "Even
nature offloads": evolution does not store chemistry in the genome. SPEC-8 makes this a training
objective (mask `D`, spend gradient on `R`) and ships the deterministic machinery for it (OG1/OG2). *(The
SPEC-9 scaling surface above qualifies this: masking `D` in the **loss** removes beneficial multi-task
signal at small capacity; the partition's load-bearing form is the **inference-time** one — verify `D`,
don't learn-then-mask it.)*

## Architecture & system design

The repo is a stack of **worlds** (filesystem v0, network SPEC-5, host SPEC-6, and now the distributed
world SPEC-7) over one shared contract — the propose→verify→correct loop — plus cross-cutting
training/packaging and a **scaling layer** (SPEC-10) that sweeps the prime-directive metric itself along
capacity/data/world-size. Every box below is dependency-free and torch-free except `model/`, `netmodel/`,
`hostmodel/`, and `train/` (the optional `[model]` extra). The **`Model` protocol is the seam**: the loop,
oracle, metrics, and benchmark never know which proposer they hold, which is what makes the contribution a
*method* rather than a model (the H22 model-invariance claim) — and is exactly what lets the SPEC-10 scaling
layer swap the flat transformer for the graph arm under the *same* harness (the proposer-dependence result,
§34). Each world **composes** the ones below it: a host runs on the network; the distributed world replicates
services *across* hosts — and it is the first world where the bit-exact global oracle becomes
**intractable**, so SPEC-7's payload is the *tiered* oracle (cheap consistency checks + rare bit-exact
replay).

```
                       ACTION a_t
                          │
                          ▼
   ┌────────────────┐  predict_delta   ┌────────────┐   apply(s,Δ̂)   ┌────────────┐
   │  Mθ  proposer  │ ───────────────▶ │  Δ̂  delta  │ ──────────────▶ │  ŝ_{t+1}   │
   │ (Model proto)  │   grammar-       └────────────┘                 └─────┬──────┘
   │ txf | graph+   │   constrained          ▲ same delta grammar           │
   │ RSSM | LLM     │                        │                              │ divergence d(ŝ, s*)
   └────────────────┘                        │                              ▼
   ┌────────────────┐  O(s,a) = (state, Δ*)  │                       ┌──────────────────┐
   │ Oracle (truth) │ ───────────────────────┘   consult on budget ρ │ H_ε(ρ) · bits-to- │
   │ deterministic  │        full | probe ─────────────────────────▶ │ correct · δ-exact │
   └────────────────┘                                                └──────────────────┘
        apply(s, O(s,a).Δ) == O(s,a).state   ← the M1 / NW1 invariant, tested by construction
```

Package map (parallel structure; `net*` mirrors v0 for the graph world):

```
  v0 filesystem (SPEC-2)        network world (SPEC-5)            cross-cutting
  ─────────────────────         ──────────────────────            ────────────────────────────
  env/      state, actions      net/        typed-graph state     train/   supervised + RLVR
  oracle/   O(s,a) truth        netoracle/  Tier-A (data-plane)    eval/    faithfulness benchmark
                                            + control-plane oracle
  delta/    Δ types, apply      netdelta/   graph Δ, apply         rl/      oracle-as-reward env
  metrics/  d, H_ε, bits        netmetrics/ d, reachability,       auto/    autoresearch ratchet
  loop/     runner, π_c, ops                delta-exact, bits      experiments/  E*, EN*, K*, EH*,
  model/    Mθ transformer      netmodel/   flat Mθ + graph+RSSM                 en8/9_scale, en8_capacity,
  data/     drivers, traj                   + grounded_train (SSL)              en9_negatives, + the
                                netdata/    drivers + OG1/OG2 factory           SPEC-10 scaling layer
                                netloop/    partial-obs runner, probe,         (horizon_*: HS1 capacity,
                                            belief filter                       HS1.2/1.3 data/joint, HS2
                                                                                host, HS3 graph + its data/
                                                                                world/joint/schedule cross-
                                                                                axes, HS-synth synthesis)

  host world (SPEC-6, HC0-HC8 — the composing world; the host oracle *composes* the FS + net sub-oracles)
  host/      bundle state (procs + per-process fds + embedded v0 fs), syscall grammar, bundle delta, config
  hostoracle/  Tier-A reference host oracle (process/fd/credential glue over the v0 FS sub-oracle)
               + invariant.py: a symbolic privilege second-oracle (the cheap EH6/H12 security check)
  hostdata/  workload drivers + trajectory JSONL + manifests/splits + the concurrency scheduler
             (interleaving-entropy chaos dial → H14: H_ε(interleaving-entropy))
  hostmetrics/  composed + per-subsystem d, bits, composition-law (H13), privilege-faithfulness, run-record
  hostmodel/   flat Mθ (HC4 incr-1) + factored interaction-graph Mθ (incr-2): GNN+RSSM over the
               process spine's lineage + shared-file edges, same grammar/decode (DD-H1: flat vs factored)
  hostloop/    composed loop (HC5): two-mode oracle, π_w which-subsystem policy (fixed/round-robin/
               uncertainty — the smart information-gain choice), SubsystemFilter per-subsystem correct
  hostsim/     the LLM-callable whole-machine simulator (HC8, §7): imagine a plan (oracle-free draft)
               + verify it (plan-level H_ε + the task-oracle "third oracle") — what an agent calls
  hostrl/      oracle-as-reward RL env (HC8, §12; the v0 rl/ shape): reset/step, reward = faithful step,
               return == composed H_ε — the verifiable-reward substrate, no learned reward model in loop
  hosteval/    composed-host faithfulness benchmark (HC8, §1.4; the v0 eval/ shape): score_host_model +
               single-step QA grader + the inspect_ai task adapter (behind the [eval] extra)
  contrib/     §16 decentralized verified-contribution protocol: accept a contributed transition/
               trajectory iff the oracle reproduces it bit-for-bit — trustless by re-execution

  distributed world (SPEC-7, the current build front — the layer *above* the host: replicated
                     services across machines, where the bit-exact global oracle becomes
                     *intractable*; core DS0/DS2/DS3/DS5/DS6 + the learned M_θ flat arm DS4 ship)
  dist/        DistributedState (per-(object,node) MVCC replicas + causal event log + in-flight
               messages + partition/crash/clock), the client (put/get/cas) + fault/time
               (advance/partition/heal/crash/restart) action grammar, the DistDelta + compositional
               apply, canonical serialization
  distoracle/  Tier-A reference DES (reference.py): a from-scratch deterministic discrete-event
               simulator of a fully-replicated KV under async replication + the fault/time medium
               (eventual-consistency last-writer-wins, the apply==oracle invariant); AND the
               **tiered oracle** (tiers.py) — SPEC-7's payload: the metamorphic ↔ cycle ↔ symbolic ↔
               bit-exact menu where cheapest_refutation spends the cheapest tier that can refute a
               prediction, recording the oracle-dollar cost (the H17 premise). (Tier-B real-DST later)
  distdata/    seeded workload+fault drivers (uniform/contention/adversarial = BUGGIFY) with the
               explicit fault-intensity (fault_prob) + partition-entropy (partition_bias) dials the
               H20/H21 sweeps need; trajectory JSONL + regenerable dataset manifests (DS2)
  distmetrics/ live-cluster divergence d(s,ŝ) (feeds the generic faithful_horizon, so distributed
               H_ε(ρ) is defined as in every world), the headline-new consistency-faithfulness
               (§9.1: did the model predict each object's converged/split state?), bits-to-correct
               / delta-exact over the DistDelta (DS3 metric core)
  distloop/    the tiered propose-verify-correct runner (DS5): model-agnostic over any DistModel
               (null/oracle-backed baselines), the π_w which-TIER policy (fixed | cheapest-refutation
               escalate), and the oracle-DOLLAR accounting — a consult spends its tier's cost, a
               refutation adds the bit-exact correction, an unrefuted prediction is trusted; the
               record carries divergences (→ H_ε) AND cumulative oracle-dollars (→ H17)
  distmodel/   the learned M_θ (DS4): the closed DistVocab (ops + commands + statuses +
               node/object/value leaves + one bounded <int:..> pool that closes the monotone
               bookkeeping counters) and the bidirectional tokenizer (state,action → Δ encode +
               exact parse). The causal-log EventAppend is a bare marker reconstructed from
               (state,action) on parse — keeping its variable-length happens_before out of the
               grammar. Round-trip parse(encode(Δ))==Δ tested exhaustively (incr 1, torch-free
               modules). Incr 2 (the [model] extra) adds the LL(1) DistDeltaGrammar (nested
               partition run + status-typed result), the NeuralDistWorldModel over v0's GPT (a
               drop-in DistModel), and the supervised dataset builders — overfit/grammar-valid/
               loop-protocol tested.

  the SYSTEM ORACLE (SPEC-11, cross-cutting — validating the *reference* oracle against reality)
  oracle/sandbox.py       SandboxOracle: a real /bin/sh over a real kernel behind the SAME Oracle
                          protocol — materialize state → render the v0 action → exec under the seal
                          → snapshot → destroy. v0-resolve path confinement; cross-POSIX (macOS+Linux)
  oracle/sandbox_seal.py  DeterminismSeal: the per-step env-scrub/umask/rlimit/no-new-privs seal
  oracle/differential.py  differential_step: ref vs system on the same (s,a); world/exit/stdout
                          channels + classify_divergence (the named modeling boundaries)
  experiments/sy{1..4}.py SY3 hermeticity (H29) → SY4 determinism (H30) → SY1 agreement (H27, retires
                          W1: structure-grammar = 1.000, residual = 0) → SY2 debugger (H28)
```

The host **bundle** is the structural novelty: state is a coupled set of subsystems (process table +
per-process fd tables + the embedded v0 filesystem) sharing references, not one tree. So a bundle delta
*composes* sub-system deltas — and the seam `M_θ` must learn is visible right in the encoding. The flat
HC4 arm (DD-H1 baseline) flattens that composition into one token stream; the factored arm (later
increment) keeps it. A `write` through fd 3 makes this concrete:

```
  syscall:  write 7 3 alpha           # pid 7 writes "alpha" through fd 3 (→ path /log)
  bundle Δ: [ FsDelta( v0:[ Create(/log, File"alpha"), SetResult(0) ] ),  SetExit(0) ]
              └── the embedded FS sub-oracle's OWN delta, applied by the v0 apply ──┘  └ host glue ┘
  flat Mθ : <fs_create> <path:/log> <c:alpha> <exit:0>   <set_exit> <exit:0>   <eos>
            └ the composition flattened to one closed-vocab stream; round-trips verbatim (§5.1) ┘
  invariant: apply(state, Δ) == oracle.step(state, action).state    # the M1/NW1-analogue, by construction
```

The composed loop (HC5) adds a second axis on top of v0's *when-to-consult* (`π_c`): **which subsystem's
truth to buy** (`π_w`, §8.2). A full consult corrects the whole bundle; a per-subsystem probe corrects
exactly one subsystem and keeps the model's belief for the rest — strictly less correction, so faithful
horizon is no greater at equal budget (the EH3 lever, native here with no v0 identity collapse):

```
  PROPOSE  Δ̂ = Mθ(ŝ, a) ;  ŝ' = apply(ŝ, Δ̂)            # cheap, every step
  CONSULT  if π_c fires (budget ρ):
     full    →  ŝ' ← truth                              cost = |all facts|     (HardReset)
     π_w=fd  →  ŝ'.fds ← truth.fds, rest kept           cost = |fd facts|      (SubsystemFilter)
                └ proc / fs / global beliefs survive verbatim ┘
  RECORD   composed d(ŝ', s*)  AND  per-subsystem d_proc/d_fd/d_fs/d_global   → HostRunRecord
                                    └ the two views H13's composition law needs ┘
```

The deterministic cores (oracle, delta/apply, divergence, the loop, the OG1/OG2 data factory) ship and are
property-tested **before** any training claim — the figure is always gated, never assumed (the NW0–NW3 /
OG1–OG2 discipline). See [SPEC-2 §10](docs/specs/SPEC-2.md) and [SPEC-5 §16](docs/specs/SPEC-5.md) for the
full module-by-module layout.

## Specifications

All specs live under [`docs/specs/`](docs/specs/); the canonical, evidence-gated build order is
[SPEC §12](docs/specs/SPEC.md#12-research-roadmap). The worlds form a ladder (filesystem → network →
host → distributed); three specs are *cross-cutting methods* every world inherits.

| Spec | Role | What it is |
|---|---|---|
| [SPEC.md](docs/specs/SPEC.md) | **the science** | why the project exists, what it claims, how we'd know we were wrong (RQs, H1–H25) |
| [SPEC-2](docs/specs/SPEC-2.md) / [SPEC-2.1](docs/specs/SPEC-2.1.md) | **v0 build** | the shell/filesystem world; the focused effort that earned a competent model and the knee result |
| [SPEC-3](docs/specs/SPEC-3.md) | depth | how the toy grows into a real simulator (system oracle, partial obs, online self-healing, info-theoretic metric) |
| [SPEC-4](docs/specs/SPEC-4.md) | **the engine** | the autonomous research engine — Verisim improving Verisim, human out of the loop |
| [SPEC-5](docs/specs/SPEC-5.md) | **world: network** | the reachability/connectivity world — NW0-NW8 built (oracle, graph delta, drivers, composed loop, the EN-series curves and the flat+graph learned arms) |
| [SPEC-6](docs/specs/SPEC-6.md) | world: host | the running computer (process tree, fds, scheduler) — **HC0-HC8 built**: the host oracle *composes* the v0 FS sub-oracle; bundle delta + `apply == oracle` invariant; workload drivers + datasets; composed + **per-subsystem** metrics with the **composition-law diagnostic** (H13); the **flat learned `M_θ` baseline** (HC4 incr-1); the **composed loop** with the `π_w` oracle-selection axis (HC5 incr-1); **the prime-directive figure (HC6)** — the composed `H_ε(ρ)` floor+cliff + the **H13 composition law = `coupled`** ([eh1_curve](figures/eh1_curve.png), [eh1_composition](figures/eh1_composition.png)); **the EH3 equal-budget operator comparison (HC7)** — per-subsystem consultation earns **~3.7× more horizon per oracle-bit** ([eh3_operators](figures/eh3_operators.png)); **the factored interaction-graph arm (HC4 incr-2) + EH4** — structure beats flat **~6.6× on delta-exact** yet the H13 coupling survives ([eh4_factored_vs_flat](figures/eh4_factored_vs_flat.png)); **EH2** — the factored arm's calibrated belief variance makes smart consultation beat fixed **~2.2×** (the first smart-`π_c` positive, [eh2_policies](figures/eh2_policies.png)); **EH5** — a smart *which-subsystem* `π_w` (per-subsystem decode entropy) gives a modest edge over round-robin ([eh5_subsystem_policy](figures/eh5_subsystem_policy.png)); **EH5-heads** — a trained per-subsystem decode *head* (opt-in) is **uncalibrated** (Spearman −0.02) where the bucketed entropy it would replace is **well-calibrated** (+0.57), closing the open HC7 lever with a negative ([eh5_heads](figures/eh5_heads.png)); the **§6.3 drift levers** (noise / self-forcing) reproduce the network's banked negative ([eh4_drift](figures/eh4_drift.png)); **H14 — the concurrency dial — is CONFIRMED**: free-running `H_ε` collapses ~8× as interleaving entropy rises (the host's defining result, [eh_h14_interleaving](figures/eh_h14_interleaving.png)); the **§7 LLM-callable whole-machine simulator** (HC8) — `imagine` a plan + `verify` it (plan-level `H_ε` + the task "third oracle"); **EH7/H22** — the floor+cliff `H_ε(ρ)` shape is **model-agnostic in the composed world too** ([eh7_invariance](figures/eh7_invariance.png)); and the **HC8 security/scaling findings** — **EH8** (aggregate faithfulness hides a **denied-recall gap**, flat 0.000 / factored 0.286, [eh8_privilege](figures/eh8_privilege.png)), **EH6** (a symbolic privilege second-oracle is redundant but **decision-sufficient in 95%** of error steps at ~3× lower cost, the host H12, [eh6_two_oracle](figures/eh6_two_oracle.png)), **EH-H13-scale** (concurrency **manufactures** the H13 coupling, [eh_h13_scale](figures/eh_h13_scale.png)), **EH9** (the denied-recall gap is a **data-balance artifact the free oracle fixes** — exposure + oversampling lift recall at no specificity cost, [eh9_denial_weighted](figures/eh9_denial_weighted.png)), **EH-stream/H15** (the experience stream **loses to the batch** at equal compute, but **replay** rescues it from collapse and the **plasticity probe** localizes HW-4 — no-replay plasticity 0.77 vs 0.95, [eh_stream](figures/eh_stream.png)), and **EH6/H16** (counterfactual replay is a **null beyond volume** for plain supervision — world-agnostic with the network's EN6, [eh6_counterfactual](figures/eh6_counterfactual.png)); plus the **oracle-as-reward RL environment** ([`hostrl/`](src/verisim/hostrl/)) whose episode return *is* the composed `H_ε` |
| [SPEC-7](docs/specs/SPEC-7.md) | world: distributed | replicated services, transactions, consensus — **the current build front**: the replicated-KV-under-partition deterministic core ([`dist/`](src/verisim/dist/), [`distoracle/`](src/verisim/distoracle/), [distributed-semantics](docs/distributed-semantics.md)) — async replication + eventual-consistency LWW, stale-read-under-partition, `apply==oracle` + goldens (DS0); the data factory with fault-intensity/partition-entropy dials (DS2); the metric core + **headline-new consistency-faithfulness** and the **tiered oracle** (metamorphic→cycle→symbolic→bit-exact, the **H17** payload, DS3); the **tiered propose-verify-correct loop** with the `π_w` which-tier axis + oracle-dollar accounting (DS5); the **distributed `H_ε(ρ)` curve + first H17 verdict** (ED1/DS6); the **learned `M_θ`** — tokenizer/vocab foundation + the LL(1)-constrained-decode flat arm over v0's `GPT` (DS4); and the **equal-dollar-budget H17/H18 frontier** on both the synthetic proposer (ED2) and the *real* learned `M_θ` (ED2-learned), the **`π_c` smart-when comparison** (ED2-smart — the H9 null carried in), the **ED3 correction-operator comparison** (the partial operator breaks v0's identity on the in-flight medium), the H21 fault-injection sweep, and the **ED4 consistency-level sweep (H20)** — the `linearizable` strong-consistency model added, the H19 gap shown to track the in-flight medium (DS7); and the **DS8 ED5 consistency-vs-bit horizon (H19) + competitive-ratio fit (H18)** — consistency-faithful horizon outlasts bit-faithful where the error hides in the consistency-invisible in-flight medium, and the loop is learning-augmented in the error axis but floor→cliff in the budget axis; and **DS8 ED6 counterfactual lift (H5)** — free oracle fault-flip branch training beats equal-volume on-policy data on held-out interventions (the honest inverse of the network/host supervision null, because the distributed medium is off-policy by nature) — plus the **ED6 two-oracle slice (H12)**: the cheap consistency oracle is redundant for verification but decision-sufficient for the split-brain question (1.00 on in-flight vs 0.00 on durable errors) at ~3.6× lower consult cost; and the **§16 verified-contribution protocol** ([`distcontrib/`](src/verisim/distcontrib/)) — trustless distributed-trace contribution by re-execution with tiered acceptance (bit-exact *or* cheap-tier admissibility, the W7 path). The first world where the bit-exact global oracle is *intractable*, so the payload is the **tiered oracle** (H17) |
| [SPEC-8](docs/specs/SPEC-8.md) | **method: oracle-grounded SSL** | put the oracle's truth in the *bulk* of the cake (self-supervised pretraining), not just the cherry (RL) |
| [SPEC-9](docs/specs/SPEC-9.md) | **method: free-oracle scaling** | because the oracle labels for free, world size is a *compute* choice, not a labeling-budget one — how large/deep the world goes on one machine, and what holds as it grows |
| [SPEC-10](docs/specs/SPEC-10.md) | **method: the faithful-horizon scaling law** | scales the *prime directive itself* (`H_ε(ρ)`) along model capacity: does free-running faithful horizon grow with scale, or is the one-step→horizon compounding gap fundamental (H26)? |
| [SPEC-11](docs/specs/SPEC-11.md) | **method: the system oracle** | validates the program's *one structural bet* — that a deterministic ground-truth oracle exists for computer worlds — by running the v0 grammar against a **real `/bin/sh` on a real kernel** in a hermetic sandbox and measuring agreement bit-for-bit (H27–H30). **The figure that retires W1.** |

Semantics docs ([filesystem](docs/semantics.md), [network](docs/network-semantics.md)) pin the normative
command semantics, paired with the reference oracles, which are the executable truth. SPEC-11's system
oracle now validates the filesystem semantics against a real kernel ([§4.1](docs/specs/SPEC-11.md)). The
full result write-up is [docs/report.md](docs/report.md).

## Status

> **Where things stand (2026-06): v0 is done; the network graph arm shipped and split the H11 verdict;
> both §6.3 drift levers, the SPEC-8 data factory, and the SPEC-8 EN8 + EN9 ablations shipped.** Filesystem v0
> (M0–M8) and the focused [SPEC-2.1](docs/specs/SPEC-2.1.md) effort are complete (K0 learner works → K1/K2
> floor ~0 → **0.86** → K3/K4 knee refuted, licensing SPEC-5). The network deterministic core (NW0–NW3),
> flat `M_θ` (NW4), partial-observation loop (NW5), prime-directive EN1 curve (NW6, the H8 negative), and
> EN2/EN3 equal-budget comparisons (NW7, the ~2.3× probe-efficiency result) all ship. **NW8** adds the
> GNN+RSSM graph arm, the EN4 graph-vs-flat comparison (the +16.5/+30.6-pt split verdict), the
> **delta-exact** metric, **both §6.3 exposure-bias levers** (noise-injection + self-forcing), the **SPEC-8
> OG1/OG2** oracle-grounded-SSL data factory, and now both **SPEC-8 EN8 / OG3** and **EN9 / OG4** ablations
> that consume it — two more split verdicts: **H23 confirmed** (the oracle-anchored target removes the
> collapse tax), **H24 a near-tie** (residual masking buys nothing at this scale), **H25 confirmed** (exact
> negatives match VICReg at preventing collapse) with a decisive **H5 lift** (the oracle's counterfactual
> negatives nearly double VICReg's interventional fidelity).
>
> **The [SPEC-9](docs/specs/SPEC-9.md) scaling surface (LS0–LS2) then carried those smoke verdicts up an 8×
> world-size range (25→200 hosts × {d64,d128}) with bootstrap CIs — the honest mixed result of [§8](#8-which-wins-survive-scaling--the-honest-mixed-verdict-spec-9):
> H23 *persists but attenuates*, H24 is *regime-dependent*, and H25/H5 *reverses* at 100–200 hosts with a
> fixed negative count — then **recovers**: the EN9 `k_negatives` S2-recovery diagnostic confirms scaling
> negatives 8→32 flips the lift back to disjoint-positive (the reversal is a negative-count artifact, fixed
> modestly by scaling negatives with the world).** **EN7/H22 model-invariance now ships** ([§9](#9-the-no-knee-shape-is-the-loops-not-the-models-network-en7--h22)):
> the floor+cliff `H_ε(ρ)` shape is the same across null / flat / graph proposers — the loop governs the
> shape, the proposer sets the floor height (H22 supported in kind). **EN5/H7 self-healing also ships**
> ([§10](#10-online-self-healing-ttt-does-not-lift-the-floor--yet-network-en5--h7-an-honest-null)):
> *neither* a minimal in-rollout TTT step *nor* the pre-registered replay-buffer self-healing budget
> lifts the floor (a robust null, consistent with EN4/EN7) — so the floor's levers are scale (SPEC-9) and
> objective grounding (SPEC-8), not adaptation. **EN6/H5 change-safety also ships** ([§11](#11-counterfactual-grounding-helps-the-contrastive-objective-not-supervision-network-en6--h5)):
> counterfactual training is a null for the predictive model beyond a matched-volume control (H5 is
> objective-dependent — it lifts the contrastive representation, not supervision). **The SPEC-9 LS3 hero
> instance also ships** — at N=300 hosts (the largest oracle-grounded world proven on one machine) the H23
> collapse gap is still disjoint-positive but nearly exhausted at fixed `d128` (rank 2.2, std 0.064),
> confirming "persistent but attenuating" at the envelope's edge. **EN10/H12 two-oracle also ships** ([§12](#12-the-control-plane-oracle-is-redundant-for-verification-but-cheaper--decision-sufficient-network-en10--h12)):
> a Batfish-style control-plane oracle is *redundant* for verification (it catches nothing the data-plane
> misses) but a **cheaper, decision-sufficient** consultation. With EN1–EN10, the network EN-series is
> complete. **The host world (SPEC-6) has now begun:** HC0 increment 1 + HC1 + HC2 + HC3 ship the
> deterministic core — the bundle host state (process table + fd tables + embedded v0 fs), the Tier-A oracle
> that *composes the v0 FS sub-oracle*, the **compositional bundle delta + `apply == oracle` invariant** (the
> M1/NW1-analogue; a `write`'s delta embeds the v0 FS sub-oracle's own `Delta` verbatim), the **data factory**
> — three state-reading workload drivers (`uniform`/`forky`/`adversarial`) + regenerable trajectory
> JSONL/manifests whose recorded deltas replay to every next_state — and now the **composed metric core**
> ([`hostmetrics/`](src/verisim/hostmetrics/)): composed **and per-subsystem** divergence and bits-to-correct
> (`proc`/`fd`/`fs`/`global`, the embedded fs reusing v0's gates verbatim), the **composition-faithfulness
> diagnostic** (`composition_law` — multiplicative ↔ weakest-link ↔ coupled, the §9.2 headline-new metric for
> H13), privilege-faithfulness, and the `HostRunRecord` schema — property-, invariant-, data-, and
> metric-tested, no GPU. **HC4 increment 1 now ships the learned model's flat baseline arm**
> ([`hostmodel/`](src/verisim/hostmodel/)): the closed `HostVocab` over the bundle DSL (the unbounded
> pid/fd families bounded by sized pools), the `(bundle_state, action) → bundle_delta` tokenizer with the
> embedded FS write delta **flattened and reconstructed verbatim** (§5.1), the LL(1) `HostDeltaGrammar` +
> constrained decode reusing v0's `GPT` (grammar-validity is structural, not learned), the supervised
> dataset adapter (v0's generic trainer reused), and `NeuralHostWorldModel` — the **DD-H1 flat-serializer
> floor the factored arm must beat**, overfit/round-trip/grammar-tested. **HC5 increment 1 now ships the
> composed propose-verify-correct loop** ([`hostloop/`](src/verisim/hostloop/), the rest of the
> deterministic core): the model-agnostic runner, the **two-mode partial-observation oracle** (full vs a
> cheap per-subsystem probe), the **`π_w` "which-subsystem" policy** (the host's new oracle-selection axis —
> *which truth-source to buy*, §8.2) and the per-subsystem **`SubsystemFilter`** operator (correct only the
> observed subsystem — the EH3 lever, native here with no v0 identity collapse), plus null/oracle-backed
> baselines, all populating HC3's composed + per-subsystem `HostRunRecord`. Loop invariants tested: `ρ=1`
> full-consult reproduces the oracle (`H_ε=T`), the perfect model never drifts at `ρ=0`, the budget is never
> exceeded and the spend-down backstop spends it exactly, and a per-subsystem consult corrects strictly less
> than a full one (horizon no greater at equal `ρ`). **HC6 now ships the prime directive**
> ([`eh1.py`](src/verisim/experiments/eh1.py), figures [`eh1_curve.png`](figures/eh1_curve.png) +
> [`eh1_composition.png`](figures/eh1_composition.png)): the composed `H_ε(ρ)` curve is the **floor+cliff**
> shape (the no-knee verdict generalizes to the bundle world), and the **composition law (H13) reads
> `coupled`** — composed per-step acceptance sits *below* the multiplicative/independence prediction, so
> the flat baseline's subsystem failures are anti-correlated (coupling is load-bearing; see finding
> [§13](#13-the-third-world-and-a-new-question-whole-machine-faithfulness-is-coupled-host-eh1--h13-hc6)).
> **HC7's EH3 also ships** ([`eh3.py`](src/verisim/experiments/eh3.py), figure
> [`eh3_operators.png`](figures/eh3_operators.png)): the equal-budget operator comparison shows the
> full-consult operators coincide on `H_ε` while per-subsystem consultation earns **~3.7× more faithful
> horizon per oracle-bit** — and that the *cheapest* subsystem wins, not the H13-*weakest*, so a *smart*
> `π_w` must trade cost against consequence (finding
> [§14](#14-which-subsystem-you-verify-is-a-real-efficiency-lever--but-the-cheapest-not-the-weakest-host-eh3--hc7)).
> **HC4 increment 2 + EH4 now ship the factored interaction-graph arm** ([`graph_model.py`](src/verisim/hostmodel/graph_model.py),
> figure [`eh4_factored_vs_flat.png`](figures/eh4_factored_vs_flat.png)): a masked message-passing GNN+RSSM
> over the process spine's lineage + shared-file edges, decoding under the same grammar as the flat arm. It
> **beats flat ~6.6× on delta-exact and ~5.3× on composed acceptance** (structure helps), **but both stay
> `coupled`** — so the H13 coupling is a genuine property of the host dynamics, not a flat-arm artifact
> (finding [§15](#15-modeling-the-composition-explicitly-helps-a-lot--but-the-coupling-is-real-not-an-artifact-host-eh4--h11--h13)).
> **EH2 also ships** ([`eh2.py`](src/verisim/experiments/eh2.py), figure [`eh2_policies.png`](figures/eh2_policies.png)):
> the consultation-policy comparison is the program's **first smart-`π_c` positive** — the flat arm
> reproduces the standing H2-negative, but the factored arm's RSSM belief variance makes
> uncertainty-triggered consultation beat fixed **~2.2×** at equal budget (finding
> [§16](#16-a-calibrated-uncertainty-signal-finally-makes-smart-consultation-pay-host-eh2--h9)).
> **EH5 then ships the smart `π_w` axis** ([`eh5.py`](src/verisim/experiments/eh5.py), figure
> [`eh5_subsystem_policy.png`](figures/eh5_subsystem_policy.png)): the factored arm's per-subsystem
> decode entropy drives an information-gain `UncertaintySubsystem` policy that gives a **modest edge**
> over round-robin (matches the best raw horizon at lower cost), though the cheapest-fixed still wins
> pure bit-efficiency (finding
> [§17](#17-the-other-consultation-axis-a-smart-which-subsystem-policy-gives-a-modest-edge-host-eh5--h10)).
> **EH5-heads then closes the open HC7 decode-*heads* lever with a negative**
> ([`eh5_heads.py`](src/verisim/experiments/eh5_heads.py), figure [`eh5_heads.png`](figures/eh5_heads.png)):
> an opt-in trained per-subsystem error head (the calibrated alternative to the post-hoc entropy bucket)
> is **uncalibrated** to held-out per-subsystem error (Spearman −0.02) where the bucketed entropy it was
> meant to replace is **well-calibrated** (+0.57) — its CE target collapses on the overfit train
> distribution; the per-subsystem echo of v0's H2 negative (finding
> [§17](#17-the-other-consultation-axis-a-smart-which-subsystem-policy-gives-a-modest-edge-host-eh5--h10)).
> Both consultation axes (when `π_c` × which `π_w`) are now measured. **The §6.3 drift levers also ship**
> ([`eh4_drift.py`](src/verisim/experiments/eh4_drift.py), figure [`eh4_drift.png`](figures/eh4_drift.png)):
> oracle-relabeled noise injection + self-forcing reproduce the network's **banked negative** — neither
> buys free-running horizon at this scale, so the one-step→horizon gap stays open (finding
> [§18](#18-the-drift-levers-dont-buy-horizon-here-either--the-same-banked-negative-host-eh4-drift--63)).
> **The concurrency scheduler + the H14 dial now ship and CONFIRM H14**
> ([`scheduler.py`](src/verisim/hostdata/scheduler.py), [`eh_h14.py`](src/verisim/experiments/eh_h14.py),
> figure [`eh_h14_interleaving.png`](figures/eh_h14_interleaving.png)): free-running `H_ε` degrades
> **monotonically** with interleaving entropy (~8×, recorded→chaos), the first quantification of HW-1's
> cost and the host world's defining result — the experiment only the host world can run (finding
> [§19](#19-concurrency-is-a-measurable-dial-not-a-binary-wall--the-host-worlds-defining-result-host-eh-h14--h14)).
> **HC8 begins with the §7 LLM-callable simulator** ([`hostsim/`](src/verisim/hostsim/)): `HostSimulator`
> both predicts the next state (the loop) and *simulates a plan* for an agent — `imagine` (oracle-free
> draft) + `verify` (the plan-level faithful horizon + the task "third oracle"), the SLM/LLM-complementarity
> payoff (finding [§20](#20-the-payoff-a-verified-whole-machine-simulator-an-llm-agent-calls-host-7--hc8)).
> And **EH7 confirms H22 in the composed world** ([`eh7.py`](src/verisim/experiments/eh7.py), figure
> [`eh7_invariance.png`](figures/eh7_invariance.png)): four proposers in the same loop share the
> floor+cliff `H_ε(ρ)` shape — the model-agnostic-primitive claim holds in the hardest (coupled,
> concurrent) world (finding [§21](#21-the-deepest-claim-holds-in-the-hardest-world-the-shape-is-the-loops-not-the-models-host-eh7--h22)).
> **Two cross-cutting findings round out the picture:** the **cross-world synthesis**
> ([`synthesis.py`](src/verisim/experiments/synthesis.py), figure [`synthesis_floor_cliff.png`](figures/synthesis_floor_cliff.png))
> overlays the normalized `H_ε(ρ)` of all three worlds onto **one floor+cliff** — the thesis in a single
> figure, model- *and* world-agnostic (finding [§22](#22-the-thesis-in-one-figure-the-floorcliff-is-the-same-in-every-world-cross-world-synthesis));
> and **EH-H14-scale** ([`eh_h14_scale.py`](src/verisim/experiments/eh_h14_scale.py), figure
> [`eh_h14_scale.png`](figures/eh_h14_scale.png)) shows the concurrency collapse **steepens with thread
> count** (~2.5×→~12× from 2→8 threads) — concurrency's cost scales with its width (finding
> [§23](#23-concurrencys-cost-scales-with-concurrencys-width-host-eh-h14-scale)).
> **Six security/scaling/continual-learning findings round out the host:** **EH8** ([§24](#24-aggregate-faithfulness-hides-a-security-critical-denial-gap-host-eh8)) — aggregate
> privilege-faithfulness (0.91–0.94) *hides* a security-critical **denial-recall gap** (flat 0.000,
> factored 0.286: the model rarely predicts the EPERM/EBADF *failures* a defender most needs); **EH6**
> ([§25](#25-a-cheap-symbolic-second-oracle-is-redundant-but-decision-sufficient-host-eh6--h12)) — a
> symbolic privilege second-oracle is redundant for verification (0%) but **decision-sufficient in 95%**
> of error steps at ~3× lower cost (the host H12); **EH-H13-scale**
> ([§26](#26-concurrency-manufactures-the-composition-coupling-host-eh-h13-scale--h13--h14)) — the H13
> coupling is in part **manufactured by concurrency** (the independence gap doubles from 2→4 threads,
> then saturates), tying H13 × H14; and **EH9** ([§27](#27-the-free-oracle-closes-the-eh8-denial-gap--it-was-data-balance-not-architecture-host-eh9))
> — the EH8 denial gap turns out to be a **data-balance artifact the free oracle can fix**: just adding
> the denial-carrying driver to training lifts flat recall 0.000→0.333 and factored 0.286→0.952, and
> **oversampling** denials lifts flat to 0.762 at **no specificity cost** (though too aggressive a factor
> backfires for the flat arm; the structured arm saturates to perfect and is robust). The banked negative
> becomes a measured intervention. **EH-stream/H15** ([§28](#28-the-experience-stream-doesnt-beat-the-batch--but-replay-saves-it-and-the-plasticity-probe-says-why-host-eh-stream--h15--hw-4))
> — the §8.5 Era-of-Experience stream **does not beat the batch** at equal compute (a pre-registered
> negative), but **experience replay** rescues it from collapse (one-step 0.47 vs the no-replay 0.10) and
> the **HW-4 plasticity probe localizes why**: the no-replay stream's plasticity decays to 0.77 vs 0.95
> for the batch/replay arms — "continual learning is hard" turned into a number with the lever (replay)
> that holds the line. And **EH6/H16** ([§29](#29-counterfactual-replay-is-just-more-data-for-plain-supervision--the-same-null-in-the-host-world-host-eh6--h16))
> — free oracle **counterfactual replay** is a **null beyond volume** for plain supervision (it beats the
> base trajectory but loses to a matched-volume control), confirming the network's EN6/H5 result is
> *world-agnostic*.
> **HC8 packaging also ships the oracle-as-reward RL environment** ([`hostrl/`](src/verisim/hostrl/)): the
> host analogue of v0's `rl/` — a `verifiers`-spec reset/step env whose reward *is* a faithful step and
> whose episode **return equals the composed faithful horizon** `H_ε`, with no learned reward model in the
> loop (the verifiable-reward substrate a future denial-aware objective plugs into).
> **And HC8 closes its dependency-free packaging:** the **§16 decentralized verified-contribution protocol**
> ([`contrib/`](src/verisim/contrib/)) — a contributed host transition or trajectory is accepted *iff
> re-running the deterministic oracle reproduces it bit-for-bit* (`verify_transition`/`verify_trajectory`),
> with chaining checks against spliced transitions and a `content_address` integrity hash; what TOPLOC
> verifies *heuristically*, the oracle verifies *exactly*, so contributed data is **trustless by
> construction** — and the **composed-host faithfulness benchmark + Inspect adapter**
> ([`hosteval/`](src/verisim/hosteval/)): `score_host_model` grades any `HostModel` through the composed
> loop, with a single-step QA grader and an `inspect_ai` task behind the `[eval]` extra — the §1.4 missing
> metrology for a *whole machine*, packaged where labs already look.
> The host-world results are written up in the **[technical report](docs/report.md#the-host-world-spec-6-does-faithfulness-compose)**
> (the SPEC-6 §18 honest write-up, per-hypothesis). Remaining: the **Tier-B system oracle**
> (rr/Hermit/gVisor) — a real-OS dependency, deferred under the no-egress posture (SPEC-6 §15).
> The per-subsystem decode *heads* shipped as EH5-heads (above), an honest negative.

**v0 — shell/filesystem world (`src/verisim/`, SPEC-2 §13): complete.**

| Milestone | What | Status |
|-----------|------|--------|
| **M0–M3** | Env + `ReferenceOracle`, `Delta`/`apply`, drivers/data, divergence + `H_ε` + run-records | ✅ |
| **M4–M5** | Neural `M_θ` (from-scratch transformer, constrained decoder) + propose–verify–correct loop | ✅ |
| **M6–M8** | E1–E4 experiments, smart policies/operators, report, faithfulness benchmark + RL env | ✅ |
| **SPEC-2.1** | K0 (learner works) → K1/K2 (floor ~0 → **0.86**) → K3/K4 (knee refuted on single-FS; licenses SPEC-5) | ✅ |

**Network world (`src/verisim/net*`, SPEC-5 §13): graph arm + EN4 + delta-exact + both §6.3 levers + SPEC-8 factory + EN8/EN9.**

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
| **NW8** | **GNN + RSSM graph arm** ([`graph_model.py`](src/verisim/netmodel/graph_model.py)) + §6.3 **noise + self-forcing** levers + **EN4 graph-vs-flat (H11)** + **delta-exact metric** ([`exact.py`](src/verisim/netmetrics/exact.py)) + **SPEC-8 OG1/OG2 data factory** ([`grounding.py`](src/verisim/netdata/grounding.py), [`negatives.py`](src/verisim/netdata/negatives.py)) + **SPEC-8 EN8/OG3 ablation** ([`en8.py`](src/verisim/experiments/en8.py), [`grounded_train.py`](src/verisim/netmodel/grounded_train.py): H23 collapse-tax removed, H24 near-tie) + **SPEC-8 EN9/OG4 ablation** ([`en9.py`](src/verisim/experiments/en9.py): H25 confirmed, H5 fidelity ~2× over VICReg) + **EN7/H22 model-invariance** ([`en7.py`](src/verisim/experiments/en7.py): the floor+cliff `H_ε(ρ)` shape is invariant across null/flat/graph proposers — H22 supported in kind) + **EN5/H7 self-healing** ([`en5.py`](src/verisim/experiments/en5.py): a robust null — neither single-example TTT nor a replay-buffer budget lifts the floor; the floor's levers are scale/objective, not adaptation) + **EN6/H5 counterfactual change-safety** ([`en6.py`](src/verisim/experiments/en6.py): a null for the predictive model beyond a matched-volume control — H5 is objective-dependent) + **EN10/H12 two-oracle** ([`en10.py`](src/verisim/experiments/en10.py) + the Batfish-style [`control_plane.py`](src/verisim/netoracle/control_plane.py): the control-plane oracle is redundant for verification but cheaper + decision-sufficient). With EN1–EN10 the network EN-series is complete | ◐ graph arm + EN4 + both levers + OG1/OG2 + EN8/OG3 + EN9/OG4 + EN7/H22 + EN5/H7 + EN6/H5 + EN10/H12 |
| **SPEC-9 LS0–LS2** | **Free-oracle scaling** ([`en8_scale.py`](src/verisim/experiments/en8_scale.py), [`en9_scale.py`](src/verisim/experiments/en9_scale.py), [`en8_capacity.py`](src/verisim/experiments/en8_capacity.py), [`scale_common.py`](src/verisim/experiments/scale_common.py)): the measured local envelope + the 8× world-size surface with bootstrap CIs ([§8](#8-which-wins-survive-scaling--the-honest-mixed-verdict-spec-9)). **S1** H23 attenuates, **S2** H25/H5 reverses, **S3** H24 regime-dependent. The [`en9_negatives.py`](src/verisim/experiments/en9_negatives.py) S2-recovery diagnostic **confirms** the lift recovers when negatives scale with the world (k 8→32 flips it back to disjoint-positive) | ✅ LS0–LS2 + S2-recovery + S3 frontier |

**SPEC-10 — the faithful-horizon scaling law (`H_ε(ρ=0)` vs capacity, H26): the floor+cliff is a resourcing story, and the lift is cross-world.**

| Milestone | What | Status |
|-----------|------|--------|
| **HS0/HS1** | Capacity-sweep harness + the network curve ([`horizon_scaling.py`](src/verisim/experiments/horizon_scaling.py), [horizon_scaling.png](figures/horizon_scaling.png)): `H_free` lifts **~9×** with capacity (1.75→15.8, disjoint CIs) then saturates — **H26 supported**; the prior floor was an under-resourcing artifact | ✅ |
| **HS1.1** | The **resourced frontier** ([horizon_scaling_xl.png](figures/horizon_scaling_xl.png)): `H_free` is **non-monotone** — peaks at `l` then declines while one-step `p` stays flat/high (the proxy goes blind); the floor lifts ~4× from resourcing even at fixed tiny capacity | ✅ |
| **HS1.2** | The **data cross-axis** ([`horizon_data_scaling.py`](src/verisim/experiments/horizon_data_scaling.py), [horizon_data_scaling.png](figures/horizon_data_scaling.png)): the decline is **data starvation, not a wall** — at fixed `xl`, data recovers `H_free` and ood η from 0.97 back to 1.90 (Chinchilla) | ✅ |
| **HS1.3** | The **joint capacity×data push** ([`horizon_joint_scaling.py`](src/verisim/experiments/horizon_joint_scaling.py), [horizon_joint_scaling.png](figures/horizon_joint_scaling.png)): a compute-optimal ladder lifts the peak to a **program-best `l@9.6k` = 19.2 id / 28.75 ood**, but returns vanish past `l` | ✅ |
| **HS2** | **Universality across worlds** ([`horizon_host_scaling.py`](src/verisim/experiments/horizon_host_scaling.py), [horizon_host_scaling.png](figures/horizon_host_scaling.png)): the *identical* axis on the harder **host** world — the lift is **cross-world** (`H_free` 1.00→5.08, disjoint CIs) but the world **re-lowers the floor ~3–5×** and re-opens the headroom ([§30](#30-capacity-buys-free-running-horizon--and-the-verdict-is-cross-world-spec-10--hs1hs2--h26)) | ✅ |
| **HS3 (incr 1)** | The **structured (graph) arm** ([`horizon_graph_scaling.py`](src/verisim/experiments/horizon_graph_scaling.py), [horizon_graph_scaling.png](figures/horizon_graph_scaling.png)): the *identical* axis with the GNN+RSSM proposer — the lift is **proposer-dependent**, it does **not** reproduce (`p` flat ~0.66, `H_free`≈0, η≈0); HS1's lift was the flat arm's p-climb, not a loop property ([§31](#31-but-the-capacity-lift-is-proposer-dependent--it-does-not-reproduce-for-the-structured-arm-spec-10--hs3--h26)) | ✅ |
| **HS3 (incr 2)** | The **graph data cross-axis** ([`horizon_graph_data_scaling.py`](src/verisim/experiments/horizon_graph_data_scaling.py), [horizon_graph_data_scaling.png](figures/horizon_graph_data_scaling.png)): fixed graph capacity, sweep the coverage set — the structured floor is **NOT data starvation** (10× data lifts neither `H_free`≈0 nor `p`, η<1 — it free-runs *shorter* than i.i.d.). The structured floor moves with **neither capacity nor data** — the first floor that doesn't dissolve into resourcing ([§32](#32-and-the-structured-floor-is-not-data-starvation-either--a-genuine-ceiling-spec-10--hs3-incr-2--h26)) | ✅ |
| **HS3 (incr 3)** | The **world-size cross-axis** ([`horizon_graph_world_scaling.py`](src/verisim/experiments/horizon_graph_world_scaling.py), [horizon_graph_world_scaling.png](figures/horizon_graph_world_scaling.png)): fixed graph capacity, sweep `n_hosts` over SPEC-9's `O(N²)` axis — the ceiling is **world-size-invariant** (`H_free`=0 at every world size 5–40, η=0, `p` *degrades*). Completes the HS3 arc: the structured floor is pinned at 0 across **capacity, data, AND world size** ([§33](#33-and-it-survives-the-world-size-axis-too--the-structured-ceiling-is-world-size-invariant-spec-10--hs3-incr-3--h26)) | ✅ |
| **HS3 (incr 4)** | The **joint capacity×world-size push** ([`horizon_graph_joint_scaling.py`](src/verisim/experiments/horizon_graph_joint_scaling.py), [horizon_graph_joint_scaling.png](figures/horizon_graph_joint_scaling.png)): a structured ladder (bigger graph arm in a bigger world). The ceiling **survives the joint push too** — `H_free`=0 at every rung (s@5h→xl@40h), vs HS1.3's *flat* joint ladder reaching the program-best 19.2/28.75. Across capacity, data, world size, **and** their product the structured floor is pinned at 0 (§33) | ✅ |
| **HS3-T** | The **trainer diagnostic** ([`horizon_graph_schedule.py`](src/verisim/experiments/horizon_graph_schedule.py), [horizon_graph_schedule.png](figures/horizon_graph_schedule.png)): is the graph `p` plateau a *flat-LR* artifact? Give it the flat arm's warmup+cosine schedule (opt-in `warmup_frac`, default-off, regression-pinned). **No** — the schedule lifts `p` only 0.66→0.68 (vs the flat arm's 0.82) and `H_free` stays 0. The plateau is the **representation, not the trainer**; the under-training caveat is refuted (§33) | ✅ |

**Distributed world (`src/verisim/dist*`, SPEC-7 §13): the current build front — the deterministic core (DS0/DS2/DS3/DS5/DS6) ships, the learned `M_θ` flat arm (DS4 incr 1+2) drops into the loop (ED1-learned), and four DS7 results land — the equal-*dollar*-budget H17/H18 frontier on the synthetic proposer (ED2) *and on the real learned `M_θ`* (ED2-learned, where the constrained decoder's subtle-only errors make bit-exact the sole budget-efficient tier — the honest inverse), the **`π_c` smart-when comparison** (ED2-smart, where entropy-gated consultation is *worse* than fixed — the H9 null carried in, sharper than a tie), the **ED3 correction-operator comparison** (where the partial `ReplicasOnlyCorrection` *breaks* v0's operator identity exactly on the in-flight medium — the `subtle` error class), and the H21 fault-injection sweep, which together pin the tiered oracle's value as *model-dependent*, localize the smart-`π_c` lever to the structured arm's belief variance, expose the in-flight medium as the distributed world's hidden state, and confirm the DST data-factory lesson. A fifth DS7 result lands — the **ED4 consistency-level sweep (H20)**: the strong `linearizable` model (synchronous, CP-under-partition, no in-flight medium) is added to the `CONSISTENCY_MODELS` axis, and sweeping the declared model shows the H19 gap is *exclusively* a weak-consistency phenomenon — it tracks the in-flight medium (+10.5 under `eventual`, 0 under `linearizable`). DS8 then opens with **ED5** (the §9.1 consistency-faithfulness metric's first loop consumer): consistency-faithful horizon **outlasts** bit-faithful where the error hides in the consistency-invisible in-flight medium (H19), and the propose-verify-correct loop is a learning-augmented algorithm whose competitive ratio degrades gracefully with prediction error but stays floor→cliff in the budget axis (H18). **ED6** then settles the counterfactual question (H5): training the flat `M_θ` on free oracle **fault-flip** branches beats equal-volume on-policy data on held-out interventions (intervention-exact 0.51 vs 0.25, disjoint CIs) — the *honest inverse* of the network/host supervision null, because the distributed **medium** is the one hidden state on-policy volume cannot reach (the held-out-intervention analogue of the H21 data-factory result). **ED6's two-oracle slice (H12)** closes it: the cheap consistency oracle is *redundant* for verification (non-redundant rate 0) but **decision-sufficient** for the split-brain question (1.00 on in-flight errors, 0.00 on durable — the in-flight medium once more) at ~3.6× lower consult cost — redundant truth, cheaper decision. Finally, the **§16 verified-contribution protocol** ([`distcontrib/`](src/verisim/distcontrib/)) packages the distributed world for the open corpus: a contributed trace is trustless by re-execution, with **tiered acceptance** (bit-exact reproduction *or* cheap-tier admissibility where bit-exact is intractable, the W7 path) — and building it surfaced and fixed a latent serialization bug (partition order made `from_canonical` non-exact).**

| Milestone | What | Status |
|-----------|------|--------|
| **DS0 (incr 1)** | The **replicated-KV-under-partition** core ([`dist/`](src/verisim/dist/), [`distoracle/`](src/verisim/distoracle/)): `DistributedState` (per-(object,node) MVCC replicas + causal event log + in-flight messages + partition/crash/clock), the client (`put`/`get`/`cas`) + fault/time (`advance`/`partition`/`heal`/`crash`/`restart`) grammar, the Tier-A async-replication DES (eventual-consistency LWW), canonical serialization, [distributed-semantics](docs/distributed-semantics.md), and golden trajectories pinning **stale-read-under-partition + convergence** — dependency-free, GPU-free, `apply==oracle` invariant tested every step | ✅ incr 1 |
| **DS2** | The **data factory** ([`distdata/`](src/verisim/distdata/)): seeded workload+fault `DistDriver`s (`uniform`/`contention`/`adversarial`) interleaving client ops + `advance` + faults, with the **explicit `fault_prob` (fault-intensity) + `partition_bias` (partition-entropy) dials** the H20/H21 sweeps need; trajectory JSONL + regenerable dataset manifests with disjoint trajectory-level splits — tested for valid-action/`apply==oracle`, determinism, and dial monotonicity | ✅ for the incr-1 world |
| **DS3 (metric core)** | The **metric core** ([`distmetrics/`](src/verisim/distmetrics/)): live-cluster **divergence** `d(s,ŝ)` (feeds the generic `faithful_horizon`, so distributed `H_ε(ρ)` is defined as in every world), the **headline-new consistency-faithfulness** (§9.1 — did the model predict each object's converged/split state? it catches a partition-split mispredicted as converged), and **bits-to-correct / delta-exact** over the `DistDelta` | ✅ |
| **DS3 (tiered oracle)** | **SPEC-7's payload** ([`distoracle/tiers.py`](src/verisim/distoracle/tiers.py)): the four-tier menu **metamorphic** ¢1 → **cycle** ¢2 → **symbolic** ¢4 → **bit-exact** ¢16, where `cheapest_refutation` spends the cheapest tier that can refute a prediction (DD-D1) and records the cumulative oracle-dollar — every error class caught at its right tier, and a subtle invariant-respecting error caught only by bit-exact (the non-redundancy **H17** measures) | ✅ |
| **DS5 (the loop)** | The **tiered propose-verify-correct loop** ([`distloop/`](src/verisim/distloop/)): the model-agnostic runner with the **`π_w` which-tier axis** (fixed | cheapest-refutation escalate) and the **oracle-dollar accounting** (each consult spends its tier's cost; a refutation adds the bit-exact correction; an unrefuted prediction is trusted) — loop invariants tested with the null/oracle-backed baselines (ρ=1 reproduces truth, perfect never drifts at $0, null drifts at step 0, budget exact). The record carries divergences (→`H_ε`) **and** oracle-dollars (→H17) | ✅ baselines |
| **DS6 (ED1, the prime directive)** | The **distributed `H_ε(ρ)` curve + the tiered-oracle (H17) measurement** ([`experiments/ed1.py`](src/verisim/experiments/ed1.py), [ed1_dist.png](figures/ed1_dist.png)): the curve is the *same floor→cliff* (0.2→40); and **H17's first verdict** — *cheap tiers win per oracle-dollar conditionally*, for **gross** errors ($9.4 vs $16/faithful-step) but not **subtle** ones ($848). On a controlled-noise proposer (the apparatus before the learned `M_θ` supplies a real error distribution) | ✅ apparatus |
| **DS6 (ED1, learned `M_θ`)** | The **real-model** distributed curve + H17 ([`experiments/ed1_learned.py`](src/verisim/experiments/ed1_learned.py), [ed1_learned.png](figures/ed1_learned.png)): the flat DS4 `M_θ` trained then run through the *same* tiered loop. Same **floor→cliff** (0.2→32). The **real-model H17 is the honest inverse of the synthetic one**: the constrained decoder removes gross errors by construction, so the model's residual errors are *subtle* — **metamorphic** catches none ($624/faithful-step), only **bit_exact** is efficient ($16), and cheapest-refutation **escalate** pays *more* ($21.6, it needs the bit-exact correction anyway). *A cheap tier helps only for cheaply-catchable errors, which a grammar-constrained learned model does not make — the tiered oracle's value is model-dependent* | ✅ learned |
| **DS4 (incr 1)** | The learned `M_θ` **serialization foundation** ([`distmodel/`](src/verisim/distmodel/), torch-free modules): the closed `DistVocab` (a single bounded `<int:..>` pool closes the monotone bookkeeping counters — the host's `max_pid` trick) and the bidirectional tokenizer with an **exact `parse_target`**. The key move: the causal-log `EventAppend` (its `happens_before` is the one variable-length field) is a bare marker **reconstructed from `(state, action)`** on parse, keeping the unbounded list out of the grammar. Round-trip `parse(encode(Δ))==Δ` tested exhaustively (every preset × seeds × 40 steps, a 5-node cluster, a multi-group partition), the decoded delta still satisfying `apply==oracle` | ✅ incr 1 |
| **DS4 (incr 2)** | The **learned (flat) arm** ([`grammar.py`](src/verisim/distmodel/grammar.py), [`decode.py`](src/verisim/distmodel/decode.py), [`world_model.py`](src/verisim/distmodel/world_model.py)): the LL(1) `DistDeltaGrammar` (two structured nonterminals the flat net/host grammars lack — the **nested partition run** and the **status-typed result**, where `advanced`→int and every other status→value), the `NeuralDistWorldModel` over v0's `GPT` (a drop-in `DistModel` with a decode-entropy uncertainty signal), and supervised dataset builders feeding the generic `verisim.train` trainers. Free-running decode surfaced a **structural bug** — an untrained model could emit `<event_append>` after a non-client action whose `args[0]` is no node — so the decoder masks it out for fault/time ops (§5.1). Tested: grammar-valid-from-untrained, overfit-to-<0.05-and-free-run-back (each still `apply==oracle`), `DistModel`-protocol, 5-node config-driven | ✅ incr 2 |
| **DS7 (ED2, equal-dollar-budget)** | **H17 in budget form + the H18 competitive ratio** ([`experiments/ed2.py`](src/verisim/experiments/ed2.py), [ed2.png](figures/ed2.png)): where ED1 reported cost *per faithful step at ρ=1*, ED2 sweeps ρ and plots the **faithful-horizon-vs-oracle-dollar frontier** per tier policy (`metamorphic`/`symbolic`/`bit_exact` + cheapest-refutation `escalate`), comparing policies **at a matched dollar budget** by interpolating each one's horizon along its Pareto envelope (a true equal-*dollar*, not equal-ρ, comparison). At the sub-linear **quarter budget** `B/4`: for **gross** errors the metamorphic tier buys **H=14.2 vs bit-exact's 4.2** (tiering wins, H18 ratio 0.36 of the full-truth ceiling at ¼ the cost); for **subtle** errors the cheap tiers are flat at the floor (**1.5 vs 4.2**) and even `escalate` *loses* to single-tier bit-exact — H17's honest negative, in the form the spec poses it | ✅ ED2 |
| **DS7 (ED2, learned `M_θ`)** | **H17/H18 in budget form on the *real* model** ([`experiments/ed2_learned.py`](src/verisim/experiments/ed2_learned.py), [ed2_learned.png](figures/ed2_learned.png)): ED2's synthetic frontier re-pointed at the trained flat DS4 `M_θ` — what ED1-learned is to ED1, this is to ED2. The constrained decoder removes the gross error class, so a real model lives entirely in ED2's `subtle` regime: at the sub-linear **quarter budget** `B/4=$128` the cheap tiers stay flat at the floor (**metamorphic 0.2, symbolic 0.8**) while only **bit_exact** buys horizon (**2.0**), and **escalate** *loses* (1.6, and spends `$691 vs $512` to reach the same H=32 ceiling). The **H18 ratio at `B/4` is 0.06** — for a grammar-constrained learned model a sub-linear budget buys little horizon however the tiers are sliced. *The budget-form of ED1-learned's per-step finding, and the honest inverse of ED2's `gross` panel: the tiered oracle's value is model-dependent, reported not assumed* | ✅ learned |
| **DS7 (ED2, smart-`π_c`)** | **The "smart-when" axis of ED2 — does entropy-gated consultation beat fixed?** ([`experiments/ed2_smart.py`](src/verisim/experiments/ed2_smart.py), [ed2_smart.png](figures/ed2_smart.png)): at a fixed interior budget `ρ`, compare `fixed` / `uncertainty` / `drift` `π_c` policies at *equal* `ρ` on the real flat `M_θ` (its constrained-decode entropy as the signal). The DS5 runner gained the uncertainty plumbing it was missing — a `_predict` helper + `DistUncertaintyModel` protocol feeding the per-step signal into the loop's `StepContext`, as the net/host runners already do. **H9 — the standing H2/H9 negative carried into the distributed world, and *sharper than a tie*:** entropy-gating does **not** beat `fixed` — it is strictly *worse* (lift **0.08–0.12×** at every budget), because faithful horizon is a *prefix* property: `fixed` consults at step 0 to protect the prefix while the entropy signal spends late and lets the model derail early. Flat decode entropy is a decode-time artifact, not a calibrated belief — localizing the smart-`π_c` lever to the (deferred) structured `M_θ`'s belief variance, the EH2 lesson | ✅ smart-π_c |
| **DS7 (ED3, operators)** | **Correction operators — and the distributed world *breaks* v0's operator identity** ([`experiments/ed3.py`](src/verisim/experiments/ed3.py), [ed3.png](figures/ed3.png); [`distloop/operator.py`](src/verisim/distloop/operator.py)): the DS5 runner gained the §8.3 correction-operator axis (`HardReset` default + `Residual`/`Projection` diagnostics + the partial `ReplicasOnlyCorrection`). v0 proved an *identity* — a full-truth consult makes `hard_reset`/`residual`/`projection` behaviorally identical on `H_ε`. ED3 breaks it, *mode-dependently*: `ReplicasOnlyCorrection` snaps the durable replicas but **trusts the model's predicted in-flight** (the stale-read medium, the `subtle` error class). For **gross** (replica-write) errors all four operators match (H=7.2, identity holds); for **subtle** (in-flight) errors the three full-correction operators hold (H=6.2) but `ReplicasOnlyCorrection` **collapses to H=1.8** (gap 4.5) — the in-flight medium is the distributed world's hidden state a partial correction cannot see, tied to the gross/subtle structure H17 turns on | ✅ ED3 |
| **DS7 (H21, fault-injection)** | The **DST/BUGGIFY data-factory lesson** ([`experiments/ed4_fault.py`](src/verisim/experiments/ed4_fault.py), [ed4_fault.png](figures/ed4_fault.png)): train two equal-volume `M_θ` — one fault-free (`fault_prob=0`), one fault-injected — then sweep eval fault-intensity **free-running**. **H21 confirmed**: as faults intensify the fault-injected model holds ~3× more free-run horizon (0.375 vs 0.125) — *and the fault-free model is the **better** clean predictor* (acc 0.60 vs 0.49) yet less fault-robust (a bonus proxy/truth-divergence instance). Fault injection buys robustness factual data cannot — DST as a *data factory*, not just a test harness | ✅ H21 |
| **DS7 (ED4, H20 consistency level)** | **Weaker consistency opens the H19 gap** ([`experiments/ed4_consistency.py`](src/verisim/experiments/ed4_consistency.py), [ed4_consistency.png](figures/ed4_consistency.png)): the `CONSISTENCY_MODELS` dial (§3.4) gains its first strong end — **`linearizable`** (synchronous all-replica writes, CP write-rejection under partition, **no in-flight medium**; [distributed-semantics §2.1](docs/distributed-semantics.md), goldens in [`test_dist_goldens`](tests/test_dist_goldens.py)). Sweeping the declared model resolves H20 *through* H19: the consistency-vs-bit gap is exclusively a *weak*-consistency phenomenon — it needs the consistency-invisible in-flight medium. The `subtle` (in-flight) gap is **+10.5 under `eventual` (in-flight rate 3.2/step) and 0 under `linearizable` (rate 0)**; the `gross` durable-replica control is 0 at both. Strong consistency buys no forgiveness because there is no hidden state to forgive — the H20 mechanism made concrete, dependency-free | ✅ H20 (gap) |
| **DS8 (ED5, H19 + H18)** | **Consistency-vs-bit horizon + the competitive-ratio fit** ([`experiments/ed5.py`](src/verisim/experiments/ed5.py), [ed5.png](figures/ed5.png)): the §9.1 consistency-faithfulness metric gets its first loop consumer — the DS5 runner now records the consistency-divergence trajectory alongside bit-exact divergence — and ED5 reads both findings off the dependency-free synthetic proposer. **H19 confirmed mode-dependently**: free-running, the consistency-faithful horizon **outlasts** the bit-faithful one for `subtle` (in-flight) errors (**H=13.1 vs 1.5, gap +11.6, disjoint CI** — the in-flight message is bit-visible but consistency-invisible until delivered) and **coincides** for `gross` (durable-replica) errors (the control). **H18 split**: the competitive ratio fit across `ρ × prediction error` confirms graceful degradation in the *error* axis (quarter-budget ratio monotone 1.00 → 0.05, trivial bound recovered for a perfect model) but reproduces the **floor→cliff / no-knee** negative in the *budget* axis (ratio near the floor at `B/4`, cliff only at ρ→1) — learning-augmented in kind, no free lunch at sub-linear budget | ✅ ED5 |
| **DS8 (ED6, H5)** | **Counterfactual grounding — the distributed world is where it finally pays** ([`experiments/ed6.py`](src/verisim/experiments/ed6.py), [ed6.png](figures/ed6.png)): three matched-count arms train the same flat `M_θ` — `trajectory` (base), `trajectory-more` (5× more on-policy data, the volume control), `+counterfactual` (base + free oracle **fault**-flip branches) — then predict **held-out fault interventions**. **H5 confirmed, the honest inverse of EN6/EH6**: `+counterfactual` beats **both** the base **and** the volume control on **both** metrics with disjoint CIs — intervention-exact **0.51 vs 0.25 vs 0.06**, medium-recall (predicts the split-brain) **0.56 vs 0.22 vs 0.05** — where the network and host found counterfactual supervision adds nothing over volume. The mechanism is the distributed **medium** (partition/crash/in-flight): a hidden state the light-fault on-policy distribution underrepresents, so volume buys little while off-policy oracle fault branches buy a lot — the held-out-intervention analogue of H21. The first distributed experiment to need the minibatched `train_batched` K2 loop (a real perf fix over the full-batch path). *Honest caveat: the branches are fault-heavier than the control, so the lift conflates counterfactual branching with the fault coverage it carries — but EN6/EH6 found null under the identical design* | ✅ ED6 |
| **DS8 (ED6 two-oracle, H12)** | **The consistency oracle is redundant but decision-sufficient and cheaper** ([`experiments/ed6_two_oracle.py`](src/verisim/experiments/ed6_two_oracle.py), [ed6_two_oracle.png](figures/ed6_two_oracle.png)): the distributed analogue of the host's privilege second-oracle (EH6) and network's control-plane oracle (EN10) — the cheap **consistency oracle** (the §9.1 split-brain decision) vs the full **bit-exact** one, teacher-forced over the fault-heavy `adversarial` workload (dependency-free). **H12 confirmed, mode-dependently**: **non-redundant rate 0** by construction (a bit-exact-correct prediction is always consistency-correct — *redundant for verification*); **consistency-sufficient 1.00 for `subtle` (in-flight) vs 0.00 for `gross` (durable-replica) errors** (disjoint CIs — the per-step form of ED5's H19 horizon gap, tracking the in-flight medium); at a **consult-fact ratio of 0.28** (~3.6× cheaper, widening under fault). *Redundant for verification, but a cheaper, decision-sufficient consult for the question an SRE/defender actually asks* | ✅ H12 |
| **DS8 (§16 verified contribution)** | **Trustless distributed-trace contribution by re-execution** ([`distcontrib/`](src/verisim/distcontrib/), [`test_distcontrib.py`](tests/test_distcontrib.py)): the dependency-free distributed analogue of the host [`contrib/`](src/verisim/contrib/) — the concrete form of the open/decentralized intent. `verify_transition`/`verify_trajectory` accept a contributed trace iff re-running the deterministic oracle reproduces it; trajectories must *chain* (no splicing); `content_address` is the tamper-evident SHA-256 manifest hash; hostile input is *rejected, never raised*. The distributed-specific novelty is **tiered acceptance**: `bit_exact` demands byte-for-byte reproduction, while the cheap `metamorphic`/`cycle`/`symbolic` tiers admit *any* next-state legal under the declared model — the **W7 path** (a contributor running an equally-valid but byte-different schedule is admitted by the consistency tier where bit-exact rejects it; a read that mutates a replica is still caught). *Fixed in the build: `from_canonical(to_canonical(s))` was non-exact because `partitions` was stored in construction order while serialization sorted it — fixed at the source (canonical partition order in `apply` + `__post_init__`), pinned by a round-trip test.* | ✅ §16 |
| **DS0 (incr 2+)** | The Raft-subset consensus group, the transaction/lock table, and the embedded SPEC-6 host / SPEC-5 net inside each node | ☐ next |
| **DS4 (graph arm)** | The service-graph message-passing + RSSM-belief `M_θ` (the structured arm, §6.1-6.2) — supplies a *real* error distribution under partial observation, mirroring the EN4/EH4 graph arms | ☐ next |

**SPEC-11 — the system oracle (validating the reference oracle against a real `/bin/sh`, H27–H30): complete.** The program's one structural bet, measured. Built **macOS-first** ([§35](#35-the-oracle-is-faithful-to-a-real-computer--the-one-structural-bet-measured-spec-11--sy1--h27)); the cross-POSIX [`SandboxOracle`](src/verisim/oracle/sandbox.py) runs on the Mac dev host and Linux CI alike, all dependency-free (pure stdlib).

| Milestone | What | Status |
|-----------|------|--------|
| **SO0** | The [`SandboxOracle`](src/verisim/oracle/sandbox.py) — a real `/bin/sh` over a real kernel behind the v0 `Oracle` protocol — with the [`DeterminismSeal`](src/verisim/oracle/sandbox_seal.py) (env/umask/rlimit/no-new-privs) and the [`differential`](src/verisim/oracle/differential.py) harness (world/exit/stdout channels + the divergence classifier). Materialize → render → exec → snapshot → destroy, with v0-`resolve` path confinement | ✅ |
| **SO1** | **SY3 hermeticity** ([`sy3.py`](src/verisim/experiments/sy3.py), [sy3_hermeticity.png](figures/sy3_hermeticity.png)): the negative-control battery — filesystem-escape, egress, privilege-gain, persistence each **denied**, the positive control allowed-then-discarded. "No action can occur" (H29) is a green table | ✅ |
| **SO2** | **SY4 determinism** ([`sy4.py`](src/verisim/experiments/sy4.py), [sy4_determinism.csv](figures/sy4_determinism.csv)): a fixed `(s,a)` battery is **bit-identical across repeats** under the seal, report all-sealed — licensing SY1 as *true* determinism, not recorded replay (H30) | ✅ |
| **SO3** | **SY1 agreement** ([`sy1.py`](src/verisim/experiments/sy1.py), [sy1_agreement.png](figures/sy1_agreement.png)): Tier-1 exhaustive + Tier-2 trajectories + the Tier-3 head-to-head curve. **Structure-building agreement = 1.000 bit-exact, residual = 0, curve gap = 0** — *the figure that retires W1* (H27) | ✅ |
| **SO4** | **SY2 debugger** ([`sy2.py`](src/verisim/experiments/sy2.py), [sy2_disagreements.csv](figures/sy2_disagreements.csv)): the divergence atlas (4 named boundaries) + the **teeth control** (a planted synthetic divergence is caught and localized). Already caught **one real reference bug** — subtree-move/copy — now fixed (H28) | ✅ |

The deterministic cores (filesystem, network, and the distributed DS0 core) **and the SPEC-11 system
oracle** have **no runtime dependencies** and need no GPU.
PyTorch is an optional `[model]` extra (see [docs/model-representation.md](docs/model-representation.md)).

## Concepts cheat-sheet

| Term | Meaning | Where |
|---|---|---|
| `O(s, a)` | the **oracle**: deterministic interpreter returning the exact next state + delta | `oracle/`, `netoracle/` |
| `Δ` (delta) | the structured edit set a step makes; `apply(s, Δ)` reconstructs `s'` | `delta/`, `netdelta/` |
| `Mθ` | the **learned proposer** (`predict_delta`); any model behind the `Model` protocol — flat serializer or factored interaction-graph (GNN+RSSM) | `model/`, `netmodel/`, `hostmodel/` |
| `d(a, b)` | **divergence**: normalized symmetric set/graph/bundle difference, `0` iff identical (host: composed + per-subsystem) | `metrics/`, `netmetrics/`, `hostmetrics/` |
| `H_ε(ρ)` | **faithful horizon**: first step where `d > ε`, as a function of consultation budget `ρ` | `metrics/horizon.py` |
| `ρ` | **consultation budget** ∈ [0,1]: fraction of steps the oracle is consulted | `loop/policy.py` |
| `p` | **one-step acceptance** (SPEC-10): teacher-forced fraction of steps whose predicted delta is *exactly* the oracle's — the per-step accuracy capacity is known to lift | `netmetrics/exact.py` (`delta_exact_rate`) |
| `H_free` | **free-running faithful horizon** `H_ε(ρ=0)`: steps the unaided model self-rolls before diverging — the SPEC-10 headline | `experiments/horizon_scaling.py` |
| `H_indep` | the **i.i.d. (geometric) baseline** `p/(1−p)`: the horizon if per-step failures were independent (no compounding), clamped at the eval cap | `experiments/horizon_scaling.py` (`independence_horizon`) |
| `η` | **horizon efficiency** `H_free / H_indep`: the scale-free compounding penalty — `η>1` = the rollout self-stabilizes (free-runs *longer* than i.i.d.); `η<1` = errors compound (free-runs *shorter*) | `experiments/horizon_scaling.py` |
| bits-to-correct | MDL of the oracle's correction of `Δ̂`; `0` iff the prediction is exactly right (host: per-subsystem) | `metrics/bits.py`, `hostmetrics/bits.py` |
| composition-law | host H13 diagnostic: is composed `H_ε` multiplicative (∏ aᵢ) ↔ weakest-link (min aᵢ) ↔ coupled? | `hostmetrics/composition.py` |
| interleaving entropy | host H14 dial: thread context-switch rate of a chaos-scheduled workload; `H_ε(interleaving-entropy)` quantifies concurrency's cost | `hostdata/scheduler.py` |
| plan `H_ε` / task oracle | §7 simulator: `imagine`/`verify` a syscall *plan*; plan-faithful-horizon = steps an agent can trust the draft; the task oracle (`Goal`) is the *third* oracle (did the plan succeed?) | `hostsim/` |
| verified contribution | §16: accept a contributed `(state, action, next_state[, delta])` iff the oracle reproduces it bit-for-bit; trajectories must *chain*; `content_address` is the integrity hash — trustless by construction | `contrib/protocol.py` |
| **delta-exact** | per-step: did free decode assemble the exact edit set? (`bits_to_correct = 0`) | `netmetrics/exact.py` |
| full / probe | oracle consultation modes: whole next-state vs one host's local view (cheap) | `netloop/observe.py` |
| `π_w` | **which-subsystem** policy (host): *which truth-source to buy* on a consult — proc/fd/fs/global; fixed / round-robin / **uncertainty** (the smart, information-gain choice from per-subsystem decode entropy); the `SubsystemFilter` corrects only that one | `hostloop/subsystem.py`, `hostloop/operator.py` |
| `D` / `R` | next-state bits the oracle **decides** vs the genuine **residual** (SPEC-8 partition) | `netdata/grounding.py` |
| oracle-anchored target | a JEPA target pinned to the *true next state* (external referent) instead of a learned EMA | `netmodel/grounded_train.py` |
| collapse readout | embedding std + effective rank — JEPA's collapse diagnostic (→ 0 / → 1 under collapse) | `netmodel/grounded_train.py` |
| noise / self-forcing | §6.3 drift levers: random input corruption vs model's-own-drift rollout, both oracle-relabeled | `netmodel/graph_train.py` |
| reachability-faithfulness | fraction of can-A-reach-service(B) entries that agree | `netmetrics/divergence.py` |
| MVCC replica | a node's `(version, value)` copy of an object; convergence is **last-writer-wins by `(version, value)`** | `dist/state.py` |
| no global state | SPEC-7's W7: there is *no* stored global snapshot — under partition replicas legitimately disagree; a consistent read is a coordinated (expensive) oracle call | `dist/state.py` |
| async replication / `advance` | a `put` writes locally + enqueues messages; they deliver only on `advance`, if due *and* reachable — the source of stale reads | `distoracle/reference.py` |
| partition / `connected` | nodes can exchange messages iff they share a partition group; the fault medium that makes the dynamics exist | `dist/state.py` |
| **tiered oracle** | SPEC-7's payload: the bit-exact global oracle is *intractable*, so the policy chooses the cheapest tier (metamorphic ↔ cycle ↔ symbolic ↔ bit-exact) that can refute the current prediction (H17, later DS) | `distoracle/` |
| consistency-faithfulness | SPEC-7's headline-new metric (§9.1): the fraction of objects whose **consistency view** — the converged/split `(version,value)` set across replicas — the model predicts right (the distributed analogue of reachability-faithfulness) | `distmetrics/divergence.py` |
| oracle-dollar / `π_w` | the distributed loop's new axis (§8.2): not just *when* to consult (`π_c`) but *which tier* (`π_w`); each consult spends its tier's **oracle-dollar** cost, recorded so `H_ε` can be plotted against dollars, not just consults (the H17 quantity) | `distloop/` |
| **system oracle** | SPEC-11: a *real* `/bin/sh` over a real kernel behind the same `Oracle` protocol — validates the from-scratch reference oracle against reality, bit-for-bit | `oracle/sandbox.py` |
| `differential_step` | runs the reference + system oracles on the same `(s,a)`; compares **world** (fs+cwd+env) / **exit** / **stdout** channels; classifies any disagreement | `oracle/differential.py` |
| hermeticity / `DeterminismSeal` | the sandbox is a *closed system* (throwaway tree, no egress, no privilege gain, no persistence, rlimits) under a per-step env/umask/rlimit seal — "no action can occur" (SY3/H29) | `oracle/sandbox.py`, `oracle/sandbox_seal.py` |
| modeling boundary | a *named, intentional* place v0 deviates from raw POSIX (root-protection / overwrite-policy / permission-enforcement / self-subtree); SY1 residual = 0 means every divergence is one of these | `oracle/differential.py` |

### SPEC-10 scaling cheat-sheet (the whole arc, one table)

The scaling layer sweeps the prime-directive metric `H_free = H_ε(ρ=0)` along resource axes and reads it
*exactly* against the oracle. The throughline: the floor's *shape* is the loop's (world/model-invariant),
but whether its *height* is a resourcing artifact is **proposer-dependent**.

| Milestone | Holds fixed → sweeps | World / proposer | Verdict |
|---|---|---|---|
| **HS1** | world → **capacity** | network / flat | `H_free` lifts ~9× (1.75→15.8), then saturates — the floor was under-resourcing |
| **HS1.1** | (resourced) → capacity ~400× | network / flat | non-monotone: peaks at `l`, then a data-starvation decline the proxy `p` can't see |
| **HS1.2** | capacity `xl` → **data** | network / flat | the decline is **data starvation** — data recovers `H_free`, ood η 0.97→1.90 |
| **HS1.3** | → **capacity × data** (ladder) | network / flat | program-best `l@9.6k` = 19.2/28.75, then returns vanish past `l` |
| **HS2** | world → capacity | **host** / flat | lift is **cross-world** (1.00→5.08), but the harder world re-lowers the floor ~3–5× |
| **HS3 (1)** | world → capacity | network / **graph** | **proposer-dependent**: capacity buys neither `p` nor `H_free` (η≈0) — lift does *not* reproduce |
| **HS3 (2)** | capacity → data | network / graph | **not** data starvation — a genuine ceiling; η<1 (compounds), the wall the flat arm escaped |
| **HS3 (3)** | capacity → **world size** | network / graph | ceiling is **world-size-invariant** (`H_free`=0, 5→40 hosts) |
| **HS3 (4)** | → **capacity × world** (ladder) | network / graph | ceiling **survives the joint push** (`H_free`=0), vs HS1.3's flat ladder reaching 19.2 |
| **HS3-T** | capacity → **LR schedule** | network / graph | the `p` plateau is the **representation, not the flat LR** (schedule lifts only 0.66→0.68) |
| **HS-synth** | — (figures-from-records) | flat vs graph overlay | the capstone: the floor is **proposer-dependent**, in one figure |

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
| **trustless by re-execution** | contributed data is accepted iff the deterministic oracle reproduces it bit-for-bit (`contrib/`, §16) | no trust to establish, no tampering to detect probabilistically — what TOPLOC checks heuristically, the oracle settles *exactly* and for free |

## Verification

The claims above are audited empirically in [docs/verification.md](docs/verification.md): the core
invariants (`apply == oracle`, serialization round-trips, the NW4 tokenizer, metric bounds, exit codes,
in- and cross-process determinism) are proven over **48,000 oracle transitions with zero failures** by
the dependency-free, torch-free [`scripts/verify_invariants.py`](scripts/verify_invariants.py) — and
additionally over the **entire action space** (448,260 state×action pairs) by construction, with
**negative controls** confirming each check detects deliberate corruptions. Every quantitative number in
the report and this README is machine-checked against the committed figure CSVs; the figures regenerate
from config + seeds with `maxΔ = 0`; the NW5 partial-observation loop invariants are tested (ρ=1
full-consult is exact; a one-host probe corrects strictly less than a full consult); and the packaging is
verified end-to-end (the RL-env return equals the faithful horizon, the benchmark separates a perfect
from a trivial model, coverage spans all 13 commands).

## Packaging for reuse

The env + metric are packaged where researchers already look (SPEC-2 §15):

- **Faithfulness benchmark** ([`verisim.eval`](src/verisim/eval/)) — dependency-free; `score_model` /
  `score_suite` grade *any* model implementing the loop `Model` protocol against the oracle's ground
  truth, and `step_labels` + `grade_prediction` expose single-step labels for question-answer frameworks.
  An `inspect_ai` task adapter ships behind the optional `[eval]` extra.
- **Oracle-as-reward RL environment** ([`verisim.rl`](src/verisim/rl/)) — a `verifiers`-spec
  `WorldModelEnv` (with the `load_environment` entrypoint) whose reward is the oracle's faithfulness
  verdict, so the episode return *is* the faithful horizon.

The host world (SPEC-6) ships the same surfaces for a **whole machine** — the metrology the
computer-use field lacks (OSWorld/TheAgentCompany grade the agent, never a simulator of the host's
predicted next state):

- **Composed-host faithfulness benchmark** ([`verisim.hosteval`](src/verisim/hosteval/)) — the host
  analogue of `verisim.eval`: `score_host_model` grades any `HostModel` through the composed loop
  (composed `H_ε`, oracle calls); `host_step_labels` / `grade_host_prediction` are the single-step QA
  form; `host_faithfulness_task` is the `inspect_ai` adapter (behind `[eval]`).
- **Oracle-as-reward host RL env** ([`verisim.hostrl`](src/verisim/hostrl/)) — episode return = the
  *composed* `H_ε`. **LLM-callable whole-machine simulator** ([`verisim.hostsim`](src/verisim/hostsim/))
  — `imagine` (oracle-free plan rollout) + `verify` (plan-level faithful horizon + task-oracle `Goal`
  agreement); propose-verify-correct lifted to the *plan* level (§7).
- **Decentralized verified-contribution protocol** ([`verisim.contrib`](src/verisim/contrib/)) — the
  concrete form of the open/decentralized intent (§16). A contributed transition or trajectory is
  accepted **iff re-running the deterministic oracle reproduces it bit-for-bit**, with chaining checks
  against spliced transitions and a `content_address` integrity hash. What TOPLOC verifies
  *heuristically* (INTELLECT-2), the oracle verifies *exactly* — so contributed data is **trustless by
  construction**, with no trust to establish and no tampering to detect probabilistically.

```python
from verisim.eval import score_model, FaithfulnessSample
from verisim.loop import OracleBackedModel
from verisim.oracle import ReferenceOracle

oracle = ReferenceOracle()
score = score_model(OracleBackedModel(oracle), FaithfulnessSample("adversarial", 200, 24), oracle=oracle)
assert score.normalized_horizon == 1.0   # a perfect model is fully faithful, unaided
```

```python
# Trustless contribution: the oracle re-executes and settles it — free and certain (§16).
from verisim.contrib import verify_trajectory
from verisim.hostdata import generate_host_trajectory
from verisim.host import DEFAULT_HOST_CONFIG
from verisim.hostoracle import ReferenceHostOracle

traj = generate_host_trajectory(ReferenceHostOracle(), DEFAULT_HOST_CONFIG, "forky", seed=7, n_steps=24)
report = verify_trajectory(traj.to_dict())
assert report.accepted                    # reproduces bit-for-bit; a forged step would be rejected
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

```python
# SPEC-11: the SAME loop, but verified against a REAL /bin/sh on a real kernel.
# The SandboxOracle is a drop-in Oracle; on the structure-building grammar it agrees
# with the reference oracle bit-for-bit (the figure that retires W1).
from verisim.oracle import ReferenceOracle, SandboxOracle
from verisim.oracle.differential import differential_step
from verisim.env import State, parse_action

ref, sysorc = ReferenceOracle(), SandboxOracle()   # SandboxOracle: pure stdlib, any POSIX host
s = State.empty()
for cmd in ["mkdir /a", "write /a/f alpha", "mkdir /a/b", "mv /a/f /a/b/g"]:
    rec = differential_step(s, parse_action(cmd), ref, sysorc)
    assert rec.agree                                # ref's predicted next state == reality, bit-exact
    s = ref.step(s, parse_action(cmd)).state
```

Reproduce every figure (E1–E4, calibration, K0/K2/K4, the EN1 curve, EN2/EN3, the EN4 graph-vs-flat
comparison, the EN8 oracle-grounded-SSL ablation, the EN9 oracle-contrastive ablation) from config +
seeds — `figures/reproduce.sh` runs them all:

```bash
bash figures/reproduce.sh
# or the NW8/SPEC-8 smoke figures on their own (each writes CSV + PNG directly):
python -m verisim.experiments.en4_graph --graph-iters 1500 --out figures/en4_graph_vs_flat.csv
python -m verisim.experiments.en8 --out figures/en8_grounding.csv
python -m verisim.experiments.en9 --out figures/en9_contrastive.csv
# the SPEC-9 scaling work (multi-seed, bootstrap CIs; the surface preset is slower):
python -m verisim.experiments.en8_scale --world-sizes 5 10 15 --seeds 0 1 2 3 --out figures/en8_scale.csv
python -m verisim.experiments.en9_scale --world-sizes 5 10 15 --seeds 0 1 2 3 --out figures/en9_scale.csv
python -m verisim.experiments.en8_capacity --out figures/en8_capacity.csv   # the H24/S3 frontier
python -m verisim.experiments.en8_scale --world-sizes 25 50 100 200 --d-models 64 128 --seeds 0 1 2 \
    --out figures/en8_surface.csv   # the §8 surface (likewise en9_scale for en9_surface.png)
# the SPEC-11 system-oracle validation (seconds, no torch; gate order SY3 -> SY4 -> SY1 -> SY2):
python -m verisim.experiments.sy3 --csv figures/sy3_hermeticity.csv --plot figures/sy3_hermeticity.png
python -m verisim.experiments.sy4 --csv figures/sy4_determinism.csv
python -m verisim.experiments.sy1 --config configs/sy1.json --csv figures/sy1_agreement.csv \
    --plot figures/sy1_agreement.png   # structure-grammar agreement = 1.000, the figure that retires W1
python -m verisim.experiments.sy2 --csv figures/sy2_disagreements.csv
```

## Layout

The package map and data flow are in [Architecture & system design](#architecture--system-design) above;
the full module-by-module layout is [SPEC-2 §10](docs/specs/SPEC-2.md) (filesystem) and
[SPEC-5 §16](docs/specs/SPEC-5.md) (network). Everything is under [src/verisim/](src/verisim/). Experiment
configs live in [configs/](configs/); plotting scripts + committed figures (PNG + CSV) in
[figures/](figures/); the run-records they read are git-ignored and regenerable from config + seeds.

## License & posture

MIT (see [LICENSE](./LICENSE)). This is a research repo: **no telemetry, no network calls at runtime, no
commercial path.** The framing and downstream agents are defensive; see
[SPEC.md §13](docs/specs/SPEC.md) for the ethics and dual-use posture.

Author: Clay Good.
