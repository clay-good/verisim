# SPEC-7 — The Distributed World

**Engineering & experiment specification for replicated services, transactions, and
consensus — the world that composes the others into a running *system*, where the
bit-exact global oracle becomes *provably intractable*, so faithfulness must be
verified at *tiered* cost, and where the deterministic-simulation-testing tradition
supplies both the oracle and the training data for free.**

This is to the *distributed system* what [SPEC-2](./SPEC-2.md) is to the filesystem,
[SPEC-5](./SPEC-5.md) is to the network, and [SPEC-6](./SPEC-6.md) is to the host.
SPEC-2 proved the *method* (propose-verify-correct against a free deterministic oracle)
on one filesystem; SPEC-5 lifted it to a multi-host network; SPEC-6 composed FS + net +
process into one running machine. SPEC-7 is the layer *above the host*: many machines
running **replicated services** that talk over the SPEC-5 network and execute on SPEC-6
hosts — databases, key-value stores, message queues, consensus groups, and the APIs in
front of them. It is the layer a computer-use or cyber-defense agent actually reasons
about ("will this config push break the cluster?"), and it is the first world where the
program's central asset — *a free, bit-exact, full-state oracle* — **stops being free**,
which is exactly why it is the world that proves the thesis matters.

It does not restate the science. Read [SPEC.md](./SPEC.md) first for *why* oracle-grounded
world models of computer environments are the one domain where long-horizon faithfulness
is measurable. This document is *how* — for a distributed system.

> **Reading order.** Prereqs: [SPEC.md](./SPEC.md) (science), [SPEC-2.md](./SPEC-2.md)
> (filesystem v0), [SPEC-5.md](./SPEC-5.md) (network world), [SPEC-6.md](./SPEC-6.md)
> (host world). Companions: [SPEC-3.md](./SPEC-3.md) (the depth roadmap whose walls
> **W1/W3/W4** and the speculative-execution framing this spec extends) and
> [SPEC-4.md](./SPEC-4.md) (the autonomous research engine that builds this with the
> human at the boundary).

> **▶ ACTIVE — DS0–DS6 + the equal-dollar-budget ED2 shipped (2026-06).** The deterministic core
> is up and the prime directive is measured: the **replicated KV under partition** slice
> ([`dist/`](../../src/verisim/dist/), [`distoracle/`](../../src/verisim/distoracle/),
> [`docs/distributed-semantics.md`](../distributed-semantics.md)) ships dependency-free and GPU-free
> with the `apply == oracle` invariant and golden trajectories (DS0–DS3); the **tiered
> propose-verify-correct loop** (DS5, [`distloop/`](../../src/verisim/distloop/)) and the flat
> learned `M_θ` (DS4) drop into it; and the distributed **`H_ε(ρ)` curve + the tiered-oracle H17
> measurement** is plotted on both the synthetic (ED1) and the real-model (ED1-learned) error
> distributions (DS6), with the **equal-*dollar*-budget H17/H18 frontier** (ED2) and the
> **fault-injection H21 data-factory result** (ED4) now in (DS7), and the **DS8 ED5 consistency-vs-bit
> horizon (H19) + competitive-ratio fit (H18)** and **ED6 counterfactual lift (H5)** in. The distributed knee + tiered
> oracle (**H17**) — the gate SPEC-7 was built around — reads *mode-dependent*: cheap tiers win per
> dollar exactly when the model's errors are cheaply catchable. ED5 sharpens the same throughline:
> consistency-faithful horizon *outlasts* bit-faithful where the error hides in the
> consistency-invisible in-flight medium (H19), and the loop is learning-augmented in the error axis
> but floor→cliff in the budget axis (H18). The **`linearizable` consistency model + the ED4
> consistency-level sweep (H20)** then show that the H19 gap is *exclusively* a weak-consistency
> phenomenon — it tracks the in-flight medium, present under `eventual` and structurally absent under
> `linearizable`. **ED6 then finds the distributed world is where counterfactual replay finally pays
> (H5):** off-policy oracle fault-branch training beats equal-volume on-policy data on held-out
> interventions (intervention-exact 0.51 vs 0.25), the honest inverse of the network/host supervision
> null — because the medium is the one hidden state on-policy volume cannot reach. **ED6's two-oracle
> slice (H12)** then closes the loop: the cheap consistency oracle is *redundant* for verification
> (non-redundant rate 0) but **decision-sufficient** for the split-brain question (1.00 on in-flight
> errors, 0.00 on durable — the in-flight medium again) at ~3.6× lower consult cost. Canonical build
> order: [SPEC §12](./SPEC.md#12-research-roadmap).

**License:** MIT. **Status:** DS0–DS6 shipped; DS7 in progress (ED2 + ED3 + ED4 H21/H20 in); DS8 in progress (ED5 H18/H19 + ED6 H5 + ED6 two-oracle H12 + the learned-`M_θ` H12 re-pointing + the §16 verified-contribution protocol + the §7 LLM-callable simulator + the `verifiers`-spec RL env + the Inspect benchmark + the DS8 technical report in; only Tier-B remains). **Audience:** the author,
collaborators, reviewers, and the autonomous research engine (SPEC-4), which reads this
file as part of its operating context.

---

## 0. Prime directive

> **SPEC-7's only job is to plot the distributed `H_ε(ρ)` curve once, cleanly, in a world
> where the full-state oracle is *intractable* — and to show whether a *tiered* oracle
> (cheap consistency checks plus rare bit-exact replay) buys more faithful horizon per
> oracle-dollar than spending the same budget on full-state truth alone (H17). It reports
> honestly if it does not.**

> **▶ First result (ED1 apparatus, DS6 — [`ed1_dist.png`](../../figures/ed1_dist.png)).** The
> distributed `H_ε(ρ)` curve is the same **floor→cliff** as every prior world (floor 0.2 at ρ=0 →
> ceiling 40 at ρ=1). And the H17 verdict, on a *controlled* error distribution (a synthetic
> tunable-noise proposer, the apparatus before the learned `M_θ`): **whether a cheap tier wins is
> not unconditional — it depends on where the model's errors fall.** For **gross** errors the cheap
> metamorphic tier buys faithful horizon at **$9.4/step vs bit-exact's $16**; for **subtle** errors
> the cheap tiers miss the drift (H≈0, **$848/step**) and full bit-exact truth is the only efficient
> verifier. So the tiered oracle is a *real lever, conditionally* — a sharper, more honest answer
> than "cheap always wins," and exactly the measurement the oracle-free distributed-systems field
> cannot make. The learned model (DS4) will supply the real error distribution this apparatus awaits.

> **▶ ED2 — the equal-*dollar*-budget form of H17 (DS7 — [`ed2.png`](../../figures/ed2.png)).** ED1
> measured cost-per-faithful-step at full consultation; ED2 asks the budget-form question the
> hypothesis is really about: *at an equal oracle-dollar budget, does a cheap or cheapest-refutation
> (`escalate`) tier policy buy more faithful horizon than spending the same dollars on bit-exact
> truth?* It sweeps ρ and plots, per tier policy, the **faithful-horizon-vs-oracle-dollar frontier**
> (the Pareto front), comparing policies at a matched budget by interpolating each one's horizon
> along its envelope — a true equal-dollar comparison, not an equal-ρ one. At the **sub-linear
> quarter budget** `B/4` (¼ of always-bit-exact's full-truth cost): for **gross** errors the
> metamorphic tier reaches **H=14.2 vs bit-exact's 4.2** (tiering wins per dollar — H17 holds, with
> a competitive ratio of **0.36** of the full-truth ceiling at ¼ the cost, the H18 readout); for
> **subtle** errors the cheap tiers are flat at the floor (H=1.5) and only bit-exact climbs (4.2),
> so `escalate` *loses* to single-tier bit-exact (H17's honest negative, ratio 0.11). The
> equal-budget frontier confirms ED1's mode-dependent verdict in the form the spec poses it.

Everything below serves that one figure. SPEC-2's prime directive was the same sentence
with "filesystem"; SPEC-5's with "network"; SPEC-6's with "composed host." SPEC-7's
difference is not a bigger world for its own sake — it is the **first world where you
*cannot* afford to verify every step even if you wanted to**, because (i) there is no
consistent global state to read in one shot (no global clock — HW-5), and (ii) checking
that an observed history is even *consistent* is NP-complete for serializability and
snapshot isolation (CAV 2025, §2.2). A budgeted oracle is no longer a convenience for
saving compute (SPEC.md §1); it is a *necessity*, and the cheapest sufficient verifier
must be *chosen*, not assumed. That choice — the **tiered oracle** — is SPEC-7's payload.

This spec is large because it is exhaustive by request, but the program is staged
(DS0–DS8, §13) so the deterministic core ships and is fully tested **with no runtime
dependencies and no GPU** before any learned model — exactly as M0–M3 did in v0, NW0–NW3
for the network, and HC0–HC3 for the host.

---

## 1. Why the distributed world, and why now

### 1.1 What SPEC-2, SPEC-5, and SPEC-6 left above them

The prior worlds model, respectively, a filesystem tree, a network of hosts, and a single
running machine. None models the thing those machines *cooperate to be*: a **service with
state replicated across machines**, kept consistent (or deliberately not) by a protocol.
A bank balance, a lock, a queue offset, a Raft log, a row under snapshot isolation — these
live *across* hosts, and their correctness is a property of the *ensemble*, not of any one
node SPEC-6 can snapshot. Three properties of the distributed world are absent from every
prior world and are exactly the properties that make the science bite next:

| Property | v0 / net / host | Distributed world (SPEC-7) |
|---|---|---|
| **Global state** | one consistent snapshot exists and is cheap to read | **none** — there is no instant at which all replicas agree; a "global state" is a *coordinated* (expensive) or *inconsistent* (cheap) construction |
| **Oracle cost** | bit-exact truth is free (`ρ=1` is the cheap ceiling) | **bit-exact truth is intractable** — checking serializability/SI of a history is NP-complete (CAV 2025); a consistent snapshot needs a protocol round |
| **Failure as first-class** | failure is an exit code | **partition, crash, message loss, reorder, clock skew** are the *medium* — the interesting dynamics only exist under fault |

These are not incremental. The disappearance of a free full-state oracle (W7) inverts the
economics SPEC.md §2 rests on, and it is *good news for the thesis*: the regime where you
*must* spend a scarce, expensive verifier wisely is precisely the regime the whole program
claims to be the metrology lab for (SPEC.md §4). The single-FS world was too easy to show
H1's knee (SPEC-2 §13, the honest negative); the distributed world is the first where
**not consulting is the only affordable default**, so the value of cheap-but-faithful
prediction is maximal.

### 1.2 The oracle survives — but it *tiers*

The reason to go here next rather than to vision/biology/robotics is unchanged (SPEC.md
§2): the oracle stays **constructible, deterministic, and free of real-world cost** —
*because the entire distributed-systems engineering field has spent a decade building
exactly the oracle we need.* The **deterministic-simulation-testing (DST)** tradition runs
the *real* distributed code inside a single-threaded discrete-event simulator with all
nondeterminism (network, disk, clock, RNG, thread schedule) quarantined behind seeded
shims, so a whole cluster's execution is a pure function of `(seed, commit)` and replays
bit-for-bit (FoundationDB's framework, Will Wilson, Strange Loop 2014; TigerBeetle's VOPR;
Antithesis's deterministic hypervisor, 2024; madsim/turmoil; Shadow, which executes real
unmodified binaries inside a deterministic network DES). What the field built as a *testing*
tool is, to us, a **free, fault-rich, perfectly-reproducible ground-truth factory**.

But "the oracle survives" now means a *spectrum* of oracles at different prices, because
bit-exact global truth is intractable (§1.1). SPEC-7's central design move is to make the
oracle **tiered** (§5):

```
oracle tier          what it verifies                         cost      analogue
-------------------  ---------------------------------------  --------  ----------------------------
bit-exact replay     the full next state, bit-for-bit         high      v0 ReferenceOracle, run on a
  (DES / Tier-B)       (a coordinated/simulated global snap)             whole cluster under DST
symbolic / formal    the protocol's next-state relation       medium    Batfish for routing (SPEC-5),
  (TLA+ / isolation)   (Raft step legal? SI satisfied?)                  now TLA+/isolation semantics
consistency-cycle    is the observed *history* admissible      low       Jepsen/Elle: cycle detection
  (Elle-style)         under the declared model (SER/SI/lin)?            over a partial observation
metamorphic / prop   does an equivalence/invariant hold        very low  SQLancer PQS/NoREC/TLP
  (SQLancer-style)     (rewrite-equivalence, no reference)?              metamorphic DB oracles
```

The cheap tiers do not return the truth; they return a *refutation or a pass*. That is
enough to gate a prediction, and it is enough to bound drift — and it is far cheaper than
reconstructing a consistent global state. **Choosing the cheapest oracle that can still
catch the model's current error is the new core of the consultation policy** (§8.2, the
`π_w` axis SPEC-6 introduced as "which sub-oracle," now generalized to "which *tier*").

### 1.3 The foils that frame the thesis: CWM and WebDreamer

Two 2024–2025 results are the cleanest external reference points, and neither is yet in the
repo's related-work set:

- **CWM — Code World Model (Meta FAIR, 2025; 32B open weights).** Mid-trained on ~120M
  Python execution traces to predict per-line local-variable state — a "neural debugger" —
  and reports strong SWE-bench-Verified and loop-termination-prediction numbers. CWM is the
  large-scale instantiation of *exactly* verisim's "predict the next program state given an
  action" thesis, at the code/host layer. **It is the foil that proves the gap:** CWM
  predicts state *un-verified* — there is no oracle in its loop, no budgeted correction, no
  faithful-horizon metric. Verisim is CWM *plus the deterministic oracle*: the same
  prediction target, but with a tiered verifier in the loop and `H_ε(ρ)` as the measured
  result CWM cannot produce. Position relative to CWM, do not compete with it on scale.
- **WebDreamer — "Is Your LLM Secretly a World Model of the Internet?" (Gu et al., 2024;
  NAACL 2025).** Uses an LLM as an *inference-time* world model to simulate the outcomes of
  web actions before committing, matching tree search at ~4–5× fewer real environment
  interactions. This is the propose-then-verify loop validated on the real web — and its
  world model is an *un-grounded* LLM. **It is the baseline to beat:** verisim's distributed
  world model is oracle-grounded, so it should sustain a longer faithful horizon per real
  interaction than an un-grounded LLM simulator, on a checkable distributed state.

> The contrast is the whole program in one line: CWM and WebDreamer show that *predicting
> system state is useful and learnable*; verisim adds the one thing the domain uniquely
> permits and they both lack — **a deterministic oracle in the loop, spent on a budget.**

And note what the foils share: CWM is a 32B transformer, WebDreamer is a prompted LLM — *different
model classes, same missing piece*. That is the point. The tiered oracle (§5) grounds **whatever**
sits in the proposer slot — a CWM-style transducer, the service-graph GNN of §6, an RSSM, or an LLM
caller — because the loop verifies the *prediction*, not the *predictor*. So the distributed
`H_ε(ρ)` curve and the tiered-oracle result (H17) are, like every prior world's, claims about the
oracle-loop rather than any one architecture: the distributed-layer instance of **deterministic
verification as a model-agnostic primitive** (SPEC.md §6 commitment 4, H22).

### 1.4 Why it is worth building for the community, not just for us

A deterministic, oracle-grounded, partially-observable model of a *distributed system* is a
benchmark the field does not have: a place to measure long-horizon faithfulness of a world
model over replicated state under fault, with ground truth, where the ground truth is
*tiered* because full truth is intractable. The DST tools (FoundationDB, TigerBeetle,
Antithesis, Jepsen) are evaluated on *bug-finding*, never on *faithful horizon under a
verification budget*; the learned-DB-component line (learned cost models, learned indexes)
optimizes performance, never predicts full consistency dynamics; the agent benchmarks
(τ-bench, AppWorld) grade *task success* against a stateful backend, but have no model of,
and no metric for, the backend's predicted next state. Verisim's contribution is the
missing metrology and the cheap faithful simulator the agents could call (§7), packaged
where researchers already look (an Inspect benchmark + a `verifiers`-spec RL env, SPEC-2
§15). This is the non-competitive contribution: not a bigger agent, but a free, honest
measuring instrument for distributed-state faithfulness and the verified simulator beneath
it.

---

## 2. Lessons folded in (and the design choice each one forces)

The "browse far and wide" deliverable, distributed-world edition. Each lesson is stated as
design guidance with its source; skeptical notes flag where the literature is hype.
Citations are consolidated in [`docs/related-work.md`](./docs/related-work.md), kept current
by the engine, **name + venue + year per that file's no-fabricated-links policy** (arXiv
IDs only where independently verified — several plausibly-recent IDs surfaced in research
are deliberately *omitted* here until verified).

### 2.1 Deterministic simulation testing — the oracle *and* the data factory

- **FoundationDB simulation (Will Wilson, Strange Loop 2014; SE-Radio 685, 2025); TigerBeetle
  VOPR ("Simulation Testing For Liveness," 2023); Antithesis "deterministic hypervisor"
  (2024); madsim / turmoil; Shadow (USENIX-pedigree, executes real binaries in a network
  DES).** Run real code in a single-threaded DES; quarantine *all* nondeterminism behind
  seeded shims; inject faults deterministically (`BUGGIFY`); a run is a pure function of
  `(seed, commit)`. TigerBeetle's time-dilation (seconds of sim ≈ tens of minutes of
  wall-clock) is the quantitative argument for a *faster-than-sim* learned surrogate. →
  **Design choices:** the Tier-A oracle (§5.1) is a from-scratch deterministic DES of a
  pinned distributed semantics, and the same DES is the **training-data factory** — seeded
  fault injection produces unlimited, perfectly-labeled, fault-rich trajectories
  (`DD-D3`); `(seed, commit)` is the reproducibility key v0 already uses (SPEC-2 §12);
  Tier-B (§5.2) can *wrap* madsim/Shadow/Antithesis-class runtimes rather than rebuild them.
- *Skeptical note:* DST's defensible core is rock-solid (these systems ship and find real
  bugs), but it explores a *seeded random walk* through fault space — robustness against
  *sampled* faults, not exhaustiveness. A model trained on DST traces inherits the
  simulator's coverage blind spots; coverage is a curriculum axis (§3.4), not a given.

### 2.2 The full oracle is intractable — and that is the point

- **"On the Complexity of Checking Mixed Isolation Levels for SQL Transactions" (CAV 2025):**
  consistency checking is polynomial for "saturable" isolation levels but **NP-complete for
  Snapshot Isolation and Serializability.** **Elle (Kingsbury & Alvaro, VLDB 2020):**
  black-box transactional-consistency checking via *cycle detection* on observed histories,
  scaling to hundreds of thousands of operations where linearizability checkers (Knossos)
  are exponential. → **Design choices:** define faithfulness against a *declared consistency
  model*, not a (nonexistent, intractable) single global state (`DD-D2`); use Elle-style
  cycle detection as the **cheap consistency-tier oracle** (§5.3) that gates predictions
  without reconstructing truth; and make the NP-hardness the *motivation* for the budgeted
  oracle, not an obstacle to it — this is the strongest H1 motivation the program has had.
- **Formal semantics as symbolic oracles: VerIso (PVLDB 2025) and weak-isolation separation
  logic (ICFP 2025)** for transactions; **Apalache (OOPSLA 2019, symbolic TLA+); Stateright;
  the P language** for consensus/protocol next-state relations. → **Design choice:** the
  *symbolic/formal tier* (§5.1) verifies "is this protocol step legal / does this history
  satisfy SI?" directly from a spec, the distributed analogue of SPEC-5's Batfish
  control-plane oracle; Stateright's *shared-code* model (the checker and the implementation
  are the same code) is the cleanest fit for a from-scratch reference.

### 2.3 Metamorphic / property oracles — verify without a reference

- **SQLancer metamorphic oracles — PQS, NoREC, TLP (Rigger & Su, OSDI/ESEC-FSE 2020);
  graph-based transactional-bug oracle (Jiang et al., OSDI 2023).** Verify behavior via
  *equivalence/invariant* relations (a query and its rewrite must agree) with **no reference
  state at all**. → **Design choice:** the cheapest oracle tier (§5.3) is metamorphic —
  invariants the predicted state must satisfy (a transferred balance conserves total; a
  dequeue cannot return an un-enqueued item; a committed read-your-writes must hold). It
  costs almost nothing and catches a large class of model hallucinations before any
  expensive tier is spent.
- *Skeptical note:* metamorphic oracles are *incomplete* — they catch violations of the
  specific relation, not arbitrary wrongness. They are a fast first filter, never the whole
  gate; the bits-to-correct headline (§9) is still anchored on the bit-exact tier where it is
  affordable.

### 2.4 Learning-augmented algorithms — the loop's missing theory

- **Algorithms with predictions / learning-augmented algorithms (the 2018–2025 line:
  learned indexes with error-bounded fallback; learning-augmented caching such as LARU; MAT
  "ML at the tail").** A prediction guides the common case; a worst-case-safe deterministic
  fallback bounds the damage when the prediction is wrong; the figure of merit is a
  **competitive ratio** that degrades gracefully with prediction error (and recovers the
  classical worst-case bound when the predictor is useless). → **Design choice:** this is
  *exactly* propose-verify-correct (`DD-D4`): the model is the predictor, the oracle is the
  worst-case-safe fallback, `ρ` is how often the fallback is consulted. SPEC-7 adopts the
  competitive-ratio *framing* as the theoretical backbone the loop has lacked, and reports
  faithful horizon as a competitive ratio against the full-oracle ceiling (H18) — giving the
  central mechanism a guarantee, not just a curve.
- *Skeptical note:* the clean competitive-ratio theorems assume a fallback that restores
  *exact* correctness each time it fires; our cheap tiers do not (they refute, they do not
  reconstruct). So the guarantee is *empirical* (a measured ratio) at the cheap tiers and
  *provable* only at the bit-exact tier. Stating which is which honestly is mandatory.

### 2.5 In-house patterns: worldify (state/belief) and securifine (change-safety eval)

- **`worldify` (sibling repo) — a deterministic, zero-dependency temporal fact store.** It
  models state as immutable, time-scoped *facts* with **supersession** (a change creates a
  new fact and marks the old one `superseded_by`, never deletes — full history is
  queryable), **confidence scores** (epistemic certainty per fact), **causal chains**
  (`caused`/`enabled`/`prevented` edges with `trace_causes`/`trace_effects`), and
  **snapshot / branch / restore** with versioned JSON export. → **Design choice (`DD-D5`):**
  the distributed world's state representation *is* this pattern — an **event-sourced causal
  log**, because a distributed system's ground truth is naturally a log of events with a
  *happens-before* (causal) partial order, replica-local **belief** (confidence) over the
  unobserved part, and **consistent snapshots** as the coordinated reads. worldify's
  supersession = MVCC versions; its causal edges = the happens-before relation that *defines*
  causal consistency; its confidence = the belief a partial-observation oracle (§5.4) leaves
  the model holding; its snapshot/branch = consistent global snapshots and the
  counterfactual branching the oracle makes free (§9, H5). The repo reuses worldify's data
  model rather than reinventing it.
- **`securifine` (sibling repo) — measurement-first differential safety eval.** A
  baseline → post-change → **differential comparison** pipeline with **severity-weighted**
  scoring, deterministic pattern-matching verifiers (not LLM-judged), config layering
  (CLI > env > file > defaults), SHA-256 artifact versioning, an abstract `ModelInterface`
  (HTTP + offline-cache), and an explicit *limitations* doc. → **Design choices:** the
  distributed world's **change-safety** evaluation (§7, §12) *is* securifine's differential
  pattern — score the *delta* in consistency-faithfulness between "before config push" and
  "after," **severity-weighted by consistency-violation class** (`DD-D8`); the
  offline-cache `ModelInterface` is the pattern for the LLM-callable simulator (§7); and the
  limitations-doc discipline carries into `docs/distributed-semantics.md`.

### 2.6 World-model-as-tool & executable distributed environments

- **WebDreamer (Gu et al., 2024; §1.3); τ-bench / τ²-bench (Sierra, 2024–2025);
  AppWorld (ACL 2024); ToolEmu (ICLR 2024).** τ-bench drives tool-agent-*user* dialogues
  against a *stateful database backend*; AppWorld has agents act via code+API over nine
  stateful apps; ToolEmu has an LM *emulate* tool execution (the un-grounded version of what
  we ground). → **Design choices:** the downstream framing (§7, §15) is the agent calling a
  **cheap faithful simulator of the service it is about to change**, and τ-bench/AppWorld are
  the *external legibility harness* (§12) the way CAGE-4 was for SPEC-5 and OSWorld for
  SPEC-6; ToolEmu's "LM-emulated tools" is the baseline verisim beats by grounding the
  emulator in an oracle.
- **SWE-Gym (2024) / R2E-Gym (2025) and the execution-based-vs-execution-free reward
  finding.** The SWE-agent field converged on *execution-based* verifiers (run the tests,
  expensive, exact) and *execution-free* reward models (cheap, approximate) being
  **complementary**. → **Design choice:** this is independent empirical support for the
  tiered-oracle design — you want both the expensive-exact and the cheap-approximate
  verifier, and the science is *how to mix them on a budget* (H17).

### 2.7 Model architecture for distributed state

- **m4 / RouteNet bipartite GNNs (SPEC-5 §2.2), DreamerV3 RSSM (SPEC-5 §2.4), and
  state-space models (Mamba/SSD, Gu & Dao, 2023–2024).** Mamba carries a *fixed-size explicit
  recurrent state* updated in linear time — a natural fit for "carry and update a compressed
  system state," and a complement to the delta-prediction target (v0's E4 finding that delta
  dominates full-state). **CLRS / TransNAR (neural algorithmic reasoning)** show GNNs that
  execute classical algorithms generalize OOD better with *step-wise intermediate-state
  supervision*, and **Jin & Rinard (ICML 2024)** show program semantics emerge in a model
  trained only on next-token prediction of code. → **Design choices:** `M_θ` is a
  message-passing predictor over the **service graph** (services ↔ shared resources, the m4
  template, §6.1) with an RSSM belief over unobserved replicas and an *optional SSM
  recurrent carry* as an EH-style architecture lever; supervise on *intermediate*
  consistency-relevant facts (commit order, lock holder, log index), not only the final
  reachable state.
- *Skeptical note:* neural algorithmic reasoning generalizes *narrowly* (single algorithms,
  modest sizes); CLRS success does not promise scalable distributed-protocol modeling.
  Architecture is an ablation axis (ED4), not a faith.

### 2.8 The SLM thesis, restated for this layer

- **"Small Language Models are the Future of Agentic AI" (NVIDIA, Belcak et al., 2025).**
  Narrow, repeated, verifiable tasks favor a small specialist that is escalated *from*, not a
  generalist. → **Design choice:** `M_θ` is a small, distributed-system-specialized
  simulator — the cheap verifiable specialist an LLM agent calls (§7), not a competitor to
  the LLM. The oracle is an *infinite, perfect distillation teacher* (unlimited labeled
  trajectories via §2.1), the unfair advantage of this domain.

---

## 3. The world (environment)

### 3.1 State as an event-sourced, replicated, partially-observable log

The distributed state `s` is **not** a single tree (SPEC-2), graph (SPEC-5), or bundle
(SPEC-6). It is a set of **replicas**, each holding a local copy (or shard) of logical
objects, plus an **event log** with a happens-before partial order, plus in-flight
messages — i.e. the worldify temporal-causal-fact model (§2.5, `DD-D5`) instantiated for a
cluster.

```
DistributedState = {
  nodes:    map[node_id -> NodeState]       # each NodeState ⊇ a SPEC-6 host (it *runs* on one)
  replicas: map[(object_id, node_id) -> ReplicaState]   # per-node copy/shard of a logical object
  log:      [ Event ]                        # the causal event log (worldify facts + happens-before)
  inflight: map[msg_id -> Message]           # messages sent, not yet delivered
  protocol: ProtocolState                    # consensus/leader/term/view, lock table, txn table
  global:   { sim_clock, rng_cursor, partition_set, last_result }
}
ReplicaState = { object_id, node_id, value_version (MVCC), confidence, last_applied_index }
Event        = { id, node, op, happens_before:[event_id], superseded_by?, confidence }
Message       = { id, src, dst, payload_kind, send_time, deliver_after }
```

- **There is no `global state` field.** A consistent global snapshot is *derived* by a
  coordinated read (an oracle call, §5), never stored — this is W7 made structural.
- **The happens-before edges are the spine.** Causal consistency, MVCC visibility, and the
  divergence metric (§9) are all defined over them; this is worldify's causal-chain model
  (`trace_causes`) doing real work.
- **Canonicalization** (sorted, hashed, volatile-IDs normalized) is mandatory exactly as in
  every prior world (SPEC-3 `DD-1`): node/replica/msg IDs are canonicalized so divergence
  measures *competence*, not identifier churn.
- **The prior worlds embed verbatim.** A `NodeState` *is* a SPEC-6 `HostState`; the links
  between nodes *are* SPEC-5's network; SPEC-7 owns only the replication/protocol/log glue.
  SPEC-7 is the integrating spec, not a fourth parallel one (mirroring how SPEC-6 embedded
  SPEC-2).

### 3.2 Actions

Three families, in a constrained grammar paired with the oracle, as every prior world pairs
a grammar with a reference oracle:

1. **Client ops (the workload):** `put/get/cas(key, val)`, `begin/commit/abort` (a
   transaction over keys), `enqueue/dequeue(queue)`, `read/write` at a declared isolation
   level, an API call against a service.
2. **Protocol / admin ops:** `propose(value)` to a consensus group, `add/remove replica`,
   `leader step-down`, `config push`, `deploy version`, `scale up/down`.
3. **Fault / time ops (the medium):** `partition(set_a | set_b)`, `heal`, `crash(node)`,
   `restart(node)`, `drop/delay/reorder(msg)`, `clock_skew(node, δ)`, and `advance Δt`
   (deliver in-flight messages, fire timers: election timeout, lease expiry, retransmit).

The fault/time family is the source of all interesting dynamics and is the distributed
analogue of SPEC-5's `advance Δt` and SPEC-6's scheduler — `partition`/`crash`/`reorder`
under a seed *is* the `BUGGIFY` of DST (§2.1).

### 3.3 Determinism contract

The world is deterministic given `(initial cluster, seed)`: the seed fixes message
delivery order, fault injection, and per-node RNG/clock — the DST contract (§2.1). Tier-A is
a pure function. Tier-B (a wrapped madsim/Shadow/Antithesis-class runtime) pins
commit + image + kernel and seeds the scheduler; **asynchrony/partition is the
nondeterminism source that cannot be sealed cheaply (HW-5)** — it is made a *seeded,
controllable input* (the fault schedule), and `determinism_report` (SPEC-3 §2.2) declares,
per figure, the consistency model assumed and whether ordering was simulated or recorded.

### 3.4 Scale & curriculum

Difficulty is a small set of dials (mirroring SPEC-2 §2.4 / SPEC-5 §3.4 / SPEC-6 §3.4):
number of nodes, replication factor, transaction concurrency, declared consistency model
(linearizable → serializable → snapshot → causal → eventual — *weaker is harder to predict*
because more histories are legal), workload contention (hot keys), and — the new ones —
**fault intensity** (the `BUGGIFY` rate) and **partition entropy** (how often/asymmetrically
the network splits). The curriculum starts at **one node, no replication, no faults** (the
distributed analogue of v0's smallest world) and ratchets up to **a partitioned, faulting,
weakly-consistent multi-replica cluster under contention**. Fault intensity and partition
entropy are explicit axes because they are exactly what H20/H21 measure.

---

## 4. The state delta `Δ`

`M_θ` predicts a structured **log/replica delta**, not a full global state and not raw wire
bytes (§2.7, and v0's E4 result). The delta vocabulary composes the prior worlds' edit types
with the new replication/protocol/log types:

```
EventAppend(node, op, happens_before)        # a new causal-log event
ReplicaWrite(object, node, new_version)      # MVCC version bump on one replica
ReplicaConverge(object, nodes, version)      # replicas agree on a version (anti-entropy/commit)
MsgSend(id, src, dst, kind) / MsgDeliver(id) / MsgDrop(id)   # message lifecycle
ProtocolStep(kind, term, index, leader?)     # consensus/leader/lease/view change
TxnBegin(id) / TxnCommit(id, order) / TxnAbort(id)           # transaction lifecycle
LockAcquire(obj, holder) / LockRelease(obj, holder)
HostDelta(...)  NetDelta(...)                 # the SPEC-6 / SPEC-5 deltas, on embedded subsystems
SetResult(status, value_token)               # client-visible result (as in v0)
```

The **M1-analogue invariant** is required and tested: `apply(state, oracle.delta) ==
oracle.next_state` for every transition, by construction, and `apply` is *compositional* —
embedded host/net deltas reuse SPEC-6/SPEC-5's `apply` verbatim, and the new types apply to
the log/replica/protocol layer. Delta↔serialization round-trips. This invariant keeps the
loop (§8) model-agnostic, as in every prior world.

---

## 5. The oracle `O` — the tiered oracle (the heart of SPEC-7)

The structural novelty (§1.2): the oracle is a *menu* at four price points, and the
consultation policy chooses *which tier* to spend, not just *when*.

> **Design decision (`DD-D1`): consult the cheapest tier that can refute the current
> prediction.** A prediction that violates a metamorphic invariant is rejected by the
> very-low-cost tier; one that violates the declared consistency model is caught by the
> low-cost cycle-detection tier; only predictions that pass both and still need
> ground-truth correction spend the high-cost bit-exact tier. *Rationale:* full-state truth
> is intractable (§2.2), so spending it indiscriminately is the one thing the distributed
> world forbids. *Alternative considered:* always bit-exact (the v0/net/host default) —
> rejected as infeasible here, which is exactly what makes this world the thesis's proving
> ground.

### 5.1 Tier-A — reference distributed oracle (the deterministic core)

A **from-scratch deterministic DES of a pinned distributed semantics**: a replicated
key-value store with MVCC and a declared isolation level, a simplified but real **consensus
group** (leader election, log replication, commit — a Raft-subset), a lock/transaction
table, a message layer with seeded delivery/partition/loss, and the embedded SPEC-6 hosts +
SPEC-5 network. It has **no runtime dependencies and needs no GPU**, like every prior core.
It is the executable truth, paired with a normative `docs/distributed-semantics.md`. Golden
trajectories pin the semantics and are denylisted from the engine (§14).

Beside the DES runs the **symbolic/formal tier**: a checker that, given the protocol spec
(a TLA+/Stateright/P-style next-state relation) and the declared isolation semantics
(VerIso/ICFP-style), answers *"is this transition legal / does this history satisfy the
model?"* directly — the distributed analogue of SPEC-5's Batfish control-plane oracle, and
deterministic and free. Whether the symbolic tier is a *non-redundant* signal over the DES
is hypothesis **H17**-adjacent and tested in ED6.

### 5.2 Tier-B — system oracle (reality check)

A **wrapped real DST runtime** — madsim/turmoil (Rust, tokio-compatible), Shadow (real
binaries in a deterministic network DES), or an Antithesis-class deterministic hypervisor —
running a real (small) distributed system (e.g., a real embedded KV store / a real
Raft library) under seeded scheduling, pinned commit + image, **no real-internet egress**
(§15). Tier-B attacks SPEC-3 wall **W1** ("the oracle is a model, not reality") for the
distributed domain; the Tier-A↔Tier-B gap is itself a reportable result and a curriculum
signal (where our pinned semantics diverges from a real system is where data should
concentrate). DST is what makes Tier-B *deterministic at all* — without it, a real cluster
is not replayable and cannot be a bit-exact oracle.

### 5.3 The cheap tiers — consistency-cycle and metamorphic

- **Consistency-cycle (low cost):** Elle-style cycle detection (VLDB 2020) over a *partial*
  observation of the history answers "is what we have seen so far admissible under the
  declared model?" in near-linear time, *without reconstructing global state* — the
  workhorse cheap oracle.
- **Metamorphic / invariant (very low cost):** SQLancer-style relations and domain
  invariants (conservation, monotonic queue offsets, read-your-writes, no-lost-update) the
  predicted state must satisfy — a first filter that costs almost nothing and catches gross
  hallucination before any expensive tier is touched (§2.3).

### 5.4 Partial observation & bits-to-correct

The oracle inherits SPEC-5/SPEC-6's **probe (cheap, localized) vs full (expensive)** modes,
and adds the §5.1–5.3 *tier* choice on top. **Bits-to-correct** (SPEC-3 §7, SPEC-4 §5)
generalizes directly and **decomposes by tier and by object**: `bits-to-correct = Σ_object
MDL(correction)`, 0 iff the prediction equals truth on the tier consulted, smooth and
unfakeable otherwise. The per-tier breakdown tells the engine (and H17) *which tier* is
buying the most faithfulness per dollar — the lever a single scalar hides.

> **Design decision (`DD-D2`): faithfulness is defined against a *declared consistency
> model*, not a single global state.** `ε` is a *consistency-level* threshold (does the
> predicted history still admit serializability / SI / causal / linearizability?), not only
> a bit distance. *Rationale:* there is no consistent global state to be bit-faithful to
> (W7); the operationally meaningful question is whether the model predicts the
> *observable consistency behavior*, which is what a defender or SRE actually relies on.
> Bit-exact divergence remains the headline *where the bit-exact tier is affordable*;
> consistency-faithfulness is the metric that survives where it is not.

---

## 6. The model `M_θ`

### 6.1 Architecture
A **message-passing predictor over the service graph** (services/replicas ↔ shared
resources: locks, logs, queues — the m4 bipartite template, SPEC-5 §6.1), action- and
clock-conditioned, emitting a structured delta under grammar-constrained decoding (the
constrained-decode machinery from M4/NW4/HC4 carries over). Message-passing depth is tuned
to the cluster diameter (a leader change `k` hops away cannot be represented with too few
steps). Heterogeneous magnitudes (log indices vs byte counters vs flags) pass through a
**symlog transform** (DreamerV3, reused). An **SSM/Mamba recurrent carry** (§2.7) is an
optional architecture arm for the long causal log.

### 6.2 Latent belief (RSSM) over the unobserved cluster
A recurrent latent carries a **belief over the replicas and in-flight messages the model
cannot see** (§2.5, worldify confidence). Under full observation it degenerates to a Markov
predictor; under partition/partial-observation it is the only way to roll forward the
unseen subgraph, and its variance is the *calibrated-by-construction* uncertainty signal
that the consultation policy reads (the H2-negative fix continuing from SPEC-5 §6.2).

### 6.3 Drift mitigations (required ablation levers, inherited)
The SPEC-5/SPEC-6 levers carry over and are mandatory ED4 arms because the distributed
world's deep causality and fault dynamics make drift compound hard: **noise-injected
rollout training** (GNS — the cheapest, highest-leverage one; here the "noise" is naturally
*fault* injection, §2.1), **self-forcing / scheduled sampling**, and the **multi-step /
latent-overshoot objective**.

### 6.4 Size & specialization (SLM)
`M_θ` is small and distributed-system-specialized (§2.8). ED4's size axis re-asks v0's
capacity question (v0: *no*, size is not the floor) in the harder, faulting, weakly-
consistent world — where the answer may differ, and the honest measurement is the point.

---

## 7. SLM/LLM complementarity — the cluster simulator an agent calls

This answers the user's "complement/integrate SLM/LLM" intent at the layer where it matters
most: a computer-use or cyber-defense agent acts on a *running distributed system*
("push this config," "fail over this primary," "drain this node"), so the distributed world
is exactly the simulator such an agent needs. The world model is not a competitor to the
LLM; it is the cheap, faithful, verifiable cluster the LLM reasons *over* — the grounded
version of WebDreamer (§1.3).

- **World model as a "what-if" the agent calls.** An LLM agent proposes a plan (a sequence
  of admin ops); `M_θ` simulates the consequences across replicas fast; the tiered oracle
  verifies on a budget. This is propose-verify-correct lifted from the *op* level to the
  *plan* level — Dreamer's "plan in imagination," made honest by a real oracle, for the
  thing τ-bench/AppWorld agents actually do (§2.6). The baseline to beat is an *un-grounded*
  LLM simulator (WebDreamer, ToolEmu).
- **Change-safety as differential faithfulness (securifine pattern, §2.5).** "Will this
  config push break consistency/availability?" is scored as the *delta* in
  consistency-faithfulness between before and after, severity-weighted by violation class
  (`DD-D8`) — securifine's baseline→post→differential pipeline, with the oracle (not a
  pattern-matcher) as the verifier.
- **Distillation from an infinite, perfect teacher.** The DES emits unlimited correctly-
  labeled, fault-rich trajectories (§2.1) → distill into the SLM. Ground truth at scale is
  the domain's unfair advantage.
- **Speculative execution / routing.** `M_θ` drafts cluster dynamics; the oracle verifies
  only when the policy fires, at the cheapest sufficient tier (`ρ` × tier). The LLM is
  escalated to only for natural-language intent → plan translation, never for simulating
  dynamics it is bad at. The execution-based-vs-execution-free complementarity (SWE-Gym,
  §2.6) is independent evidence this mixture is the right shape.

The integration is a **protocol**, specified here and built in DS8: a `Model` that
implements both "predict next cluster state" (for the loop) and "simulate a plan" (for an
LLM caller), packaged like v0's `eval`/`rl` modules and exposed behind a securifine-style
abstract interface (offline-cache + live), so external agents can call the verified cluster
simulator. Open question §17.8 (defining `H_ε` for a *plan* unit) is inherited from SPEC-5
§17.8 / SPEC-6 §17.8.

> **◐ shipped** ([`distsim/`](../../src/verisim/distsim/),
> [`test_distsim.py`](../../tests/test_distsim.py)): the LLM-callable cluster simulator, the
> dependency-free distributed analogue of the host [`hostsim/`](../../src/verisim/hostsim/).
> `DistSimulator` both *predicts the next cluster state* (the loop interface) and *simulates a plan*:
> `imagine` rolls `M_θ` over a proposed admin-op plan with **no oracle** (Dreamer's cheap draft),
> and `verify` runs that imagination against the oracle step-by-step, returning a `DistPlanReport`.
> Beyond the host's bit-exact plan horizon, the report carries the two readouts the cluster world
> makes meaningful: (1) a **consistency-faithful plan horizon** distinct from the bit-exact one — how
> many leading plan steps the agent can trust the model's split-brain prediction, which (per
> ED5/H19) outlasts the bit-exact horizon when the error hides in the in-flight medium; and (2)
> **change-safety as differential consistency-faithfulness** (the securifine pattern, §2.5,
> `DD-D8`) — *"will this plan break consistency?"* scored as the change in the cluster's consistency
> health (fraction of objects converged) from before to after, with the **model-vs-oracle agreement**
> on the safe/unsafe verdict as the task-level faithfulness. A composing **task oracle** (`DistGoal`,
> the §7 "third oracle": "object `x` converged everywhere", "no split-brain", "node back up") gives
> goal-level agreement too. Dependency-free except the model — the loop's dependency-free baselines
> satisfy `DistModel`, so the protocol is exercised in CI with no torch. *(Deferred: the offline-cache
> live/abstract split and the plan-level RL reward, inherited from the host packaging.)*

---

## 8. The loop (tiered, partially-observed propose-verify-correct)

Same skeleton as SPEC.md §5.2, with the distributed extensions (tier choice, consistency-
correction, the fault-driven advance).

```
for each step t:
    Δ̂ ← M_θ(belief, a_t)                          # PROPOSE: predict the log/replica delta
    ŝ' ← apply(ŝ, Δ̂); belief ← update(belief, Δ̂)
    if π_c decides to consult (budget ρ):                  # WHEN to consult
        tier ← π_w(belief, Δ̂)                              # WHICH TIER/oracle (§8.2) — the new axis
        v ← O[tier](s, a_t)                                # metamorphic | cycle | symbolic | bit-exact
        d ← divergence(v, ŝ')                              # refutation, consistency, or full
        ŝ', belief ← C(ŝ', belief, v)                      # CORRECT / belief-update (§8.3)
        replay.add(s_t, a_t, v)                            # accumulate the experience stream (SPEC-6 §8.5)
        if healer: θ ← heal(θ, replay)                     # SELF-HEAL: gated TTT step (SPEC-6 §8.4)
    s ← O.true_next(s, a_t)                                # the cluster advances regardless (under the seed)
```

### 8.1 Consultation policy `π_c` (when)
The v0/net/host policies carry over (`fixed`, `drift_triggered`, `uncertainty_triggered`,
`learned`), now reading the RSSM belief variance over the unobserved cluster (§6.2) — the
calibrated-by-construction signal partial observability supplies.

### 8.2 Tier/oracle policy `π_w` (which) — the central new axis
Given a consultation, *which tier* to spend (§5): metamorphic (almost free, incomplete),
consistency-cycle (cheap, refutes consistency violations), symbolic (medium, protocol-step
legality), or bit-exact replay (expensive, full truth). The information-gain choice — spend
the cheapest tier that can refute the *current* most-uncertain, most-consequential predicted
edit — is the generalization of SPEC-5's `π_o` (what-to-observe) and SPEC-6's `π_w`
(which-sub-oracle) to *which-price-of-truth-to-buy*. It is the heart of H17 and ED2, and it
is the operational form of `DD-D1`.

### 8.3 Correction / belief operators `C`
`hard_reset` (snap the observed slice to truth), `residual`, `projection` (onto the nearest
*consistency-model-consistent* state — e.g. repair the history so it re-admits SI), and the
`belief-filter` (Bayesian/particle update of the unobserved replicas). Partial observation +
weak consistency make these genuinely different (no v0 identity collapse) — ED3.

### 8.4 Self-healing & the experience stream (optional, gated, inherited)
Each consultation is a free labeled example; take a gated gradient step (SPEC-3 §6.3
recipe — small-lr, replay, PEFT, trust-region revert under the keep-if-better referee).
The SPEC-6 §8.5 experience stream applies: a never-ending sandboxed cluster run from which
the model predicts, the tiered oracle verifies on a budget, and the model heals — with the
plasticity probe (SPEC-6 §9.4, HW-4) tracked. Off by default so the static baseline is
clean (`DD-5`).

---

## 9. Metrics

### 9.1 Consistency-faithfulness (the headline-new metric)
Over the horizon, does the model's predicted history admit the *declared consistency model*
when the true one does (and forbid it when the true one does)? Graded by Elle-style cycle
detection over the predicted vs true partial histories. This is the operationally meaningful
number (`DD-D2`) and the one that survives W7. Reported alongside bit-exact divergence
wherever the bit-exact tier is affordable.

### 9.2 Composed / consistency divergence `d(s, ŝ)`
Normalized symmetric difference over canonical replica/log/protocol tuples (the worldify
fact-set), **plus** a consistency-class distance (do predicted and true histories sit in the
same level of the consistency hierarchy?). `d = 0` iff replicas, log, and consistency class
all agree. `ε` and what counts as "ε-close" for a consistency class is open question §17.3.

### 9.3 Faithful horizon `H_ε(ρ)` and the competitive ratio
`H_ε(ρ)` unchanged in definition (max steps within `ε`), now over a partitioned, faulting
cluster under tiered consultation. The headline curve (ED1, DS6). **New:** report it as a
**competitive ratio** `H_ε(ρ) / H_ε(ρ=full-oracle-ceiling)` against prediction error — the
learning-augmented-algorithms figure of merit (`DD-D4`, H18), with the cheap-tier ratio
labeled *empirical* and the bit-exact-tier ratio labeled *provable* (§2.4 skeptical note).

### 9.4 Bits-to-correct (per-tier), probe/tier efficiency, calibration
- **Bits-to-correct**, per-tier and per-object (§5.4): the scale-free gate.
- **Faithful horizon per oracle-dollar**, where each tier has a different dollar cost — the
  distributed enrichment of `ρ`, and the quantity H17 is about.
- **Belief calibration:** does RSSM belief variance predict error? (continuing SPEC-2 §7.2 /
  SPEC-5 §9.4 — decides whether smart `π_c`/`π_w` *can* work).

---

## 10. Hypotheses

SPEC-7 operationalizes two **already-stated** hypotheses that needed an intractable-oracle,
weakly-consistent world to bite, and adds four **new** ones (H17–H20, plus the data-factory
H21), each falsifiable and each naming its honest negative (SPEC.md §9: "the favorable curve
might not exist").

### 10.1 Existing hypotheses this spec operationalizes (do not re-coin)
- **H8 (SPEC-3 §13) — the interesting interior lives in harder worlds.** The distributed
  world — combinatorial reachability over replicas under fault, with an *intractable* full
  oracle — is the strongest test yet of H1's favorable knee (≥80% of ceiling horizon at
  ≤20% consultation). **This, with H17, is SPEC-7's headline** (ED1, DS6). *Honest negative:*
  the interior is flat/linear here too → the knee is not about world hardness; report it.
- **H5 (SPEC.md §9) — counterfactual lift.** Oracle-grounding improves interventional
  fidelity ("what if this node had not been partitioned at step `t`?") on identical data,
  trained on **branch-replay counterfactuals** the deterministic DES makes free (re-run from
  `(seed, t)` with one fault flipped) — ED6. *Honest negative:* counterfactual data adds
  nothing over factual.

### 10.2 New hypotheses (H17–H21, non-colliding with H1–H16)
- **H17 — tiered oracles dominate single-tier.** A budgeted *mix* of cheap (metamorphic +
  consistency-cycle) and rare expensive (bit-exact/symbolic) oracle calls, scheduled by
  `π_w` (§8.2), achieves higher faithful horizon **per oracle-dollar** than spending the same
  dollar budget entirely on bit-exact full-state checks. This is the central claim and has no
  prior-world analogue (every prior world had a single cheap full oracle). *Honest negative:*
  the cheap tiers refute too rarely to matter, so all the value is in the expensive tier and
  tiering buys nothing — in which case the distributed world is no different in kind, only in
  cost.
- **H18 — the loop is a learning-augmented algorithm with a bounded ratio.** The
  oracle-gated loop achieves faithful horizon within a *bounded factor* of the full-oracle
  ceiling at sub-linear oracle cost, and the factor degrades gracefully with the model's
  prediction error (recovering the trivial bound when the model is useless) — i.e.
  propose-verify-correct has a competitive ratio (`DD-D4`, §2.4). *Honest negative:* the
  ratio is unbounded / grows with horizon → cheap-tier correction does not bound drift and
  only bit-exact reset does (which would itself sharpen *why*). **Result — SPLIT, both
  halves reported (ED5, DS8, [`experiments/ed5.py`](../../src/verisim/experiments/ed5.py),
  [`ed5.png`](../../figures/ed5.png)).** Fitting the competitive ratio `H_ε(ρ)/ceiling`
  across `ρ × prediction error` (the noise dial) at the bit-exact tier (where ρ maps
  linearly to oracle-dollars, so the quarter ρ *is* the `B/4` budget ED2 reads): the
  **graceful-degradation-with-error half is CONFIRMED** — the quarter-budget ratio is
  monotone in the model's competence (1.00 → 0.45 → 0.11 → 0.07 → 0.05 as per-step error
  rises 0.0 → 1.0), recovering the trivial bound (ratio 1.0) for a perfect model and
  collapsing toward the free-running floor for a useless one, exactly the learning-augmented
  signature. But the **bounded-ratio-at-*sub-linear*-cost half reproduces the program's
  recurring floor→cliff / no-knee negative** — at the quarter budget a competent-but-noisy
  model's ratio sits near the floor (~0.11), and the cliff to the ceiling only appears as
  ρ→1, because a discrete-error world's faithful horizon is a *prefix* property only
  near-full consultation protects (the E1/EN1/EH1/ED1 finding, now in competitive-ratio
  form). So the loop *is* learning-augmented in the error axis, but the budget axis buys no
  free lunch on this world ([`test_ed5`](../../tests/test_ed5.py), dependency-free).
- **H19 — consistency-faithful outlasts bit-faithful.** Under a weak consistency model, a
  model is *consistency-faithful* (predicts the observable consistency behavior, §9.1) for
  materially longer than it is *bit-faithful* (predicts the exact replica state), because
  many bit-states map to the same admissible history. *Honest negative:* the two horizons
  coincide → consistency adds no slack, and W7's "no global state" does not buy the model
  any forgiveness. **Result — CONFIRMED, mode-dependently (ED5, DS8).** On the same
  free-running rollout (ρ=0, which exposes the model not the loop) the **consistency-faithful
  horizon outlasts the bit-faithful one for the `subtle` (in-flight) error class — H=13.1 vs
  H=1.5, a gap of +11.6 steps with a disjoint bootstrap CI [3.1, 21.8]** — because the
  asynchronous-replication in-flight message is the gap: a corrupted in-flight payload is
  immediately *bit*-visible (the message fact differs) but **consistency-invisible** (the
  per-object converged/split view reads only replicas) until that message is delivered by
  `advance` and writes a replica. For the `gross` (durable-replica) error class the two
  horizons coincide (gap +0.8, CI includes 0 — the control), exactly the gross/subtle
  structure H17/ED3 turn on. So W7's "no global state" *does* buy the model forgiveness, but
  only where the error lives in the consistency-invisible medium — reported, not assumed.
- **H20 — weaker consistency is harder to predict.** Faithful horizon *decreases*
  monotonically as the declared consistency model weakens (linearizable → eventual), because
  weaker models admit exponentially more legal histories and the model must track which one
  actually occurred. The curve `H_ε(consistency-level)` is the first quantification of
  "consistency strength vs predictability." *Honest negative:* horizon is flat across
  consistency levels → predictability is set by fault intensity, not consistency strength.
  **Result — the mechanism CONFIRMED, dependency-free (ED4 consistency-level arm, DS7,
  [`experiments/ed4_consistency.py`](../../src/verisim/experiments/ed4_consistency.py),
  [`ed4_consistency.png`](../../figures/ed4_consistency.png)).** The `CONSISTENCY_MODELS` axis
  (§3.4) gains its first implemented strong end — **`linearizable`**: synchronous all-replica
  writes, CP write-rejection under partition, so no replica is ever stale and there is **no
  in-flight medium** ([`docs/distributed-semantics.md` §2.1](../distributed-semantics.md),
  goldens in [`test_dist_goldens`](../../tests/test_dist_goldens.py)). Sweeping the declared
  model resolves H20 *through its connection to H19*: the consistency-vs-bit gap (H19) is
  **exclusively a weak-consistency phenomenon** — it needs the consistency-invisible in-flight
  medium to hide errors in. Measured, free-running, with an *exact* in-flight-only error class:
  the `subtle` gap is **+10.5 steps under `eventual` (in-flight rate 3.2/step) and exactly 0
  under `linearizable` (in-flight rate 0)**, while the `gross` durable-replica control is 0 at
  both levels. So strong consistency buys the model no forgiveness because there is no hidden
  state to forgive — the H20 mechanism made concrete. *Honest scope:* the synthetic proposer's
  error distribution is tied to the eventual world's structure, so the *absolute*-predictability
  form of H20 (a monotone `H_ε(level)` curve) is left to the learned `M_θ`; this arm reports the
  gap, which the synthetic proposer measures cleanly ([`test_ed4_consistency`](../../tests/test_ed4_consistency.py)).
- **H21 — fault-injected training beats fault-free (the DST/BUGGIFY lesson).** A model
  trained on DST-style seeded-fault trajectories is more faithful *under fault* than one
  trained on equal-volume fault-free trajectories, at equal clean accuracy. *Honest
  negative:* fault-free training transfers to faulting rollout for free → the fault
  distribution is already implied by the fault-free dynamics.

### 10.3 Outcome → implication: where each distributed result routes the program

Per the epistemic engine (SPEC.md §10.1), each hypothesis is pre-registered to a forward move on *both*
branches. The distributed world is the sharpest illustration of the project's defining move — **a
limitation, faced honestly, becomes the contribution.** Here the limitation is fundamental: bit-exact
full-state truth is *intractable* (serializability/SI checking is NP-complete; there is no consistent
global state without coordination — the wall W7). A lesser program would call that the end of the road.
Instead it is the *premise* of SPEC-7: precisely *because* the full oracle is too expensive, faithfulness
must be verified at **tiered** cost (metamorphic → consistency-cycle → symbolic → bit-exact), and choosing
the cheapest sufficient tier (`π_w`) becomes the central new science. The wall did not stop us; it
*defined the spec*.

- **H17 (tiered oracles dominate single-tier).** *Confirmed* → the central claim holds: budgeted cheap
  tiers buy more faithful horizon per dollar than all-bit-exact → tiering is the distributed-world method.
  *Refuted* → the cheap tiers refute too rarely to matter and all value is in the expensive tier → the
  distributed world is no different *in kind* from the host world, only in cost — a clean simplification
  that retires a whole axis of complexity. Either branch is a real answer to "is tiering worth it?"
- **H18 (the loop has a bounded competitive ratio).** *Confirmed* → propose-verify-correct is a
  learning-augmented algorithm with a provable ratio that degrades gracefully with model error → the loop
  gets *theory*, not just curves (DD-D4, §2.4). *Refuted* → the ratio is unbounded / grows with horizon →
  cheap-tier correction does not bound drift and only bit-exact reset does, which **sharpens *why*** and
  tells us exactly where the cheap tiers fail — a negative that advances the theory by ruling out the easy
  conjecture.
- **H19 (consistency-faithful outlasts bit-faithful).** *Confirmed* → many bit-states map to one
  admissible history, so predicting *observable consistency* buys real slack → the right faithfulness
  target under weak consistency. *Refuted* → the horizons coincide and W7's "no global state" buys no
  forgiveness → a precise statement of when consistency-level abstraction *doesn't* help.
- **H20 (weaker consistency is harder to predict).** *Confirmed* → `H_ε(consistency-level)` is the first
  quantification of consistency-strength vs. predictability — a genuinely new measurement. *Refuted* →
  horizon is set by fault intensity, not consistency strength → redirects modeling effort to fault
  handling, a useful reprioritization.
- **H21 (fault-injected training beats fault-free — the DST/BUGGIFY lesson).** *Confirmed* → seeded-fault
  trajectories (the FoundationDB/TigerBeetle tradition, §2.1) train fault-robustness factual data cannot →
  validates DST as a *data factory*, not just a test harness. *Refuted* → fault-free transfers for free →
  bounds the value of fault injection for *modeling* (as opposed to testing).

The throughline, stated for the hardest world so it is unmistakable: **we do not retreat from
intractability; we tier around it, measure what the tiers buy, and report the number whichever way it
falls.** A wall that is named, quantified, and engineered around is not a limit on the program — it is the
program's next theorem.

---

## 11. Walls (relative to SPEC-3 / SPEC-5 / SPEC-6)

SPEC-7 makes concrete SPEC-3's **W1** (oracle-is-a-model) via Tier-B DST runtimes (§5.2),
inherits SPEC-5's **W5** (asynchronous/temporally-extended effects, now under partition) and
SPEC-6's **W6** (composed multi-subsystem state) and **HW-4** (plasticity loss, via the
experience stream). It adds one genuinely new wall and one hard wall:

- **W7 — there is no consistent global state to be faithful to.** Every prior world had one
  consistent snapshot you could read cheaply and compare bit-for-bit. The distributed world
  has none — a global state is either *coordinated* (expensive) or *inconsistent* (cheap and
  wrong). W7 is what forces `DD-D2` (faithfulness against a consistency model) and the tiered
  oracle (`DD-D1`), and it is what the prime directive attacks.
- **HW-5 (new) — asynchrony & partition.** No global clock; FLP impossibility; the CAP
  tradeoff. Message ordering and partition timing are the nondeterminism source that
  record/replay only tames at the cost of *fixing a schedule* (the DST move). SPEC-7 does not
  pretend to solve it; it makes the fault schedule a *seeded, declared* input (§3.3) and the
  consistency model an explicit choice, and `determinism_report` states both per figure.

---

## 12. Experiments (ED-series)

Non-colliding with E1–E4, the reserved E5/E6 (SPEC-2 §9), EN1–EN9 (SPEC-5), and EH1–EH6
(SPEC-6). The distributed suite is its own namespace, **ED1–ED6**. Each mirrors a prior
experiment's role and names the hypotheses it tests (§10). Every figure regenerates from
config + seeds (the `figures/reproduce.sh` discipline) and **negative results are
first-class** (the repo norm).

- **ED1 — the distributed `H_ε(ρ)` curve** (role of E1/EN1/EH1; the prime directive, DS6).
  Sweep `ρ × ε × difficulty × consistency-level × fault-intensity`. Bootstrap-CI aggregation.
  Reported also as the competitive ratio (§9.3). *Does the knee appear — H8 — and is it
  bigger here because the full oracle is dear?*
- **ED2 — when × which-tier policies** (role of E2/EN2/EH2): cross `π_c` (when) with `π_w`
  (which tier, §8.2), at equal *dollar* budget. *Does the cheapest-sufficient-tier mixture
  beat all-bit-exact and all-cheap — H17 — and does belief-variance scheduling beat fixed?*
  **◐ shipped (the fixed-tier × `escalate` arm at equal dollar budget)**
  ([`experiments/ed2.py`](../../src/verisim/experiments/ed2.py),
  [`ed2.png`](../../figures/ed2.png), [`ed2.csv`](../../figures/ed2.csv)): the
  **faithful-horizon-vs-oracle-dollar frontier** per tier policy on the synthetic proposer
  (dependency-free, GPU-free), with policies compared at a matched budget by interpolating each
  one's horizon along its Pareto envelope (a true equal-*dollar* comparison) and the **H18
  competitive ratio** read off at the sub-linear quarter budget. **H17 in budget form, confirmed
  mode-dependently:** at `B/4`, the metamorphic tier beats bit-exact for **gross** errors
  (H=14.2 vs 4.2, ratio 0.36) and loses for **subtle** errors (H=1.5 vs 4.2, where `escalate`
  also loses to single-tier bit-exact — the honest negative) ([`test_ed2`](../../tests/test_ed2.py)).
  **◐ the `π_c` "smart-when" half also ships** ([`experiments/ed2_smart.py`](../../src/verisim/experiments/ed2_smart.py),
  [`ed2_smart.png`](../../figures/ed2_smart.png), [`ed2_smart.csv`](../../figures/ed2_smart.csv)): at
  a *fixed* interior budget `ρ`, compare the three §6.1 policies (`fixed`/`uncertainty`/`drift`) at
  equal `ρ` on the real flat `M_θ`, the signal being its constrained-decode entropy (wired into the
  loop's `StepContext` by the DS5 runner — the network/host runners already did this; the distributed
  runner gained the `_predict`/`DistUncertaintyModel` plumbing here). **H9 — the standing H2/H9
  negative carried into the distributed world, and *sharper* than a tie:** entropy-gated consultation
  does **not** beat `fixed` — it is strictly *worse*, lift **0.08–0.12×** at every budget, because
  faithful horizon is a *prefix* property (the first divergence step) and `fixed` consults at step 0
  to protect the prefix while the entropy signal spends its budget on late high-entropy steps and lets
  the model derail early. The flat decode-entropy signal is a decode-time artifact, not a calibrated
  belief; this localizes the smart-`π_c` lever to the (deferred) structured `M_θ`'s RSSM belief
  variance — exactly the EH2 lesson, where the host's factored arm's belief variance beat fixed ~2.2×
  where the flat arm's entropy could not ([`test_ed2_smart`](../../tests/test_ed2_smart.py), torch
  extra). The **learned-`M_θ` equal-dollar arm** also shipped (see DS7, [`ed2_learned`](../../figures/ed2_learned.png)).
  *Deferred: the smart-`π_w` (which-tier) scheduling and the structured-arm `π_c` the flat-arm null motivates.*
- **ED3 — correction / belief operators** (role of E3/EN3/EH3): `hard_reset` vs `residual`
  vs `projection` (consistency-model-consistent) vs `belief-filter`. *Do operators differ
  under partition + weak consistency (no v0 identity), and does correction teach over the
  stream — H7?*
  **◐ shipped — and the distributed world *does* break the v0 identity, mode-dependently**
  ([`experiments/ed3.py`](../../src/verisim/experiments/ed3.py), [`ed3.png`](../../figures/ed3.png),
  [`ed3.csv`](../../figures/ed3.csv); the `distloop/operator.py` correction operators the DS5 runner
  was missing). v0 proved an *identity*: a consult returns the full one-step truth, so
  `hard_reset`/`residual`/`projection` all snap to the same `s'` and are behaviorally identical on
  `H_ε` (they differ only in diagnostics). ED3 asks whether the distributed world breaks it — and it
  does, because the cluster state has a part a *partial* correction can decline to fix: the **in-flight
  replication messages**, the stale-read source under partition and exactly the `subtle` error class
  the cheap tiers also miss (§5). The new `ReplicasOnlyCorrection` snaps the durable replicas to truth
  but **trusts the model's predicted in-flight**. Result (synthetic proposer, dependency-free): for
  **gross** (corrupted replica write) errors all four operators recover the same horizon (**H=7.2**,
  the v0 identity holds); for **subtle** (corrupted in-flight) errors the three full-correction
  operators hold the identity (**H=6.2**) but `ReplicasOnlyCorrection` **collapses to H=1.8** (gap
  4.5) — it trusts the corrupted in-flight and the coupled state keeps drifting. *The v0 operator
  identity holds for full correction and breaks for partial correction exactly on the in-flight
  medium — the distributed world's hidden state a partial correction cannot see, tied to the same
  gross/subtle structure H17 turns on* ([`test_ed3`](../../tests/test_ed3.py)). The residual/projection
  diagnostics (bits-to-correct, repaired fraction) quantify how much truth each correction injects.
  *Deferred: a consistency-model `projection` that corrects to the nearest weak-consistency-legal
  state (needs the multi-consistency DS0 increment); online correction-teaches-the-stream (H7).*
- **ED4 — representation & drift ablation** (role of E4/EN4/EH4): service-graph GNN vs flat
  serializer (H11's distributed analogue); RSSM-belief vs Markov; SSM-carry on/off;
  fault-injection (noise) on/off — the **H21 arm**; self-forcing on/off; size;
  **consistency-level sweep** (H20) and **fault-intensity / partition-entropy** as the new
  axes. *Which lesson buys horizon, and how does `H_ε` fall with weaker consistency — H20 —
  and does fault-injected training transfer — H21?*
  **◐ the H21 fault-injection arm shipped (DS7, [`ed4_fault.png`](../../figures/ed4_fault.png)).
  ◐ the consistency-level arm (H20) ships** ([`experiments/ed4_consistency.py`](../../src/verisim/experiments/ed4_consistency.py),
  [`ed4_consistency.png`](../../figures/ed4_consistency.png), [`ed4_consistency.csv`](../../figures/ed4_consistency.csv)),
  dependency-free: it gives the `CONSISTENCY_MODELS` axis (§3.4) its first strong end —
  **`linearizable`** (synchronous all-replica writes, CP write-rejection under partition, no
  in-flight medium) — and sweeps the declared model. **H20 mechanism confirmed through H19:** the
  consistency-vs-bit gap is exclusively a *weak*-consistency phenomenon — it needs the
  consistency-invisible in-flight medium. The `subtle` (in-flight) gap is **+10.5 under `eventual`
  (in-flight rate 3.2/step) and 0 under `linearizable` (rate 0)**, the `gross` durable-replica
  control 0 at both levels — strong consistency buys no forgiveness because there is no hidden
  state to forgive ([`test_ed4_consistency`](../../tests/test_ed4_consistency.py)). *Deferred: the
  service-graph GNN/RSSM representation arm; the absolute-predictability `H_ε(level)` curve on the
  learned `M_θ` (the synthetic proposer measures the gap cleanly but not the absolute horizon).*
- **ED5 — consistency-faithful vs bit-faithful & the competitive ratio** (role of E4
  objective axis): measure the gap between the consistency-faithful and bit-faithful
  horizons (H19) and fit the learning-augmented competitive ratio (H18) across `ρ` and
  prediction error. *Does consistency buy slack — H19 — and is the loop's ratio bounded —
  H18?* **◐ shipped** ([`experiments/ed5.py`](../../src/verisim/experiments/ed5.py),
  [`ed5.png`](../../figures/ed5.png), [`ed5.csv`](../../figures/ed5.csv); the
  consistency-faithfulness trajectory the DS5 runner now records alongside bit-exact
  divergence — the §9.1 headline-new metric's first loop consumer). Both findings on the
  dependency-free synthetic proposer. **H19 confirmed mode-dependently:** free-running
  (ρ=0), the **consistency-faithful horizon outlasts the bit-faithful one for `subtle`
  (in-flight) errors — H=13.1 vs 1.5, gap +11.6, disjoint CI [3.1, 21.8]** — the in-flight
  message is bit-visible but consistency-invisible until delivered by `advance`; for `gross`
  (durable-replica) errors the two coincide (the control). **H18 split:** the
  competitive-ratio fit across `ρ × prediction error` shows graceful degradation with error
  **confirmed** (quarter-budget ratio monotone 1.00 → 0.05 as error rises, recovering the
  trivial bound for a perfect model) while the bounded-ratio-at-sub-linear-cost half
  reproduces the **floor→cliff / no-knee negative** (ratio near the floor at `B/4`, the cliff
  only at ρ→1) — the learning-augmented property holds in the *error* axis, no free lunch in
  the *budget* axis ([`test_ed5`](../../tests/test_ed5.py)). *Deferred: the consistency-level
  H20 sweep and the provable-vs-empirical competitive-ratio split across tiers (needs the
  multi-consistency-model DS0 increment); the counterfactual ED6.*
- **ED6 — counterfactual & multi-tier grounding** (H5, H17-adjacent): train with branch-
  replay counterfactuals (re-run from `(seed, t)` with one fault flipped, §10.1); add the
  symbolic/formal tier (§5.1) on top of the DES; measure interventional fidelity and whether
  the symbolic tier is a non-redundant signal (the distributed analogue of SPEC-5's H12 /
  SPEC-6's EH6). **◐ the counterfactual / H5 arm ships** ([`experiments/ed6.py`](../../src/verisim/experiments/ed6.py),
  [`ed6.png`](../../figures/ed6.png), [`ed6.csv`](../../figures/ed6.csv)): three matched-count arms
  train the same flat DS4 `M_θ` — `trajectory` (base light-fault on-policy), `trajectory-more`
  (5× more on-policy data, the volume control), `+counterfactual` (base + free oracle **fault**-flip
  branches, the near-miss partitions/crashes §17 Q7) — then predict held-out fault interventions,
  scored bit-exact (full next cluster state) and by **medium recall** (predicts the partition/crash
  split-brain). **H5 — and the distributed world is where it finally pays, the honest inverse of
  EN6/EH6.** `+counterfactual` beats **both** the base **and** the matched-volume control on **both**
  metrics with disjoint CIs (intervention-exact **0.51 vs 0.25 vs 0.06**, medium-recall **0.56 vs
  0.22 vs 0.05**) — where the network (EN6) and host (EH6/H16) found counterfactual supervision adds
  nothing over volume. The mechanism is the distributed **medium** (partition/crash/in-flight): a
  hidden state the light-fault on-policy distribution underrepresents, so on-policy *volume* buys
  little (0.06→0.25) while off-policy oracle **fault branches** buy a lot (0.25→0.51) — the held-out-
  intervention analogue of H21 (fault-injection beats fault-free at equal volume). *Honest caveat:*
  the counterfactual branches are fault-heavier than the on-policy control, so the lift conflates
  counterfactual *branching* with the fault *coverage* it carries — but this is the identical
  methodology under which EN6/EH6 found null, so the distributed positive under the same design is
  the result; the branching-vs-coverage split is future work (tied to H21) ([`test_ed6`](../../tests/test_ed6.py),
  torch extra; the matched-*volume* arm needs the minibatched `train_batched` K2 loop, the first
  distributed experiment to). **◐ the two-oracle / H12 slice ships**
  ([`experiments/ed6_two_oracle.py`](../../src/verisim/experiments/ed6_two_oracle.py),
  [`ed6_two_oracle.png`](../../figures/ed6_two_oracle.png),
  [`ed6_two_oracle.csv`](../../figures/ed6_two_oracle.csv)): the distributed analogue of SPEC-5's
  H12 / SPEC-6's EH6 — the cheap **consistency oracle** (the §9.1 split-brain decision: is each
  object converged or split?) as a *second oracle* against the full **bit-exact** one, scored
  teacher-forced over the fault-heavy `adversarial` workload on the dependency-free synthetic
  proposer. **H12 confirmed, mode-dependently:** (1) **non-redundant rate 0** by construction — the
  consistency view is a pure function of the replica state, so a bit-exact-correct prediction is
  always consistency-correct (the cheap oracle catches *nothing* the full one misses: *redundant for
  verification*); (2) **consistency-sufficient rate tracks the in-flight medium** — of the steps
  where the model's *full* prediction is wrong, it is still consistency-faithful **1.00 for `subtle`
  (in-flight) errors vs 0.00 for `gross` (durable-replica) errors** (disjoint CIs), the per-step
  teacher-forced form of ED5's free-running H19 horizon gap; (3) at a **consult-fact ratio of 0.28**
  — the consistency answer is ~3.6× cheaper than the full state, the gap *widening under fault*
  because the medium inflates the full state but never enters the consistency view. So the
  consistency oracle is *redundant* but a **cheaper, decision-sufficient** consult for the question
  an SRE/defender actually asks — the tiered-oracle premise (§5) made concrete, dependency-free
  ([`test_ed6_two_oracle`](../../tests/test_ed6_two_oracle.py)). **◐ the learned-`M_θ` re-pointing of
  this slice ships** ([`experiments/ed6_two_oracle_learned.py`](../../src/verisim/experiments/ed6_two_oracle_learned.py),
  [`ed6_two_oracle_learned.png`](../../figures/ed6_two_oracle_learned.png),
  [`ed6_two_oracle_learned.csv`](../../figures/ed6_two_oracle_learned.csv)): what ED1-learned is to
  ED1, this is to the two-oracle slice — train the flat DS4 `M_θ` (exactly as ED2-learned does) and
  run the **same** teacher-forced H12 measurement on the *real* error distribution rather than the
  dialled one (no `gross`/`subtle` knob — one model, one mixed error distribution). **H12 confirmed on
  the real model, and it is the honest *mirror* of ED2-learned read through the other oracle:**
  non-redundant **0.0** (unchanged — structural), the consistency oracle **decision-sufficient on
  0.57 [0.53, 0.61]** of the model's bit-wrong steps at a **consult-fact ratio of 0.28 (~3.6× cheaper)**
  — *even though the full prediction is wrong 87% of the time* (the uniform-trained model on the
  fault-heavy `adversarial` eval). The 0.57 lands **between the synthetic `gross` (0.0) and `subtle`
  (1.0) poles**: ED2-learned showed the constrained decoder removes the `gross` *out-of-vocab* class
  so the cheap *refutation* tiers are useless, but the residual learned errors are a **mixture**
  (predominantly the consistency-invisible in-flight class, not purely it), so the cheap *decision*
  oracle is sufficient on a majority but not all of them. The same model, same decoder: the cheap
  oracle **loses as a verifier** (ED2-learned's tiers refute nothing) yet is **decision-sufficient on
  the majority of errors as a decision oracle** — the clearest single statement of why the tiered
  oracle's value depends on *which question you ask it*, on the model that actually exists
  ([`test_ed6_two_oracle_learned`](../../tests/test_ed6_two_oracle_learned.py), torch extra).
  *Deferred: the symbolic/legality tier as a third oracle.*

**External harness (optional, for community legibility).** Where cheap, ED1/ED3/ED5 are
additionally reported against **τ-bench / AppWorld**-class stateful-backend agent tasks
(§2.6) and against a **Jepsen**-style consistency test suite — gyms the distributed-systems
and agent communities already trust — with the speedup-vs-fidelity Pareto (model alone →
model + budgeted tiered oracle → full oracle) the way DST tools report bugs-per-CPU-hour and
learned simulators report speedup vs ns-3. Defensive tasks only (§15).

---

## 13. Milestones (DS0–DS8)

SPEC-7 is the buildable expansion of the distributed/service concern implicit in SPEC.md
§11's in-scope list ("web applications, APIs, key-value stores") and the integration of
SPEC-5 and SPEC-6. The `DS` series is to the distributed world what `M0–M8` were to the
filesystem, `NW0–NW8` to the network, and `HC0–HC8` to the host: deterministic core first
(DS0–DS3, **no runtime deps, no GPU**), learned model after. It does not collide with
`M0–M8`, `S1–S6`, `AR0–AR5`, `NW0–NW8`, or `HC0–HC8`.

| Milestone | What | Gate |
|---|---|---|
| **DS0** | Distributed env: event-sourced/replicated `State` (worldify-style log + happens-before), action grammar (client/protocol/fault), canonical serialization + **Tier-A reference DES** (replicated KV + Raft-subset + txn/lock table, embedding SPEC-6 hosts / SPEC-5 net) + `docs/distributed-semantics.md` + golden trajectories | property tests + goldens — **◐ increment 1 shipped**: the **replicated KV under partition** core ([`dist/`](../../src/verisim/dist/), [`distoracle/`](../../src/verisim/distoracle/)) — `DistributedState` (replicas + causal log + in-flight messages + partition/crash/clock), the client (`put`/`get`/`cas`) + fault/time (`advance`/`partition`/`heal`/`crash`/`restart`) grammar, the Tier-A async-replication DES with eventual-consistency LWW, canonical serialization, [`docs/distributed-semantics.md`](../distributed-semantics.md), and golden trajectories pinning stale-read-under-partition + convergence ([`test_dist_core`](../../tests/test_dist_core.py), [`test_dist_goldens`](../../tests/test_dist_goldens.py); dependency-free, GPU-free). **◐ a second consistency model ships** for the H20 sweep: **`linearizable`** (synchronous all-replica writes, CP write-rejection under partition, so no replica is ever stale and there is no in-flight medium — [`docs/distributed-semantics.md` §2.1](../distributed-semantics.md), goldens pinning synchronous replication + partitioned-write rejection). *Deferred to later DS0 increments: the intermediate consistency models (serializable/snapshot/causal), the Raft-subset consensus group, the transaction/lock table, and the embedded SPEC-6 host / SPEC-5 net inside each node.* |
| **DS1** | Log/replica `Delta` types, compositional `apply` (reusing SPEC-2/5/6 `apply` for embedded subsystems), delta↔serialization; the `apply == oracle` invariant | invariant tests — **◐ shipped for the DS0-increment-1 slice**: the `DistDelta` edit vocabulary (`ReplicaWrite`/`MsgSend`/`MsgDeliver`/`EventAppend`/`PartitionSet`/`NodeDown`/…), `apply` as a pure function, delta↔serialization round-trips, and the `apply(state, oracle.delta) == oracle.next_state` invariant tested on every transition ([`dist/delta.py`](../../src/verisim/dist/delta.py)). The embedded host/net delta composition arrives with their embedding (later DS0 increment). |
| **DS2** | Drivers (workload + seeded fault injection = `BUGGIFY`; topology/replication/consistency generators), trajectory JSONL, manifests/splits, the **fault-intensity / partition-entropy dials** | data tests — **◐ shipped for the DS0-increment-1 world** ([`distdata/`](../../src/verisim/distdata/)): the seeded `DistDriver` (`uniform`/`contention`/`adversarial`) interleaving client ops + `advance` + faults, with the **explicit `fault_prob` (fault-intensity) and `partition_bias` (partition-entropy) dials** the H20/H21 sweeps need; trajectory JSONL + regenerable dataset manifests with disjoint trajectory-level splits; tested for valid-action/`apply==oracle`, determinism, dial monotonicity, and preset distinctness ([`test_dist_data`](../../tests/test_dist_data.py)). Extends with the consensus/transaction ops as DS0 grows. |
| **DS3** | Consistency-faithfulness (Elle-style cycle detection), consistency/composed divergence `d`, `H_ε`, per-tier bits-to-correct, run-record schema; the **tiered-oracle interface** (`O[tier]`) with metamorphic + cycle + symbolic + bit-exact tiers | metric + oracle-tier tests — **◐ the metric core AND the tiered oracle ship**. *Metrics* ([`distmetrics/`](../../src/verisim/distmetrics/)): the **live-cluster divergence** `d(s, ŝ)` (a normalized fact-set difference over replicas + in-flight + partition/crash/clock, feeding the generic `faithful_horizon` so distributed `H_ε(ρ)` is defined exactly as in every prior world), the **headline-new consistency-faithfulness** (§9.1 — the fraction of objects whose converged/split *consistency view* the model predicts right, which catches a model that mispredicts a partition split as converged), and **bits-to-correct / delta-exact** over the `DistDelta` ([`test_dist_metrics`](../../tests/test_dist_metrics.py)). *The tiered oracle — SPEC-7's payload (§5, DD-D1)* ([`distoracle/tiers.py`](../../src/verisim/distoracle/tiers.py)): the four-tier menu (**metamorphic** ¢1 → **cycle** ¢2 → **symbolic** ¢4 → **bit-exact** ¢16) with `cheapest_refutation` spending the cheapest tier that can refute a prediction, the cumulative oracle-dollar cost recorded — every error class caught at its right tier, and a subtle invariant-respecting error caught only by bit-exact (the non-redundancy H17 measures, [`test_dist_tiers`](../../tests/test_dist_tiers.py)). *Deferred: the Elle-style cross-object cycle detection for stronger consistency models (the cycle tier is the eventual-consistency form here), and the run-record schema for the DS5 loop.* |
| **DS4** | `M_θ`: service-graph message-passing + RSSM belief (+ optional SSM carry), constrained delta decode, supervised training (SLM-sized) | model tests (torch extra) — **◐ increment 1 shipped — the dependency-free serialization foundation** ([`distmodel/`](../../src/verisim/distmodel/)): the closed token [`DistVocab`](../../src/verisim/distmodel/vocab.py) (specials + structure markers + the 10 delta ops + 8 commands + 5 result statuses + the node/object/value leaf pools + a single bounded **integer pool** `<int:0..max_int>` that closes the one unbounded family — the monotone bookkeeping counters `version`/`msg_id`/`deliver_after`/`clock` — the host's `max_pid`/`max_fd` trick), and the bidirectional [`tokenizer`](../../src/verisim/distmodel/tokenizer.py) mapping `<bos> state action <gen>` → `Δ <eos>` with an **exact inverse `parse_target`**. The design move that makes the distributed delta tokenizable: the causal-log `EventAppend` (whose `happens_before` is the one genuinely variable-length field) is encoded as a bare `<event_append>` marker and **reconstructed deterministically from `(state, action)`** on parse — `id`/`node`/`op`/`clock`/`happens_before` are all pure functions of the step context, exactly as the network tokenizer omits the always-1 `ClockAdvance` amount, so the unbounded list never enters the token grammar. The serialization module files stay torch-free if imported directly, so they remain in the dependency-free core; the **round-trip `parse(encode(Δ)) == Δ`** is tested exhaustively over every preset × 6 seeds × 40 steps (full edit-vocabulary coverage), on a 5-node/3-object cluster, on a multi-group partition, and the decoded delta is shown to still satisfy the M1 invariant `apply(state, Δ) == oracle.next_state` ([`test_dist_model`](../../tests/test_dist_model.py)). **◐ increment 2 shipped — the learned (flat) arm** ([`grammar.py`](../../src/verisim/distmodel/grammar.py), [`decode.py`](../../src/verisim/distmodel/decode.py), [`world_model.py`](../../src/verisim/distmodel/world_model.py), [`dataset.py`](../../src/verisim/distmodel/dataset.py)): the LL(1) constrained-decode `DistDeltaGrammar` (the distributed analogue of v0's `DeltaGrammar`, carrying two structured nonterminals the flat net/host grammars do not need — the **nested partition run** `<pgroup> NODE+ … <pgroups_end>` and the **status-typed result** where `advanced` is followed by an int and every other status by a value), the `NeuralDistWorldModel` over v0's `GPT` (a drop-in `DistModel` for the DS5 loop, with a `predict_delta_with_uncertainty` decode-entropy signal for `π_c`), and the supervised dataset builders feeding the generic `verisim.train` trainers. A **structural-bug fix found by free-running decode**: an untrained model could emit `<event_append>` after a non-client action (whose `args[0]` is not a coordinator node), so the decoder now masks `<event_append>` out of the top-level op set for fault/time ops — the one op whose reconstruction reads `action.args[0]`, kept to the client-op context the oracle's language actually produces (§5.1). Tested ([`test_dist_model_decode`](../../tests/test_dist_model_decode.py), torch extra): constrained decode is **grammar-valid from an untrained model** across the partition/advance/put shapes, a tiny cluster **overfits to <0.05 loss** and free-runs the training deltas back (each still satisfying the M1 invariant), the model **satisfies the `DistModel` loop protocol**, and decode is config-driven on a 5-node/3-object cluster. *Deferred: the service-graph message-passing + RSSM-belief arm (the structured `M_θ`, SPEC-7 §6.1-6.2) — under full observability it degenerates to this flat Markov predictor (§6.2), so it lands with the partial-observation work; supervised dataset JSONL/manifest persistence.* |
| **DS5** | Tiered propose-verify-correct loop with `π_c` × `π_w` (when × which-tier), consistency/belief operators, experience-stream scaffolding, baselines | loop invariants — **✅ the tiered loop ships** ([`distloop/`](../../src/verisim/distloop/)): the model-agnostic runner over any `DistModel` (with `DistNullModel`/`DistOracleBackedModel` baselines), the **`π_w` which-tier axis** ([`tier_policy.py`](../../src/verisim/distloop/tier_policy.py): `FixedTierPolicy` + the cheapest-refutation `EscalatingTierPolicy`), and the **oracle-dollar accounting** — each consult spends its tier's cost, a refutation adds the bit-exact correction cost, and a prediction the tier cannot refute is *trusted*; the run-record carries the divergence trajectory (→ `H_ε`) **and** the cumulative oracle-dollars (→ H17). Loop invariants tested ([`test_dist_loop`](../../tests/test_dist_loop.py)): ρ=1 reproduces truth (`H_ε=T`), the perfect model never drifts at ρ=0 spending $0, the null model drifts at step 0, the budget is spent exactly, and the oracle-dollar reflects the tier policy (escalation pays the cheap tiers before bit-exact when errors are caught late — the genuine H17 nuance). The §8.3 **correction operators `C`** now ship too ([`operator.py`](../../src/verisim/distloop/operator.py): `HardReset` (default) + `Residual`/`Projection` diagnostics + the partial `ReplicasOnlyCorrection`), wired into the runner via the `operator` parameter and the `π_c` uncertainty signal via `_predict`/`DistUncertaintyModel`. *Deferred: the experience stream; online correction-teaches-the-stream (H7).* |
| **DS6** | **ED1 distributed `H_ε(ρ)` curve** + the **tiered-oracle measurement (H17)** + competitive-ratio fit (H18) + bootstrap-CI aggregation + figure | **the prime directive** — **◐ the apparatus + the first distributed curve ship** ([`experiments/ed1.py`](../../src/verisim/experiments/ed1.py), [`ed1_dist.png`](../../figures/ed1_dist.png), [`ed1_dist.csv`](../../figures/ed1_dist.csv)): the distributed **`H_ε(ρ)` curve** (floor 0.2 at ρ=0 → ceiling 40 at ρ=1, bootstrap-CI over seeds — the standard prime-directive shape, comparable to v0/EN1/EH1) **and the H17 tiered-oracle measurement** — oracle-dollar *per faithful step* for each fixed tier × proposer error class. **H17 verdict (apparatus, on a controlled error distribution):** *whether a cheap tier buys more faithful horizon per oracle-dollar depends on where the model's errors fall.* For **gross** (out-of-vocab) errors the metamorphic tier is cheaper per faithful step ($9.4) than always-bit-exact ($16); for **subtle** (in-flight) errors the cheap tiers miss the drift entirely (H≈0, $848/step) and bit-exact is the only efficient choice. Run on a synthetic tunable-noise proposer ([`DistNoisyModel`](../../src/verisim/distloop/model.py)) before the learned `M_θ` (DS4) supplies a *real* error distribution; the loop + tiered-oracle + oracle-dollar machinery is exercised and the H17 tradeoff is exact ([`test_ed1`](../../tests/test_ed1.py)). **◐ the learned-model curve ships** ([`experiments/ed1_learned.py`](../../src/verisim/experiments/ed1_learned.py), [`ed1_learned.png`](../../figures/ed1_learned.png), [`ed1_learned.csv`](../../figures/ed1_learned.csv)): the flat DS4 `M_θ` trained on seeded rollouts and run through the *same* tiered loop, so the curve and H17 are measured on a **real** error distribution. The curve is the same **floor→cliff** (floor 0.2 at ρ=0 → ceiling 32 at ρ=1, in-distribution eval — the EN1/EH1 step for this world). The **real-model H17 finding, and it is the honest inverse of the synthetic one**: the constrained decoder (DS4 incr 2) removes the *gross* (out-of-vocab) error class by construction, so the learned model's residual errors are *subtle* — the cheap **metamorphic** tier catches none of them (H=0.2, **$624/faithful-step**), **symbolic** few (H=0.8, $411), and only **bit_exact** is efficient (H=32, **$16**); the cheapest-refutation **escalate** policy reaches full horizon but pays **more** ($21.6) because a real model's errors need the bit-exact correction anyway. *So a cheap tier helps exactly when a model makes catchable-cheaply errors — and a grammar-constrained learned model, by design, does not; the tiered oracle's value is model-dependent, reported not assumed* ([`test_ed1_learned`](../../tests/test_ed1_learned.py), torch extra). *Deferred: the competitive-ratio fit (H18).* |
| **DS7** | Smart `π_w` + consistency operators + drift mitigations + the **consistency-level (H20) and fault-injection (H21) sweeps**; ED2/ED3/ED4 (equal-dollar-budget, CIs) | comparison figures — **◐ the equal-dollar-budget ED2 (H17/H18) ships** ([`experiments/ed2.py`](../../src/verisim/experiments/ed2.py), [`ed2.png`](../../figures/ed2.png), [`ed2.csv`](../../figures/ed2.csv)): the **faithful-horizon-vs-oracle-dollar frontier** per tier policy (`metamorphic`/`symbolic`/`bit_exact` fixed + the cheapest-refutation `escalate`) on the synthetic proposer, dependency-free and GPU-free. Where ED1 reported cost *per faithful step at ρ=1*, ED2 sweeps ρ and compares policies **at a matched dollar budget** — interpolating each policy's horizon along its Pareto envelope (a true equal-*dollar*, not equal-ρ, comparison) — and reads the **H18 competitive ratio** off the same frontier at the sub-linear quarter budget `B/4`. **H17 in budget form, confirmed mode-dependently:** at `B/4` the metamorphic tier buys **H=14.2 vs bit-exact's 4.2** for **gross** (cheaply-catchable) errors (tiering wins, ratio 0.36 of the full-truth ceiling at ¼ the cost) but is flat at the floor (**H=1.5 vs 4.2**) for **subtle** (bit-exact-only) errors, where even `escalate` *loses* to single-tier bit-exact — H17's honest negative reported, not hidden ([`test_ed2`](../../tests/test_ed2.py)). **◐ the H21 / fault-injection arm of ED4 ships** ([`experiments/ed4_fault.py`](../../src/verisim/experiments/ed4_fault.py), [`ed4_fault.png`](../../figures/ed4_fault.png), [`ed4_fault.csv`](../../figures/ed4_fault.csv)): the **DST/BUGGIFY data-factory lesson**, made measurable by the DS2 driver's `fault_prob` dial — train two DS4 `M_θ` of **equal volume**, one fault-free (`fault_prob=0`) and one fault-injected, then sweep the eval workload's fault-intensity **free-running** (ρ=0 exposes the model, not the loop). **H21 confirmed, with the sharpest possible control:** at zero eval-fault the two coincide, but as faults intensify the **fault-injected** model holds ~3× more free-run faithful horizon (0.375 vs 0.125 steps) — *even though the fault-free model is the **better** clean predictor* (teacher-forced accuracy 0.60 vs 0.49). The fault-free model never saw a partition/crash/heal, so under fault it derails immediately; fault injection buys robustness factual data cannot — validating DST as a *data factory*, not just a test harness. A bonus in-figure instance of the program's proxy/truth divergence: higher per-token clean accuracy, lower compounding free-run horizon. The dataset builders gained the `fault_prob`/`partition_bias` dials the H20/H21 axes need ([`distmodel/dataset.py`](../../src/verisim/distmodel/dataset.py), [`test_ed4_fault`](../../tests/test_ed4_fault.py), torch extra). **◐ the learned-`M_θ` equal-dollar arm of ED2 ships** ([`experiments/ed2_learned.py`](../../src/verisim/experiments/ed2_learned.py), [`ed2_learned.png`](../../figures/ed2_learned.png), [`ed2_learned.csv`](../../figures/ed2_learned.csv)): ED2's synthetic frontier re-pointed at the **real** flat DS4 `M_θ` (trained exactly as ED1-learned), so the equal-dollar H17/H18 question is answered on a *real* error distribution — what ED1-learned is to ED1, this is to ED2. **The finding is the honest inverse of ED2's `gross` panel, in budget form, and the budget-form of ED1-learned's per-step H17:** the constrained decoder removes the gross (out-of-vocab) error class by construction, so a real model lives entirely in ED2's `subtle` regime — at the sub-linear quarter budget `B/4=$128` the cheap tiers stay flat at the floor (**metamorphic H=0.2, symbolic H=0.8**) while only **bit_exact buys horizon (H=2.0)**, and the cheapest-refutation **escalate** policy *loses* to single-tier bit-exact (H=1.6, and at every ρ it spends strictly more — `$691 vs $512` to reach the same H=32 ceiling — because it pays the cheap probes before the bit-exact correction a real model's subtle errors always need). The **H18 competitive ratio at `B/4` is just 0.06** of the full-truth ceiling: for a grammar-constrained learned model a sub-linear budget buys little horizon however the tiers are sliced — H17/H18's honest negative for the real model, *reported not assumed* ([`test_ed2_learned`](../../tests/test_ed2_learned.py), torch extra). *So the tiered oracle's value is model-dependent: a cheap tier helps exactly when a model makes catchable-cheaply errors, and a grammar-constrained learned model, by design, does not.* **◐ the `π_c` "smart-when" arm of ED2 ships** ([`experiments/ed2_smart.py`](../../src/verisim/experiments/ed2_smart.py), [`ed2_smart.png`](../../figures/ed2_smart.png), [`ed2_smart.csv`](../../figures/ed2_smart.csv)): the missing *when* axis of ED2 — at a fixed interior budget `ρ`, does spending the consults on the steps the flat `M_θ` is least sure about (its constrained-decode entropy) beat spreading them evenly? The DS5 runner gained the uncertainty plumbing it had been missing — a `_predict` helper + a `DistUncertaintyModel` protocol that feed the per-step decode entropy into the loop's `StepContext`, exactly as the network/host runners already do (the genuine gap this increment closed). **H9 — the standing H2/H9 negative carried into the distributed world, and sharper than a tie:** entropy-gated consultation does **not** beat `fixed` — it is strictly *worse* (lift **0.08–0.12×** at every budget), because faithful horizon is a *prefix* property and `fixed` consults at step 0 to protect the prefix while the entropy signal spends late and lets the model derail early. The flat decode-entropy signal is a decode-time artifact, not a calibrated belief; this localizes the smart-`π_c` lever to the (deferred) structured `M_θ`'s RSSM belief variance — the EH2 lesson (the host's factored belief-variance beat fixed ~2.2× where flat entropy could not), now the flat-arm baseline the distributed structured arm must beat ([`test_ed2_smart`](../../tests/test_ed2_smart.py), torch extra). **◐ ED3 — the correction operators ship, and the distributed world breaks v0's operator identity** ([`experiments/ed3.py`](../../src/verisim/experiments/ed3.py), [`ed3.png`](../../figures/ed3.png), [`ed3.csv`](../../figures/ed3.csv); [`distloop/operator.py`](../../src/verisim/distloop/operator.py)): the DS5 runner gained the §8.3 correction-operator axis it was missing (`HardReset` default + `Residual`/`Projection` diagnostics + the partial `ReplicasOnlyCorrection`). v0 proved an identity — a full-truth consult makes `hard_reset`/`residual`/`projection` behaviorally identical on `H_ε` — and ED3 shows the **distributed world breaks it, mode-dependently**, because the partial `ReplicasOnlyCorrection` snaps the durable replicas but **trusts the model's predicted in-flight** (the stale-read medium, the `subtle` error class §5). For **gross** (replica-write) errors all four operators recover the same horizon (H=7.2, identity holds); for **subtle** (in-flight) errors the three full-correction operators hold the identity (H=6.2) but `ReplicasOnlyCorrection` **collapses to H=1.8** (gap 4.5) — the in-flight medium is the distributed world's hidden state a partial correction cannot see, tied to the same gross/subtle structure H17 turns on ([`test_ed3`](../../tests/test_ed3.py), dependency-free). **◐ the consistency-level H20 arm of ED4 ships** ([`experiments/ed4_consistency.py`](../../src/verisim/experiments/ed4_consistency.py), [`ed4_consistency.png`](../../figures/ed4_consistency.png)): the `CONSISTENCY_MODELS` axis (§3.4) gains its first strong end — **`linearizable`** (synchronous all-replica writes, CP write-rejection under partition, no in-flight medium) — and the sweep resolves H20 *through* H19: the consistency-vs-bit gap is exclusively a *weak*-consistency phenomenon (it needs the consistency-invisible in-flight medium), measuring **+10.5 under `eventual` / in-flight rate 3.2 vs 0 under `linearizable` / in-flight rate 0**, gross control 0 at both ([`test_ed4_consistency`](../../tests/test_ed4_consistency.py), dependency-free). *Deferred: the smart-`π_w` (which-tier) scheduling, the consistency-model `projection` operator, the absolute-predictability `H_ε(level)` curve on the learned `M_θ`, and the GNN/RSSM representation arm of ED4 (which the flat-arm smart-`π_c` null motivates).* |
| **DS8** | Consistency-vs-bit horizon + competitive ratio (ED5/H18/H19), counterfactual replay (ED6/H5), **Tier-B system oracle** (madsim/Shadow/Antithesis-class), the **LLM-callable cluster-simulator protocol** (§7), the **verified-contribution protocol** (SPEC-6 §16), Inspect benchmark + `verifiers`-spec distributed RL env, technical report | packaging + report — **◐ ED5 (H18/H19) ships** ([`experiments/ed5.py`](../../src/verisim/experiments/ed5.py), [`ed5.png`](../../figures/ed5.png), [`ed5.csv`](../../figures/ed5.csv)): the §9.1 consistency-faithfulness metric gets its first loop consumer (the DS5 runner now records the consistency-divergence trajectory alongside bit-exact divergence), and ED5 reads both findings off the dependency-free synthetic proposer. **H19 confirmed mode-dependently** — free-running consistency-faithful horizon outlasts bit-faithful for `subtle` in-flight errors (H=13.1 vs 1.5, gap +11.6, disjoint CI; the in-flight message is bit-visible but consistency-invisible until delivery) and coincides for `gross` durable-replica errors (the control). **H18 split** — the competitive-ratio fit across `ρ × prediction error` confirms graceful degradation in the *error* axis (quarter-budget ratio monotone 1.00 → 0.05, trivial bound recovered for a perfect model) but reproduces the floor→cliff *no-knee* negative in the *budget* axis (ratio near the floor at `B/4`, cliff only at ρ→1) — learning-augmented in kind, no free lunch at sub-linear budget ([`test_ed5`](../../tests/test_ed5.py)). **◐ ED6 (H5 counterfactual lift) ships** ([`experiments/ed6.py`](../../src/verisim/experiments/ed6.py), [`ed6.png`](../../figures/ed6.png), [`ed6.csv`](../../figures/ed6.csv)): three matched-count arms train the same flat DS4 `M_θ` — `trajectory` (base light-fault on-policy), `trajectory-more` (5× more on-policy, the volume control), `+counterfactual` (base + free oracle **fault**-flip branches, the §10.1 "re-run from `(seed,t)` with one fault flipped") — then predict **held-out fault interventions**, scored bit-exact (full next cluster state) and by **medium recall** (predicts the partition/crash split-brain, §17 Q7). **H5 confirmed — and the distributed world is where it finally pays, the honest inverse of EN6/EH6:** `+counterfactual` beats **both** the base **and** the matched-volume control on **both** metrics, disjoint CIs (intervention-exact **0.51 vs 0.25 vs 0.06**, medium-recall **0.56 vs 0.22 vs 0.05**), where the network (EN6) and host (EH6/H16) found counterfactual supervision adds nothing over volume. The mechanism is the distributed **medium** (partition/crash/in-flight) — a hidden state the light-fault on-policy distribution underrepresents, so on-policy *volume* buys little (0.06→0.25) while off-policy oracle **fault branches** buy a lot (0.25→0.51): the held-out-intervention analogue of H21. *Honest caveat:* the branches are fault-heavier than the on-policy control, so the lift conflates counterfactual *branching* with the fault *coverage* it carries — but EN6/EH6 found null under the identical design, so the distributed positive is the result; the disentanglement is future work (tied to H21). The matched-*volume* arm is the first distributed experiment to need the minibatched `train_batched` K2 loop (a real perf fix vs the full-batch path the small-dataset learned arms use — [`test_ed6`](../../tests/test_ed6.py), torch extra). **◐ the ED6 two-oracle / H12 slice ships** ([`experiments/ed6_two_oracle.py`](../../src/verisim/experiments/ed6_two_oracle.py), [`ed6_two_oracle.png`](../../figures/ed6_two_oracle.png), [`ed6_two_oracle.csv`](../../figures/ed6_two_oracle.csv)): the distributed analogue of SPEC-5's H12 / SPEC-6's EH6 — the cheap **consistency oracle** (the §9.1 split-brain decision) as a *second oracle* against the full bit-exact one, teacher-forced over the fault-heavy `adversarial` workload on the dependency-free synthetic proposer. **H12 confirmed, mode-dependently:** **non-redundant rate 0** by construction (the consistency view is a pure function of replicas, so a bit-exact-correct prediction is always consistency-correct — *redundant for verification*); **consistency-sufficient rate 1.00 for `subtle` (in-flight) vs 0.00 for `gross` (durable-replica) errors** (disjoint CIs — the per-step teacher-forced form of ED5's H19 horizon gap, tracking the in-flight medium); at a **consult-fact ratio of 0.28** (~3.6× cheaper, the gap widening under fault as the medium inflates the full state but never the consistency view). The consistency oracle is *redundant* but a **cheaper, decision-sufficient** consult for the question an SRE/defender actually asks ([`test_ed6_two_oracle`](../../tests/test_ed6_two_oracle.py)). **◐ the §16 verified-contribution protocol ships** ([`distcontrib/`](../../src/verisim/distcontrib/), [`test_distcontrib.py`](../../tests/test_distcontrib.py)): the dependency-free distributed analogue of the host `contrib/` — `verify_transition`/`verify_trajectory` accept a contributed trace iff re-running the oracle reproduces it, with the distributed-specific **tiered acceptance** (`bit_exact` demands byte-for-byte, the cheap tiers admit any next-state legal under the declared model — the W7 path), trajectory chaining (no splicing), SHA-256 content-addressing, and hostile-input safety. *Fixed in the build:* the `from_canonical(to_canonical(s))` round-trip was non-exact because `partitions` was stored in construction order while serialization sorted it — fixed at the source (canonical partition order in `apply` + `DistributedState.__post_init__`), pinned by a round-trip test ([`test_dist_core`](../../tests/test_dist_core.py)). **◐ the §7 LLM-callable cluster simulator ships** ([`distsim/`](../../src/verisim/distsim/), [`test_distsim.py`](../../tests/test_distsim.py)): the dependency-free distributed analogue of the host `hostsim/` — `DistSimulator.imagine` (oracle-free plan rollout, the cheap draft) + `verify` (against the oracle → a `DistPlanReport`) with the two distributed-specific readouts: a **consistency-faithful plan horizon** distinct from the bit-exact one (ED5/H19 lifted to the plan level — the agent trusts the model's split-brain prediction longer than its byte prediction) and **change-safety as the differential in consistency health** (the securifine pattern: does the plan break consistency, and does the model agree with the oracle on that verdict?), plus a composing `DistGoal` task oracle. Exercised in CI with no torch (the dependency-free baselines satisfy `DistModel`). **◐ the learned-`M_θ` re-pointing of the ED6 two-oracle / H12 slice ships** ([`experiments/ed6_two_oracle_learned.py`](../../src/verisim/experiments/ed6_two_oracle_learned.py), [`ed6_two_oracle_learned.png`](../../figures/ed6_two_oracle_learned.png), [`ed6_two_oracle_learned.csv`](../../figures/ed6_two_oracle_learned.csv)): what ED1-learned is to ED1, this is to the two-oracle slice — train the flat DS4 `M_θ` (exactly as ED2-learned) and run the **same** teacher-forced H12 measurement on its *real* (un-dialled) error distribution. **H12 confirmed on the real model, the honest mirror of ED2-learned read through the other oracle:** non-redundant **0.0** (structural, unchanged), the consistency oracle **decision-sufficient on 0.57 [0.53, 0.61]** of the model's bit-wrong steps at a **consult-fact ratio 0.28 (~3.6× cheaper)** — even though the full prediction is wrong **87%** of the time on the fault-heavy eval. The 0.57 sits **between the synthetic `gross` (0.0) and `subtle` (1.0) poles** because a real error distribution is a *mixture* (predominantly the consistency-invisible in-flight class). The same model, same constrained decoder, **loses as a verifier** (ED2-learned's cheap tiers refute nothing) yet is **decision-sufficient on the majority of errors as a decision oracle** — the clearest single statement that the tiered oracle's value depends on *which question you ask it* ([`test_ed6_two_oracle_learned`](../../tests/test_ed6_two_oracle_learned.py), torch extra). **◐ the distributed world is packaged for reuse** (the DoD §4 deliverable): the **`verifiers`-spec distributed RL env** ([`distrl/`](../../src/verisim/distrl/), [`test_distrl.py`](../../tests/test_distrl.py)) — the distributed analogue of [`hostrl/`](../../src/verisim/hostrl/), a dependency-free reset/step env whose **reward is the tiered oracle's faithfulness verdict** (no learned reward model in the loop — the verifier-as-reward thesis), teacher-forced so the episode **return *is* the faithful horizon `H_ε`**, with the one distributed-specific knob `reward_mode ∈ {bit_exact, consistency}` so an agent can be graded on the §9.1 split-brain *decision* (the SRE/defender's question, which outlasts the bit-exact horizon where the error hides in the in-flight medium — ED5/H19) rather than on bytes; and the **Inspect benchmark** ([`disteval/`](../../src/verisim/disteval/), [`test_disteval.py`](../../tests/test_disteval.py)) — the distributed analogue of [`hosteval/`](../../src/verisim/hosteval/), a framework-agnostic faithfulness benchmark (rollout `score_dist_model` reporting **both** the bit-faithful horizon **and** the consistency-faithful one **and** the tiered **oracle-dollars**, plus single-step labels + divergence graders) with a lazily-imported `inspect_ai` task adapter — the metrology SPEC-7 §1.4 argues the field lacks (Jepsen grades a running system's history, never a simulator's predicted next cluster state). **◐ the DS8 technical report ships** ([`docs/report.md`](../report.md) — the distributed-world section): the honest hypothesis-by-hypothesis write-up the DoD §3 requires, reading the committed numbers off the CSVs for H8 (the floor→cliff carried into the fourth world), H17 (the model-dependent tiered oracle, synthetic + learned), H18 (learning-augmented in the error axis, floor→cliff in the budget axis), H19 (the consistency-vs-bit horizon gap that tracks the in-flight medium), H20 (the gap is a weak-consistency phenomenon), H21 (fault-injected training beats fault-free at equal volume), H5 (the counterfactual-replay positive, the honest inverse of EN6/EH6), and H12 (the redundant-but-decision-sufficient consistency oracle, synthetic + learned) — each with its honest negative and caveat, plus the SPEC-7 reproduce commands and a distributed threat-to-validity note. *Deferred: Tier-B (the real-DST re-execution path).* |

DS0–DS3 + the DS5 loop are the deterministic core. `M_θ` (DS4) drops into the loop via the
same model-agnostic interface every prior world uses. Tier-B, torch, and the LLM client are
optional extras; the deterministic core has no runtime deps.

---

## 14. The autonomous research engine over the distributed world

SPEC-7 is built, as far as possible, with the human at the boundary — extending SPEC-4's
ratchet, not replacing it.

- **The gate is unfakeable, and now per-tier.** You cannot lower bits-to-correct (§5.4) or
  raise consistency-faithfulness (§9.1) without predicting the truth; the per-tier
  decomposition gives the engine a *richer* gradient — it can search for "improve the tier /
  consistency class that is leaking" — while staying ground-truth (SPEC-4 §9). And because
  the DES is the data factory (§2.1), the engine can *generate its own harder curriculum*
  (raise fault intensity, weaken consistency) under the same denylist.
- **Search space (knobs the proposer may turn):** consistency-level and fault-intensity
  curriculum, service-graph/SSM architecture, RSSM and SSM-carry toggles, drift-mitigation
  strength (fault-noise σ, schedule), `π_c`/`π_w` policies (including the tier mixture),
  TTT/stream learning rate and replay size, RLVR/GRPO settings.
- **Frozen distributed eval cells.** A held-out set of cluster configs + workloads + **fault
  seeds** the proposer never sees or mutates; anomalous jumps re-evaluate on a second
  held-out set (the SPEC-4 §5.3 tripwire). *Critical (W7-specific):* the eval must fix the
  **consistency model and the tiered-oracle cost schedule**, or a proposer could "win" by
  silently checking a weaker model or a cheaper tier — a new reward-hacking surface the
  denylist must close.
- **Denylist (the judge is not a knob, `DD-AR2`).** The proposer cannot edit the oracle (any
  tier), the metrics, the goldens, the gate, `docs/distributed-semantics.md`, the consistency
  checker, or the tier-cost schedule.
- **The four irreducibles stay with the human** (SPEC-4 §8), restated: the **objective**
  (faithfulness; bits-to-correct down, the H17 tier mixture and H18 ratio understood), the
  **safety/ethics boundary** (defensive-only, sandbox, **no real-internet egress**,
  editable-path denylist, §15), the **kill-switch + resource cap**, and **promotion to main**.

The distributed world *strengthens* the autoresearch story twice over: the per-tier signal
is denser than a single scalar, and the DES-as-data-factory lets the engine manufacture the
exact harder world (more faults, weaker consistency) that the science needs — the cleanest
realization yet of "the engine builds its own curriculum, the oracle keeps it honest."

---

## 15. Safety & ethics

This subsystem simulates a running distributed system, closer to operational capability than
any prior toy. The posture (SPEC.md §13 / SPEC-3 §14 / SPEC-4 §9 / SPEC-5 §15 / SPEC-6 §15)
holds and tightens:

- **Defensive framing only.** Downstream use is autonomous **defense**, **change-safety**
  ("will this config push / failover / deploy break the cluster?"), SRE, incident response,
  and capacity planning — predicting the consequences of *your own* changes in a sandbox
  before they touch production (the τ-bench/AppWorld framing, §2.6, §7). Not offense, not
  exploitation, not third-party targeting. No exploit-bearing workload is a goal or
  deliverable; any environment encoding real attack dynamics (e.g., a consensus-protocol
  exploit) is reviewed before release and may be held back (SPEC.md §13).
- **No real-internet egress.** Tier-A is a pure simulator. Tier-B runs only under DST
  runtimes (madsim/Shadow/Antithesis-class) in sandboxes with egress disabled and
  self-contained, seeded workloads. The model never touches a system it does not own.
- **Reproducibility as a safety property.** Everything replays from `(seed, commit)`; no
  telemetry, no runtime network call (the repo-wide posture).
- **Denylist + kill-switch + resource cap** govern the engine (§14); goldens pin semantics so
  the engine cannot quietly repurpose them or weaken the consistency model it is judged
  against.
- **Dual-use note.** A faithful distributed-system simulator is dual-use; the denylist, the
  defensive task framing, and the sandbox-only tiered oracle are the structural mitigations.
  Consistency-faithfulness is modeled to make the simulator *trustworthy about failures*
  (a defender's need — does the model correctly predict that a partition causes a stale read
  or a write rejection?), not to model exploitation.

---

## 16. Open, decentralized, verified contribution (continued)

SPEC-6 §16 specified the protocol; SPEC-7 strengthens it, because the distributed world is
where the user's "open, freely-available, decentralized" intent has the most natural form
and the cleanest verification story (research methodology and community infrastructure only;
no service, no commercial path — SPEC.md §11 holds).

> **◐ shipped** ([`distcontrib/`](../../src/verisim/distcontrib/),
> [`test_distcontrib.py`](../../tests/test_distcontrib.py)): the distributed verified-contribution
> protocol, the dependency-free distributed analogue of the host [`contrib/`](../../src/verisim/contrib/).
> `verify_transition` / `verify_trajectory` accept a contributed `(state, action, next_state[,
> delta, observation])` iff re-running the deterministic oracle reproduces it — and add the one
> thing the host/network protocols could not need, the **tiered acceptance** (`tier=`): `bit_exact`
> demands byte-for-byte reproduction, while `metamorphic`/`cycle`/`symbolic` accept *any* next-state
> the cheap tier admits as legal under the declared model — the W7 path made concrete (a contributor
> running an equally-valid but byte-different schedule is admitted by the consistency tier where
> bit-exact rejects it; a genuinely illegal one — e.g. a read that mutates a replica — is still
> caught). Trajectories must also *chain* (`next_state[i] == state[i+1]`) so transitions cannot be
> spliced; `content_address` gives the corpus its tamper-evident SHA-256 manifest hash; hostile input
> is *rejected, never raised*. **Bug fixed in the build:** the round-trip `from_canonical(to_canonical(s))`
> was *not* exact — `partitions` was stored in the oracle's construction order while serialization
> sorted it, so a re-executed genuine trajectory spuriously refuted at the first non-sorted partition.
> Fixed at the source (canonical sorted partition order in `apply` + `DistributedState.__post_init__`),
> pinned by a round-trip test on a non-sorted partition. *(Deferred: the Tier-B real-DST re-execution
> path, and the §16 (b)/(c) contributable-oracle / golden-trajectory ingestion.)*

- **The oracle makes contributed cluster traces trustless by construction.** As in SPEC-6
  (vs Prime Intellect's TOPLOC heuristic, §SPEC-6 2.9), any contributed
  `(state, action, next_state)` or `(state, action, delta)` is verified by re-running the
  deterministic DES on `(seed, commit)` and comparing bit-for-bit — *or*, where bit-exact is
  intractable (W7), by checking the contributed history against the cheap consistency tier.
  A contribution is accepted iff it reproduces (or admits the declared consistency model).
- **What can be contributed.** (a) Oracle-verified cluster trajectories (expand coverage of
  the fault/consistency space — the §2.1 curriculum lever, the part DST coverage blind spots
  most need); (b) new tiered oracles for a subsystem behind the existing `O[tier]` protocol
  (a new symbolic checker, a new metamorphic invariant, a new consistency model); (c) golden
  trajectories pinning additional semantics (denylist review, §14). Each carries its manifest
  + content-hash (SPEC-2 §4, §12).
- **Why this is the defensible community contribution.** The artifact others get is not a
  model to compete with but a **free, exact (or consistency-checked) verification layer** for
  distributed-state world-model data, plus a benchmark/RL-env anyone can extend and have
  certified by the tiered oracle — the open, decentralized contribution the deterministic
  oracle (and, where it is intractable, the cheap consistency tier) makes uniquely possible.

---

## 17. Open questions

The v0/§17 discipline: record them, resolve them in the open.

1. **Semantics boundary.** How much of MVCC / isolation levels / a real consensus protocol
   does Tier-A pin before it overfits to a toy or becomes too costly to keep correct? (The
   single most important design call — the distributed analogue of SPEC-5 §17.1 / SPEC-6
   §17.1.)
2. **Tier-B determinism under partition.** How far madsim/Shadow/Antithesis-class runtimes
   seal asynchrony/partition timing (HW-5), and where a regime is irreducibly *stochastic* so
   the divergence metric must account for it (SPEC-5 §17.2 / SPEC-6 §17.2, harder here).
3. **`ε` across consistency classes.** What "ε-close" means when the target is a consistency
   *class*, not a state — a per-class threshold, a distance in the consistency hierarchy, or
   weakest-admissible? Tied to H19/H20.
4. **Tier-cost calibration.** The *dollar* cost of each oracle tier (§5) sets the whole H17
   result; how to price metamorphic vs cycle vs symbolic vs bit-exact honestly and
   reproducibly so the per-dollar comparison is fair.
5. **Coverage of the fault space.** DST samples faults (§2.1 skeptical note); how to measure
   and report the coverage blind spots a model inherits, and whether the engine's curriculum
   search (§14) closes them.
6. **Competitive-ratio rigor.** Where the learning-augmented guarantee (H18) is *provable*
   (bit-exact tier, exact fallback) vs only *empirical* (cheap tiers that refute but do not
   reconstruct) — and whether a cheap-tier fallback can be made provably drift-bounding.
7. **Counterfactual sampling.** Which fault-flip distribution produces counterfactuals that
   transfer (H5) — random flips, or targeted "near-miss" partitions / split-brain scenarios
   (the operationally decisive ones)?
8. **Plan-level loop (inherited, SPEC-5 §17.8 / SPEC-6 §17.8).** How to define divergence and
   `H_ε` for the LLM-integration case (§7) where the unit is a *plan* (a sequence of admin
   ops), not a single op.

---

## 18. Definition of done

SPEC-7 is done when:

1. DS0–DS6 ship, tested, with the deterministic core dependency-free, GPU-free.
2. The **distributed `H_ε(ρ)` curve (ED1)** is plotted once, cleanly, regenerable from
   config + seeds — *whatever it shows* — **and** the **tiered-oracle measurement (H17,
   §9.4)** is reported: does spending a budget across cheap + rare-expensive tiers beat
   spending it all on bit-exact truth? A flat interior or a no-tier-benefit result is a
   reportable result (the honest negative), not a failure.
3. **✅ shipped.** The honest write-up (the `docs/report.md` discipline) states, for each
   hypothesis in §10 (the operationalized H5/H8 and the new H17–H21), what was found and what the
   honest negative looked like — including the consistency-vs-bit horizon gap (H19), the
   `H_ε(consistency-level)` result (H20), the fault-injection transfer result (H21), and the
   competitive-ratio fit (H18) — in the distributed-world section of
   [`docs/report.md`](../report.md), each number read off its committed CSV.
4. **✅ shipped.** The distributed world is packaged for reuse (Inspect benchmark
   [`disteval/`](../../src/verisim/disteval/) + `verifiers`-spec distributed RL env
   [`distrl/`](../../src/verisim/distrl/) + the LLM-callable cluster-simulator protocol of §7
   [`distsim/`](../../src/verisim/distsim/) + the verified-contribution protocol of §16
   [`distcontrib/`](../../src/verisim/distcontrib/)), so the community can measure long-horizon
   faithfulness of a *running distributed system* with ground truth — at *tiered* cost,
   because full truth is intractable — the contribution of §1.4. All four pieces are
   dependency-free at their core (the Inspect adapter alone needs the optional `[eval]` extra).

The science is one curve, again — but for the first time it is the curve of a world where
the full oracle is *unaffordable*, so the question underneath it is no longer "how little
oracle can we get away with" but "which *price of truth* do we buy, and when." That is the
question that decides whether oracle-grounded world models reach the systems that actually
run the internet, or stop at the machines that host them.

---

## 19. Provenance and reading order

- **Prereqs:** [SPEC.md](./SPEC.md) (science), [SPEC-2.md](./SPEC-2.md) (filesystem v0),
  [SPEC-5.md](./SPEC-5.md) (network world), [SPEC-6.md](./SPEC-6.md) (host world). SPEC-7
  composes the network and host worlds into a distributed system and assumes their builds.
  Companions: [SPEC-3.md](./SPEC-3.md) (the depth roadmap; W1/W3/W4 and the
  speculative-execution framing extended here) and [SPEC-4.md](./SPEC-4.md) (the engine; §14
  extends it).
- **Lessons grounding this spec** (name + venue + year, per
  [`docs/related-work.md`](./docs/related-work.md)'s no-fabricated-links policy; arXiv IDs
  only where independently verified — deliberately omitted where not): deterministic
  simulation testing — FoundationDB (Will Wilson, Strange Loop 2014; SE-Radio 685, 2025),
  TigerBeetle VOPR (2023), Antithesis deterministic hypervisor (2024), madsim / turmoil,
  Shadow (USENIX-pedigree); consistency checking — Elle (Kingsbury & Alvaro, VLDB 2020),
  mixed-isolation complexity (CAV 2025), VerIso (PVLDB 2025), weak-isolation separation logic
  (ICFP 2025); metamorphic DB oracles — SQLancer PQS/NoREC/TLP (Rigger & Su, OSDI/ESEC-FSE
  2020), graph-based transactional oracle (Jiang et al., OSDI 2023); formal protocol oracles —
  Apalache (OOPSLA 2019), Stateright, the P language; learning-augmented algorithms (learned
  indexes with error-bounded fallback; LARU; MAT); world-model-as-tool — WebDreamer
  (Gu et al., 2024; NAACL 2025), CWM / Code World Model (Meta FAIR, 2025); executable
  environments — τ-bench / τ²-bench (Sierra, 2024–2025), AppWorld (ACL 2024), ToolEmu
  (ICLR 2024), SWE-Gym (2024), R2E-Gym (2025); architecture — m4 / RouteNet bipartite GNNs,
  DreamerV3 RSSM, Mamba/SSD (Gu & Dao, 2023–2024), CLRS / TransNAR, emergent program
  semantics (Jin & Rinard, ICML 2024); the SLM thesis (NVIDIA / Belcak et al., 2025); and the
  in-house sibling repos `worldify` (temporal-causal fact store → the state/belief model,
  §2.5) and `securifine` (differential severity-weighted safety eval → the change-safety
  evaluation, §2.5). *(These should be added to `docs/related-work.md` when this spec moves
  from design to build.)*
- **Author:** Clay Good. **License:** MIT. The distributed oracle runs only inside the
  sandbox; no real-internet egress, no telemetry, defensive framing (§15).
- A living spec: as milestones (§13) land they are marked and their figures linked (mirroring
  SPEC-2 §13 / SPEC-5 §13 / SPEC-6 §13); as a hypothesis (§10) is tested its result is
  recorded in SPEC.md §9. The spec is the record of what we believed and what we learned.
