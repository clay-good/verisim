# SPEC-5 — The Network World

**Engineering & experiment specification for the multi-host, partially-observable
computer-network world: the next smallest world with the *hard* property.**

> **▶ ACTIVE — NW0–NW6 shipped (2026-05); the prime-directive curve is plotted.** SPEC-2.1
> refuted the knee on the single-filesystem world (discrete errors), which **licenses this
> network world** — where drift is gradual and observation is partial (SPEC §12). The
> **deterministic core NW0–NW3 is built and tested** (no GPU): the typed-graph reachability world
> ([`net/`](../../src/verisim/net/)), the Tier-A reference oracle
> ([`netoracle/`](../../src/verisim/netoracle/)), graph deltas
> ([`netdelta/`](../../src/verisim/netdelta/)), drivers ([`netdata/`](../../src/verisim/netdata/)),
> and metrics ([`netmetrics/`](../../src/verisim/netmetrics/)) — see [`docs/network-semantics.md`](../network-semantics.md)
> and §13. **NW4's supervised model ships:** a from-scratch, grammar-constrained network `M_θ`
> ([`netmodel/`](../../src/verisim/netmodel/)) that drops into the loop exactly as v0's did.
> **NW5 now ships the partial-observation loop** ([`netloop/`](../../src/verisim/netloop/)): the
> two-mode (full / probe) partial-observation oracle (§5.3), probe policies `π_o` (§8.2),
> correction/belief operators (§8.3), baselines, and the model-agnostic runner.
> **NW6 plots the headline EN1 `H_ε(ρ)` curve** ([`experiments/en1.py`](../../src/verisim/experiments/en1.py),
> [`figures/en1_curve.png`](../../figures/en1_curve.png)): on the flat-Markov `M_θ` the
> network interior is **near-flat (the H8 honest negative)** — the network analogue of v0's
> H1 floor (§10.1, [report](../report.md)). **NW7 is underway on the flat arm:** EN2
> ([`en2.py`](../../src/verisim/experiments/en2.py)) compares consultation policies `π_c`
> (H9 — the uncertainty-triggered policy *leads* but with overlapping CIs, suggestive not
> conclusive, unlike v0's clean negative), and EN3 ([`en3.py`](../../src/verisim/experiments/en3.py))
> shows the partial-observation payoff directly: the full-consult operators coincide (the v0
> identity) **but the cheap one-host probe + belief filter breaks that collapse** and earns
> **~2.3× more faithful horizon per oracle-bit** (§8.3, §9.4). **Remaining NW7:** the
> message-passing + RSSM graph arm (H11), the smart information-gain probe policy `π_o` (H10),
> the drift mitigations (noise injection, self-forcing; the GNS/m4 levers v0 lacked), and EN4
> — the levers that must lift `M_θ` off the EN1 floor.

This is to the network what [SPEC-2](./SPEC-2.md) is to the filesystem. SPEC-2 made
the single-host shell/filesystem world buildable and proved the *method* —
propose-verify-correct against a free deterministic oracle. SPEC-5 makes the
**network** buildable, because the filesystem world was too easy for the central
science to bite. It promotes the network sketch in [SPEC-3](./SPEC-3.md) (wall **W3**:
"the world is tiny and fully observable") from a paragraph to a buildable program,
realizes it under the autonomous research engine of [SPEC-4](./SPEC-4.md), and folds
in the world-model, small-language-model, and learned-network-simulator lessons that
2024–2026 made available.

It does not restate the science. Read [SPEC.md](./SPEC.md) first for *why* oracle-grounded
world models of computer environments are the one domain where long-horizon
faithfulness is measurable. This document is *how* — for networks.

---

## 0. Prime directive

> **SPEC-5's only job is to plot the network `H_ε(ρ)` curve once, cleanly, in a world
> hard enough that the interior is informative — and to report honestly if the
> favorable knee still does not appear.**

Everything below serves that one figure. SPEC-2's prime directive was the same
sentence with "filesystem" in place of "network." The difference is the world: v0's
`H_ε(ρ)` sat on the floor because the world was fully observable, one-step, and
low-entropy (see the honest E1/E2/E4 negatives in [docs/report.md](./docs/report.md)).
The network world is engineered to be exactly the regime where pure-neural rollouts
drift fast **and** oracle truth is expensive — the regime where SPEC.md's hypothesis
H1 (a favorable curve exists) can finally be tested under load.

This spec is large because it is exhaustive by request, but the program is staged
(NW0–NW8, §13) so that the deterministic core ships and is testable before any GPU is
touched, exactly as M0–M3 did in v0.

---

## 1. Why the network world, and why now

### 1.1 What v0 could not show, and why

The v0 world has three properties that, together, keep `H_ε(ρ)` on the floor:

1. **Full observability.** The whole filesystem is in the state; nothing is hidden.
   There is no belief to maintain, so a smart consultation policy has nothing to be
   smart *about*. This is why E2's `uncertainty_triggered` lost to `fixed` — the
   model's uncertainty signal had nothing latent to track (the calibration diagnostic,
   SPEC-2 §7.2, measured Pearson ≈ 0.11).
2. **One-step, synchronous effects.** A command's entire effect lands in a single
   transition. There is no delayed consequence, no convergence, no retransmit. This is
   why E3's three correction operators *collapsed to an identity*: with a full-state
   one-step oracle, every operator snaps the coupled state to the same `s'`.
3. **Low branching, low entropy.** The drift per step is small, so the gap between the
   `ρ→0` floor and the `ρ→1` ceiling is small, so there is no informative interior to
   buy with cheap consultation.

These are not bugs in the method. They are properties of *the smallest world*. The
method is correct; the world is too easy.

### 1.2 The network world removes all three at once

| v0 property | Network world | Consequence for the science |
|---|---|---|
| Fully observable | **Partially observable** by construction (an operator sees NetFlow / SNMP / pcap samples, never global truth) | Belief-state estimation matters; smart sensing can beat dumb (recovers H2) |
| One-step, synchronous | **Asynchronous, temporally extended** (route convergence, TCP retransmit, timer expiry, in-flight packets) | Correction operators genuinely differ; drift compounds over many steps (stresses H1, H3) |
| Low entropy | **Combinatorial reachability** over a graph of hosts, routes, ACLs, flows | Large floor-to-ceiling gap → an informative `H_ε(ρ)` interior to measure |

Crucially, **the deterministic oracle stays free.** A discrete-event network
simulator is reproducible given a seed; Linux network namespaces with a pinned kernel,
deterministic traffic generators, and no real-internet egress are replayable truth.
The one property that makes computer environments unique (SPEC.md §2) survives the
jump to networks — that is the entire reason to go here next rather than to vision,
biology, or robotics (those domains are named in the user's long-horizon roadmap, but
they have no free oracle; networks do).

### 1.3 Why it is worth building for the community, not just for us

A deterministic, oracle-grounded, partially-observable network world is a **reusable
benchmark the field does not have**: a place to measure long-horizon faithfulness of
world models *with ground truth*, on graph-structured state, under partial
observability. Learned network simulators exist (§2.2) but are evaluated on aggregate
performance metrics (throughput, latency error), never on *faithful horizon under a
consultation budget*. Verisim's contribution is the metrology, packaged where
researchers already look (an Inspect benchmark and a `verifiers`-spec RL environment,
as in SPEC-2 §15 / M8). This is the non-competitive contribution: not a bigger model,
but a free, honest measuring instrument and the cheap faithful simulator that other
agents can call.

---

## 2. Lessons folded in (and the design choice each one forces)

This section is the "browse far and wide" deliverable: each lesson is stated as design
guidance with its source, so every non-obvious choice in §3–§9 traces to evidence.
Skeptical notes are included where the literature is hype.

### 2.1 Small / specialist models (SLM)

- **NVIDIA, "Small Language Models are the Future of Agentic AI" (Belcak et al., 2025).**
  For narrow, repeated, verifiable tasks, a small specialist beats a large generalist
  on cost, latency, and reliability; the right architecture routes most calls to small
  specialists and escalates rarely.
  → **Design choice:** `M_θ` is deliberately **small and specialized** — a fast
  faithful *network* simulator, not a generalist. Verisim's edge is being the cheap,
  verifiable specialist an LLM agent calls, not a competitor to the LLM.
- **Distillation with a verifiable teacher.** Distillation usually loses signal because
  the teacher is itself fallible. Here the "teacher" is a *perfect deterministic
  oracle* that emits unlimited correctly-labeled trajectories.
  → **Design choice:** treat the oracle as an infinite distillation source (§5, §8.4).
  This is the rare regime where you have ground truth at scale.
- *Skeptical note:* the SLM thesis is partly positioning. The defensible core — small
  specialists win on verifiable narrow tasks — is exactly verisim's case, so we lean
  on the defensible core only.

### 2.2 Learned network simulators & network digital twins

- **ns-3 / OMNeT++** are deterministic discrete-event simulators — i.e. *exactly the
  oracle shape we need*. → **Design choice:** the Tier-A oracle (§5.1) is a from-scratch
  DES of a pinned semantics; the Tier-B oracle (§5.2) can wrap a real DES or namespaces.
- **RouteNet / RouteNet-Erlang (GNN models of network performance)** show graph neural
  nets predict per-path delay/jitter and *generalize across topologies* — but
  generalization across topology/scale is the known failure mode of the whole subfield.
  → **Design choice:** topology generalization is a first-class experiment axis (EN4)
  and open question (§17), not an afterthought; the curriculum (§3.4) sweeps topology.
- **MimicNet, DeepQueueNet, m3** accelerate packet-level simulation with learned
  components but are evaluated on *aggregate* fidelity, never faithful horizon under a
  verification budget. → **Design choice:** that gap is the contribution (§1.3).
- **m4 (2025) — the closest validated analog.** A flow-level learned simulator trained
  on ~4,000 ns-3 runs of small (~32-host) topologies that generalizes to ~49k-host
  fat-trees, using a spatial-temporal split: a **bipartite "entity ↔ shared-resource"
  GNN** (flows/connections ↔ links/queues) plus recurrent updates between discrete
  events, with **dense intermediate supervision** (remaining flow size, queue length),
  not just end-to-end answers. Cautionary detail: it **one-hot-encodes** the
  congestion-control protocol, so it *cannot* predict unseen protocols. → **Design
  choices:** adopt m4's bipartite + recurrent decomposition as `M_θ`'s template (§6.1);
  supervise densely on FIB entries / link occupancy / connection-state transitions, not
  only reachability; and treat anything one-hot (CC algorithm, ACL action, OS) as a hard
  generalization wall — prefer feature-based encodings that interpolate.
- **Batfish — a free symbolic control-plane oracle.** Batfish deterministically computes
  RIBs/FIBs and ACL/reachability behavior from device configs, with correctness
  guarantees. → **Design choice:** use it as a **second oracle** for the *control plane*
  (routing/firewall/reachability) alongside the DES / namespaces for the *data plane*
  (§5.1) — neuro-symbolic grounding obtained without training anything, and the basis of
  hypothesis **H12**.

### 2.3 Graph / mesh learned simulators

- **DeepMind GNS ("Learning to Simulate Complex Physics with Graph Networks") and
  MeshGraphNets.** Message-passing over a graph rolls structured state forward; the
  single most important trick for long rollouts is **training-time noise injection** on
  inputs so the model learns to correct its own drift.
  → **Design choice:** `M_θ` is a message-passing predictor over the network graph
  (§6.1), and **noise-injected rollout training** is a required ablation lever (EN4),
  the lever v0 lacked.

### 2.4 World models & long-horizon drift

- **DreamerV3 (RSSM).** A recurrent latent state model that *plans in imagination*;
  the latent carries information the single observation does not.
  → **Design choice:** `M_θ` carries an **RSSM-style latent belief** over the
  unobserved subgraph (§6.2), enabling partial observability and imagination-time
  planning for the LLM-integration case (§7).
- **JEPA / V-JEPA 2.** Predict in *latent space*; do not reconstruct raw observations
  (pixels — here, raw packets). Reconstruction wastes capacity on irrelevant detail.
  → **Design choice:** the model predicts structured *deltas* and latent belief, never
  raw packet bytes. (This also matches v0's E4 finding that the delta representation
  dominates full-state.)
- **Diffusion forcing / scheduled sampling.** Mixing teacher-forced and free-run
  prediction during training reduces autoregressive exposure bias.
  → **Design choice:** scheduled-sampling / free-run mixing is a drift-mitigation lever
  (EN4), alongside noise injection.
- **Genie (action-conditioned world models).** Conditioning the predictor on the action
  is what makes the world *controllable*. → **Design choice:** every prediction is
  conditioned on the action *and* on a clock/event token for asynchrony (§6.1).

### 2.5 Partial observability & belief states

- **POMDP belief-state estimation; particle filters; RSSM as an amortized filter.**
  When you see part of the state, maintain a distribution over the rest.
  → **Design choice:** a partial-observation oracle (§5.3) returns cheap probes
  (ping/traceroute/flow sample) or expensive full truth; the model maintains a belief
  and the loop adds a **probe-selection** axis (what to observe), not just a
  consultation axis (when to observe) — §8.2.
- **Network tomography** is the classical, grounded way to infer unobserved internal
  link state (delay/loss) from end-to-end edge measurements — typically an
  *underdetermined* linear system (path metric = sum of link metrics) solved by sparse
  recovery. → **Design choices:** frame observation ingestion (NetFlow/SNMP/pcap/BGP) as
  tomography, bake the additive structure in as inductive bias, and accept that some
  links are *unidentifiable* — model bounds, not false certainty. And do **not** use a
  diagonal-Gaussian belief (the literature is explicit it is too weak for real
  multi-modal network uncertainty); use a categorical/particle/flow-based belief.

### 2.6 Test-time training / online adaptation

- **Test-time training (TTT) and online continual learning (2024–2026).** A model can
  take gradient steps on a self-supervised or verifiable signal *at deployment*.
  → **Design choice:** every oracle consultation is a free, perfectly-labeled example;
  the **self-healing loop** (§8.4) takes a gradient step on each correction during the
  rollout. This realizes SPEC-3's wall **W4** ("the model is static") for the network
  world, in a world long enough for it to matter.

### 2.7 Verifiable rewards, reward hacking, and autoresearch

- **RLVR / GRPO and Karpathy's autoresearch (≈630-line keep-if-better loop, 2026).**
  Optimization against a *verifiable* reward, with the human out of the loop, works only
  if the reward is unfakeable. → **Design choice:** the network oracle is unfakeable —
  you cannot lower **bits-to-correct** (§9.4) or raise **reachability-faithfulness**
  (§9.2) without actually predicting the truth — so SPEC-4's ratchet extends directly
  (§14). The denser, graded network reward should lift RLVR off the H1-floor null that
  v0 reported.

### 2.8 Computer/network agent benchmarks

- **OSWorld, terminal-bench, CTF / incident-response / SRE tasks.** The community values
  long-horizon, multi-step computer tasks with checkable outcomes; network change-safety
  and autonomous *defense* are high-value, under-served, and unambiguously defensive.
  → **Design choice:** the downstream framing (§15) is defensive — change-safety, SRE,
  autonomous network defense — never offense.

---

## 3. The world (environment)

### 3.1 State as a typed graph

The network state `s` is a typed, attributed multigraph plus a small global block. This
replaces v0's flat `(filesystem, cwd, env, last_result)` record with a graph, because
the domain *is* a graph and graph structure is what the model exploits (§2.3).

**Node types** and their per-node state:

| Node type | State attributes (illustrative, pinned in `docs/network-semantics.md`) |
|---|---|
| `host` | id, interfaces, up/down, FIB (routing table), ARP/neighbor table, conntrack (connection table), service set |
| `router`/`switch` | id, interfaces, FIB, ACL/firewall ruleset, up/down |
| `interface` | id, owner node, address(es), link, up/down, counters (rx/tx bytes, packets) |
| `service` | id, owner host, port/proto, listening/up/down |
| `flow`/`conn` | id, 5-tuple, TCP/conn state (e.g. `SYN_SENT`,`ESTABLISHED`,`CLOSED`), seq/ack summary, bytes |

**Edge types:** `link` (with capacity, base latency, loss), `route` (next-hop in a FIB),
`rule` (an ACL/firewall entry binding a node to a match/action), `member` (service→host,
interface→node). The graph is canonicalized (sorted, hashed) exactly as v0 canonicalizes
the filesystem, so serialization is deterministic and divergence (§9.1) is well-defined.

**Global block:** simulated clock, RNG seed/cursor, pending event queue summary, and a
`last_result` (command stdout/exit) mirroring v0.

**Derived (not stored, computed by the oracle):** the **reachability matrix** `R[a][s]`
(can host `a` reach service `s`?), shortest path, RTT estimate, throughput estimate.
These are the operationally meaningful quantities and the basis of
reachability-faithfulness (§9.2).

### 3.2 Actions

Two families, both expressed in a constrained grammar paired with the oracle, exactly as
v0 pairs a shell grammar with `ReferenceOracle`:

1. **Config / admin ops** (the "commands"): `ip addr add/del`, `ip route add/del`,
   `ip link set up/down`, firewall `rule add/del` (nftables-style), `service up/down`,
   NAT/`masquerade` add/del.
2. **Traffic / time ops** (the source of asynchrony): `connect a→s`, `send n bytes`,
   `close`, and `advance Δt` (deliver in-flight packets, fire timers: TCP retransmit,
   route convergence, ARP timeout).

Actions are the controllable input; `advance Δt` is what makes effects temporally
extended and is the single most important difference from v0.

### 3.3 Determinism contract

The world is deterministic given `(initial topology, seed)`. Tier-A is a pure function.
Tier-B (namespaces) pins kernel version, container image hash, traffic-generator seeds,
and disables real egress; residual scheduler nondeterminism is handled per §17.2.

### 3.4 Scale & curriculum

Difficulty is a small set of dials (mirroring v0's difficulty axis): number of hosts,
number of links, topology family (line → star → tree → mesh → datacenter fat-tree →
random `G(n,p)`), ruleset complexity, traffic intensity, and `advance` granularity.
The curriculum starts at **2 hosts, 1 link, 1 flow** (the network analogue of v0's
smallest world) and ratchets up. Topology generalization (train on some families, test
on held-out families) is an explicit axis (EN4), because it is the field's known
failure mode (§2.2).

---

## 4. The state delta `Δ`

`M_θ` predicts a structured **graph delta**, not a full state and not raw packets (§2.4,
and v0's E4 result that deltas dominate). The delta vocabulary mirrors v0's
`Create/Delete/Modify/Move/Chmod/SetCwd/SetEnv/SetResult` with network types:

```
RouteAdd / RouteDel / FibUpdate        # routing table edits
NeighborUpdate                          # ARP/neighbor table
RuleAdd / RuleDel                       # firewall/ACL edits
IfaceUp / IfaceDown / AddrAdd / AddrDel # interface/address state
ServiceUp / ServiceDown                 # service state
ConnOpen / ConnState / ConnClose        # connection/flow state machine transitions
CounterUpdate                           # rx/tx bytes/packets deltas
ClockAdvance                            # global clock move
EventFire                               # a timer/in-flight delivery fired
SetResult                               # command stdout/exit (as in v0)
```

The **M1-analogue invariant** is required and tested: `apply(state, oracle.delta) ==
oracle.next_state` for every transition, by construction. Delta↔serialization is
round-trippable. This invariant is the contract that lets the loop (§8) stay
model-agnostic, exactly as in v0.

---

## 5. The oracle `O`

Two tiers, mirroring SPEC-3's system-oracle plan (Tier A / Tier B), specialized to the
network.

### 5.1 Tier A — reference network oracle (the deterministic core)

A **from-scratch deterministic discrete-event simulator** of a pinned semantics:

- IPv4, longest-prefix-match routing over per-node FIBs.
- Stateless + stateful firewall (5-tuple match, conntrack).
- A simplified but real TCP state machine (handshake, data, teardown, retransmit timer).
- A fixed link model (capacity, base latency, optional loss) and an event queue advanced
  by `advance Δt`.

It has **no runtime dependencies and needs no GPU**, like v0's M0–M3 core. It is the
executable truth, paired with a normative `docs/network-semantics.md` (the analogue of
`docs/semantics.md`). Golden trajectories pin the semantics and are denylisted from the
autoresearch engine (§14).

**Two planes, two oracles.** The DES above is the *data-plane* oracle (what packets
actually do). A **control-plane** oracle — a Batfish-style symbolic solver that computes
the FIB/RIB and ACL/reachability truth directly from the configuration (§2.2) — runs
beside it. The data-plane oracle is the ground truth for flows, queues, and counters; the
control-plane oracle is the ground truth for reachability and routing. Both are
deterministic and free; consulting them is two different (and differently priced) oracle
calls, and whether the control-plane oracle is a *non-redundant* signal is hypothesis
**H12** (§10).

The pinned subset is deliberately *small enough to be correct and replayable, large
enough to make the science bite* — choosing that boundary is open question §17.1.

### 5.2 Tier B — system oracle (reality check)

Real **Linux network namespaces** (`ip netns`), real kernel routing, nftables, services,
and deterministic traffic generators, captured via pcap/conntrack. Pinned kernel +
image hash; **no real-internet egress** (§15). Tier B is how we attack SPEC-3's wall
**W1** ("the oracle is a model, not reality") for the network domain: it measures the
gap between Tier-A semantics and a real kernel, which is itself a reportable result.

### 5.3 Partial-observation oracle

Unlike v0, the oracle exposes **two consultation modes**:

- **Probe (cheap):** a localized observation — `ping`, `traceroute`, a NetFlow/SNMP
  sample, a single host's view of its FIB/conntrack. Returns partial truth at low cost.
- **Full (expensive):** the complete next state, as in v0.

This is what makes partial observability native rather than bolted on, and it creates
the new probe-selection axis (§8.2).

### 5.4 Bits-to-correct over graph deltas

The scale-free gate metric from SPEC-3/SPEC-4 generalizes directly:
**`bits-to-correct(Δ̂, Δ)` = the MDL of the oracle's correction of the model's predicted
graph delta** — 0 iff the prediction equals truth, smooth and unfakeable otherwise. It
is the autoresearch gate (§14) and a per-step diagnostic.

---

## 6. The model `M_θ`

> **Build status (NW4).** The **flat** arm of `M_θ` ships now
> ([`netmodel/`](../../src/verisim/netmodel/)): a from-scratch decoder-only transducer over
> the serialized `(state, action) → Δ` sequence, with a closed token vocabulary, an LL(1)
> graph-delta grammar, grammar-constrained greedy decode, and supervised training — reusing
> v0's transformer (`GPT`) and trainer (`train_supervised`) unchanged, because the
> example/padding shapes are identical. It is the H11 *flat-Markov baseline* and the NW5-loop
> drop-in (`predict_delta`) — it runs in the shipped NW5 loop and plots the NW6 EN1 curve. The
> **message-passing encoder (§6.1)** and **RSSM belief (§6.2)** land at **NW7**, where the
> probe-driven partial observability makes the belief non-degenerate (§6.2): under full
> observability it degenerates to exactly this Markov predictor, so building it earlier would
> be unused machinery; the graph-vs-flat comparison it enables is EN4 (§12).

### 6.1 Architecture

A **message-passing predictor over the network graph**, action- and clock-conditioned,
emitting a structured delta under grammar-constrained decoding (the constrained-decode
machinery from v0's M4 carries over). Message passing is chosen for the reasons in §2.3;
constrained decoding guarantees every prediction is a *valid* delta, as in v0.

Concretely, follow **m4's validated template** (§2.2): a **bipartite "entity ↔
shared-resource" graph** (flows/connections on one side, links/queues/firewalls on the
other) with message passing for the spatial coupling and a recurrent update between
discrete `advance Δt` events for the temporal dynamics. Message-passing depth is tuned to
the network diameter, since a firewall change three hops away cannot be represented with
too few steps (§2.3). Large-magnitude, heterogeneous quantities (byte counters vs. binary
flags vs. small integers) are passed through a **symlog transform** (DreamerV3, §2.4) so
one model handles scales differing by orders of magnitude with fixed hyperparameters.

### 6.2 Latent belief (RSSM)

A recurrent latent carries a **belief over the unobserved subgraph** (§2.5). Under full
observability it degenerates to v0's Markov predictor; under partial observability it is
the only way to roll forward state the model cannot see. The belief's variance is the
*calibrated-by-construction* uncertainty signal that EN2 tests (§12) — the thing v0
lacked when its decode-entropy signal failed calibration (SPEC-2 §7.2).

### 6.3 Drift mitigations (required ablation levers)

- **Noise-injected rollout training** (GNS, §2.3) — the single highest-leverage,
  cheapest drift mitigation: corrupt input state during training so the training input
  distribution matches the noisy distribution the model eats during oracle-free rollout.
  Noise scale `σ` is a key hyperparameter, not a default (too much hurts one-step
  accuracy, too little fails to cover drift).
- **Self-forcing / scheduled sampling** (§2.4): roll the model out on *its own* outputs
  during training, not pure teacher forcing, to close the train/deploy exposure-bias gap.
  Noise injection and self-forcing attack the same problem from two angles; both are levers.
- **Multi-step (latent-overshooting) objective** (RSSM/PlaNet, §2.4): penalize compounding
  error directly by training k-step-ahead predictions, not only one-step.

Each is an on/off lever in EN4 so we can attribute any horizon gain to a specific
lesson rather than to unattributed tuning.

### 6.4 Size & specialization (SLM)

`M_θ` is small and network-specialized by design (§2.1). The EN4 size ablation tests
whether horizon is capacity-bound (v0's answer was *no* at its scale; the network world
re-asks the question where the world is harder).

---

## 7. SLM/LLM complementarity (the world model as a callable simulator)

This section answers the user's "complement/integrate SLM/LLM" intent directly. The
world model is not a competitor to LLMs; it is the **cheap, faithful, verifiable
simulator an LLM agent calls.**

- **World model as a tool.** An LLM agent proposes a *plan* (a sequence of network
  changes — "add this route, tighten this ACL"). `M_θ` simulates the consequences fast;
  the oracle verifies on a budget. This is propose-verify-correct lifted from the *step*
  level (v0) to the *plan* level, and it is Dreamer's "plan in imagination" (§2.4) made
  honest by a real oracle.
- **Distillation.** The oracle emits unlimited perfectly-labeled trajectories → distill
  into the SLM (§2.1). Ground truth at scale is the unfair advantage of this domain.
- **Routing / speculative execution.** `M_θ` drafts the dynamics; the oracle verifies
  only when the consultation policy fires. This is *speculative decoding as a
  consultation policy* (SPEC-3's framing), and `ρ` is precisely the verification rate.
  The LLM is escalated to only for natural-language intent → action translation, not for
  simulating dynamics it is bad at.
- **Verifiable-reward training.** The SLM is trained against the oracle's
  faithful-horizon / bits-to-correct reward (RLVR, §2.7). Unfakeable by §5.4.

The integration is a *protocol*, specified here and built in NW8: a `Model` that
implements both "predict next state" (for the loop) and "simulate a plan" (for an LLM
caller), packaged like v0's `eval` and `rl` modules so external agents can call it.

---

## 8. The loop (partial-observation propose-verify-correct)

Same skeleton as SPEC.md §5.2, with three network-specific extensions.

```
for each step t:
    Δ̂ ← M_θ(belief, a_t)            # PROPOSE: predict the graph delta
    ŝ' ← apply(ŝ, Δ̂); belief ← update(belief, Δ̂)
    if π_c decides to consult (budget ρ):           # WHEN to consult
        o ← O(s, a_t, mode = π_o(belief))           # WHAT to observe (§8.2)
        d ← divergence(o, ŝ')                       # partial or full
        ŝ', belief ← C(ŝ', belief, o)               # CORRECT / belief-update (§8.3)
        θ ← θ − η·∇ loss(Δ̂, o)        # SELF-HEAL: free TTT step (§8.4, optional)
    s ← O.true_next(s, a_t)                          # the world advances regardless
```

### 8.1 Consultation policy `π_c` (when)

The v0 policies carry over: `fixed(k)`, `drift_triggered(τ)`, `uncertainty_triggered(τ)`,
`learned`. Now `uncertainty_triggered` reads the *belief variance* (§6.2), which is
calibrated by construction — the fix for the H2 negative.

### 8.2 Probe policy `π_o` (what) — new axis

Given a consultation, *what* to observe: random probe, uncertainty-targeted probe,
or **information-gain** probe (observe the subgraph that most reduces belief entropy).
This is active sensing / optimal experiment design, and it has no v0 analogue. It is the
heart of EN2.

### 8.3 Correction / belief operators `C`

`hard_reset` (snap the observed subgraph to truth), `residual`, `projection` (onto the
nearest *reachability-consistent* state), and **belief-filter** (Bayesian/particle update
of the unobserved subgraph from a partial observation). Unlike v0 (where operators
collapsed to an identity), partial observability makes these genuinely different — EN3.

### 8.4 Self-healing online TTT (optional, gated)

Each consultation is a free labeled example; take a gradient step to heal drift mid-
rollout (§2.6). This is SPEC-3 wall **W4** realized. It is measured by *faithful horizon
per oracle-bit* (does healing pay for its compute?) in EN5, and it is off by default so
the static-model baseline is clean.

---

## 9. Metrics

### 9.1 Graph divergence `d(s, ŝ)`

Normalized symmetric difference over typed node/edge tuples (the graph analogue of v0's
filesystem set-difference), **plus** the Hamming distance of the reachability matrix
`R`. `d = 0` iff the graphs and reachability agree. `ε` and what counts as "ε-close" for
`R` is open question §17.3.

### 9.2 Reachability-faithfulness

The operationally meaningful metric: over the horizon, does the model correctly predict
`R[a][s]` (can-A-reach-S)? This is what a defender or SRE actually needs, and it is
graded as a first-class number alongside `H_ε`.

### 9.3 Faithful horizon `H_ε(ρ)`

Unchanged in definition (max steps within `ε`), now over distributed state. This is the
headline (EN1, NW6).

### 9.4 Bits-to-correct, probe efficiency, calibration

- **Bits-to-correct** (§5.4): the scale-free gate.
- **Probe efficiency:** faithful horizon per oracle-bit and per probe — the network
  enrichment of `ρ`.
- **Belief calibration:** does belief variance predict error? (Extends SPEC-2 §7.2;
  this is the diagnostic that decides whether EN2's smart sensing *can* work.)

---

## 10. Hypotheses

SPEC-5 is, first, the place where four **already-stated** hypotheses finally become
testable, because the network world is the regime they were written for; and second, it
adds three **new** hypotheses (H10–H12) unique to a partially-observable, multi-oracle
network world. Each is falsifiable; each names what an **honest negative** looks like,
per SPEC.md §9 ("the favorable curve might not exist").

### 10.1 Existing hypotheses this spec operationalizes (do not re-coin)

- **H8 (SPEC-3 §13) — the interesting interior lives in networks.** The multi-host
  world exhibits H1's favorable knee (≥80% of ceiling horizon at ≤20% consultation)
  where the single-FS world did not. **This is SPEC-5's headline** (EN1, NW6).
  *Honest negative:* the network interior is also flat/linear → the knee is not about
  world hardness but about model/metric design; report it and move to EN4/EN5.
- **H7 (SPEC-3 §13) — correction teaches online.** Self-healing (§8.4) reduces
  divergence *after* corrections within a single rollout, beating `hard_reset` — tested
  here in a world long enough for it to bite (EN5).
- **H9 (SPEC-3 §13) — speculative scheduling dominates.** Consultation policies derived
  from acceptance-rate / belief-variance estimates beat fixed-interval at equal budget,
  now with the calibrated-by-construction signal partial observability supplies (EN2).
- **H5 (SPEC.md §9) — counterfactual lift.** Oracle-grounding improves interventional
  fidelity over an oracle-free model on identical data, trained on branch-replay
  counterfactuals the deterministic oracle makes free (EN6).

### 10.2 New hypotheses (H10–H12, non-colliding with H1–H9)

- **H10 — what-to-observe beats when-to-consult-alone (active sensing).** Under partial
  observability, an **information-gain probe-selection** policy `π_o` (§8.2) — choosing
  *which* cheap probe to spend — lifts faithful horizon per oracle-bit above any
  when-only consultation policy (`π_c`) holding the probe budget fixed. This axis has no
  v0 or SPEC-3 analogue: H2/H9 schedule *when* to verify; H10 is the first hypothesis
  about *what* to look at.
  *Honest negative:* probe selection adds nothing over uniform probing → the belief's
  uncertainty does not localize where truth is most valuable.
- **H11 — graph + belief beats flat-Markov.** A message-passing predictor over the
  network graph with an RSSM belief latent dominates a flat-sequence Markov predictor on
  `H_ε` at matched compute — the network analogue of v0's "delta dominates full-state"
  (E4), and the GNS/m4 bet (§2.2–§2.3) that structure is the lever once the world is a
  graph.
  *Honest negative:* no gap → structure/belief is not the lever at this scale; a flat
  serializer suffices.
- **H12 — two-oracle grounding lowers bits-to-correct.** Adding a symbolic
  **control-plane** oracle (Batfish-style reachability/FIB/ACL truth, §5.1) on top of
  the **data-plane** oracle (DES / namespaces) lowers bits-to-correct more than either
  oracle alone at equal consultation budget — i.e. cheap, exact reachability supervision
  is a strong free signal the data-plane oracle does not localize.
  *Honest negative:* the control-plane oracle is redundant given the data-plane oracle →
  reachability is already implied by the state the model predicts.

---

## 11. Walls (relative to SPEC-3)

SPEC-5 is the buildable assault on SPEC-3's wall **W3** ("the world is tiny and fully
observable") for the network domain. It also makes **W1** (oracle-is-a-model) concrete
via Tier-B namespaces (§5.2) and **W4** (static model) concrete via online TTT (§8.4).

It adds one genuinely new wall to SPEC-3's taxonomy:

- **W5 — effects are synchronous and one-step.** v0's one-step oracle hid all delayed
  dynamics. The network world's `advance Δt`, route convergence, and retransmit timers
  make effects asynchronous and temporally extended. W5 is what gives correction
  operators something to differ on (§8.3) and what makes long-horizon drift compound.

---

## 12. Experiments (EN-series)

Non-colliding with E1–E4 and with the *reserved* E5/E6 (Phase-1 system oracle / Phase-3
counterfactuals, SPEC-2 §9). The network suite is its own namespace, **EN1–EN6**. Each
mirrors a v0 experiment's role and names the hypothesis it tests (§10).

- **EN1 — the network `H_ε(ρ)` curve** (role of E1; the prime directive, NW6). Sweep
  `ρ × ε × difficulty`. Bootstrap-CI aggregation. *Does the knee appear — **H8**?*
- **EN2 — consultation × probe policies** (role of E2): cross `π_c` (when: fixed vs
  belief-variance-triggered, **H9**) with `π_o` (what: random vs uncertainty vs
  information-gain, §8.2, **H10**), at equal probe budget. *Does smart scheduling, and
  then smart sensing, beat dumb?*
- **EN3 — correction/belief operators** (role of E3): `hard_reset` vs `residual` vs
  `projection` vs `belief-filter`. *Do operators differ now that observation is partial
  (no v0 identity collapse), and does correction teach within a rollout — **H7**?*
- **EN4 — representation & drift-mitigation ablation** (role of E4): graph vs flat;
  RSSM-belief vs Markov; noise-injection on/off; self-forcing/scheduled-sampling on/off;
  size; **topology generalization** (train families → held-out families — the field's
  known failure mode, §2.2). *Which lesson actually buys horizon — **H11**?*
- **EN5 — objective ablation** (role of E4 objective axis): supervised vs +RLVR vs
  +online-TTT (self-healing). *Does verifiable reward + self-healing lift the curve where
  v0's RLVR was a null — **H7**, in a world with horizon to extend?*
- **EN6 — counterfactual & two-oracle grounding** (**H5**, **H12**): train with
  branched-replay counterfactuals; add the Batfish-style control-plane oracle (§5.1);
  measure faithfulness and downstream change-safety on held-out incidents.

**External harness (optional, for community legibility).** Where it is cheap to do so,
EN1/EN3/EN5 are additionally reported against a **CAGE-4 / CybORG**-class scenario
(SPEC-3 §3.3) — a partially-observable enterprise-defense gym the ACD community already
trusts — and the change-safety framing of EN6 maps to incident-response / root-cause
benchmarks (OpenRCA-class). This makes the speedup-vs-fidelity Pareto curve (model alone →
model + budgeted oracle → full oracle) legible to readers outside the world-model field,
the way learned network simulators report speedup vs. ns-3.

Every figure regenerates from config + seeds (the `figures/reproduce.sh` discipline),
and **negative results are first-class** (the v0 norm).

---

## 13. Milestones (NW0–NW8)

SPEC-5 is the buildable expansion of SPEC-3 §15's roadmap milestones **S4 (network
world)** and **S5 (partial observability)** — and it consciously leans on **S1** (the
Tier-A system oracle) for its Tier-B namespaces. The `NW` series below is to S4/S5 what
SPEC-2's `M0–M8` were to the v0 sketch: the same staging discipline, where the
deterministic core (NW0–NW3) ships and is fully tested with **no runtime dependencies and
no GPU** before any learned model. It does not collide with `M0–M8`, `S1–S6`, or `AR0–AR5`.

| Milestone | What | Gate | Status |
|---|---|---|---|
| **NW0** | Network env: typed-graph `State` ([`net/`](../../src/verisim/net/)), action grammar, canonical serialization + **Tier-A reference oracle** ([`netoracle/`](../../src/verisim/netoracle/)) + [`docs/network-semantics.md`](../network-semantics.md) + golden trajectories | property tests + goldens | ✅ |
| **NW1** | Graph `Delta` types, `apply(state, delta)`, delta↔serialization ([`netdelta/`](../../src/verisim/netdelta/)); the `apply == oracle` invariant | invariant tests | ✅ |
| **NW2** | Drivers (uniform/weighted/adversarial topology+traffic), trajectory generation ([`netdata/`](../../src/verisim/netdata/)) | data tests | ✅ |
| **NW3** | Graph divergence `d`, reachability-faithfulness, bits-to-correct ([`netmetrics/`](../../src/verisim/netmetrics/)); `H_ε` + run-record schema reused from v0 | metric tests | ✅ |
| **NW4** | `M_θ`: constrained graph-delta decode + supervised training (SLM-sized) ([`netmodel/`](../../src/verisim/netmodel/)). The **flat** arm — a from-scratch transducer over serialized `(state, action) → Δ` reusing v0's transformer + trainer — ships and is tested; it is the H11 flat-Markov baseline and the NW5-loop drop-in | model tests (torch extra) | ◐ flat arm shipped |
| **NW5** | Propose-verify-correct loop with **partial-observation oracle** (full / probe modes), probe-policy interface `π_o`, correction/belief operators, baselines, model-agnostic runner ([`netloop/`](../../src/verisim/netloop/)). The **message-passing + RSSM `M_θ`** is deferred to NW7, where partial observability makes the belief non-degenerate (§6.2) and the graph arm becomes the H11 contender — exactly as v0 shipped the M5 loop with baselines before the neural model bit | loop invariants | ✅ (graph arm → NW7) |
| **NW6** | **EN1 network `H_ε(ρ)` curve** + bootstrap-CI aggregation + figure ([`experiments/en1.py`](../../src/verisim/experiments/en1.py), [`figures/en1_curve.png`](../../figures/en1_curve.png)) | **the prime directive** | ✅ (H8 honest negative on the flat arm) |
| **NW7** | Smart probe policies + belief operators + drift mitigations; EN2/EN3/EN4 (equal-budget, CIs). **EN2** (consultation policy `π_c`, H9) and **EN3** (correction/belief operators, §8.3) ship on the flat arm ([`en2.py`](../../src/verisim/experiments/en2.py), [`en3.py`](../../src/verisim/experiments/en3.py)): EN3 breaks v0's operator identity collapse and shows the probe earns ~2.3× more faithful horizon per oracle-bit. The graph/RSSM arm (H11), the smart info-gain `π_o` (H10), the drift mitigations, and EN4 remain | comparison figures | ◐ EN2/EN3 (flat arm) |
| **NW8** | RLVR + online-TTT (EN5), counterfactual training (EN6), **LLM-callable simulator protocol** (§7), Inspect benchmark + `verifiers`-spec network RL env, technical report | packaging + report |

NW0–NW3 + the NW5 loop are the deterministic core. `M_θ` (NW4) drops into the loop via
the same model-agnostic interface v0 uses (`NeuralWorldModel`).

---

## 14. The autonomous research engine over the network world

SPEC-5 is built, as far as possible, with the human out of the loop — extending SPEC-4's
ratchet, not replacing it. The user's "replicate Karpathy's autoresearch" intent lands
here.

- **The gate is unfakeable.** You cannot lower bits-to-correct (§5.4) or raise
  reachability-faithfulness (§9.2) without actually predicting the truth. This is the
  property that makes autonomous search safe to run unattended (SPEC-4 §9).
- **Search space (the knobs the proposer may turn):** topology curriculum, model arch
  (layers/width/message-passing depth), drift-mitigation strength (noise σ, schedule),
  probe policy, consultation policy, TTT learning rate, RLVR/GRPO settings.
- **Frozen network eval cells.** A held-out set of topologies + golden trajectories the
  proposer never sees or mutates. Anomalous jumps re-evaluate on a second held-out set.
- **Denylist (the judge is not a knob).** The proposer cannot edit the oracle, the
  metric, the goldens, the gate, or `docs/network-semantics.md`.
- **The four irreducibles stay with the human** (SPEC-4 §8), restated for this domain:
  1. **The objective** — what "better" means (faithfulness; bits-to-correct down,
     `H_ε(ρ)` knee up).
  2. **The safety/ethics boundary** — defensive-only, namespace sandbox, **no
     real-internet egress**, editable-path denylist (§15).
  3. **The kill-switch + resource cap** — a compute ceiling the engine cannot raise.
  4. **Promotion to main** — the engine proposes on a branch; a human or trusted CI
     merges.

The network world *strengthens* the autoresearch story: longer horizons and graded
reachability reward give a denser, harder-to-hack signal than the filesystem floor.

---

## 15. Safety & ethics

This subsystem simulates networks; the posture is therefore stated explicitly and
mirrors SPEC.md §13 / SPEC-3 §14 / SPEC-4 §9.

- **Defensive framing.** The downstream use is autonomous network **defense**,
  change-safety, capacity planning, and SRE — predicting the consequences of *your own*
  configuration changes in a sandbox before they touch production. Not offense, not
  scanning third parties, not exploitation.
- **No real-internet egress.** Tier-A is a pure simulator. Tier-B runs only in network
  namespaces with egress disabled and deterministic, self-contained traffic generators.
  The model never touches a network it does not own.
- **Reproducibility as a safety property.** Everything is replayable from seed; there is
  no telemetry and no runtime network call (the repo-wide posture).
- **Denylist + kill-switch + resource cap** govern the autoresearch engine (§14).
- **Dual-use note.** A faithful network simulator is dual-use; the denylist, the
  defensive task framing, and the sandbox-only oracle are the structural mitigations, and
  the golden trajectories pin semantics so the engine cannot quietly repurpose them.

---

## 16. Repo layout additions

New packages, mirroring v0's structure so the conventions hold:

```
src/verisim/
  net/            # typed-graph State, Action grammar, canonical serialization, config
  netoracle/      # Tier-A reference DES; Tier-B namespace adapter (optional extra)
  netdelta/       # graph Delta types, apply(), serialization  (the NW1 invariant)
  netmodel/       # M_θ: vocab/tokenizer/grammar + constrained delta decode + supervised dataset
                  #   (flat arm shipped, reusing v0's GPT/trainer); message-passing/RSSM with NW5
  netloop/        # partial-observation propose-verify-correct runner, π_c, π_o, operators
  netmetrics/     # graph divergence, reachability-faithfulness, H_ε, bits-to-correct
  netdata/        # topology + traffic drivers, trajectory JSONL, manifests/splits
  neteval/        # faithfulness benchmark (Inspect adapter) for the network world
  netrl/          # verifiers-spec network RL environment (oracle-as-reward)
  experiments/    # e5_1 … e5_6 entry points (alongside existing e1…e4)
docs/
  network-semantics.md   # normative Tier-A semantics (paired with netoracle)
configs/  e5_1.json … e5_6.json
figures/  plot_e5_1.py …      # regenerated from run-records only
```

Tier-B and torch remain optional extras; the deterministic core has no runtime deps.

---

## 17. Open questions

The v0 §17 discipline: record them, resolve them in the open.

1. **Semantics boundary.** How much of TCP / routing / firewalling does Tier-A model
   before it overfits to a toy or becomes too costly to keep correct? (The single most
   important design call.)
2. **Tier-B determinism.** Kernel scheduler and clock nondeterminism in namespaces — how
   to pin enough that the oracle is replayable. If it cannot be fully pinned, Tier-B
   becomes a *stochastic* oracle and the divergence metric must account for it.
3. **`ε` for reachability.** What counts as "ε-close" for a reachability matrix and for
   continuous quantities (RTT, throughput)? Is reachability-faithfulness a separate axis
   or folded into `d`?
4. **Observation realism vs tractability.** How faithfully to model NetFlow/SNMP/pcap
   sampling without making the partial-observation oracle the bottleneck.
5. **Scale ceiling.** How many hosts/links before training cost dominates; where the
   curriculum should stop for v1.
6. **Topology generalization.** Per-topology-specialized SLM vs topology-general model —
   the field's known failure mode (§2.2). Decide what EN4 must demonstrate.
7. **Counterfactual sampling.** Which branch distribution produces counterfactuals that
   transfer (H5). Random action flips, or targeted "near-miss misconfigurations"?
8. **Plan-level loop.** How to define divergence and `H_ε` for the LLM-integration case
   (§7) where the unit is a *plan*, not a step.

---

## 18. Definition of done

SPEC-5 is done when:

1. NW0–NW6 ship, tested, with the deterministic core dependency-free.
2. The **network `H_ε(ρ)` curve (EN1)** is plotted once, cleanly, regenerable from
   config + seeds — *whatever it shows.* A flat interior is a reportable result
   (the H8 negative), not a failure.
3. The honest write-up (the `docs/report.md` discipline) states, for each hypothesis in
   §10 (the operationalized H5/H7/H8/H9 and the new H10–H12), what was found and what an
   honest negative looked like.
4. The network world is packaged for reuse (Inspect benchmark + `verifiers`-spec RL env
   + the LLM-callable simulator protocol of §7), so the community can measure
   long-horizon faithfulness under partial observability with ground truth — the
   contribution of §1.3.

The science is one curve, again. The world is finally hard enough to draw it.
