# SPEC-20 — The Usefulness Proof: An Agent Trained Inside the World Model, Transferred to Reality

**Application-and-transfer specification: the program has measured, exhaustively, whether the world model
is *faithful* — `H_ε(ρ)` curves, scaling laws, conformal coverage, an exact SCM. It has never measured
whether the world model is *useful*. The field's gold-standard test of a world model is not its
faithfulness number; it is whether you can **train a policy inside it and have that policy work in
reality** (Dreamer's learn-in-imagination, the entire premise of model-based RL). Verisim has the rare
ability to run that test rigorously, because its "reality" — the system oracle — is checkable. SPEC-20
trains a downstream defensive (ACD) agent *inside* the oracle-grounded learned network world model, then
transfers it to the system oracle and measures task success. It is Phase 5 of the SPEC.md §12 roadmap,
and it is the experiment that converts "a brilliant measurement instrument" into "a world model good for
something."**

> **▶ PRIORITY — APPLICATION / TRANSFER SPEC — proposed 2026-06-11.** A *downstream-application* spec,
> the first in the program: where every prior spec studied the world model as the object, SPEC-20 uses it
> as the *environment* for a separate learned policy. It depends on the [SPEC-19](./SPEC-19.md) flagship
> checkpoint (the trained network `M_θ`, kept faithful by the loop) as the training environment, the
> [SPEC-5](./SPEC-5.md) network world as the task substrate, and the highest-fidelity available network
> oracle as the transfer target (the Tier-A [`ReferenceNetworkOracle`](../../src/verisim/netoracle/reference.py)
> as the deterministic anchor, with the SPEC-11 system-oracle line as the reality stand-in §6). It invents
> **no new world or oracle.** What it adds is the *policy layer above the world model* and the
> *transfer measurement* — a thing the program has never built and the field has never cleanly measured.

> **NOT the existing RLVR.** [`rl/environment.py`](../../src/verisim/rl/environment.py) and
> [`train/rlvr.py`](../../src/verisim/train/rlvr.py) train the **world model itself** against the oracle
> reward (faithful horizon as return — the model *is* the policy, the oracle *is* the reward). SPEC-20 is
> the opposite direction: a **separate agent** whose actions are *defensive operations*, whose environment
> *is the frozen world model* (not the oracle), and whose reward is *task success* (containment), not
> faithfulness. The world model here is the cheap simulator the agent learns in; the oracle's role is to
> keep that simulator faithful *during the agent's training rollouts*, and to be the reality the trained
> agent is finally tested against. Confusing the two collapses the whole point — so the spec keeps them
> rigorously apart (§3).

Read [SPEC.md §4.1](./SPEC.md) (the ACD sim-to-emulation gap this spec closes), [SPEC.md §12 Phase 5](./SPEC.md)
(this is that phase), [SPEC.md §14](./SPEC.md) (#4, the bridge result — this spec lands it), and
[SPEC-19](./SPEC-19.md) (the flagship checkpoint that is this spec's training environment). This document
is *whether a policy learned in the cheap oracle-grounded model survives contact with the oracle's
reality — and whether the oracle-grounding is what makes it survive.*

---

## 0. One-paragraph thesis

A world model earns its name by being something you can *act inside*. The model-based RL bet (Dreamer,
MuZero, DayDreamer) is that a learned model is cheap enough to generate millions of training rollouts a
real environment could never afford, so a policy trained in imagination transfers to reality at a
fraction of the real-environment sample cost. That bet has a known failure mode: the model drifts, the
policy learns to exploit the model's hallucinations, and transfer collapses — and in every domain the bet
is made in, *you cannot tell whether the model drifted*, because there is no oracle. Verisim is the one
place the bet can be made honestly: the oracle keeps the training-time model faithful (SPEC-19), and the
oracle *is* the reality the policy is finally tested against, so transfer is measured exactly. The thesis,
stated to be falsified: *a defensive policy trained inside the oracle-grounded learned network world model
transfers to the oracle's reality at task-success competitive with a policy trained directly against the
oracle, at far lower training-rollout cost — and the oracle-grounding is load-bearing: the same policy
trained inside an oracle-free (uncorrected) model transfers materially worse.* If instead oracle-grounding
buys no transfer advantage — the uncorrected model trains an equally good policy — that is a deep,
bankable negative: faithfulness, the program's whole product, does not convert into downstream usefulness
in this world, and the §4.1 ACD bridge is weaker than claimed. Either branch is the first quantified
sim-to-emulation *policy-transfer* result the ACD field has, against a checkable reality.

---

## 1. Why this is the experiment that makes the program matter

SPEC-18 measured *faithfulness* transfer: `ΔH=0` on the validated grammar, the world model's predictions
match the system oracle. That is necessary but not sufficient — it shows the model is *accurate*, not that
it is *useful*. A model can be faithful and useless (if no decision depends on the dynamics it gets right)
or faithful and useful (if a policy trained in it transfers). The field's reviewers, and the ACD
practitioners SPEC.md §4.1 targets, care about the second. SPEC-20 supplies it.

It also closes the program's largest open loop. The roadmap (SPEC.md §12) lists Phase 5 — "train a
downstream agent inside the Verisim world model and measure transfer to the real system — the ultimate
test of a world model's usefulness" — and every shipped spec is upstream of it. Phase 5 has never been
attempted. SPEC-20 is Phase 5.

## 2. The task: defensive containment on the network world

The downstream task is a **defensive cyber (ACD) containment episode** on the SPEC-5 network world — the
defensive framing SPEC.md §13 commits to, no offensive agent built or trained.

- **Environment dynamics:** the SPEC-5 network world — a typed reachability graph of hosts/services, with
  a scripted adversary process that spreads along reachable edges (the existing
  [`netdata/`](../../src/verisim/netdata/) drivers supply the dynamics; the adversary is a fixed,
  *non-learned* spreader, so this spec trains *only* a defender — the §13 ethics commitment).
- **Agent (defender) actions:** defensive operations drawn from a fixed grammar — isolate a host (cut its
  edges), patch a service (close a reachability), restore a quarantined host. Partial observation: the
  defender sees the probe-revealed subgraph (the SPEC-5 §5.3 partial-observation oracle), not the whole
  network — the realistic ACD condition.
- **Reward (task success):** containment — fraction of the network kept un-compromised over the episode,
  minus an operational cost per defensive action (isolating everything trivially "wins" and is penalized).
  This is *task success*, orthogonal to faithful horizon.
- **Why this task:** it is the canonical ACD benchmark shape (CybORG/CyGIL containment), it exercises the
  network world's reachability dynamics (so getting the dynamics right *matters* to the policy — ruling
  out the faithful-but-useless degenerate case), and it has a clean scalar success metric.

## 3. The three environments (kept rigorously apart)

The whole experiment turns on not confusing three distinct things the word "environment" could mean:

| Name | What it is | Role | Cost |
|---|---|---|---|
| **`E_oracle` (reality)** | the network world stepped by the exact oracle | the **test** environment; also the *expensive* training baseline | high (every step is an oracle call) |
| **`E_grounded` (the product)** | the frozen SPEC-19 flagship `M_θ`, rolled out with the oracle-in-the-loop correction at budget `ρ` | the **cheap, faithful** training environment | low (`ρ` oracle calls per step) |
| **`E_free` (the ablation)** | the *same* frozen `M_θ`, rolled out **with no oracle** (`ρ=0`, uncorrected) | the **oracle-free** training environment — the control that isolates grounding | lowest (no oracle calls) |

The agent is trained three ways — in `E_oracle`, `E_grounded`, `E_free` — and **all three are tested in
`E_oracle`**. The comparison `E_grounded` vs `E_free` is the money measurement: it isolates *what the
oracle-grounding buys for downstream transfer*, holding the model, the task, and the rollout budget fixed.

## 4. Hypotheses (H73–H76)

Pre-registered with both branches, per SPEC.md §10.1.

- **H73 (learn-in-imagination works — the bridge result).** A defender trained in `E_grounded` reaches
  containment success in `E_oracle` (reality) competitive with one trained directly in `E_oracle`
  (within a pre-registered margin), at far lower training-rollout oracle cost (target: ≥5× fewer oracle
  calls for ≥90% of the reality-trained policy's success). *Refuted if* the `E_grounded`-trained policy's
  reality success is materially below the `E_oracle`-trained policy — the cheap faithful model is not a
  usable training environment, and learn-in-imagination fails here. Tested as **UA1**.
- **H74 (oracle-grounding is load-bearing — the money hypothesis).** The `E_grounded`-trained defender
  transfers to reality **materially better** than the `E_free`-trained defender at matched rollout budget
  and policy class — i.e. the oracle-in-the-loop correction during training is what makes the policy
  transfer, not merely the model's raw predictions. *Refuted if* `E_free` transfers as well as
  `E_grounded` — a deep bankable negative: in this world the policy does not exploit the uncorrected
  model's drift enough to hurt transfer, so faithfulness does not convert into downstream usefulness, and
  the §4.1 ACD bridge is weaker than the program claimed. Tested as **UA2**. *This is the experiment the
  whole spec exists for.*
- **H75 (the sim-to-emulation policy gap is small and quantified).** The transfer gap — Δ(containment
  success) between training-in-`E_grounded` and deploying-in-`E_oracle` — is small and bounded, and we
  *report the number* (the policy-level analogue of SPEC-18's `ΔH=0` faithfulness transfer). *Refuted if*
  the gap is large — the model is faithful (SPEC-18) yet policies trained in it still do not transfer, a
  precise and surprising result that would say faithful one-step dynamics are insufficient for policy
  transfer (the decisions depend on dynamics the faithfulness metric does not weight). Tested as **UA3**.
- **H76 (grounding's value tracks the consultation budget).** The transfer advantage of `E_grounded` over
  `E_free` (H74) is monotone in the training-time consultation budget `ρ` — more grounding during the
  agent's rollouts buys more transfer — recovering `E_oracle`'s transfer as `ρ→1` and `E_free`'s as
  `ρ→0`. *Refuted if* the advantage is flat in `ρ` (a little grounding is as good as a lot, or none of it
  matters) — which would itself sharply characterize how much faithfulness a downstream policy actually
  needs. Tested as **UA4**.

## 5. Milestones (UA0–UA5)

- **UA0 — the containment env.** Build the defender action grammar + containment reward on the SPEC-5
  world as a `verifiers`/Gymnasium env (reuse the [`rl/environment.py`](../../src/verisim/rl/environment.py)
  protocol shape, but a *policy* env, not the world-model-as-policy env — §3). Plug-swappable backend:
  `E_oracle` / `E_grounded` / `E_free` selected by a flag, so the three training runs share one code path.
- **UA1 — learn-in-imagination (H73).** Train the defender in `E_grounded`; evaluate in `E_oracle`;
  compare to the `E_oracle`-trained baseline on success and on training oracle-call cost. Committed figure
  [`figures/ua1_imagination_transfer.png`](../../figures/ua1_imagination_transfer.png).
- **UA2 — the grounding ablation (H74).** The `E_grounded` vs `E_free` head-to-head at matched budget;
  disjoint-CI bar of reality success. *The deliverable.*
  [`figures/ua2_grounding_ablation.png`](../../figures/ua2_grounding_ablation.png).
- **UA3 — the policy transfer gap (H75).** Report Δ(success) train-vs-deploy as a number, the policy-level
  sim-to-emulation measurement; cross-link SPEC-18's `ΔH`.
- **UA4 — the budget sweep (H76).** Transfer advantage vs training-time `ρ`.
- **UA5 — the usefulness report + the `verisim-acd-train` artifact.** One section of
  [`docs/report.md`](../report.md), and the env hardened into the SPEC-18 `verisim-acd` package as a
  *trainable* environment (not just a faithfulness surface) — closing SPEC.md §14 #4, the ACD bridge.

The agent itself is deliberately the *smallest* policy that does the job (a tabular / small-MLP defender
under REINFORCE or PPO — the point is the *transfer measurement*, not RL sophistication, exactly as
SPEC-2 §5.3's RLVR kept the optimizer minimal). The honest reason: a fancy policy would confound "the
agent is good" with "the world model is a good training environment," and only the latter is the result.

## 6. Reality, and the honest transfer-target caveat

The strongest version of this experiment transfers to the SPEC-11 *system oracle* — a real kernel/network
stack — so "reality" is reality, not a model of it. The network world's system-oracle line (Tier-B,
SPEC-7 §5.2; the SPEC-11 `SandboxOracle` for the host-level operations) is less mature than the
filesystem `SandboxOracle`. SPEC-20 therefore pre-registers a *graded* reality target, weakest-to-
strongest, and reports which rung it reaches:

1. **Tier-A reference reality** — train in `E_grounded`/`E_free`, test against the exact
   [`ReferenceNetworkOracle`](../../src/verisim/netoracle/reference.py). Proves the *mechanism* of policy
   transfer (faithful to a model of reality). Always achievable.
2. **Tier-B / system-oracle reality** — test against the highest-fidelity sandboxed network/host oracle
   available (SPEC-11 line). Proves transfer survives contact with *reality*. `skipif`-guarded on the
   system-oracle maturity; deferred with the GPU bet if not ready, and that deferral is stated, not
   hidden (SPEC.md §13).

This mirrors SPEC.md §2.1's reference-vs-system oracle discipline: Tier-A proves the mechanism cleanly;
Tier-B proves fidelity to reality. SPEC-20 lands rung 1 as the committed result and reaches for rung 2.

## 7. Gate and what each branch licenses

**Gate: H74** (oracle-grounding is load-bearing for transfer).

- **H74 confirmed** → the program's product is *useful*, not just faithful: oracle-grounded learned world
  models train transferable ACD policies cheaply, the §4.1 bridge is real and quantified. Next: scale the
  adversary (learned red team, still defensive framing), the GPU policy class, and the Tier-B reality rung.
- **H74 refuted** (`E_free` transfers as well) → the bankable negative: faithfulness does not convert into
  downstream usefulness in this world. This does not kill the program — the measurement, scaling, and
  causal contributions stand — but it *redirects* it: the next question becomes *which* tasks make
  faithfulness load-bearing (a task-taxonomy fork), since this one did not.

## 8. Status

| ID | Hypothesis / artifact | State | Result |
|---|---|---|---|
| UA0 | containment env, three backends | ✅ shipped (CPU core) | the containment MDP + the three-backend seam ship ([`acd/containment.py`](../../src/verisim/acd/containment.py)): a scripted adversary spreads over the SPEC-5 reachability graph (compromise tracked in the env's own episode state, on top of the shipped `connected_hosts`), a defender `isolate`/`patch`/`noop`s (compiled to `host_down`/`svc_down`/`advance` NetActions), reward = containment − action cost. `OracleBackend`/`GroundedBackend`(ρ-corrected)/`FreeBackend` differ *only* in how the net state evolves — the UA2/H74 seam — so the adversary spreads on the backend's (possibly drifted) reachability. 7 torch-free tests (CI-runnable): isolation halts spread, ρ=1 grounded ≡ oracle, seeded topology connected + reproducible. |
| UA1 | H73 — learn-in-imagination transfers | ▶ proposed | — |
| UA2 | H74 — grounding is load-bearing | ▶ proposed (the deliverable) | — |
| UA3 | H75 — policy transfer gap quantified | ▶ proposed | — |
| UA4 | H76 — advantage tracks `ρ` | ▶ proposed | — |
| UA5 | usefulness report + `verisim-acd-train` | ▶ proposed | — |

Honest caveats, stated up front: (1) the adversary is scripted, not learned — this spec trains a defender
only (§13 ethics) and the learned-red-team fork is future work. (2) Rung-1 reality is a reference oracle
(a model of reality); rung-2 (system oracle) is reached only if the network/host system-oracle line is
mature enough, else deferred and stated. (3) The policy is deliberately small; "the world model is a good
training environment" is the result, not "the agent is clever."
