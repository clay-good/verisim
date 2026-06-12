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
| UA1 | H73 — learn-in-imagination transfers | ✅ shipped + **frontier run** — **SUPPORTED** ([`ua2_grounding_rho0.2.csv`](../../figures/ua2_grounding_rho0.2.csv)) | the torch-free REINFORCE engine ([`acd/policy.py`](../../src/verisim/acd/policy.py), [`acd/train.py`](../../src/verisim/acd/train.py)). **Frontier result — H73 SUPPORTED:** the defender *learns* (reality containment **0.42** vs a 0.21 noop baseline; a hand-coded isolate-exposed heuristic ceilings at 0.51), and at **ρ=0.2** the `E_grounded`-trained policy reaches reality containment **≥** the `E_oracle`-trained one (**0.420 vs 0.390**) at **5× lower training oracle cost** (720 vs 3,600 calls) — both the ≥90%-quality and ≥5×-cheaper bars met. Learning in the cheap faithful model transfers. |
| UA2 | H74 — grounding is load-bearing | ✅ shipped + **frontier run** — **REFUTED (the bankable negative)** ([`ua2_grounding_ablation.csv`](../../figures/ua2_grounding_ablation.csv)) | the three-environment contrast ([`experiments/ua_transfer.py`](../../src/verisim/experiments/ua_transfer.py)). **Frontier result — H74 REFUTED:** `E_grounded`-trained and `E_free`-trained defenders transfer to reality **identically** (0.420 = 0.420; advantage **0.000**, and slightly *negative* at a lighter schedule). Oracle-grounding during training buys **no** transfer advantage. **Mechanism (diagnosed):** both backends teach the *same* effective policy — "isolate exposed hosts" — because that policy keys on compromise/exposure features that **survive the model's reachability drift**, so the flat model's errors never change the preferred action. Per SPEC-20 §7 this is the deep bankable negative: faithfulness does not convert into downstream usefulness *for this task* — redirecting to the task-taxonomy fork (which tasks make the optimal policy depend on the dynamics the model drifts on). |
| UA3 | H75 — policy transfer gap quantified | ✅ shipped + **frontier run** — **SUPPORTED** | `transfer_gap` ([`experiments/ua_transfer.py`](../../src/verisim/experiments/ua_transfer.py)). **Frontier result — H75 SUPPORTED:** the grounded policy performs **identically** in-model and in reality (gap = **0.000**) — the world model is a *faithful training environment* (zero sim-to-emulation policy gap), the clean positive. The dissociation with UA2 is the finding: a **faithful** training env (H75) whose faithfulness is **not load-bearing** for this task (H74). |
| UA4 | H76 — advantage tracks `ρ` | ✅ shipped + **frontier run** — **REFUTED** | `budget_sweep` ([`experiments/ua_transfer.py`](../../src/verisim/experiments/ua_transfer.py)). **Frontier result — H76 REFUTED:** the grounding advantage is **flat and ≤ 0 across ρ** (−0.11 to −0.21 at ρ ∈ {0.1, 0.3, 0.5, 0.8}; 0.000 at full config), never monotone-rising — consistent with the H74 null (no advantage to grow with ρ). |
| UA5 | usefulness report + `verisim-acd-train` | ✅ written from frontier numbers | the dissociation result (faithful env, non-load-bearing faithfulness) is written into [`docs/report.md`](../report.md) (SPEC-19/20 section); the env is the `verisim-acd-train` trainable surface. |
| UA6 | H78 — the task-taxonomy fork (when is faithfulness load-bearing?) | ✅ shipped + **frontier run** — **REFUTED, diagnosed** ([`experiments/ua_taxonomy.py`](../../src/verisim/experiments/ua_taxonomy.py), [`ua6_taxonomy.csv`](../../figures/ua6_taxonomy.csv)) | the H74 redirect: contrast the drift-robust task (UA0 local features) with a **drift-sensitive** task (the `marginal_cut` structural feature + a tight `cut_budget`, so *which* host you isolate depends on multi-hop reachability). **Frontier result — H78 REFUTED, but the diagnostic is the finding:** the grounding advantage is +0.000 (robust, reproducing H74) and **−0.010** (sensitive) — grounding still not load-bearing. *Why:* `feature_drift_diagnostic` shows the marginal-cut feature is **97.4% identical** on drifted (`E_free`) vs true states — the flat model's drift is **connectivity-structure-preserving** (raw links drift ~24% but the coarse reachability the policy reads survives), so the task never became drift-sensitive and there was nothing for grounding to fix. This *sharpens* the redirect: faithfulness needs a task whose optimal policy depends on the **fine-grained state the model actually drifts on** (specific services/flows), not the coarse connectivity that is itself drift-robust. |
| (diag) | the drift profile — *where* does the flat model drift? | ✅ measured | free-running the flagship alongside the oracle, the per-dimension disagreement is: **host up/down 0.000**, firewall 0.003, services (host,port) 0.011, links 0.044, **flows 0.252** (at 12 steps; services/links grow to 0.255/0.145 by 40 steps, past `H_free≈18`). The model is faithful on the *control-relevant* state and drifts almost entirely on *control-irrelevant flows* — the mechanistic root of the UA2/UA6/UA7 nulls. |
| UA7 | H79 — predictive control needs faithfulness | ✅ shipped + **frontier run** — **REFUTED; the characterization is complete** ([`acd/predictive.py`](../../src/verisim/acd/predictive.py), [`experiments/ua_predictive.py`](../../src/verisim/experiments/ua_predictive.py), [`ua7_predictive.csv`](../../figures/ua7_predictive.csv)) | the boundary's positive side: a fixed model-predictive defender that rolls its model forward `k` steps to choose isolations, planning with the exact oracle (**faithful**) vs the raw `M_θ` (**free**) vs a model-free **reactive** baseline, all against true dynamics. **Frontier result — H79 REFUTED, and decisive:** closed-loop predictive planning **helps a lot** (0.800 vs reactive 0.558, **+0.242**) but its *faithfulness is irrelevant* — faithful planner ≡ free planner at every `k` (gap +0.000), and the same in **open-loop** (plan the whole episode with no re-observation: faithful ≡ free ≡ 0.558). *Why:* the flat model **perfectly predicts the control lever** (`host_down`: 0% drift) and the planning-relevant reachability is drift-robust, so a drifted model plans identically. **The complete SPEC-20 finding across six formulations** (reactive, structural feature, long-horizon, closed-loop predictive, open-loop predictive): oracle-grounded faithfulness is **structurally not load-bearing for control in this domain — the model is faithful where control needs it.** The sharp boundary: faithfulness-for-control appears only where the model is *bad at the control-relevant dynamics* — the natural next test is the **host world** (SPEC-6, `H_free≈5` vs the network's ≈18). |
| HFL0 | the **host** flagship — the harder world | ✅ shipped + **frontier run** ([`experiments/host_flagship.py`](../../src/verisim/experiments/host_flagship.py)) | the FL0 lifecycle on the SPEC-6 host world: train + freeze a flat host `M_θ` at the HS2 frontier (`l`, 110k params), gated on reload-determinism + the HS2 band. **Frontier result:** `H_free` = **9.25 id / 6.00 ood**, `p` = 0.700 id — materially **less faithful** than the network flagship (18.75 / 0.875), exactly the harder regime the boundary predicted to probe. |
| (host diag) | the host drift profile — does the harder model's drift reach the control lever? | ✅ shipped + **frontier run** — **the cross-world law** ([`experiments/host_drift.py`](../../src/verisim/experiments/host_drift.py), [`host_drift.json`](../../figures/host_drift.json)) | per-action 1-step accuracy: **exit 1.00, close 0.96, fork 0.93** (discrete *structure* — faithful) vs **open 0.55, write 0.36** (*content* — drifts); free-running drift: **procs 0.000** (process structure perfect) vs **fds 0.258, fs 0.322** (file content). **The law, now confirmed on both worlds:** the flat model learns the discrete **structural** dynamics faithfully (network reachability / host process-tree) and drifts on the **content** dynamics (network flows / host file-writes). So control tasks that key on structure are drift-robust in *both* worlds — and the drift-sensitive opportunity (the predicted positive side of the boundary) is a **content-keyed** task: **file integrity**, keyed on `write`/`fs`, where the host model drifts ~30–64%. That is the precise next experiment. |
| UA8 | H80 — content-keyed control needs faithfulness | ✅ shipped + **frontier run** — **SUPPORTED; the boundary closes from both sides** ([`acd/host_integrity.py`](../../src/verisim/acd/host_integrity.py), [`experiments/ua_host_integrity.py`](../../src/verisim/experiments/ua_host_integrity.py), [`ua8_host_integrity.csv`](../../figures/ua8_host_integrity.csv)) | the predicted positive: a **predictive file-integrity** defender protects the budget files it predicts will be corrupted, scored on how many true corruptions it caught — a decision that depends entirely on the model's prediction of *which files get written* (the content the host model drifts ~25% on). Faithful predictor (oracle rollout) vs free predictor (raw `M_θ` rollout), swept over horizon. **Frontier result — H80 SUPPORTED, the positive:** the faithful predictor catches **every** corruption (reward 1.000) while the free predictor catches only **0.50–0.73**, and the **gap widens with horizon** as content drift compounds (+0.29 at h=6 → +0.50 at h=14). **This closes the SPEC-20 boundary from both sides:** world-model faithfulness is *not* load-bearing for control that keys on the **structure** the model learns faithfully (UA2–UA7, six nulls, both worlds) and *is* load-bearing for control that keys on the **content** the model drifts on (UA8). The complete law: **faithfulness-for-control is load-bearing exactly when the task's optimal policy depends on the dynamics the model gets wrong.** |
| UA9 | H81 — the **useful knee**: buy content-keyed faithfulness at budget ρ | ✅ shipped + **frontier run** — **SUPPORTED; the synthesis** ([`acd/host_integrity.py`](../../src/verisim/acd/host_integrity.py) `grounded_defense_reward`, [`experiments/ua_host_grounded.py`](../../src/verisim/experiments/ua_host_grounded.py), [`ua9_grounded_knee.csv`](../../figures/ua9_grounded_knee.csv)) | the missing synthesis of SPEC-19 (`H_ε(ρ)`) and SPEC-20 (usefulness): UA8 measured only the two extremes (ρ=1 faithful / ρ=0 free); the program's whole thesis is that the oracle-in-the-loop *buys back* faithfulness cheaply at a budget ρ — but that curve had never been run on the *downstream* task where faithfulness is load-bearing (on the structural tasks UA2–UA7 there was no advantage for ρ to recover; H76/UA4 found it **flat** in ρ). UA9 adds the **ρ-grounded predictor** — free-run `M_θ`, re-anchor to the oracle's truth every `round(1/ρ)` steps (propose–verify–correct on the predictive rollout) — and sweeps ρ on the content-keyed file-integrity task. **Frontier result — H81 SUPPORTED, the useful knee:** the catch rate rises **monotonically** with ρ (0.50 → 0.67 → 0.85 → **1.000**), recovering the every-step faithful predictor's perfect catch at **ρ=0.5 — half the oracle calls (7 vs 14)**. Two findings in one: (1) the **H76/UA4 mirror** — the grounding advantage that was *flat* on structural control **is monotone in ρ here**, on the task whose optimal policy depends on the content the model drifts on; (2) the **useful knee** — SPEC-19's "buy faithfulness cheaply" mechanism, demonstrated for the first time on SPEC-20's *downstream task success*, not just on faithful horizon. The cheap-faithful-model story holds **exactly where it has to**: on the one task where faithfulness is load-bearing, you can still buy it at sub-linear oracle cost. |
| UA10 | H82 — the boundary law + useful knee are **cross-world** | ✅ shipped + **frontier run** — **SUPPORTED; the cross-world confirmation** ([`acd/net_integrity.py`](../../src/verisim/acd/net_integrity.py), [`experiments/ua_net_integrity.py`](../../src/verisim/experiments/ua_net_integrity.py), [`ua10_net_integrity.csv`](../../figures/ua10_net_integrity.csv), [`ua10_net_integrity.png`](../../figures/ua10_net_integrity.png)) | UA8/UA9 drew the law and the knee on the **host** world (content = file-writes); UA10 asks whether both reproduce on the **network** world, whose content dimension is **flows** (the net flagship drifts ~0.252 on the live-flow set, faithful on structure). The content-keyed task: an adversarial workload (from a connected seed topology, the flow-bearing regime) opens connections; the defender predicts which flows will be established over the episode (the **cumulative** flow set — detect every connection the adversary made) and protects the budget flows it predicts. **Frontier result — H82 SUPPORTED, both reproduce on the network, more sharply than host:** (1) the **content-keyed positive** — the faithful predictor catches every flow (1.000) while the free predictor **collapses 0.583 → 0.083** as flow drift compounds (gap +0.42 at h=8 → **+0.92** at h=28; the network model drifts *harder* on its content than the host model, so the free predictor fails more completely than host's 0.50–0.73); (2) the **useful knee** — the ρ-grounded predictor recovers the faithful ceiling from a near-zero floor (0.083) at **ρ=0.2 — 4 oracle calls vs 20 (5× cheaper)**, monotone in ρ. The complete cross-world picture: the boundary law (faithfulness load-bearing iff the task keys on the content the model drifts on) **and** its cheap-purchase (the useful knee) hold in *both* worlds, with magnitudes that scale with how hard each world's model drifts on its content. The law is not host-specific. |
| UA11 | H85 — the boundary holds on the **third world** (distributed) — two is a pattern, three is a law | ✅ shipped + **frontier run** — **SUPPORTED, with a distributed-specific wrinkle** ([`experiments/dist_boundary.py`](../../src/verisim/experiments/dist_boundary.py), [`ua11_dist_boundary.csv`](../../figures/ua11_dist_boundary.csv), [`ua11_dist_boundary.png`](../../figures/ua11_dist_boundary.png)) | UA8–UA10 confirmed the boundary on host + network; UA11 lands the third world, the **distributed** (SPEC-7) — the hardest, where the global oracle is intractable and the state is replicated values under partition. The structure/content split: **partition-control** (structure, keyed on the partition groups) vs **value-integrity** (content, keyed on the replicated *(object, value)* pairs the model drifts on). A trained distributed `M_θ`, faithful-vs-free predictive-defense on each. **Frontier result — H85 SUPPORTED:** the content gap **+0.50** materially exceeds the structure gap **+0.23** (faithful 1.000 on both; the boundary holds on the third world, completing host + network + distributed). **The distributed wrinkle (honest):** (1) the structure gap is *not* ~0 like host/network — the distributed world is hard enough that even partition structure isn't perfectly learned at this model scale; (2) the content is load-bearing **but not cheaply buyable** — the useful knee is at **ρ=1** (full grounding), unlike host's ρ≈0.25 and network's ρ≈0.2, because the in-flight/partition medium (SPEC-7 H19/H20, where errors hide between re-anchors) makes partial grounding insufficient. So the *gradient* (content > structure) is the cross-world law; the *cheap knee* is a host/network property the distributed medium breaks. |

| UA12 | H92 — the **operational** detection characteristic: drift costs *precision*, not just recall | ✅ shipped + **frontier run** — **SUPPORTED; the operational completion of UA8** ([`acd/host_detection.py`](../../src/verisim/acd/host_detection.py), [`experiments/ua_host_detection.py`](../../src/verisim/experiments/ua_host_detection.py), [`ua12_host_detection.csv`](../../figures/ua12_host_detection.csv), [`ua12_host_detection.png`](../../figures/ua12_host_detection.png)) | UA8 proved faithfulness load-bearing for the content-keyed file-integrity task by the **catch rate** — but scored a *budget-limited recall only*, which is **structurally blind to false alarms**: it caps the defender's flags at `budget` and counts only the hits, so a model that flags the *wrong* files pays nothing. A real detector flags every file it predicts corrupted, and a SOC gates on its **precision** (false-alarm rate), not its recall. UA12 scores the **full confusion matrix** over the uncapped predicted-vs-true written-file set (precision / recall / F1), because a drifting model's predicted set differs from truth in *both* directions — it misses real corruptions **and** flags untouched files. **Frontier result — H92 SUPPORTED:** the faithful detector holds P = R = F1 = **1.000** at every horizon, while the free (trained `M_θ`) detector loses **both** — precision falls to **0.69–0.79** (≈1 in 4 alarms is false, the cost UA8 could not see) *and* recall to 0.50–0.73, so the **F1 (deployability) gap reaches +0.48** at h=14 (vs UA8's recall-only +0.50). Two findings: (1) **drift degrades precision alongside recall** — the operational cost is a flood of false alarms, not just missed corruptions, so a drifting world model gives an *undeployable* detector even where its recall looks survivable; (2) **the cheap knee restores the whole operating point** — the ρ-grounded detector recovers deployable P + R + F1 (≥0.93) by **ρ≈0.1–0.2 (2–4 oracle calls of 20)**, the same sub-linear regime as UA9's recall knee (host ρ≈0.25). So the SPEC-20 boundary law, read operationally: faithfulness is what makes a content-keyed detector *deployable* (precision-grade), and the oracle-in-the-loop buys that deployability cheaply. *(Honest: the ρ-knee wiggles non-monotonically at ρ=0.3 — the re-anchoring grid's interaction with the cumulative keyed set and whether the final step is a re-anchor, the same UA9 quantization caveat; the knee is cheap, ρ≈0.1–0.2.)* |

Honest caveats, stated up front: (1) the adversary is scripted, not learned — this spec trains a defender
only (§13 ethics) and the learned-red-team fork is future work. (2) Rung-1 reality is a reference oracle
(a model of reality); rung-2 (system oracle) is reached only if the network/host system-oracle line is
mature enough, else deferred and stated. (3) The policy is deliberately small; "the world model is a good
training environment" is the result, not "the agent is clever."
