# SPEC-22 — The Agent-in-the-Loop: A Verified World Model as a Safety Layer for Computer-Use Agents and Autonomous Cyber Defense

**Application-capstone specification: every prior spec studied the world model as an *object* — is it
faithful (SPEC-2/5/6/7/10), where is faithfulness load-bearing (SPEC-20), does the boundary scale
(SPEC-21), does the benchmark discriminate (SPEC-21 CL1). SPEC-22 closes the loop the program exists
to close: it shows the one *deployment* that turns all of that into a directly useful thing for an AI
agent operating a computer and for an autonomous cyber defender — the **safety gate**. A capable
computer-use agent does not fire a risky action blind; it *previews* the consequence with a world
model ("look before you leap"), checks the predicted outcome against a guardrail, and executes only if
the preview says it is safe. This spec measures the one question that makes or breaks that pattern:
can the preview be *trusted*? The answer is the whole program in one sentence, now at the point of
action: a free (unverified) world model is unsafe to act on exactly where the guardrail keys on the
content the model drifts on — the agent executes credential-corrupting plans it previewed as safe —
and the oracle-in-the-loop is what makes the preview trustworthy, cheaply.**

> **▶ PROPOSED — APPLICATION / DEPLOYMENT SPEC — 2026-06-12.** A *downstream-application* spec, sibling
> to [SPEC-20](./SPEC-20.md) (the usefulness proof it operationalizes) and a direct answer to the
> standing question *"how does this lead to real computer-use agents and cyber defense?"*. It invents
> **no new world, oracle, or model**: it runs on the [SPEC-6](./SPEC-6.md) host world, the shipped
> [`ReferenceHostOracle`](../../src/verisim/hostoracle/reference.py), the change-safety predicates
> already in [`hostsim/goal.py`](../../src/verisim/hostsim/goal.py), the agent-callable simulator
> ([`hostsim/simulator.py`](../../src/verisim/hostsim/simulator.py), `imagine`/`verify`), and the
> trained host `M_θ` (the SPEC-20 HFL0 flagship). What it adds is the **gate framing** — the agent's
> allow/abort decision and its *safety* confusion matrix (the asymmetric, catastrophic *missed-danger*
> error) — and the demonstration that the boundary law and the cheap knee govern whether an agent can
> act safely on its model.

Read [SPEC-20 §7](./SPEC-20.md) (the boundary law: faithfulness is load-bearing iff control keys on
the content the model drifts on — the law this spec deploys), [SPEC-19](./SPEC-19.md) (the useful
knee, the cheap-verification mechanism), and [SPEC-6 §2.6/§7](./SPEC-6.md) (the change-safety /
incident-response task family and the agent-callable simulator). This document is *whether a verified
world model lets a computer-use agent and a cyber defender act safely, and what it costs.*

---

## 0. One-paragraph thesis

A computer-use agent's core unsafe move is acting on a prediction that is wrong in the one way that
matters. SPEC-22 makes that concrete and measures it: a battery of host action plans, each genuinely
safe or unsafe by the *oracle's* verdict; an agent that previews each plan through a predictor and
**allows** it iff the preview says a guardrail holds; and the **missed-danger rate** — truly-unsafe
plans the agent wrongly executed. The thesis: **on a content guardrail (a credential file is not
overwritten — keyed on the file writes the host model drifts on) a free preview misses real dangers,
so the agent executes destructive plans; the oracle preview misses none; and the cheap ρ-knee
(re-anchor the preview to the oracle every `round(1/ρ)` steps) drives missed-danger to zero at a
fraction of the verification cost. On a structure guardrail (a protected process stays alive — keyed
on the process tree the model learns faithfully) the free preview already gates correctly.** The
boundary law, read at the point of action: *verification is what makes a world model safe for an agent
to act on, exactly where the agent's guardrail keys on the dynamics the model gets wrong — and it is
cheap.* The opposite branch is first-class and would be a clean negative: if the free preview gated
the content guardrail correctly too, then faithfulness would not be load-bearing for safe computer use
in this world, and the agent could act on a cheap unverified model.

---

## 1. Why the safety gate is the experiment that makes the value legible

The program's results are sharp but they read as *metrology*: faithful-horizon curves, structure/
content gaps, scale laws, discriminative leaderboards. A reader asking "so an AI agent uses this *how*,
exactly?" has had to infer the answer. The safety gate *is* the answer, in the form the question is
asked. It is the deployment pattern a frontier computer-use agent already uses — predict the
consequence of a risky action before taking it — and it makes every prior result land as a property of
*the agent's safety*, not of an abstract model:

- the **faithful horizon** (SPEC-10/19) becomes *how many plan steps the agent can trust before it
  must re-verify*;
- the **structure/content boundary** (SPEC-20) becomes *which guardrails a free preview can gate and
  which it cannot* — the agent can self-govern structural guardrails for free but needs the oracle for
  content guardrails;
- the **useful knee** (SPEC-19/UA9) becomes *the consultation budget that buys a safe gate*;
- the **real-kernel anchor** (SPEC-11) becomes *the gate is safe against a real `/bin/sh`, not a model
  of one*.

And the metric is the one a security or safety reviewer actually gates on: not a catch rate but a
**missed-danger rate**, the asymmetric error where the agent does the irreversible bad thing.

## 2. The deployment picture (the architecture)

The agent-in-the-loop loop, the thing a computer-use agent or an autonomous cyber defender runs (see
[`figures/cu_architecture.png`](../../figures/cu_architecture.png)):

1. **Agent (LLM).** Translates a natural-language intent into a host action plan (a sequence of
   syscalls: open/write/fork/setuid/…). The LLM is good at intent→plan; it is *bad* at simulating
   dynamics, which is why it does not gate itself.
2. **World model `M_θ` (cheap).** `imagine` rolls the plan forward with no oracle — a fast "look before
   you leap" preview. Cheap, but it drifts.
3. **Oracle (free, exact).** `verify` re-anchors the preview to ground truth at rate ρ (the reference
   oracle, or a real `/bin/sh` via SPEC-11). The propose–verify–correct loop spends a small ρ budget
   to keep the preview faithful.
4. **Safety gate.** A guardrail over the predicted final state. SAFE → ALLOW (execute on the real
   computer); UNSAFE → ABORT/flag. The gate is only as trustworthy as the preview, which is why
   verification is the load-bearing piece — exactly on the content the model drifts on (SPEC-20).

The honest scope: this is shell/file/process computer use (the oracle-grounded slice), the agent is a
fixed planner not a learned policy (the science is the *gate's* trustworthiness, not the agent's
cleverness), and the adversary/workload is scripted (defender-side only, the SPEC.md §13 ethics
commitment — no offensive agent is built).

## 3. The guardrails (the measurement substrate)

Two guardrails span the structure→content split the boundary law turns on, each a change-safety
predicate over the final host state (reusing [`hostsim/goal.py`](../../src/verisim/hostsim/goal.py)):

| guardrail | keyed dimension | the threat it encodes | predicted |
|---|---|---|---|
| **content** — `/passwd` not overwritten | file content (writes) | credential / config tampering | free preview **misses dangers** |
| **structure** — a protected process stays alive | process tree | killing a defensive daemon | free preview **gates correctly** |

The content guardrail is the canonical cyber/computer-use safety check ("do not let this plan tamper
with the credential file"). The host model drifts ~25–36% on the written-file set (SPEC-20 host
diagnostic), so the free preview mis-predicts whether `/passwd` is hit. The structure guardrail keys on
the process tree the model learns faithfully (~0% drift), so the boundary law predicts the free preview
gates it correctly — the agent can self-govern there without the oracle.

## 4. The headline measurement (CU1) and hypothesis (H93)

**CU1** ([`experiments/cu_safety_gate.py`](../../src/verisim/experiments/cu_safety_gate.py)): the plan
battery × {free, oracle, ρ-grounded} preview × {content, structure} guardrail, scored by the safety
confusion matrix ([`acd/safety_gate.py`](../../src/verisim/acd/safety_gate.py)). The headline figure
[`figures/cu1_safety_gate.png`](../../figures/cu1_safety_gate.png): the missed-danger rate by preview
and guardrail (left), and the missed-danger knee vs ρ on the content guardrail (right).

- **H93 (a computer-use agent needs a verified model to gate its actions safely — and the oracle buys
  it cheaply).** On the content guardrail the free preview's missed-danger rate is materially positive
  (the agent executes credential-corrupting plans it previewed as safe), the oracle preview's is zero,
  and the ρ-grounded preview drives missed-danger to zero at a sub-linear ρ (the UA9 knee, on agent
  safety). On the structure guardrail the free preview's missed-danger rate is already low (faithfulness
  not load-bearing). *Refuted if* the free preview gates the content guardrail as safely as the oracle
  (faithfulness not load-bearing for safe computer use in this world — the agent can act on a cheap
  unverified model, a clean and publishable negative that would redirect the deployment claim). Tested
  as **CU1**.
- **H94 (the safety gate is verified against a real `/bin/sh` — real computer-use).** The gate's
  missed-danger rate, measured against the deterministic reference oracle, holds when the SPEC-11
  `SandboxOracle` (a real `/bin/sh` on a real kernel) replaces it as the reality anchor — so the agent's
  safety claim is about real computer-use dynamics, not a model of them. *Refuted if* real-kernel
  semantics move the gate's verdict. Tested as **CU2-sys** (the gate sibling of CS3/H90). **Result —
  SUPPORTED:** on the content grammar (where SY1/H27 proved ref ≡ sandbox bit-exact), the missed-danger
  rate is **anchor-invariant — bit-identical against the real `/bin/sh` and the reference oracle (max
  Δ = 0)** at every capacity-proxy rung, *and* a free preview misses real dangers (0.71 at low α,
  receding to 0) even against the real kernel. The agent's safety gate, verified against reality.

- **H95 (the certified safety gate — provable, not just empirical, agent safety).** CU1/CU2 made the
  gate *empirically* safe (the oracle catches dangers). H95 makes it *provably* safe: using the free
  oracle as a conformal calibration set (SPEC-15), the agent attaches a distribution-free, finite-sample
  certificate `P(missed danger) ≤ α` to its gate. *Refuted if* the certificate cannot be made valid, or
  if its cost is independent of faithfulness (verification buys no cheaper guarantee). Tested as **CU3**.
  **Result — SUPPORTED, the program's deepest synthesis:** the certificate is **valid at every
  consultation budget ρ** (split-averaged missed-danger ≤ α = 0.1 at every rung), but its **false-block
  cost collapses with faithfulness** — **1.00 at ρ=0** (a drifting preview can only honor the guarantee
  by aborting *everything*, a safe-but-useless agent) → **0.01 at ρ=0.2** → **0.00**, with the gate then
  aborting exactly the unsafe fraction. So *any* world model can be made safe by being useless, and
  **only a faithful one is safe *and* useful** — the consultation budget ρ buys the safety certificate
  down to ≈ free (the safe-and-useful knee). Faithfulness, the boundary, the knee, and the certificate
  in one object.

- **H96 (the un-gameable gate — worst-case robustness, not average faithfulness).** CU1–CU3 measured
  the gate against a *random* world; cyber is *adversarial*. H96 asks whether the gate survives an
  attacker who knows the deployed model and fires only the plans it previews as safe (its blind spots).
  *Refuted if* a free gate's adversarial missed-danger matches its average-case rate (the gate is not
  gameable), or if verification does not close the worst case. Tested as **CU4**. **Result —
  SUPPORTED, two warnings:** (1) a **free gate is fully gameable** — its adversarial missed-danger is
  **1.0** (every crafted attack succeeds, vs 0.46 average) — and verification collapses *both* to ≈0 at
  the cheap knee (**un-gameable by ρ=0.2**); (2) the adversarial worst case at ρ=0 is **1.0 for *any*
  model fidelity** (0.71/0.46/0.22 *average* at φ=0.4/0.6/0.8, but **1.00 adversarial at all three**) —
  so a more faithful model is no safer against an adversary; average-case faithfulness is a *false sense
  of security*, and **only verification removes the worst case**. The oracle's value is not (only)
  average faithfulness but **worst-case robustness** — exactly what a security threat model requires.

- **H97 (the closed-loop safe agent — finishing the job without the irreversible harm).** CU1–CU4
  scored the gate's *verdict* on a fixed plan pool — the safety filter in isolation. But a computer-use
  agent *acts in a loop*: propose, preview, execute-if-safe / abort, repeat until the task is done. H97
  asks the loop's question: *can the agent finish the job without ever doing the irreversible bad
  thing?* The agent is scored on **both axes at once** — `task_success_rate` (utility) and `unsafe_rate`
  (safety) — and the deeper question (H97b) is *where* the consultation budget should be spent.
  *Refuted if* grounding does not move the agent from the unsafe/unreliable corner to the safe/reliable
  one, or if a stakes-aware consultation schedule buys no cheaper safety than a uniform one. Tested as
  **CU5**. **Result — SUPPORTED:** (1) a **free agent is in the bad corner — unsafe *and* unreliable**
  (task success **0.28**, unsafe-episode rate **0.57**: it both fails the job and does the irreversible
  bad thing), while the **oracle agent is safe *and* reliable** (1.00 / 0.00), and ρ is the path
  between them; (2) **where you spend the budget matters** — a *stakes-aware* schedule (consult the
  actions the model is most **uncertain** about, the SPEC-15 thesis at the action level) reaches the
  safe-and-reliable corner at **ρ=0.5**, vs a uniform schedule's **ρ=1.0** (the full oracle): the knee
  is bought by spending verification on the model's own blind spots. The closed loop is the literal
  "computer use for an AI agent," and a verified world model is what makes it both safe and useful.

- **H98 (the replanning agent — capability and safety trade off without the oracle).** CU5 gave each
  goal one route; a *capable* agent **replans** — when the gate blocks one route it tries another way
  to the same goal. H98 asks what that capability costs. *Refuted if* replanning does not lift
  capability, or if a free replanner is no more dangerous than a one-shot agent (persistence is free),
  or if grounding fails to make replanning safe. Tested as **CU6**. **Result — SUPPORTED, the warning
  capable-agent builders need:** replanning is real capability (it recovers the goals a one-shot agent
  abandons: success **0.52 → 0.88** free, **0.57 → 1.00** oracle) — but **for a free agent that
  capability is danger**: replanning **amplifies the harm rate** (one-shot **0.05** → replanner
  **0.12** at ρ=0, a +0.06 persistence penalty), because the retry loop turns every false-aborted safe
  route into one more step toward a dangerous one and every extra try into another draw at the model's
  blind spots — the agent becomes an **unintentional CU4 adversary against its own gate**. Verification
  dissolves the tension: the oracle truly aborts every dangerous route, so no number of retries can
  execute one — replanning becomes **pure benefit** (success 1.00, harm 0.00), and ρ is the path from
  the capable-but-dangerous free corner to the capable-and-safe oracle corner. The headline: **only a
  verified agent is both capable and safe — the more capable the agent, the more it needs the oracle.**

- **H99 (verify-before-commit — where you verify beats how much).** CU6 warned that free replanning
  amplifies harm; CU5 answered "spend a budget where the model is uncertain." H99 gives the
  *structural* answer an agent builder can act on: the harm only happens at the **moment of commit**
  (the agent can only do the irreversible bad thing by *executing* a route, and it only executes a
  route it believes safe), so verification belongs at exactly one place — the route the agent is about
  to commit to. A **verify-before-commit** agent replans *freely* (cheap model search) and spends one
  oracle call to verify the route it commits to: it **verifies the model's "yes" and trusts its "no"**
  (an abort can never cause harm). *Refuted if* this gives no cost advantage over uniform verification,
  or if it cannot reach the zero-harm guarantee. Tested as **CU7**. **Result — SUPPORTED:** a
  verify-before-commit agent reaches the **zero-harm guarantee *by construction*** (every executed
  route is oracle-verified) at **2.1× lower oracle cost** than a full-verification agent (**1.19 vs
  2.52 calls/goal**) — because **58% of full verification is wasted** on routes the model already calls
  dangerous (a "no" the agent would have aborted anyway), and verify-before-commit removes exactly that
  waste. The saving **grows with how adversarial the environment is** (more dangerous shortcuts → more
  wasted "no" verifications). The constructive resolution of CU6: the same persistence that was a
  liability becomes safe *and* cheap once verification is placed at the commit point. **Where you
  verify beats how much.**

- **H100 (the trained-arm confirmation — the closed loop survives a real learned model).** CU5–CU7
  measured the closed loop on a *controlled stand-in* (a φ-dial hash-coin drifter) — legitimate
  methodology, but it leaves the program's biggest credibility question open: *does the closed loop
  close under a real learned model's real drift, or only under synthetic drift?* H100 answers it on the
  substrate where it is feasible — the **network** world, whose trained flagship `M_θ` (a transformer
  world-model, CU2-net's) is cheap to run and whose live-flow content is exactly what it drifts on
  (~0.252, UA10). An agent works a connectivity goal (flows to the work hosts) from a workload that
  tempts it with exfiltration (flows to the protected crown-jewel hosts), previewing each action through
  the *real* model (ρ-grounded) and executing iff it shows no exfil. *Refuted if* the closed loop does
  not close on the real model (free ≈ oracle, or verification buys no safety). Tested as **CU5-net**
  (93 contested episodes, horizon 24, the real trained `M_θ`). **Result — SUPPORTED, with an honest
  refinement the stand-in could not show:** on the real model a **free agent opens *every*
  exfiltration flow** (unsafe-episode rate **1.00**, mean **1.29** missed exfil flows — the net model's
  free-running flow prediction collapses, as UA10/CU2-net found), and **verification drives it to 0.00**
  (1.00 → 0.96 → 0.85 → 0.69 → 0.56 → **0.00** as ρ: 0 → 1) — the **load-bearing safety axis closes
  exactly as on the stand-in**. But task success stays pinned at **1.00** at every ρ: the **real drift
  is one-sided** — the model *omits* flows (so it misses exfil) but never *hallucinates* a protected
  flow (so it never false-aborts a benign connect), so the *utility* axis the two-sided synthetic
  stand-in exercised is not triggered here. **The half that matters — verification is load-bearing for
  safety — is no longer a property of synthetic drift; it survives a real learned model.** (CPU-only;
  the trained arm is torch-gated, the one ~11-min train paid once; the closed-loop core is torch-free.)

- **H101 (the drift asymmetry — world models hide danger by omission).** CU5-net found the trained
  model's drift is *one-sided*; H101 characterizes it directly, because if it is robust it is a
  structural safety law, not a quirk. A teacher-forced probe of the real trained `M_θ` (predict each
  step from the oracle's true state) classifies every flow-prediction error as an **omission** (the
  oracle opened a flow the model missed — a hidden danger, the gate's *missed-danger* source) or a
  **hallucination** (the model invented a flow — a false alarm, the gate's *false-block* source), split
  by protected (danger) vs work (benign) host. *Refuted if* the errors are roughly symmetric
  (omissions ≈ hallucinations), or if the protected hosts show no special asymmetry. Tested as **CU8**
  (300 workloads, 7,200 steps, the real `M_θ`). **Result — SUPPORTED, decisively:** drift is
  **overwhelmingly omission-biased** — **417 omissions vs 14 hallucinations** overall (30:1) — and on
  the danger hosts the asymmetry is extreme: the model **missed 146 of 149 real exfiltration flows
  while hallucinating just 1** (a **146:1** missed-danger-to-false-alarm ratio; it foresaw only **2%**
  of true exfil flows). The mechanism is the point, not a pathology: consequential events (a connection
  establishing) are *rare*, so the model's safe default is to predict no consequence — and danger is
  exactly a rare consequence it then misses. **The catastrophic missed-danger cell is the one drift
  inflates; the model hides danger, it does not invent it.** This is the structural reason verification
  is load-bearing for *safety* specifically (CU5-net's safety axis needed the oracle; its utility axis
  did not), and it doubles the program's core asymmetry: the most costly confusion-matrix cell is also
  the one a learned world model's drift is biased toward. (Measured on the network world's trained arm,
  where it is cheap; the host arm is the deferred extension. Torch-gated trained arm; torch-free probe.)

- **H102 (the agent-safety horizon — unverified safety is a clock that runs out).** CU8 showed the
  model hides danger by omission, so an unverified agent misses essentially every exfil opportunity.
  H102 asks the *deployment* question that follows: how long can an unverified agent run before it does
  the irreversible bad thing, and how much does verification extend that? It is the safety-outcome
  analogue of SPEC-10's *faithful horizon* (which measured how long the model's *predictions* stay
  faithful) — CU9 measures how long the agent's *actions* stay safe. The agent runs the CU5-net closed
  loop over a long deployment on the real trained `M_θ`; we record the step of its first breach and
  build the **survival curve** (fraction still safe after `t` steps) per consultation budget ρ.
  *Refuted if* the free agent's survival does not decay with deployment length, or verification does
  not extend it. Tested as **CU9** (200 deployments, horizon 48, the real `M_θ`). **Result —
  SUPPORTED, and stark:** a free agent's survival **decays toward zero** — **breach rate 0.995** over a
  48-step deployment, safe for only **~20 steps on average** (median safe horizon **17**) — it breaches
  at its first dangerous opportunity, and over a long enough run that is a near certainty. Verification
  **flattens the curve and extends the safe horizon**: ρ=0.3 → ~26 safe steps (breach 0.81), ρ=0.5 →
  ~31 safe steps (breach 0.65), and the **oracle never breaches** (survival flat at 1.0). The
  deployment lesson, legible to any practitioner: **unverified safety is not a property an agent has,
  it is a clock that runs out — and on an omission-biased model the worse the drift, the more
  verification a long deployment needs.** (Network trained arm; host arm deferred. Torch-gated trained
  arm; torch-free survival core.)

- **H103 (targeted verification — what to verify beats how much).** CU9 measured verification on a
  *blind, uniform* schedule (re-anchor every `round(1/ρ)` steps) and found it only reaches zero
  breach at the full oracle. H103 asks the practitioner's follow-up: if you cannot verify
  everything, *which* steps should the budget buy? The tempting answer — let the model flag when it
  is unsure — is exactly wrong, and CU8 says why: a model that drifts by **omission** mis-predicts
  danger by predicting *no* consequence, so it cannot flag its own blind spots. The answer that
  works is **structural** and the defender's to supply: danger here is *grammar-localized* (every
  exfiltration flow to a protected host is opened by a `connect` whose destination is that host — an
  empirically exact property of the v0 net grammar), so a defender who knows the crown jewels can
  verify exactly the rare action class that can touch them. *Refuted if* structure targeting does
  not reach the oracle's breach rate, or buys no call saving over uniform, or if model
  self-targeting matches it. Tested as **CU10** (200 deployments, horizon 48, the real `M_θ`).
  **Result — SUPPORTED, decisively:** **uniform** needs the full oracle (48 calls) to reach zero
  breach (even ρ=0.5 breaches 0.65); **model self-targeting fails** — breach **0.995** at **0.07**
  calls, the free agent's danger, because the omitting model never expects the activity that matters;
  **structure targeting reaches the oracle's zero breach at 4.07 calls — 11.8× cheaper than the full
  oracle**, for identical safety. The constructive flip of CU8/CU9: the omission bias is a warning,
  but its *structure* is the remedy — because danger is concentrated and the model cannot self-report
  it, a defender targets verification at the crown-jewel surface and defends it cheaply. This is
  CU7's "where you verify beats how much" carried from the host stand-in's *commit point* to the
  trained network arm's *danger surface*, plus the new negative that the model itself cannot tell you
  where that surface is. (Network trained arm; host deferred. Torch-gated trained arm; torch-free
  verification core.)

- **H104 (un-gameable targeting — the adversary controls *when* danger happens).** CU10 measured the
  three schedules on a *random* workload; a cyber defender faces an **adversary** who knows the
  deployed schedule and chooses the *timing* of the exfiltration to evade it (CU4's worst-case move,
  carried from the safety *gate* to the verification *schedule*). The result turns on *what each
  schedule keys on*: **uniform** consults on the **clock phase** (a signal the attacker reads off the
  schedule and steps around), **model** consults on the **model's own output** (which omits on
  exactly the steps that matter, CU8), but **structure** consults on a **grammar-fixed property of
  the action** — in the v0 net grammar a flow is born *only* by `connect`, addressed to the connect's
  destination (the only edit that opens a flow is `FlowOpen`, emitted solely by `connect` — a
  structural invariant of `ReferenceNetworkOracle`), so the danger surface is **static** and the
  attacker cannot relocate an exfil off the `connect`-to-protected action class. *Refuted if*
  structure's adversarial breach exceeds 0, or uniform's adversarial breach does not exceed its
  random breach (the attacker gains nothing), or structure does not separate from uniform. Tested as
  **CU11** (200 deployments, horizon 48, the real `M_θ`). **Result — SUPPORTED, decisively:**
  **uniform and model targeting are gameable** — adversarial breach jumps to **1.000 at every ρ<1**
  (the CU9/CU10 knee is a mirage: the random-timing 0.65 at ρ=0.5 becomes 1.000 under an attacker
  who picks the step; only the *full* oracle at 48 calls is adversarially safe) — while **structure
  targeting is un-gameable** — adversarial breach **0.000 at 4.07 calls**, identical to its random
  breach, because the danger surface is grammar-fixed. The defender principle: **target verification
  at what the adversary cannot move.** This is CU4's worst-case-robustness thesis carried to CU10's
  targeting, with a cheap constructive winner. (Network trained arm; host deferred. Torch-gated
  trained arm; torch-free verification core.)

- **H105 (knowledge-free targeting — target the grammar, not the assets).** CU10/CU11 made targeted
  verification cheap *and* un-gameable by spending the oracle on the `connect`-to-crown-jewel actions
  — but that rests on an assumption a real defender cannot always meet: that the **crown-jewel
  inventory is complete**. Inventories drift, miss shadow services, and lag the network; an adversary
  exfiltrates to the host you *didn't* flag. There are two structural targets: **asset-indexed**
  (verify `connect` to a *known* jewel `K`, the CU10/CU11 target — cheap but blind to any sensitive
  host outside `K`) and **grammar-indexed** (verify *every* `connect`, the whole flow-genesis surface
  — needs **zero asset knowledge**, since a flow is born only by `connect`, so every exfil to *any*
  host is caught). H105 measures the asset target against the **true** sensitive set as the
  inventory becomes incomplete. *Refuted if* asset-indexed breach does not rise as the inventory
  becomes incomplete, or grammar-indexed targeting does not reach ~0 breach, or it costs about as
  much as the full oracle. Tested as **CU12** (200 deployments, horizon 48, the real `M_θ`, true
  sensitive set `{h0,h4}`). **Result — SUPPORTED, decisively:** with a 50%-complete inventory
  (`K={h0}`, `h4` unflagged) the asset target breaches **0.635 random / 0.960 adversarial** — nearly
  the unverified rate, a false sense of security, and fully gameable by an adversary who exfiltrates
  to the unflagged host; the **grammar-indexed target reaches 0.000 breach inventory-independently at
  9.35 calls — 5.1× cheaper than the full oracle (48)**. The defender principle: **when you cannot
  trust your asset inventory, target the grammar, not the assets** — the most robust target is the
  grammar's flow-genesis surface, which needs no asset list and is still cheap. (Network trained arm;
  host deferred. Torch-gated trained arm; torch-free verification core.)

- **H106 (capability under real drift — the false-alarm channel prices CU6 and CU7).** CU6 (free
  replanning *amplifies* harm by +0.06) and CU7 (verify-before-commit reaches zero harm at *2.1×*
  lower cost) were both measured on the *two-sided* synthetic stand-in. CU5-net then showed the real
  trained `M_θ` drifts *one-sided* (it omits exfil flows, never hallucinates one). H106 asks whether
  those two capable-agent results survive a real one-sided model — and isolates the mechanism: both
  are priced by the model's **"no" channel**, but by different halves. **CU6's amplification is
  priced by the FALSE-ALARM rate** (a *wrong* "no" false-aborts a safe route, forcing the agent to
  retry onto a blind-spotted danger), and **CU7's saving is priced by the danger RECALL** (a *right*
  "no" on a truly-dangerous route is a call full-verify wastes and verify-before-commit skips). The
  real `M_θ` has neither — it says "yes" to *every* route — so both effects should vanish. *Refuted
  if* the false-alarm sweep does not move the amplification, or the recall sweep does not move the
  saving, or the real model does not sit at the origin of both. Tested as **CU13** (200 net
  replanning goals, the real `M_θ`, two dials on a synthetic net model). **Result — SUPPORTED,
  decisively:** on the false-alarm dial (recall fixed at 0) the harm-amplification rises **0.000 →
  0.160**; on the recall dial (false-alarm fixed at 0) the verify-before-commit saving rises **1.00×
  → 1.70×** (wasted-call fraction 0 → 0.41); and the **real `M_θ` anchors at the origin of both**
  (measured false-alarm **0.000**, recall **0.004**) with **amplification exactly 0.000 and cost
  saving exactly 1.000×**. So CU6's capable-agent warning and CU7's verify-where economics are both
  **properties of a model that says "no"** — a real omission-biased world model does not, so neither
  appears. The honest other half: the danger does **not** vanish (the agent's one-shot harm is
  **0.53** either way — it blind-spots the dangerous route it walks), and verify-before-commit keeps
  its **zero-harm guarantee** at every rung. What survives to the real model is the *structural*
  guarantee and the omission danger itself; the *quantitative* knee and saving are two-sided
  artifacts. (Network trained arm; host deferred. Torch-gated trained arm; torch-free replanning
  core.)

- **H107 (the defended incident — the whole stack on one named scenario).** CU1–CU13 each isolated
  one face of the gate; H107 puts them together on a single, concrete, named scenario a defender
  reads end to end. An autonomous incident-response agent must restore work connectivity
  (`h1/h2/h3`) on a compromised segment salted with exfiltration lures (`connect` to the crown
  jewels `h0/h4`), under four defenses — **undefended** (trust the model), **paranoid** (block every
  `connect`, the "safe by being useless" corner of CU3), **structure** (verify the
  `connect`-to-jewel actions, CU10/CU12's grammar target), **full oracle** (verify every step) —
  scored on **all three axes at once**: mission completed (utility), exfiltrated (safety), oracle
  calls (cost). The prediction: only **structure** is in the all-good corner — it completes the
  mission (the one-sided model never false-aborts a benign connect, so targeting the danger surface
  costs the mission nothing, CU13), reaches the oracle's zero breach, and does so at a fraction of
  the oracle's cost. *Refuted if* structure does not reach the oracle's safety, or sacrifices the
  mission, or saves no cost over the full oracle. Tested as **CU14** (200 contested incidents,
  horizon 48, the real `M_θ`). **Result — SUPPORTED, decisively:** the **undefended** agent
  completes the mission (1.00) but **exfiltrates (0.99)**; **paranoid** is safe (0.00) but
  **abandons the mission (0.00)**; **full oracle** is safe and on-mission at **48 calls**; and
  **structure** is the only all-good corner — **safe (0.00 breach), on-mission (1.00), at 4.0 calls
  — 12× cheaper than the full oracle**. The representative-incident playback makes the mechanism
  legible: on the *same* action sequence the undefended agent walks the one true lure (`connect h2
  h4 22`, a breach) while structure spends an oracle call on exactly it (abort) and still finishes
  the work connects. The synthesis statement: a verified world model is the safety layer that lets a
  computer-use agent and a network defender complete the mission without the irreversible bad thing,
  and verifying the world's flow-genesis surface is cheap. (Network trained arm; host deferred.
  Torch-gated trained arm; torch-free incident core.)

- **H108 (the verification-exhaustion attack — the cost axis under an adversary).** CU11 proved
  structure targeting is un-gameable on the *safety* axis (an attacker who controls the *timing* of
  an exfiltration cannot make it breach, because the danger surface is grammar-fixed). But CU14
  noted structure spends a call on *every* `connect`-to-jewel, most of them benign — which opens a
  different attack the safety results never measured: an adversary who cannot make structure
  *breach* can still make it *expensive*, by flooding the danger surface with benign-looking
  activity to exhaust the verification budget (the real cyber phenomenon of *alert fatigue /
  denial of budget*). H108 carries CU4/CU11's worst-case threat model from the safety axis to the
  **cost** axis: a deployment of fixed length whose steps an adversary poisons with attacker
  `connect`-to-jewel actions at saturation `s`, each schedule read on *both* axes (breach, calls).
  The claim: an adversary can move **exactly one axis** of a sub-oracle schedule — **structure's
  cost** (clock-free, so the flood inflates it) or **uniform's safety** (clock-fixed cost, so the
  flood evades it) — and the defender should prefer the schedule whose movable axis is the *bill*,
  because structure's cost stays **bounded by and weakly dominates the full oracle** (`calls ≤
  horizon`, `= horizon` only at full saturation) and the attack is **self-limiting** (each attacker
  action adds one defender call and zero breaches). *Refuted if* structure's cost does not rise with
  `s` (no attack), or its breach rises above 0 (not safety-immovable, contradicting CU11), or its
  cost ever exceeds the full oracle. Tested as **CU15** (200 deployments, horizon 48, the real
  `M_θ`). **Result — SUPPORTED, decisively:** structure's **safety is immovable** (breach **0.000**
  at every `s`) while its **cost is gameable** — mean calls climb **4.07 → 15.07 → 26.09 → 36.99 →
  48.00** as `s`: 0 → 1, up to *exactly* the full oracle and never past it (it **weakly dominates**
  the full oracle at every `s`); the attack is **self-limiting** at **0.92 defender calls per
  attacker action and 0.000 breaches bought**. Uniform is the mirror image: its **cost is immovable**
  (24.00 calls at every `s`, the `ρ=0.5` clock) but its **safety is gameable** (breach 0.650 →
  0.965 as the flood places exfils off-clock). Only the full oracle is immovable on *both* axes — at
  the maximum price. The defender principle: an adversary always moves one axis of a sub-oracle
  schedule; prefer the schedule whose movable axis is a bill you can cap (≤ the full oracle, and the
  attacker pays its whole budget to impose it) over one that is a breach. The cost-axis analogue of
  CU4's lesson that average-case faithfulness is a false sense of security: average-case cheapness is
  a false sense of *economy*, but structure's worst case is still safe and still ≤ the price of total
  safety. (Network trained arm; host deferred. Torch-gated trained arm; torch-free verification core.)

- **H109 (cross-world targeting — the danger surface is grammar-fixed on the host too).** The whole
  targeting arc (CU10 cheap, CU11 un-gameable, CU12 knowledge-free) was measured on the *network*
  world, where danger is born by a single action-visible event (a flow to a crown jewel is opened
  only by a `connect` addressed to it). H109 asks whether the program's most-quoted result is
  network-specific or a general property of an oracle-grounded world, by carrying it to the **host**
  world (credential / config tampering — CU1's content guardrail). The host grammar invariant is just
  as exact: a file's content becomes non-empty *only* by a `write`, and a `write` reaches a path
  *only* through a file descriptor previously `open`-ed at it — so a `/passwd` corruption is born only
  by a `write` to an fd bound to that path. The host adds a sharper twist the network could not show:
  unlike the `connect` (whose destination is a literal argument), the host danger surface is the
  **composition of the action with the fd→path binding**, and that binding lives in the *process
  structure* — the part of the state the boundary law says the model learns **faithfully** (host `M_θ`
  drifts ~25-36% on file content but ~0% on the process/fd table; SPEC-20 §7). So structure targeting
  localizes danger using the structure the model is faithful on, even when the danger itself is a
  content corruption it drifts on. *Refuted if* structure does not reach the oracle's breach rate, or
  buys no call saving, or is gameable by adversarial timing (then host danger is not grammar-localized
  and the cheap defense is network-specific). Tested as **CU16** (200 deployments, horizon 48). The
  host trained `M_θ` is the deferred GPU arm (its rollout over fork-heavy workloads is pathologically
  slow on the throttled CPU — the LP7 rule and CU2-threats' lesson), so the schedule result, which
  keys on the oracle and the grammar not the model's competence, runs a **worst-case content omitter**
  stand-in (faithful on structure, omits writes — the realistic drift CU8 measured and CU1 confirmed
  the real host `M_θ` exhibits: a free preview misses 0.38 of real `/passwd` corruptions). **Result —
  SUPPORTED, decisively:** the network result generalizes — **uniform** needs the full oracle (48
  calls) to reach zero breach and its sub-oracle knee is a mirage (adversarial breach **1.000 at every
  ρ<1**); **model self-targeting fails** (breach **1.000** at 0 calls — the omitter cannot flag its
  own blind spots); and **structure targeting reaches zero breach at 3.49 calls — 13.8× cheaper than
  the full oracle — and is un-gameable** (adversarial breach **0.000**). The targeting result is not a
  network artifact: it is a property of any oracle-grounded world whose danger has a grammar-fixed
  genesis surface, and the host shows that surface can be localized through the very structure the
  model is faithful on. (Host world; the trained host arm is the deferred GPU extension. Torch-free
  verification core and stand-in.)

- **H110 (the genesis-grammar boundary — target the danger's genesis, not a single action class).**
  The whole targeting arc (CU10 cheap, CU11 un-gameable, CU12 knowledge-free, CU16 cross-world)
  rested on one assumption it never examined: that danger is born on a single, syntactically
  visible action class (the `connect` to a crown jewel). H110 tests whether the targeting
  *principle* is a real result or an artifact of that sparse grammar, by exhibiting a second,
  recognizable danger in the *same* world with a genuinely richer genesis: **network-segmentation
  exposure**, a crown jewel becoming *reachable* from an untrusted host
  (`can_reach(untrusted, jewel, port)` flipping `False → True`). Unlike an exfil flow, that
  reachability is not born by a `connect`: it is opened by the *config* grammar — `svc_up`,
  `fw_allow`, `host_up`, and above all `link_up` (a link completes a *path*, which is **multi-hop**,
  so a `link_up` between two hosts that are *neither* the jewel can still expose it). The danger
  surface is therefore **semantic** (reachability), not **syntactic** (an action class), and to
  enumerate it you must compute the reachability *closure* — the SPEC-12 landmark-reachability
  machinery. Four schedules on a battery of segmented deployments: uniform(ρ), the CU10–CU16
  `connect` target carried over verbatim, a *syntactic* genesis-grammar target (verify the genesis
  action *types* that name a jewel, plus — since a link is multi-hop — *every* `link_up`), and the
  *semantic* `closure` target (verify exactly the actions that flip `can_reach` to a jewel). *Refuted
  if* the `connect` target is as safe here as on the flow danger (the cheap target transfers
  unchanged), or `closure` does not reach the oracle's breach rate, or buys no call saving over the
  full oracle. Tested as **CU17** (200 segmented deployments, horizon 48). The trained host/network
  `M_θ` is the deferred GPU arm (per LP7); the schedule result keys on the oracle and the grammar,
  not the model's competence, so it runs a worst-case content omitter stand-in. **Result —
  SUPPORTED, decisively:** the cheap `connect` target **does not transfer** — it is blind to the
  config genesis, so its breach stays at the **free rate (1.000)** while spending **3.86 calls** (a
  false sense of security); the *syntactic* `grammar` target reaches near-zero random breach
  (**0.025**) but **leaks through multi-hop intermediates** (adversarial breach **0.370** — an
  attacker exposes a jewel via a `host_up` of a non-jewel relay it cannot name) and overpays
  (**13.72 calls**); only the *semantic* `closure` target reaches the oracle's **0.000 breach,
  un-gameable (0.000 adversarial), at 4.17 calls — 11.5× cheaper than the full oracle** (and it
  dominates the syntactic target on *both* axes). The principle: **target the danger's genesis
  grammar, and that grammar is whatever the world's transition relation says opens the danger —
  compute its reachability closure, do not pattern-match an action class.** The cheapness of CU10–CU16
  was a property of a *sparse* genesis grammar, not magic: a richer danger needs a richer (but still
  bounded, still sub-oracle) target, getting the grammar wrong gives false security, and the precise
  closure is computed by the world's reachability relation (SPEC-12). (Network world; trained arm
  deferred. Torch-free verification core and stand-in.)

## 5. Milestones

- **CU0 — the safety-gate core.** `Guardrail` + the safety confusion matrix (`SafetyOutcome`, the
  missed-danger / false-block / caught / correct cells) + the free/oracle/ρ-grounded gate evaluators,
  on the shipped predictive-rollout machinery. Deterministic tests; torch-free core. ✅
- **CU1 — the headline run.** The plan battery × previews × guardrails, the missed-danger figure + the
  ρ-knee, on the trained host `M_θ`. The committed result. ✅
- **CU2 — deepening: more threats, the real kernel, cross-world. ✅** Three deepenings: **CU2-sys** the
  gate against a real `/bin/sh` (H94, anchor-invariant); **CU2-threats** a recognizable threat spectrum
  (service kill / privilege escalation / credential tampering, ordered structure→content); **CU2-net**
  the cross-world exfiltration / flow-tamper gate on the network world.
- **CU3 — the certified safety gate. ✅** From empirically safe to *provably* safe: a distribution-free
  certificate `P(missed danger) ≤ α` on the gate, calibrated free by the oracle (H95). The headline is
  not just the guarantee but its *cost*: the false-block rate collapses with faithfulness (1.00 at ρ=0 →
  ≈0 by ρ=0.2), so only a faithful model is safe *and* useful — the safe-and-useful knee.
- **CU4 — the un-gameable gate. ✅** From the average case to the *adversarial* worst case (H96): a free
  gate is fully gameable (an attacker who knows the model achieves a 1.0 missed-danger rate), and
  verification makes it un-gameable at the cheap knee. The deepest warning: the worst case is
  fidelity-independent (1.0 for *any* model), so faithfulness alone is a false sense of security — only
  the oracle removes the worst case.
- **CU5 — the closed-loop safe agent. ✅** From the gate's verdict on a fixed pool to the *agent acting
  in a loop* (H97): an agent works a task, previewing each action and executing the safe ones / aborting
  the rest, scored on **both** task success and irreversible harm. A free agent is unsafe *and*
  unreliable; the oracle agent is safe *and* reliable; and a stakes-aware consultation schedule (spend
  the budget where the model is uncertain) reaches the safe-and-reliable corner at half a uniform
  schedule's budget — the knee. The literal "computer use for an AI agent."
- **CU6 — the replanning agent. ✅** From one route per goal to *replanning* (H98): a capable agent
  tries another way when the gate blocks the first. Replanning lifts capability — but for a *free* agent
  that capability is danger: the retry loop searches its own gate's blind spots, so replanning amplifies
  the harm rate (the agent becomes an unintentional CU4 adversary against itself). The oracle dissolves
  the tension — it aborts every dangerous route regardless of retries, so persistence becomes pure
  benefit (success 1.0, harm 0). The headline: only a verified agent is both capable and safe.
- **CU7 — verify-before-commit. ✅** The constructive fix to CU6 (H99): the harm only happens at the
  *moment of commit*, so verify the route the agent is about to execute and nothing else — verify the
  model's "yes," trust its "no." A verify-before-commit agent reaches the zero-harm guarantee (every
  executed route is verified) at **2.1× lower oracle cost** than verifying everything, because most of
  full verification is wasted on "no" decisions that can't cause harm. Where you verify beats how much.
- **CU5-net — the trained-arm confirmation. ✅** The rigor move (H100): re-run the closed loop on the
  *real* trained network `M_θ`, not a stand-in. On the real model a free agent opens **every**
  exfiltration flow and verification drives the unsafe rate to **0.00** — the load-bearing safety axis
  closes exactly as on the stand-in — while an honest refinement appears that only a real model could
  show: the drift is **one-sided** (omission, never hallucination), so task success stays at 1.0 and the
  stand-in's utility axis is not triggered. The closed loop's safety half survives a real learned model.
- **CU8 — the drift asymmetry. ✅** Characterizes CU5-net's one-sided finding into a structural law
  (H101): a teacher-forced probe of the real trained `M_θ` shows drift is **overwhelmingly
  omission-biased** (417 vs 14 errors overall; **146:1** on the danger hosts — 146 missed exfil flows
  vs 1 hallucinated, only 2% exfil recall). The model **hides danger, it does not invent it**, so the
  gate's errors concentrate in the catastrophic missed-danger cell — the structural reason verification
  is load-bearing for safety specifically.
- **CU9 — the agent-safety horizon. ✅** Turns the omission bias into a deployment-level safety metric
  (H102): the survival curve of the closed-loop agent on the real trained `M_θ` over a long deployment.
  A free agent breaches **99.5%** over a 48-step run and is safe for only **~20 steps on average**
  (median horizon 17); verification flattens the curve (ρ=0.5 → ~31 safe steps, breach 0.65) and the
  oracle never breaches. The safety-outcome analogue of SPEC-10's faithful horizon: **unverified safety
  is a clock that runs out; verification stops it.**
- **CU10 — targeted verification. ✅** Turns CU9's bleak clock into a constructive defense (H103):
  the question is not *how much* to verify but *what*. On the real trained `M_θ`, the blind uniform
  schedule needs the full oracle (48 calls) to reach zero breach; **model self-targeting fails**
  (breach 0.995 at 0.07 calls — the omitting model can't flag its own blind spots, the CU8 lesson);
  and **structure targeting** (verify the rare `connect`-to-crown-jewel actions the defender knows)
  reaches the oracle's zero breach at **4.07 calls — 11.8× cheaper**. You can't ask the omitter where
  it omits; you target verification at the danger surface, and danger is cheap to defend because it
  is concentrated. CU7's "where beats how much," carried to the trained network arm.
- **CU11 — un-gameable targeting. ✅** The adversarial worst case of CU10 (H104), CU4's threat-model
  move carried from the gate to the schedule: an attacker who knows the schedule chooses *when* to
  exfiltrate. On the real trained `M_θ`, **uniform and model targeting are gameable** — adversarial
  breach **1.000 at every ρ<1** (the CU9/CU10 knee is a mirage; only the full oracle at 48 calls is
  adversarially safe) — while **structure targeting is un-gameable** — adversarial breach **0.000 at
  4.07 calls**, because the danger surface is grammar-fixed (a flow is born only by `connect`,
  addressed to its destination) and the attacker cannot move danger off it. The defender principle:
  **target verification at what the adversary cannot move** — structure targeting is not just cheap
  on average, it is the only sub-oracle schedule whose guarantee survives an adversary.
- **CU12 — knowledge-free targeting. ✅** The robustness boundary CU10/CU11 left open (H105): the
  cheap asset-indexed target assumes the crown-jewel inventory is complete, but a real inventory is
  wrong, and an adversary exfiltrates to the unflagged host. On the real trained `M_θ`, a 50%-complete
  inventory (`K={h0}`, `h4` unflagged) breaches **0.635 random / 0.960 adversarial** — nearly the
  unverified rate, a false sense of security — while the **grammar-indexed target** (verify *every*
  `connect`, the whole flow-genesis surface) reaches **0.000 breach inventory-independently at 9.35
  calls, 5.1× cheaper than the full oracle**. The defender principle: **when you cannot trust your
  asset inventory, target the grammar, not the assets** — the flow-genesis surface needs no asset
  list and is still cheap. Completes the targeting arc (CU10 cheap → CU11 un-gameable → CU12
  knowledge-free).
- **CU13 — capability under real drift. ✅** The trained-arm confirmation of CU6 and CU7 (H106): both
  were measured on the two-sided stand-in, and CU5-net showed the real `M_θ` drifts one-sided. CU13
  isolates the mechanism with two dials on a net replanning world — **CU6's harm-amplification is
  priced by the false-alarm rate** (0.000 → 0.160 as it rises), **CU7's verify-before-commit saving
  by the danger recall** (1.00× → 1.70×) — and shows the real `M_θ` anchors at the **origin of both**
  (measured false-alarm 0.000, recall 0.004 → it says "yes" to every route): amplification exactly
  0.000, saving exactly 1.000×. CU6's capable-agent warning and CU7's verify-where win are both
  properties of a model that says "no"; a real omission-biased one does not, so neither appears. The
  danger does not vanish (one-shot harm 0.53 either way) and verify-before-commit keeps its zero-harm
  guarantee — the *structural* result survives, the *quantitative* one is a two-sided artifact.
- **CU14 — the defended incident. ✅** The synthesis (H107): the whole stack on one named scenario.
  An incident-response agent restores work connectivity on a compromised segment salted with
  exfiltration lures, under four defenses scored on all three axes at once (mission / breach / cost).
  **Committed trained run (193 contested incidents, horizon 48, real `M_θ`):** undefended completes
  the mission (1.00) but exfiltrates (0.99); paranoid is safe (0.00) but off-mission (0.00); full
  oracle is safe and on-mission at 48 calls; and **structure is the only all-good corner — safe
  (0.00), on-mission (1.00), at 4.0 calls, 12× cheaper than the full oracle**. The
  representative-incident playback replays the same action sequence undefended vs structure — the
  undefended agent walks the one true lure (a breach), structure verifies exactly it (abort) and
  still finishes the work. The legible bridge from metrology to deployment.
- **CU15 — the verification-exhaustion attack. ✅** The cost-axis worst case CU11 left open (H108):
  CU11 proved structure targeting un-gameable on *safety*; CU15 asks whether an adversary who cannot
  breach it can still exhaust its *cost* (alert fatigue / denial of budget), by flooding the
  `connect`-to-jewel surface structure must verify. **Committed trained run (200 deployments,
  horizon 48, real `M_θ`):** an adversary can move **exactly one axis** of a sub-oracle schedule —
  **structure's cost** climbs 4.07 → 48.00 calls as the flood grows (safety stays immovable at 0.000
  breach) while **uniform's safety** degrades 0.650 → 0.965 breach (its clock-fixed cost stays at
  24.00). But structure's cost stays **bounded by and weakly dominates the full oracle** (≤ horizon
  always, = only at full saturation) and the attack is **self-limiting** (0.92 defender calls per
  attacker action, 0.000 breaches bought). Only the full oracle is immovable on both axes, at the
  maximum price. The defender principle: prefer the schedule whose movable axis is a *bill* you can
  cap, not a *breach* — the cost-axis analogue of CU4 (average cheapness is a false sense of economy).
- **CU16 — cross-world host targeting. ✅** The generality test the targeting arc left open (H109):
  CU10–CU12 were all on the network world; CU16 carries the headline to the **host** world
  (credential tampering). The host grammar invariant is just as exact (a `/passwd` corruption is born
  only by a `write` to an fd bound to it), with a sharper twist — the danger surface is the action
  composed with the fd→path binding, which lives in the *structure the model learns faithfully*, so
  structure targeting localizes content danger through faithful structure. **Worst-case content
  omitter (200 deployments, horizon 48):** uniform needs the full oracle and its knee is a mirage
  (adversarial breach 1.000 at every ρ<1); model self-targeting fails (1.000 at 0 calls); **structure
  reaches zero breach at 3.49 calls — 13.8× cheaper than the full oracle — and is un-gameable (0.000
  adversarial)**. The targeting result is a property of any oracle-grounded world with a grammar-fixed
  danger genesis, not a network artifact.
- **CU17 — the genesis-grammar boundary. ✅** The falsifiable boundary the targeting arc left open
  (H110): CU10–CU16 always targeted a single, syntactically visible danger action; CU17 exhibits a
  second danger in the *same* world with a richer genesis — **network-segmentation exposure** (a
  crown jewel becoming reachable from an untrusted host), born by the config grammar
  (`link_up`/`svc_up`/`host_up`/`fw_allow`), not by `connect`, and **multi-hop** (a `link_up` between
  two non-jewel hosts can complete the path). **Worst-case content omitter (200 segmented
  deployments, horizon 48):** the cheap `connect` target **does not transfer** (breach **1.000**, the
  free rate, at 3.86 calls — a false sense of security); a *syntactic* genesis-grammar target reaches
  near-zero random breach (0.025) but **leaks through multi-hop intermediates** (adversarial breach
  **0.370**) and overpays (13.72 calls); only the *semantic* **reachability-closure** target reaches
  **0.000 breach, un-gameable, at 4.17 calls — 11.5× cheaper than the full oracle** (dominating the
  syntactic target on both axes). The principle: **target the danger's genesis grammar — compute its
  reachability closure (SPEC-12), do not pattern-match an action class.** CU10–CU16's cheapness was a
  property of a sparse genesis grammar, not magic.
- **The writeup.** Fold the gate into the SPEC-21 essay / README "from foundation to application"
  section — the legible bridge from the metrology to the deployment.

## 6. Gate and what each branch licenses

**Gate: H93** (the free preview misses content dangers the oracle catches).

- **H93 confirmed** → the program's most legible application result: *a verified world model is the
  safety layer that lets a computer-use agent and a cyber defender act on a cheap learned model
  without doing the irreversible bad thing — and the verification is cheap (the knee).* Licenses the
  "foundation → application" framing and the deployment story.
- **H93 refuted** (the free preview gates content safely too) → faithfulness is not load-bearing for
  safe computer use in this world; the agent can act on an unverified model, and the deployment claim
  narrows to worlds/guardrails where the model drifts on the gated dimension. A clean negative, and the
  oracle is what makes it trustworthy.

## 7. Honest caveats, stated up front

- **The agent is a fixed planner, not a learned policy.** The science is whether the *gate* can be
  trusted; "a smarter agent" is out of scope (the SPEC-20 §13 discipline). CU5 closes the loop with a
  *fixed* candidate-queue agent (it aborts and moves on; it does not learn a policy) — the variable
  under study is the gate and the consultation schedule, never the planner.
- **Defender-side only.** The workload is scripted; no offensive/red-team agent is built (SPEC.md §13).
- **Shell/file/process, not GUI.** The oracle-grounded slice (SPEC.md §11).
- **The real-`/bin/sh` anchor is shipped (CU2-sys / H94).** The gate's missed-danger rate is
  anchor-invariant against a real shell (max Δ = 0) on the validated content grammar; the *trained*-arm
  anchor (a real model gated against `/bin/sh`) is the deferred GPU extension, per the LP7 rule.

## 8. Status

| ID | Hypothesis / artifact | State | Result |
|---|---|---|---|
| CU0 | the safety-gate core | ✅ shipped (CPU core) | `Guardrail` + `SafetyOutcome` (the asymmetric safety confusion matrix) + free/oracle/ρ-grounded gate evaluators ([`acd/safety_gate.py`](../../src/verisim/acd/safety_gate.py)), on the shipped `host_integrity` rollouts + the `hostsim.goal` change-safety predicates. 7 torch-free tests. |
| CU1 | H93 — the agent needs a verified model to gate safely; the oracle buys it cheaply | ✅ shipped + **frontier run** — **SUPPORTED; the boundary law on the safety gate** ([`experiments/cu_safety_gate.py`](../../src/verisim/experiments/cu_safety_gate.py), [`cu1_safety_gate.csv`](../../figures/cu1_safety_gate.csv), [`.png`](../../figures/cu1_safety_gate.png)) | a 60-plan battery on the trained host `M_θ`, 29 plans truly overwriting `/passwd`. **Content guardrail — the free preview misses real dangers:** missed-danger **0.38** — the agent **executed 11 of 29 credential-corrupting plans** it previewed as safe (plus a 0.19 false-block rate, the over-caution cost) — while the **oracle preview misses 0** and the **ρ-knee drives missed-danger to zero at ρ=0.30 (6 oracle calls of 18, ~⅓ the budget)**: 0.38 → 0.28 (ρ0.1) → 0.10 (ρ0.2) → **0.00 (ρ0.3)**. **Structure guardrail (process stays alive, 17 truly unsafe) — the free preview already gates correctly:** missed-danger **0.00** (0 destructive plans executed), the boundary-law null. So a computer-use agent acting on an *unverified* world model executes credential-tampering plans exactly where the guardrail keys on the content the model drifts on, the oracle is what makes the preview safe to act on, and that safety is cheap (the knee). H93 SUPPORTED; the structure/content split the program proved as *metrology* now governs whether an agent can *act safely*. |
| CU2-sys | H94 — the gate is verified against a real `/bin/sh` | ✅ shipped + **frontier run** — **SUPPORTED; anchor-invariant** ([`experiments/cu2_system_gate.py`](../../src/verisim/experiments/cu2_system_gate.py), [`cu2_system_gate.csv`](../../figures/cu2_system_gate.csv), [`.png`](../../figures/cu2_system_gate.png)) | the gate sibling of CS3/H90: on the v0 fs content grammar (where SY1/H27 proved ref ≡ sandbox bit-exact), the agent's missed-danger rate is swept across a capacity-proxy α-ladder (a write-drifting `M_θ` stand-in, the trained arm deferred per LP7) and scored against **both** the reference oracle and a real `/bin/sh`. **Committed 4-rung run (real `/bin/sh`, platform=darwin; Linux CI reproduces):** the missed-danger rate is **anchor-invariant — bit-identical against the real kernel and the reference oracle (max Δ = 0)** at every rung, *and* a free preview misses real dangers **even against the real shell** (0.71 → 0.36 → 0.21 → 0.00 as α rises). The agent's safety gate is verified against reality, not a model of it. `skipif`-guarded + §2.5-disclosed. |
| CU2-threats | H93 deepened — the gate across a recognizable cyber threat spectrum | ✅ shipped (experiment + guardrails + CI-asserted spectrum) | the gate on a spectrum of named cyber threats, each ranked by the dynamics it keys on ([`experiments/cu2_threats.py`](../../src/verisim/experiments/cu2_threats.py)): the two clean boundary endpoints are **service kill** (a protected daemon killed — *structure*, the process tree the model learns faithfully) and **credential tampering** (`/passwd` overwrite — *content*, the file writes it drifts on). The free preview's missed-danger rate is **ordered by the keyed dimension** (service kill ≈ 0 — the agent self-governs it; credential tampering high — the agent needs the oracle), the oracle catching every danger on both — the boundary law as a *threat ranking*, asserted in CI ([`tests/test_cu2_gates.py`](../../tests/test_cu2_gates.py)). A third threat, **privilege escalation** via `setuid 0` (keyed on process *uid*, near-structure), ships as a tested guardrail ([`no_root_escalation`](../../src/verisim/acd/safety_gate.py)) — at smoke scale the free preview misses ~16% of root escalations (oracle 0), the predicted middle of the spectrum. The committed full-model *figure* is deferred (the run completes, but the trained host `M_θ`'s per-step rollout over fork-heavy workloads is too slow to render on the local CPU); the *result* is the CI-asserted ordering, not a figure. |
| CU2-net | H93 cross-world — the exfiltration / flow-tamper gate (network) | ✅ shipped + **frontier run** — **SUPPORTED; cross-world, even sharper than host** ([`experiments/cu2_net_gate.py`](../../src/verisim/experiments/cu2_net_gate.py), [`cu2_net_gate.csv`](../../figures/cu2_net_gate.csv), [`.png`](../../figures/cu2_net_gate.png)) | the canonical network threat: **exfiltration / unauthorized lateral movement** — *the plan opens no flow to a protected server* (`{h0, h4}`, the crown-jewel hosts), keyed on the live-flow content the net flagship drifts ~0.252 on (UA10). 40 plans, 15 truly opening a protected flow. **Committed run on the trained network `M_θ`:** the **free preview's missed-danger rate is 1.00** — the agent **ran all 15 exfil plans** it previewed as safe (the net model's free-running flow prediction collapses, so it never foresees the connection — UA10's 0.083 floor), while the **oracle preview misses 0** and the cheap knee drives missed-danger to zero at **ρ=0.50 (8 oracle calls of 16)**: 1.00 → 0.40 (ρ0.1) → 0.07 (ρ0.2) → **0.00 (ρ0.5)**. The agent-safety value is not host-specific: a verified world model is the safety layer for a network defender too, and the oracle buys it cheaply. |
| CU3 | H95 — the **certified** safety gate (provable, not just empirical) | ✅ shipped + **frontier run** — **SUPPORTED; the program's deepest synthesis** ([`acd/certified_gate.py`](../../src/verisim/acd/certified_gate.py), [`experiments/cu3_certified_gate.py`](../../src/verisim/experiments/cu3_certified_gate.py), [`cu3_certified_gate.csv`](../../figures/cu3_certified_gate.csv), [`.png`](../../figures/cu3_certified_gate.png)) | the agent attaches a **distribution-free, finite-sample certificate** `P(missed danger) ≤ α` to its gate, using the free oracle as a conformal calibration set — the SPEC-15 conformal idea ([`conformal.calibrate_threshold`](../../src/verisim/conformal/calibrate.py)) applied to the agent's allow/abort decision (a plan is a breach iff the oracle's true rollout violates the guardrail; the gate aborts when an ensemble-of-ρ-grounded-previews risk score exceeds the calibrated τ; a missed danger is a breach allowed). The trained arm deferred (the LP7 rule), a two-sided write-drifting stand-in on the v0 fs world (so it composes with the CU2-sys real-`/bin/sh` anchor). **Committed run (200 plans, α=0.1, split-averaged):** the certificate is **valid at every consultation budget ρ** (missed-danger ≤ 0.1 at all rungs), and its **false-block cost collapses with faithfulness — 1.00 at ρ=0 (a drifting preview can only be safe by aborting *everything*, useless) → 0.01 at ρ=0.2 → 0.00**, the gate then aborting exactly the unsafe fraction. So *any* model can be made safe by being useless; **only a faithful one is safe *and* useful**, and ρ buys the certificate down to ≈ free (the safe-and-useful knee). CPU-only, torch-free, seconds. |
| CU4 | H96 — the **un-gameable** safety gate (adversarial, not average-case) | ✅ shipped + **frontier run** — **SUPPORTED; the cyber threat-model result** ([`acd/adversarial_gate.py`](../../src/verisim/acd/adversarial_gate.py), [`experiments/cu4_adversarial_gate.py`](../../src/verisim/experiments/cu4_adversarial_gate.py), [`cu4_adversarial_gate.csv`](../../figures/cu4_adversarial_gate.csv), [`.png`](../../figures/cu4_adversarial_gate.png)) | the question a threat model demands: *is the gate gameable by an attacker who knows the deployed model?* Attacks are plans that truly write a protected prefix; the gate previews each (ρ-grounded) and allows iff the preview shows no protected write; the **adversarial** missed-danger is over the attacker's arsenal — the attacks the *free* model previews as safe (its blind spots), fired by choice. **Committed run (240 plans, φ=0.6, 5 deployed models, v0 fs / omit-only stand-in, the trained arm deferred):** **(1) a free gate is fully gameable** — adversarial missed-danger **1.00** vs **0.46** average — and verification collapses *both* to ≈0 at the cheap knee (**un-gameable by ρ=0.2**); **(2) the worst case is fidelity-independent** — at ρ=0 the *average* missed-danger falls with model fidelity (0.71 → 0.46 → 0.22 at φ=0.4/0.6/0.8) but the *adversarial* one is **1.00 at all three** — so a "better" model is no safer against an adversary; average-case faithfulness is a **false sense of security**, and only verification removes the worst case. The oracle's value is **worst-case robustness**, exactly what cyber needs. CPU-only, torch-free, seconds. |
| CU5 | H97 — the **closed-loop** safe agent (acting in a loop, not judging a pool) | ✅ shipped + **frontier run** — **SUPPORTED; the "computer use for an AI agent" result** ([`acd/closed_loop_agent.py`](../../src/verisim/acd/closed_loop_agent.py), [`experiments/cu5_closed_loop.py`](../../src/verisim/experiments/cu5_closed_loop.py), [`cu5_closed_loop.csv`](../../figures/cu5_closed_loop.csv), [`.png`](../../figures/cu5_closed_loop.png)) | the loop's question, the one that decides whether any of the gate work matters for real agents: *can the agent finish the job without ever doing the irreversible bad thing?* An agent works a task (benign files it must write) from a candidate queue salted with **dangerous traps** (writes to a protected prefix), previews each action through its world model's risk estimate, and EXECUTES the safe ones / ABORTS the rest — scored on **both axes at once**: `task_success_rate` (did it finish?) and `unsafe_rate` (did it ever execute a truly-dangerous action?). The danger labels are oracle-grounded (a write's real delta under the prefix), so the loop composes with the CU2-sys real-`/bin/sh` anchor. **Committed run (200 episodes, 16 actions each, φ=0.6, v0 fs / risk stand-in, trained arm deferred):** **(1) a free agent is in the bad corner — unsafe *and* unreliable** (task success **0.28**, unsafe-episode rate **0.57**) — while the **oracle agent is safe *and* reliable** (1.00 / 0.00), ρ the path between; **(2) where you spend the budget matters** — a *stakes-aware* schedule (consult the actions the model is most uncertain about, the SPEC-15 thesis at the action level) reaches the safe-and-reliable corner at **ρ=0.5** vs a uniform schedule's **ρ=1.0** (the knee). The closed loop is the literal computer-use deployment; a verified world model is what makes a fixed agent both safe and useful, cheaply. CPU-only, torch-free, seconds. |
| CU6 | H98 — the **replanning** agent (capability vs safety; persistence amplifies harm) | ✅ shipped + **frontier run** — **SUPPORTED; the capable-agent warning** ([`acd/closed_loop_replan.py`](../../src/verisim/acd/closed_loop_replan.py), [`experiments/cu6_closed_loop_replan.py`](../../src/verisim/experiments/cu6_closed_loop_replan.py), [`cu6_closed_loop_replan.csv`](../../figures/cu6_closed_loop_replan.csv), [`.png`](../../figures/cu6_closed_loop_replan.png)) | CU5 gave each goal one route; CU6 gives it several and lets the agent **replan** — try another way when the gate blocks the first. Each goal offers safe routes and dangerous shortcuts (oracle-grounded labels); the agent walks them in order and stops at the first its gate allows. A *one-shot* agent (retry budget 1) vs a *replanner* (full budget) across ρ. **Committed run (300 goals, 6 routes each, φ=0.6, v0 fs / risk stand-in, trained arm deferred):** replanning is real **capability** — it recovers the goals a one-shot agent abandons (success **0.52 → 0.88** free, **0.57 → 1.00** oracle) — but **for a free agent that capability is danger**: replanning **amplifies the harm rate** (one-shot **0.05** → replanner **0.12** at ρ=0, a +0.06 persistence penalty), because the retry loop turns every false-aborted safe route into a step toward a dangerous one and every extra try into another draw at the model's blind spots (an **unintentional CU4 adversary against its own gate**). The oracle dissolves the tension — it truly aborts every dangerous route, so no number of retries can execute one — and replanning becomes **pure benefit** (success **1.00**, harm **0.00**); ρ is the path from the capable-but-dangerous free corner to the capable-and-safe oracle corner. **Only a verified agent is both capable and safe — the more capable the agent, the more it needs the oracle.** CPU-only, torch-free, seconds. |
| CU7 | H99 — **verify-before-commit** (where you verify beats how much) | ✅ shipped + **frontier run** — **SUPPORTED; the constructive fix to CU6** ([`acd/closed_loop_verify.py`](../../src/verisim/acd/closed_loop_verify.py), [`experiments/cu7_verify_before_commit.py`](../../src/verisim/experiments/cu7_verify_before_commit.py), [`cu7_verify_before_commit.csv`](../../figures/cu7_verify_before_commit.csv), [`.png`](../../figures/cu7_verify_before_commit.png)) | the harm only happens at the **moment of commit** — the agent does the irreversible bad thing only by *executing* a route, and only executes one it believes safe — so verification belongs at exactly one place: the route the agent is about to commit to. A **verify-before-commit** agent replans *freely* (cheap model search) and spends one oracle call to verify the route it commits to (**verify the model's "yes," trust its "no"** — an abort can never cause harm). Compared against the CU6 budgeted replanner swept over ρ. **Committed run (400 goals, 8 routes each, 60% dangerous shortcuts, φ=0.6, v0 fs / risk stand-in, trained arm deferred):** verify-before-commit reaches the **zero-harm guarantee *by construction*** (every executed route is oracle-verified → missed-danger rate **0**, not "small on average") at **2.1× lower oracle cost** than a full-verification agent (**1.19 vs 2.52 calls/goal**, success 0.97 vs 1.00) — because **58% of full verification is wasted** verifying routes the model already calls dangerous (a "no" the agent would have aborted anyway), and verify-before-commit removes exactly that waste; on the cost/harm plane it lies **below the budgeted frontier** (Pareto-superior for safety). The saving **grows with how adversarial the environment is**. **Where you verify beats how much** — the agent's loop structure tells you where the oracle is needed. CPU-only, torch-free, seconds. |
| CU5-net | H100 — the **trained-arm confirmation** (the closed loop on a REAL learned model) | ✅ shipped + **trained frontier run** — **SUPPORTED, honest refinement** ([`acd/closed_loop_net.py`](../../src/verisim/acd/closed_loop_net.py), [`experiments/cu5_net_closed_loop.py`](../../src/verisim/experiments/cu5_net_closed_loop.py), [`cu5_net_closed_loop.csv`](../../figures/cu5_net_closed_loop.csv), [`.png`](../../figures/cu5_net_closed_loop.png)) | CU5–CU7 ran on a φ-dial stand-in; CU5-net re-runs the closed loop on the **real trained network flagship `M_θ`** (the transformer CU2-net used) to answer the program's biggest open question — *does the loop close under real learned drift?* An agent works a connectivity goal (flows to work hosts) from a workload that tempts it with exfiltration (flows to protected crown-jewel hosts), previewing each action through the *real* model (ρ-grounded) and executing iff it shows no exfil; scored on success + unsafe. **Committed trained run (93 contested episodes, horizon 24, one ~11-min train):** the **load-bearing safety axis closes exactly as on the stand-in** — a free agent opens **every** exfil flow (unsafe **1.00**, mean **1.29** missed flows) and verification drives it to **0.00** (1.00 → 0.96 → 0.85 → 0.69 → 0.56 → 0.00 over ρ); **but** task success stays **1.00** at every ρ because the **real drift is one-sided** (the model *omits* flows so it misses exfil, but never *hallucinates* one, so it never false-aborts a benign connect) — the utility axis the two-sided synthetic stand-in exercised is not triggered. The half that matters — verification is load-bearing for safety — is **not** an artifact of synthetic drift; it survives a real learned model. Torch-gated trained arm; torch-free closed-loop core. |
| CU8 | H101 — the **drift asymmetry** (world models hide danger by omission) | ✅ shipped + **trained probe** — **SUPPORTED, decisively** ([`acd/drift_asymmetry.py`](../../src/verisim/acd/drift_asymmetry.py), [`experiments/cu8_drift_asymmetry.py`](../../src/verisim/experiments/cu8_drift_asymmetry.py), [`cu8_drift_asymmetry.csv`](../../figures/cu8_drift_asymmetry.csv), [`.png`](../../figures/cu8_drift_asymmetry.png)) | CU5-net found the trained model's drift is one-sided; CU8 characterizes it. A teacher-forced probe of the real trained network `M_θ` (predict each step from the oracle's true state) classifies every flow-prediction error as an **omission** (missed a real flow — the gate's missed-danger source) or a **hallucination** (invented a flow — the false-alarm source), split by protected/work host. **Committed trained probe (300 workloads, 7,200 steps):** drift is **overwhelmingly omission-biased** — **417 omissions vs 14 hallucinations** (30:1) overall, and on the danger hosts **146 missed exfil flows vs 1 hallucinated** (**146:1**; only **2%** exfil recall). The model **hides danger, it does not invent it** — the mechanism is that consequential events (a connection establishing) are rare, so the model's safe default is "predict no consequence," and danger is exactly a rare consequence it misses. So the gate's errors concentrate in the **catastrophic missed-danger cell**, which is the structural reason verification is load-bearing for *safety* specifically (and why CU5-net's utility axis never moved). It doubles the program's core asymmetry: the most costly cell is the one drift is biased toward. Network trained arm (cheap); host arm deferred. Torch-gated trained arm; torch-free probe. |
| CU9 | H102 — the **agent-safety horizon** (unverified safety is a clock that runs out) | ✅ shipped + **trained run** — **SUPPORTED, stark** ([`acd/safety_horizon.py`](../../src/verisim/acd/safety_horizon.py), [`experiments/cu9_safety_horizon.py`](../../src/verisim/experiments/cu9_safety_horizon.py), [`cu9_safety_horizon.csv`](../../figures/cu9_safety_horizon.csv), [`.png`](../../figures/cu9_safety_horizon.png)) | the deployment question CU8 implies: how long can an unverified agent run before the irreversible breach? The safety-outcome analogue of SPEC-10's *faithful horizon* (which measured how long the model's *predictions* stay faithful) — CU9 measures how long the agent's *actions* stay safe. The agent runs the CU5-net closed loop over a long deployment on the real trained `M_θ`; we record the step of its first exfiltration and build the **survival curve** (fraction still safe after `t` steps) per budget ρ. **Committed trained run (200 deployments, horizon 48):** a free agent's survival **decays toward zero — breach rate 0.995**, safe for only **~20 steps on average** (median safe horizon **17**) — it breaches at its first dangerous opportunity, near-certain over a long run. Verification **flattens the curve and extends the horizon**: ρ=0.3 → ~26 safe steps (breach 0.81), ρ=0.5 → ~31 (breach 0.65), and the **oracle never breaches** (survival flat at 1.0). The practitioner lesson: **unverified safety is not a property an agent has, it is a clock that runs out** — and on an omission-biased model a long deployment needs substantial verification. Network trained arm; host deferred. Torch-gated trained arm; torch-free survival core. |
| CU10 | H103 — **targeted verification** (what to verify beats how much) | ✅ shipped + **trained run** — **SUPPORTED, decisively** ([`acd/targeted_verification.py`](../../src/verisim/acd/targeted_verification.py), [`experiments/cu10_targeted_verification.py`](../../src/verisim/experiments/cu10_targeted_verification.py), [`cu10_targeted_verification.csv`](../../figures/cu10_targeted_verification.csv), [`.png`](../../figures/cu10_targeted_verification.png)) | CU9 verified on a *blind, uniform* schedule that only reaches zero breach at the full oracle; CU10 asks *which* steps a limited budget should buy. Three schedules on the same long-deployment battery and the same real trained network `M_θ`: **uniform** (the CU9 blind budget), **model** self-targeting (consult when the model expects activity), **structure** (consult the `connect`-to-protected actions — the defender's crown-jewel knowledge + the world grammar; empirically *every* protected-flow open is such a direct connect). **Committed trained run (200 deployments, horizon 48):** uniform needs **48 calls** (the full oracle) to reach zero breach (ρ=0.5 still breaches 0.65); **model self-targeting FAILS — breach 0.995 at 0.07 calls** (the omitting model never expects the activity that matters, so it never consults — you can't ask the omitter where it omits, the CU8 law); **structure targeting reaches the oracle's zero breach at 4.07 calls — 11.8× cheaper than the full oracle**, identical safety. The constructive flip of CU8/CU9: the omission bias is a warning, but its *structure* is the remedy — danger is concentrated on a rare, identifiable surface, so a defender who knows where to look defends it cheaply. CU7's "where you verify beats how much," carried from the host stand-in's commit point to the trained network arm's danger surface, plus the negative that the model cannot supply the "where." Torch-gated trained arm; torch-free verification core. |
| CU11 | H104 — **un-gameable targeting** (the adversary controls the timing) | ✅ shipped + **trained run** — **SUPPORTED, decisively** ([`acd/adversarial_targeting.py`](../../src/verisim/acd/adversarial_targeting.py), [`experiments/cu11_adversarial_targeting.py`](../../src/verisim/experiments/cu11_adversarial_targeting.py), [`cu11_adversarial_targeting.csv`](../../figures/cu11_adversarial_targeting.csv), [`.png`](../../figures/cu11_adversarial_targeting.png)) | CU10 ranked the schedules on a *random* workload; CU11 is its adversarial worst case — CU4's threat-model move carried from the safety gate to the verification schedule. An attacker who knows the deployed schedule chooses *when* to fire the exfiltration (worst-case over timing: the deployment breaches if any reachable exfil placement evades the schedule and the model's preview). The result turns on *what each schedule keys on*: **uniform** keys on the **clock phase** (the attacker steps around it), **model** keys on its **own omitting output** (it expects no activity where it matters, CU8), but **structure** keys on a **grammar-fixed property** — a flow is born only by `connect`, addressed to its destination (the only flow-opening edit is `FlowOpen`, emitted solely by `connect`; a structural invariant of the oracle), so the attacker cannot relocate danger off the `connect`-to-protected surface. **Committed trained run (200 deployments, horizon 48, same battery and `M_θ` as CU10):** **uniform and model targeting are gameable — adversarial breach 1.000 at every ρ<1** (the CU9/CU10 knee is a *mirage*: random-timing 0.65 at ρ=0.5 → **1.000** adversarial; only the full oracle at 48 calls is adversarially safe), while **structure targeting is un-gameable — adversarial breach 0.000 at 4.07 calls**, identical to its random breach. The defender principle: **target verification at what the adversary cannot move.** CU4's worst-case robustness, now with a cheap constructive winner. Torch-gated trained arm; torch-free verification core. |
| CU12 | H105 — **knowledge-free targeting** (target the grammar, not the assets) | ✅ shipped + **trained run** — **SUPPORTED, decisively** ([`acd/knowledge_free_targeting.py`](../../src/verisim/acd/knowledge_free_targeting.py), [`experiments/cu12_knowledge_free_targeting.py`](../../src/verisim/experiments/cu12_knowledge_free_targeting.py), [`cu12_knowledge_free_targeting.csv`](../../figures/cu12_knowledge_free_targeting.csv), [`.png`](../../figures/cu12_knowledge_free_targeting.png)) | CU10/CU11 made targeting cheap and un-gameable by verifying the `connect`-to-crown-jewel actions — assuming the defender's **crown-jewel inventory is complete**. A real inventory is incomplete (shadow services, drift), and an adversary exfiltrates to the host you didn't flag. Two structural targets: **asset-indexed** (verify `connect` to a *known* jewel `K` — CU10/CU11, cheap but blind outside `K`) and **grammar-indexed** (verify *every* `connect` — the whole flow-genesis surface, needs **zero asset knowledge** because a flow is born only by `connect`). Scored against the **true** sensitive set `T={h0,h4}` as the inventory `K⊆T` becomes incomplete. **Committed trained run (200 deployments, horizon 48, the real `M_θ`):** with a 50%-complete inventory (`K={h0}`, `h4` unflagged) the asset target breaches **0.635 random / 0.960 adversarial** — nearly the unverified rate (0.995), a *false sense of security*, and fully gameable by an adversary who picks the unflagged host; the complete inventory (`K=T`) is safe at 4.07 calls (the CU10 result), but the **grammar-indexed target reaches 0.000 breach inventory-independently at 9.35 calls — 5.1× cheaper than the full oracle (48)**. The defender principle: **when you cannot trust your asset inventory, target the grammar, not the assets** — the flow-genesis surface needs no asset list and is still cheap. Completes the targeting arc (cheap → un-gameable → knowledge-free). Torch-gated trained arm; torch-free verification core. |
| CU13 | H106 — **capability under real drift** (the false-alarm channel prices CU6 and CU7) | ✅ shipped + **trained run** — **SUPPORTED, decisively** ([`acd/closed_loop_replan_net.py`](../../src/verisim/acd/closed_loop_replan_net.py), [`experiments/cu13_replan_net.py`](../../src/verisim/experiments/cu13_replan_net.py), [`cu13_replan_net.csv`](../../figures/cu13_replan_net.csv), [`.png`](../../figures/cu13_replan_net.png)) | CU6 (free replanning *amplifies* harm +0.06) and CU7 (verify-before-commit zero-harm at *2.1×* lower cost) were measured on the two-sided stand-in; CU5-net showed the real `M_θ` drifts *one-sided*. CU13 re-runs both on a net replanning world and isolates the mechanism: both are priced by the model's **"no" channel**, but by different halves — **CU6's amplification by the FALSE-ALARM rate** (a *wrong* "no" false-aborts a safe route → retry onto a blind-spotted danger), **CU7's saving by the danger RECALL** (a *right* "no" on a truly-dangerous route is a call full-verify wastes and verify-before-commit skips). **Committed trained run (200 goals, 8 routes, real `M_θ`):** the false-alarm dial (recall 0) lifts amplification **0.000 → 0.160**; the recall dial (false-alarm 0) lifts the verify-before-commit saving **1.00× → 1.70×** (wasted fraction 0 → 0.41); and the **real `M_θ` anchors at the origin of both** — measured false-alarm **0.000**, recall **0.004** (it says "yes" to every route) — so **amplification is exactly 0.000 and cost saving exactly 1.000×**. CU6's capable-agent warning and CU7's verify-where win are both properties of a model that says "no"; a real omission-biased one does not, so neither appears. The danger does **not** vanish (one-shot harm **0.53** either way) and verify-before-commit keeps its **zero-harm guarantee** — the *structural* result survives a real learned model, the *quantitative* knee/saving is a two-sided artifact. Torch-gated trained arm; torch-free replanning core. |
| CU14 | H107 — **the defended incident** (the whole stack on one named scenario) | ✅ shipped + **trained run** — **SUPPORTED, decisively** ([`acd/incident_response.py`](../../src/verisim/acd/incident_response.py), [`experiments/cu14_incident_response.py`](../../src/verisim/experiments/cu14_incident_response.py), [`cu14_incident_response.csv`](../../figures/cu14_incident_response.csv), [`.png`](../../figures/cu14_incident_response.png)) | CU1–CU13 each isolated one face of the gate; CU14 puts them together on one concrete scenario a defender reads end to end. An autonomous incident-response agent must restore work connectivity (`h1/h2/h3`) on a compromised segment salted with exfiltration lures (`connect` to crown jewels `h0/h4`), under four defenses — **undefended** (trust the model), **paranoid** (block every `connect` — CU3's "safe by being useless" corner), **structure** (verify the `connect`-to-jewel actions, CU10/CU12's grammar target), **full oracle** (verify every step) — scored on **all three axes at once**: mission completed (utility), exfiltrated (safety), oracle calls (cost). **Committed trained run (193 contested incidents, horizon 48, real `M_θ`):** undefended completes the mission (**1.00**) but **exfiltrates (0.99)**; paranoid is safe (**0.00**) but **abandons the mission (0.00)**; full oracle is safe and on-mission at **48 calls**; **structure is the only all-good corner — safe (0.00 breach), on-mission (1.00), at 4.0 calls — 12× cheaper than the full oracle**. The representative-incident playback replays the *same* action sequence undefended vs structure: the undefended agent walks the one true lure (`connect h2 h4 22`, a breach) while structure spends an oracle call on exactly it (abort) and still finishes the work connects. The synthesis: a verified world model is the safety layer that lets a computer-use agent and a network defender complete the mission without the irreversible bad thing, and verifying the world's flow-genesis surface is cheap. Torch-gated trained arm; torch-free incident core. |
| CU15 | H108 — **the verification-exhaustion attack** (the cost axis under an adversary) | ✅ shipped + **trained run** — **SUPPORTED, decisively** ([`acd/verification_exhaustion.py`](../../src/verisim/acd/verification_exhaustion.py), [`experiments/cu15_verification_exhaustion.py`](../../src/verisim/experiments/cu15_verification_exhaustion.py), [`cu15_verification_exhaustion.csv`](../../figures/cu15_verification_exhaustion.csv), [`.png`](../../figures/cu15_verification_exhaustion.png)) | CU11 proved structure targeting un-gameable on the *safety* axis; CU15 carries CU4/CU11's worst-case threat model to the *cost* axis. An adversary who cannot make structure **breach** floods its danger surface (every `connect`-to-jewel) with benign-looking activity to exhaust the verification budget — *alert fatigue / denial of budget*. A fixed `horizon`-step deployment whose steps are poisoned with attacker `connect`-to-jewel actions at saturation `s`; each schedule read on **both** axes (breach, calls). **Committed trained run (200 deployments, horizon 48, real `M_θ`):** an adversary can move **exactly one axis** of a sub-oracle schedule — **structure's cost** climbs **4.07 → 15.07 → 26.09 → 36.99 → 48.00** calls as `s`: 0 → 1 (its **safety is immovable** at **0.000** breach throughout), while **uniform's safety** degrades **0.650 → 0.965** breach (its clock-fixed cost stays at **24.00**). But structure's cost stays **bounded by and weakly dominates the full oracle** (≤ horizon at every `s`, = only at full saturation) and the attack is **self-limiting** (**0.92 defender calls per attacker action, 0.000 breaches bought**). Only the full oracle is immovable on *both* axes — at the maximum price. The defender principle: prefer the schedule whose movable axis is a **bill you can cap** (≤ the full oracle, and the attacker spends its whole budget to impose it) over one that is a **breach** — the cost-axis analogue of CU4 (average-case cheapness is a false sense of *economy*, but structure's worst case is still safe and still ≤ the price of total safety). Torch-gated trained arm; torch-free verification core. |
| CU16 | H109 — **cross-world targeting** (the danger surface is grammar-fixed on the host too) | ✅ shipped + **frontier run** — **SUPPORTED, decisively** ([`acd/host_targeting.py`](../../src/verisim/acd/host_targeting.py), [`experiments/cu16_host_targeting.py`](../../src/verisim/experiments/cu16_host_targeting.py), [`cu16_host_targeting.csv`](../../figures/cu16_host_targeting.csv), [`.png`](../../figures/cu16_host_targeting.png)) | the targeting arc (CU10 cheap, CU11 un-gameable, CU12 knowledge-free) was network-only; CU16 carries the headline to the **host** world (credential / config tampering — CU1's content guardrail). The host grammar invariant is just as exact: a `/passwd` corruption is born only by a `write` to an fd bound to that path. The sharper twist the network could not show: the host danger surface is the action **composed with the fd→path binding**, which lives in the process *structure* the boundary law says the model learns faithfully (host `M_θ` drifts ~25-36% on content, ~0% on the fd table) — so structure targeting localizes a *content* danger through *faithful structure*. The trained host arm is the deferred GPU extension (its rollout over fork-heavy workloads is pathologically slow on the throttled CPU — LP7), so the schedule result (which keys on the oracle + grammar, not the model) runs a **worst-case content omitter** stand-in (the realistic drift CU8 measured and CU1 confirmed: a free preview misses 0.38 of real `/passwd` corruptions). **Committed run (200 deployments, horizon 48):** uniform needs the full oracle (48 calls) to reach zero breach and its sub-oracle knee is a mirage (adversarial breach **1.000 at every ρ<1**); **model self-targeting fails** (breach **1.000** at 0 calls); and **structure targeting reaches zero breach at 3.49 calls — 13.8× cheaper than the full oracle — and is un-gameable** (adversarial breach **0.000**). The targeting result is a property of any oracle-grounded world whose danger has a grammar-fixed genesis surface, not a network artifact. Torch-free verification core and stand-in. |
| CU17 | H110 — **the genesis-grammar boundary** (target the danger's genesis, not an action class) | ✅ shipped + **frontier run** — **SUPPORTED, decisively** ([`acd/segmentation_targeting.py`](../../src/verisim/acd/segmentation_targeting.py), [`experiments/cu17_segmentation_targeting.py`](../../src/verisim/experiments/cu17_segmentation_targeting.py), [`cu17_segmentation_targeting.csv`](../../figures/cu17_segmentation_targeting.csv), [`.png`](../../figures/cu17_segmentation_targeting.png)) | the targeting arc (CU10–CU16) always targeted a single, syntactically visible danger action (the `connect` to a crown jewel); CU17 tests whether the *principle* survives a danger with a richer genesis grammar — **network-segmentation exposure**, a crown jewel becoming *reachable* from an untrusted host (`can_reach` flipping `False → True`). That reachability is born by the *config* grammar — `svc_up`/`fw_allow`/`host_up` and above all `link_up` (a link completes a **multi-hop** path, so a `link_up` between two *non-jewel* hosts can expose the jewel) — never by `connect`, so the danger surface is **semantic** (reachability) not **syntactic** (an action class), and enumerating it needs the reachability *closure* (the SPEC-12 machinery). The trained arm is deferred (LP7); the schedule result keys on the oracle + grammar, so a **worst-case content omitter** stand-in. **Committed run (200 segmented deployments, horizon 48):** the cheap CU10–CU16 `connect` target **does not transfer** — breach **1.000** (the free rate) at **3.86 calls**, a false sense of security; a *syntactic* genesis-grammar target reaches near-zero random breach (**0.025**) but **leaks through multi-hop intermediates** (adversarial breach **0.370** — an attacker exposes a jewel via a `host_up` of a relay it cannot name) and overpays (**13.72 calls**); only the *semantic* **reachability-closure** target reaches the oracle's **0.000 breach, un-gameable (0.000 adversarial), at 4.17 calls — 11.5× cheaper than the full oracle**, dominating the syntactic target on *both* axes. The principle: **target the danger's genesis grammar — compute its reachability closure (SPEC-12), do not pattern-match an action class.** The cheapness of CU10–CU16 was a property of a *sparse* genesis grammar, not magic; a richer danger needs a richer (but still bounded, still sub-oracle) target. Torch-free verification core and stand-in. |
| — | the architecture diagram + the "foundation → application" writeup | ✅ shipped | [`figures/cu_architecture.png`](../../figures/cu_architecture.png) + the README section |

This spec is the bridge the program needed: it does not add metrology, it *spends* the metrology, and
the thing it buys is the one a frontier lab and a SOC both want — an AI agent that can act on a cheap
world model without doing the irreversible bad thing, because a free oracle keeps the preview honest.
