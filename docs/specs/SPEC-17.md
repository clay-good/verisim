# SPEC-17 — Causal & Counterfactual World Models: The Oracle as an Exact Structural Causal Model

**Method specification: the program's single most-cited open result is H5 — the counterfactual lift is
*world- and objective-dependent*: it exists for the contrastive representation (EN9) and for the
distributed world's fault-branch replay (ED6), but is *null* for plain next-state supervision in the
on-policy-complete worlds (EN6 network, EH6 host). SPEC-17 gives that mixed result the only formalism
that explains it cleanly — Pearl's ladder of causation — by observing that a deterministic, resettable,
seedable oracle is not *like* a Structural Causal Model; it *is* one, exactly and for free. From that one
identification two things follow that no oracle-free domain can do: (1) verisim can compute all three
rungs — observe, intervene (`do`), and the hardest rung, **counterfactual** (abduct the exogenous state,
intervene, predict) — exactly, so it can generate true counterfactual ground truth; and (2) it can
finally **disentangle counterfactual *branching* from fault *coverage*** — the named confound that
qualifies the ED6 positive — by constructing interventions matched on coverage. This spec is the
do-calculus reading of H5 and the experiment that resolves its open caveat.**

> **✅ SHIPPED (pure-oracle core + the confound-resolver + the CoDA contrast) — METHOD SPEC, 2026-06
> (CX0 H60 + CX1 H61 on all four worlds + CX5 H64 on the real system oracles + CX3 H62 — the
> matched-coverage cut: **H62 REFUTED, the ED6 lift was fault coverage, not counterfactual branching**
> — + CX4 H63 — exact-oracle vs learned-model (CoDA) counterfactual augmentation: **H63 SUPPORTED, a
> learned augmenter's 6%-valid counterfactuals corrupt training below baseline while the exact oracle's
> lift it, validating SPEC §1.1's unverifiability thesis**; only the CX2 three-world *learned* lift
> remains deferred). See §8 for the status table.**
> A *cross-world method* in the lineage of [SPEC-8](./SPEC-8.md)
> (oracle-grounded self-supervision) and [SPEC-12](./SPEC-12.md) (planning over the world model): it
> invents **no new world** — it runs on the SPEC-5 network, SPEC-6 host, and SPEC-7 distributed worlds
> and their shipped oracles — and **no new oracle**. What it adds is a *formal layer*: the recognition
> that [`ReferenceNetworkOracle`](../../src/verisim/netoracle/reference.py),
> [`ReferenceDistOracle`](../../src/verisim/distoracle/reference.py) and their kin are SCMs whose
> structural equations are their `step` functions and whose exogenous noise is the seed/clock the DES
> consumes — so the abduction-action-prediction recipe for Pearl's third rung is *executable*. The
> counterfactual-branch factory the program already ships
> ([`netdata/negatives.py:counterfactual_branches`](../../src/verisim/netdata/negatives.py)) is, in this
> reading, a *partial* rung-3 generator: it does intervention from the *same* state but does not yet
> abduct an exogenous latent. SPEC-17 completes it and uses it to settle H5's open question.

Read [SPEC.md §3 (RQ4)](./SPEC.md), [SPEC.md §9 (H5 — read the full mixed result)](./SPEC.md),
[SPEC.md §7 (counterfactual result)](./SPEC.md), [SPEC-7 §10.1 + the ED6 milestone](./SPEC-7.md) (the
fault-branch counterfactual and its honest branching-vs-coverage caveat), and the shipped
counterfactual machinery in [`netdata/negatives.py`](../../src/verisim/netdata/negatives.py) and
[`experiments/ed6.py`](../../src/verisim/experiments/ed6.py).

---

## 0. One-paragraph thesis

Pearl's ladder of causation has three rungs — **association** (seeing: `P(y|x)`), **intervention**
(doing: `P(y|do(x))`), and **counterfactual** (imagining: "would `y` have happened had `x` been `x'`,
*given what actually happened*") — and the third rung is unreachable from data alone: it requires a
Structural Causal Model, because answering it means *abducting* the exogenous noise that produced the
factual outcome, *intervening* on it, and *predicting* the alternative. Every oracle-free world model is
stuck below rung 3; it has no SCM and no way to get exact counterfactual ground truth, which is why
"causal poverty" is a named open problem of the field ([SPEC.md §1](./SPEC.md), point 3). Verisim's
claim is that **a deterministic, resettable, seedable oracle is an exact SCM**: its `step(s, a)` *is* the
set of structural equations, and the seed/clock the discrete-event oracle consumes *is* the exogenous
noise `U`. So verisim can climb all three rungs exactly — and the third one for free, because
resettability *is* abduction (re-run from `(seed, t)` recovers the exact `U` that produced the factual
state) and intervention *is* a one-action edit before the re-run. This (1) turns the mixed H5 result into
a one-line do-calculus statement — counterfactual training helps exactly when the world has
**off-policy exogenous state** the on-policy distribution `P(U)` underrepresents, and adds nothing where
on-policy supervision already covers `P(s'|do(a))` — and (2) lets verisim **resolve the program's named
ED6 confound**: ED6's fault branches are fault-*heavier* than the on-policy control, so its lift
conflates counterfactual *branching* with fault *coverage* ([SPEC-7](./SPEC-7.md), ED6 caveat). With an
exact SCM verisim can construct counterfactual branches **matched on fault coverage** to the control and
re-run the ED6 design — isolating whether counterfactual *structure per se* carries the lift, or whether
ED6 was coverage all along. Both branches are bankable (SPEC §10.1): a matched-coverage lift confirms
counterfactual structure is the active ingredient; a matched-coverage *null* re-attributes ED6 to
coverage (tied to H21) and *strengthens* the off-policy-hidden-state mechanism rather than weakening it.

---

## 1. Why now: H5 is the program's most-qualified result, and it has a formalism waiting

H5 (counterfactual lift) is the only headline hypothesis in [SPEC.md §9](./SPEC.md) whose result is a
*paragraph* of caveats rather than a verdict. Read it carefully and it says four things at once:

1. **A lift exists for the contrastive representation** — EN9: oracle counterfactual negatives nearly
   double VICReg's branch-retrieval top-1 ([`experiments/en9.py`](../../src/verisim/experiments/en9.py),
   [`en9_negatives.py`](../../src/verisim/experiments/en9_negatives.py)).
2. **No lift for plain next-state supervision in on-policy-complete worlds** — EN6 (network,
   [`en6.py`](../../src/verisim/experiments/en6.py)) and EH6/H16 (host,
   [`eh6_counterfactual.py`](../../src/verisim/experiments/eh6_counterfactual.py)): a counterfactual is
   "just another labeled transition" there; matched-volume trajectory data ties it.
3. **A decisive lift in the distributed world** — ED6: fault-branch replay beats the matched-*volume*
   on-policy control, intervention-exact **0.51 vs 0.25** (base 0.06), medium-recall **0.56 vs 0.22**,
   disjoint CIs ([`experiments/ed6.py`](../../src/verisim/experiments/ed6.py)).
4. **An honest caveat that the program flags as open** — ED6's branches are *fault-heavier* than the
   control, so the lift "conflates counterfactual *branching* with the fault *coverage* it carries …
   the branching-vs-coverage disentanglement is future work, tied to H21."

The program already names the mechanism that reconciles 1–3: counterfactual training helps where the
world has **off-policy hidden state** the on-policy distribution underrepresents (the distributed
*medium*: partition/crash/in-flight), and adds nothing where on-policy supervision already covers the
dynamics. That mechanism is a *causal* statement in everything but vocabulary. Stated in Pearl's:
on-policy supervision gives `P(s'|do(a))` over the support `P(U)` the policy visits; a counterfactual
demands `P(s'_{a'}|s, a, U)` for the abducted `U` — and only diverges from what supervision already
knows when `U`'s relevant support lies *off* the on-policy distribution. The network and host worlds are
**on-policy `U`-complete** (their nondeterminism is seeded-but-thin and the policy spans it); the
distributed world is not (the fault medium is exogenous state the light-fault policy rarely enters). The
do-calculus says the lift *must* be hidden-state-dependent — which is exactly what EN6/EH6/ED6 found
empirically, before the formalism was written down. SPEC-17 writes it down and then uses it to do the
one thing the empirical result could not: cut branching from coverage.

The second reason to do this now is the artifact. The counterfactual-branch generator already ships
([`counterfactual_branches`](../../src/verisim/netdata/negatives.py)) but does intervention only from the
*same* state `s` (`do(a')` from `s`) — that is rung **2**, not rung 3. A true counterfactual abducts the
exogenous `U` that produced a *factual* downstream state and asks the alternative under *that same* `U`.
The oracle's resettability makes that abduction exact and free. SPEC-17 is the spec that turns the
shipped rung-2 generator into a rung-3 one and measures whether the difference matters.

---

## 2. The identification: a deterministic resettable oracle *is* an SCM (and why that is the whole spec)

A Structural Causal Model is a tuple `(U, V, F, P(U))`: exogenous variables `U` (noise, drawn from
`P(U)`), endogenous variables `V` (the state), structural equations `F` (`V := f(parents, U)`), and the
noise distribution. The three rungs are then:

- **Rung 1 — association** `P(V)`: sample `U ~ P(U)`, run `F`, read off `V`. *Verisim:* roll the oracle
  on the on-policy action distribution — exactly the EN/EH/ED trajectory data.
- **Rung 2 — intervention** `P(V | do(X=x))`: replace the equation for `X` with `X := x`, sample
  `U ~ P(U)`, run `F`. *Verisim:* `oracle.step(s, a')` — set the action, take the truth. This is what
  [`counterfactual_branches`](../../src/verisim/netdata/negatives.py) ships today.
- **Rung 3 — counterfactual** `P(V_{X=x'} | V=v, X=x)`: **abduct** `P(U | V=v, X=x)` (recover the noise
  consistent with the factual observation), **intervene** `do(X=x')` on the abducted model, **predict**
  `V` by re-running `F` with the *same* `U`. *Verisim:* re-run from `(seed, t)` — which recovers the
  exact `U` — with one action/fault flipped. The ED6 recipe ("re-run from `(seed, t)` with one fault
  flipped", [SPEC-7 §10.1](./SPEC-7.md)) is *already* abduction-action-prediction; it just was not named
  as such.

The identification is exact, not metaphorical, because the oracle satisfies the SCM contract literally:

- **`F` is `step`.** [`ReferenceNetworkOracle.step`](../../src/verisim/netoracle/reference.py) and
  [`ReferenceDistOracle.step`](../../src/verisim/distoracle/reference.py) are pure functions of
  `(state, action)` — the structural equations, with the state factored into the atomic facts
  `negatives.py` already enumerates (host up/down, service, firewall, link, flow; the distributed
  replica/partition/crash/in-flight medium).
- **`P(U)` is the seed/clock.** The distributed DES is "a pure function of `(state, action)`"
  ([reference.py](../../src/verisim/distoracle/reference.py)) once the fault/time medium is fixed; the
  medium *is* the exogenous draw. The network oracle's `advance` clock is its thin `U`.
- **Abduction is reset+replay.** Because the oracle is resettable and the seed is recorded, recovering
  the exact `U` behind a factual state is `O(1)`: re-run from `(seed, t)`. In an oracle-free SCM
  abduction is an *inference* problem (often intractable, never exact); here it is a *lookup*.

> **This is the asymmetry of [SPEC §2](./SPEC.md), sharpened to its strongest form.** Oracle-free world
> models are barred from rung 3 not by engineering but by epistemics — they have no SCM and cannot
> abduct exact noise. Verisim is barred from nothing: it owns the SCM, so it generates *exact
> counterfactual ground truth*. Every result below is downstream of this one identification.

The one honest subtlety, named up front: the *reference* oracle is an SCM of a *model* of reality, not
of reality ([SPEC §2.1](./SPEC.md)). The system oracles (SPEC-11 host `SandboxOracle`, SPEC-7's Tier-B
[`distoracle/system.py`](../../src/verisim/distoracle/system.py)) are SCMs of the real execution — and
they are *still resettable and seedable* (the DST scheduler's seed is `U`), so the rung-3 recipe
survives the move to the system oracle. That is the H4 (mechanism-survives-reality) reading of this spec.

---

## 3. The architecture: three rungs, one factory, one matched-coverage cut

```
                 EXOGENOUS U                STRUCTURAL EQUATIONS F            ENDOGENOUS V
   rung 1  obs   seed s ~ P(U)  ──────────▶  oracle.step(·)  ──────────────▶  trajectory  (EN/EH/ED data)
   rung 2  do    seed s ~ P(U)  ──do(a')──▶  oracle.step(s, a')  ──────────▶  branch       (counterfactual_branches, shipped)
   rung 3  cf    ABDUCT Û from (seed,t)  ──do(a')──▶  re-run F with Û  ─────▶  TRUE counterfactual  (NEW: cx_abduct.py)
                                          │
              ┌───────────────────────────┴───────────────────────────────────┐
              │  MATCHED-COVERAGE CUT (resolves the ED6 confound, §1 point 4)   │
              │  construct cf branches matched to the on-policy control on      │
              │  fault-coverage statistics  ⇒  any residual lift is BRANCHING,  │
              │  not coverage (else ED6 was coverage; both banked, §5)          │
              └────────────────────────────────────────────────────────────────┘
```

Four commitments, each tied to a measured verisim fact and the do-calculus reading of H5:

1. **The counterfactual is rung 3, not rung 2.** The shipped generator does `do(a')` from `s` (rung 2);
   SPEC-17 adds abduction (`Û` from `(seed, t)`), so the training target is the *true counterfactual*
   "what the factual rollout would have become under the same noise had one action differed." The CX1
   ablation measures whether the rung-2→rung-3 upgrade changes anything — a clean test of whether
   abduction (as opposed to mere alternative-action branching) is load-bearing on these worlds.
2. **The lift is pre-registered as hidden-state-dependent.** Per the §1 mechanism: null on the
   on-policy-complete worlds (network/host) is the **expected, banked** outcome consistent with EN6/EH6,
   *not* a failure; the distributed world (off-policy exogenous medium) is where rung-3 supervision is
   predicted to pay (CX2). This is H5 restated as a falsifiable do-calculus prediction (H60/H61).
3. **Coverage is held fixed to isolate branching.** The load-bearing experiment (CX3) constructs
   counterfactual branches **matched on fault-coverage** to the on-policy control, re-running ED6 — the
   only way to cut counterfactual *structure* from fault *coverage*. The matching statistic is the
   `_medium` the ED6 code already computes ([`experiments/ed6.py`](../../src/verisim/experiments/ed6.py)).
4. **The metric is a counterfactual `H_ε`.** The headline dependent variable is divergence on held-out
   *counterfactual* queries, scored against the oracle's *exact* counterfactual ground truth (the rung-3
   re-run) — RQ4 made a measured quantity with units, the way `H_ε(ρ)` is for rung 1.

---

## 4. The load-bearing assumption (and its banked alternative)

The whole spec rests on one empirical claim: that **abduction adds information over intervention on
exactly the worlds with off-policy hidden state, and nothing elsewhere.** That is testable, not assumed
— it is CX1×CX2 crossed. The honest pre-registration:

- **If rung-3 abduction beats rung-2 intervention only on the distributed world**, the SCM framing is
  vindicated: the exogenous medium is the thing abduction recovers and supervision-from-data cannot, and
  the network/host null is the do-calculus prediction confirmed. This is the hoped-for branch.
- **If rung-3 ties rung-2 even on the distributed world**, the lift was never about abduction — it was
  about *coverage of the fault medium*, reachable by rung-2 `do(a')` branches alone. This is a clean,
  bankable result: it says the active ingredient is *visiting* the off-policy state (rung 2 suffices),
  not *counterfactually re-deriving* it (rung 3) — and it folds directly into H21 (fault-injection beats
  fault-free at equal volume). Either way the program learns precisely which rung the distributed lift
  lives on, which no prior experiment could say.

This is the program's epistemic engine at design time: the assumption that would silently sink an
oracle-free causal-RL paper is, here, a measurement with a forward move banked on the failing branch.

---

## 5. Hypotheses (pre-registered; continuing the global H-ID space past SPEC-16's H59)

- **H60 — the oracle is an exact SCM and rung-3 counterfactuals are exact and free.** For every world
  with a deterministic resettable seedable oracle, abduction-action-prediction (re-run from `(seed, t)`
  with one action/fault flipped) yields the *exact* counterfactual next state — verified by comparing
  two independent recoveries of `U` and confirming bit-identical counterfactual outcomes. *Refuted if*
  the recovered `U` does not reproduce the factual rollout bit-for-bit (the oracle's nondeterminism is
  not fully seed-captured — itself a clean, important result that bounds which worlds admit exact rung 3,
  and re-points the spec at the seedable subset). Tested as **CX0**.

- **H61 — the counterfactual lift is hidden-state-dependent (the do-calculus reading of H5).** Training
  on rung-3 counterfactual targets lifts held-out counterfactual `H_ε` **iff** the world has off-policy
  exogenous state the on-policy distribution underrepresents: null on the on-policy-complete worlds
  (network EN6, host EH6), positive on the distributed world (off-policy fault medium, ED6). *Refuted if*
  the lift appears on the on-policy-complete worlds too (counterfactual structure helps even where
  supervision is complete — a stronger, surprising positive that would generalize the result beyond
  hidden state) *or* is null on the distributed world under the exact rung-3 target (the ED6 positive
  was an artifact of the rung-2 generator). The null on network/host is the **expected, banked** branch,
  consistent with EN6/EH6 — not a failure. Tested as **CX1** (rung-2 vs rung-3) × **CX2** (the
  three-world sweep).

- **H62 — counterfactual *branching* carries the ED6 lift, net of coverage (THE confound-resolver).**
  When counterfactual fault branches are constructed **matched on fault-coverage** (the `_medium`
  statistic) to the on-policy volume control, the `+counterfactual` arm *still* beats the control on
  held-out intervention-exact and medium-recall with disjoint CIs — i.e. counterfactual *structure per
  se*, not the fault coverage it happened to carry, is the active ingredient. *Refuted if* matched
  coverage erases the lift (the control ties `+counterfactual`) — in which case **ED6 was coverage, not
  counterfactual structure**, the program's named caveat resolves *against* branching, and the result
  re-attributes to H21 (fault coverage at equal volume). Both branches resolve the SPEC-7 open caveat
  cleanly and are banked. Tested as **CX3** (the matched-coverage ED6 re-run).

- **H63 — exact-oracle counterfactual augmentation beats learned-model augmentation (the CoDA contrast).**
  Counterfactual data augmentation grounded by the *exact* oracle (rung-3 branches with verified noise)
  improves held-out counterfactual `H_ε` more, per augmented sample, than CoDA-style augmentation
  grounded by a *learned* local model of the same dynamics — because the oracle's counterfactuals are
  causally valid by construction while the learned model's inherit its drift (the [SPEC §1.1](./SPEC.md)
  unverifiability the whole program exists to remove). *Refuted if* a learned-model augmenter matches the
  oracle's at equal sample count (the dynamics are easy enough that a learned local model is already
  causally valid — a result about *which* worlds need the exact oracle vs a good-enough learned one).
  Tested as **CX4**.

- **H64 — the SCM framing is a cross-world method (fork).** The rung-3 recipe and the matched-coverage
  cut transfer across all three shipped worlds with no per-world causal-discovery step, because the SCM
  is *given* (the oracle), not *learned* (the standing assumption of neural-causal-model work, Ke et
  al.) — so verisim skips structure learning entirely and the method's reach is exactly the set of
  seedable oracles (H60). *Refuted if* a world admits a resettable oracle but *not* an exact rung 3
  (H60 fails there), narrowing the method to the seed-complete subset — itself the precise statement of
  the method's domain. Tested as **CX5** (deferred fork; runs after the CX3 confound-resolver lands).

---

## 6. Experiments (prefix **CX** — counterfactual / causal; world noted per experiment)

Each follows the house template: a `Config` dataclass with `from_json_file`, a CLI entry point, a JSONL
record stream, a `plot_*.py` emitting a committed `.png` + `.csv`, regenerable from `reproduce.sh`,
deterministic and seeded ([SPEC-2 §12](./SPEC-2.md)). All reuse the shipped oracles, the
[`negatives.py`](../../src/verisim/netdata/negatives.py) factory, and the ED6/EN6/EH6 training harnesses
— SPEC-17 adds the abduction generator, the matched-coverage sampler, and these harnesses. The non-CoDA
block is dependency-free; CX4's learned-model augmenter is the one torch-gated piece.

- **CX0 — the oracle is an exact SCM (H60).** For each world, sample factual rollouts; abduct `Û` from
  `(seed, t)`, re-run, and assert the recovery is bit-identical to the factual state (two independent
  recoveries agree); then flip one action and emit the exact counterfactual. Report the
  abduction-exactness rate per world (expected 1.0 on the reference oracles; < 1.0 names any
  seed-incomplete nondeterminism). **The gate that licenses rung 3.** `experiments/cx0.py`,
  `configs/cx0.json`.

- **CX1 — rung 2 vs rung 3: does abduction add anything? (H61).** On each world, train the matched flat
  `M_θ` (the ED6/EN6 arm) three ways at matched count: `trajectory` (base), `+intervention` (rung-2
  `do(a')` branches from `s` — the shipped [`counterfactual_branches`](../../src/verisim/netdata/negatives.py)),
  `+counterfactual` (rung-3 abducted branches — NEW). Held-out counterfactual `H_ε`. Isolates whether
  abduction (rung 3) beats mere alternative-action branching (rung 2). `experiments/cx1.py`.

- **CX2 — the three-world hidden-state sweep (H61, the headline).** Run CX1's `+counterfactual` arm vs
  the matched-*volume* trajectory control on network (EN6 apparatus), host (EH6 apparatus), and
  distributed (ED6 apparatus). Report counterfactual `H_ε` lift with bootstrap CIs per world, against the
  do-calculus prediction: null on network/host, disjoint-positive on distributed. **The figure that
  shows the lift tracking off-policy hidden state.** `experiments/cx2.py`.

- **CX3 — the matched-coverage cut (H62, the confound-resolver).** The ED6 re-run with one change:
  `+counterfactual` branches are subsampled/constructed to **match the on-policy control on the
  `_medium` fault-coverage statistic** ([`ed6.py`](../../src/verisim/experiments/ed6.py)). Three arms,
  matched count *and* matched coverage; head-to-head on intervention-exact and medium-recall. Any
  residual lift is counterfactual *branching*; a tie attributes ED6 to coverage. **The experiment that
  settles the program's named ED6 caveat (tied to H21).** `experiments/cx3.py`.

- **CX4 — exact-oracle vs learned-model counterfactual augmentation (H63, the CoDA contrast).** Compare
  rung-3 oracle augmentation against a CoDA-style learned-local-model augmenter (Pitis et al.) at equal
  augmented-sample count, on the distributed world (where the lift exists). Reports counterfactual `H_ε`
  per augmented sample and the *causal-validity rate* of each augmenter's samples (the oracle's is 1.0
  by construction; the learned model's is what it is). `experiments/cx4.py`. *(Torch-gated; `skipif`
  with disclosure, never counted as a result when skipped.)*

- **CX5 — system-oracle fork (H64, ✅ SHIPPED).** Re-runs the CX0 abduction gate (and rung-3
  exactness) on the system oracles — the real `/bin/sh` `SandboxOracle` (filesystem) and the Tier-B
  [`distoracle/system.py`](../../src/verisim/distoracle/system.py) (distributed) — confirming the rung-3
  recipe survives the move from a model-of-reality SCM to a reality SCM (the H4 reading). The
  *buildable* part (CX0 on the system oracles, pure-oracle) shipped now, ahead of the CX2 learned arm;
  shipped as a single [`experiments/cx5.py`](../../src/verisim/experiments/cx5.py) over both system
  oracles. Result (§8): abduction + rung-3 are bit-exact on both (H64 transfer) — the SY4 seal and the
  DST seeded scheduler are what make the real system an exact SCM.

---

## 7. What is confidently buildable now vs gated on a result

The instruction — *build what is confidently positive, experiment on what is not* — maps onto the
dependency order:

- **Confidently buildable now (the machinery exists, the result is near-certain):**
  - The **rung-3 abduction generator** (CX0's apparatus). The oracles are resettable and seeded; the
    ED6 recipe already re-runs from `(seed, t)`. Abduction is a *lookup*, not a bet; CX0 is a build that
    near-certainly returns rate 1.0 on the reference oracles (and any < 1.0 is itself the finding).
  - The **matched-coverage sampler** (CX3's apparatus). The `_medium` statistic is already computed in
    [`ed6.py`](../../src/verisim/experiments/ed6.py); matching the control on it is deterministic
    bookkeeping over the shipped counterfactual factory.
- **Gated on CX0 (the §2 identification check):** whether a given world admits *exact* rung 3 (H60). The
  reference oracles near-certainly do; the system oracles (CX5) are the genuine question.
- **The genuine bets (must be measured):** CX1 (does abduction beat intervention), CX2 (the hidden-state
  prediction), CX3 (the confound-resolver — the program's open caveat), CX4 (the CoDA contrast). CX2 is
  *high-confidence* — the do-calculus and EN6/EH6/ED6 all point the same way — but "the prediction holds
  under the exact rung-3 target with disjoint CIs" is exactly the claim the program measures before
  asserting. CX3 is the one with no safe prior: branching-vs-coverage is *genuinely open*, and either
  outcome is a clean result.

**Recommended build order:** CX0 (license rung 3) → CX1 (rung-2 vs rung-3) → CX2 (the three-world
headline) → CX3 (the confound-resolver) → CX4 (CoDA contrast) → CX5 fork. Each rung graduates on a
committed figure or a banked negative that licenses the next ([SPEC §10.1](./SPEC.md), §12).

---

## 8. The lineage folded in (and the design choice each forces)

### 8.1 The causal hierarchy (the formalism)

- **Pearl's ladder of causation** (Pearl & Mackenzie, *The Book of Why*, 2018; Pearl, *Causality*, 2nd
  ed., 2009; Bareinboim et al., "On Pearl's Hierarchy and the Foundations of Causal Inference", 2020).
  Association / intervention / counterfactual; rung 3 requires an SCM and abduction-action-prediction. →
  **Design choice:** the three rungs map one-to-one onto observe / `oracle.step(s,a')` / abduct-from-
  `(seed,t)` (§2). The contribution is that verisim *owns* the SCM, so rung 3 is exact and free where
  every oracle-free world model is barred from it.

### 8.2 Causal representation & dynamics learning (the foil)

- **Toward Causal Representation Learning** (Schölkopf et al., 2021, arXiv:2102.11107) and **Causal
  Dynamics Learning** (Wang et al., ICML 2022, arXiv:2206.13452). The field's program is to *discover*
  causal variables and dependencies from observation — hard, approximate, and the bottleneck. → **Design
  choice:** verisim *skips structure learning entirely* — the SCM is given by the oracle (the factored
  state of [`negatives.py`](../../src/verisim/netdata/negatives.py) is the variable set; `step` is `F`).
  The method's reach is therefore the set of seedable oracles (H64), not the set of learnable structures.
- **Learning Neural Causal Models from Unknown Interventions** (Ke et al., 2019, arXiv:1910.01075). The
  hard case verisim avoids: here the intervention identity is *known* (verisim chooses `a'`) and the
  model is *given*, so there is no discrete structure search.

### 8.3 Counterfactual dynamics & data augmentation (the closest method neighbors)

- **CoPhy — Counterfactual Learning of Physical Dynamics** (Baradel et al., ICLR 2020,
  arXiv:1909.12000). Predicts the alternative future under an intervention on initial conditions, with
  the confounders learned *unsupervised* from visual input. → **Design choice / the foil:** CoPhy must
  *learn* the latent confounders because it has no oracle; verisim *abducts them exactly* from the seed
  (§2). CoPhy is rung 3 approximated; CX1's `+counterfactual` arm is rung 3 exact — the contrast is the
  whole point of the SCM identification.
- **CoDA — Counterfactual Data Augmentation using Locally Factored Dynamics** (Pitis, Creager, Garg,
  NeurIPS 2020, arXiv:2007.02863). Generates counterfactual experiences that are causally valid *in a
  learned local model*. → **Design choice:** CX4 contrasts CoDA's *learned*-model augmentation against
  verisim's *exact*-oracle augmentation — the same augmentation idea, with verisim's samples causally
  valid by construction rather than by a model that can drift ([SPEC §1.1](./SPEC.md)).
- **Invariant Risk Minimization** (Arjovsky et al., 2019, arXiv:1907.02893). OOD generalization via
  invariant causal mechanisms across environments. → **Reading:** the off-policy hidden state of H61 is
  precisely the "environment" axis IRM cares about; verisim can *construct* the environments exactly
  (intervene on the fault medium) rather than hope they appear in the data.

### 8.4 What the program already built that this sits on

- **The counterfactual factory ships** ([`counterfactual_branches`](../../src/verisim/netdata/negatives.py)):
  rung-2 `do(a')` branches. SPEC-17 adds the rung-3 abduction wrapper around it.
- **The ED6 design ships** ([`experiments/ed6.py`](../../src/verisim/experiments/ed6.py)) with the
  `_medium` coverage statistic — CX3 reuses it for the matched-coverage cut.
- **The three worlds and their oracles ship** — SPEC-17 builds almost no new primitive; it adds a
  formalism, an abduction wrapper, a coverage matcher, and the CX harnesses.

---

## 9. Build, reproduce, CI

### 9.1 Module layout (additive only)

```
src/verisim/causal/                   # NEW — the SCM/counterfactual layer (no new world, no new oracle)
  abduct.py            # rung-3: recover U from (seed, t), re-run with do(a') → exact counterfactual
  coverage.py          # the matched-coverage sampler over the ED6 _medium statistic (CX3)
  augment.py           # exact-oracle counterfactual data augmentation (CX4 oracle arm)
src/verisim/experiments/
  cx0.py … cx5_*.py    # NEW — the CX experiments
figures/
  plot_cx0.py … plot_cx4.py  # NEW — committed-figure generators
configs/
  cx0.json … cx4.json        # NEW — committed sweep configs
```

The layer consumes the shipped oracles, the [`negatives.py`](../../src/verisim/netdata/negatives.py)
factory, and the ED6/EN6/EH6 training harnesses **unchanged** — nothing in the deterministic core, the
oracles, the metrics, or any existing experiment is edited. SPEC-17 is a layer *beside* the data
factory, not a change *to* it (DD-AR2: never edit the oracle, metric, or gate).

### 9.2 `reproduce.sh` (new CX block, in dependency order)

```bash
echo "== CX0: the oracle is an exact SCM — abduction-exactness per world (H60) — gates rung 3 =="
python -m verisim.experiments.cx0 --config configs/cx0.json --out runs/cx0/records.jsonl --plot figures/cx0_abduction_exact.png
echo "== CX1: rung 2 (do) vs rung 3 (abduct) — does abduction add anything? (H61) =="
python -m verisim.experiments.cx1 --config configs/cx1.json --out runs/cx1/records.jsonl --plot figures/cx1_rung2_vs_rung3.png
echo "== CX2: the three-world hidden-state sweep (H61) — THE HEADLINE =="
python -m verisim.experiments.cx2 --config configs/cx2.json --out runs/cx2/records.jsonl --plot figures/cx2_hidden_state_lift.png
echo "== CX3: the matched-coverage cut (H62) — settles the ED6 branching-vs-coverage caveat =="
python -m verisim.experiments.cx3 --config configs/cx3.json --out runs/cx3/records.jsonl --plot figures/cx3_matched_coverage.png
# CX4 (CoDA contrast, torch-gated) and CX5 (system-oracle fork) follow; CX5 gated on CX3.
```

The non-CoDA CX block runs on CPU (the deterministic gate); CX4's learned-model augmenter is the one
optional torch dependency and is `skipif`-guarded with disclosure (never counted as a result when
skipped). CI (`ubuntu-latest`) stays the free Linux confirmation; CX tests assert structural invariants
(abduction recovers the factual state bit-for-bit; matched-coverage arms have equal `_medium`
statistics; the lift sign per world), not exact magnitudes, so the same tests pass on the macOS primary
host and Linux CI (the macOS-first principle).

---

## 10. Scope, non-goals, honest caveats

- **This is a formalism + a confound-resolver, not a new world or a new dynamics claim.** It runs on the
  three shipped worlds and their oracles. Over-claiming a new dynamics result from SPEC-17 is the failure
  mode this line forbids; the result is about *which rung the H5 lift lives on* and *whether branching
  survives matched coverage*.
- **The headline (CX2) is high-confidence but unproven.** The do-calculus and EN6/EH6/ED6 all predict
  hidden-state-dependence, but "it holds under the exact rung-3 target with disjoint CIs" is a
  measurement the program will not skip. A refutation (lift on network/host too, or null on distributed)
  is a deep, bankable result either way.
- **CX3 is the genuinely-open one.** Branching-vs-coverage has no safe prior; the program flagged it as
  future work tied to H21 ([SPEC-7 §10.1](./SPEC-7.md)). A matched-coverage null does *not* weaken the
  off-policy-hidden-state mechanism — it sharpens it (rung-2 `do` suffices; abduction is not the active
  ingredient). Both branches are banked and stated in advance.
- **Exact rung 3 requires seed-complete nondeterminism.** H60/CX0 is the check; a world whose
  nondeterminism escapes the seed (real-OS edges, [SPEC §2.1](./SPEC.md)) admits at best approximate
  rung 3, which narrows the method to the seedable subset (H64). The system oracles' DST schedulers are
  seeded, so the recipe is *designed* to survive CX5 — but that is a measurement, not an assumption.
- **The reference oracle is an SCM of a model of reality, not reality** ([SPEC §2.1](./SPEC.md)). CX5
  (system oracles) is the move to a reality SCM; until it lands, the rung-3 results are about the
  reference semantics, stated as such.
- **The CoDA contrast (CX4) needs an external model.** It is the one torch-gated experiment; everything
  else is dependency-free. The harness isolates it so the spec's spine runs without it.

---

## 11. Provenance & reading order

SPEC-17 is the formal layer the program's most-qualified result has been waiting for: [SPEC §3 (RQ4)](./SPEC.md)
posed counterfactual fidelity; [SPEC §9 (H5)](./SPEC.md) measured it and found a world- and
objective-dependent lift with one open caveat (branching vs coverage); SPEC-7's ED6 produced the
distributed positive and named the caveat. SPEC-17 supplies the do-calculus that explains the mix in one
line (off-policy hidden state), the SCM identification that makes rung-3 counterfactuals exact and free,
and the matched-coverage experiment that settles the caveat. The lineage is Pearl's ladder (the
formalism), CoPhy and CoDA (the closest counterfactual-dynamics neighbors, both *learning* what verisim
*abducts exactly*), and the causal-representation/dynamics line (whose structure-learning step verisim
*skips* because the oracle gives the SCM). It is a *method* spec: it advances how the program *reasons
about* its world model, not what the model is.

Reading order for a newcomer: [SPEC §2 (the oracle asymmetry)](./SPEC.md) →
[SPEC §3 (RQ4)](./SPEC.md) → [SPEC §9 (H5 — the full mixed result)](./SPEC.md) →
[SPEC-7 ED6 + §10.1 (the distributed positive and its caveat)](./SPEC-7.md) → this document
(§1 motivation, §2 the SCM identification, §3 architecture, §4 the load-bearing assumption, §5 the
hypotheses) → [`src/verisim/netdata/negatives.py`](../../src/verisim/netdata/negatives.py) and
`src/verisim/causal/abduct.py` (the concrete rung-3 build, once shipped).

---

## 8. Status (2026-06-09) — CX0 + CX1 + CX3 + CX4 + CX5 SHIPPED (only the CX2 learned lift remains)

The pure-oracle causal core is built and committed. The package is
[`causal/`](../../src/verisim/causal/): `scm.py` (the world-generic SCM machinery -- `abduct_and_replay`,
`abduction_exact`, `rung2_branch`, `rung3_counterfactual`, `downstream_amplification`). The world
bundles, including the **distributed** world (the off-policy world where H5/H61 predicts the largest
counterfactual structure), are in
[`experiments/cx_common.py`](../../src/verisim/experiments/cx_common.py). Both experiments run on
network/host/filesystem/distributed, CPU-only, deterministic and seeded -- no learner, no GPU.

- ✅ **CX0 — the oracle is an exact SCM (H60 SUPPORTED — the gate).** Abduction-action-prediction is
  **bit-exact on every world** (rate 1.0): recovering `U` from the seed and replaying `F` reproduces
  the factual trajectory bit-for-bit, so rung-3 counterfactuals are *exact and free* -- abduction is an
  `O(1)` reset+replay, not the intractable inference an oracle-free SCM faces. The rung-3 trajectory
  genuinely differs from the factual (cf-differs 0.81–1.00), so the recovered `U` is producing a real
  counterfactual. This is the identification the whole spec rests on, made empirical -- a *build*, not a
  bet. [`cx0`](../../src/verisim/experiments/cx0.py), [`figures/cx0_scm_gate.png`](../../figures/cx0_scm_gate.png).
- ✅ **CX1 — the counterfactual effect is hidden-state-dependent (H61 effect-size SUPPORTED — the
  do-calculus reading of H5).** Sweeping interventions across depths/seeds and measuring the rung-2
  immediate vs rung-3 downstream effect: the **distributed** world's counterfactual effect amplifies
  **~3.6× downstream** (its persistent partition/crash medium carries the intervention forward, 65% of
  interventions consequential), while the **on-policy-complete network/host** worlds amplify ~1× (the
  effect washes out, 0–41% consequential). The clean ordering distributed ≫ host > filesystem ≫ network
  is the do-calculus reading of the mixed H5: counterfactual *structure* is large exactly where the
  world has off-policy exogenous hidden state. [`cx1`](../../src/verisim/experiments/cx1.py),
  [`figures/cx1_counterfactual_effect.png`](../../figures/cx1_counterfactual_effect.png).
- ✅ **CX5 — the system-oracle fork (H64 TRANSFER — the SCM survives reality).** The objection to CX0
  is that bit-exact abduction is a property of the reference *abstraction*, not reality. CX5 re-runs the
  abduction gate on the **system oracles** — the real `/bin/sh` + coreutils `SandboxOracle` (filesystem)
  and the Tier-B `SystemDistOracle` (distributed, autonomous actors under a seeded scheduler) — with
  the action sequence replayed *through the system oracle*. Result: abduction-exactness, rung-3
  counterfactual-exactness, and cf-differs are **all 1.0 on both system oracles**, matching the
  reference anchor — so exact, free rung-3 counterfactuals survive the move to the real system. The
  honest, load-bearing reading: this holds **because** the filesystem oracle is sealed against
  clock/RNG/concurrency (the SY4 `DeterminismSeal`) and the distributed oracle drives its real
  concurrency with a **seeded** scheduler (the DST thesis) — a real system *without* the seal/seed is
  *not* an SCM, so CX5 measures that the seal/seed is exactly what buys an exact rung 3 on reality.
  Pure-oracle; a genuinely unavailable system oracle is a disclosed skip, never a pass.
  [`cx5`](../../src/verisim/experiments/cx5.py), [`configs/cx5.json`](../../configs/cx5.json),
  [`figures/cx5_system_oracle.png`](../../figures/cx5_system_oracle.png).
- ◐ **CX3 — the matched-coverage cut (H62 REFUTED — branching was coverage; the program's open caveat,
  closed).** A genuine CPU-scale trained-arm experiment ([`cx3`](../../src/verisim/experiments/cx3.py),
  the new [`causal/coverage.py`](../../src/verisim/causal/coverage.py) sampler): the ED6 re-run with the
  `+counterfactual` arm and a *factual* control matched on **both** example count *and* fault-coverage
  (the fraction of training examples whose action changes the `_medium`). The factual-matched control is
  a fault-heavy on-policy trajectory (high coverage, no branching); the +counterfactual-matched arm is
  branches off the light-fault on-policy states (the same states, many alternative fault futures — the
  abduction/re-grounding structure), subsampled to the identical coverage — so the two differ in
  **branching alone**. **Result: at matched count and coverage (0.78) the factual control STRICTLY
  beats the counterfactual arm** on held-out intervention-exact (0.569 vs 0.426) and medium-recall
  (0.639 vs 0.480), **disjoint CIs both ways**. So ED6's ~2× lift was **fault coverage, not
  counterfactual structure** — re-attributed to H21 (fault coverage at equal volume); branching per se
  not only fails to help, a fault-heavy factual sequence does *better* (it visits deeper drifted states
  in-sequence). The SPEC-7 §10.1 caveat resolves **decisively against branching** — the pre-registered
  "no safe prior" question, answered. (The raw arms reproduce the original lift: trajectory 0.245 →
  +counterfactual 0.425, at coverage 0.10 → 0.59 — the lift tracks coverage.)
  [`figures/cx3_matched_coverage.png`](../../figures/cx3_matched_coverage.png).
- ◐ **CX4 — exact-oracle vs learned-model counterfactual augmentation (H63 SUPPORTED — the CoDA contrast;
  the unverifiability thesis, made a measured cost).** A CPU-scale trained-arm experiment
  ([`cx4`](../../src/verisim/experiments/cx4.py)) on the distributed world: build one counterfactual
  query set (alternative fault actions at visited on-policy states) and label it two ways — the **exact
  oracle** ``O(s, a')`` (causally valid by construction) and a **learned local model** ``M_local`` (a
  small `M_θ` trained on the on-policy trajectory, the CoDA stand-in) predicting ``M_local(s, a')`` (the
  same prompts, labels that inherit its drift). Both augment the same base to the same sample count; a
  `base` arm is the no-augmentation reference. **Result: H63 supported, decisively.** +oracle-aug lifts
  held-out intervention-exact to **0.394** (over base 0.277) while +learned-aug **collapses to 0.064 —
  *below* the no-augmentation base** — disjoint CIs ([0.307,0.485] vs [0.049,0.084]). The mechanism is
  causal validity: the learned model's counterfactual samples are only **0.058** valid (the oracle's are
  1.00 by construction), so its augmentation injects ~94% causally-invalid data that *corrupts* training.
  This is [SPEC §1.1](./SPEC.md)'s unverifiability claim made a measured cost — a learned model's
  counterfactual augmentation is not merely useless but actively harmful, exactly where the exact oracle
  is the unique leverage over the CoDA line. [`figures/cx4_coda_contrast.png`](../../figures/cx4_coda_contrast.png).
- ✅ **H64 (the cross-world method) — supported in kind *and* on the system oracle.** CX0/CX1 run on all
  four worlds with the *identical* SCM machinery and no per-world causal-discovery step (the SCM is
  *given* by the oracle, not learned); CX5 then shows the contract holds on the real system oracles too.

**Deferred to the trained/contrastive arm (the LP7 rule, §7), disclosed and never counted on a
stand-in:** only **CX2** (the three-world *learned* lift -- does training `M_θ` on rung-3 targets improve
held-out counterfactual prediction across network/host/distributed). Its headline is now nuanced by CX3
(the distributed counterfactual lift is fault coverage, not branching), so CX2 would measure where the
coverage-driven counterfactual-training lift appears across worlds — the one remaining learned-arm bet.
Everything else (CX0/CX1 identification + effect-size law, CX3 confound-resolver, CX4 CoDA contrast, CX5
system-oracle transfer) ships.
