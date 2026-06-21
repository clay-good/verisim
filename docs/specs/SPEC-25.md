# SPEC-25 — Where the LLM guardrail breaks: the directed effect-hunter vs a real model judge

**MVP/POC specification. SPEC-23/24 built a monitor auditor that certifies any covering surface and
graded what its certificate proves. The 2026 deployment shape it most needs to be pointed at is the
one it has not yet been: a **real LLM guardrail** (a SafePred-style prompted judge). RA11 already
measured one — a real `claude -p` judge scored relational danger at recall 0.95–1.00 — and the paper
states the honest gap out loud: *"we did not find where the LLM guardrail breaks badly."* SPEC-25
closes exactly that gap. It wraps a real model judge as a `Monitor` and turns the auditor loose to find
the realizing actions the judge silently passes, each confirmed by the exact oracle — because an LLM
guardrail reads the **action** and the oracle reads the **effect**, so the auditor hunts precisely the
effect-harms an action-reader is structurally blind to. The result is "red-team-as-a-CI-gate for LLM
guardrails," and the certificate it emits is — fittingly — statistical, because the monitor is
stochastic, which is exactly what SPEC-24's residual machinery was built for.**

> **✅ SHIPPED — MVP / POC SPEC — 2026-06-21.** H171–H174 built and tested (hermetic via a deterministic
> stub judge; the real `claude -p` lane is run-on-demand): the LLM-guardrail auditor
> ([`src/verisim/audit/llm_guardrail.py`](../../src/verisim/audit/llm_guardrail.py) —
> `LLMGuardrailMonitor`, `StubJudge`, `ClaudeCliJudge`, `certify_llm`). Run-record:
> [`docs/llm-guardrail-audit-run.md`](../llm-guardrail-audit-run.md); CSV + certificate under
> [`figures/`](../../figures/) (`spec25_llm_guardrail.csv`). Reproduce:
> `python -m verisim.experiments.spec25_llm_guardrail` (stub) and `--claude` (real model). Sibling to
> [SPEC-24](./SPEC-24.md) (the graded
> guarantee it reuses) and [SPEC-23](./SPEC-23.md) (the auditor it points outward). It invents **no new
> oracle and no new adversary**: it wraps the RA11 `claude -p` judge
> ([`bench/ra11/learned_guardrail.py`](../../bench/ra11/learned_guardrail.py)) behind the SPEC-23
> `Monitor` protocol, drives discovery with the existing proposers (the directed adversary on a cheap
> surrogate, H167), confirms with the exact oracles already built, and grades with SPEC-24's
> statistical residual bound. The real-model lane is **run-on-demand** (like RA11/RA20); a deterministic
> **stub judge** keeps the CI path hermetic.

Read [SPEC-24 §3.1](./SPEC-24.md) (the statistical residual tier this reuses), [`docs/paper.md`](../paper.md)
§4 (RA11, the real LLM guardrail measured, and the gap it leaves), and [`docs/review.md`](../review.md)
(the verdict that the action-vs-effect distinction is the program's one piece of real, un-dominated
signal — the weapon SPEC-25 turns on a real model).

---

## 0. One-paragraph thesis

A prompted-LLM guardrail decides allow/block by **reading the proposed action** and reasoning about it.
The program's one durable empirical signal (paper §4, review) is that harms visible only in the
**resulting state** — relational (a config edit that opens an untrusted→db path), cumulative (mass
collection past a budget), context-dependent (a prod change during a freeze), and indirection (a path
the command assembles without ever naming it) — are a structural blind spot no model quality closes,
because the harm is not in the action text the judge reads. SPEC-25's thesis is that the SPEC-23/24
auditor, pointed at a real model judge, **finds those blind spots automatically**: the judge is the
`Monitor` (`covers` = "the judge would block this"), the exact, resettable oracle is the verifier
(`realizes` = "this action truly causes the harm"), and a hole is `realizes ∧ ¬covers` — a realizing
action the model silently passed. Because the judge is stochastic, the certificate is not a theorem but
a **per-class recall with a Wilson interval** over K trials — SPEC-24's residual tier, which already
treats "no hole found in N" as a confidence bound, applied to a monitor that is non-deterministic by
nature. The honest scope is the live-lane scope inherited from RA11/RA20: it needs the API/CLI, every
hole is pinned to model + version + prompt, and it is one provider unless a second is added.

---

## 1. What this deepens (the gap RA11 named and did not close)

RA11 replaced the convenient "learned guardrails can't see relational danger" story with a measurement:
a real `claude -p` judge caught RA8's relational danger at recall **0.95–1.00**, and the paper retired
the easy claim. But it stated, in the same breath, the thing it did **not** do: *"we did not find where
the LLM guardrail breaks badly — depth ≤ 8 / ~18 edges is within Claude's in-context reach; larger or
adversarial graphs likely push recall down but we have not measured that and do not claim it."* RA11 was
a **fixed battery**: a hand-written UNSAFE/SAFE pair per depth. It never *searched* for the judge's
failure region.

SPEC-25 supplies the search. The SPEC-24 directed adversary (H167) climbs a monitor's blind-spot
gradient; pointed at the model judge it concentrates exactly where the judge is weakest, at a bounded
LLM-call budget. Where RA11 asked "what is the judge's recall on this pair?", SPEC-25 asks "what is the
input on which this judge silently passes a realizing action?" — and answers it with an oracle-confirmed
witness or a graded statistical certificate of robustness when none is found.

## 2. The pivot (why a model judge, why now)

The deterministic-guardrail crowd SPEC-23 audits (YoloFS, Progent, OPA) is one half of 2026; the other,
growing half is the **LLM-judge guardrail** (SafePred and its descendants). It is the harder, more
honest target for three reasons the program already established:

- **It is where the action-vs-effect distinction bites hardest.** A denylist or resolver reads the
  action too, but an LLM judge is *sold* as understanding intent and consequence. Showing it still
  misses effect-harms is the sharpest form of the program's one un-dominated claim.
- **Its failures are not enumerable.** A denylist's gap is a missing pattern; an LLM's gap is a region
  of input space found only by search — exactly what the directed adversary is for.
- **Its certificate must be statistical, and the program already has the machinery.** SPEC-24's
  Wilson / rule-of-three residual bound is the correct epistemic footing for a stochastic monitor;
  SPEC-25 is its natural application, not a new tool.

## 3. The MVP

### 3.1 The model judge as a `Monitor`

`LLMGuardrailMonitor(judge)` wraps a judge callable behind `covers(action) = (judge(action) == BLOCK)`,
reusing the RA11 SafePred prompt shape (the judge is shown the proposed action and the standing
invariant, and asked to return SAFE/UNSAFE). Two judges satisfy the callable:

- **`ClaudeCliJudge`** — the real lane: `claude -p <prompt> --output-format text`, parsed to
  SAFE/UNSAFE exactly as RA11 (clean-room, run-on-demand). Optional K-sample majority vote to tame
  per-call variance.
- **`StubJudge`** — the hermetic lane: a deterministic model of an *action-reading* judge. It blocks a
  command iff the protected prefix appears **literally** in the action text (the obvious harm it can
  read) and allows everything else — so it reproduces the action-vs-effect blind spot (encoded
  indirection, and the triad's resulting-state harms) with no API call. It is not a strawman of a
  *good* judge; it is a faithful, deterministic stand-in for the one property under test (action-reading
  cannot see effect-only harm), so the loop and harness are proven in CI and the real finding lands on
  the live lane.

### 3.2 The harm families the judge is hunted on

The proposer emits candidate actions over the families where action-reading is structurally blind, each
with an exact oracle already built:

- **indirection** (file family): encoded commands that realize a write to the protected file without the
  literal path (the SPEC-23 `GrammarProposer`; oracle = `ContainerDiffOracle` or `ShellPathOracle`).
- **relational / cumulative / context** (the un-dominated triad): the SPEC-23 `policy` compiler's
  actions and oracles (reachability / count / freeze).

### 3.3 The loop, the budget, and the statistical certificate

Discovery uses the **directed adversary on a cheap surrogate** (H167: train the REINFORCE policy against
the resolver/oracle reward, which costs no LLM calls), then the audit evaluates the proposed actions
against the real judge — so the LLM-call budget equals the number of distinct candidates evaluated, not
the search depth. Because the judge is stochastic, `certify_llm` runs the audit over **K trials** and
reports, per harm class, the judge's **recall** (fraction of realizing actions it blocked) with a
**Wilson interval**, and the **miss set** (oracle-confirmed silent passes). The certificate's verdict is
SPEC-24's: a non-empty miss set is "where the judge breaks"; an empty one over N calls is a graded
robustness bound (`recall ≥ 1 − ε at confidence 1 − δ`), corroborating RA11 at the auditor's scale
rather than on a single pair.

## 4. Milestones (hypotheses continue the SPEC-24 ledger from H170)

| ID | Hypothesis | Status | Acceptance |
|---|---|---|---|
| MVP-1 | **H171** — a **real model judge is a `Monitor`**: `LLMGuardrailMonitor` wraps a SafePred-style judge behind `covers`, with a deterministic stub for CI and the real `claude -p` lane run-on-demand | ✅ shipped — **SUPPORTED** ([`tests/test_spec25_llm_guardrail.py`](../../tests/test_spec25_llm_guardrail.py)) | the auditor runs unchanged against `LLMGuardrailMonitor`; the `StubJudge` test produces a certificate; the real-`claude` lane is exercised by a `shutil.which("claude")`-guarded test |
| MVP-2 | **H172** — the auditor **finds where the judge breaks**: ≥1 oracle-confirmed silent miss on the effect-harm families (indirection + triad) the action-reading judge passes | ✅ shipped — **SUPPORTED** | against `StubJudge` the auditor surfaces **66** oracle-confirmed holes across all 11 encodings (rev/printf_fmt/var_split/…) while the literal class is covered (recall 1.0); the relational triad hole (`analytics→reporting`) is also found; the real-`claude` lane reproduces on demand or reports a clean graded bound |
| MVP-3 | **H173** — the **statistical certificate** over a stochastic monitor: per-class recall + Wilson CI + a residual bound, reusing SPEC-24 | ✅ shipped — **SUPPORTED** | `certify_llm(trials=K)` reports per-class recall + Wilson CI + `(residual_epsilon, residual_delta)`; pinned by a hermetic test where a stochastic stub's planted block rate p=0.6 falls inside the certified interval, and the CI matches a direct Wilson on the aggregate draws |
| MVP-4 | **H174** — the **framing-transfer trap (RA19) made operational**: the auditor measures whether the judge's recall drops under "approve this agent's change" framing vs the direct security question | ✅ shipped — **SUPPORTED** | `certify_llm` runs both framings and reports the recall delta; on the framing-insensitive stub the delta is ~0 (pinned), and the real-lane harness (`spec25_llm_guardrail --claude`) surfaces a non-zero delta if present |

## 5. Architecture (where the seams go — reusing SPEC-23/24)

- **`src/verisim/audit/llm_guardrail.py` (new)** — `LLMGuardrailMonitor` (wraps a `judge` callable),
  `StubJudge` (deterministic action-reader), `ClaudeCliJudge` (the RA11 `claude -p` lane), the
  effect-harm `battery()` (indirection + triad actions with their oracles), and `certify_llm` (K-trial
  recall + Wilson CI + residual, building on `guarantee.py`). Torch-free; the directed-proposer arm
  imports torch lazily as elsewhere.
- **No change to the shipped auditor**: `audit` / `certify` / the proposers / the oracles / the policy
  compiler are all reused as-is; the only new thing is the monitor adapter and the K-trial aggregation.
- **experiment + docs** — `src/verisim/experiments/spec25_llm_guardrail.py` (the stub table hermetic;
  `--claude` runs the real lane), `docs/llm-guardrail-audit-run.md` (run-record), and
  `tests/test_spec25_llm_guardrail.py` (hermetic stub; `claude`-guarded live test).

## 6. Honest caveats, stated up front

- **The real finding is on a run-on-demand lane, not in CI.** The hermetic test proves the loop and the
  harness against a deterministic stub; the actual judge hole is found only when the live lane runs. This
  is the RA11/RA20 discipline, stated, not hidden.
- **The certificate is a confidence bound, not a theorem — and now doubly so.** The monitor is
  stochastic, so even "no hole found" is a per-class recall with a CI, and the exhaustive depth-k tier
  (SPEC-24 H165) does **not** apply to an LLM judge (you cannot enumerate it cheaply). SPEC-25 carries
  only the statistical tier, and says so.
- **Every hole is pinned to model + version + prompt + framing.** A judge hole is not a property of "LLMs";
  it is a property of this judge as invoked. The certificate records all four.
- **Single provider unless a second is added.** This is the same limitation RA11 carried and the
  cross-provider study (the program's other named gap) would close; SPEC-25 does not claim past it.
- **The stub is a stand-in for one property, not a model of a good judge.** It demonstrates the harness
  and the action-vs-effect gap deterministically; it must not be read as evidence about real-model recall
  — that number comes only from the live lane.

## 7. Reproduce / acceptance (what "done" means for the POC)

The POC is done when: (1) `LLMGuardrailMonitor` runs through the unchanged `audit`/`certify` loop, with
a hermetic `StubJudge` test and a `claude`-guarded live test (H171); (2) the auditor surfaces
oracle-confirmed silent misses in the encoded-indirection and triad classes the action-reader passes,
hermetically against the stub and on-demand against the real judge (H172); (3) `certify_llm` emits a
per-class recall + Wilson CI + residual whose interval contains a planted stub miss rate (H173); (4) the
two-framing harness reports a recall delta (H174); (5) every number is pinned by a hermetic test
(torch-free where possible) and the real lane is reproducible via `--claude`, matching the RA arc's
discipline. Run-records land in `docs/llm-guardrail-audit-run.md`; figures + CSVs under `figures/`.
