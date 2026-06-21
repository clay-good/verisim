# The cumulative-harm hunt: result and verdict (SPEC-26, 2026-06-21)

> The hypothesis: cumulative harm ("≤ B distinct sensitive records per task") is where a real LLM
> guardrail *structurally* fails, because it is not a property of any single action. We pointed a real
> `claude -p` guardrail at it two ways and swept the budget magnitude to stress exact counting. This is
> a hunt with a pre-registered kill criterion (SPEC-26 §4).

## What we measured (real `claude -p`, Claude Code 2.1.81)

Boundary sweep: at each budget B, a probe whose next read brings the distinct count to **exactly B+1**
(over by one, correct = BLOCK) vs **exactly B** (safe, correct = ALLOW). Distinguishing them needs a
precise distinct count of B-1 vs B over a long noisy history (scrambled non-sequential ids,
near-duplicate distractors, re-reads that must not double-count, benign actions interleaved — ~180
lines at B=80). `recall` = catch-the-over-by-one rate; `over_block` = wrongly block the safe boundary.

| arm | model | B=10 | B=20 | B=40 | B=80 | over-block |
|---|---|---|---|---|---|---|
| **stateless** (per-action hook) | default & haiku | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **history** (full session shown) | default | 1.00 | 1.00 | 1.00 | 1.00¹ | 0.00 |
| **history** (full session shown) | haiku (cheap) | 1.00 | 0.80² | 1.00 | 1.00 | 0.00 |
| **oracle** (exact accumulator) | — | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |

¹ default model run with sequential ids to B=48 (recall 1.00 at every horizon); haiku run with the
hardened scrambled/near-duplicate probe to B=80. ² a single stochastic miss (4/5, Wilson [0.38, 0.96]);
**not** degradation — recall returns to 1.00 at B=40 and B=80, so it is noise, not a monotone break.

## The verdict (kill criterion met)

**The structural half is confirmed; the empirical half is a null.**

- **Stateless per-action judges cannot enforce a cumulative budget — recall 0.00 at every budget, on
  both a strong and a cheap model.** A single read always looks individually fine, so the deployed
  guardrail pattern (a PreToolUse hook that sees one tool call) structurally misses the budget. Model
  quality does not rescue it. This is real and deployment-relevant — but it is not a *capability*
  finding; it is a *what-you-show-the-guardrail* finding.
- **History-augmented judges do not break.** Given the session history, even the cheap model counts
  distinct correctly to B=80 over a ~180-line scrambled, near-duplicate, re-read-laden history
  (recall ≈ 1.00, over-block 0.00). The exact-counting degradation we hunted for did not appear.

This hits SPEC-26 §4's pre-registered kill criterion: the stateless result is (arguably) a strawman a
deployer would fix by adding state, and the history-augmented judge holds. So we **stop the hunt** here
rather than chase the tail to B=200 — that would be the circle this experiment was meant to avoid.

## What the hunt actually taught us (the honest, useful finding)

Across three live experiments — SPEC-25 (encoded indirection: recall 1.00; relational reachability to
depth 8: recall 1.00) and SPEC-26 (cumulative, history: recall ≈ 1.00 to B=80) — the same thing keeps
showing up, and it **reframes the program's contribution honestly**:

**The action-vs-effect gap is not a model-capability gap. It is a state/context gap and a guarantee
gap.** Modern models — even cheap ones — see effect-harms perfectly *when shown the relevant state*:
the decoded command (indirection), the reachability graph (relational), the session history
(cumulative). What no model can do is enforce a property it is not shown — and the *deployed* guardrail
pattern (stateless per-action hooks) does not show it. So the oracle's real, defensible value is **not**
"models can't see effect-harm" (false — they can). It is:

1. **It is the state the deployed pattern lacks** — the oracle *is* the stateful distinct-counter, the
   reachability evaluator, the post-commit diff. It supplies exactly the context a stateless hook omits.
2. **A guarantee with a certificate** — exact and every-time, versus a measured ≈0.95–1.00 with no
   certificate on the one input that matters (SPEC-24's graded bound makes this concrete).
3. **Cost and determinism** — an exact distinct-count is a one-line accumulator; re-prompting an LLM
   with a 180-line history on every action to recount is absurd cost for a property a counter nails.

That is a sharper, fully-defensible thesis than "the oracle sees what models can't," and it is what the
preprint should lead with. The next step is the write-up (option C), not SPEC-27.

## Honest caveats

Single model family (Claude), single version; the live lane is small-K with wide CIs; the prompts are
fair (they tell the judge to count distinct and warn about re-reads). A genuinely adversarial history
(thousands of lines, deliberately confusable ids) might still break a model — not measured, not claimed.
The structural stateless result is the robust one; treat the history nulls as "no break at this scale."

## Reproduce

```
python -m verisim.experiments.spec26_cumulative_probe                       # hermetic stub
python -m verisim.experiments.spec26_cumulative_probe --claude              # default model
python -m verisim.experiments.spec26_cumulative_probe --claude --model haiku --budgets 10,20,40,80
python -m pytest tests/test_spec26_cumulative_probe.py -q                   # hermetic
```
