# Where the LLM guardrail breaks: the directed effect-hunter vs a real model judge (SPEC-25, 2026-06-21)

> RA11 measured a real `claude -p` SafePred-style judge — relational danger, recall 0.95–1.00 — and
> named the gap it left: *"we did not find where the LLM guardrail breaks."* RA11 was a fixed battery;
> it never *searched*. SPEC-25 supplies the search. An LLM guardrail reads the **action**; the exact
> oracle reads the **effect**; so the SPEC-23/24 auditor, with the model judge in the `Monitor` slot,
> hunts the realizing actions the judge silently passes — the effect-only harms an action-reader is
> structurally blind to.

## What SPEC-25 adds

[`src/verisim/audit/llm_guardrail.py`](../src/verisim/audit/llm_guardrail.py):

- **`LLMGuardrailMonitor`** — wraps a judge callable behind the SPEC-23 `Monitor` protocol
  (`covers` = "the judge would block this"); runs through the unchanged `audit`/`certify` loop.
- **`StubJudge`** — a deterministic *action-reader* (blocks iff the protected path appears literally;
  passes encoded indirection and resulting-state harm). Keeps CI hermetic; a faithful stand-in for the
  one property under test, **not** a model of a good judge.
- **`ClaudeCliJudge`** — the real lane: `claude -p <prompt>` parsed to SAFE/UNSAFE (the RA11
  invocation), optional K-sample majority vote. Run-on-demand.
- **`certify_llm`** — the statistical certificate over a stochastic monitor: per-class recall + Wilson
  CI + a residual bound (SPEC-24's tier, which is the correct footing for a non-deterministic judge).

No change to the shipped auditor, proposers, oracles, or policy compiler — SPEC-25 is a monitor adapter
plus K-trial aggregation.

## The result (hermetic stub; the real model is a run-on-demand lane)

`python -m verisim.experiments.spec25_llm_guardrail` — the deterministic action-reader, audited over
the file-corruption family:

| class | recall (stub) | misses |
|---|---|---|
| literal | **1.00** [0.82, 1.00] | 0 |
| rev / cut / printf_fmt / var_split / ansi_hex / … (11 encodings) | **0.00** [0.00, 0.18] | all |

Overall recall **0.083** — the action-reader blocks the one literal form and silently passes every
encoding (66 oracle-confirmed holes, `found_break=True`). On the **triad**, a per-action judge passes
the relational exposing edge (`analytics→reporting`, which transitively opens `→db`) because the harm is
in the *resulting graph*, not the action it reads. This is the action-vs-effect gap, made into a
searched certificate rather than a single pair.

The certificate is **statistical by construction**: a stochastic judge blocking at a planted rate p has
its Wilson interval bracket p over K trials (pinned by a hermetic test), so a *clean* real-lane result
(no hole found) is a graded robustness bound — `recall ≥ 1 − ε at conf 1 − δ` — corroborating RA11 at
the auditor's scale, and a *non-clean* one is the witness RA11 never found.

## The honest takeaway

The hermetic stub proves the loop and the harness and the action-vs-effect gap; **the real finding lands
only on the `--claude` lane** (RA11/RA20 discipline). The certificate is a confidence bound, not a
theorem — doubly so, because the monitor is stochastic, and the exhaustive depth-k tier (SPEC-24 H165)
does **not** apply to an LLM (it cannot be enumerated cheaply); SPEC-25 carries only the statistical
tier and says so. Every hole is pinned to **model + version + prompt + framing** — a judge hole is a
property of *this judge as invoked*, not of "LLMs." Single provider unless a second is added (the
program's standing cross-provider gap). And the stub must not be read as evidence about real-model
recall — that number comes only from the live lane.

## Reproduce

```
python -m verisim.experiments.spec25_llm_guardrail                 # hermetic stub
python -m verisim.experiments.spec25_llm_guardrail --claude        # real claude -p lane (run on demand)
python -m pytest tests/test_spec25_llm_guardrail.py -q             # hermetic; claude lane skipif-guarded
```
