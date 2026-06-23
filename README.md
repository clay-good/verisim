# Verisim

Verisim is two things, and the order matters:

1. **[`verisim-certify`](src/verisim/certify/README.md)** — a tool that *proves* an AI-agent guardrail's
   completeness instead of asserting it. Point it at the guardrail you shipped; it returns the exact
   commands that realize harm off your checked surface, each proven against a real shell, plus a
   coverage certificate with a confidence interval. **This is the active project.**
2. **A concluded research program** on oracle-grounded world models and complete-mediation safety for
   computer-use agents. Its honest result, and the battle-tested engine underneath the tool above, is
   the *certify-don't-assert* loop. The full record is the **[research log](RESEARCH-LOG.md)**.

> *Verisim* (n., from *verisimilitude*, "the quality of resembling reality"): the one lucky fact this
> work is built on is that a computer's state is **exact and checkable for free** — unlike vision,
> language, or robotics, you can run the real thing and read the true next state, bit for bit. Call
> that an **oracle**. Everything here follows from having one.

---

## `verisim-certify` — prove your guardrail's completeness, don't assert it

The 2026 agent-security market is split into red-teamers that *find* prompt attacks and guardrails that
*assert* completeness. The empty middle is the measurement that connects them: *is the tool-call rail
you deployed actually complete, and here is the adversarial proof.* That is this tool.

```bash
# audit a deployed Claude Code PreToolUse hook; write a human-readable certificate
python -m verisim.certify audit --hook ./my_hook.py --report cert.md

# audit a denylist directly (no hook script); nonzero exit on a leak -> drops into CI
python -m verisim.certify audit --denylist "/etc/shadow,rm -rf /etc"

# prove each bypass against a real /bin/sh in a sandbox (the strongest evidence)
python -m verisim.certify audit --hook ./my_hook.py --oracle real
```

**The moat is the free oracle.** Every red-teamer says "an LLM judge thinks this is risky." This one
*executes* the candidate in a sandbox and reads the true effect against a real `/bin/sh`, so a reported
bypass is proven, and the coverage number is grounded in ground truth, not a model's opinion.

**What it found on its first two real targets** ([full writeup](docs/certifier-findings.md)): on a
hand-rolled denylist hook, **66 of 72** harm-realizing commands bypass it (8.3% coverage) — every
indirection encoding sails past a substring rule. On the repo's own *hardened* resolver hook, **18 of
72** (75% coverage, Wilson 95% CI [16.4%, 36.1%]) — and the subtle one: its reversibility router is
itself fooled by obfuscation, so an obfuscated `rm` of `/etc/shadow` is allowed at the hook layer.

Tool docs and the bring-your-own-guardrail interface: **[src/verisim/certify/README.md](src/verisim/certify/README.md)**.
Vision and scope: **[SPEC-28](docs/specs/SPEC-28.md)**.

![The verisim agent-in-the-loop architecture: an LLM agent turns intent into a host action plan; a cheap learned world model previews it; the free exact oracle verifies the preview at a small consultation rate and re-anchors to ground truth; a safety gate checks the predicted final state and allows or aborts. The model is cheap but drifts; the oracle is exact but costs a call; the loop spends a small budget to keep the preview faithful so the gate can be trusted.](figures/cu_architecture.png)

---

## The research, distilled

The whole program tried to prove one tempting story — "an LLM guardrail reads the *action*, the oracle
reads the *effect*, so the oracle catches harms the model can't" — and the data killed it. The honest
result is smaller and truer. The 5-minute version is the essay
**[Your agent guardrail is stateless, and that is the whole problem](docs/essay-state-not-capability.md)**.

**What is true, and useful:**

- **It is a state/deployment gap, not a capability gap.** A modern model *shown the relevant state*
  decodes every shell-indirection trick (recall 1.00), traces relational reachability to depth 8, and
  counts a cumulative budget correctly. But the deployed pattern — a *stateless per-action* hook —
  never shows it the state, so it cannot enforce any harm that spans actions (cumulative, relational,
  contextual): recall **0.00**, no matter how smart the model. The fix is plumbing or a cheap exact
  check, not a bigger model.
- **Certify, don't assert.** Because the oracle is free, you can run an adversary that *proves* a
  checker honest instead of trusting it. Pointed at our own resolver it found a real soundness bug a
  human red team missed. This is the durable contribution — and the product.
- **The danger surface is sparse.** On **123,195** real Claude Code commands, only **~1.1%** can touch
  a protected resource, so a covering gate waves 99% through and checks the rest.

![Sparse surface, measured on real traffic (RA21): replaying 123,195 real Claude Code tool calls through the coverage gate, only ~1.1% of actions touch the danger surface, so 98.9% auto-approve with no prompt while a fair denylist over-prompts 8.7x as often and still leaks an injected harm arsenal the gate catches entirely.](figures/ra21_cc_corpus.png)

![The oracle safety gate drives missed-danger to zero, cheaply (CU1): an unverified agent executes the irreversible bad action; with the free oracle in the loop the missed-danger rate is 0, and a small consultation budget buys it back to zero at a cheap knee.](figures/cu1_safety_gate.png)

**Honest scope.** This is for **integrity / state-change** harms. **Confidentiality / covert-channel
exfiltration is out** — there is no sparse surface to cover (an honest negative). And the program's
honesty is enforced, not claimed: an [adversarial self-review](docs/review.md) by three hostile
reviewers forced the reframing, and the most-quoted positive number (a "learned adversary finds bugs
18× faster") was [retracted by a pre-registered evaluation](docs/learned-proposer-eval-run.md) when a
competent baseline matched it — a *Revisiting-Neural-Program-Smoothing* result the program ran on
itself.

---

## Read the research

| document | what it is |
|---|---|
| [docs/essay-state-not-capability.md](docs/essay-state-not-capability.md) | **the 5-minute distilled thesis** (start here) |
| [docs/paper.md](docs/paper.md) | the ~8-page technical preprint |
| [docs/review.md](docs/review.md) | the hostile self-review that forced the honesty |
| [docs/certifier-findings.md](docs/certifier-findings.md) | what `verisim-certify` caught on real hooks |
| [docs/learned-proposer-eval-run.md](docs/learned-proposer-eval-run.md) | the pre-registered null that retracted the 18× claim (SPEC-27) |
| [docs/related-work.md](docs/related-work.md) | the maintained bibliography and positioning |
| [RESEARCH-LOG.md](RESEARCH-LOG.md) | **the complete, preserved record** — the full CU1–35 / RA1–26 ledger, every figure, every run |
| [docs/specs/](docs/specs/) | the specifications (SPEC-1 … SPEC-28); [SPEC.md](docs/specs/SPEC.md) is the master science |
| [docs/](docs/) | the per-experiment run docs (the live Docker/Claude arc, Terminal-Bench, the LLM-guardrail audit, …) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the system/code architecture |

The 218 result figures are in [figures/](figures/); reproductions are in [RESEARCH-LOG.md](RESEARCH-LOG.md#quickstart).

---

## Repository layout

| path | contents |
|---|---|
| `src/verisim/certify/` | **the Coverage Certifier** (the product) |
| `src/verisim/audit/` | the certify-don't-assert engine: `audit()`, monitors, oracles, proposers, the certificate |
| `src/verisim/` | the research codebase: the world models, oracles, loops, and metrics for the four worlds (host, network, distributed, filesystem) |
| `scripts/` | runnable hooks and examples (incl. the Claude Code coverage hook the certifier audits) |
| `docs/`, `docs/specs/` | the write-ups, run docs, and specifications |
| `figures/`, `runs/` | result figures and (gitignored) run artifacts |
| `tests/` | the test suite (language-agnostic, real tests, no stubs) |

## Quickstart

```bash
pip install -e .                                   # or: uv pip install -e .

# the product
python -m verisim.certify audit --denylist "/etc/shadow"      # see a guardrail leak, with proof

# research reproductions (a few; the full set is in RESEARCH-LOG.md)
python -m verisim.experiments.spec27_proposer_eval --seeds 30 # the honest proposer evaluation (the null)
python -m pytest tests/test_certify.py -q                     # the certifier's tests
```

## License & posture

MIT. This is defensive security research and tooling: it measures whether a guardrail covers the
actions that can realize harm, and it is honest about what it does not cover (confidentiality is out)
and about what its certificate is worth (empirical over a sampled space, with a stated residual bound,
not a proof over all of bash).
