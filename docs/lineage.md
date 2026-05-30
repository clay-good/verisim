# Lineage — from the author's verifiers to Verisim

> The explicit mapping SPEC.md §8 promises: how Verisim generalizes the author's
> prior work. The through-line across that work is one idea — **a deterministic
> verifier sitting over a stochastic process** — and Verisim is the point where the
> verifier stops guarding a single action and starts grounding a whole simulated
> world.

## The common shape

Every prior project pairs a *stochastic producer* (an agent, a model, a pipeline)
with a *deterministic checker* that adjudicates the producer's output against rules
that hold by construction. The checker is cheap, exact, and reproducible; that is
what makes it trustworthy as a gate. Verisim keeps exactly this shape and changes
one thing: the thing being checked is no longer a single decision but the **next
state of an autoregressively-rolled world model**, where errors compound over a
horizon. The checker becomes the §3 oracle; the gate becomes the §5.2
propose–verify–correct loop; the "is this allowed?" question becomes "is this
predicted state faithful?"

## The mapping

| Prior work | Stochastic producer | Deterministic verifier | What Verisim generalizes |
|------------|--------------------|------------------------|--------------------------|
| **Proxilion** | agent / proxy actions | policy checker over a security boundary | the checker now verifies *predicted dynamics*, not a single boundary crossing |
| **Invariant** | a process that must preserve properties | invariant checker | the "invariant" becomes faithfulness to the oracle's true next state, measured continuously as `d(s, ŝ)` |
| **Mantissa** | numeric/financial computation | exact recomputation as ground truth | exact recomputation becomes the reference oracle `O(s, a)` for an environment |
| **agent-replay** | a recorded agent trajectory | deterministic replay/verification | replay-as-truth becomes the ground-truth rollout the coupled rollout is scored against |
| **Vaulytica** | a stochastic analysis pipeline | golden-corpus verification | the golden-corpus discipline becomes Verisim's committed semantics goldens (SPEC-2 §12, `tests/test_goldens.py`) |
| **enklayve** (tax engine) | rule application over inputs | golden test corpus, manifest + content-hash regeneration | the manifest/content-hash + golden regime is mirrored directly in Verisim's dataset manifests and reproducibility regime (SPEC-2 §4, §12) |

## What changes when you lift it to a world

Three things get strictly harder going from "verify one action" to "ground a world
model," and they are exactly Verisim's research questions:

1. **Errors compound.** A single-action verifier never has to reason about the *50th*
   step depending on the *1st*. A world model does — that is the whole point of the
   faithful horizon `H_ε` (RQ1 / H1).
2. **Verification has a budget.** A gate can run on every action cheaply. Grounding a
   rollout means choosing *when* to spend an oracle call (RQ2 / H2) and *how* to apply
   the correction (RQ3 / H3) — the verifier becomes something you optimize the *use*
   of, not just run.
3. **The verifier can teach.** A gate accepts or rejects. An oracle-in-the-loop can
   feed its corrections back as a verifiable reward (RLVR; SPEC.md §6.3,
   [`verisim.rl`](../src/verisim/rl/)) — the verifier becomes a training signal, not
   just a filter.

This is why Verisim is framed as the *generalization* rather than another instance:
it takes the author's standing verifier-as-reward thesis and asks what it becomes when
the thing under verification is an entire learned simulation of a computer world.
