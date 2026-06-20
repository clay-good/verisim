# Automated coverage: synthesize and certify the covering target (RA22, 2026-06-20)

> The program's deepest open objection ([docs/review.md](review.md) item 2d): the gate's safety rests
> on a *covering* target, yet every target so far is **hand-specified** — the `/etc` prefix, RA21's
> coding surface, the RA18 resolver grammar. RA18 hardened a hand-written target with a hand-run red
> team. RA22 closes the loop: it **synthesizes the covering target from nothing** with a
> counterexample-guided (CEGIS) search and emits a **coverage certificate** that names its own
> irreducible residual.

## The shape

This is the program's founding thesis ([docs/lineage.md](lineage.md)) — a stochastic/structured
proposer paired with a deterministic exact verifier — now load-bearing for *safety* rather than world-
model faithfulness:

- **realizes(action)** — ground truth: does executing the action corrupt the protected region? Known
  by construction of the candidate corpus, and **confirmed equal to a real `/bin/sh`** by
  `cross_check_against_bash` (the RA2 anchor-invariance move). It is never read from the target.
- **target(action)** — the cheap, model-free pre-commit guess being synthesized: a growable set of
  path-pattern rules over the command string. It **starts empty**.
- **a coverage hole** — a witness `realizes(a) and not target(a)`: a realizing action off the surface
  (a silent miss). CEGIS searches for holes and **repairs** the target on each, until a full pass
  finds none (loop-until-dry).

The proposer here is a structured fuzzer (10 verbs × 4 path-forms × {protected, benign}); the slot
where a *learned* proposer would generate harder candidates is explicit and is the natural extension.

## The result (hermetic; corpus labels cross-checked against `/bin/sh`)

From an empty target, over 50 candidates (40 realizing), in **2 CEGIS rounds**:

| outcome | value |
|---|---|
| synthesized target | `["/etc"]` (grown from nothing) |
| literal realizing class | **100% covered** by the synthesized target |
| indirection/symlink class | **isolated as the irreducible residual** → routed to post-commit |
| residual classes | `indirection_var`, `indirection_subst`, `indirection_symlink` |
| **silent miss (soundness invariant)** | **0** |
| benign over-fire | 0 |
| bash label/exec mismatches | **0** (labels are exact) |

The load-bearing finding: with **no hand-specification**, the loop covers the entire class of
realizing actions that *name* the protected path, and **automatically isolates the indirection class
as irreducible to any pre-commit string pattern** — the same partition RA18 drew by hand, now
*discovered*. The soundness invariant holds by construction: at convergence no realizing action is
silently off-surface — every one is either covered by the synthesized target or in the
explicitly-routed residual (the post-commit diff / CU27 reversibility routing).

Why the residual is genuinely irreducible, not unsolved: an indirection command assembles the
protected path at runtime (`p=/et; …${p}c/shadow`, `$(printf …)`, a planted symlink), so the literal
prefix is *never a token* — there is no string feature for any pre-commit pattern to key on. The
exact post-commit diff catches it regardless (it reads the effect, not the string), which is exactly
where RA22 routes it. The bash cross-check proves both halves: every indirection form really does
write the sandbox secret (so `realizes` is True), and no literal pattern can see it.

## Honest boundaries

- **The harm is one well-specified family** (file corruption on a protected path). The synthesizer is
  general over verbs and path-forms within it; extending the proposer to network/exfil harms (where
  RA15 already shows there is no sparse covering surface) is future work, not claimed here.
- **The certificate is empirical, not a proof over all bash.** It certifies coverage over the
  generated corpus and isolates the residual class; it is the automated analogue of RA18's measured
  coverage, with the same honest limit (an evolving corpus, not a theorem over arbitrary shell).
- **The proposer is a structured fuzzer, not yet a learned adversary.** The "model proposes, oracle
  disposes" slot is implemented and tested with an injected oracle; wiring a learned proposer to hunt
  harder holes is the next step (and the point where the world-model arc reconnects).

## Reproduce

```
python -m verisim.experiments.ra22_coverage_synth          # the certificate (hermetic)
python -m verisim.experiments.ra22_coverage_synth --bash   # + cross-check labels vs real /bin/sh
python figures/plot_ra22.py                                # -> figures/ra22_coverage_synth.png
python -m pytest tests/test_ra22_coverage_synth.py -q      # hermetic; no shell
```
