# The learned adversarial proposer: oracle-as-reward hole hunting (RA23, 2026-06-20)

> RA22 synthesized the covering target with a **structured fuzzer** as the proposer — an enumerative
> grid of verb × path-form. RA23 wires in the piece the program was built for ([docs/lineage.md](lineage.md),
> [verisim.rl](../src/verisim/rl/)): a **learned proposer trained by the exact oracle's own verdict**.
> The reward is whether a proposed action is a *coverage hole* (`realizes(a) ∧ ¬target(a)`) — read from
> the free exact oracle, never a learned reward model. This is the verifier-as-reward / RLVR thesis
> applied to **adversarial test generation**: the model proposes, the oracle disposes, and the disposal
> is the training signal.

## The mechanism

A factorized softmax policy over three categorical dimensions — (attack target {protected, benign},
verb, path-transform) — trained by REINFORCE with a running-mean baseline. Tabular, torch-free, seeded
(deterministic). It runs inside the RA22 CEGIS loop: propose → query oracle → reward = hole? → update
policy → repair the target on coverable (literal) holes → repeat. As the loop covers the literal class,
protected+literal stops paying reward, so the policy is *forced* to discover where holes still live.

The point is the mechanism — an adaptive adversary driven by a verifiable reward — not that RL is hard
here (it isn't; the space is small and the reward dense). The value over the RA22 grid is measured.

## The result (hermetic; seeded; labels cross-checked against `/bin/sh`)

600 oracle calls, learned vs a blind uniform proposer over the *same* operator space:

| metric | learned (oracle-as-reward) | blind uniform |
|---|---|---|
| coverage holes found | **572** | 251 |
| oracle calls to reach all residual classes | **11** | 17 |
| final mass on indirection transforms | **0.99** | 0.80 (uniform) |
| final mass on the protected target | **0.997** | 0.50 (uniform) |
| synthesized target | `["/etc"]` | `["/etc"]` |
| **silent miss (soundness)** | **0** | **0** |
| bash label/exec mismatches | **0** | — |

Three findings, all measured:

1. **Adaptivity / sample-efficiency.** The learned adversary finds **2.3× more holes** per oracle call
   than blind search, because the oracle's hole-verdict (a verifiable reward) concentrates its
   proposals on hole-rich regions. As the CEGIS loop repairs the literal class, the policy's mass on
   `literal` collapses (to ~0.01) and flows to the indirection transforms that still pay — a visible
   literal→indirection learning curve (`figures/ra23_learned_proposer.png`).
2. **A residual class beyond the grid.** The operator space includes **quote-splice** indirection
   (`/et""c/shadow`), which RA22's enumerative grid did not contain. The learned adversary surfaces it,
   and the oracle/bash confirms it realizes the harm and that no string pattern covers it — so it joins
   the proven-irreducible residual, widening the command space the certificate is tested over.
3. **Soundness under an adaptive adversary.** Every hole the learner finds is realizing-and-routed
   (covered by the synthesized target, or in the explicitly-routed residual). The soundness invariant —
   no realizing action silently off-surface — **survives the adversary** (silent miss = 0), and the
   literal class still gets covered (`["/etc"]`).

This is where the two arcs of the program meet: the world-model / RL arc supplies the proposer; the
oracle-coverage arc supplies the verifiable reward and the soundness guarantee. The proposer learns to
attack; the certificate stays sound.

## Honest boundaries

- **One well-specified harm family** (file corruption on a protected path), same scope as RA18/RA22.
- **A tabular policy over a small operator space**, not a neural sequence model; the dense reward makes
  this the easy-RL regime. The contribution is the *mechanism* and the soundness-under-adversary
  result, not an RL difficulty claim. A larger compositional grammar + a neural proposer is the
  natural scale-up (and where a real learned world model would slot in).
- **The certificate is empirical** over the sampled command space, not a proof over all bash — the
  adversary tests it harder than the grid did, but it remains a measured coverage, not a theorem.

## Reproduce

```
python -m verisim.experiments.ra23_learned_proposer          # learned vs blind (hermetic)
python -m verisim.experiments.ra23_learned_proposer --bash   # + cross-check labels vs real /bin/sh
python figures/plot_ra23.py                                  # -> figures/ra23_learned_proposer.png
python -m pytest tests/test_ra23_learned_proposer.py -q      # hermetic; no shell
```
