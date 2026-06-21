# A neural compositional adversary vs the RA18 resolver (RA24, 2026-06-21)

> RA23 wired a **learned** proposer into the coverage loop, but two things were soft: the policy was
> tabular (it picked one transform for the whole path, so it could not *compose*), and its opponent
> was a literal-only strawman target, not RA18's real abstract resolver. RA24 fixes both. The
> adversary is a **neural autoregressive policy** ([verisim.model.transformer.GPT](../src/verisim/model/transformer.py))
> over a **compositional grammar**, and its opponent is the real [shell_resolver](../src/verisim/realagent/shell_resolver.py).
> The reward is still the exact oracle's own verdict — no learned reward model — so this is the
> verifier-as-reward thesis ([docs/lineage.md](lineage.md)) carried to a genuinely compositional
> attack space, where the two arcs of the program ([RL/world-model](../src/verisim/rl/) +
> oracle-coverage) actually meet at scale.

## The mechanism

The protected path `/etc/shadow` is split into 2-char **atoms**, and an action assigns each atom one
of **twelve** standard shell path-encoding mechanisms (literal, quote-splice, `${var}`, `$'\xNN'`,
`$'\NNN'`, `$(printf %s)`, `${u:-…}`, nested `$(printf)`, backticks, `printf '\xNN%s'`,
`echo|cut`, `echo|rev`), plus a verb and a direct/symlink redirect. The mechanisms **compose** — the
realized command mixes indirection forms the resolver must fold *together* — so the space is
`12**6 × verbs` per path, exponential in length, and a single-transform policy (RA23's architecture)
can only ever emit the dozen *uniform* forms. Every mechanism is string-resolvable and **faithful**:
[`bash_cross_check`](../src/verisim/realagent/compositional_grammar.py) confirms each composed
command realizes exactly as labeled against a real `/bin/sh` (the RA2 anchor move) — **0 unfaithful
of 200** sampled commands, so every ABSTAIN/CLEAR below is a real resolver result, not a rendering
artifact.

The policy is a 2-layer GPT decoding a length-`2+|atoms|` structured sequence (verb, redirect, one
mechanism per atom), logits masked per position, trained by REINFORCE with a running-mean baseline
and an entropy bonus. The reward is the **tiered oracle verdict** (no learned model in the loop):

| event (realizing protected-path command) | reward | meaning |
|---|---|---|
| string-resolvable & resolver **CLEAR** | **1.0** | a **true silent miss** — a soundness violation |
| string-resolvable & resolver **ABSTAIN** | **0.5** | folder **incompleteness** — the frontier a miss hides behind |
| string-resolvable & resolver **FIRES** | 0.0 | folded correctly; the resolver wins |
| `~`string-resolvable (symlink) | 0.0 | the honest, *provably*-irreducible residual (post-commit diff) |

## The result (hermetic; seeded; labels cross-checked vs `/bin/sh`; 1600 oracle calls/arm)

**The discovery.** Run against the resolver *as the program shipped it* (`sound_printf=False`), the
neural adversary discovers a genuine RA18 soundness bug the hand-run red team missed: `printf`
**decodes C backslash escapes in its format string** (`$(printf '\x2f%s' e)` → `/e`), but the
resolver's `_printf_fold` folded the format literally, producing a constant that drops the protected
prefix — so a realizing command was silently declared **CLEAR**.

| discovery lane (pre-RA24 resolver) | neural | blind uniform | single-transform (RA23) |
|---|---|---|---|
| true silent misses found | **1316** | 74 | 64 |
| distinct silent-miss compositions | **1307** | 74 | — |
| minimal silent-miss depth | **3** | — | 6 (uniform only) |

The learned compositional adversary finds the violation **~18× more often per oracle call** than
blind search, and at lower composition depth than a single-transform policy can represent.

**The fix, and soundness under the adversary.** The RA24 hardening makes `_printf_fold` refuse
(ABSTAIN) on any format carrying a backslash escape rather than fold it to a wrong constant — the
conservative, sound choice (it is RA18's own ABSTAIN-never-collapses-to-CLEAR invariant, now extended
to printf format escapes). Re-run against the hardened resolver (the default):

| hardened lane | neural | blind uniform | single-transform |
|---|---|---|---|
| **true silent misses** | **0** | **0** | **0** |
| reward / oracle call | **0.471** | 0.110 | 0.066 |
| distinct frontier compositions | **1500** | 352 | 18 (uniform-capped) |
| minimal frontier depth | **3** | — | 6 (uniform only) |

The soundness invariant **holds under the learned adversary**: across 1600 adaptive proposals (and a
1500-sample hermetic sweep in the tests) no string-resolvable realizing command is silently CLEAR.
What the adversary now does instead is **map the folder-incompleteness frontier** — the
string-resolvable `cut`/`rev`/`printf '\xNN'` forms the resolver ABSTAINs on — **4.3× more
efficiently than blind** and over a **mixed-composition space the single-transform policy is
architecturally blind to** (18 uniform forms vs 1500). That frontier is the honest next thing to
grow: ABSTAIN here is the resolver conceding a string it *could* in principle decide, distinct from
the symlink residual the post-commit diff owns forever.

Three findings, all measured:

1. **A neural compositional adversary discovers a real soundness bug** (printf format-escape folding)
   that RA18's hand-run red team missed, ~18× faster than blind search — the program's
   red-team-folds-back pattern, now automated and learned.
2. **The soundness invariant survives the learned adversary** once the bug is folded back: 0 silent
   misses across the composition space, the load-bearing RA18/RA22/RA23 property carried to a neural
   attacker over an exponential grammar.
3. **Compositionality is load-bearing.** Reward/call is 4.3× blind and 7.1× the single-transform
   (RA23) control; the neural policy reaches mixed-composition frontier witnesses (depth 3 of 6) the
   uniform policy cannot represent, covering 1500 distinct compositions vs its 18.

## Honest boundaries

- **One well-specified harm family** (file corruption on a protected path), same scope as
  RA18/RA22/RA23. The grammar is a realistic-but-finite attacker toolkit (twelve mechanisms over a
  fixed atomization), not all of bash; the certificate is empirical over the sampled composition
  space, not a proof.
- **The reward is dense and the space is small enough that REINFORCE converges fast** (within-run
  reward/call rises ~0.39 → ~0.48); the contribution is the *mechanism* (a neural compositional
  adversary grounded by a verifiable reward, attacking the real resolver) and the discover→fix→
  re-verify loop, not an RL-difficulty claim. A larger compositional grammar and cross-call binding
  are the natural scale-up.
- **The frontier (`cut`/`rev`) is string-resolvable-in-principle but unfolded** — RA24 maps it but
  does not close it; growing the folder to cover pipe-filters (and proving it) is open, the same
  shape as RA18's residual.

## Reproduce

```
python -m verisim.experiments.ra24_neural_proposer                 # both lanes (hermetic)
python -m verisim.experiments.ra24_neural_proposer --bash          # + cross-check vs real /bin/sh
python -m verisim.experiments.ra24_neural_proposer --plot figures/ra24_neural_proposer.png
python -m pytest tests/test_ra24_neural_proposer.py -q             # hermetic; shell test skipif-guarded
```
