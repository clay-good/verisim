# The coverage gate on a real Claude Code corpus (RA21, 2026-06-20)

> RA17 made the §8 product reading runnable (the oracle coverage gate as a Claude Code `PreToolUse`
> hook) and measured it against a denylist on a **synthetic 18-call battery**. RA21 answers the
> question that battery cannot: **on real agent traffic, how much approval does the coverage operating
> point actually cost, and would it leak?** It replays every state-mutating tool call this machine's
> Claude Code installation has logged through the gate and the denylist status quo, and reports the
> real off-surface density against the denylist's real approval fatigue.

## What was run

`bench/cc_corpus/extract.py` walks `~/.claude/projects/**/*.jsonl`, extracts every **Bash / Edit /
Write / NotebookEdit** tool call with its `cwd`, and replays each through three arms
([`verisim.realagent.cc_corpus`](../src/verisim/realagent/cc_corpus.py)):

- **no gate** (allow-all baseline),
- **permission denylist** — a *fair* destructive-verb + real-file-overwrite list (`rm`/`mv`/`dd`/
  `sudo`/…, and `>`/`>>` to a real path; it deliberately excludes `2>&1` and `>/dev/null` so its
  fatigue is not inflated), plus the literal `/etc` / `.env` paths it enumerated for Edit/Write,
- **oracle coverage** — the model-free covering target over a realistic **coding-agent surface**:
  system dirs, secret files (`.env`, `*.pem`, `id_rsa`, cloud/ssh credentials), VCS integrity
  (`.git/hooks`, `.git/config`), and CI definitions (`.github/workflows`).

The harm-coverage half stays honestly labeled: real benign sessions contain no labeled harms, so
missed-harm is measured on an **injected, labeled arsenal** (10 credential/integrity harms spanning
the verb classes) against the *same* surface definition — the benign distribution is real, the harm
distribution is labeled.

**Privacy.** The extractor never emits raw command strings or file paths. Each call is reduced to a
non-sensitive `Record` (tool name, the leading argv token for breakdowns, and the per-arm decisions),
so the committed `RESULTS.csv` / `summary.json` carry **rates, never the user's commands**.

## The result (123,195 real state-mutating tool calls, snapshot 2026-06-20)

| metric | no gate | permission denylist | oracle coverage |
|---|---|---|---|
| **prompt rate** (fraction of real calls escalated) | 0.000 | **0.094** | **0.011** |
| — of which off-surface (pure approval fatigue) | — | 0.093 | 0.000 |
| **off-surface auto-approved** | 100% | — | **98.9%** |
| **missed-harm** (injected labeled arsenal) | 1.00 | **0.70** | **0.00** |

Read it as the §8 claim, now on real traffic:

- **The coverage budget is sparse.** Only **1.1%** of real agent actions touch the coding-agent
  danger surface; **98.9% are auto-approvable**. The approval budget a human should spend is small and
  it is exactly the surface a human should have been asked about.
- **The denylist's prompts are almost all fatigue.** It fires on **9.4%** of actions — ~8.7× more
  than the coverage gate — and **99%** of those prompts are off-surface (benign `rm`/`mv`/redirect of
  build artifacts), the approval-fatigue failure mode, with no harm on the surface.
- **And it still leaks.** On the labeled arsenal the denylist misses **70%** (the `chmod`/`sed -i`/
  `cp`/`python -c`/integrity classes its verb list never enumerated), while the coverage gate misses
  **none**. Adding `chmod` to the list only moves the leak to the next un-enumerated verb — the
  enumeration treadmill the coverage surface escapes by keying on the effect's genesis, not on verbs.

This is the operating point RA16 named in control terms (the approval prompts *are* the audit budget):
the coverage gate turns approval-fatigue-vs-safety from a tradeoff into one lever — fewer prompts
*and* higher coverage at once.

## Cross-project generalization (H153): the ~99% is not one workflow

The first thing a reviewer asks of a one-machine number is whether it is an artifact of one workflow.
The corpus spans **31 distinct projects** (different repos, languages, and domains), which are
quasi-independent sub-corpora, so we break the off-surface density down per project (`BY_PROJECT.csv`,
anonymized to ordinal ids; **29 projects** have ≥20 calls).

| per-project off-surface density | value |
|---|---|
| median | **0.991** |
| p25 / p75 (IQR) | 0.981 / 0.995 |
| min | 0.556 |
| projects below 0.95 | 3 of 29 |

The central tendency is high and **tight** — the headline is a property of how a coding agent acts,
not of one project. The interesting part is the three low outliers, and they are not noise: they are
small, **security-work** projects. The 0.556 outlier is the repo's own **RA7 credential-tamper
clean-room** (`/private/tmp/ra7-clean`), a project whose actual work is manipulating `/etc/shadow`,
`.env`, and credentials; the others are a security tool and a scratch dir for this very arc. There the
on-surface rate is high **because the surface is the work** — which is the gate doing exactly the right
thing: its approval budget **auto-scales to how security-sensitive the real work is** (~1% in ordinary
coding, ~44% when the job is touching secrets). This is also why "coverage prompts ≤ denylist" is not
true in *every* project — in a credential-heavy project the effect-surface gate correctly prompts more
than a blunt verb denylist that ignores secrets; the denylist is *under*-prompting on the dangerous
work, not the gate over-prompting (see `figures/ra21_by_project.png`).

The honest limit remains: this is cross-*project* within one machine, not cross-*machine*. The
extractor is drop-in ready for the real thing — point `--root` at a second person's corpus and merge.

## Honest boundaries

- **The benign distribution is one user's corpus** (3,836 sessions, this machine). It generalizes
  across the 29 projects within it (above), but it is not a sampled population of deployments; the
  cross-*machine* check is the obvious next step and is not yet done.
- **Harms are injected, not observed.** Real sessions contained no labeled harmful actions; the 0.00
  vs 0.70 missed-harm is on a hand-built labeled arsenal, the same honesty boundary as RA17.
- **The surface is hand-specified.** `SURFACE` is a deployer's covering target for a coding agent, not
  derived from an external harm definition. Automating and certifying the surface is the open research
  frontier (the direction-#2 work).
- **The pre-commit target is syntactic.** This corpus replay uses the syntactic coding-surface target;
  the indirection edge (paths built without the literal prefix) is already characterized in RA4 and
  closed for the string-resolvable slice in RA18 — orthogonal to the density measurement here.
- **`cwd` is present on ~83% of events**; calls without a `cwd` resolve relative paths against `/`,
  which can only *under*-fire the surface, so the off-surface density is if anything conservative.

## Reproduce

```
python bench/cc_corpus/extract.py            # ~/.claude/projects -> RESULTS.csv, BY_PROJECT.csv, summary.json
python figures/plot_ra21_cc_corpus.py        # -> figures/ra21_cc_corpus.png (the headline)
python figures/plot_ra21_by_project.py       # -> figures/ra21_by_project.png (cross-project)
python -m pytest tests/test_ra21_cc_corpus.py -q   # hermetic; does not read ~/.claude
```

To run the real cross-machine check, point `--root` at a second corpus:
`python bench/cc_corpus/extract.py --root /path/to/other/.claude/projects --out /tmp/other`.

The count drifts slightly run-to-run because the live corpus grows; the rates are stable.
