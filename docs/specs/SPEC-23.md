# SPEC-23 — The Monitor Auditor: certified-complete effect-monitoring as a tool you point at *someone else's* guardrail

**MVP/POC specification. Every prior spec studied the oracle-grounded gate as *our* monitor over *one*
harm family (a protected path, `/etc/shadow`-shaped): SPEC-22 deployed it as a safety gate, the RA arc
(RA22–RA25) turned its own free oracle into the verifiable reward of a neural compositional adversary
and ran discover→fix→re-verify on the shell-path resolver. SPEC-23 makes the sharp, slightly
subversive move the program's own RA25 conclusion points at: the certification loop — the crown jewel,
the one thing the 2026 deterministic-guardrail crowd does not have — is currently spent on the single
substrate where it is *least* needed, because reversibility routing to the post-commit diff already
makes the resolver's ABSTAINs sound with no folder at all (RA25/H157). The loop's real value lives in
the two places with no post-commit-diff fallback: auditing *other people's* monitors, and *new harm
families* (especially irreversible ones). SPEC-23 scopes the smallest artifact that captures that
value — a `Monitor`/`Oracle`-protocol auditor that takes an arbitrary covering surface and an exact
oracle and returns certified holes + a coverage certificate — and the policy-language harm model it
feeds.**

> **✅ SHIPPED — MVP / POC SPEC — 2026-06-21.** All of H158–H164 are built and tested (hermetic,
> torch-free where possible, `--bash`/`--neural` arms run on demand): the protocol-driven auditor
> ([`src/verisim/audit/`](../../src/verisim/audit/)) and the declarative policy language
> ([`src/verisim/policy/`](../../src/verisim/policy/)), with the run-record in
> [`docs/monitor-auditor-run.md`](../monitor-auditor-run.md) and certificates/CSV under
> [`figures/`](../../figures/) (`spec23_monitor_auditor.csv`). Reproduce:
> `python -m verisim.audit resolver-unsound shell-path` (the RA24 printf hole through the protocol,
> exit 1) and `python -m verisim.experiments.spec23_monitor_auditor --neural --bash`. Sibling to
> [SPEC-22](./SPEC-22.md) (the deployment it
> generalizes) and built on the shipped RA arc. It invents **no new oracle and no new adversary
> architecture**: it lifts the existing neural compositional proposer
> ([`src/verisim/realagent/neural_proposer.py`](../../src/verisim/realagent/neural_proposer.py)), the
> CEGIS coverage synthesizer
> ([`src/verisim/realagent/coverage_synth.py`](../../src/verisim/realagent/coverage_synth.py)), and
> the abstract resolver ([`src/verisim/realagent/shell_resolver.py`](../../src/verisim/realagent/shell_resolver.py))
> off their hardcoded `/etc` target and behind two narrow protocols, so the *same* discover→fix→re-verify
> loop runs against a monitor it did not write. The MVP is **Direction A (the auditor) feeding Direction
> B (a declarative policy language for the harms a static sandbox cannot express)**.

Read [SPEC-22 §3](./SPEC-22.md) (the gate and its missed-danger measurement), [`docs/paper.md`](../paper.md)
§7 (the 2026 crowd this audits — YoloFS, ePCA, Progent, GoEX — each *asserts* its covering surface is
complete and none *certifies* it), and [`docs/review.md`](../review.md) (the hostile self-review whose
verdict — the oracle's *only* un-dominated territory is the relational/cumulative/context triad — is the
reason Direction B exists). This document is *the minimum buildable thing that turns "we certified our
own toy monitor" into "we are the completeness auditor for any agent guardrail."*

---

## 0. One-paragraph thesis

A covering surface is only as sound as the human who enumerated it, and the entire deterministic-guardrail
crowd of 2026 *asserts* completeness rather than *certifying* it (paper §7). In a computer world the
monitor's own exact oracle makes its soundness-hole predicate `realizes(s,a) ∧ ¬covers(s,a)` a dense,
model-free reward, so completeness is a learning problem with a free verifier — and the program already
built the learner (RA23–RA24) and the loop (RA22). SPEC-23's thesis is that **this loop is monitor-agnostic
the moment the monitor and the oracle are protocols rather than hardcoded `/etc` predicates**, and that
the agnostic auditor is the differentiated artifact: point it at a customer denylist, a staging predicate,
or a policy engine, give it an exact oracle for the harm, and it returns the encodings that realize the
harm *off* the checked surface, plus a coverage certificate over the sampled space. The MVP proves the
loop survives the abstraction (it reproduces the RA22–RA25 holes through a `Monitor` interface with no
literal path baked in, H158–H159), emits the certificate as a CI-consumable artifact (H160), grounds the
oracle in a real container state-diff so the certificate means "realizes the harm" and not merely "names
the path" (H161), and then exercises the auditor against the one harm class that justifies an oracle over
a sandbox — a declarative policy language for relational, cumulative, and context-dependent danger
(H162–H163). The honest scope is unchanged from the RA arc: a finite sampled space, an empirical
certificate not a proof, and the irreversible-exfiltration edge (RA15) still open.

---

## 1. What this is, honestly (the assessment that motivates the MVP)

Stripped of prose, Verisim is one good idea on a deliberately small substrate: computer worlds have a
free, exact, resettable oracle, so complete mediation (Saltzer–Schroeder, 1975) is cheap — gate only the
sparse genesis surface where harm is born. The 2026 reframing pushed the headline to the novel part: a
monitor's own oracle makes `realizes ∧ ¬target` a model-free reward, so **certifying a monitor's
completeness is a learning problem with a free verifier**; the neural adversary found a real `printf`
format-escape soundness bug a human red team missed (RA24).

**Two things are genuinely ahead of the crowd.** (1) The **action-vs-effect distinction is real signal**:
every deployed defense — denylists, LLM judges, even a strong measured LLM guardrail (RA11 at recall
0.95) — reads the *action*; harms visible only in the *resulting state* (relational RA8, cumulative RA9,
context-dependent RA12) are a structural blind spot no model quality closes. (2) **"Certify the monitor,
don't assert it"** is exportable and uncontested: the CEGIS-with-a-free-oracle loop is the only thing in
the program nobody else has.

**As a deployable defense it is not there yet, and the MVP exists to move it.** The certification headline
currently rides on `realizes = a filesystem diff names a protected path` over one harm family — the
real-`/bin/sh` cross-check is good rigor, but the harm is still "names a protected prefix," so the
certificate means "no shell encoding hides the path," not "no action realizes the harm." Confidentiality
/ exfiltration is a disclaimed negative (RA15: a read leaves no diff). And on single-resource harms a
hardened sandbox (`chattr +i`, MAC) *dominates* the oracle (RA13); the oracle's only un-dominated
territory is the relational/cumulative/context triad — currently three bespoke Python scenarios, not a
capability.

## 2. The pivot (why the MVP points outward, not at our own resolver)

RA25/H157 is the program's own evidence for the pivot: hardening the resolver against the neural adversary
is an **enumeration treadmill** (`rev`, `cut`, then `tr`, `sed`, `base64 -d`, `xxd -r`, … unboundedly),
and the principled closer is not a bigger folder but reversibility routing to the post-commit diff, which
makes every reversible ABSTAIN already safe **with no folder at all**. So "the neural adversary hardens
our own monitor" partly eats itself — the diff was always the soundness guarantee for the reversible
slice. The loop earns its keep precisely where that fallback is absent:

- **Auditing a monitor we did not write** (Direction A) — a third party's covering surface has no
  guaranteed post-commit diff behind it, so a certified hole is a real finding, not a usability note.
- **New / irreversible harm families** (Direction B + RA15) — where the post-commit diff cannot recover
  the effect (an egress send), the covering surface is the *only* line, so certifying it is load-bearing.

## 3. The MVP: Direction A feeds Direction B

### 3.1 Direction A — the third-party monitor auditor (highest leverage, smallest lift)

Two narrow protocols, both already latent in the codebase:

```python
# the surface under audit — anything that classifies an action as on/off the checked surface
class Monitor(Protocol):
    def covers(self, action: Action, ctx: Context) -> bool: ...   # True == "I would check this"
    name: str

# the free verifier — the exact, resettable ground truth of harm for this monitor's harm family
class Oracle(Protocol):
    def realizes(self, action: Action, state: State) -> bool: ...  # True == "this truly causes the harm"
```

A soundness hole is exactly `oracle.realizes(a, s) and not monitor.covers(a, ctx)`. The existing
artifacts become *instances* of these protocols rather than the whole program:

| today (hardcoded `/etc`) | becomes (protocol instance) |
|---|---|
| `command_agnostic.command_targets_protected` | a `SyntacticPathMonitor(prefix)` |
| `claude_code_gate.denylist_decision` | a `DenylistMonitor(patterns)` |
| `shell_resolver.abstract_targets_protected` | a `ResolverMonitor(prefix)` |
| `coverage_synth` real-`/bin/sh` diff | a `ShellPathOracle(prefix)` |
| `neural_proposer` (REINFORCE over the compositional grammar) | the **proposer**, target passed in, not baked |

The deliverable is a CLI/library entrypoint — `audit(monitor, oracle, proposer, budget) -> Certificate` —
that runs discover→fix→re-verify and emits a **coverage certificate**: the holes found (each confirmed
bit-for-bit by the oracle), the sampled composition space, the per-class on/off-surface verdict, and the
fail-closed routing for the irreducible residual. This is "red-team-as-a-CI-gate for agent guardrails,"
and it is the part of the repo with no competitor.

### 3.2 Direction B — a declarative policy language for the un-dominated triad

The oracle's only defensible territory (per the self-review) is harms a static per-resource policy
structurally cannot express. RA8/RA9/RA12 prove the point on bespoke scenarios; the MVP turns them into a
small declarative language whose monitors are then auditable by Direction A:

- **relational** — an invariant over a configuration/reachability graph (`no untrusted ~> db`); the
  oracle evaluates the *resulting* reachability of a proposed edit (RA8).
- **cumulative** — a session budget/counter (`distinct sensitive records touched <= B`); the oracle keeps
  the stateful count and gates the access that would cross it (RA9). Lead with this — its harm is an
  unambiguous integer, repairing the contestable-label problem the review flagged for `disclose`.
- **context-dependent** — a predicate over external state (`deny prod change while freeze active`); the
  oracle reads the live flag, not the runbook (RA12).

Each policy compiles to a `Monitor` (its covering surface) plus an `Oracle` (the resulting-state check),
so Direction A audits *its* completeness with the same loop — closing the program's loop: the policy
engine is the new monitor, and the auditor certifies it.

## 4. Milestones (hypotheses continue the SPEC-22 ledger from H157)

| ID | Hypothesis | Status | Acceptance |
|---|---|---|---|
| MVP-A1 | **H158** — the certification loop survives abstraction: lifting `Monitor`/`Oracle` behind protocols reproduces the RA22–RA25 holes through the interface with **no literal `/etc` baked into proposer or loop** | ✅ shipped — **SUPPORTED** ([`tests/test_spec23_auditor.py`](../../tests/test_spec23_auditor.py)) | the protocol-driven auditor re-derives the RA24 `printf` hole (6 silent, one per verb; hardened → 0) and the RA22 partition (`SyntacticPathMonitor` grown **from empty** → `['/etc']`, literal covered, indirection routed-residual, 0 silent) against `SyntacticPathMonitor`/`ResolverMonitor` instances; numbers match the hardcoded RA22/RA24 runs |
| MVP-A2 | **H159** — the neural proposer **generalizes off the hardcoded path**: target prefix is a parameter, and the proposer finds holes in a *monitor it was not tuned against* (a synthetic/external denylist) | ✅ shipped — **SUPPORTED** ([`tests/test_spec23_generalization.py`](../../tests/test_spec23_generalization.py)) | retargeted to `/srv/secret/key` (not `/etc`), the neural proposer surfaces ≥1 (5 at seed 0) oracle-confirmed denylist hole at `neural_holes > blind_holes` across seeds 0–4 (blind = 0); `neural_proposer`/`compositional_grammar` gained a `prefix`/`atoms` target param (RA24 byte-identical default) |
| MVP-A3 | **H160** — the auditor emits a **CI-consumable certificate**: `audit(monitor, oracle, proposer, budget) -> Certificate` (JSON: holes, sampled space, per-class verdict, irreversible routing) | ✅ shipped — **SUPPORTED** ([`src/verisim/audit/__main__.py`](../../src/verisim/audit/__main__.py)) | `python -m verisim.audit <monitor> <oracle>` writes a versioned certificate JSON and **exits non-zero on any silent (CLEAR-on-realizing) hole** (exit 1 unsound resolver, exit 0 hardened); pinned by a hermetic test |
| MVP-A4 | **H161** — the oracle is a **real container state-diff**, not a path predicate, so the certificate means "realizes the harm": wire the TB lane / `trace/` oracle into the `Oracle` slot | ✅ shipped — **SUPPORTED** ([`src/verisim/audit/oracles.py`](../../src/verisim/audit/oracles.py)) | `audit` runs against `ShellPathOracle` *and* the real-`/bin/sh` `ContainerDiffOracle` on the same monitor; both agree on the reversible printf class (6 holes), and the diff oracle **additionally catches the 6 symlink-indirection holes** the syntactic oracle cannot express |
| MVP-B1 | **H162** — a **declarative policy language** expresses relational + cumulative + context harms, each compiling to a `Monitor` + `Oracle` | ✅ shipped — **SUPPORTED** ([`tests/test_spec23_policy.py`](../../tests/test_spec23_policy.py)) | RA8/RA9/RA12 re-expressed as policies in [`src/verisim/policy/`](../../src/verisim/policy/); oracle is both safe and useful on each (matches the existing RA results), cumulative leads with the integer-budget harm |
| MVP-B2 | **H163** — **B consumes A**: the auditor certifies a *policy* monitor's completeness (finds a hole in a deliberately under-specified policy, confirms none in the repaired one) | ✅ shipped — **SUPPORTED** | the loop finds the 2 oracle-confirmed exposing edges in an empty-blocklist relational policy and certifies the `closure_complete()` policy clean (0 silent) over the sampled space; same shape for cumulative (per-resource → accumulator) and context (static → context-aware) |
| MVP-B3 | **H164** — **credibility fixes** land alongside the MVP: cross-harm-family (≥2 families through one auditor) and the contestable `disclose` label dropped from the headline | ✅ shipped — **SUPPORTED** | the identical `audit()` entrypoint runs unchanged across four harm families (file-corruption + the triad); empirics lead with RA9's count-based integer harm, not a contestable label |

## 5. Architecture (where the seams go)

- **`src/verisim/audit/` (new)** — `protocols.py` (`Monitor`, `Oracle`, `Action`, `Certificate`),
  `auditor.py` (the protocol-driven discover→fix→re-verify loop, extracted from `coverage_synth`), and
  `monitors.py` (the table-3 adapters wrapping today's hardcoded functions). No change to the shipped RA
  modules' behavior — they are *imported* as instances, fail-closed routing (CU27 reversibility) preserved.
- **proposer slot** — `neural_proposer.py` gains a `target`/`grammar` parameter (default = current `/etc`
  behavior, so RA24 is byte-identical); the auditor passes the monitor's harm target in.
- **oracle slot** — `ShellPathOracle` (today's diff-names-the-path) and a `ContainerDiffOracle` (the TB
  lane's execute-diff-rollback) both satisfy `Oracle`; the certificate records *which* oracle produced it.
- **policy layer** — `src/verisim/policy/` (new): a small spec → (`Monitor`, `Oracle`) compiler for the
  three triad shapes, reusing the existing RA8/RA9/RA12 evaluators as the resulting-state checks.

## 6. Honest caveats, stated up front (the RA arc's, inherited)

- **Empirical, not a proof.** The certificate is over a finite *sampled* composition space, exactly as
  RA22–RA25. "No hole found" means "none in the sampled space," not a theorem over all bash / all actions.
- **One free-oracle assumption.** The whole method needs an exact, resettable oracle for the harm. It
  exists for integrity/state-change harms; it does **not** exist for confidentiality/covert-channel
  exfiltration (RA15), so the auditor is honest-silent there — the irreversible slice fails closed to a
  human, it is not certified.
- **Auditing a real third-party monitor is the *point*, and still partly future.** The MVP proves the
  loop is monitor-agnostic on instances we adapt (denylist, resolver, policy); running it against an
  unmodified external system (YoloFS staging predicate, ePCA axioms) is the next step, named not claimed.
- **The sandbox still dominates single-resource harms (RA13).** Direction B exists because that is the
  *only* place the oracle is un-dominated; the MVP must not re-demonstrate harms a `chattr +i` beats.

## 7. Reproduce / acceptance (what "done" means for the POC)

The POC is done when: (1) `python -m verisim.audit <monitor> <oracle>` reproduces the RA24 `printf` hole
through the protocol with no hardcoded path (H158–H159); (2) it writes a certificate JSON and exits
non-zero on a silent hole (H160); (3) the same command runs against a container-diff oracle (H161) and
against a relational-policy monitor, finding a planted hole and certifying the repair (H162–H163); (4)
every number is pinned by a hermetic test (torch-free where possible) and a `--bash` real-`/bin/sh`
cross-check, matching the RA arc's reproduce discipline. Run-records land in `docs/monitor-auditor-run.md`;
figures + CSVs under `figures/`.
