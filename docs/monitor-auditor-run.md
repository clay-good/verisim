# The monitor auditor: certified-complete monitoring you point at someone else's guardrail (SPEC-23, 2026-06-21)

> The RA arc (RA22–RA25) certified *our* monitor over *one* harm family with the protected path
> (`/etc/shadow`) baked into the proposer, the resolver, and the loop. RA25's own conclusion is that
> on that substrate the certification loop is *least* needed — reversibility routing to the
> post-commit diff already makes every reversible ABSTAIN safe with no folder at all. SPEC-23 makes the
> move that conclusion points at: lift the monitor and the oracle behind two narrow protocols, and the
> *same* discover→fix→re-verify loop becomes a tool you point at a guardrail you did not write, and at
> harm families a static sandbox cannot express.

## What SPEC-23 adds

Two new packages, no new oracle and no new adversary architecture:

- **[`src/verisim/audit/`](../src/verisim/audit/)** — the [`Monitor`](../src/verisim/audit/protocols.py)
  / [`Oracle`](../src/verisim/audit/protocols.py) protocols, the
  [`audit(monitor, oracle, proposer, budget) -> Certificate`](../src/verisim/audit/auditor.py) loop
  (extracted from `coverage_synth.synthesize`), and the table-3 adapters
  ([`monitors.py`](../src/verisim/audit/monitors.py), [`oracles.py`](../src/verisim/audit/oracles.py),
  [`proposers.py`](../src/verisim/audit/proposers.py)) that wrap today's shipped predicates as protocol
  instances. A soundness hole is exactly `oracle.realizes(a, s) and not monitor.covers(a, ctx)`.
- **[`src/verisim/policy/`](../src/verisim/policy/)** — a small declarative language for the
  relational / cumulative / context-dependent triad (the oracle's only un-dominated territory per
  [docs/review.md](review.md)); each policy compiles to a `Monitor` + `Oracle` the same auditor audits.

`neural_proposer.py` and `compositional_grammar.judge` gain a `prefix`/`atoms` target parameter whose
default reproduces the RA24 `/etc/shadow` behavior byte-identically (the RA22/24/25 tests pass
unchanged); the auditor passes a monitor's harm target in, so nothing is hardcoded in the loop.

## The result (hermetic; seeded; `--bash` cross-checked vs `/bin/sh`)

Run records are the certificates under [`figures/`](../figures/) (`spec23_monitor_auditor.csv` + one
JSON per run). One auditor, every monitor:

### Direction A — the certification loop survives the abstraction (H158/H160)

| run | monitor | oracle | silent holes | sound |
|---|---|---|---|---|
| reproduce RA24 (pre-hardening) | `resolver` (`sound_printf=False`) | shell-path | **6** (the printf-format-escape class) | **False** |
| reproduce RA24 (hardened) | `resolver` | shell-path | **0** | **True** |
| reproduce RA22 partition | `syntactic` grown **from empty** | shell-path | **0** (literal covered; indirection → routed residual) | True |

The auditor re-derives the RA24 `printf` silent miss and the RA22 partition **through the `Monitor`
interface with no literal `/etc` in the loop**, and the numbers match the hardcoded runs. The
certificate is CI-consumable: `python -m verisim.audit resolver-unsound shell-path` writes a versioned
JSON and **exits non-zero on the silent hole**; the hardened resolver exits zero.

### H159 — the neural proposer generalizes off the hardcoded path

Retargeted to `/srv/secret/key` (not `/etc`) and tuned on the resolver reward, the neural compositional
adversary surfaces **5 oracle-confirmed holes** an arbitrary denylist misses, where a blind uniform
proposer over the same retargeted space finds **0** — `neural_holes > blind_holes` across seeds 0–4.
The denylist catches the literal path and the "loud" foldable encodings; the learned adversary explores
the deep mixed compositions it does not enumerate.

### H161 — the oracle is a real container state-diff, not a path predicate

Auditing the **same** monitor with the syntactic `ShellPathOracle` and the real-`/bin/sh`
`ContainerDiffOracle`: both agree on the reversible printf class (**6 holes**), and the diff oracle
**additionally catches the 6 symlink-indirection holes** the syntactic oracle is structurally blind to
(its realized target depends on on-disk link state the string never names). So the certificate means
"realizes the harm," not merely "names the path."

### Direction B — the declarative policy triad, audited by the same loop (H162/H163/H164)

| policy | monitor | silent holes | sound |
|---|---|---|---|
| relational (RA8), under-specified blocklist | `relational_policy` | **2** (`analytics→reporting`, `analytics→db`) | False |
| relational, closure-complete (repaired) | `relational_policy` | **0** | **True** |
| cumulative (RA9), stateless per-resource | `cumulative_policy` | **25** (every over-budget plan) | False |
| cumulative, stateful accumulator (repaired) | `cumulative_policy` | **0** | **True** |
| context (RA12), static allow-posture in a freeze | `context_policy` | **1** | False |
| context-aware (repaired) | `context_policy` | **0** | **True** |

**B consumes A**: the auditor finds an oracle-confirmed hole in each under-specified policy and
certifies the repaired one clean over the sampled space (H163). The cumulative harm leads with an
unambiguous integer budget (no contestable label, H164), and the identical `audit()` entrypoint runs
unchanged across four harm families (file-corruption + the triad).

## The honest takeaway (the RA arc's caveats, inherited)

The certificate is over a finite **sampled** composition space — "no hole found" means "none in the
sampled space," not a theorem. The method needs an exact, resettable oracle; it exists for
integrity/state-change harms and does **not** exist for confidentiality/covert-channel exfiltration
(RA15), so the irreversible slice fails closed to a human — it is not certified. Auditing an
*unmodified* third-party system (a YoloFS staging predicate, ePCA axioms) is the next step: the MVP
proves the loop is monitor-agnostic on instances we adapt (resolver, denylist, policy), which is the
prerequisite, not the full product.

## Reproduce

```
python -m verisim.audit resolver-unsound shell-path        # reproduce the RA24 printf hole (exit 1)
python -m verisim.audit resolver shell-path                 # hardened resolver is sound (exit 0)
python -m verisim.audit syntactic shell-path --proposer corpus   # the RA22 partition from empty
python -m verisim.experiments.spec23_monitor_auditor --neural --bash   # the full table + certificates
python -m pytest tests/test_spec23_auditor.py tests/test_spec23_policy.py tests/test_spec23_generalization.py -q
```
